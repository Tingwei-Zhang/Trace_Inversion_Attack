#!/usr/bin/env python3
import json
import os
import argparse
from tqdm import tqdm

class SummarizationDataFormatter:
    """Format data for summarization model training using the summarization prompt."""
    
    def __init__(self):
        self.system_prompt = """You are a model trained to convert informal internal reasoning into a clear, structured sequence of “reasoning bubbles.”  
These bubbles should summarize the key steps of thought, capturing both logical flow and meaningful insight — not just surface-level summaries.

When processing a <think>...</think> trace:

1. Read the full reasoning carefully and extract only the **meaningful logical advances** (e.g. observations, deductions, decisions, failed attempts that change direction).  
2. Summarize each such idea as a **self-contained bubble**, ideally one to three sentences each.  
3. Maintain the **logical flow** of the original trace, showing how the reasoning unfolds.  
4. Keep each bubble:
   - Abstracted (not just copied phrases)
   - Logically complete and well-phrased
   - High-information and nontrivial
5. Do **not** include filler thoughts, aimless speculation, or mechanical calculations unless critical.

You should produce **a few numbered reasoning bubbles**, depending on the depth of the input. Each bubble should contribute meaningfully to the progression of thought.

**Format:**
1. [Detailed, insight-capturing reasoning bubble]
2. [Next reasoning bubble showing development or pivot]
3. ..."""
    
    def format_summarization_data(self, input_file, output_file):
        """Format data for summarization model training."""
        print(f"Loading data from {input_file}...")
        
        try:
            # Load data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            print(f"Found {len(data)} entries to process")
            
            formatted_data = []
            
            for item in tqdm(data, desc="Formatting data"):
                # Extract assistant_thinking content
                thinking_content = self._extract_thinking(item)
                
                if thinking_content and thinking_content.strip():
                    # Create formatted conversation for summarization training
                    formatted_item = {
                        "messages": [
                            {"content": self.system_prompt, "role": "system"},
                            {"content": f"Transform this thinking process into clear reasoning bubbles:<think>\n\n{thinking_content}</think>", "role": "user"},
                            {"content": "", "role": "assistant"}  # Left blank for inference
                        ]
                    }
                    formatted_data.append(formatted_item)
            
            # Save formatted data
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in formatted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"✓ Formatted {len(formatted_data)} samples for summarization training")
            print(f"✓ Output saved to: {output_file}")
            
            # Update dataset_info.json
            self._update_dataset_info(output_file)
            
        except Exception as e:
            print(f"Error formatting data: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_thinking(self, item):
        """Extract thinking content from assistant_thinking role."""
        if "messages" not in item:
            return ""
        for message in item["messages"]:
            if message.get("role") == "assistant_thinking":
                return message.get("content", "").strip()
        return ""
    
    def _update_dataset_info(self, output_file):
        """Update data/dataset_info.json with the new dataset entry."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        dataset_info_path = os.path.join(project_root, "data/dataset_info.json")
        
        # Extract dataset name from output file (e.g., "open_thoughts_to_summarize_r1" from path)
        dataset_name = os.path.splitext(os.path.basename(output_file))[0]
        
        # Always store paths relative to data/ for portability across devices.
        data_dir = os.path.join(project_root, "data")
        abs_output_file = os.path.abspath(output_file)
        if abs_output_file.startswith(os.path.abspath(data_dir) + os.sep):
            relative_path = os.path.relpath(abs_output_file, data_dir)
        elif output_file.startswith("data/"):
            relative_path = output_file[5:]
        else:
            relative_path = output_file
        
        # Create dataset entry
        dataset_entry = {
            "file_name": relative_path,
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system"
            }
        }
        
        # Load existing dataset_info.json or create new
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}
        
        # Add/update entry
        dataset_info[dataset_name] = dataset_entry
        
        # Save updated dataset_info.json
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Updated {dataset_info_path} with entry '{dataset_name}'")

def main():
    # Get project root relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Define path sets for each model
    path_sets = {
        "r1": {
            "input": os.path.join(project_root, "data/step0_preprocessed_data/processed_open_thoughts_20k_r1.jsonl"),
            "output": os.path.join(project_root, "data/step1_summarized_data/open_thoughts_to_summarize_r1.jsonl")
        },
        "r1_distill": {
            "input": os.path.join(project_root, "data/step0_preprocessed_data/processed_open_thoughts_20k_r1_distill.jsonl"),
            "output": os.path.join(project_root, "data/step1_summarized_data/open_thoughts_to_summarize_r1_distill.jsonl")
        }
    }
    
    parser = argparse.ArgumentParser(description="Format data for summarization model training")
    parser.add_argument("--model", default="r1", choices=["r1", "r1_distill"],
                       help="Model type to use (default: r1)")
    parser.add_argument("--input", default=None, 
                       help="Input JSONL file path (overrides model default)")
    parser.add_argument("--output", default=None, 
                       help="Output JSONL file path (overrides model default)")
    args = parser.parse_args()
    
    # Use model defaults if input/output not explicitly provided
    input_path = args.input if args.input else path_sets[args.model]["input"]
    output_path = args.output if args.output else path_sets[args.model]["output"]
    
    formatter = SummarizationDataFormatter()
    formatter.format_summarization_data(input_path, output_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.model=="r1_distill":
        os.system(f"bash {os.path.join(script_dir, 'qwen2_5_summarization_r1_distill.sh')}")
    elif args.model=="r1":
        os.system(f"bash {os.path.join(script_dir, 'qwen2_5_summarization_r1.sh')}")

if __name__ == "__main__":
    main() 