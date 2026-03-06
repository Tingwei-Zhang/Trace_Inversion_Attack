#!/usr/bin/env python3
"""
Process teacher model inference output to extract entries with thinking content.
Extracts entries containing </think> delimiter and separates thinking from answer.
"""

import json
import os
from tqdm import tqdm

def process_teacher_inference(input_file, output_dir):
    """
    Process teacher model inference output and extract entries with thinking content.
    
    Args:
        input_file: Path to the teacher model inference JSONL file
        output_dir: Directory to save the processed files
    """
    print(f"Processing teacher model inference from: {input_file}")
    
    # Create output directories
    os.makedirs(f"{output_dir}/step3_reasoning_model_training_data", exist_ok=True)
    os.makedirs(f"{output_dir}/step0_preprocessed_data", exist_ok=True)
    
    # Lists to store different formats
    with_thinking = []
    comprehensive = []
    inference_only = []
    
    # Counters
    total_entries = 0
    filtered_entries = 0
    
    # Process the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing entries"):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                # Extract prompt and prediction
                prompt = data.get('prompt', '')
                prediction = data.get('predict', '')
                
                # Check if prediction contains </think> delimiter - skip if not present
                if '</think>' not in prediction:
                    filtered_entries += 1
                    continue
                
                # Parse the prompt to extract system and user messages
                # The prompt format is: <|im_start|>system\n...<|im_start|>user\n...<|im_start|>assistant\n
                parts = prompt.split('<|im_start|>')
                system_content = ""
                user_content = ""
                
                for part in parts:
                    if part.startswith('system'):
                        system_content = part[6:].strip()  # Remove 'system' prefix
                    elif part.startswith('user'):
                        user_content = part[4:].strip()    # Remove 'user' prefix
                
                # Separate thinking and answer using </think> delimiter
                if '</think>' in prediction:
                    start = 0
                    end = prediction.find('</think>')
                    if start != -1 and end != -1:
                        thinking = prediction[0:end].strip()  # Remove '<think>' (7 chars)
                        answer = prediction[end + 8:].strip()         # Remove '</think>' (8 chars)
                    else:
                        thinking = ""
                        answer = prediction
                else:
                    thinking = ""
                    answer = prediction
                
                # Base messages for training data
                base_messages = [
                    {"content": system_content, "role": "system"},
                    {"content": user_content, "role": "user"}
                ]
                
                # Format 1: With thinking (full response)
                with_thinking.append({
                    "messages": base_messages + [{"content": prediction, "role": "assistant"}]
                })
                
                # Format 2: Comprehensive (all roles including thinking and answer)
                comprehensive_msgs = base_messages + [{"content": prediction, "role": "assistant"}]
                if thinking:
                    comprehensive_msgs.append({"content": thinking, "role": "assistant_thinking"})
                if answer:
                    comprehensive_msgs.append({"content": answer, "role": "assistant_answer"})
                comprehensive.append({"messages": comprehensive_msgs})
                
                # Format 3: Inference only (system + user prompts only)
                inference_base = [
                    {"content": " Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.", "role": "system"},
                    {"content": user_content, "role": "user"}
                ]
                inference_only.append({
                    "messages": inference_base + [{"content": "", "role": "assistant"}]
                })
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                continue
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
    
    # only take first 10000 entries for with_thinking
    with_thinking = with_thinking[:10000]
    # Save the processed data
    output_files = [
        (
            f"{output_dir}/step3_reasoning_model_training_data/open_thoughts_with_thinking_r1_distill.jsonl",
            with_thinking,
            "open_thoughts_with_thinking_r1_distill",
        ),
        (
            f"{output_dir}/step0_preprocessed_data/processed_open_thoughts_20k_r1_distill.jsonl",
            comprehensive,
            "processed_open_thoughts_20k_r1_distill",
        ),
    ]

    # Load existing dataset_info.json if it exists
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}
        os.makedirs(output_dir, exist_ok=True)

    for filepath, data, info_key in output_files:
        print(f"Saving {len(data)} entries to {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            for item in tqdm(data, desc=f"Saving {os.path.basename(filepath)}"):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Add dataset info only if it doesn't already exist
        if info_key not in dataset_info:
            relative_path = filepath[5:]
            dataset_info[info_key] = {
                "file_name": relative_path,
                "formatting": "sharegpt",
                "columns": {
                    "messages": "messages",
                },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                },
            }
        else:
            print(f"✓ Dataset info for '{info_key}' already exists, skipping.")

    # Save updated dataset_info.json
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Total entries processed: {total_entries}")
    print(f"✓ Entries filtered out (no </think> delimiter): {filtered_entries}")
    print(f"✓ Processed {len(with_thinking)} samples with thinking content")
    print(f"✓ Training datasets saved to: {output_dir}/step3_reasoning_model_training_data/")
    print(f"✓ Comprehensive format saved to: {output_dir}/step0_preprocessed_data/")

def main():
    """Main function to run the processing."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    input_file = os.path.join(project_root, "output/step0_data_preprocess/processed_open_thoughts_20k.jsonl")
    output_dir = os.path.join(project_root, "data")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return
    
    process_teacher_inference(input_file, output_dir)

if __name__ == "__main__":
    os.system("bash src/step0_data_preprocess/r1_distill_inference.sh")
    main()


