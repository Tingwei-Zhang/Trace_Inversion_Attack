#!/usr/bin/env python3
import argparse
import json
import os
from tqdm import tqdm

# System prompt for inversion model (same as format_data.py --scenario r1_distill)
INVERSION_SYSTEM_PROMPT = """You are a language model that reconstructs full internal reasoning traces from high-level bubble summaries.

You will be given:
- A problem **input** (e.g., a math or logic problem)
- A final **output** or solution
- A list of numbered **reasoning bubbles**, where each bubble summarizes one key insight, step, or decision made during the problem-solving process

These bubbles are **condensed summaries** of what was originally a much longer, richer internal thought process.

Your task is to reconstruct that full process.

Below are high-level bubble summaries representing condensed thoughts or decisions. Your task is to reconstruct the full thinking trace that might have led to each summary. For each bubble, expand it into a **detailed internal monologue or reasoning chain**, showing how one idea leads to the next.

Include:
- Assumptions and background intuitions
- Intermediate steps, definitions, and subcases
- Natural questions or doubts raised during reasoning
- Alternatives that were considered and rejected
- Transitions that make the reasoning coherent and plausible

Use informal, introspective language — as if the person is thinking out loud. Add math expressions in \\( \\LaTeX \\) where appropriate.

Do **not** invent new reasoning steps outside the bubbles. Use the **input** and **output** only for context and consistency. Your goal is to **flesh out the bubbles**, not to re-solve the problem from scratch.

The full trace should:
- Be logically consistent and cohesive from start to finish
- Sound like a realistic thought process that could plausibly result in the given answer
- Span multiple paragraphs per bubble and up to 20,000 characters overall if needed
- Be output as one continuous trace, wrapped in `<think>...</think>` tags

You are not summarizing the bubbles. You are recovering the internal narrative that *generated* them.
"""

def parse_prompt(prompt):
    """Parse the chat template prompt to extract system and user content."""
    # The prompt format is: <|im_start|>system\n...<|im_start|>user\n...<|im_start|>assistant\n
    system_start = "<|im_start|>system\n"
    user_start = "<|im_start|>user\n"
    assistant_start = "<|im_start|>assistant\n"
    
    system_content = ""
    user_content = ""
    
    if system_start in prompt:
        system_end = prompt.find(user_start)
        if system_end != -1:
            system_content = prompt[len(system_start):system_end].strip()
    
    if user_start in prompt:
        user_end = prompt.find(assistant_start)
        if user_end != -1:
            user_content = prompt[prompt.find(user_start) + len(user_start):user_end].strip()
    
    return system_content, user_content


def _dataset_entry(file_name):
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        },
    }


def _update_dataset_info(output_file):
    abs_output = os.path.abspath(output_file)
    abs_data_dir = os.path.abspath("data")
    if not abs_output.startswith(abs_data_dir + os.sep):
        return

    relative_path = os.path.relpath(abs_output, abs_data_dir)
    dataset_name = os.path.splitext(os.path.basename(output_file))[0]
    dataset_info_path = os.path.join(abs_data_dir, "dataset_info.json")

    if os.path.exists(dataset_info_path):
        try:
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        except json.JSONDecodeError:
            dataset_info = {}
    else:
        dataset_info = {}

    dataset_info[dataset_name] = _dataset_entry(relative_path)
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"✓ Updated {dataset_info_path} with entry '{dataset_name}'")

def process_chatgpt_data(
    input_file="data/step0_preprocessed_data/chatgpt_inference_raw_20k_gpt_5_mini.jsonl",
    comprehensive_file="data/step0_preprocessed_data/processed_open_thoughts_20k_chatgpt.jsonl",
    no_thinking_file="data/step3_reasoning_model_training_data/open_thoughts_with_no_thinking_chatgpt.jsonl",
    with_bubble_file="data/step3_reasoning_model_training_data/open_thoughts_with_bubble_chatgpt.jsonl",
    inversion_output_file="data/step2_inversion_model_training_data/open_thoughts_to_invert_chatgpt_val.jsonl",
):
    """Process ChatGPT inference JSONL into multiple formats."""
    preprocessed_output_dir = "data/step0_preprocessed_data"
    reasoning_output_dir = "data/step3_reasoning_model_training_data"
    inversion_output_dir = "data/step2_inversion_model_training_data"
    
    os.makedirs(preprocessed_output_dir, exist_ok=True)
    os.makedirs(reasoning_output_dir, exist_ok=True)
    os.makedirs(inversion_output_dir, exist_ok=True)
    
    comprehensive_data = []
    no_thinking_data = []
    with_bubble_data = []
    inversion_data = []
    skipped_entries = []
    
    # Store raw data for inversion format
    raw_data_for_inversion = []
    
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing"):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
                prompt = example.get('prompt', '')
                summary = example.get('summary', '').strip()
                output = example.get('output', '').strip()
                
                # Parse system and user content from prompt
                system_content, user_content = parse_prompt(prompt)
                
                # Base messages format (same as download_dataset.py)
                base = [
                    {"content": system_content, "role": "system"},
                    {"content": user_content, "role": "user"}
                ]

                # Format 0: Comprehensive preprocessed file (same role schema as processed_open_thoughts_20k_r1.jsonl)
                if user_content and output:
                    comprehensive_data.append(
                        {
                            "messages": base
                            + [{"content": output, "role": "assistant"}]
                            + [{"content": summary, "role": "assistant_thinking"}]
                            + [{"content": output, "role": "assistant_answer"}]
                        }
                    )
                
                # Format 1: No thinking (answer only)
                no_thinking_data.append({
                    "messages": base + [{"content": output, "role": "assistant"}]
                })
                
                # Format 2: With bubble (summary as thinking + output)
                # Only save if there's a summary
                if summary:
                    # Wrap summary with <|begin_of_thought|> and <|end_of_thought|> tags
                    # Wrap output with <|begin_of_solution|> and <|end_of_solution|> tags
                    thinking_content = f"<|begin_of_thought|>\n\n{summary}\n\n<|end_of_thought|>\n\n"
                    full_assistant_content = f"{thinking_content}<|begin_of_solution|>\n\n{output}\n\n<|end_of_solution|>".strip()
                    with_bubble_data.append({
                        "messages": base + [{"content": full_assistant_content, "role": "assistant"}]
                    })
                
                # Store data for inversion format
                if summary and output and user_content:
                    raw_data_for_inversion.append({
                        "user_prompt": user_content,
                        "assistant_answer": output,
                        "bubbles_output": summary
                    })
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
    
    # Create inversion training data (format matching format_data.py --scenario r1_distill)
    print("\nCreating inversion training data...")
    for idx, raw_item in enumerate(tqdm(raw_data_for_inversion, desc="Creating inversion data")):
        user_prompt = raw_item["user_prompt"]
        assistant_answer = raw_item["assistant_answer"]
        bubbles_output = raw_item["bubbles_output"]
        
        # Note: We don't have full thinking trace in ChatGPT data
        # The format structure is created, but thinking content is empty
        # This matches the format from format_data.py --scenario r1_distill
        formatted_item = {
            "messages": [
                {"content": INVERSION_SYSTEM_PROMPT, "role": "system"},
                {"content": f"The original problem input is: {user_prompt}\nThe final answer is: {assistant_answer}\nTransform these thinking bubbles into clear full reasoning traces: <think>\n{bubbles_output}\n</think>", "role": "user"},
                {"content": "", "role": "assistant"}  # Empty - full thinking not available in ChatGPT data
            ]
        }
        inversion_data.append(formatted_item)
        
        if not bubbles_output:
            skipped_entries.append({
                "index": idx,
                "reason": "missing_bubbles",
                "user_prompt_preview": (user_prompt[:100] + "...") if len(user_prompt) > 100 else user_prompt
            })
    
    # Save comprehensive preprocessed format
    print(f"\nSaving {comprehensive_file}...")
    with open(comprehensive_file, "w", encoding="utf-8") as f:
        for item in tqdm(comprehensive_data, desc="Writing comprehensive"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    _update_dataset_info(comprehensive_file)

    # Save no_thinking format
    print(f"\nSaving {no_thinking_file}...")
    with open(no_thinking_file, 'w', encoding='utf-8') as f:
        for item in tqdm(no_thinking_data, desc="Writing no_thinking"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    _update_dataset_info(no_thinking_file)
    
    # Save with_bubble format
    print(f"\nSaving {with_bubble_file}...")
    with open(with_bubble_file, 'w', encoding='utf-8') as f:
        for item in tqdm(with_bubble_data, desc="Writing with_bubble"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    _update_dataset_info(with_bubble_file)
    
    # Save inversion format (all data in one file, no train/val split)
    if inversion_data:
        print(f"\nSaving {inversion_output_file}...")
        os.makedirs(os.path.dirname(inversion_output_file), exist_ok=True)
        with open(inversion_output_file, 'w', encoding='utf-8') as f:
            for item in tqdm(inversion_data, desc="Writing inversion data"):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✓ Created {len(inversion_data)} samples for inversion model")
        print(f"✓ Inversion data saved to: {inversion_output_file}")
        _update_dataset_info(inversion_output_file)
    
    # Report on skipped entries
    if skipped_entries:
        print(f"\n⚠️  Skipped {len(skipped_entries)} entries:")
        for entry in skipped_entries[:5]:  # Show first 5
            print(f"  Index {entry['index']}: {entry['reason']}")
            print(f"    Preview: {entry['user_prompt_preview']}")
        if len(skipped_entries) > 5:
            print(f"  ... and {len(skipped_entries) - 5} more")
    else:
        print("✓ No entries were skipped")
    
    # Note about missing thinking
    if inversion_data:
        print(f"\n⚠️  Note: Inversion data created with empty thinking content (full thinking traces not available in ChatGPT input data).")
        print(f"   The format structure matches format_data.py --scenario r1_distill, but assistant messages are empty.")
        print(f"   Full thinking traces need to be added to make this data useful for training.")
    
    print(f"\n✓ Processed {len(no_thinking_data)} samples")
    print(f"✓ Comprehensive preprocessed format: {comprehensive_file}")
    print(f"✓ No thinking format: {no_thinking_file}")
    print(f"✓ With bubble format: {with_bubble_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess ChatGPT inference output into training/eval datasets")
    parser.add_argument(
        "--input_file",
        default="data/step0_preprocessed_data/chatgpt_inference_raw_20k_gpt_5_mini.jsonl",
        help="ChatGPT inference JSONL containing prompt/summary/output/token fields",
    )
    parser.add_argument(
        "--comprehensive_file",
        default="data/step0_preprocessed_data/processed_open_thoughts_20k_chatgpt.jsonl",
        help="Output comprehensive preprocessed file with assistant_thinking/assistant_answer roles",
    )
    parser.add_argument(
        "--no_thinking_file",
        default="data/step3_reasoning_model_training_data/open_thoughts_with_no_thinking_chatgpt.jsonl",
        help="Output no-thinking reasoning file",
    )
    parser.add_argument(
        "--with_bubble_file",
        default="data/step3_reasoning_model_training_data/open_thoughts_with_bubble_chatgpt.jsonl",
        help="Output bubble reasoning file",
    )
    parser.add_argument(
        "--inversion_output_file",
        default="data/step2_inversion_model_training_data/open_thoughts_to_invert_chatgpt_val.jsonl",
        help="Output inversion dataset file",
    )
    args = parser.parse_args()

    process_chatgpt_data(
        input_file=args.input_file,
        comprehensive_file=args.comprehensive_file,
        no_thinking_file=args.no_thinking_file,
        with_bubble_file=args.with_bubble_file,
        inversion_output_file=args.inversion_output_file,
    )
