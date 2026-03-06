#!/usr/bin/env python3
import json
import os
import argparse
from tqdm import tqdm
from datasets import load_dataset

class OpenThoughtsDatasetManager:
    """Process OpenThoughts-114k dataset into multiple formats."""
    
    def process_open_thoughts_dataset(self):
        """Process dataset into 3 formats: with thinking, no thinking, and comprehensive."""
        print("Loading OpenThoughts-114k dataset...")
        
        try:
            dataset = load_dataset("llamafactory/OpenThoughts-114k")
            train_data = dataset['train']
            
            with_thinking, no_thinking, comprehensive, inference_only = [], [], [], []
            inference_only_50k = []
            
            max_examples = 50000
            for idx, example in enumerate(tqdm(train_data, desc="Processing")):
                if idx >= max_examples:
                    break
                messages = example.get('messages', [])
                if len(messages) < 3:
                    continue
                    
                system_msg, user_msg, assistant_msg = messages[:3]
                if not (system_msg.get('role') == 'system' and 
                       user_msg.get('role') == 'user' and 
                       assistant_msg.get('role') == 'assistant'):
                    continue
                
                # Extract content
                system_content = system_msg.get('content', '').strip()
                user_content = user_msg.get('content', '').strip()
                assistant_response = assistant_msg.get('content', '').strip()
                
                # Parse thinking and answer
                thinking, answer = "", assistant_response
                if "<think>" in assistant_response and "</think>" in assistant_response:
                    start = assistant_response.find("<think>")
                    end = assistant_response.find("</think>")
                    if start != -1 and end != -1:
                        thinking = assistant_response[start + 7:end].strip()
                        answer = assistant_response[end + 8:].strip()
                
                # Base messages
                base = [
                    {"content": system_content, "role": "system"},
                    {"content": user_content, "role": "user"}
                ]
                inference_base = [
                    {"content": " Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process.", "role": "system"},
                    {"content": user_content, "role": "user"}
                ]
                # Format 4: Inference only (system + user prompts only) - 50k version
                inference_entry = {"messages": inference_base + [{"content": "", "role": "assistant"}]}
                inference_only_50k.append(inference_entry)
                
                # First 20k samples also go into the other formats
                if idx < 20000:
                    # Format 1: With thinking (full response)
                    with_thinking.append({
                        "messages": base + [{"content": assistant_response, "role": "assistant"}]
                    })
                    
                    # Format 2: No thinking (answer only)
                    no_thinking.append({
                        "messages": base + [{"content": answer, "role": "assistant"}]
                    })
                    
                    # Format 3: Comprehensive (all roles)
                    comprehensive_msgs = base + [{"content": assistant_response, "role": "assistant"}]
                    if thinking:
                        comprehensive_msgs.append({"content": thinking, "role": "assistant_thinking"})
                    if answer:
                        comprehensive_msgs.append({"content": answer, "role": "assistant_answer"})
                    comprehensive.append({"messages": comprehensive_msgs})
                    
                    # Format 4: Inference only - 20k version
                    inference_only.append(inference_entry)
            # Save files
            os.makedirs("data/step3_reasoning_model_training_data", exist_ok=True)
            os.makedirs("data/step0_preprocessed_data", exist_ok=True)
            
            files = [
                ("data/step0_preprocessed_data/processed_open_thoughts_20k_to_inference.jsonl", inference_only, "processed_open_thoughts_20k_to_inference"),
                ("data/step0_preprocessed_data/processed_open_thoughts_50k_to_inference.jsonl", inference_only_50k, "processed_open_thoughts_50k_to_inference"),
                ("data/step0_preprocessed_data/processed_open_thoughts_20k_r1.jsonl", comprehensive, "processed_open_thoughts_20k_r1"),
                ("data/step3_reasoning_model_training_data/open_thoughts_with_thinking_r1.jsonl", with_thinking, "open_thoughts_with_thinking"),
                ("data/step3_reasoning_model_training_data/open_thoughts_no_thinking_r1.jsonl", no_thinking, "open_thoughts_no_thinking"),
            ]
            
            # Load existing dataset_info.json if it exists
            dataset_info_path = "data/dataset_info.json"
            if os.path.exists(dataset_info_path):
                try:
                    with open(dataset_info_path, 'r', encoding='utf-8') as f:
                        dataset_info = json.load(f)
                except json.JSONDecodeError:
                    # Handle empty or corrupted JSON file by reinitializing
                    print(f"Warning: {dataset_info_path} is invalid JSON. Reinitializing it.")
                    dataset_info = {}
            else:
                dataset_info = {}
                os.makedirs("data", exist_ok=True)
            
            for filepath, data, info_key in files:
                if os.path.exists(filepath):
                    print(f"✓ {filepath} already exists, skipping.")
                    continue
                with open(filepath, 'w', encoding='utf-8') as f:
                    for item in tqdm(data, desc="Saving"):
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                # Add dataset info only if it doesn't already exist
                if info_key not in dataset_info:
                    relative_path = filepath[5:]
                    dataset_info[info_key] = {
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
                else:
                    print(f"✓ Dataset info for '{info_key}' already exists, skipping.")
            
            # Save updated dataset_info.json
            with open(dataset_info_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Processed {len(with_thinking)} samples (20k formats)")
            print(f"✓ Processed {len(inference_only_50k)} samples (50k inference format)")
            print(f"✓ Training datasets: data/step3_reasoning_model_training_data/")
            print(f"✓ Comprehensive format: data/step0_preprocessed_data/processed_open_thoughts_20k_r1.jsonl")
            print(f"✓ Inference format (20k): data/step0_preprocessed_data/processed_open_thoughts_20k_to_inference.jsonl")
            print(f"✓ Inference format (50k): data/step0_preprocessed_data/processed_open_thoughts_50k_to_inference.jsonl")
            print(f"✓ Updated dataset info: {dataset_info_path}")
            
        except Exception as e:
            print(f"Error: {e}")
def main():
    manager = OpenThoughtsDatasetManager()
    manager.process_open_thoughts_dataset()

if __name__ == "__main__":
    main()