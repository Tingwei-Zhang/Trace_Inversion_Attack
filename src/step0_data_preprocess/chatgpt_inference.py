import json
import os
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
from typing import Dict, Optional

client = AsyncOpenAI()

def format_prompt(system_content: str, user_content: str) -> str:
    """Format prompt to match `vllm_infer.py` style."""
    boxed_answer_instruction = (
        "Think thoroughly and step-by-step before answering.\n\n"
        "If the task expects a final numerical/symbolic answer (math/logic):\n"
        "- Fully evaluate arithmetic.\n"
        "- Output ONLY the final answer on the last line in the form: \\boxed{...}\n"
        "- Do not write anything after the boxed answer.\n\n"
        "If the task expects code (programming):\n"
        "- Output ONLY the code (no explanations, no markdown, no \\boxed{}).\n"
        "- The code should be complete and runnable/compilable as required by the prompt."
    )
    system_block = system_content.strip()
    if system_block:
        system_block = f"{system_block}\n\n{boxed_answer_instruction}"
    else:
        system_block = boxed_answer_instruction

    return (
        "<|im_start|>system\n"
        f"{system_block}\n"
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_start|>assistant\n"
    )

async def process_single_entry(line: str, semaphore: asyncio.Semaphore, pbar: tqdm) -> Optional[Dict]:
    """Process a single JSONL line and run one API call."""
    if not line.strip():
        return None
    
    async with semaphore:  # Limit concurrent requests
        data = json.loads(line.strip())

        # Extract system and user messages
        system_content = ""
        user_content = ""

        for msg in data.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content

        if not user_content:
            pbar.write("Warning: No user content found in entry, skipping")
            return None

        # Format prompt for saving (matching vllm_infer.py format)
        prompt_text = format_prompt(system_content, user_content)

        # Single API call without retry/error handling
        resp = await client.responses.create(
            model="gpt-5-mini-2025-08-07",
            input=prompt_text,
            reasoning={"effort": "medium", "summary": "auto"},
            store=True,
        )

        # Extract final output
        output = getattr(resp, "output_text", "") or ""

        # Extract reasoning summary
        reasoning_summary = ""
        output_items = getattr(resp, "output", None) or []
        for item in output_items:
            if getattr(item, "type", None) == "reasoning" and getattr(
                item, "summary", None
            ):
                reasoning_summary = "\n".join(s.text for s in item.summary)
                break

        # Extract token usage (including reasoning tokens)
        input_tokens = 0
        output_tokens = 0
        usage = getattr(resp, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0

        result = {
            "prompt": prompt_text,
            "summary": reasoning_summary,
            "output": output,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

        pbar.update(1)
        return result

async def process_dataset_async(
    input_file: str,
    output_file: str,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    max_concurrent: int = 10,
    batch_size: int = 100,
) -> None:
    """Process a JSONL dataset and run ChatGPT inference concurrently."""
    print(f"Processing dataset from: {input_file}")
    print(f"Output will be saved to: {output_file}")
    print(f"Start index: {start_index}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Batch size for writing: {batch_size}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = lines[start_index:]
        if max_samples:
            lines = lines[:max_samples]
    
    # Filter out empty lines
    valid_lines = [l for l in lines if l.strip()]
    total_lines = len(valid_lines)
    print(f"Total entries to process: {total_lines}")
    
    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Progress bar
    pbar = tqdm(total=total_lines, desc="Processing entries")
    
    # Process all entries concurrently
    tasks = [process_single_entry(line, semaphore, pbar) for line in valid_lines]

    results = []
    errors = 0
    batch_results = []

    with open(output_file, "w", encoding="utf-8") as output_f:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is None:
                errors += 1
                continue

            batch_results.append(result)
            results.append(result)

            if not result.get("output", ""):
                errors += 1

            if len(batch_results) >= batch_size:
                for res in batch_results:
                    output_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                output_f.flush()
                batch_results = []

        if batch_results:
            for res in batch_results:
                output_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            output_f.flush()

    pbar.close()

    print(f"\n✓ Processing complete!")
    print(f"✓ Successfully processed: {len(results)} entries")
    print(f"✓ Errors: {errors} entries")
    print(f"✓ Results saved to: {output_file}")

def process_dataset(
    input_file: str,
    output_file: str,
    start_index: int = 0,
    max_samples: Optional[int] = None,
    max_concurrent: int = 10,
    batch_size: int = 100,
) -> None:
    """Synchronous wrapper around the async dataset processor."""
    asyncio.run(
        process_dataset_async(
            input_file=input_file,
            output_file=output_file,
            start_index=start_index,
            max_samples=max_samples,
            max_concurrent=max_concurrent,
            batch_size=batch_size,
        )
    )

def main() -> None:
    """CLI entrypoint for running inference on the preprocessed dataset."""
    input_file = (
        "data/step0_preprocessed_data/processed_open_thoughts_20k_to_inference.jsonl"
    )
    output_file = (
        "data/step0_preprocessed_data/chatgpt_inference_raw_20k_gpt_5_mini.jsonl"
    )

    # Configuration (tweak here for local runs)
    start_index = 20000
    max_samples = 10000
    max_concurrent = 50
    batch_size = 100

    process_dataset(
        input_file=input_file,
        output_file=output_file,
        start_index=start_index,
        max_samples=max_samples,
        max_concurrent=max_concurrent,
        batch_size=batch_size,
    )

if __name__ == "__main__":
    main()