#!/usr/bin/env python3
import argparse
from collections import defaultdict, deque
from difflib import SequenceMatcher
import json
import os
import re
from typing import Any

from tqdm import tqdm


PRESET_CONFIG = {
    "r1_distill_on_r1": {
        "prompt_input": "data/step0_preprocessed_data/processed_open_thoughts_20k_r1.jsonl",
        "inversion_input": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1_distill/r1.jsonl",
        "reasoning_output": "data/step3_reasoning_model_training_data/open_thoughts_with_inverted_thinking_r1_distill_on_r1.jsonl",
        "prompt_start_idx": 10000,
    },
    "r1_on_r1": {
        "prompt_input": "data/step0_preprocessed_data/processed_open_thoughts_20k_r1.jsonl",
        "inversion_input": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1/r1.jsonl",
        "reasoning_output": "data/step3_reasoning_model_training_data/open_thoughts_with_inverted_thinking_r1_on_r1.jsonl",
        "prompt_start_idx": 10000,
    },
    "r1_no_bubble_on_r1": {
        "prompt_input": "data/step0_preprocessed_data/processed_open_thoughts_20k_r1.jsonl",
        "inversion_input": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1_no_bubble/r1.jsonl",
        "reasoning_output": "data/step3_reasoning_model_training_data/open_thoughts_with_inverted_thinking_r1_no_bubble_on_r1.jsonl",
        "prompt_start_idx": 10000,
    },
    "r1_distill_on_chatgpt": {
        "prompt_input": "data/step0_preprocessed_data/processed_open_thoughts_20k_chatgpt.jsonl",
        "inversion_input": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1_distill/chatgpt.jsonl",
        "reasoning_output": "data/step3_reasoning_model_training_data/open_thoughts_with_inverted_thinking_r1_distill_on_chatgpt.jsonl",
        "prompt_start_idx": 10000,
    },
    "r1_on_chatgpt": {
        "prompt_input": "data/step0_preprocessed_data/processed_open_thoughts_20k_chatgpt.jsonl",
        "inversion_input": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1/chatgpt.jsonl",
        "reasoning_output": "data/step3_reasoning_model_training_data/open_thoughts_with_inverted_thinking_r1_on_chatgpt.jsonl",
        "prompt_start_idx": 0,
    },
}


class ReasoningDataFormatter:
    """Format step3 reasoning data using matched inversion outputs."""

    def format_data(
        self,
        prompt_input: str,
        inversion_input: str,
        reasoning_output: str,
        debug_mismatch_output: str | None = None,
        prompt_start_idx: int = 0,
    ):
        if os.path.exists(reasoning_output):
            print(f"✓ Output file {reasoning_output} already exists. Skipping processing.")
            return

        print(f"Loading data from {prompt_input} and {inversion_input}...")
        with open(prompt_input, "r", encoding="utf-8") as f:
            input_data = [json.loads(line) for line in f if line.strip()]
        if prompt_start_idx > 0:
            input_data = input_data[prompt_start_idx:]
        with open(inversion_input, "r", encoding="utf-8") as f:
            inversion_data = [json.loads(line) for line in f if line.strip()]
        print(
            f"Found {len(input_data)} input entries and {len(inversion_data)} inversion entries to process "
            f"(prompt_start_idx={prompt_start_idx})"
        )

        paired_entries = self._pair_by_thinking_content(
            input_data=input_data,
            inversion_data=inversion_data,
            debug_mismatch_output=debug_mismatch_output,
        )
        print(f"✓ Matched {len(paired_entries)} entries by thinking content")

        self._create_reasoning_data(paired_entries=paired_entries, output_file=reasoning_output)
        self._update_dataset_info(reasoning_output)

    def _create_reasoning_data(self, paired_entries: list[tuple[dict[str, Any], dict[str, Any]]], output_file: str):
        print("Creating reasoning training data...")
        formatted_data = []
        skipped_entries = 0
        for input_item, inversion_item in tqdm(paired_entries, desc="Formatting"):
            user_prompt = self._extract_role(input_item, "user")
            system_prompt = self._extract_role(input_item, "system")
            assistant_answer = self._extract_assistant_answer(input_item)
            inversion_output = self._extract_inversion_output(inversion_item)

            if not (user_prompt and assistant_answer and inversion_output):
                skipped_entries += 1
                continue

            thinking_content = f"<think>\n{inversion_output}\n</think>\n"
            full_assistant_content = f"{thinking_content}\n{assistant_answer}".strip()
            formatted_data.append(
                {
                    "messages": [
                        {"content": system_prompt, "role": "system"},
                        {"content": user_prompt, "role": "user"},
                        {"content": full_assistant_content, "role": "assistant"},
                    ]
                }
            )

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for item in formatted_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✓ Created {len(formatted_data)} samples for reasoning training")
        if skipped_entries:
            print(f"⚠️  Skipped {skipped_entries} entries due to missing prompt/answer/prediction")
        print(f"✓ Output saved to: {output_file}")

    def _pair_by_thinking_content(
        self,
        input_data: list[dict[str, Any]],
        inversion_data: list[dict[str, Any]],
        debug_mismatch_output: str | None = None,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        input_thinkings = []
        exact_index: dict[str, deque[int]] = defaultdict(deque)
        for idx, item in enumerate(input_data):
            canon = self._canonicalize_thinking(self._extract_role(item, "assistant_thinking"))
            input_thinkings.append(canon)
            if canon:
                exact_index[canon].append(idx)

        used_input_indices = set()
        paired_entries = []
        mismatch_debug_records = []
        missing_reference_count = 0
        mismatch_count = 0

        for idx, inversion_item in enumerate(tqdm(inversion_data, desc="Matching entries")):
            inversion_reference = self._canonicalize_thinking(self._extract_reference_thinking(inversion_item))
            if not inversion_reference:
                missing_reference_count += 1
                if len(mismatch_debug_records) < 200:
                    mismatch_debug_records.append(
                        {
                            "index": idx,
                            "reason": "missing_inversion_reference_thinking",
                            "inversion_prompt_preview": str(inversion_item.get("prompt", ""))[:500],
                        }
                    )
                continue

            matched_input_idx = None
            candidates = exact_index.get(inversion_reference)
            while candidates:
                candidate_idx = candidates.popleft()
                if candidate_idx not in used_input_indices:
                    matched_input_idx = candidate_idx
                    break

            if matched_input_idx is None:
                best_idx = None
                best_sim = 0.0
                for input_idx, input_thinking in enumerate(input_thinkings):
                    if input_idx in used_input_indices:
                        continue
                    is_match, sim = self._is_same_thinking(input_thinking, inversion_reference)
                    if is_match and sim > best_sim:
                        best_sim = sim
                        best_idx = input_idx
                        if sim >= 0.999:
                            break
                matched_input_idx = best_idx

            if matched_input_idx is None:
                mismatch_count += 1
                if len(mismatch_debug_records) < 200:
                    mismatch_debug_records.append(
                        {
                            "index": idx,
                            "reason": "no_matching_preprocessed_entry",
                            "inversion_reference_preview": inversion_reference[:400],
                            "inversion_prompt_preview": str(inversion_item.get("prompt", ""))[:500],
                        }
                    )
                continue

            used_input_indices.add(matched_input_idx)
            paired_entries.append((input_data[matched_input_idx], inversion_item))

        if missing_reference_count:
            print(f"⚠️  Missing inversion reference thinking for {missing_reference_count} entries.")
        if mismatch_count:
            print(f"⚠️  Could not match {mismatch_count} inversion entries to preprocessed data.")
        if debug_mismatch_output:
            self._write_debug_records(debug_mismatch_output, mismatch_debug_records)
            print(f"✓ Wrote mismatch debug records to: {debug_mismatch_output}")

        return paired_entries

    @staticmethod
    def _extract_role(item: dict[str, Any], role: str) -> str:
        for message in item.get("messages", []):
            if message.get("role") == role:
                return message.get("content", "").strip()
        return ""

    @staticmethod
    def _extract_assistant_answer(item: dict[str, Any]) -> str:
        answer = ReasoningDataFormatter._extract_role(item, "assistant_answer")
        return answer if answer else ReasoningDataFormatter._extract_role(item, "assistant")

    @staticmethod
    def _extract_inversion_output(item: dict[str, Any]) -> str:
        if isinstance(item, dict):
            return str(item.get("predict", "")).strip()
        if isinstance(item, list) and item:
            return str(item[0].get("predict", "")).strip()
        return ""

    def _extract_reference_thinking(self, inversion_item: dict[str, Any]) -> str:
        label = str(inversion_item.get("label", "")).strip()
        if label:
            return label

        prompt = str(inversion_item.get("prompt", ""))
        marker = "Transform these thinking bubbles into clear full reasoning traces: <think>"
        marker_pos = prompt.rfind(marker)
        if marker_pos != -1:
            content = prompt[marker_pos + len(marker) :]
            think_end = content.rfind("</think>")
            if think_end != -1:
                content = content[:think_end]
            return content.strip()

        think_start = prompt.rfind("<think>")
        if think_start == -1:
            return ""
        content = prompt[think_start + len("<think>") :]
        think_end = content.rfind("</think>")
        if think_end != -1:
            content = content[:think_end]
        return content.strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").strip().split())

    @staticmethod
    def _canonicalize_thinking(text: str) -> str:
        if not text:
            return ""
        value = str(text)
        value = value.replace("\\n", "\n")
        value = re.sub(r"</?think>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"<\|begin_of_thought\|>|<\|end_of_thought\|>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"<\|im_start\|>assistant\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"<\|im_end\|>", "", value, flags=re.IGNORECASE)
        return ReasoningDataFormatter._normalize_text(value)

    @staticmethod
    def _head_tail(text: str, window: int = 800) -> str:
        if len(text) <= window * 2:
            return text
        return text[:window] + " " + text[-window:]

    @staticmethod
    def _is_same_thinking(left: str, right: str) -> tuple[bool, float]:
        if not left or not right:
            return False, 0.0
        if left == right:
            return True, 1.0

        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        length_ratio = len(shorter) / max(len(longer), 1)
        if shorter in longer and length_ratio >= 0.9:
            return True, 1.0

        prefix_window = 1200
        prefix_ratio = SequenceMatcher(None, left[:prefix_window], right[:prefix_window]).ratio()
        if prefix_ratio >= 0.98 and min(len(left), len(right)) >= 400:
            return True, prefix_ratio

        ratio = SequenceMatcher(None, ReasoningDataFormatter._head_tail(left), ReasoningDataFormatter._head_tail(right)).ratio()
        return ratio >= 0.9, ratio

    @staticmethod
    def _write_debug_records(path: str, records: list[dict[str, Any]]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _dataset_entry(file_name: str):
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

    def _update_dataset_info(self, output_file: str):
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

        dataset_info[dataset_name] = self._dataset_entry(relative_path)
        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        print(f"✓ Updated {dataset_info_path} with entry '{dataset_name}'")


def main():
    parser = argparse.ArgumentParser(description="Format step3 reasoning data with robust inversion matching")
    parser.add_argument("--preset", choices=sorted(PRESET_CONFIG.keys()), default="r1_distill_on_r1")
    parser.add_argument("--prompt_input", help="Override prompt input path")
    parser.add_argument("--inversion_input", help="Override inversion input path")
    parser.add_argument("--reasoning_output", help="Override reasoning output path")
    parser.add_argument("--prompt_start_idx", type=int, help="Start index for prompt_input slicing before matching")
    parser.add_argument(
        "--debug_mismatch_output",
        help="Optional path to write mismatch debug JSONL (stores up to 200 mismatch samples)",
    )
    args = parser.parse_args()

    config = dict(PRESET_CONFIG[args.preset])
    if args.prompt_input:
        config["prompt_input"] = args.prompt_input
    if args.inversion_input:
        config["inversion_input"] = args.inversion_input
    if args.reasoning_output:
        config["reasoning_output"] = args.reasoning_output
    if args.prompt_start_idx is not None:
        config["prompt_start_idx"] = args.prompt_start_idx

    formatter = ReasoningDataFormatter()
    formatter.format_data(
        prompt_input=config["prompt_input"],
        inversion_input=config["inversion_input"],
        reasoning_output=config["reasoning_output"],
        debug_mismatch_output=args.debug_mismatch_output,
        prompt_start_idx=config.get("prompt_start_idx", 0),
    )


if __name__ == "__main__":
    main()