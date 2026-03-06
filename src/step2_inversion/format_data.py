#!/usr/bin/env python3
import argparse
from difflib import SequenceMatcher
import json
import os
import re

from tqdm import tqdm

SCENARIO_CONFIG = {
    "r1": {
        "prompt_input": "data/step0_preprocessed_data/processed_open_thoughts_20k_r1.jsonl",
        "bubble_input": "output/step1_summarization/bubbles_r1.jsonl",
        "reasoning_output": "data/step3_reasoning_model_training_data/open_thoughts_with_bubble_r1.jsonl",
        "inversion_output": "data/step2_inversion_model_training_data/open_thoughts_to_invert_r1.jsonl",
        "use_bubbles": True,
        "create_reasoning_data": True,
        "fixed_train_count": 10000,
    },
    "r1_no_bubble": {
        "prompt_input": "data/step0_preprocessed_data/processed_open_thoughts_20k_r1.jsonl",
        "bubble_input": None,
        "reasoning_output": None,
        "inversion_output": "data/step2_inversion_model_training_data/open_thoughts_to_invert_r1_no_bubble.jsonl",
        "use_bubbles": False,
        "create_reasoning_data": False,
        "fixed_train_count": 10000,
    },
    "r1_distill": {
        "prompt_input": "data/step0_preprocessed_data/processed_open_thoughts_20k_r1_distill.jsonl",
        "bubble_input": "output/step1_summarization/bubbles_r1_distill.jsonl",
        "reasoning_output": "data/step3_reasoning_model_training_data/open_thoughts_with_bubble_r1_distill.jsonl",
        "inversion_output": "data/step2_inversion_model_training_data/open_thoughts_to_invert_r1_distill.jsonl",
        "use_bubbles": True,
        "create_reasoning_data": True,
        "fixed_train_count": 10000,
    },
}


class InversionDataFormatter:
    """Format inversion training data for multiple scenarios."""

    def __init__(self):
        self.system_prompt_with_bubble = """You are a language model that reconstructs full internal reasoning traces from high-level bubble summaries.

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

Use informal, introspective language as if the person is thinking out loud. Add math expressions in \\( \\LaTeX \\) where appropriate.

Do **not** invent new reasoning steps outside the bubbles. Use the **input** and **output** only for context and consistency. Your goal is to **flesh out the bubbles**, not to re-solve the problem from scratch.

The full trace should:
- Be logically consistent and cohesive from start to finish
- Sound like a realistic thought process that could plausibly result in the given answer
- Span multiple paragraphs per bubble and up to 20,000 characters overall if needed
- Be output as one continuous trace, wrapped in `<think>...</think>` tags

You are not summarizing the bubbles. You are recovering the internal narrative that *generated* them.
"""
        self.system_prompt_no_bubble = """You are a language model that reconstructs full internal reasoning traces from only an **input**
(e.g., a math or logic problem) and a corresponding **output** (final solution or answer).

You will be given:
- A problem **input**
- A final **output** or solution

Your task is to reconstruct the full internal reasoning process that could plausibly connect the input
to the output. This should be a long, detailed, introspective trace, not a short summary.

Guidelines for the reasoning trace:
- Write in the style of an informal, introspective monologue, as if the person is thinking out loud.
- Include assumptions, intuitions, and background facts as they arise naturally.
- Show intermediate steps, calculations, logical deductions, definitions, and subcases.
- Raise natural questions or doubts during reasoning, and explain how they are resolved.
- Explore alternative approaches, even ones that are discarded, and explain why.
- Make transitions clear so the reasoning feels like a coherent train of thought.
- Use \\( \\LaTeX \\) for math expressions where helpful.
- Do not introduce new information inconsistent with the input or output.
- The goal is depth, not brevity: expand ideas fully, elaborate with multiple paragraphs, and let the reasoning unfold gradually.

Formatting:
- Wrap the full reasoning trace in <think>...</think> tags.
"""

    def format_data(
        self,
        prompt_input,
        bubble_input,
        reasoning_output,
        inversion_output,
        use_bubbles,
        create_reasoning_data,
        fixed_train_count=10000,
        debug_mismatch_output=None,
    ):
        reasoning_exists = (not reasoning_output) or os.path.exists(reasoning_output)
        inversion_exists = os.path.exists(inversion_output)
        if reasoning_exists and inversion_exists:
            print("✓ All output files already exist. Skipping processing.")
            return

        print(f"Loading data from {prompt_input}...")
        with open(prompt_input, "r", encoding="utf-8") as f:
            input_data = [json.loads(line) for line in f if line.strip()]

        bubble_data = None
        paired_entries = None
        if use_bubbles:
            with open(bubble_input, "r", encoding="utf-8") as f:
                bubble_data = [json.loads(line) for line in f if line.strip()]
            print(f"Found {len(input_data)} input entries and {len(bubble_data)} bubble entries to process")
            paired_entries = self._pair_by_thinking_content(
                input_data=input_data,
                bubble_data=bubble_data,
                debug_mismatch_output=debug_mismatch_output,
            )
            print(f"✓ Matched {len(paired_entries)} entries by thinking content")
        else:
            print(f"Found {len(input_data)} input entries to process")

        if create_reasoning_data and (not reasoning_exists):
            self._create_bubble_reasoning_data(paired_entries, reasoning_output)
            self._update_dataset_info(reasoning_output)
        if not inversion_exists:
            self._create_inversion_training_data(
                input_data=input_data,
                paired_entries=paired_entries,
                output_file=inversion_output,
                use_bubbles=use_bubbles,
                fixed_train_count=fixed_train_count,
            )
            self._update_dataset_info_for_split_outputs(inversion_output)

    def _create_bubble_reasoning_data(self, paired_entries, output_file):
        print("Creating reasoning training data...")
        formatted_data = []
        for input_item, bubble_item in tqdm(paired_entries, desc="Creating reasoning data"):
            user_prompt = self._extract_role(input_item, "user")
            system_prompt = self._extract_role(input_item, "system")
            assistant_answer = self._extract_assistant_answer(input_item)
            bubbles_output = self._extract_bubbles_output(bubble_item)
            if not (user_prompt and user_prompt.strip()):
                continue
            thinking_content = (
                f"<|begin_of_thought|>\n\n{bubbles_output}\n\n<|end_of_thought|>\n\n" if bubbles_output else ""
            )
            full_assistant_content = (
                f"{thinking_content}<|begin_of_solution|>\n\n{assistant_answer}\n\n<|end_of_solution|>"
            ).strip()
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
        print(f"✓ Output saved to: {output_file}")

    def _create_inversion_training_data(
        self,
        input_data,
        paired_entries,
        output_file,
        use_bubbles,
        fixed_train_count=10000,
    ):
        print("Creating inversion model training data...")
        inversion_data = []
        skipped_entries = []

        iterator = paired_entries if use_bubbles else ((item, None) for item in input_data)
        for idx, (input_item, bubble_item) in enumerate(tqdm(iterator, desc="Creating inversion data")):
            thinking_content = self._extract_role(input_item, "assistant_thinking")
            user_prompt = self._extract_role(input_item, "user")
            assistant_answer = self._extract_assistant_answer(input_item)
            bubbles_output = self._extract_bubbles_output(bubble_item) if use_bubbles else ""

            missing_thinking = not (thinking_content and thinking_content.strip())
            missing_bubbles = use_bubbles and not (bubbles_output and bubbles_output.strip())
            if missing_thinking or missing_bubbles:
                reason = []
                if missing_thinking:
                    reason.append("missing_thinking")
                if missing_bubbles:
                    reason.append("missing_bubbles")
                skipped_entries.append(
                    {
                        "index": idx,
                        "reason": " + ".join(reason),
                        "user_prompt_preview": (user_prompt[:100] + "...") if user_prompt and len(user_prompt) > 100 else user_prompt,
                    }
                )
                continue

            system_prompt = self.system_prompt_with_bubble if use_bubbles else self.system_prompt_no_bubble
            user_content = (
                f"The original problem input is: {user_prompt}\n"
                f"The final answer is: {assistant_answer}\n"
            )
            if use_bubbles:
                user_content += (
                    "Transform these thinking bubbles into clear full reasoning traces: <think>\n"
                    f"{bubbles_output}\n"
                    "</think>"
                )
            else:
                user_content += "Generate full reasoning traces."

            inversion_data.append(
                {
                    "messages": [
                        {"content": system_prompt, "role": "system"},
                        {"content": user_content, "role": "user"},
                        {"content": thinking_content, "role": "assistant"},
                    ]
                }
            )

        if skipped_entries:
            print(f"\n⚠️  Skipped {len(skipped_entries)} entries:")
            for entry in skipped_entries[:20]:
                print(f"  Index {entry['index']}: {entry['reason']}")
                print(f"    Preview: {entry['user_prompt_preview']}")
            if len(skipped_entries) > 20:
                print(f"  ... and {len(skipped_entries) - 20} more")
        else:
            print("✓ No entries were skipped")

        train_data, val_data = self._split_data(
            inversion_data=inversion_data,
            fixed_train_count=fixed_train_count,
        )

        base_path = output_file.rsplit(".", 1)[0]
        extension = output_file.rsplit(".", 1)[1] if "." in output_file else "jsonl"
        train_output_file = f"{base_path}_train.{extension}"
        val_output_file = f"{base_path}_val.{extension}"

        os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
        with open(train_output_file, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        os.makedirs(os.path.dirname(val_output_file), exist_ok=True)
        with open(val_output_file, "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✓ Created {len(train_data)} training samples and {len(val_data)} validation samples")
        print(f"✓ Training data saved to: {train_output_file}")
        print(f"✓ Validation data saved to: {val_output_file}")

    @staticmethod
    def _split_data(inversion_data, fixed_train_count=10000):
        train_count = fixed_train_count or 10000
        return inversion_data[:train_count], inversion_data[train_count:]

    @staticmethod
    def _extract_role(item, role):
        for message in item.get("messages", []):
            if message.get("role") == role:
                return message.get("content", "").strip()
        return ""

    @staticmethod
    def _extract_assistant_answer(item):
        answer = InversionDataFormatter._extract_role(item, "assistant_answer")
        return answer if answer else InversionDataFormatter._extract_role(item, "assistant")

    @staticmethod
    def _extract_bubbles_output(item):
        if isinstance(item, dict):
            return item.get("predict", "").strip()
        if isinstance(item, list) and item:
            return item[0].get("predict", "").strip()
        return ""

    def _pair_by_thinking_content(self, input_data, bubble_data, debug_mismatch_output=None):
        if len(input_data) != len(bubble_data):
            print(
                f"⚠️  Length mismatch: input={len(input_data)}, bubble={len(bubble_data)}. "
                "Matching only aligned prefix."
            )

        paired_entries = []
        mismatch_count = 0
        missing_thinking_in_bubble = 0
        mismatch_debug_records = []
        missing_parse_debug_records = []

        for idx, (input_item, bubble_item) in enumerate(zip(input_data, bubble_data)):
            input_thinking = self._canonicalize_thinking(self._extract_role(input_item, "assistant_thinking"))
            bubble_thinking = self._canonicalize_thinking(self._extract_thinking_from_bubble_prompt(bubble_item))

            if not bubble_thinking:
                missing_thinking_in_bubble += 1
                if len(missing_parse_debug_records) < 200:
                    missing_parse_debug_records.append(
                        {
                            "index": idx,
                            "input_user_preview": self._extract_role(input_item, "user")[:200],
                            "bubble_prompt_preview": str(bubble_item.get("prompt", ""))[:500],
                        }
                    )
                continue

            is_match, sim = self._is_same_thinking(input_thinking, bubble_thinking)
            if is_match:
                paired_entries.append((input_item, bubble_item))
            else:
                mismatch_count += 1
                if len(mismatch_debug_records) < 200:
                    mismatch_debug_records.append(
                        {
                            "index": idx,
                            "similarity": round(sim, 4),
                            "input_user_preview": self._extract_role(input_item, "user")[:200],
                            "input_thinking_preview": input_thinking[:400],
                            "bubble_thinking_preview": bubble_thinking[:400],
                        }
                    )

        if mismatch_count:
            print(f"⚠️  Dropped {mismatch_count} entries due to thinking mismatch.")
        if missing_thinking_in_bubble:
            print(f"⚠️  Could not parse thinking from bubble prompt for {missing_thinking_in_bubble} entries.")
        if debug_mismatch_output:
            self._write_debug_records(debug_mismatch_output, mismatch_debug_records)
            print(f"✓ Wrote mismatch debug records to: {debug_mismatch_output}")
            missing_path = f"{debug_mismatch_output}.missing_parse.jsonl"
            self._write_debug_records(missing_path, missing_parse_debug_records)
            print(f"✓ Wrote missing-parse debug records to: {missing_path}")

        return paired_entries

    @staticmethod
    def _extract_thinking_from_bubble_prompt(bubble_item):
        if not isinstance(bubble_item, dict):
            return ""

        prompt = str(bubble_item.get("prompt", ""))
        if not prompt:
            return ""

        # Support two prompt templates:
        # 1) special-token style: <|im_start|>user ... <|im_end|>
        # 2) plain text style: system\n...user\n...assistant\n...
        user_text = prompt
        token_style = re.search(r"<\|im_start\|>user\s*\n(.*?)<\|im_end\|>", prompt, flags=re.DOTALL)
        if token_style:
            user_text = token_style.group(1)
        else:
            plain_style = re.search(r"^system\n.*?\nuser\n(.*?)(?:\nassistant\n|$)", prompt, flags=re.DOTALL)
            if plain_style:
                user_text = plain_style.group(1)

        marker = "Transform this thinking process into clear reasoning bubbles:<think>"
        marker_pos = user_text.rfind(marker)
        if marker_pos != -1:
            content = user_text[marker_pos + len(marker):]
        else:
            # There may be an instructional "<think>...</think>" in the system prompt.
            # Use the LAST <think> so we pick the user trace, not instruction text.
            think_start = user_text.rfind("<think>")
            source = user_text
            if think_start == -1:
                think_start = prompt.rfind("<think>")
                source = prompt
            if think_start == -1:
                return ""
            content = source[think_start + len("<think>"):]

        think_end = content.rfind("</think>")
        if think_end != -1:
            content = content[:think_end]
        content = content.strip()
        return "" if not content or content == "..." else content

    @staticmethod
    def _normalize_text(text):
        return " ".join((text or "").strip().split())

    @staticmethod
    def _canonicalize_thinking(text):
        if not text:
            return ""
        value = str(text)
        value = value.replace("\\n", "\n")
        value = re.sub(r"</?think>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"<\|begin_of_thought\|>|<\|end_of_thought\|>", "", value, flags=re.IGNORECASE)
        value = re.sub(r"<\|im_start\|>assistant\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"<\|im_end\|>", "", value, flags=re.IGNORECASE)
        return InversionDataFormatter._normalize_text(value)

    @staticmethod
    def _head_tail(text, window=800):
        if len(text) <= window * 2:
            return text
        return text[:window] + " " + text[-window:]

    @staticmethod
    def _is_same_thinking(left, right):
        if not left or not right:
            return False, 0.0
        if left == right:
            return True, 1.0

        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        length_ratio = len(shorter) / max(len(longer), 1)
        if shorter in longer and length_ratio >= 0.9:
            return True, 1.0

        # Handle truncated wrappers where starts match strongly but one side is shorter.
        prefix_window = 1200
        prefix_ratio = SequenceMatcher(None, left[:prefix_window], right[:prefix_window]).ratio()
        if prefix_ratio >= 0.98 and min(len(left), len(right)) >= 400:
            return True, prefix_ratio

        # Compare only head+tail windows to stay robust and efficient for long traces.
        ratio = SequenceMatcher(None, InversionDataFormatter._head_tail(left), InversionDataFormatter._head_tail(right)).ratio()
        return ratio >= 0.9, ratio

    @staticmethod
    def _write_debug_records(path, records):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
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

    def _update_dataset_info_for_split_outputs(self, inversion_output):
        base_path = inversion_output.rsplit(".", 1)[0]
        extension = inversion_output.rsplit(".", 1)[1] if "." in inversion_output else "jsonl"
        train_output_file = f"{base_path}_train.{extension}"
        val_output_file = f"{base_path}_val.{extension}"
        self._update_dataset_info(train_output_file)
        self._update_dataset_info(val_output_file)

    def _update_dataset_info(self, output_file):
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
    parser = argparse.ArgumentParser(description="Format inversion data for multiple scenarios")
    parser.add_argument("--scenario", choices=sorted(SCENARIO_CONFIG.keys()), default="r1")
    parser.add_argument("--prompt_input", help="Override prompt input path")
    parser.add_argument("--bubble_input", help="Override bubble input path")
    parser.add_argument("--reasoning_output", help="Override reasoning output path")
    parser.add_argument("--inversion_output", help="Override inversion output path")
    parser.add_argument("--fixed_train_count", type=int, help="Override fixed train sample count")
    parser.add_argument(
        "--debug_mismatch_output",
        help="Optional path to write mismatch debug JSONL (stores up to 200 mismatch samples)",
    )
    args = parser.parse_args()

    config = dict(SCENARIO_CONFIG[args.scenario])
    if args.prompt_input:
        config["prompt_input"] = args.prompt_input
    if args.bubble_input:
        config["bubble_input"] = args.bubble_input
    if args.reasoning_output:
        config["reasoning_output"] = args.reasoning_output
    if args.inversion_output:
        config["inversion_output"] = args.inversion_output
    if args.fixed_train_count is not None:
        config["fixed_train_count"] = args.fixed_train_count

    formatter = InversionDataFormatter()
    formatter.format_data(
        prompt_input=config["prompt_input"],
        bubble_input=config["bubble_input"],
        reasoning_output=config["reasoning_output"],
        inversion_output=config["inversion_output"],
        use_bubbles=config["use_bubbles"],
        create_reasoning_data=config["create_reasoning_data"],
        fixed_train_count=config.get("fixed_train_count"),
        debug_mismatch_output=args.debug_mismatch_output,
    )


if __name__ == "__main__":
    main()