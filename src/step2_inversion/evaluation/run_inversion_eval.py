#!/usr/bin/env python3
"""Run step2 inversion eval with normalized preset configs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

from evaluate_similarity import SimilarityMetrics, load_prediction_label_pairs


REPO_ROOT = Path(__file__).resolve().parents[3]
VLLM_INFER = REPO_ROOT / "llama_factory" / "scripts" / "vllm_infer.py"

COMMON_ARGS = {
    "template": "qwen",
    "dataset_dir": "data",
    "cutoff_len": 16384,
    "max_new_tokens": 8192,
    "max_samples": 10000,
    "batch_size": 1024,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "pipeline_parallel_size": 2,
    "vllm_config": {
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.9,
        "max_num_batched_tokens": 32768,
        "max_num_seqs": 256,
    },
}

PRESETS: Dict[str, Dict[str, str]] = {
    "surrogate_r1_distill_on_r1": {
        "model_name_or_path": "output/step2_inversion/models/inversion_qwen2_5_surrogate_r1_distill",
        "dataset": "open_thoughts_to_invert_r1_val",
        "save_name": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1_distill/r1.jsonl",
    },
    "surrogate_r1_distill_on_r1_distill": {
        "model_name_or_path": "output/step2_inversion/models/inversion_qwen2_5_surrogate_r1_distill",
        "dataset": "open_thoughts_to_invert_r1_distill_val",
        "save_name": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1_distill/r1_distill.jsonl",
    },
    "surrogate_r1_distill_on_chatgpt": {
        "model_name_or_path": "output/step2_inversion/models/inversion_qwen2_5_surrogate_r1_distill",
        "dataset": "open_thoughts_to_invert_chatgpt_val",
        "save_name": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1_distill/chatgpt.jsonl",
    },
    "surrogate_r1_distill_on_chatgpt_high_effort": {
        "model_name_or_path": "output/step2_inversion/models/inversion_qwen2_5_surrogate_r1_distill",
        "dataset": "open_thoughts_to_invert_chatgpt_high_effort_val",
        "save_name": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1_distill/chatgpt_high_effort.jsonl",
    },
    "surrogate_r1_on_chatgpt": {
        "model_name_or_path": "output/step2_inversion/models/inversion_qwen2_5_surrogate_r1",
        "dataset": "open_thoughts_to_invert_chatgpt_val",
        "save_name": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1/chatgpt.jsonl",
    },
    "surrogate_r1_no_bubble_on_r1": {
        "model_name_or_path": "output/step2_inversion/models/inversion_qwen2_5_surrogate_r1_no_bubble",
        "dataset": "open_thoughts_to_invert_r1_no_bubble_val",
        "save_name": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1_no_bubble/r1_no_bubble.jsonl",
    },
    "surrogate_r1_on_r1": {
        "model_name_or_path": "output/step2_inversion/models/inversion_qwen2_5_surrogate_r1",
        "dataset": "open_thoughts_to_invert_r1_val",
        "save_name": "output/step2_inversion/eval/inversion_qwen2_5_surrogate_r1/r1.jsonl",
    },
}


def build_command(
    preset_name: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> list[str]:
    if preset_name not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    config = {**COMMON_ARGS, **PRESETS[preset_name]}
    config["pipeline_parallel_size"] = pipeline_parallel_size
    config["vllm_config"] = {
        **COMMON_ARGS["vllm_config"],
        "tensor_parallel_size": tensor_parallel_size,
    }
    command = [sys.executable, str(VLLM_INFER)]
    for key, value in config.items():
        if key == "vllm_config":
            value = json.dumps(value)
        command.extend([f"--{key}", str(value)])
    return command


def run_similarity_eval(prediction_jsonl: Path, tokenizer_name: str, output_name: str) -> None:
    predictions, labels = load_prediction_label_pairs(str(prediction_jsonl))
    calculator = SimilarityMetrics(tokenizer_name=tokenizer_name)
    metrics = calculator.calculate(predictions, labels)

    print("Similarity metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    output_path = prediction_jsonl.parent / output_name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved similarity metrics to: {output_path}")


def ensure_preset_paths(preset_name: str) -> None:
    preset = PRESETS[preset_name]
    model_path = REPO_ROOT / preset["model_name_or_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # vllm_infer writes directly to save_name and does not create parent dirs.
    save_path = REPO_ROOT / preset["save_name"]
    save_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standardized vLLM eval presets.")
    parser.add_argument(
        "--preset",
        required=False,
        help="Preset name to execute. Use --list to show all options.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command only.")
    parser.add_argument("--list", action="store_true", help="List preset names and exit.")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=COMMON_ARGS["vllm_config"]["tensor_parallel_size"],
        help="vLLM tensor parallel size. For Qwen2-7B, use a divisor of 28 (e.g., 4 or 7).",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=COMMON_ARGS["pipeline_parallel_size"],
        help="vLLM pipeline parallel size. Use with tensor parallel to utilize all GPUs.",
    )
    parser.add_argument(
        "--with-metrics",
        action="store_true",
        help="Compute post-run similarity metrics (off by default).",
    )
    parser.add_argument(
        "--metrics-tokenizer",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer used for token-length stats in similarity metrics.",
    )
    parser.add_argument(
        "--metrics-output-name",
        default="similarity_metrics.json",
        help="Output filename for similarity metrics JSON (saved alongside prediction JSONL).",
    )
    args = parser.parse_args()

    if args.list:
        for name in sorted(PRESETS):
            print(name)
        return

    if not args.preset:
        parser.error("--preset is required unless --list is used.")

    ensure_preset_paths(args.preset)

    command = build_command(
        args.preset,
        args.tensor_parallel_size,
        args.pipeline_parallel_size,
    )
    print("Running command:")
    print(" ".join(command))
    if args.dry_run:
        return

    subprocess.run(command, check=True, cwd=REPO_ROOT)
    if not args.with_metrics:
        return

    save_name = PRESETS[args.preset]["save_name"]
    prediction_jsonl = REPO_ROOT / save_name
    if not prediction_jsonl.exists():
        raise FileNotFoundError(f"Prediction file not found after inference: {prediction_jsonl}")

    run_similarity_eval(
        prediction_jsonl=prediction_jsonl,
        tokenizer_name=args.metrics_tokenizer,
        output_name=args.metrics_output_name,
    )


if __name__ == "__main__":
    main()
