#!/usr/bin/env python3
"""
Simple evaluation script that runs the same command as the shell scripts
but allows you to modify the model name and task name.
"""

import os
import subprocess
import argparse
import json


def _count_visible_gpus(cuda_visible_devices):
    if cuda_visible_devices:
        return len([gpu for gpu in cuda_visible_devices.split(",") if gpu.strip()])
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len([line for line in output.splitlines() if line.startswith("GPU ")])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 1


def _auto_tensor_parallel_size(visible_gpu_count, model_path):
    """Pick largest TP that is likely safe for this model config."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return visible_gpu_count

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return visible_gpu_count

    num_heads = config.get("num_attention_heads")
    vocab_size = config.get("vocab_size")
    for tp in range(visible_gpu_count, 0, -1):
        if isinstance(num_heads, int) and num_heads % tp != 0:
            continue
        if isinstance(vocab_size, int) and vocab_size % tp != 0:
            continue
        return tp
    return 1


def run_evaluation(
    model_name,
    tasks,
    dry_run=False,
    cuda_visible_devices=None,
    tensor_parallel_size=None,
):
    """
    Run evaluation with the specified model and tasks.
    
    Args:
        model_name: Name of the model (e.g., "meta-llama/Llama-3.2-3B-Instruct")
        tasks: Comma-separated task names (e.g., "AMC23,AIME25,LiveCodeBench,GPQADiamond")
        dry_run: If True, only print the command without executing
    """
    repo_root = os.path.abspath(os.getcwd())
    evalchemy_dir = os.path.join(repo_root, "evalchemy")

    # Resolve local model paths from repo root so subprocess cwd does not break loading.
    resolved_model = model_name
    if not os.path.isabs(model_name) and not model_name.startswith("./") and not model_name.startswith("../"):
        candidate = os.path.join(repo_root, model_name)
        if os.path.exists(candidate):
            resolved_model = candidate
    elif not os.path.isabs(model_name):
        resolved_model = os.path.abspath(os.path.join(repo_root, model_name))

    # Set environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{repo_root}"

    # Default to all server GPUs unless a specific subset is requested.
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    elif "CUDA_VISIBLE_DEVICES" not in env:
        # Do not set this var so child processes can see all GPUs.
        pass

    effective_visible_devices = env.get("CUDA_VISIBLE_DEVICES")
    visible_gpu_count = _count_visible_gpus(effective_visible_devices)
    effective_tp_size = tensor_parallel_size or _auto_tensor_parallel_size(
        visible_gpu_count, resolved_model
    )
    
    # Build the command
    command = [
        "accelerate", "launch",
        "--num-processes", "1",
        "--num-machines", "1",
        "-m", "evalchemy.eval.eval",
        "--model", "vllm",
        "--tasks", tasks,
        "--model_args",
        (
            f"pretrained={resolved_model},"
            f"tensor_parallel_size={effective_tp_size},"
            "gpu_memory_utilization=0.9"
        ),
        "--batch_size", "16",
        "--seed", "1234",
        "--output_path", "../output/step3_reasoning/eval"
    ]
    
    print("=" * 80)
    print("EVALUATION COMMAND:")
    print("=" * 80)
    print(f"Working directory: {evalchemy_dir}")
    print(f"Environment variables:")
    print(f"  PYTHONPATH: {env['PYTHONPATH']}")
    print(f"  CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', 'all available')}")
    print(f"  tensor_parallel_size: {effective_tp_size}")
    print(f"Command: {' '.join(command)}")
    print("=" * 80)
    
    if dry_run:
        print("DRY RUN - Command not executed")
        return
    
    # Change to evalchemy directory and run command
    try:
        result = subprocess.run(
            command,
            cwd=evalchemy_dir,
            env=env,
            check=True
        )
        print("Evaluation completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation: {e}")
    except KeyboardInterrupt:
        print("Evaluation interrupted by user")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with configurable model and tasks")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Model name (e.g., 'meta-llama/Llama-3.2-3B-Instruct')")
    parser.add_argument("--tasks", default="GPQADiamond", help="Comma-separated task names (e.g., 'AMC23,AIME25,LiveCodeBench,GPQADiamond')")
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Comma-separated GPU ids to use (e.g., '0,1,2,3'). Defaults to all available GPUs.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor parallel size. Defaults to number of visible GPUs.",
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        args.model,
        args.tasks,
        args.dry_run,
        cuda_visible_devices=args.cuda_visible_devices,
        tensor_parallel_size=args.tensor_parallel_size,
    )


if __name__ == "__main__":
    main()
