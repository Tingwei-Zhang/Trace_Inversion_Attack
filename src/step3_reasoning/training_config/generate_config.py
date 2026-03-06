#!/usr/bin/env python3
"""
Training config generator for trace inversion experiments.

Replaces 26+ redundant YAML files with a single generator that combines:
- Base config (shared training parameters)
- Model presets (qwen, llama)
- Experiment presets (r1/*, chatgpt/*)

Usage:
    # Generate config for qwen model with r1 source data
    python generate_config.py --model qwen --preset r1/bubble_on_r1

    # Generate with neat_packing enabled
    python generate_config.py --model llama --preset chatgpt/no_thinking_on_chatgpt --neat-packing

    # Override output directory
    python generate_config.py --model qwen --preset r1/inverted_thinking_r1_on_r1 --output-dir /custom/path

    # Use custom dataset (no preset)
    python generate_config.py --model qwen --dataset my_custom_dataset --output-suffix my_experiment

    # Save to file instead of stdout
    python generate_config.py --model qwen --preset r1/bubble_on_r1 --save config.yaml

    # Generate config, save under training_config/r1, and launch training
    python generate_config.py --model qwen --preset r1/inverted_thinking_r1_on_r1 --train

    # Save to a custom path and launch training
    python generate_config.py --model qwen --preset r1/inverted_thinking_r1_on_r1 --save config.yaml --train

    # List available presets
    python generate_config.py --list-presets
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[2]
BASE_CONFIG_PATH = SCRIPT_DIR / "base_config.yaml"
PRESETS_DIR = SCRIPT_DIR / "presets"
MODELS_PATH = PRESETS_DIR / "models.yaml"
DATASET_INFO_PATH = REPO_ROOT / "data" / "dataset_info.json"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_base_config() -> dict:
    return load_yaml(BASE_CONFIG_PATH)


def load_models() -> dict:
    return load_yaml(MODELS_PATH)


def load_preset(preset_path: str) -> dict:
    """Load a preset from category/name format (e.g., 'r1/bubble_on_r1')."""
    parts = preset_path.split("/")
    if len(parts) != 2:
        raise ValueError(f"Preset must be in 'category/name' format, got: {preset_path}")
    
    category, name = parts
    preset_file = PRESETS_DIR / f"{category}.yaml"
    
    if not preset_file.exists():
        raise FileNotFoundError(f"Preset category not found: {preset_file}")
    
    presets = load_yaml(preset_file)
    if name not in presets:
        available = ", ".join(presets.keys())
        raise KeyError(f"Preset '{name}' not found in {category}. Available: {available}")
    
    return presets[name]


def list_presets() -> None:
    """Print all available presets."""
    print("Available models:")
    models = load_models()
    for name, config in models.items():
        print(f"  {name}: {config['model_name_or_path']}")
    
    print("\nAvailable presets:")
    for preset_file in sorted(PRESETS_DIR.glob("*.yaml")):
        if preset_file.name == "models.yaml":
            continue
        category = preset_file.stem
        presets = load_yaml(preset_file)
        print(f"  {category}/")
        for name, config in presets.items():
            print(f"    {name}: dataset={config['dataset']}")


def dataset_exists(dataset_name: str) -> bool:
    """Check whether dataset is registered in data/dataset_info.json."""
    if not DATASET_INFO_PATH.exists():
        return False
    try:
        with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
            info = yaml.safe_load(f) or {}
    except Exception:
        return False
    return dataset_name in info


def generate_output_dir(model_name: str, output_suffix: str, base_dir: str = "output/step3_reasoning/models") -> str:
    """Generate output directory path based on model and experiment."""
    model_short = "qwen" if "qwen" in model_name.lower() else "llama3.1"
    return f"{base_dir}/{model_short}_sft_{output_suffix}"


def default_config_save_path(model: str, preset: str) -> Path:
    """Choose deterministic config path under training_config/r1 or chatgpt."""
    category, name = preset.split("/", 1)
    if category not in {"r1", "chatgpt"}:
        raise ValueError(f"Unsupported preset category '{category}'. Use r1/* or chatgpt/*, or pass --save.")
    return SCRIPT_DIR / category / f"{model}_sft_{name}.yaml"


def generate_config(
    model: str,
    preset: str | None = None,
    dataset: str | None = None,
    output_suffix: str | None = None,
    output_dir: str | None = None,
    neat_packing: bool = False,
    save_steps: int | None = None,
    max_samples: int | None = None,
    num_epochs: float | None = None,
    learning_rate: float | None = None,
) -> dict:
    """Generate a complete training config."""
    config = load_base_config()
    models = load_models()
    
    if model not in models:
        available = ", ".join(models.keys())
        raise KeyError(f"Model '{model}' not found. Available: {available}")
    
    model_config = models[model]
    config["model_name_or_path"] = model_config["model_name_or_path"]
    config["template"] = model_config["template"]
    
    if preset:
        preset_config = load_preset(preset)
        config["dataset"] = preset_config["dataset"]
        effective_suffix = preset_config["output_suffix"]
    elif dataset:
        config["dataset"] = dataset
        effective_suffix = output_suffix or dataset.replace("open_thoughts_", "").replace("with_", "")
    else:
        raise ValueError("Either --preset or --dataset must be provided")
    
    if output_suffix:
        effective_suffix = output_suffix
    
    if output_dir:
        config["output_dir"] = output_dir
    else:
        config["output_dir"] = generate_output_dir(model_config["model_name_or_path"], effective_suffix)
    
    if neat_packing:
        config["neat_packing"] = True
    
    if save_steps is not None:
        config["save_steps"] = save_steps
    
    if max_samples is not None:
        config["max_samples"] = max_samples
    
    if num_epochs is not None:
        config["num_train_epochs"] = num_epochs
    
    if learning_rate is not None:
        config["learning_rate"] = learning_rate

    if not dataset_exists(config["dataset"]):
        raise ValueError(
            f"Dataset '{config['dataset']}' is not defined in {DATASET_INFO_PATH}. "
            "Please generate/register it first (e.g., run src/step3_reasoning/format_data.py with the matching preset)."
        )
    
    return config


def format_value(value) -> str:
    """Format a single value for YAML output."""
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return value
    elif isinstance(value, float):
        if value == int(value):
            return str(value)
        elif value < 0.001:
            return f"{value:.1e}"
        return str(value)
    else:
        return str(value)


def format_yaml(config: dict) -> str:
    """Format config as YAML with section comments."""
    lines = ["### model"]
    lines.append(f"model_name_or_path: {config.pop('model_name_or_path')}")
    lines.append("")
    
    lines.append("### method")
    method_keys = ["stage", "do_train", "finetuning_type", "deepspeed", 
                   "enable_liger_kernel", "packing", "neat_packing"]
    for key in method_keys:
        if key in config:
            lines.append(f"{key}: {format_value(config.pop(key))}")
    lines.append("")
    
    lines.append("### dataset")
    dataset_keys = ["dataset", "template", "cutoff_len", "max_samples", 
                    "overwrite_cache", "preprocessing_num_workers"]
    for key in dataset_keys:
        if key in config:
            lines.append(f"{key}: {format_value(config.pop(key))}")
    lines.append("")
    
    lines.append("### output")
    output_keys = ["output_dir", "logging_steps", "save_steps", "plot_loss",
                   "overwrite_output_dir", "save_only_model"]
    for key in output_keys:
        if key in config:
            lines.append(f"{key}: {format_value(config.pop(key))}")
    lines.append("")
    
    lines.append("### train")
    for key, value in config.items():
        lines.append(f"{key}: {format_value(value)}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training config for trace inversion experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--model", "-m", choices=["qwen", "llama"],
                        help="Model to train (qwen or llama)")
    parser.add_argument("--preset", "-p",
                        help="Preset to use (e.g., r1/bubble_on_r1, chatgpt/no_thinking_on_chatgpt)")
    parser.add_argument("--dataset", "-d",
                        help="Custom dataset name (alternative to --preset)")
    parser.add_argument("--output-suffix", "-s",
                        help="Output directory suffix (default: derived from dataset)")
    parser.add_argument("--output-dir", "-o",
                        help="Full output directory path (overrides auto-generation)")
    parser.add_argument("--neat-packing", action="store_true",
                        help="Enable neat_packing")
    parser.add_argument("--save-steps", type=int,
                        help="Override save_steps (default: 10000)")
    parser.add_argument("--max-samples", type=int,
                        help="Override max_samples (default: 10000)")
    parser.add_argument("--num-epochs", type=float,
                        help="Override num_train_epochs (default: 3.0)")
    parser.add_argument("--learning-rate", type=float,
                        help="Override learning_rate (default: 1e-5)")
    parser.add_argument("--save", type=str, metavar="FILE",
                        help="Save config to file instead of stdout")
    parser.add_argument("--train", action="store_true",
                        help="Run 'llamafactory-cli train' with generated config (auto-saves under training_config/{r1|chatgpt})")
    parser.add_argument("--list-presets", action="store_true",
                        help="List all available models and presets")
    
    args = parser.parse_args()
    
    if args.list_presets:
        list_presets()
        return
    
    if not args.model:
        parser.error("--model is required (unless using --list-presets)")
    
    if not args.preset and not args.dataset:
        parser.error("Either --preset or --dataset must be provided")
    
    try:
        config = generate_config(
            model=args.model,
            preset=args.preset,
            dataset=args.dataset,
            output_suffix=args.output_suffix,
            output_dir=args.output_dir,
            neat_packing=args.neat_packing,
            save_steps=args.save_steps,
            max_samples=args.max_samples,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )
        
        yaml_output = format_yaml(config)
        
        if args.train:
            if args.save:
                config_path = Path(args.save)
            elif args.preset:
                config_path = default_config_save_path(args.model, args.preset)
            else:
                raise ValueError("--save is required with --train when using --dataset without --preset.")

            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(yaml_output)
            print(f"Config saved to {config_path}", file=sys.stderr)

            train_cmd = ["llamafactory-cli", "train", str(config_path)]
            print(f"Running: {' '.join(train_cmd)}", file=sys.stderr)
            subprocess.run(train_cmd, check=True)
        else:
            if args.save:
                with open(args.save, "w", encoding="utf-8") as f:
                    f.write(yaml_output)
                print(f"Config saved to {args.save}", file=sys.stderr)
            else:
                print(yaml_output)
            
    except (KeyError, FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
