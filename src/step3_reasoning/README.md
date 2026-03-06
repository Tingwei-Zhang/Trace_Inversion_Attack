# Step 3: Reasoning Model Training

Train reasoning models on inverted thinking traces.

## Quick Start

```bash
# 1. Format data (combine prompts + inverted thinking)
python src/step3_reasoning/format_data.py --preset r1_distill_on_r1

# 2. Generate training config
python src/step3_reasoning/training_config/generate_config.py \
    --model qwen --preset r1/inverted_thinking_r1_distill_on_r1 --train
# (auto-saves config to src/step3_reasoning/training_config/r1/)

# 4. Evaluate
python src/step3_reasoning/evaluation/run_evaluation.py \
    --model output/step3_reasoning/models/qwen_sft_inverted_thinking_r1_distill_on_r1 \
    --tasks MATH500
```

## Data Formatting

```bash
# Use predefined preset
python src/step3_reasoning/format_data.py --preset r1_distill_on_r1

# Or custom paths
python src/step3_reasoning/format_data.py \
    --prompt_input data/step0_preprocessed_data/processed_open_thoughts.jsonl \
    --inversion_input output/step2_inversion/eval/your_model/predictions.jsonl \
    --reasoning_output data/step3_reasoning_model_training_data/my_dataset.jsonl
```

## Config Generator

```bash
# List presets
python src/step3_reasoning/training_config/generate_config.py --list-presets

# Generate config
python src/step3_reasoning/training_config/generate_config.py \
    --model qwen --preset r1/bubble_on_r1 --save config.yaml

# With overrides
python src/step3_reasoning/training_config/generate_config.py \
    --model llama --preset chatgpt/no_thinking_on_chatgpt \
    --neat-packing --max-samples 5000 --save config.yaml
```

| Option | Description |
|--------|-------------|
| `--model` | `qwen` or `llama` |
| `--preset` | e.g., `r1/bubble_on_r1`, `chatgpt/inverted_thinking_r1_on_chatgpt` |
| `--dataset` | Custom dataset (alternative to preset) |
| `--neat-packing` | Enable neat_packing |
| `--max-samples` | Limit training samples |
| `--save FILE` | Save to file |
| `--train` | Generate config and run `llamafactory-cli train` directly |

## Evaluation

```bash
python src/step3_reasoning/evaluation/run_evaluation.py \
    --model output/step3_reasoning/models/your_model \
    --tasks "MATH500,GPQADiamond"
```
