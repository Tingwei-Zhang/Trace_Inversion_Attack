# Step 0: Data Preprocessing

Prepares OpenThoughts-114k dataset for reasoning and inversion model training.

## Pipeline

```
OpenThoughts-114k
       │
       ▼
download_dataset.py  ──►  Preprocessed prompts (20k/50k) + Training data
       │
       ├──────────────────────┐
       ▼                      ▼
   R1 Distill Path       ChatGPT Path
   (local model)         (API-based)
       │                      │
       ▼                      ▼
   Training data         Training data +
   (with thinking)       Inversion data
```

## Files

| Script | What it does |
|--------|--------------|
| `download_dataset.py` | Downloads OpenThoughts-114k, creates inference prompts + training data (with/without thinking) |
| `r1_distill_inference.sh` | Runs DeepSeek-R1-Distill-Qwen-1.5B inference via vLLM |
| `preprocess_r1_distill.py` | Extracts `<think>` content from R1 outputs |
| `chatgpt_inference.py` | Calls GPT-5-mini API with reasoning mode |
| `preprocess_chatgpt_inference.py` | Formats ChatGPT outputs for training |
| `script.sh` | Runs the pipeline |

## Quick Start

```bash
# 1. Download and preprocess base dataset
python src/step0_data_preprocess/download_dataset.py

# 2. Choose one path:
# R1 path (GPU required)
python src/step0_data_preprocess/preprocess_r1_distill.py

# ChatGPT path (API key required)
python src/step0_data_preprocess/chatgpt_inference.py
python src/step0_data_preprocess/preprocess_chatgpt_inference.py
```

## Outputs

- `data/step0_preprocessed_data/` - Inference prompts and comprehensive formats
- `data/step3_reasoning_model_training_data/` - Training data (with/without thinking)
