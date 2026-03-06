# Step 1: Summarization

Converts full reasoning traces (`<think>...</think>`) into condensed "reasoning bubbles" using Qwen2.5-7B-Instruct.

## Pipeline

```
step0_preprocessed_data/processed_open_thoughts_20k_*.jsonl
       │
       ▼
data_formatter.py  ──►  step1_summarized_data/open_thoughts_to_summarize_*.jsonl
       │
       ▼
qwen2_5_summarization_*.sh (vLLM inference)
       │
       ▼
output/step1_summarization/bubbles_*.jsonl
```

## Files

| Script | What it does |
|--------|--------------|
| `data_formatter.py` | Extracts `assistant_thinking` content, formats for summarization inference |
| `qwen2_5_summarization_r1.sh` | Runs Qwen2.5-7B inference on R1 data |
| `qwen2_5_summarization_r1_distill.sh` | Runs Qwen2.5-7B inference on R1-distill data |
| `script.sh` | Runs both pipelines |

## Usage

```bash
# Process R1 data
python src/step1_summarization/data_formatter.py --model r1

# Process R1-distill data
python src/step1_summarization/data_formatter.py --model r1_distill

# Or run both
bash src/step1_summarization/script.sh
```

## Outputs

- `data/step1_summarized_data/` - Formatted prompts for summarization inference
- `output/step1_summarization/` - Generated reasoning bubbles
