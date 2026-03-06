# Step2 Inversion

This directory handles inversion data formatting, inversion training, and inversion evaluation.

## One formatter entrypoint

Use a single script for all scenarios:

```bash
python src/step2_inversion/format_data.py --scenario r1
python src/step2_inversion/format_data.py --scenario r1_distill
python src/step2_inversion/format_data.py --scenario r1_no_bubble
```

Scenario behavior is controlled in `SCENARIO_CONFIG` inside `format_data.py`.

### Outputs from formatter

- Reasoning data (bubble + final answer) for step3 reasoning training:
  - `open_thoughts_with_bubble_r1.jsonl`
  - `open_thoughts_with_bubble_r1_distill.jsonl`
- Inversion train/val data:
  - `open_thoughts_to_invert_*_train.jsonl`
  - `open_thoughts_to_invert_*_val.jsonl`

`format_data.py` also updates `data/dataset_info.json` for generated files under `data/`.

## Training

Example:

```bash
llamafactory-cli train src/step2_inversion/training/inversion_qwen2_5_surrogate_r1.yaml
llamafactory-cli train src/step2_inversion/training/inversion_qwen2_5_surrogate_r1_distill.yaml
llamafactory-cli train src/step2_inversion/training/inversion_qwen2_5_surrogate_r1_no_bubble.yaml
```

## Evaluation

Evaluation is centralized:

```bash
python src/step2_inversion/evaluation/run_inversion_eval.py --list
python src/step2_inversion/evaluation/run_inversion_eval.py --preset surrogate_r1_distill_on_chatgpt
python src/step2_inversion/evaluation/run_inversion_eval.py --preset surrogate_r1_distill_on_chatgpt --dry-run
```

By default, `run_inversion_eval.py` runs inference only.
To also compute prediction-vs-label similarity metrics, add `--with-metrics`.
Metrics are saved as `similarity_metrics.json` next to the generated prediction file.

`model_name_or_path` values are aligned with `output_dir` values in `src/step2_inversion/training/*.yaml`.

Current clean wrapper scripts include:

- `eval_surrogate_r1_distill_on_deepseek.sh`
- `eval_surrogate_r1_distill_on_r1_distill.sh`
- `eval_surrogate_r1_distill_on_chatgpt.sh`
- `eval_surrogate_r1_distill_on_chatgpt_high_effort.sh`
- `eval_surrogate_r1_on_chatgpt.sh`
- `eval_surrogate_r1_no_bubble_on_no_bubble.sh`

### Similarity metrics (prediction vs label)

Metrics are integrated into `run_inversion_eval.py` and are opt-in:

```bash
python src/step2_inversion/evaluation/run_inversion_eval.py \
  --preset surrogate_r1_on_r1 \
  --with-metrics
```

Wrapper scripts support passthrough args, so this also works:

```bash
sh src/step2_inversion/evaluation/eval_surrogate_r1_distill_on_chatgpt.sh --with-metrics
```
