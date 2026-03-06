# Trace_Inversion_Attack
Train inversion models to synthesize full reasoning traces from reasoning summaries ("bubbles"), then use those synthesized traces to train student models to perform capability stealing attack..

The active workflow lives under `src/`:
- `src/step0_data_preprocess`
- `src/step1_summarization`
- `src/step2_inversion`
- `src/step3_reasoning`

## Setup

```bash
git clone https://github.com/Tingwei-Zhang/trace_inversion.git
cd trace_inversion

conda create -n trace_inversion python=3.10 -y
conda activate trace_inversion
pip install -r requirements.txt

git submodule update --init --recursive
cd llama_factory && pip install -e ".[torch,metrics]" --no-build-isolation
cd ../evalchemy && pip install -e . && pip install -e eval/chat_benchmarks/alpaca_eval
cd ..
```

For gated Hugging Face models:
```bash
huggingface-cli login
```

## End-to-End Pipeline

```text
Step 0 (preprocess) -> Step 1 (summarize to bubbles) ->
Step 2 (train/eval inversion model) -> Step 3 (train/eval reasoning model)
```

## Step 0: Data Preprocessing

Generate base processed datasets and optional ChatGPT path.

```bash
# Base dataset download + preprocessing
python src/step0_data_preprocess/download_dataset.py

# R1-distill preprocessing path
python src/step0_data_preprocess/preprocess_r1_distill.py

# ChatGPT path (requires OPENAI_API_KEY)
python src/step0_data_preprocess/chatgpt_inference.py
python src/step0_data_preprocess/preprocess_chatgpt_inference.py
```

See `src/step0_data_preprocess/README.md` for details.

## Step 1: Summarization (Create Bubbles)

Create bubble summaries from full thinking traces.

```bash
python src/step1_summarization/data_formatter.py --model r1
python src/step1_summarization/data_formatter.py --model r1_distill
```

See `src/step1_summarization/README.md` for details.

## Step 2: Inversion Model

### 2.1 Format inversion data

```bash
python src/step2_inversion/format_data.py --scenario r1
python src/step2_inversion/format_data.py --scenario r1_distill
python src/step2_inversion/format_data.py --scenario r1_no_bubble
```

### 2.2 Train inversion model

```bash
llamafactory-cli train src/step2_inversion/training/inversion_qwen2_5_surrogate_r1.yaml
llamafactory-cli train src/step2_inversion/training/inversion_qwen2_5_surrogate_r1_distill.yaml
llamafactory-cli train src/step2_inversion/training/inversion_qwen2_5_surrogate_r1_no_bubble.yaml
```

### 2.3 Run inversion evaluation

```bash
python src/step2_inversion/evaluation/run_inversion_eval.py --list
python src/step2_inversion/evaluation/run_inversion_eval.py --preset surrogate_r1_distill_on_chatgpt
python src/step2_inversion/evaluation/run_inversion_eval.py --preset surrogate_r1_distill_on_chatgpt --with-metrics
```

See `src/step2_inversion/README.md` for preset names and outputs.

## Step 3: Reasoning Model

### 3.1 Format reasoning training data

```bash
python src/step3_reasoning/format_data.py --preset r1_distill_on_r1
```

### 3.2 Generate training config (preset-based)

```bash
python src/step3_reasoning/training_config/generate_config.py --list-presets

# Directly generate config and train (auto-saves under training_config/r1 or training_config/chatgpt)
python src/step3_reasoning/training_config/generate_config.py \
  --model qwen \
  --preset r1/inverted_thinking_r1_distill_on_r1 \
  --train

# Or save config explicitly
python src/step3_reasoning/training_config/generate_config.py \
  --model qwen \
  --preset r1/inverted_thinking_r1_distill_on_r1 \
  --save /tmp/step3_train.yaml
```

### 3.3 Train and evaluate reasoning model

```bash
# If you used --save above:
llamafactory-cli train /tmp/step3_train.yaml

python src/step3_reasoning/evaluation/run_evaluation.py \
  --model output/step3_reasoning/models/qwen_sft_inverted_thinking_r1_distill_on_r1 \
  --tasks "MATH500,GPQADiamond"
```

See `src/step3_reasoning/README.md` for naming conventions and additional examples.

## Notes

- Scripts under `src/` are the maintained path.
- Many data formatter scripts auto-update `data/dataset_info.json` for generated datasets.
- Model training is done through `llama_factory`, and benchmark evaluation is done through `evalchemy`.


Please feel free to email: [tz362@cornell.edu](mailto:tz362@cornell.edu) or raise an issue.