python -m eval.eval \
    --model openai-chat-completions \
    --tasks JEEBench \
    --model_args "model=gpt-5-mini-2025-08-07,num_concurrent=32" \
    --batch_size 16 \
    --output_path ../output/step3_reasoning/eval

python -m eval.eval \
    --model openai-chat-completions \
    --tasks MATH500 \
    --model_args "model=gpt-5-mini-2025-08-07,num_concurrent=32" \
    --batch_size 16 \
    --output_path ../output/step3_reasoning/eval

python -m eval.eval \
    --model openai-chat-completions \
    --tasks LiveCodeBench \
    --model_args "model=gpt-5-mini-2025-08-07,num_concurrent=32" \
    --batch_size 16 \
    --output_path ../output/step3_reasoning/eval

python -m eval.eval \
    --model openai-chat-completions \
    --tasks LiveCodeBench \
    --model_args "model=gpt-4o-mini-2024-07-18,num_concurrent=32" \
    --batch_size 16 \
    --output_path ../output/step3_reasoning/eval

