python src/step3_reasoning/evaluation/run_evaluation.py --model "meta-llama/Llama-3.2-3B-Instruct" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.2_full_sft_with_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.2_full_sft_with_inverted_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.2_full_sft_with_bubble_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.2_full_sft_with_no_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.2_full_sft_with_inverted_thinking_no_bubble_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"

python src/step3_reasoning/evaluation/run_evaluation.py --model "Qwen/Qwen2.5-7B-Instruct" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/qwen_full_sft_with_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/qwen_full_sft_with_inverted_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/qwen_full_sft_with_bubble_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/qwen_full_sft_with_no_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/qwen_full_sft_with_inverted_thinking_no_bubble_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"

python src/step3_reasoning/evaluation/run_evaluation.py --model "meta-llama/Llama-3.1-8B-Instruct" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.1_full_sft_with_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.1_full_sft_with_inverted_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.1_full_sft_with_bubble_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"
python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.1_full_sft_with_no_thinking_tingwei" --tasks "AMC23,AIME25,LiveCodeBench,GPQADiamond"


python src/step3_reasoning/evaluation/run_evaluation.py --model "../output/step3_reasoning/models/llama3.2_full_sft_with_no_thinking_tingwei_open_thoughts3" --tasks "MATH500"
