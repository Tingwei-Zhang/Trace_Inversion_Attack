cd evalchemy
python -m eval.eval \
  --model bedrock \
  --tasks JEEBench \
  --apply_chat_template \
  --batch_size 1 \
  --model_args "model=arn:aws:bedrock:us-east-1:122997907744:inference-profile/us.deepseek.r1-v1:0,region=us-east-1,num_concurrent=8" \
  --output_path ../output/step3_reasoning/eval

cd evalchemy
python -m eval.eval \
  --model bedrock \
  --tasks LiveCodeBench \
  --apply_chat_template \
  --batch_size 1 \
  --model_args "model=arn:aws:bedrock:us-east-1:122997907744:inference-profile/us.deepseek.r1-v1:0,region=us-east-1,num_concurrent=8" \
  --output_path ../output/step3_reasoning/eval

cd evalchemy
python -m eval.eval \
  --model bedrock \
  --tasks MATH500 \
  --apply_chat_template \
  --batch_size 1 \
  --model_args "model=arn:aws:bedrock:us-east-1:122997907744:inference-profile/us.deepseek.r1-v1:0,region=us-east-1,num_concurrent=8" \
  --output_path ../output/step3_reasoning/eval