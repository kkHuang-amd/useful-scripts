MODEL=/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/

lm_eval \
  --model local-completions \
  --model_args model=${MODEL},base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
