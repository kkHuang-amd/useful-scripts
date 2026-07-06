#!/usr/bin/env bash
# GSM-8K via the lm-eval-harness (local-completions backend) against a running
# SGLang server on B200. Companion to gsm8k_b200.sh (which uses the in-tree
# few_shot_gsm8k). Requires `pip install lm-eval`.
#
#   MODEL = served model path (default = B200 DeepSeek-V4-Pro)
#   PORT  = server port       (default 8000)
#   NCON  = num_concurrent    (default 64)
set -euo pipefail

MODEL="${MODEL:-/dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro}"
PORT="${PORT:-8000}"
NCON="${NCON:-64}"

set -x
exec lm_eval \
  --model local-completions \
  --model_args model=${MODEL},base_url=http://localhost:${PORT}/v1/completions,num_concurrent=${NCON},max_retries=3,tokenized_requests=False \
  --tasks gsm8k \
  --num_fewshot 5
