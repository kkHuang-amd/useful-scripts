#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/dockerx/data/huggingface-cache}"

exec sglang serve \
  --model-path moonshotai/Kimi-K3 \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend trtllm_mla \
  --dtype bfloat16 \
  --mem-fraction-static 0.85 \
  --cuda-graph-max-bs-decode 32 \
  --max-running-requests 32 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mamba-full-memory-ratio 0.3 \
  --mamba-ssm-dtype bfloat16 \
  --enable-int8-mamba-checkpoint \
  --mamba-track-interval 1024 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --enable-cache-report \
  --host 0.0.0.0 \
  --port 30000
