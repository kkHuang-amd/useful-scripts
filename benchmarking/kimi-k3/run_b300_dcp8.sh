#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-/dockerx/data/huggingface-cache}"

exec sglang serve \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 8 \
  --dcp-size 8 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mamba-full-memory-ratio 8.82 \
  --host 0.0.0.0 \
  --port 30000
