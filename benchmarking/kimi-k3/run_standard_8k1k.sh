#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
MODEL="${MODEL:-/dockerx/data/models/Kimi-K3}"
TOKENIZER="${TOKENIZER:-moonshotai/Kimi-K3}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
INPUT_LENGTH="${INPUT_LENGTH:-8192}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1024}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"
WARMUPS="${WARMUPS:-64}"
SEED="${SEED:-42}"
TAG="${TAG:-kimi-k3-8k1k-c32}"
RESULT_DIR="${RESULT_DIR:-$(pwd)/results/${TAG}}"

mkdir -p "$RESULT_DIR"

python -m sglang.benchmark.serving \
  --backend sglang-oai \
  --base-url "$BASE_URL" \
  --dataset-name random \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --num-prompts "$NUM_PROMPTS" \
  --random-input-len "$INPUT_LENGTH" \
  --random-output-len "$OUTPUT_LENGTH" \
  --random-range-ratio 1.0 \
  --max-concurrency "$MAX_CONCURRENCY" \
  --warmup-requests "$WARMUPS" \
  --seed "$SEED" \
  --output-file "$RESULT_DIR/result.jsonl" \
  2>&1 | tee "$RESULT_DIR/client.log"
