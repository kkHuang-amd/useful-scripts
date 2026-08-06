#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/dockerx/data/models/Kimi-K3}"
TOKENIZER="${TOKENIZER:-moonshotai/Kimi-K3}"
URL="${URL:-http://localhost:8000}"
PREFIX_PROMPTS="${PREFIX_PROMPTS:-8}"
PREFIX_LENGTH="${PREFIX_LENGTH:-63240}"
INPUT_LENGTH="${INPUT_LENGTH:-4760}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-350}"
WARMUPS="${WARMUPS:-32}"
CONCURRENCY="${CONCURRENCY:-6,12,24,32}"
REQUEST_COUNT="${REQUEST_COUNT:-60,72,96,128}"
SEED="${SEED:-42}"
TAG="${TAG:-kimi-k3-68k}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$(pwd)/results/${TAG}/artifacts}"

mkdir -p "$ARTIFACT_DIR"

exec aiperf profile \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --url "$URL" \
  --endpoint-type chat \
  --streaming \
  --use-server-token-count \
  --num-prefix-prompts "$PREFIX_PROMPTS" \
  --prompt-prefix-length "$PREFIX_LENGTH" \
  --synthetic-input-tokens-mean "$INPUT_LENGTH" \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean "$OUTPUT_LENGTH" \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --extra-inputs "min_tokens:${OUTPUT_LENGTH}" \
  --extra-inputs "max_tokens:${OUTPUT_LENGTH}" \
  --warmup-request-count "$WARMUPS" \
  --sweep-type zip \
  --concurrency "$CONCURRENCY" \
  --request-count "$REQUEST_COUNT" \
  --random-seed "$SEED" \
  --ui simple \
  --artifact-dir "$ARTIFACT_DIR"
