#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/sgl-workspace/kvv-bench/kvv-k3-0727-update}"
BEAM_DIR="$BENCH_DIR/beam"
MODEL_PATH="${MODEL_PATH:-/dockerx/data/models/Kimi-K3}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
BEAM_API_KEY="${BEAM_API_KEY:-EMPTY}"
CONCURRENCY="${CONCURRENCY:-16}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$BEAM_DIR/results}"
OUTPUT="${OUTPUT:-$OUTPUT_DIR/answers_${RUN_TAG}.jsonl}"
LOG="${LOG:-$OUTPUT_DIR/generate_${RUN_TAG}.log}"

test -f "$BEAM_DIR/beam_generate.py" || {
  echo "error: beam_generate.py not found in $BEAM_DIR" >&2
  exit 1
}
test -f "$MODEL_PATH/tokenizer_config.json" || {
  echo "error: Kimi-K3 tokenizer not found in $MODEL_PATH" >&2
  exit 1
}

mkdir -p "$OUTPUT_DIR"
export BEAM_API_KEY

echo "BEAM output: $OUTPUT"
echo "BEAM log:    $LOG"
echo "Resume by reusing RUN_TAG='$RUN_TAG' or OUTPUT='$OUTPUT'."

(
  cd "$BEAM_DIR"
  uv run python beam_generate.py \
    --model "$MODEL_PATH" \
    --base-url "$BASE_URL" \
    --temperature 1.0 \
    --top-p 0.95 \
    --max-tokens 32768 \
    --thinking-json \
      '{"chat_template_kwargs":{"thinking":true,"preserve_thinking":true,"thinking_effort":"max"}}' \
    --tokenizer "$MODEL_PATH" \
    --concurrency "$CONCURRENCY" \
    --output "$OUTPUT"
) 2>&1 | tee "$LOG"

