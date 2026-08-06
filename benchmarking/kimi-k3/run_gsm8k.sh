#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
MODEL_PATH="${MODEL_PATH:-/dockerx/data/models/Kimi-K3}"
SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
NUM_THREADS="${NUM_THREADS:-64}"
NUM_SHOTS="${NUM_SHOTS:-5}"
MAX_TOKENS="${MAX_TOKENS:-512}"
RESULT_LOG="${RESULT_LOG:-/tmp/kimi-k3-gsm8k.log}"

served_id="$(
  curl -fsS --max-time 10 "$BASE_URL/v1/models" |
    python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])'
)"
if [[ "$served_id" != "$MODEL_PATH" ]]; then
  echo "error: expected model '$MODEL_PATH', endpoint serves '$served_id'" >&2
  exit 1
fi

cd "$SGLANG_DIR"
python -m sglang.test.run_eval \
  --base-url "$BASE_URL" \
  --model "$served_id" \
  --eval-name gsm8k \
  --api completion \
  --num-examples "$NUM_EXAMPLES" \
  --num-threads "$NUM_THREADS" \
  --num-shots "$NUM_SHOTS" \
  --max-tokens "$MAX_TOKENS" \
  --temperature 0 \
  2>&1 | tee "$RESULT_LOG"
