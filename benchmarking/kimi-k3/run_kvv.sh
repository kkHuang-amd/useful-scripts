#!/usr/bin/env bash
set -euo pipefail

BENCH="${1:-}"
if [[ ! "$BENCH" =~ ^(ocrbench|mmmu|toolcall|all)$ ]]; then
  echo "usage: $0 {ocrbench|mmmu|toolcall|all}" >&2
  exit 2
fi

BENCH_DIR="${BENCH_DIR:-/sgl-workspace/kvv-bench/kvv-k3-0727-update}"
MODEL_PATH="${MODEL_PATH:-/dockerx/data/models/Kimi-K3}"
BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"
MAX_CONNECTIONS="${MAX_CONNECTIONS:-50}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-$BENCH_DIR/logs/kimi-k3-$RUN_TAG}"
INSPECT_MODEL="${INSPECT_MODEL:-opensource/${MODEL_PATH#/}}"

test -f "$BENCH_DIR/eval.py" || {
  echo "error: eval.py not found in $BENCH_DIR" >&2
  exit 1
}

mkdir -p "$LOG_DIR"
export KIMI_BASE_URL="$BASE_URL"
export KIMI_API_KEY="$API_KEY"

common=(
  --model "$INSPECT_MODEL"
  --thinking
  --think-mode opensource
  --stream
  --max-connections "$MAX_CONNECTIONS"
  --temperature 1.0
  --top-p 0.95
  --thinking-effort max
)

run_one() {
  local name="$1"
  local max_tokens dataset=()

  case "$name" in
    ocrbench)
      max_tokens=16384
      ;;
    mmmu)
      max_tokens=98304
      ;;
    toolcall)
      max_tokens=32768
      dataset=(
        --dataset
        "$BENCH_DIR/toolcall_benchmark/toolcall_thinking_samples.jsonl"
      )
      ;;
  esac

  local eval_name="$name"
  [[ "$name" == "toolcall" ]] && eval_name="kimi_toolcall"

  echo "Running $name; log=$LOG_DIR/$name.log"
  (
    cd "$BENCH_DIR"
    uv run python eval.py "$eval_name" \
      --max-tokens "$max_tokens" \
      "${common[@]}" \
      "${dataset[@]}"
  ) 2>&1 | tee "$LOG_DIR/$name.log"
}

if [[ "$BENCH" == "all" ]]; then
  run_one ocrbench
  run_one mmmu
  run_one toolcall
else
  run_one "$BENCH"
fi

