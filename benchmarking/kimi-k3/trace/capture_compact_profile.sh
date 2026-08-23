#!/usr/bin/env bash
set -euo pipefail

# Capture a compact CPU/GPU profile from an already-running SGLang server.
# The caller owns server startup/shutdown; this script owns only the client and
# /start_profile -> /stop_profile window.

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVER_LOG="${SERVER_LOG:?set SERVER_LOG to the active server log}"
RESULT_DIR="${RESULT_DIR:?set RESULT_DIR}"
MODEL="${MODEL:-/shared_nfs/models/Kimi-K3}"
TOKENIZER="${TOKENIZER:-$MODEL}"
BENCH_MODULE="${BENCH_MODULE:-sglang.bench_serving}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
INPUT_LENGTH="${INPUT_LENGTH:-8192}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-64}"
CONCURRENCY="${CONCURRENCY:-64}"
PROFILE_SECONDS="${PROFILE_SECONDS:-2}"
EXPECTED_RANKS="${EXPECTED_RANKS:-8}"
MAX_TRACE_MIB="${MAX_TRACE_MIB:-500}"

TRACE_DIR="$RESULT_DIR/traces"
CLIENT_LOG="$RESULT_DIR/client.log"
CLIENT_JSON="$RESULT_DIR/client.jsonl"
CLIENT_PID=

mkdir -p "$TRACE_DIR"

cleanup() {
  if [[ -n "$CLIENT_PID" ]] && kill -0 "$CLIENT_PID" 2>/dev/null; then
    kill -TERM "$CLIENT_PID" 2>/dev/null || true
    wait "$CLIENT_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

prefill_before=$(rg -c "Prefill batch" "$SERVER_LOG" || true)
decode_before=$(rg -c "Decode batch" "$SERVER_LOG" || true)

python -m "$BENCH_MODULE" \
  --backend sglang-oai \
  --base-url "$BASE_URL" \
  --dataset-name random \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --num-prompts "$NUM_PROMPTS" \
  --random-input-len "$INPUT_LENGTH" \
  --random-output-len "$OUTPUT_LENGTH" \
  --random-range-ratio 1.0 \
  --max-concurrency "$CONCURRENCY" \
  --request-rate inf \
  --warmup-requests 0 \
  --seed 42 \
  --output-file "$CLIENT_JSON" \
  >"$CLIENT_LOG" 2>&1 &
CLIENT_PID=$!

for _ in $(seq 1 600); do
  prefill_now=$(rg -c "Prefill batch" "$SERVER_LOG" || true)
  if (( prefill_now > prefill_before )); then
    break
  fi
  kill -0 "$CLIENT_PID"
  sleep 0.2
done

profile_body=$(python - "$TRACE_DIR" <<'PY'
import json
import sys

print(json.dumps({
    "output_dir": sys.argv[1],
    "activities": ["CPU", "GPU"],
    "with_stack": False,
    "record_shapes": False,
    "profile_by_stage": False,
    "merge_profiles": False,
}))
PY
)
curl -fsS -X POST "$BASE_URL/start_profile" \
  -H "Content-Type: application/json" \
  -d "$profile_body" >"$RESULT_DIR/start-profile.json"

for _ in $(seq 1 600); do
  decode_now=$(rg -c "Decode batch" "$SERVER_LOG" || true)
  if (( decode_now > decode_before )); then
    break
  fi
  kill -0 "$CLIENT_PID"
  sleep 0.2
done

sleep "$PROFILE_SECONDS"
curl -fsS -X POST "$BASE_URL/stop_profile" \
  -H "Content-Type: application/json" \
  -d '{}' >"$RESULT_DIR/stop-profile.json"

wait "$CLIENT_PID"
CLIENT_PID=

for _ in $(seq 1 300); do
  trace_count=$(compgen -G "$TRACE_DIR/*.trace.json.gz" | wc -l || true)
  if (( trace_count >= EXPECTED_RANKS )); then
    break
  fi
  sleep 1
done

mapfile -t traces < <(compgen -G "$TRACE_DIR/*.trace.json.gz")
[[ "${#traces[@]}" -eq "$EXPECTED_RANKS" ]]
for trace in "${traces[@]}"; do
  gzip -t "$trace"
  (( $(stat -c %s "$trace") <= MAX_TRACE_MIB * 1024 * 1024 ))
done

echo "TRACE_COMPLETE traces=${#traces[@]} dir=$TRACE_DIR"
