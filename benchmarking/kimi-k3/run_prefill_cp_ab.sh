#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/dockerx/data/models/Kimi-K3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CONCS="${CONCS:-1 2 4 8}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_ROOT="${RESULT_ROOT:-/dockerx/home/wunhuang/tmp/benchmark-results/kimi-k3-prefill-cp}"
RUN_DIR="$RESULT_ROOT/$RUN_TAG"
CURRENT_SERVER_PID=""

mkdir -p "$RUN_DIR"

cleanup() {
  if [[ -n "$CURRENT_SERVER_PID" ]] && kill -0 "$CURRENT_SERVER_PID" 2>/dev/null; then
    "$SCRIPT_DIR/stop_server.sh" || true
    wait "$CURRENT_SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

stop_server() {
  "$SCRIPT_DIR/stop_server.sh"
  CURRENT_SERVER_PID=""
  sleep 10
}

record_environment() {
  {
    date -u
    hostname
    python3 -c 'import importlib.metadata as m,sys; print("python="+sys.version.split()[0]); print("sglang="+m.version("sglang"))'
    git -C /sgl-workspace/sglang rev-parse HEAD
    rocm-smi --showproductname --showmemuse --showuse --showpids
  } >"$RUN_DIR/environment.log" 2>&1
}

start_server() {
  local mode="$1"
  local enable_cp="$2"
  local server_log="$RUN_DIR/${mode}_server.log"

  "$SCRIPT_DIR/stop_server.sh"
  (
    ENABLE_PREFILL_CP="$enable_cp" \
    CP_STRATEGY=zigzag \
    RADIX_CACHE=0 \
    HOST=0.0.0.0 \
    PORT="$PORT" \
    MODEL_PATH="$MODEL_PATH" \
      "$SCRIPT_DIR/launch_server.sh"
  ) >"$server_log" 2>&1 &
  CURRENT_SERVER_PID=$!
  printf '%s\n' "$CURRENT_SERVER_PID" >"$RUN_DIR/${mode}_server.pid"

  SERVER_LOG="$server_log" TIMEOUT_SECONDS=5400 \
    "$SCRIPT_DIR/wait_server.sh" | tee "$RUN_DIR/${mode}_wait.log"

  curl -fsS --max-time 30 "http://${HOST}:${PORT}/server_info" \
    >"$RUN_DIR/${mode}_server_info.json"
  python3 - "$RUN_DIR/${mode}_server_info.json" "$enable_cp" <<'PY'
import json
import sys

path, expected = sys.argv[1], sys.argv[2] == "1"
with open(path) as f:
    info = json.load(f)
actual = bool(info.get("enable_prefill_cp"))
if actual != expected:
    raise SystemExit(f"enable_prefill_cp mismatch: expected={expected}, actual={actual}")
attn_cp_size = int(info.get("attn_cp_size", 1))
expected_size = 8 if expected else 1
if attn_cp_size != expected_size:
    raise SystemExit(
        f"attn_cp_size mismatch: expected={expected_size}, actual={attn_cp_size}"
    )
print(f"enable_prefill_cp={actual} attn_cp_size={attn_cp_size}")
PY

  BASE_URL="http://${HOST}:${PORT}/v1" MODEL_PATH="$MODEL_PATH" \
    "$SCRIPT_DIR/smoke_test.sh" >"$RUN_DIR/${mode}_smoke.log" 2>&1
}

run_sweep() {
  local mode="$1"
  local result_dir="$RUN_DIR/$mode"
  TAG="$mode" RESULT_DIR="$result_dir" HOST="$HOST" PORT="$PORT" \
    MODEL="$MODEL_PATH" CONCS="$CONCS" NP_MULT=4 WARM_MULT=1 \
    "$SCRIPT_DIR/bench_long_context.sh"
}

record_environment

# Validate the experimental path first; do not collect A/B data if CP is not real.
start_server prefill_cp 1
TAG=cp_correctness RESULT_DIR="$RUN_DIR/cp_correctness" \
  HOST="$HOST" PORT="$PORT" MODEL="$MODEL_PATH" CONCS=1 \
  NP_MULT=1 WARM_MULT=0 "$SCRIPT_DIR/bench_long_context.sh"
python3 - "$RUN_DIR/cp_correctness/cp_correctness_isl68000_osl350_c1.jsonl" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as f:
    result = json.loads(f.readline())
if result.get("completed") != 1:
    raise SystemExit(f"correctness request did not complete: {result.get('completed')}")
if result.get("total_input_tokens", 0) < 68000:
    raise SystemExit(f"unexpected input token count: {result.get('total_input_tokens')}")
for key in ("mean_ttft_ms", "mean_tpot_ms", "mean_e2e_latency_ms"):
    if not math.isfinite(float(result[key])):
        raise SystemExit(f"non-finite {key}: {result[key]}")
print("68k/350 correctness gate passed")
PY
run_sweep prefill_cp
stop_server

start_server baseline 0
run_sweep baseline
stop_server

python3 "$SCRIPT_DIR/summarize_prefill_cp.py" \
  --baseline "$RUN_DIR/baseline" \
  --cp "$RUN_DIR/prefill_cp" \
  --output "$RUN_DIR/summary.md"

printf 'RUN_COMPLETE %s\n' "$RUN_DIR"
