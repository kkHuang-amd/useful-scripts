#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
RESULT_ROOT="${RESULT_ROOT:-$(pwd)/results/b300-kimi-k3-68k-c6-traces-two-cases}"
PROFILE_STEPS="${PROFILE_STEPS:-5}"

export HF_HOME="${HF_HOME:-/dockerx/data/huggingface-cache}"
export PATH="$(pwd)/.venv-aiperf/bin:$PATH"

server_pid=""
client_pid=""

cleanup() {
  if [[ -n "$client_pid" ]]; then
    kill -TERM "$client_pid" 2>/dev/null || true
    wait "$client_pid" 2>/dev/null || true
    client_pid=""
  fi
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup EXIT

run_case() {
  local case_name="$1"
  local multi_stream="$2"
  local pdl="$3"
  local case_dir="$RESULT_ROOT/$case_name"
  local profile_dir="$case_dir/profile"
  local client_log="$case_dir/client.log"

  mkdir -p "$profile_dir" "$case_dir/artifacts"

  SGLANG_OPT_USE_MULTI_STREAM_OVERLAP="$multi_stream" \
  TRTLLM_ENABLE_PDL="$pdl" \
  SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS="$((pdl * 8192))" \
  ./run_b300_68k_tuned.sh >"$case_dir/server.log" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 90); do
    if curl --silent --fail --max-time 3 "$BASE_URL/v1/models" >/dev/null; then
      break
    fi
    kill -0 "$server_pid"
    sleep 10
  done
  curl --silent --fail --max-time 5 "$BASE_URL/v1/models" >/dev/null

  ulimit -n 65535
  MODEL="moonshotai/Kimi-K3" \
  TOKENIZER="moonshotai/Kimi-K3" \
  URL="$BASE_URL" \
  WARMUPS=32 \
  CONCURRENCY=6 \
  REQUEST_COUNT=60 \
  SEED=42 \
  TAG="$case_name" \
  ARTIFACT_DIR="$case_dir/artifacts" \
  ./run_aiperf_68k_sweep.sh >"$client_log" 2>&1 &
  client_pid=$!

  for _ in $(seq 1 1800); do
    if rg -q "Phase profiling started" "$client_log"; then
      break
    fi
    kill -0 "$client_pid"
    sleep 1
  done
  rg -q "Phase profiling started" "$client_log"

  curl --silent --show-error --fail \
    --request POST "$BASE_URL/start_profile" \
    --header "Content-Type: application/json" \
    --data "{
      \"output_dir\": \"$profile_dir\",
      \"num_steps\": $PROFILE_STEPS,
      \"activities\": [\"CPU\", \"GPU\"],
      \"profile_by_stage\": true,
      \"profile_prefix\": \"$case_name-c6\"
    }" \
    >"$case_dir/start_profile_response.json"

  wait "$client_pid"
  client_pid=""
  cleanup

  for _ in $(seq 1 60); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
      break
    fi
    sleep 2
  done
}

mkdir -p "$RESULT_ROOT"
run_case "no-dcp" 1 1
run_case "no-dcp-single-stream-no-pdl" 0 0

echo "All c6 traces saved under: $RESULT_ROOT"
