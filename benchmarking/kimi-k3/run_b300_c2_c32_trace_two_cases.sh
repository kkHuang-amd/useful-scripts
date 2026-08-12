#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-moonshotai/Kimi-K3}"
TOKENIZER="${TOKENIZER:-moonshotai/Kimi-K3}"
BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
RESULT_ROOT="${RESULT_ROOT:-$(pwd)/results/b300-kimi-k3-traces-c2-c32-two-cases}"
PROFILE_STEPS="${PROFILE_STEPS:-5}"

export HF_HOME="${HF_HOME:-/dockerx/data/huggingface-cache}"

server_pid=""

cleanup_server() {
  if [[ -n "$server_pid" ]]; then
    kill -TERM "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    server_pid=""
  fi
}
trap cleanup_server EXIT

run_case() {
  local case_name="$1"
  local multi_stream="$2"
  local pdl="$3"
  local case_dir="$RESULT_ROOT/$case_name"

  mkdir -p "$case_dir"

  SGLANG_OPT_USE_MULTI_STREAM_OVERLAP="$multi_stream" \
  TRTLLM_ENABLE_PDL="$pdl" \
  SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS="$((pdl * 8192))" \
  sglang serve \
    --trust-remote-code \
    --model-path "$MODEL" \
    --tp-size 8 \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --mamba-full-memory-ratio 0.9 \
    --host 0.0.0.0 \
    --port 30000 \
    >"$case_dir/server.log" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 90); do
    if curl --silent --fail --max-time 3 "$BASE_URL/v1/models" >/dev/null; then
      break
    fi
    kill -0 "$server_pid"
    sleep 10
  done
  curl --silent --fail --max-time 5 "$BASE_URL/v1/models" >/dev/null

  for concurrency in 2 32; do
    local warmups=$((concurrency * 2))
    local num_prompts=$((concurrency * 8))
    local run_dir="$case_dir/c${concurrency}"
    local profile_dir="$run_dir/profile"
    mkdir -p "$profile_dir"

    python -m sglang.benchmark.serving \
      --backend sglang-oai \
      --base-url "$BASE_URL" \
      --dataset-name random \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --num-prompts "$num_prompts" \
      --random-input-len 8192 \
      --random-output-len 1024 \
      --random-range-ratio 1.0 \
      --max-concurrency "$concurrency" \
      --warmup-requests "$warmups" \
      --seed 42 \
      --profile \
      --profile-activities CPU GPU \
      --profile-by-stage \
      --profile-num-steps "$PROFILE_STEPS" \
      --profile-output-dir "$profile_dir" \
      --profile-prefix "${case_name}-c${concurrency}" \
      --output-file "$run_dir/result.jsonl" \
      2>&1 | tee "$run_dir/client.log"
  done

  cleanup_server
  for _ in $(seq 1 60); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
      break
    fi
    sleep 2
  done
}

mkdir -p "$RESULT_ROOT"
run_case "no-radix-cache" 1 1
run_case "no-radix-cache-single-stream-no-pdl" 0 0

echo "All traces saved under: $RESULT_ROOT"
