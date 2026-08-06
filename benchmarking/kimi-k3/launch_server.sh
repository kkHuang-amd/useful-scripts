#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/dockerx/data/models/Kimi-K3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
DCP_SIZE="${DCP_SIZE:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-aiter}"
MEM_FRACTION="${MEM_FRACTION:-0.93}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-8}"
MAMBA_FULL_MEMORY_RATIO="${MAMBA_FULL_MEMORY_RATIO:-0.3}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-1024}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-8192}"
RADIX_CACHE="${RADIX_CACHE:-0}"
ENABLE_PREFILL_CP="${ENABLE_PREFILL_CP:-0}"
CP_STRATEGY="${CP_STRATEGY:-zigzag}"
ATTN_CP_SIZE="${ATTN_CP_SIZE:-$TP}"
ENABLE_INT8_MAMBA_CHECKPOINT="${ENABLE_INT8_MAMBA_CHECKPOINT:-1}"
ENABLE_CACHE_REPORT="${ENABLE_CACHE_REPORT:-1}"
DSPARK_DRAFT_MODEL_PATH="${DSPARK_DRAFT_MODEL_PATH:-}"
DSPARK_BLOCK_SIZE="${DSPARK_BLOCK_SIZE:-7}"

if [[ "$DCP_SIZE" -gt 1 && "$ENABLE_PREFILL_CP" == "1" ]]; then
  echo "error: decode DCP and Prefill CP must not be enabled together" >&2
  exit 1
fi

test -f "$MODEL_PATH/config.json" || {
  echo "error: model config not found at $MODEL_PATH/config.json" >&2
  exit 1
}

command -v sglang >/dev/null 2>&1 || {
  echo "error: 'sglang' is not installed in this image" >&2
  exit 1
}

export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
export SGLANG_AITER_K3_OPT="${SGLANG_AITER_K3_OPT:-1}"
export AITER_FLYDSL_FORCE="${AITER_FLYDSL_FORCE:-1}"
export AITER_SITUV2_A8W4="${AITER_SITUV2_A8W4:-1}"
export SGLANG_K3_FLYDSL_AR_NORM="${SGLANG_K3_FLYDSL_AR_NORM:-1}"

args=(
  serve
  --model-path "$MODEL_PATH"
  --trust-remote-code
  --tp-size "$TP"
  --attention-backend "$ATTENTION_BACKEND"
  --dtype bfloat16
  --mem-fraction-static "$MEM_FRACTION"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
  --mamba-full-memory-ratio "$MAMBA_FULL_MEMORY_RATIO"
  --mamba-ssm-dtype bfloat16
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-prefill-tokens "$MAX_PREFILL_TOKENS"
  --host "$HOST"
  --port "$PORT"
)

if [[ "$DCP_SIZE" -gt 1 ]]; then
  args+=(--dcp-size "$DCP_SIZE")
fi

if [[ "$RADIX_CACHE" == "0" ]]; then
  args+=(--disable-radix-cache)
fi

if [[ "$ENABLE_PREFILL_CP" == "1" ]]; then
  args+=(
    --enable-prefill-cp
    --cp-strategy "$CP_STRATEGY"
    --attn-cp-size "$ATTN_CP_SIZE"
  )
fi

if [[ "$ENABLE_INT8_MAMBA_CHECKPOINT" == "1" ]]; then
  args+=(--enable-int8-mamba-checkpoint)
fi

if [[ "$ENABLE_CACHE_REPORT" == "1" ]]; then
  args+=(--enable-cache-report)
fi

if [[ -n "$DSPARK_DRAFT_MODEL_PATH" ]]; then
  export SGLANG_RAGGED_VERIFY_MODE="${SGLANG_RAGGED_VERIFY_MODE:-static}"
  export SGLANG_PREP_IN_CUDA_GRAPH="${SGLANG_PREP_IN_CUDA_GRAPH:-1}"
  args+=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DSPARK_DRAFT_MODEL_PATH"
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --speculative-attention-mode decode
  )
fi

if [[ -n "${SGL_EXTRA_ARGS:-}" ]]; then
  # Intentional shell-style splitting for command-line overrides.
  read -r -a extra_args <<<"$SGL_EXTRA_ARGS"
  args+=("${extra_args[@]}")
fi

printf 'Launching: sglang'
printf ' %q' "${args[@]}"
printf '\n'

exec sglang "${args[@]}"

