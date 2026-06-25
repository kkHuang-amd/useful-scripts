#!/usr/bin/env bash
# ATOM DeepSeek-V4-Pro launch, ALIGNED with run_sgl_dsv4_aligned.sh for
# apple-to-apple. Explicitly pins the knobs that previously differed from the
# aligned SGLang config:
#   --block-size 256            (match SGLang --page-size 256; ATOM default was 128/16)
#   --no-enable_prefix_caching  (explicit OFF; ATOM auto-disables for DSV4 anyway,
#                                matches SGLang --disable-radix-cache)
#   --kv_cache_dtype fp8        (match SGLang --kv-cache-dtype fp8_e4m3)
#   --max-num-batched-tokens 16384   (== SGLang --chunked-prefill-size 16384)
#   --max-num-seqs 512          (== SGLang --max-running-requests 512)
#   --gpu-memory-utilization 0.9(== SGLang --mem-fraction-static 0.90)
# cudagraph-capture default already [1..512] == SGLang --cuda-graph-max-bs 512.
#
# DP_MODE=tp8     -> plain TP8 (no dp-attention)
# DP_MODE=tp8dp8  -> TP8 + --enable-dp-attention   (default)
set -euo pipefail

DP_MODE="${DP_MODE:-tp8dp8}"
MODEL="${MODEL:-/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/}"
PORT="${PORT:-8000}"

export ATOM_DISABLE_MMAP=true
export ATOM_MOE_GU_ITLV=1
export AITER_BF16_FP8_MOE_BOUND=0

DP_ARGS=""
if [ "$DP_MODE" = "tp8dp8" ]; then
    DP_ARGS="--enable-dp-attention"
fi

# Optional torch profiler dir (enables /start_profile & /stop_profile dumps).
PROF_ARGS=""
if [ -n "${ATOM_TORCH_PROFILER_DIR:-}" ]; then
    PROF_ARGS="--torch-profiler-dir ${ATOM_TORCH_PROFILER_DIR}"
fi

set -x
exec python3 -m atom.entrypoints.openai_server \
    --model "${MODEL}" \
    --server-port "${PORT}" \
    -tp 8 \
    --trust-remote-code \
    ${DP_ARGS} \
    ${PROF_ARGS} \
    --kv_cache_dtype fp8 \
    --block-size 256 \
    --no-enable_prefix_caching \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 512 \
    --gpu-memory-utilization 0.9 \
    ${ATOM_EXTRA_ARGS:-}
