#!/usr/bin/env bash
# SGLang DeepSeek-V4-Pro launch, ALIGNED with run_atom_dsv4.sh for apple-to-apple.
# Based on run_sgl_dsv4.sh, plus knobs matched to ATOM's resolved config:
#   --kv-cache-dtype fp8_e4m3   (ATOM kv_cache_dtype=fp8)
#   --chunked-prefill-size 16384(ATOM max_num_batched_tokens/attn_prefill_chunk_size=16384)
#   --cuda-graph-max-bs 512     (ATOM cudagraph capture max = 512)
#   --max-running-requests 512  (ATOM max_num_seqs=512)
#   --mem-fraction-static 0.90  (ATOM gpu_memory_utilization=0.9)
#   radix/prefix cache OFF      (ATOM auto-disables prefix cache for DSV4)
# Remaining known diff: page-size 256 (SGLang) vs ATOM KV block_size 128 (minor).
#
# DP_MODE=tp8     -> plain TP8 (no dp-attention)
# DP_MODE=tp8dp8  -> TP8 + --dp 8 --enable-dp-attention   (default)
set -euo pipefail

DP_MODE="${DP_MODE:-tp8dp8}"
EP_MODE="${EP_MODE:-""}"
MODEL="${MODEL:-/dockerx/data/deepseek-ai/DeepSeek-V4-Pro}"
PORT="${PORT:-8000}"

# --- env (verbatim from run_sgl_dsv4.sh) ---
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=${SGLANG_USE_ROCM700A:-0}
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_AITER_INDEXER=true
export SGLANG_OPT_USE_TILELANG_INDEXER=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_EAGER_INPUT_NO_COPY=true

export SGLANG_SHARED_EXPERT_TP1=1
export SGLANG_DP_SHARED_EXPERT_LOCAL=1
export SGLANG_DP_USE_GATHERV=1
export SGLANG_DP_USE_REDUCE_SCATTER=1

DP_ARGS=""
if [ "$DP_MODE" = "tp8dp8" ]; then
    DP_ARGS="--dp 8 --enable-dp-attention"
    # DELAYER=off removes --enable-prefill-delayer (A/B knob); default ON.
    if [ "${DELAYER:-on}" != "off" ]; then
        DP_ARGS="$DP_ARGS --enable-prefill-delayer"
    fi
fi

EP_ARGS=""
if [ "$EP_MODE" = "mori" ]; then
    EP_ARGS="--ep-size 8 --moe-a2a-backend mori --load-balance-method round_robin --deepep-mode normal --moe-dense-tp-size 1 --enable-dp-lm-head"
fi

set -x
exec sglang serve \
    --model-path "${MODEL}" \
    --trust-remote-code \
    --tp 8 \
    ${DP_ARGS} \
    ${EP_ARGS} \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --page-size 256 \
    --mem-fraction-static 0.90 \
    --swa-full-tokens-ratio 0.15 \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size 16384 \
    --cuda-graph-max-bs 1024 \
    --max-running-requests 1024 \
    --port "${PORT}" \
    ${SGL_EXTRA_ARGS:-}
