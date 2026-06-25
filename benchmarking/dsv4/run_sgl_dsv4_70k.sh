#!/usr/bin/env bash
# SGLang DeepSeek-V4-Pro launch for the 70k/200 low-concurrency sweep (2026-06-25).
# Pins the sglang-upstream clone via PYTHONPATH (avoids the /sgl-workspace
# namespace-shadow that otherwise breaks `import sglang`), and exposes the three
# sweep knobs as env vars.
#
#   MODE   = tp8 | tp8dp8         (default tp8; tp8dp8 only meaningful for conc>=8)
#   CHUNK  = per-rank chunk tokens(default 32768; dp mode multiplies by dp_size=8)
#   SWA    = swa-full-tokens-ratio(default 0.1; only tune at conc 16/32)
#   DELAYER= on | off             (default on)
#   PORT   = server port          (default 8000)
#
# Example:
#   MODE=tp8    CHUNK=32768 bash run_sgl_dsv4_70k.sh
#   MODE=tp8dp8 CHUNK=32768 bash run_sgl_dsv4_70k.sh
set -euo pipefail

# --- pin sglang-upstream (verified 2026-06-25; keep mori+aiter on the path) ---
export PYTHONPATH=/sgl-workspace/sglang-upstream/python:/sgl-workspace/mori:/sgl-workspace/aiter:${PYTHONPATH:-}

MODE="${MODE:-tp8}"
CHUNK="${CHUNK:-32768}"
SWA="${SWA:-0.1}"
DELAYER="${DELAYER:-on}"
MODEL="${MODEL:-/dockerx/data/deepseek-ai/DeepSeek-V4-Pro}"
PORT="${PORT:-8000}"
CTXLEN="${CTXLEN:-73728}"          # 70k in + 200 out + headroom, multiple of page 256
CGMAXBS="${CGMAXBS:-64}"           # conc<=32; small to free HBM for the big 70k KV
MAXRUN="${MAXRUN:-64}"
MEM="${MEM:-0.92}"                 # mem-fraction-static; raise for the 70k KV ceiling

# --- env (verbatim perf set from run_sgl_dsv4_aligned.sh) ---
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

DP_ARGS=""
CHUNK_EFF="$CHUNK"
if [ "$MODE" = "tp8dp8" ]; then
    DP_ARGS="--dp 8 --enable-dp-attention"
    CHUNK_EFF=$((CHUNK * 8))        # dp-attn divides chunked_prefill_size by dp_size
    # dp-only collective optimizations (prefill gatherv + decode reduce_scatter + SE-local)
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    if [ "$DELAYER" != "off" ]; then
        DP_ARGS="$DP_ARGS --enable-prefill-delayer"
    fi
fi

set -x
exec sglang serve \
    --model-path "${MODEL}" \
    --trust-remote-code \
    --tp 8 \
    ${DP_ARGS} \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --page-size 256 \
    --mem-fraction-static "${MEM}" \
    --swa-full-tokens-ratio "${SWA}" \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --kv-cache-dtype fp8_e4m3 \
    --context-length "${CTXLEN}" \
    --chunked-prefill-size "${CHUNK_EFF}" \
    --cuda-graph-max-bs "${CGMAXBS}" \
    --max-running-requests "${MAXRUN}" \
    --port "${PORT}" \
    ${SGL_EXTRA_ARGS:-}
