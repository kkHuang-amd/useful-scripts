#!/usr/bin/env bash
# SGLang DeepSeek-V4-Pro launch for the 70k/300 low-concurrency sweep on B200.
# B200 (NVIDIA) flavor of run_sgl_dsv4_70k.sh: keeps the canonical B200 env set
# (flashinfer_mxfp4 MoE, JIT norm/indexer, custom_all_reduce_v2, swa-split-leaf)
# instead of the ROCm/MI35x env (AITER/ROCm700A/flashmla/...), and adds the
# 70k-specific serving args (context-length, fp8 KV, small cuda-graph bs).
#
# Pins sglang-upstream via PYTHONPATH (default `import sglang` on this box points
# at /sgl-workspace/sglang, NOT the upstream clone that carries the DP work).
#
#   MODE   = tp8 | tp8dp8          (default tp8; tp8dp8 = --dp 8 --enable-dp-attention)
#   CHUNK  = per-rank chunk tokens (default 32768; dp mode multiplies by dp_size)
#   SWA    = swa-full-tokens-ratio (default 0.1)
#   DELAYER= on | off             (default off; OFF wins for prefill-dominated 70k)
#   DP_COLL= on | off             (default on; dp reduce_scatter+gatherv collectives)
#   MEM    = mem-fraction-static   (default 0.90; dp MoE may need lower to avoid OOM)
#   KVDTYPE= kv-cache-dtype        (default fp8_e4m3; memory-critical at 70k)
#   CTXLEN = context-length        (default 73728 = 70k in + 300 out + headroom)
#   SE_LOCAL = on | off            (default off; dp shared-expert-local PoC, perf only)
#   PORT   = server port           (default 8000)
#
# Example:
#   MODE=tp8    CHUNK=32768 bash run_sgl_dsv4_70k_b200.sh
#   MODE=tp8dp8 CHUNK=16384 MEM=0.80 bash run_sgl_dsv4_70k_b200.sh
set -euo pipefail

# --- pin sglang-upstream ---
export PYTHONPATH=/sgl-workspace/sglang-upstream/python:${PYTHONPATH:-}

MODE="${MODE:-tp8}"
CHUNK="${CHUNK:-32768}"
SWA="${SWA:-0.1}"
DELAYER="${DELAYER:-off}"
DP_COLL="${DP_COLL:-on}"
MEM="${MEM:-0.90}"
KVDTYPE="${KVDTYPE:-fp8_e4m3}"
CTXLEN="${CTXLEN:-73728}"
SE_LOCAL="${SE_LOCAL:-off}"
TP="${TP:-8}"
CGMAXBS="${CGMAXBS:-64}"
MAXRUN="${MAXRUN:-64}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro}"

# --- B200 env (verbatim from the canonical B200 command) ---
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1

DP_ARGS=""
CHUNK_EFF="$CHUNK"
SE_FUSION_ARG=""
if [ "$MODE" = "tp8dp8" ]; then
    DP_ARGS="--dp ${TP} --enable-dp-attention"
    CHUNK_EFF=$((CHUNK * TP))      # dp-attn divides chunked_prefill_size by dp_size
    # validated-correct dp collectives (gsm8k acc 0.97 on B200); DP_COLL=off to compare:
    if [ "$DP_COLL" = "off" ]; then
        export SGLANG_DP_USE_REDUCE_SCATTER=0
        export SGLANG_DP_USE_GATHERV=0
    else
        export SGLANG_DP_USE_REDUCE_SCATTER=1
        export SGLANG_DP_USE_GATHERV=1
    fi
    if [ "$SE_LOCAL" = "on" ]; then
        export SGLANG_SHARED_EXPERT_TP1=1
        export SGLANG_DP_SHARED_EXPERT_LOCAL=1
        SE_FUSION_ARG="--disable-shared-experts-fusion"
    fi
    if [ "$DELAYER" != "off" ]; then
        DP_ARGS="$DP_ARGS --enable-prefill-delayer"
    fi
fi

set -x
exec sglang serve \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --tp "${TP}" \
    ${DP_ARGS} \
    ${SE_FUSION_ARG} \
    --disable-radix-cache \
    --max-running-requests "${MAXRUN}" \
    --mem-fraction-static "${MEM}" \
    --swa-full-tokens-ratio "${SWA}" \
    --moe-runner-backend flashinfer_mxfp4 \
    --chunked-prefill-size "${CHUNK_EFF}" \
    --disable-flashinfer-autotune \
    --kv-cache-dtype "${KVDTYPE}" \
    --context-length "${CTXLEN}" \
    --cuda-graph-max-bs "${CGMAXBS}" \
    ${SGL_EXTRA_ARGS:-}
