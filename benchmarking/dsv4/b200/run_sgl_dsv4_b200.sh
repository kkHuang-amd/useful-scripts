#!/usr/bin/env bash
# SGLang DeepSeek-V4-Pro launch for B200 (NVIDIA), 2026-06-25.
# Faithful to the canonical B200 command (flashinfer_mxfp4 MoE, JIT norm/indexer,
# custom_all_reduce_v2, ...) and adds a DP-attention toggle to validate the
# dp-only collective optimizations on B200:
#   - SGLANG_DP_USE_REDUCE_SCATTER=1   (decode reduce_scatter, default OFF on CUDA)
#   - SGLANG_DP_USE_GATHERV=1          (prefill gatherv, default OFF)
#
# Pins the sglang-upstream clone via PYTHONPATH because the default `import sglang`
# on this box resolves to /sgl-workspace/sglang (HEAD a17753e), NOT the upstream
# clone at /sgl-workspace/sglang-upstream (HEAD ffb1afd5e) that carries the DP work.
#
#   MODE = tp8 | tp8dp8   (default tp8dp8; tp8dp8 = --dp 8 --enable-dp-attention)
#   TP   = tensor-parallel size            (default 8)
#   PORT = server port                     (default 8000)
#   MEM  = mem-fraction-static             (default 0.90)
#   CHUNK= --chunked-prefill-size          (default 8192; per the B200 command)
#
# Example:
#   MODE=tp8dp8 bash run_sgl_dsv4_b200.sh
#   MODE=tp8    bash run_sgl_dsv4_b200.sh     # baseline, no dp
set -euo pipefail

# --- pin sglang-upstream (the DP-attention work lives here, not /sgl-workspace/sglang) ---
export PYTHONPATH=/sgl-workspace/sglang-upstream/python:${PYTHONPATH:-}

MODE="${MODE:-tp8dp8}"
TP="${TP:-8}"
PORT="${PORT:-8000}"
MEM="${MEM:-0.90}"
CHUNK="${CHUNK:-8192}"
MODEL="${MODEL:-/dockerx/raid/models--deepseek-ai--DeepSeek-V4-Pro}"

# --- B200 env (verbatim from the canonical B200 command) ---
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1

DP_ARGS=""
if [ "$MODE" = "tp8dp8" ]; then
    DP_ARGS="--dp ${TP} --enable-dp-attention"
    # dp-only collective optimizations under validation on B200:
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export SGLANG_DP_USE_GATHERV=1
fi

set -x
exec sglang serve \
    --model-path "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --trust-remote-code \
    --tp "${TP}" \
    ${DP_ARGS} \
    --disable-radix-cache \
    --max-running-requests 256 \
    --mem-fraction-static "${MEM}" \
    --swa-full-tokens-ratio 0.1 \
    --moe-runner-backend flashinfer_mxfp4 \
    --chunked-prefill-size "${CHUNK}" \
    --disable-flashinfer-autotune \
    ${SGL_EXTRA_ARGS:-}
