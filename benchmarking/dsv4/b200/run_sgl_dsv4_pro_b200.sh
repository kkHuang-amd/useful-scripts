#!/usr/bin/env bash
# SGLang DeepSeek-V4-Pro high-performance launch for B200 (NVIDIA).
#
# Distilled from the InferenceX agentic benchmark launcher:
#   https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/agentic/dsv4_fp4_b200_sglang.sh
#
# It keeps only the server-launch part (drops the InferenceX benchmark harness,
# AIPerf, trace replay and result plumbing) so it can be used directly for our
# own serving / benchmarking.
#
# Two launch profiles:
#   MODE=dp8   (default) high-throughput DP-attention + DeepEP + mega-MoE path.
#              This is the "效能好" recipe from the InferenceX agentic run:
#              --dp 8 --enable-dp-attention --moe-a2a-backend deepep, plus the
#              SGLANG_OPT_*_MEGA_MOE optimizations and a 32k per-rank chunk.
#   MODE=tp8   plain TP-only baseline (flashinfer_mxfp4 MoE, no dp-attention),
#              matching the canonical single-node B200 command.
#
# Env knobs (all optional):
#   MODE       = dp8 | tp8                (default dp8)
#   TP         = tensor-parallel size     (default 8)
#   EP_SIZE    = expert-parallel size     (default 8, dp8 only)
#   CONC       = target concurrency; sizes cuda-graph-bs / max-running-requests
#                                         (default 256)
#   PORT       = server port              (default 8000)
#   MEM        = mem-fraction-static      (default 0.88, per the B200 recipe)
#   HICACHE    = on | off                 (default off; on = DRAM KV offload tier)
#   HICACHE_RATIO = host/device token ratio for hicache (default 8, max 8)
#   MODEL      = model path               (default DeepSeek-V4-Pro below)
#
# Examples:
#   bash run_sgl_dsv4_pro_b200.sh                 # default dp8 high-throughput
#   MODE=tp8 bash run_sgl_dsv4_pro_b200.sh        # TP-only baseline
#   HICACHE=on CONC=512 bash run_sgl_dsv4_pro_b200.sh
set -euo pipefail

MODE="${MODE:-dp8}"
TP="${TP:-8}"
EP_SIZE="${EP_SIZE:-8}"
CONC="${CONC:-256}"
PORT="${PORT:-8000}"
MEM="${MEM:-0.88}"
HICACHE="${HICACHE:-off}"
HICACHE_RATIO="${HICACHE_RATIO:-8}"
MODEL="${MODEL:-/dockerx/mnt/models/deepseek-ai/DeepSeek-V4-Pro}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Pro}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- common B200 env (verbatim from the InferenceX / canonical B200 command) ---
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1

# Pick ptxas from the CUDA / nvidia wheels for Triton JIT (as InferenceX does).
TRITON_PTXAS_PATH=$(find \
    /usr/local/cuda* \
    /usr/local/lib/python*/dist-packages/nvidia \
    /usr/local/lib/python*/site-packages/nvidia \
    -type f -name ptxas -perm -u+x -print -quit 2>/dev/null || true)
if [ -n "$TRITON_PTXAS_PATH" ]; then
    export TRITON_PTXAS_PATH
    echo "Using ptxas for Triton: $TRITON_PTXAS_PATH"
fi

# AgentX-style sizing: allow request fan-out to exceed CONC without clipping.
MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$CONC
[ "$CUDA_GRAPH_MAX_BS" -gt 64 ] && CUDA_GRAPH_MAX_BS=64

PARALLEL_ARGS=(--tp "$TP")

if [ "$MODE" = "dp8" ]; then
    # --- high-throughput DP-attention + DeepEP + mega-MoE path ---
    DEEPEP_CONFIG='{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'
    export SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1
    export SGLANG_OPT_FIX_HASH_MEGA_MOE=1
    export SGLANG_OPT_USE_FAST_MASK_EP=1
    export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=4096
    export SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=0
    PARALLEL_ARGS+=(
        --dp "$TP"
        --tokenizer-worker-num "$TP"
        --enable-dp-attention
        --enable-dp-attention-local-control-broadcast
        --incremental-streaming-output
        --stream-interval 20
        --dist-init-addr "127.0.0.1:$((PORT + 2000))"
        --ep-size "$EP_SIZE"
        --moe-a2a-backend deepep
        --deepep-config "$DEEPEP_CONFIG"
    )
    CHUNKED_PREFILL_SIZE=32768
else
    # --- TP-only baseline ---
    PARALLEL_ARGS+=(
        --moe-runner-backend flashinfer_mxfp4
        --disable-flashinfer-autotune
    )
    CHUNKED_PREFILL_SIZE=8192
fi

CACHE_ARGS=()
if [ "$HICACHE" = "on" ]; then
    # DRAM KV offload tier. DSv4 HiCache uses a host/device token ratio (<=8),
    # not a byte size. write_through + direct io + page_first_direct layout.
    if [ "$HICACHE_RATIO" -gt 8 ]; then
        echo "Error: HICACHE_RATIO=$HICACHE_RATIO exceeds configured limit 8" >&2
        exit 1
    fi
    export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy write_through
        --hicache-io-backend direct
        --hicache-mem-layout page_first_direct
    )
    echo "HiCache DRAM tier enabled: ratio=$HICACHE_RATIO"
fi

# DeepSeek-V4 thinking chat template (ships alongside the InferenceX launcher);
# fall back gracefully if it is not present next to this script.
CHAT_TEMPLATE_ARGS=()
CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/deepseek_v4_thinking.jinja}"
if [ -f "$CHAT_TEMPLATE" ]; then
    CHAT_TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE")
fi

nvidia-smi || true

set -x
exec python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    "${PARALLEL_ARGS[@]}" \
    --mem-fraction-static "$MEM" \
    --swa-full-tokens-ratio 0.1 \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --watchdog-timeout 1800 \
    --weight-loader-prefetch-checkpoints \
    --enable-metrics \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${CACHE_ARGS[@]}" \
    ${SGL_EXTRA_ARGS:-}
