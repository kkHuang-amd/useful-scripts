#!/usr/bin/env bash
# Unified SGLang DeepSeek-V4-Pro launcher (8x MI355X, tp8). One script, best-perf
# defaults, selectable MODE. Supersedes run_sgl_dsv4{,_aligned,_mori-ep}.sh.
#
# MODE (default: dp):
#   dp           TP8 + DP-attention, TP-MoE (gatherv+reduce_scatter). Best baseline.
#                8k/1k: ~37.9k tok/s @c512, ~30.0k @c256.
#   dp-tbo       dp + non-EP DP two-batch-overlap (overlaps DP all_gatherv +
#                reduce_scatterv with the other ubatch's attn+MoE compute).
#                Requires the TBO build (DSV4 non-EP DP TBO op decomposition).
#                NOTE: stable at <=8192/rank & <=conc256; larger crashes (HSA, WIP).
#   mori-ep      TP8 + DP-attention + EP via mori a2a (no TBO).
#   mori-ep-tbo  mori-ep + EP two-batch-overlap (overlaps mori dispatch/combine).
#   flydsl       FlyDSL-EP dynamic-recv + guarded prefill delayer.
#   flydsl-tbo   flydsl + prefill TBO, dedicated comm stream, block64 tuning;
#                guarded delayer defaults ON (DELAYER=off reproduces tuning baseline).
#   tp8          plain TP8, no DP-attention (low-concurrency / long-context).
#
# Knobs (env): MODEL PORT CHUNK MEM CGBS MAXRUN SWA DELAYER CTXLEN
#   CHUNK is the GLOBAL --chunked-prefill-size; sglang divides by dp_size(8) for
#   the per-rank chunk (e.g. CHUNK=65536 -> 8192/rank). DP best = 65536.
#
# TBO modes auto-set GPU_MAX_HW_QUEUES=5 (ROCm HSA stability for sglang TBO) and
# --enable-two-batch-overlap. Baseline modes leave the HW-queue pool uncapped
# (capping throttles the non-TBO baseline ~3-12%).
#
# The TBO code lives on branch feat/dsv4-ep-tbo-prefill; this script pins
# PYTHONPATH to the editable tree so `sglang`/launch_server uses it.
set -euo pipefail

MODE="${MODE:-dp}"
MODEL="${MODEL:-/dockerx/data/deepseek-ai/DeepSeek-V4-Pro}"
PORT="${PORT:-8000}"
AITER_REPO="${AITER_REPO:-/sgl-workspace/aiter}"
CHUNK="${CHUNK:-65536}"        # global; /8 = per-rank (65536 -> 8192/rank, DP best)
MEM="${MEM:-0.90}"
CGBS="${CGBS:-1024}"
MAXRUN="${MAXRUN:-1024}"
SWA="${SWA:-0.15}"
CTXLEN="${CTXLEN:-}"           # optional --context-length
DELAYER="${DELAYER:-auto}"     # auto|on|off; auto is mode-specific below

case "$MODE" in
  dp|dp-tbo|mori-ep|mori-ep-tbo|mori-epv2|tp8|megamoe|flydsl|flydsl-tbo) ;;
  *) echo "Unknown MODE='$MODE' (use dp|dp-tbo|mori-ep|mori-ep-tbo|mori-epv2|tp8|megamoe|flydsl|flydsl-tbo)"; exit 1 ;;
esac

is_tbo=0;  [[ "$MODE" == *tbo* ]] && is_tbo=1
is_mori=0; [[ "$MODE" == mori-ep* ]] && is_mori=1
is_mori_epv2=0; [[ "$MODE" == mori-epv2 ]] && is_mori_epv2=1
[[ "$is_mori_epv2" = "1" ]] && is_mori=0
is_megamoe=0; [[ "$MODE" == megamoe ]] && is_megamoe=1
is_flydsl=0; [[ "$MODE" == flydsl* ]] && is_flydsl=1

# --- common env (DSV4 fused-kernel optimal set, ROCM700A=0) ---
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_USE_AITER=1
# Hard-set ROCM700A=0 (NOT ${:-0}): =0 is the optimal DSV4 path (aiter MAX_LEN
# decode kernels, ~+5% over =1) and must not be overridden by a polluted shell env.
export SGLANG_USE_ROCM700A=0
export SGLANG_OPT_USE_FUSED_COMPRESS=true
export SGLANG_HACK_FLASHMLA_BACKEND=${SGLANG_HACK_FLASHMLA_BACKEND:-unified_kv_triton}
export SGLANG_OPT_FP8_WO_A_GEMM=false
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=false
export SGLANG_OPT_USE_TOPK_V2=false
export SGLANG_OPT_USE_AITER_INDEXER=${SGLANG_OPT_USE_AITER_INDEXER:-true}
export SGLANG_OPT_USE_TILELANG_INDEXER=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export SGLANG_ROCM_USE_MULTI_STREAM=false
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_EAGER_INPUT_NO_COPY=true
export SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=${SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD:-1}

# Use the editable tree (TBO ops live there; harmless for baseline modes).
export PYTHONPATH=${SGLANG_PYTHONPATH:-/sgl-workspace/sglang/python}:${PYTHONPATH:-}

PARALLEL_ARGS="--tp 8"
EXTRA_ARGS=""

if [ "$is_flydsl" = "1" ]; then
  # FlyDSL un-fused a2a: replaces mori-ep's dispatch/combine with FlyDSL's
  # standalone intranode dispatch/combine (aiter PR #3924), keeping aiter
  # fused_moe for the GEMM. Same EP/DP-attention topology as mori-ep; only the
  # a2a backend differs. Shared-expert TP1 is handled in-code (is_flydsl()).
  export SGLANG_SHARED_EXPERT_TP1=0
  export SGLANG_DP_SHARED_EXPERT_LOCAL=0
  export SGLANG_DP_USE_GATHERV=0
  export SGLANG_DP_USE_REDUCE_SCATTER=0
  # FlyDSL transports over mori's symmetric heap / intranode P2P.
  export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-enp81s0f1}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp81s0f1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp81s0f1}
  export MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-8G}
  export PYTHONPATH=/sgl-workspace/flydsl-0.2.4-py:/sgl-workspace/sglang-flydsl-a2a/python:${AITER_REPO}:${PYTHONPATH}
  export SGLANG_FLYDSL_DISPATCH_DTYPE=${SGLANG_FLYDSL_DISPATCH_DTYPE:-bf16}
  export SGLANG_FLYDSL_DYNAMIC_RECV_CAP=${SGLANG_FLYDSL_DYNAMIC_RECV_CAP:-1}
  # DSV4 FlyDSL-TBO tuning: a smaller comm grid leaves CUs for the overlapped
  # ubatch GEMMs. Ignored by the dispatcher when TBO is disabled.
  export SGLANG_FLYDSL_TBO_BLOCK_NUM=${SGLANG_FLYDSL_TBO_BLOCK_NUM:-64}
  PARALLEL_ARGS="--tp 8 --ep-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend flydsl --moe-dense-tp-size 1 --enable-dp-lm-head --load-balance-method round_robin"
  if [ "$is_tbo" = "1" ]; then
    # Guarded TBO+delayer is validated; DELAYER=off reproduces the tuning baseline.
    [ "$DELAYER" != "off" ] && PARALLEL_ARGS="$PARALLEL_ARGS --enable-prefill-delayer"
  else
    # Guarded delayer is the validated FlyDSL no-TBO throughput/TTFT winner.
    [ "$DELAYER" != "off" ] && PARALLEL_ARGS="$PARALLEL_ARGS --enable-prefill-delayer"
  fi
elif [ "$is_megamoe" = "1" ]; then
  # FlyDSL MegaMoE: fused single-op MoE (dispatch+gemm1+quant+gemm2+combine) over
  # mori intranode a2a. Same EP/DP-attention topology as mori-ep; only the MoE
  # backend differs. Requires the sglang-megamoe checkout (mega_moe_flydsl.py +
  # cohere2 fix + a8w4-interleave weight-prep fix) and the FlyDSL mega_moe_v1
  # workspace on PYTHONPATH.
  export SGLANG_SHARED_EXPERT_TP1=0
  export SGLANG_DP_SHARED_EXPERT_LOCAL=0
  export SGLANG_DP_USE_GATHERV=0
  export SGLANG_DP_USE_REDUCE_SCATTER=0
  export SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1
  export SGLANG_AMD_FLYDSL_KERNELS_PATH=/sgl-workspace/FlyDSL
  export SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=${SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR:-8192}
  export MORI_SHMEM_HEAP_SIZE=${MORI_SHMEM_HEAP_SIZE:-40G}
  # MegaMoE requires mori's STATIC_HEAP mode (default, single-node). Do NOT set
  # MORI_SHMEM_MODE=ISOLATION here — it leaves the symmetric heap unallocated
  # ("Pointer not in symmetric heap [0x0,0x0)" -> NULL-deref in FlyDSL stage1).
  # megamoe integration now lives in sglang-upstream (ported), so it shares the
  # common PYTHONPATH set above — same codebase as dp / mori-ep for a fair A/B.
  PARALLEL_ARGS="--tp 8 --ep-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend megamoe --moe-dense-tp-size 1 --enable-dp-lm-head --load-balance-method round_robin"
elif [ "$is_mori_epv2" = "1" ]; then
  # MORI EPv2: FlyDSL kernels over cco-LSA. Keep this distinct from MORI v1.
  export SGLANG_SHARED_EXPERT_TP1=0
  export SGLANG_DP_SHARED_EXPERT_LOCAL=0
  export SGLANG_DP_USE_GATHERV=0
  export SGLANG_DP_USE_REDUCE_SCATTER=0
  export MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-enp81s0f1}
  export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp81s0f1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp81s0f1}
  export SGLANG_MORI_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_EPV2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-8192}
  export SGLANG_MORI_EPV2_PER_RANK_VMM_GB=${SGLANG_MORI_EPV2_PER_RANK_VMM_GB:-4}
  export AITER_FLYDSL_EP_NO_FAKE_EXPERT=1
  export PYTHONPATH=/tmp/sglang-mori-epv2-compare/python:/tmp/mori-epv2-serving-fix-install:/sgl-workspace/flydsl-0.2.4-py:${AITER_REPO}:${PYTHONPATH}
  [ "$DELAYER" != "off" ] && EXTRA_ARGS="$EXTRA_ARGS --enable-prefill-delayer"
  PARALLEL_ARGS="--tp 8 --ep-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend mori-epv2 --deepep-mode normal --moe-dense-tp-size 1 --enable-dp-lm-head --load-balance-method round_robin"
elif [ "$is_mori" = "1" ]; then
  # EP via mori a2a. MoE comm handled by mori -> DP gatherv path OFF.
  export SGLANG_SHARED_EXPERT_TP1=0
  export SGLANG_DP_SHARED_EXPERT_LOCAL=0
  export SGLANG_DP_USE_GATHERV=0
  export SGLANG_DP_USE_REDUCE_SCATTER=0
  # mori runtime tuning (from run_sgl_dsv4_mori-ep.sh).
  export SGLANG_MORI_DISPATCH_DTYPE=${SGLANG_MORI_DISPATCH_DTYPE:-auto}
  export SGLANG_MORI_COMBINE_DTYPE=${SGLANG_MORI_COMBINE_DTYPE:-auto}
  export SGLANG_MORI_QP_PER_TRANSFER=4
  export SGLANG_MORI_NUM_WORKERS=4
  export MORI_IO_SQ_BACKOFF_TIMEOUT_US=50000
  export MORI_IO_QP_MAX_SEND_WR=16384
  export MORI_IO_QP_MAX_CQE=32768
  export MORI_IO_QP_MAX_SGE=4
  export MORI_SHMEM_MODE=ISOLATION
  export MORI_MAX_DISPATCH_TOKENS_PREFILL=8192
  export MORI_MAX_DISPATCH_TOKENS_DECODE=256
  export MORI_MOE_MAX_INPUT_TOKENS_DECODE=2048
  export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=4096
  export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
  export MORI_APP_LOG_LEVEL=INFO
  export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
  export GLOO_SOCKET_IFNAME=enp81s0f1
  export NCCL_SOCKET_IFNAME=enp81s0f1
  export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-16384}
  export SGLANG_ENABLE_SPEC_V2=1
  export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
  PARALLEL_ARGS="--tp 8 --ep-size 8 --dp-size 8 --enable-dp-attention --moe-a2a-backend mori --deepep-mode ${DEEPEP_MODE:-normal} --moe-dense-tp-size 1 --enable-dp-lm-head --load-balance-method round_robin"
elif [ "$MODE" = "tp8" ]; then
  # plain TP8, no DP-attention.
  export SGLANG_SHARED_EXPERT_TP1=0
  export SGLANG_DP_SHARED_EXPERT_LOCAL=0
  export SGLANG_DP_USE_GATHERV=0
  export SGLANG_DP_USE_REDUCE_SCATTER=0
  PARALLEL_ARGS="--tp 8"
else
  # dp / dp-tbo: DP-attention + TP-MoE gatherv path.
  export SGLANG_SHARED_EXPERT_TP1=1
  export SGLANG_DP_SHARED_EXPERT_LOCAL=1
  export SGLANG_DP_USE_GATHERV=1
  export SGLANG_DP_USE_REDUCE_SCATTER=1
  PARALLEL_ARGS="--tp 8 --dp 8 --enable-dp-attention"
  [ "$DELAYER" != "off" ] && PARALLEL_ARGS="$PARALLEL_ARGS --enable-prefill-delayer"
fi

if [ "$is_tbo" = "1" ]; then
  EXTRA_ARGS="$EXTRA_ARGS --enable-two-batch-overlap"
  export GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES:-5}
  # Non-EP DP TBO auto-enables from --enable-dp-attention + --enable-two-batch-overlap
  # (a2a backend none); mori/flydsl TBO use their EP a2a paths.
fi

[ -n "$CTXLEN" ] && EXTRA_ARGS="$EXTRA_ARGS --context-length $CTXLEN"

set -x
# LAUNCHER override allows dry-runs: LAUNCHER=echo MODE=dp-tbo bash run_sgl_dsv4_unified.sh
exec ${LAUNCHER:-python3 -m sglang.launch_server} \
  --model-path "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  ${PARALLEL_ARGS} \
  --attention-backend dsv4 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 256 \
  --swa-full-tokens-ratio "${SWA}" \
  --mem-fraction-static "${MEM}" \
  --chunked-prefill-size "${CHUNK}" \
  --cuda-graph-max-bs "${CGBS}" \
  --max-running-requests "${MAXRUN}" \
  --disable-radix-cache \
  --disable-shared-experts-fusion \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  ${EXTRA_ARGS} \
  ${SGL_EXTRA_ARGS:-}
