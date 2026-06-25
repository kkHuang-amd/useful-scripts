#!/bin/bash
set -euo pipefail

export SGLANG_USE_AITER=1
export SGLANG_MORI_DISPATCH_DTYPE=auto
export SGLANG_MORI_COMBINE_DTYPE=auto
export SGLANG_MORI_QP_PER_TRANSFER=4
export SGLANG_MORI_NUM_WORKERS=4
export MORI_IO_SQ_BACKOFF_TIMEOUT_US=50000
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export MORI_SHMEM_MODE=ISOLATION
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_LOG_MS=true
export SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS=32
export MORI_MAX_DISPATCH_TOKENS_PREFILL=8192
export MORI_MAX_DISPATCH_TOKENS_DECODE=256
export MORI_MOE_MAX_INPUT_TOKENS_DECODE=2048
export SGLANG_MORI_DISPATCH_INTER_KERNEL_SWITCH_THRESHOLD=4096
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export MORI_APP_LOG_LEVEL=INFO
export MORI_RDMA_SL=3
export MORI_RDMA_TC=96
export MORI_IO_SL=3
export MORI_IO_TC=96
export MORI_IO_TC_DISABLE=0
export NCCL_IB_HCA=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export GLOO_SOCKET_IFNAME=enp81s0f1
export NCCL_SOCKET_IFNAME=enp81s0f1
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_USE_ROCM700A=0
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

export SGLANG_SHARED_EXPERT_TP1=0
export SGLANG_DP_SHARED_EXPERT_LOCAL=0
export SGLANG_DP_USE_GATHERV=0
export SGLANG_DP_USE_REDUCE_SCATTER=0

MODEL="${MODEL:-/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/}"

python3 -m sglang.launch_server \
  --model-path ${MODEL} \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tp-size 8 \
  --ep-size 8 \
  --dp-size 8 \
  --decode-log-interval 100 \
  --watchdog-timeout 3600 \
  --load-balance-method round_robin \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend dsv4 \
  --page-size 256 \
  --swa-full-tokens-ratio 0.1 \
  --disable-shared-experts-fusion \
  --tool-call-parser deepseekv4 \
  --reasoning-parser deepseek-v4 \
  --moe-a2a-backend mori \
  --deepep-mode normal \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 131072 \
  --context-length 9217 \
  --max-running-requests 1024 \
  --max-total-tokens 262144 \
  --disable-radix-cache
