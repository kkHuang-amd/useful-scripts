#!/bin/bash
# launch_gptoss_decode_tp_ep.sh — Decode server for GPT-OSS-120B with EP+TP (EP=8, TP=8, no DP)
# Run INSIDE the container on gpu-23 (decode node: 10.2.84.7)
#
# EP-only changes vs TP-only:
#   --ep-size 8 added
#   --max-running-requests: 128 → 256
#   --cuda-graph-bs: seq 1 128 → seq 1 256
#   SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK: 128 → 160
#   MORI_EP_LAUNCH_CONFIG_MODE=MANUAL added (MORI-EP topology init)

source /opt/venv/bin/activate 2>/dev/null

export GLOO_SOCKET_IFNAME=enp196s0
export NCCL_SOCKET_IFNAME=enp196s0
export NCCL_IB_HCA=`rdma dev show | awk -F': ' '{print $2}' | awk '{printf "%s:1,", $1}' | sed 's/,$/\n/'`
export NCCL_IB_HCA="ionic_0:1,ionic_1:1,ionic_2:1,ionic_3:1"
export NCCL_IB_GID_INDEX=1
export IBDEVICES=`rdma dev show | awk -F': ' '{print $2}' | awk '{print $1}' | paste -sd ','`
export IBDEVICES="ionic_0,ionic_1,ionic_2,ionic_3"
export MORI_RDMA_SL=3
export MORI_RDMA_TC=104
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# Hardcoded MORI RDMA buffer value from set_env_vars.sh (MORI_MAX_DISPATCH_TOKENS_DECODE=160)
# Source: set_env_vars.sh line 70 — empirically tuned for this hardware, no formula in repo
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160
export MORI_APP_LOG_LEVEL=INFO
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_USE_AITER=1

# MORI-IO variables
export MORI_SHMEM_MODE=ISOLATION
export MORI_IO_QP_MAX_SEND_WR=16384
export MORI_IO_QP_MAX_CQE=32768
export MORI_IO_QP_MAX_SGE=4

# EP-specific: controls MORI-EP expert dispatch topology initialization
export MORI_EP_LAUNCH_CONFIG_MODE=MANUAL

export SGLANG_MORI_DISPATCH_DTYPE=fp4
export SGLANG_MORI_FP8_COMB=True
export SGLANG_MORI_FP4_DISP=True
#export SGLANG_MORI_FP8_DISP=False

export PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/aiter:$PYTHONPATH

ln -sf /usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71 /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null
ln -sf /usr/lib/x86_64-linux-gnu/libionic.so.1 /usr/lib/x86_64-linux-gnu/libionic.so 2>/dev/null
ln -sf /usr/lib/x86_64-linux-gnu/libionic.so.1.0.54.0-149.g3304be71 /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so 2>/dev/null
ldconfig 2>/dev/null

TORCH_LIB=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib'))" 2>/dev/null)
export LD_LIBRARY_PATH=${TORCH_LIB}:${LD_LIBRARY_PATH}

python3 -m sglang.launch_server \
--model-path /dockerx/data/models/gpt-oss-120b/ \
--disaggregation-mode decode \
--disaggregation-transfer-backend mori \
--disaggregation-ib-device $IBDEVICES \
--host 0.0.0.0 \
--port 8000 \
--tp-size 8 \
--ep-size 8 \
--trust-remote-code \
--ep-dispatch-algorithm fake \
--load-balance-method round_robin \
--max-running-requests 256 \
--mem-fraction-static 0.85 \
--attention-backend aiter \
--kv-cache-dtype auto \
--decode-log-interval 100 \
--watchdog-timeout 3600 \
--cuda-graph-bs $(seq 1 256) \
--prefill-round-robin-balance \
2>&1 | tee /models/sglang_decode_ep_$(date +%Y%m%d_%H%M%S).log
