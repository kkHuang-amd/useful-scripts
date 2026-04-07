#!/bin/bash
# startup_1p1d_prefil_benchmark_tp_ep.sh — 1P1D orchestrator for GPT-OSS-120B with EP+TP (EP=8, TP=8, no DP)
# Launches prefill server, waits for HTTP 200 readiness, launches router, optionally runs sweep
# Run INSIDE the container on gpu-22 (prefill node: 10.2.84.6)
#
# Prerequisites:
#   1. Decode server already running on gpu-23: launch_gptoss_decode_tp_ep.sh
#   2. Model at /models/gpt-oss-120b
#
# Usage:
#   bash /sgl-workspace/scripts/startup_1p1d_prefil_benchmark_tp_ep.sh              # servers + router only
#   bash /sgl-workspace/scripts/startup_1p1d_prefil_benchmark_tp_ep.sh --sweep      # servers + router + sweep
#   bash /sgl-workspace/scripts/startup_1p1d_prefil_benchmark_tp_ep.sh --sweep --fp8  # with FP8 KV cache
#
# EP-only changes vs TP-only:
#   --ep-size 8 added to prefill server
#   --max-running-requests: 128 → 256
#   --chunked-prefill-size 262144 added
#   SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK: set to 12288  (from set_env_vars.sh MORI_MAX_DISPATCH_TOKENS_PREFILL, empirically tuned)
#   MORI_EP_LAUNCH_CONFIG_MODE=MANUAL added

set -e

source /opt/venv/bin/activate 2>/dev/null

PREFILL_HOST="${PREFILL_HOST:-10.235.58.248}"
DECODE_HOST="${DECODE_HOST:-10.235.58.247}"
SERVER_PORT=8000
ROUTER_PORT=30000
MODEL_PATH="${MODEL_PATH:-/dockerx/data/models/gpt-oss-120b/}"
KV_CACHE_DTYPE="auto"
RUN_SWEEP=false
SWEEP_OUTPUT_DIR="/models/sweep_results_ep"

for arg in "$@"; do
    case $arg in
        --sweep) RUN_SWEEP=true ;;
        --fp8)   KV_CACHE_DTYPE="fp8_e4m3"; SWEEP_OUTPUT_DIR="/models/sweep_results_ep_fp8kv" ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_PREFIX="./p-node"
[ "$KV_CACHE_DTYPE" = "fp8_e4m3" ] && LOG_PREFIX="${LOG_PREFIX}_fp8kv"

echo "=============================================="
echo "  GPT-OSS-120B 1P1D EP+TP Benchmark Launcher"
echo "=============================================="
echo "  Prefill host:   $PREFILL_HOST"
echo "  Decode host:    $DECODE_HOST"
echo "  Server port:    $SERVER_PORT"
echo "  Router port:    $ROUTER_PORT"
echo "  TP size:        8"
echo "  EP size:        8"
echo "  KV cache dtype: $KV_CACHE_DTYPE"
echo "  Run sweep:      $RUN_SWEEP"
echo "  Timestamp:      $TIMESTAMP"
echo "=============================================="
echo ""

# --- Step 0: Environment & library setup ---
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

# Hardcoded MORI RDMA buffer value from set_env_vars.sh (MORI_MAX_DISPATCH_TOKENS_PREFILL=12288)
# Source: set_env_vars.sh line 69 — empirically tuned for this hardware, no formula in repo
export SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=12288
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

# --- Step 1: Kill any existing SGLang processes ---
echo "[Step 1/4] Cleaning up existing SGLang processes..."
pkill -f "sglang" 2>/dev/null || true
sleep 3

# --- Step 2: Verify decode server is reachable ---
echo "[Step 2/4] Checking decode server on ${DECODE_HOST}:${SERVER_PORT}..."
DECODE_WAIT=0
while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://${DECODE_HOST}:${SERVER_PORT}/health" 2>/dev/null)
    echo $STATUS
    [ "$STATUS" = "200" ] && break
    sleep 5
    DECODE_WAIT=$((DECODE_WAIT + 1))
    if [ $DECODE_WAIT -ge 120 ]; then
        echo "ERROR: Decode server not reachable at http://${DECODE_HOST}:${SERVER_PORT}"
        echo "Start it first: docker exec sglang_mori bash /sgl-workspace/scripts/launch_gptoss_decode_tp_ep.sh"
        exit 1
    fi
    echo "  ... waiting for decode server ($((DECODE_WAIT * 5))s elapsed)"
done
echo "  Decode server is ready!"

# --- Step 3: Start prefill server (background) ---
echo "[Step 3/4] Starting prefill server on port $SERVER_PORT..."
PREFILL_LOG="${LOG_PREFIX}_prefill_${TIMESTAMP}.log"

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mori \
    --disaggregation-ib-device $IBDEVICES \
    --host 0.0.0.0 \
    --port $SERVER_PORT \
    --tp-size 8 \
    --ep-size 8 \
    --trust-remote-code \
    --ep-dispatch-algorithm fake \
    --load-balance-method round_robin \
    --max-running-requests 256 \
    --mem-fraction-static 0.8 \
    --attention-backend aiter \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --chunked-prefill-size 262144 \
    --decode-log-interval 100 \
    --watchdog-timeout 3600 \
    --disable-radix-cache \
    2>&1 | tee "$PREFILL_LOG" &
PREFILL_PID=$!

# Wait for "fired up" in the server log — the definitive readiness signal.
# /health returns 503 in disaggregation mode even when server is ready, so we check the log instead.
echo "  Prefill server launching (PID: $PREFILL_PID), waiting for ready..."
WAIT_COUNT=0
while ! grep -q "The server is fired up and ready to roll" "$PREFILL_LOG" 2>/dev/null; do
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $WAIT_COUNT -ge 120 ]; then
        echo "ERROR: Prefill server did not start within $((WAIT_COUNT * 5)) seconds"
        kill $PREFILL_PID 2>/dev/null
        exit 1
    fi
    if [ $((WAIT_COUNT % 12)) -eq 0 ]; then
        echo "  ... still waiting ($((WAIT_COUNT * 5))s elapsed)"
    fi
done
echo "  Prefill server is ready!"

# --- Step 4: Launch router + optionally run sweep ---
echo "[Step 4/4] Launching router on port $ROUTER_PORT..."
ROUTER_LOG="${LOG_PREFIX}_router_${TIMESTAMP}.log"

python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --port $ROUTER_PORT \
    --policy random \
    --prefill-policy random \
    --decode-policy random \
    --prefill "http://${PREFILL_HOST}:${SERVER_PORT}" \
    --decode "http://${DECODE_HOST}:${SERVER_PORT}" \
    2>&1 | tee "$ROUTER_LOG" &
ROUTER_PID=$!

sleep 5
# Wait for HTTP 200 on /readiness — router uses this endpoint (not /health)
ROUTER_WAIT=0
while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${ROUTER_PORT}/readiness" 2>/dev/null)
    [ "$STATUS" = "200" ] && break
    sleep 2
    ROUTER_WAIT=$((ROUTER_WAIT + 1))
    if [ $ROUTER_WAIT -ge 60 ]; then
        echo "ERROR: Router failed to start on port $ROUTER_PORT"
        kill $ROUTER_PID 2>/dev/null
        kill $PREFILL_PID 2>/dev/null
        exit 1
    fi
done
echo "  Router is ready on port $ROUTER_PORT!"

echo ""
echo "=============================================="
echo "  1P1D EP+TP Stack Running"
echo "  Prefill:  http://${PREFILL_HOST}:${SERVER_PORT}  (PID: $PREFILL_PID)"
echo "  Router:   http://localhost:${ROUTER_PORT}        (PID: $ROUTER_PID)"
echo "  Decode:   http://${DECODE_HOST}:${SERVER_PORT}   (remote)"
echo ""
echo "  Benchmark endpoint: http://localhost:${ROUTER_PORT}"
echo "=============================================="
echo ""

if [ "$RUN_SWEEP" = true ]; then
    echo "Running sweep benchmark against router (port $ROUTER_PORT)..."
    BASE_URL="http://localhost:${ROUTER_PORT}" \
    OUTPUT_DIR="$SWEEP_OUTPUT_DIR" \
        bash /sgl-workspace/scripts/sweep_benchmark_tp_ep.sh
else
    echo "Skipping sweep (use --sweep to run automatically)"
    echo ""
    echo "To run benchmark manually:"
    echo "  BASE_URL=http://localhost:${ROUTER_PORT} bash /sgl-workspace/scripts/sweep_benchmark_tp_ep.sh"
    echo ""
    echo "Press Ctrl+C to stop all servers."
    wait $PREFILL_PID
fi
