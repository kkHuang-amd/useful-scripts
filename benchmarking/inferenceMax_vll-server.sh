export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_USE_AITER_MHA=0
#export VLLM_TORCH_PROFILER_DIR=/root/traces/
#--profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}' \

PORT=${PORT:-8000}
MODEL="openai/gpt-oss-120b"
TP=8
MAX_MODEL_LEN=131072

set -x
vllm serve $MODEL --port $PORT \
--tensor-parallel-size=$TP \
--gpu-memory-utilization 0.95 \
--max-model-len $MAX_MODEL_LEN \
--compilation-config  '{"cudagraph_mode": "FULL_AND_PIECEWISE"}' \
--block-size=64 \
--no-enable-prefix-caching \
--disable-log-requests
