#!/bin/bash

# ===== Default parameters =====
INPUT_LEN=${1:-8192}
OUTPUT_LEN=${2:-1024}
ENABLE_PROFILE=${3:-0}   # 1 = enable profile, 0 = disable

# ===== Timestamp =====
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "INPUT_LEN=${INPUT_LEN}"
echo "OUTPUT_LEN=${OUTPUT_LEN}"
echo "PROFILE=${ENABLE_PROFILE}"
echo "TIMESTAMP=${TIMESTAMP}"

for concurrency in 128 256
do
    prompt=$((concurrency * 2))

    #warmup
    CMD="python3 -m sglang.bench_serving \
        --port 8000 \
        --dataset-name random \
        --random-input ${INPUT_LEN} \
        --random-output ${OUTPUT_LEN} \
        --random-range-ratio 1.0 \
        --max-concurrency ${concurrency} \
        --num-prompt ${prompt}"
    eval ${CMD}


    # start to run
    prompt=$((concurrency * 8))
    LOG_FILE="mi35x_${INPUT_LEN}_${OUTPUT_LEN}_tp8-dp8_c-${concurrency}_${TIMESTAMP}.log"

    CMD="python3 -m sglang.bench_serving \
	--port 8000 \
        --dataset-name random \
        --random-input ${INPUT_LEN} \
        --random-output ${OUTPUT_LEN} \
        --random-range-ratio 1.0 \
        --max-concurrency ${concurrency} \
        --num-prompt ${prompt}"

    # ===== Optional profile =====
    if [ "${ENABLE_PROFILE}" -eq 1 ]; then
        CMD="${CMD} --profile --profile-num-steps 4 --profile-by-stage"
    fi

    echo "Running: ${CMD}"
    echo "Log: ${LOG_FILE}"

    eval ${CMD} 2>&1 | tee ${LOG_FILE}
done
