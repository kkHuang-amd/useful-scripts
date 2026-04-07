#!/bin/bash
# GPT-OSS-120B 1P1D EP+TP Sweep Benchmark
# Prefill-heavy workload: 10:1 input-to-output ratio
# Run inside the container on gpu-22 (where the router is at port 30000)
#
# EP-only changes vs TP-only sweep:
#   CONCURRENCIES extended to 256 (= max-running-requests for EP, vs 128 for TP-only)
#   RANDOM_RANGE_RATIO=1 added (from golden repo set_env_vars.sh / bench.sh)
#   OUTPUT_DIR defaults to /models/sweep_results_ep
#
# Usage:
#   bash /sgl-workspace/scripts/sweep_benchmark_tp_ep.sh
#   BASE_URL=http://localhost:30000 OUTPUT_DIR=/models/sweep_results_ep bash /sgl-workspace/scripts/sweep_benchmark_tp_ep.sh

BASE_URL="${BASE_URL:-http://localhost:30000}"
MODEL="${MODEL:-/dockerx/data/models/gpt-oss-120b}"
NUM_PROMPTS_MULTIPLIER="${NUM_PROMPTS_MULTIPLIER:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-./sweep_results_ep}"
# From golden repo (BENCH_RANDOM_RANGE_RATIO=1 in sglang_disagg_server.sh defaults)
# Controls how much input/output lengths vary around the chosen ISL/OSL
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1}"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${OUTPUT_DIR}/sweep_summary_${TIMESTAMP}.txt"
CSV_FILE="${OUTPUT_DIR}/sweep_data_${TIMESTAMP}.csv"

INPUT_LENS=(512 1024 2048 4096 8192 8192)
OUTPUT_LENS=(50  100  200  400  800  1024)
# EP supports max-running-requests=256 (vs 128 for TP-only); extend concurrencies accordingly
CONCURRENCIES=(1 8 16 32 64 256)

echo "================================================================" | tee "$SUMMARY_FILE"
echo "  GPT-OSS-120B 1P1D EP+TP Sweep Benchmark" | tee -a "$SUMMARY_FILE"
echo "  Date: $(date)" | tee -a "$SUMMARY_FILE"
echo "  Ratio: 10:1 (prefill-heavy)" | tee -a "$SUMMARY_FILE"
echo "  Prompts: concurrency x ${NUM_PROMPTS_MULTIPLIER}" | tee -a "$SUMMARY_FILE"
echo "  Endpoint: $BASE_URL" | tee -a "$SUMMARY_FILE"
echo "  Input lengths:     ${INPUT_LENS[*]}" | tee -a "$SUMMARY_FILE"
echo "  Output lengths:    ${OUTPUT_LENS[*]}" | tee -a "$SUMMARY_FILE"
echo "  Concurrencies:     ${CONCURRENCIES[*]}" | tee -a "$SUMMARY_FILE"
echo "  Random range ratio: $RANDOM_RANGE_RATIO" | tee -a "$SUMMARY_FILE"
echo "  Warmup requests:   10" | tee -a "$SUMMARY_FILE"
echo "  Prefill mem-fraction-static: 0.85" | tee -a "$SUMMARY_FILE"
echo "  Decode  mem-fraction-static: 0.85" | tee -a "$SUMMARY_FILE"
echo "  KV-cache dtype: fp8_e4m3" | tee -a "$SUMMARY_FILE"
echo "  EP size: 8  TP size: 8" | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# CSV header
echo "isl,osl,concurrency,num_prompts,req_per_s,input_tok_s,output_tok_s,total_tok_s,mean_e2e_ms,median_e2e_ms,p99_e2e_ms,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_itl_ms,median_itl_ms,p99_itl_ms,duration_s,success_count" > "$CSV_FILE"

printf "%-8s %-8s %-6s %-10s %-10s %-10s %-12s %-10s %-10s %-10s\n" \
    "ISL" "OSL" "Conc" "Req/s" "InTok/s" "OutTok/s" "TotalTok/s" "TTFT_med" "TPOT_med" "E2E_med" | tee -a "$SUMMARY_FILE"
printf "%s\n" "$(printf '%.0s-' {1..100})" | tee -a "$SUMMARY_FILE"

TOTAL_TESTS=$(( ${#INPUT_LENS[@]} * ${#CONCURRENCIES[@]} ))
TEST_NUM=0

WARMUP_PROMPTS=10

for i in "${!INPUT_LENS[@]}"; do
    ISL="${INPUT_LENS[$i]}"
    OSL="${OUTPUT_LENS[$i]}"

    # Warmup run for this ISL/OSL pair
    WARMUP_LOG="${OUTPUT_DIR}/warmup_isl${ISL}_osl${OSL}_${TIMESTAMP}.log"
    WARMUP_JSON="${OUTPUT_DIR}/warmup_isl${ISL}_osl${OSL}_${TIMESTAMP}.json"
    echo "" | tee -a "$SUMMARY_FILE"
    echo "[Warmup] ISL=${ISL} OSL=${OSL} NumPrompts=${WARMUP_PROMPTS} ..." | tee -a "$SUMMARY_FILE"

    python3 -m sglang.bench_serving \
        --backend sglang \
        --base-url "$BASE_URL" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$ISL" \
        --random-output-len "$OSL" \
        --random-range-ratio "$RANDOM_RANGE_RATIO" \
        --num-prompts "$WARMUP_PROMPTS" \
        --max-concurrency 1 \
        --output-file "$WARMUP_JSON" \
        --warmup-requests 10 \
        --disable-tqdm \
        2>&1 | tee "$WARMUP_LOG" > /dev/null

    echo "[Warmup] ISL=${ISL} OSL=${OSL} done." | tee -a "$SUMMARY_FILE"

    for CONC in "${CONCURRENCIES[@]}"; do
        TEST_NUM=$((TEST_NUM + 1))
        NUM_PROMPTS=$((CONC * NUM_PROMPTS_MULTIPLIER))
        RESULT_FILE="${OUTPUT_DIR}/bench_isl${ISL}_osl${OSL}_c${CONC}_${TIMESTAMP}.json"
        LOG_FILE="${OUTPUT_DIR}/bench_isl${ISL}_osl${OSL}_c${CONC}_${TIMESTAMP}.log"

        echo "" | tee -a "$SUMMARY_FILE"
        echo "[${TEST_NUM}/${TOTAL_TESTS}] ISL=${ISL} OSL=${OSL} Concurrency=${CONC} NumPrompts=${NUM_PROMPTS} ..." | tee -a "$SUMMARY_FILE"

        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url "$BASE_URL" \
            --model "$MODEL" \
            --dataset-name random \
            --random-input-len "$ISL" \
            --random-output-len "$OSL" \
            --random-range-ratio "$RANDOM_RANGE_RATIO" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONC" \
            --output-file "$RESULT_FILE" \
            --warmup-requests 10 \
            --disable-tqdm \
            2>&1 | tee "$LOG_FILE"

        if [ $? -eq 0 ] && [ -f "$RESULT_FILE" ]; then
            # Extract all metrics from JSON
            python3 -c "
import json, sys
d = json.load(open('$RESULT_FILE'))
req_s     = d.get('request_throughput', 0)
in_tok    = d.get('input_throughput', 0)
out_tok   = d.get('output_throughput', 0)
total_tok = d.get('total_throughput', in_tok + out_tok)
me2e      = d.get('mean_e2e_latency_ms', 0)
mde2e     = d.get('median_e2e_latency_ms', 0)
p99e2e    = d.get('p99_e2e_latency_ms', 0)
mttft     = d.get('mean_ttft_ms', 0)
mdttft    = d.get('median_ttft_ms', 0)
p99ttft   = d.get('p99_ttft_ms', 0)
mtpot     = d.get('mean_tpot_ms', 0)
mdtpot    = d.get('median_tpot_ms', 0)
p99tpot   = d.get('p99_tpot_ms', 0)
mitl      = d.get('mean_itl_ms', 0)
mditl     = d.get('median_itl_ms', 0)
p99itl    = d.get('p99_itl_ms', 0)
dur       = d.get('duration', 0)
succ      = d.get('completed', d.get('successful_requests', 0))
# CSV row
print(f'${ISL},${OSL},${CONC},${NUM_PROMPTS},{req_s:.2f},{in_tok:.0f},{out_tok:.0f},{total_tok:.0f},{me2e:.1f},{mde2e:.1f},{p99e2e:.1f},{mttft:.1f},{mdttft:.1f},{p99ttft:.1f},{mtpot:.2f},{mdtpot:.2f},{p99tpot:.2f},{mitl:.2f},{mditl:.2f},{p99itl:.2f},{dur:.1f},{succ}')
" >> "$CSV_FILE" 2>/dev/null

            # Pretty print summary row
            python3 -c "
import json
d = json.load(open('$RESULT_FILE'))
req_s     = d.get('request_throughput', 0)
in_tok    = d.get('input_throughput', 0)
out_tok   = d.get('output_throughput', 0)
total_tok = d.get('total_throughput', in_tok + out_tok)
mdttft    = d.get('median_ttft_ms', 0)
mdtpot    = d.get('median_tpot_ms', 0)
mde2e     = d.get('median_e2e_latency_ms', 0)
print(f'$ISL      $OSL      $CONC    {req_s:<10.2f}{in_tok:<10.0f}{out_tok:<10.0f}{total_tok:<12.0f}{mdttft:<10.1f}{mdtpot:<10.2f}{mde2e:<10.1f}')
" | tee -a "$SUMMARY_FILE" 2>/dev/null
        else
            printf "%-8s %-8s %-6s FAILED\n" "$ISL" "$OSL" "$CONC" | tee -a "$SUMMARY_FILE"
            echo "${ISL},${OSL},${CONC},${NUM_PROMPTS},0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0" >> "$CSV_FILE"
        fi
    done
done

echo "" | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"
echo "  Sweep complete at $(date)" | tee -a "$SUMMARY_FILE"
echo "  Results dir: $OUTPUT_DIR" | tee -a "$SUMMARY_FILE"
echo "  Summary:     $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "  CSV data:    $CSV_FILE" | tee -a "$SUMMARY_FILE"
echo "================================================================" | tee -a "$SUMMARY_FILE"

#echo ""
#echo "Generating visualization charts..."
# Interactive HTML charts (Plotly) — hover, zoom, pan
#python3 /sgl-workspace/scripts/plot_sweep_interactive.py "$CSV_FILE" "${OUTPUT_DIR}/charts_${TIMESTAMP}"
# Static PNG charts (matplotlib) — uncomment below if preferred
# python3 /sgl-workspace/scripts/plot_sweep.py "$CSV_FILE" "${OUTPUT_DIR}/charts_${TIMESTAMP}"
#echo "Done! Charts saved to ${OUTPUT_DIR}/charts_${TIMESTAMP}/"
