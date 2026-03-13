#!/usr/bin/env bash
set -euo pipefail

# ===================== User-adjustable params =====================
HOST="localhost"
PORT="8000"
MODEL="openai/gpt-oss-120b"
#MODEL="/dockerx/data/models/openai/gpt-oss-120b/"

result_dir="."

DATASET="random"

input_tokens=8192
output_tokens=1024
#output_tokens=10

random_range_ratio=0.8

# Where to save outputs
CSV_OUT="bench_results.csv"
PLOT_OUT="throughput_vs_median_e2e_latency.png"
# =================================================================
concurrencies=(4 8 16)

# Start fresh CSV
echo "concurrency,median_e2e_ms,total_token_throughput_tok_s" > "$CSV_OUT"
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

for c in "${concurrencies[@]}"; do
  max_concurrency="$c"
  num_prompts=$((max_concurrency * 10))

  echo "=== Running benchmark: concurrency=${c}, num_prompts=${num_prompts} ==="
  tmp_log="server-mode_benchmark_results_max_concurrency${max_concurrency}_${RUN_TIMESTAMP}.log"

  python3 benchmark_serving.py \
        --model "${MODEL}" \
        --backend "vllm" \
        --base-url "http://0.0.0.0:${PORT}" \
        --dataset-name random \
        --random-input-len "${input_tokens}" \
        --random-output-len "${output_tokens}" \
        --random-range-ratio "${random_range_ratio}" \
        --num-prompts "$num_prompts" \
        --max-concurrency "$max_concurrency" \
        --request-rate inf \
        --ignore-eos \
        --save-result \
        --num-warmups "$((2 * max_concurrency))" \
        --percentile-metrics 'ttft,tpot,itl,e2el' \
        --result-dir "$result_dir" \
        --result-filename "${tmp_log}.json"

#  python3 -m sglang.bench_serving \
#      --host "${HOST}" \
#      --port "${PORT}" \
#      --model "${MODEL}" \
#      --dataset-name "${DATASET}" \
#      --random-input "${input_tokens}" \
#      --random-output "${output_tokens}" \
#      --random-range-ratio "${random_range_ratio}" \
#      --max-concurrency "${max_concurrency}" \
#      --port ${PORT} \
#      --profile \
#      --num-prompt "${num_prompts}" 2>&1 | tee "${tmp_log}"
done
