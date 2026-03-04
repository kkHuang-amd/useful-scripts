#!/usr/bin/env bash
set -euo pipefail

# ===================== User-adjustable params =====================
HOST="localhost"
PORT="8000"
MODEL="/dockerx/data/models/openai/gpt-oss-120b/"

DATASET="random"

input_tokens=8000
output_tokens=1000

random_range_ratio=1.0

# Where to save outputs
CSV_OUT="bench_results.csv"
PLOT_OUT="throughput_vs_median_e2e_latency.png"
# =================================================================
concurrencies=(1 2 4 8 16)

# Start fresh CSV
echo "concurrency,median_e2e_ms,total_token_throughput_tok_s" > "$CSV_OUT"
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

for c in "${concurrencies[@]}"; do
  max_concurrency="$c"
  num_prompts=$((max_concurrency * 8))

  echo "=== Running benchmark: concurrency=${c}, num_prompts=${num_prompts} ==="
  tmp_log="server-mode_benchmark_results_max_concurrency${max_concurrency}_${RUN_TIMESTAMP}.log"

  python3 -m sglang.bench_serving \
      --host "${HOST}" \
      --port "${PORT}" \
      --model "${MODEL}" \
      --dataset-name "${DATASET}" \
      --random-input "${input_tokens}" \
      --random-output "${output_tokens}" \
      --random-range-ratio "${random_range_ratio}" \
      --max-concurrency "${max_concurrency}" \
      --port ${PORT} \
      --num-prompt "${num_prompts}" 2>&1 | tee "${tmp_log}"
done
