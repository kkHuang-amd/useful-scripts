#!/usr/bin/env bash
set -euo pipefail

# Benchmark an already-running SGLang endpoint. Keep server configuration and
# variant orchestration outside this reusable client matrix.

BASE_URL="${BASE_URL:-http://127.0.0.1:30000}"
MODEL="${MODEL:-/shared_nfs/models/Kimi-K3}"
TOKENIZER="${TOKENIZER:-$MODEL}"
RESULT_DIR="${RESULT_DIR:?set RESULT_DIR}"
CONCURRENCIES="${CONCURRENCIES:-2 4 8 16}"
WORKLOADS="${WORKLOADS:-8k:8192:1024:8:64 68k:68000:350:3:1}"
SEED="${SEED:-42}"
BENCH_MODULE="${BENCH_MODULE:-sglang.bench_serving}"

mkdir -p "$RESULT_DIR"
printf 'workload\tconcurrency\tsuccessful\tthroughput\tttft_ms\ttpot_ms\n' \
  >"$RESULT_DIR/summary.tsv"

for workload_spec in $WORKLOADS; do
  IFS=: read -r workload input_len output_len waves warmups <<<"$workload_spec"
  for concurrency in $CONCURRENCIES; do
    prompts=$((concurrency * waves))
    prefix="$RESULT_DIR/${workload}-c${concurrency}"
    python -m "$BENCH_MODULE" \
      --backend sglang-oai \
      --base-url "$BASE_URL" \
      --dataset-name random \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --num-prompts "$prompts" \
      --random-input-len "$input_len" \
      --random-output-len "$output_len" \
      --random-range-ratio 1.0 \
      --max-concurrency "$concurrency" \
      --output-file "$prefix.jsonl" \
      --seed "$SEED" \
      --warmup-requests "$warmups" \
      >"$prefix.log" 2>&1

    successful=$(awk '/Successful requests:/{print $NF}' "$prefix.log")
    throughput=$(awk '/Total token throughput/{print $NF}' "$prefix.log")
    ttft=$(awk '/Median TTFT/{print $NF}' "$prefix.log")
    tpot=$(awk '/Median TPOT/{print $NF}' "$prefix.log")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$workload" "$concurrency" "$successful" "$throughput" "$ttft" "$tpot" \
      | tee -a "$RESULT_DIR/summary.tsv"
  done
done
