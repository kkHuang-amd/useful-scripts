#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-/dockerx/data/Kimi-K3}"
ISL="${ISL:-68000}"
OSL="${OSL:-350}"
CONCS="${CONCS:-1 2 4 8}"
NP_MULT="${NP_MULT:-4}"
WARM_MULT="${WARM_MULT:-1}"
RATIO="${RATIO:-1.0}"
TAG="${TAG:-run}"
RESULT_DIR="${RESULT_DIR:-$(pwd)/results/${TAG}}"

mkdir -p "$RESULT_DIR"
summary="$RESULT_DIR/summary.txt"
: >"$summary"

for conc in $CONCS; do
  num_prompts=$((conc * NP_MULT))
  warmups=$((conc * WARM_MULT))
  name="${TAG}_isl${ISL}_osl${OSL}_c${conc}"
  output="$RESULT_DIR/${name}.jsonl"
  log="$RESULT_DIR/${name}.log"

  printf 'run=%s prompts=%d warmups=%d\n' "$name" "$num_prompts" "$warmups" |
    tee -a "$summary"

  if python3 -m sglang.bench_serving \
    --backend sglang \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio "$RATIO" \
    --num-prompts "$num_prompts" \
    --warmup-requests "$warmups" \
    --max-concurrency "$conc" \
    --request-rate inf \
    --output-file "$output" >"$log" 2>&1; then
    printf 'PASS %s\n' "$name" | tee -a "$summary"
    rg "Successful requests|Benchmark duration|Request throughput|Input token throughput|Output token throughput|Total token throughput|Mean TTFT|Mean TPOT|Median E2E" "$log" |
      tee -a "$summary" || true
  else
    printf 'FAIL %s log=%s\n' "$name" "$log" | tee -a "$summary"
    rg "Traceback|Error|error|failed|OOM|OutOfMemory" "$log" |
      tee -a "$summary" || true
    exit 1
  fi
done

printf 'results=%s\n' "$RESULT_DIR" | tee -a "$summary"
