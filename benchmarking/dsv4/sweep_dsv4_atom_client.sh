#!/usr/bin/env bash
# Sweep DeepSeek-V4 serving perf using ATOM's OWN benchmark client
# (atom.benchmarks.benchmark_serving), mirroring the ROCm/ATOM recipe:
#   https://github.com/ROCm/ATOM/blob/main/recipes/DeepSeek-V4.md
# with --num-prompts fixed to CONC*8 (recipe uses CONC*10) and an explicit
# CONC*2 warmup so it lines up with sweep_dsv4_sglang_client.sh.
#
# Prereq: an OpenAI-compatible server already running on $PORT (e.g. launched
# via run_atom_dsv4.sh). This script is client-only; it does NOT start a server.
#
# Env vars:
#   MODEL        served model path (default /dockerx/data/deepseek-ai/DeepSeek-V4-Pro/)
#   HOST         server host                         (default localhost)
#   PORT         server port                         (default 8000)
#   RESULT_DIR   where to write results              (default /workspace/bench_results_dsv4_atom)
#   RATIO        random-range-ratio                  (default 1.0 = fixed lengths)
#   WORKLOADS    space-separated "ISL:OSL" pairs     (default "8192:1024 1024:1024")
#   CONCS        space-separated concurrency list    (default "128 256")
#
# Example (only the two requested 8k/1k points):
#   WORKLOADS="8192:1024" CONCS="128 256" bash sweep_dsv4_atom_client.sh
set -euo pipefail

MODEL="${MODEL:-/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"
RESULT_DIR="${RESULT_DIR:-/workspace/bench_results_dsv4_atom}"
RATIO="${RATIO:-1.0}"
WORKLOADS="${WORKLOADS:-8192:1024 1024:1024}"
CONCS="${CONCS:-128 256}"

mkdir -p "$RESULT_DIR"
SUMMARY="${RESULT_DIR}/atomBench_sweep_summary.txt"
: > "$SUMMARY"

run_one() {
    local isl="$1" osl="$2" conc="$3"
    local name="atomBench_dsv4_isl${isl}_osl${osl}_c${conc}"
    local log="${RESULT_DIR}/${name}.log"
    local nprompts=$((conc * 8))
    local nwarm=$((conc * 2))
    echo "============================================================"
    echo "[atom-client] $name prompts=$nprompts warmups=$nwarm ratio=$RATIO"
    echo "============================================================"
    if python3 -m atom.benchmarks.benchmark_serving \
        --model="$MODEL" --backend=vllm --base-url="http://${HOST}:${PORT}" \
        --dataset-name=random \
        --random-input-len="$isl" --random-output-len="$osl" \
        --random-range-ratio="$RATIO" \
        --num-prompts="$nprompts" \
        --max-concurrency="$conc" \
        --num-warmups="$nwarm" \
        --request-rate=inf --ignore-eos \
        --save-result --result-dir="$RESULT_DIR" \
        --result-filename="${name}.json" \
        --percentile-metrics="ttft,tpot,itl,e2el" \
        2>&1 | tee "$log"; then
        echo "[atom-client] OK   $name" | tee -a "$SUMMARY"
    else
        echo "[atom-client] FAIL $name (see $log)" | tee -a "$SUMMARY"
    fi
}

for w in $WORKLOADS; do
    isl="${w%%:*}"; osl="${w##*:}"
    for c in $CONCS; do run_one "$isl" "$osl" "$c"; done
done

echo "============================================================"
echo "[atom-client] DONE results=$RESULT_DIR"
cat "$SUMMARY"
