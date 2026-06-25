#!/usr/bin/env bash
# Sweep DeepSeek-V4 serving perf using SGLang's bench_serving client (via the
# bench_dsv4.py wrapper, which adds the ATOM streaming-chunk shim). This mirrors
# the InferenceX-style invocation used for the gpt-oss sweeps so numbers are
# directly comparable across engines.
#
# Prereq: an OpenAI-compatible server already running on $PORT (e.g. launched
# via run_atom_dsv4.sh). Client-only; does NOT start a server.
#
# Against an SGLang server you do not need the wrapper and can set
#   BENCH="python3 -m sglang.bench_serving"
# (the shim is only needed for ATOM's usage-only final SSE chunk).
#
# Env vars:
#   MODEL        served model path (default /dockerx/data/deepseek-ai/DeepSeek-V4-Pro/)
#   HOST         server host                         (default 127.0.0.1)
#   PORT         server port                         (default 8000)
#   BACKEND      bench_serving --backend             (default sglang-oai)
#   RESULT_DIR   where to write results              (default /workspace/bench_results_dsv4_atom)
#   RATIO        random-range-ratio                  (default 1.0 = fixed lengths)
#   WORKLOADS    space-separated "ISL:OSL" pairs     (default "8192:1024 1024:1024")
#   CONCS        space-separated concurrency list    (default "128 256")
#
# Example (only the two requested 8k/1k points):
#   WORKLOADS="8192:1024" CONCS="128 256" bash sweep_dsv4_sglang_client.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-python3 ${SCRIPT_DIR}/bench_dsv4.py}"

MODEL="${MODEL:-/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BACKEND="${BACKEND:-sglang-oai}"
RESULT_DIR="${RESULT_DIR:-/workspace/bench_results_dsv4_atom}"
RATIO="${RATIO:-1.0}"
WORKLOADS="${WORKLOADS:-8192:1024 1024:1024}"
CONCS="${CONCS:-128 256}"
NP_MULT="${NP_MULT:-8}"      # num-prompts  = conc * NP_MULT
WARM_MULT="${WARM_MULT:-2}"  # warmups      = conc * WARM_MULT

mkdir -p "$RESULT_DIR"
SUMMARY="${RESULT_DIR}/sglangClient_sweep_summary.txt"
: > "$SUMMARY"

run_one() {
    local isl="$1" osl="$2" conc="$3"
    local name="sglangClient_dsv4_isl${isl}_osl${osl}_c${conc}"
    local log="${RESULT_DIR}/${name}.log"
    local nprompts=$((conc * NP_MULT))
    local nwarm=$((conc * WARM_MULT))
    echo "============================================================"
    echo "[sglang-client] $name prompts=$nprompts warmups=$nwarm ratio=$RATIO"
    echo "============================================================"
    if $BENCH \
        --backend "$BACKEND" --base-url "http://${HOST}:${PORT}" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$isl" --random-output-len "$osl" \
        --random-range-ratio "$RATIO" \
        --num-prompts "$nprompts" --max-concurrency "$conc" \
        --request-rate inf --warmup-requests "$nwarm" \
        --output-file "${RESULT_DIR}/${name}.jsonl" \
        2>&1 | tee "$log"; then
        echo "[sglang-client] OK   $name" | tee -a "$SUMMARY"
    else
        echo "[sglang-client] FAIL $name (see $log)" | tee -a "$SUMMARY"
    fi
}

for w in $WORKLOADS; do
    isl="${w%%:*}"; osl="${w##*:}"
    for c in $CONCS; do run_one "$isl" "$osl" "$c"; done
done

echo "============================================================"
echo "[sglang-client] DONE results=$RESULT_DIR"
cat "$SUMMARY"
