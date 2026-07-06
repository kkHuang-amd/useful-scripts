#!/usr/bin/env bash
# Drive the InferenceX agentic trace-replay (aiperf, inferencex-agentx-mvp
# scenario) against an already-running SGLang DeepSeek-V4-Pro server, reusing
# the upstream benchmark_lib.sh helpers verbatim so the replay command and
# aggregation match the InferenceX dashboard pipeline.
#
# Prereqs:
#   - server up on $PORT (served-model-name must equal $MODEL)
#   - aiperf venv at $AIPERF_VENV (built via install_agentic_deps / this session)
#   - dataset semianalysisai/cc-traces-weka-062126 pre-downloaded in HF cache
#
# All paths/config are overridable via env vars (defaults in parentheses):
#   INFMAX_CONTAINER_WORKSPACE  InferenceX repo root (/dockerx/home/wunhuang/InferenceX)
#   AIPERF_VENV                 isolated aiperf venv  (/tmp/inferencex-agentic-venv)
#   AGENTIC_DIR / AIPERF_DIR    derived from workspace unless set
#   MODEL                       served-model-name     (DeepSeek-V4-Pro)
#   MODEL_PATH                  local weights = tokenizer source
#                               (/dockerx/mnt/models/deepseek-ai/DeepSeek-V4-Pro)
#   MODEL_PREFIX                corpus family selector (dsv4)
#   KV_OFFLOADING               none | dram           (none)
#   CONC                        concurrency           (128)
#   DURATION                    benchmark seconds; <900 auto-adds --unsafe-override (900)
#   PORT                        server port           (8000)
#   RESULT_DIR                  output dir (<script_dir>/agentic_results/conc<CONC>_dur<DURATION>)
set -euo pipefail

export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-/dockerx/home/wunhuang/InferenceX}"
export AGENTIC_DIR="${AGENTIC_DIR:-$INFMAX_CONTAINER_WORKSPACE/utils/agentic-benchmark}"
export AIPERF_DIR="${AIPERF_DIR:-$INFMAX_CONTAINER_WORKSPACE/utils/aiperf}"

# Reuse the already-built isolated venv; skip the reinstall in install_agentic_deps.
export AIPERF_VENV="${AIPERF_VENV:-/tmp/inferencex-agentic-venv}"
export AIPERF_DEPS_READY="${AIPERF_DEPS_READY:-1}"

export PORT="${PORT:-8000}"
export CONC="${CONC:-128}"
export DURATION="${DURATION:-900}"

# DSv4 => full-context corpus loader (semianalysis_cc_traces_weka_062126).
export MODEL_PREFIX="${MODEL_PREFIX:-dsv4}"
export MODEL="${MODEL:-DeepSeek-V4-Pro}"                                   # must match served-model-name
export MODEL_PATH="${MODEL_PATH:-/dockerx/mnt/models/deepseek-ai/DeepSeek-V4-Pro}" # tokenizer source
export KV_OFFLOADING="${KV_OFFLOADING:-none}"

# Fail early with a clear message if the InferenceX harness / venv are missing.
if [ ! -f "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh" ]; then
    echo "ERROR: benchmark_lib.sh not found under INFMAX_CONTAINER_WORKSPACE=$INFMAX_CONTAINER_WORKSPACE" >&2
    echo "       Set INFMAX_CONTAINER_WORKSPACE to your InferenceX repo root." >&2
    exit 1
fi
if [ "$AIPERF_DEPS_READY" = "1" ] && [ ! -x "$AIPERF_VENV/bin/aiperf" ]; then
    echo "ERROR: AIPERF_DEPS_READY=1 but no aiperf at $AIPERF_VENV/bin/aiperf" >&2
    echo "       Point AIPERF_VENV at the built venv, or set AIPERF_DEPS_READY=0 to (re)install." >&2
    exit 1
fi

RESULT_DIR="${RESULT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/agentic_results/conc${CONC}_dur${DURATION}}"
mkdir -p "$RESULT_DIR"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
export RESULT_FILENAME="dsv4_agentic_conc${CONC}"

source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"

resolve_trace_source

build_replay_cmd "$RESULT_DIR"
# benchmark_lib does not set a tokenizer; our served-model-name is a short alias
# ("DeepSeek-V4-Pro") that is not a resolvable HF repo, so point aiperf's dataset
# tokenizer at the local weights directory.
REPLAY_CMD+=" --tokenizer $MODEL_PATH"
REPLAY_CMD+=" --server-metrics http://localhost:$PORT/metrics"

echo "=== REPLAY_CMD ==="
echo "$REPLAY_CMD"
echo "=================="

run_agentic_replay_and_write_outputs "$RESULT_DIR"
echo "=== DONE: results in $RESULT_DIR ==="
