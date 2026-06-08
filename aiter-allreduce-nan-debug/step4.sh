#!/usr/bin/env bash
# ============================================================
#  Step 4 — Run AgentX v0.3 trace replay against the router.
#
#  Run this ON the bench host (e.g. 08-25), NOT inside the
#  bench container. The script resolves PREFILL_NODE IP, checks
#  router health, then `docker exec`s into ${BENCH_CONTAINER}
#  to run scripts/run_trace_replay_v0.3.sh with the right deps.
#
#  Prereq: bench container is already up. If not, on bench host:
#    WORKSPACE=/home/wunhuang/workspace \
#    INFERENCEX_DIR=/home/yanfwang/workspace/InferenceX \
#    bash /home/wunhuang/workspace/mori-scheduler/scripts/multi_node/launch_bench_container.sh --detach
#
#  Override examples:
#    BENCH_CONC=4 BENCH_DURATION=120 bash step4.sh   # quick smoke
#    PREFILL_IP=10.235.192.82 bash step4.sh          # skip DNS
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

prefill_ip="${PREFILL_IP:-$(node_ip "${PREFILL_NODE}")}"
ENDPOINT="http://${prefill_ip}:${ROUTER_PORT}"

echo "[step4] Endpoint:     ${ENDPOINT}"
echo "[step4] Bench cont.:  ${BENCH_CONTAINER}"
echo "[step4] Concurrency:  ${BENCH_CONC}"
echo "[step4] Duration:     ${BENCH_DURATION}s"
echo "[step4] Context len:  ${CONTEXT_LENGTH}"

if ! curl -sf -m 5 "${ENDPOINT}/health" >/dev/null 2>&1; then
  echo "ERROR: router not healthy at ${ENDPOINT}" >&2
  echo "       Check step3 ran successfully on ${PREFILL_NODE}." >&2
  exit 1
fi
echo "[step4] Router /health OK"

if ! docker inspect "${BENCH_CONTAINER}" >/dev/null 2>&1; then
  echo "ERROR: bench container '${BENCH_CONTAINER}' not found on $(hostname -s)." >&2
  echo "       Start it first with launch_bench_container.sh (see comment header)." >&2
  exit 1
fi

LOGDIR="${LOGDIR:-${REPO_DIR}/outputs/trace_replay}"
AIPERF_DATASET_MMAP_CACHE_DIR="${AIPERF_DATASET_MMAP_CACHE_DIR:-${REPO_DIR}/outputs/aiperf_mmap_cache}"
# The HuggingFace cache defaults to /root/.cache inside the container, which
# lives on the node's docker overlay (/data) and is frequently 100% full. Point
# it at the NFS workspace instead so the ~4.5 GiB trace dataset has room.
HF_HOME="${HF_HOME:-${WORKSPACE_DIR}/.hf_cache}"
mkdir -p "${LOGDIR}" "${AIPERF_DATASET_MMAP_CACHE_DIR}" "${HF_HOME}"
# The bench container runs as root, but these dirs live on NFS with
# root_squash, so the container's root is mapped to nobody and cannot write
# into our (uid-owned) dirs. run_trace_replay_v0.3.sh and the HF downloader
# create subdirs from inside the container, so make the parents sticky
# world-writable (1777) — same approach as run_1p1d_no_umbp_v03.sh.
chmod 1777 "${LOGDIR}" "${AIPERF_DATASET_MMAP_CACHE_DIR}" "${HF_HOME}" 2>/dev/null || true

# Use -t only when we actually have a TTY, so this works both interactively
# and from a backgrounded loop (e.g. repro_loop.sh) where stdin/stdout aren't TTYs.
DOCKER_TTY="-i"
if [ -t 0 ] && [ -t 1 ]; then DOCKER_TTY="-it"; fi

docker exec ${DOCKER_TTY} "${BENCH_CONTAINER}" bash -lc "
  cd '${REPO_DIR}' && \
  ENDPOINT='${ENDPOINT}' \
  MODEL='${MODEL_NAME}' \
  TOKENIZER='${TOKENIZER}' \
  MAX_CONTEXT_LENGTH='${CONTEXT_LENGTH}' \
  LOGDIR='${LOGDIR}' \
  AIPERF_DATASET_MMAP_CACHE_DIR='${AIPERF_DATASET_MMAP_CACHE_DIR}' \
  INFERENCEX_DIR='${INFERENCEX_DIR}' \
  HF_HOME='${HF_HOME}' \
  SKIP_DEP_INSTALL='${SKIP_DEP_INSTALL:-0}' \
  bash scripts/run_trace_replay_v0.3.sh ${BENCH_CONC} ${BENCH_DURATION}
"
