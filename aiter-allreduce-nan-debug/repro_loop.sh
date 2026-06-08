#!/usr/bin/env bash
# ============================================================
#  Re-run step4 (benchmark) in a loop until the decode (or
#  prefill) GPU process dies, to reproduce the intermittent
#  HSA_STATUS_ERROR_EXCEPTION GPU fault.
#
#  Run this ON the bench/prefill host (08-21), where the bench
#  container lives. decode + router must already be up.
#
#  Stops as soon as a worker stops answering /health (i.e. the
#  crash we want), leaving the core file in /var/tmp/cores/ on
#  the node that faulted.
#
#  Override:
#    REPRO_RUNS       number of iterations (default 20)
#    BENCH_CONC       concurrency per run (default 8 — known crashing config)
#    BENCH_DURATION   seconds per run (default 300)
#    PREFILL_IP/DECODE_IP  pin IPs (recommended)
# ============================================================
set -uo pipefail   # intentionally NOT -e: we handle failures ourselves
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPRO_RUNS="${REPRO_RUNS:-20}"
export BENCH_CONC="${BENCH_CONC:-8}"
export BENCH_DURATION="${BENCH_DURATION:-300}"

prefill_ip="${PREFILL_IP:-$(node_ip "${PREFILL_NODE}")}"
decode_ip="${DECODE_IP:-$(node_ip "${DECODE_NODE}")}"
export PREFILL_IP="$prefill_ip"
export DECODE_IP="$decode_ip"

echo "============================================================"
echo "  repro_loop start $(date -Is)"
echo "  prefill=${PREFILL_NODE} (${prefill_ip}:${PREFILL_PORT})"
echo "  decode =${DECODE_NODE} (${decode_ip}:${DECODE_PORT})"
echo "  router :${ROUTER_PORT}  conc=${BENCH_CONC} dur=${BENCH_DURATION} runs=${REPRO_RUNS}"
echo "============================================================"

health() {  # host port -> echoes http code or 000
  curl -sf -m 5 -o /dev/null -w '%{http_code}' "http://$1:$2/health" 2>/dev/null || echo "000"
}

worker_dead() {
  local p d
  p="$(health "$prefill_ip" "$PREFILL_PORT")"
  d="$(health "$decode_ip" "$DECODE_PORT")"
  echo "    [health] prefill=$p decode=$d"
  [[ "$p" != "200" || "$d" != "200" ]]
}

# Confirm workers healthy before we start
if worker_dead; then
  echo "ERROR: a worker is already unhealthy before starting. Fix step1/step2/step3 first." >&2
  exit 1
fi

# Dependency install is slow (~7-10 min: it git-clones transformers). Only do it
# on the first run; subsequent runs reuse the already-installed deps in the bench
# container. Caller may set SKIP_DEP_INSTALL=1 to skip even the first run.
RUN1_SKIP="${SKIP_DEP_INSTALL:-0}"

for ((i=1; i<=REPRO_RUNS; i++)); do
  echo
  echo "################ RUN ${i}/${REPRO_RUNS}  $(date -Is) ################"
  if [[ $i -eq 1 ]]; then
    export SKIP_DEP_INSTALL="$RUN1_SKIP"
  else
    export SKIP_DEP_INSTALL=1
  fi
  echo "    [deps] SKIP_DEP_INSTALL=${SKIP_DEP_INSTALL}"
  bash "${SCRIPT_DIR}/step4.sh"
  rc=$?
  echo "    [step4 exit] rc=${rc}"

  # Authoritative crash signal: did a GPU worker stop answering?
  if worker_dead; then
    echo
    echo "############################################################"
    echo "  CRASH DETECTED on run ${i}  $(date -Is)"
    echo "  A worker is no longer healthy — likely the GPU fault."
    echo "  Check cores:  /var/tmp/cores/  on prefill(${PREFILL_NODE}) / decode(${DECODE_NODE})"
    echo "############################################################"
    exit 7
  fi

  echo "    [ok] both workers still healthy after run ${i}"
done

echo
echo "repro_loop finished ${REPRO_RUNS} runs without a worker crash $(date -Is)"
exit 0
