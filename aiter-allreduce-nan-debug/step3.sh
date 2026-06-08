#!/usr/bin/env bash
# ============================================================
#  Step 3 — Launch no-UMBP mori-sched router on PREFILL_NODE.
#
#  Run this ON the prefill host (defaults to 08-29), AFTER
#  step1 + step2 are healthy.
#
#    ssh smci355-ccs-aus-n08-29.prov.aus.ccs.cpe.ice.amd.com
#    bash /home/wunhuang/dbg-1p1d-gpu-fault/step3.sh
#
#  IPs are resolved at run-time from PREFILL_NODE / DECODE_NODE
#  hostnames (see common.sh::node_ip). To bypass resolution
#  (e.g. transient DNS issue), export PREFILL_IP / DECODE_IP.
#
#  Logs:
#    docker exec mori-router-no-umbp-${USER} tail -f /tmp/router_no_umbp.log
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

prefill_ip="${PREFILL_IP:-$(node_ip "${PREFILL_NODE}")}"
decode_ip="${DECODE_IP:-$(node_ip "${DECODE_NODE}")}"

echo "[step3] PREFILL_NODE=${PREFILL_NODE} -> ${prefill_ip}:${PREFILL_PORT}"
echo "[step3] DECODE_NODE =${DECODE_NODE} -> ${decode_ip}:${DECODE_PORT}"
echo "[step3] ROUTER listen :${ROUTER_PORT}"

cd "${REPO_DIR}"
source "${CONFIG_FILE}"

WORKSPACE_DIR="${WORKSPACE_DIR}" \
PREFILL_URLS="http://${prefill_ip}:${PREFILL_PORT}" \
DECODE_URLS="http://${decode_ip}:${DECODE_PORT}" \
ROUTER_PORT="${ROUTER_PORT}" \
TOKENIZER_PATH="/models/${MODEL_NAME}" \
POLICY=round_robin \
PREFILL_POLICY=round_robin \
DECODE_POLICY=round_robin \
bash scripts/multi_node/launch_router_no_umbp.sh
