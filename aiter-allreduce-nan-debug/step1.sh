#!/usr/bin/env bash
# ============================================================
#  Step 1 — Launch no-UMBP prefill SGLang on PREFILL_NODE.
#
#  Run this ON the prefill host (defaults to 08-29).
#    ssh smci355-ccs-aus-n08-29.prov.aus.ccs.cpe.ice.amd.com
#    bash /home/wunhuang/dbg-1p1d-gpu-fault/step1.sh
#
#  Logs:
#    docker exec mori-prefill-no-umbp-${USER} tail -f /tmp/prefill_no_umbp.log
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

echo "[step1] PREFILL_NODE=${PREFILL_NODE}  port=${PREFILL_PORT}  ctx=${CONTEXT_LENGTH}"

cd "${REPO_DIR}"
source "${CONFIG_FILE}"

WORKSPACE_DIR="${WORKSPACE_DIR}" \
PREFILL_ID=0 \
PORT="${PREFILL_PORT}" \
CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
bash scripts/multi_node/launch_prefill_no_umbp.sh
