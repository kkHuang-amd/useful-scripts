#!/usr/bin/env bash
# ============================================================
#  Step 2 — Launch no-UMBP decode SGLang on DECODE_NODE.
#
#  Run this ON the decode host (defaults to 08-33).
#    ssh smci355-ccs-aus-n08-33.prov.aus.ccs.cpe.ice.amd.com
#    bash /home/wunhuang/dbg-1p1d-gpu-fault/step2.sh
#
#  Logs:
#    docker exec mori-decode-no-umbp-${USER} tail -f /tmp/decode_no_umbp.log
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

echo "[step2] DECODE_NODE=${DECODE_NODE}  port=${DECODE_PORT}  ctx=${CONTEXT_LENGTH}"

cd "${REPO_DIR}"
source "${CONFIG_FILE}"

WORKSPACE_DIR="${WORKSPACE_DIR}" \
DECODE_ID=0 \
PORT="${DECODE_PORT}" \
CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
bash scripts/multi_node/launch_decode_no_umbp.sh
