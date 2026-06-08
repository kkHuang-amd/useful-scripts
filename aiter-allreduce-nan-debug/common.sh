#!/usr/bin/env bash
# ============================================================
#  Shared config for 1p1d-no-umbp debug runs.
#
#  Source from each step script. Every variable here is
#  overridable from the caller's environment, so for one-off
#  changes you can do e.g.
#
#    PREFILL_NODE=smci355-ccs-aus-n08-21 bash step1.sh
#    CONTEXT_LENGTH=8192 bash step1.sh   # shrink to repro faster
# ============================================================

# --- Topology -----------------------------------------------------------------
PREFILL_NODE="${PREFILL_NODE:-smci355-ccs-aus-n08-29.prov.aus.ccs.cpe.ice.amd.com}"
DECODE_NODE="${DECODE_NODE:-smci355-ccs-aus-n08-33.prov.aus.ccs.cpe.ice.amd.com}"

# --- Workspace / repo ---------------------------------------------------------
WORKSPACE_DIR="${WORKSPACE_DIR:-/home/wunhuang/workspace}"
REPO_DIR="${REPO_DIR:-${WORKSPACE_DIR}/mori-scheduler}"
CONFIG_FILE="${CONFIG_FILE:-scripts/multi_node/configs/deepseek_r1_mxfp4.env}"

# --- Ports --------------------------------------------------------------------
PREFILL_PORT="${PREFILL_PORT:-30020}"
DECODE_PORT="${DECODE_PORT:-30030}"
ROUTER_PORT="${ROUTER_PORT:-8300}"

# --- Model --------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528-MXFP4-th}"
TOKENIZER="${TOKENIZER:-/apps/data/models/${MODEL_NAME}}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"

# --- Benchmark ----------------------------------------------------------------
INFERENCEX_DIR="${INFERENCEX_DIR:-/home/yanfwang/workspace/InferenceX}"
BENCH_CONTAINER="${BENCH_CONTAINER:-mori-bench-${USER:-wunhuang}}"
BENCH_CONC="${BENCH_CONC:-8}"
BENCH_DURATION="${BENCH_DURATION:-300}"

# --- Helpers ------------------------------------------------------------------
#
# node_ip <hostname>
#
# Resolve the canonical 10.235.x management IP of a node.
#   1. Try DNS (`getent hosts`) first; on these clusters it returns 10.235.x.
#   2. If DNS gives nothing useful (e.g. 127.0.1.1 when looking up self), fall
#      back to SSH'ing into the node and reading `hostname -I`.
#   3. Final fallback: scan local interfaces (for self lookup without ssh).
#
# You can short-circuit this by exporting PREFILL_IP / DECODE_IP directly.
node_ip() {
  local node=$1
  local ip

  ip="$(getent hosts "$node" 2>/dev/null | awk 'NR==1 {print $1}')"
  if [[ -n "$ip" && ! "$ip" =~ ^127\. ]]; then
    echo "$ip"
    return 0
  fi

  ip="$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$node" \
        "hostname -I | tr ' ' '\n' | grep '^10\\.235\\.' | head -1" 2>/dev/null)"
  if [[ -n "$ip" ]]; then
    echo "$ip"
    return 0
  fi

  # Self lookup without ssh.
  local short
  short="$(hostname -s 2>/dev/null)"
  if [[ "$node" == "$short"* ]]; then
    ip="$(hostname -I | tr ' ' '\n' | grep '^10\.235\.' | head -1)"
    if [[ -n "$ip" ]]; then
      echo "$ip"
      return 0
    fi
  fi

  echo "ERROR: cannot resolve IP for ${node}" >&2
  return 1
}
