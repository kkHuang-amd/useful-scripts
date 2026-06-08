#!/usr/bin/env bash
# ============================================================
#  dbg_helpers.sh — one-liner bash functions for debugging
#  a 1p1d no-UMBP run (especially GPU core dumps).
#
#  Usage:
#    source /home/wunhuang/dbg-1p1d-gpu-fault/dbg_helpers.sh
#    dbg_help        # list everything
#    dbg_status      # snapshot of all containers, health, gpu
#    dbg_logs prefill
#    dbg_shell decode
#    dbg_cleanup --yes
#
#  Convention: role is one of {prefill, decode, router, bench}.
#    prefill / router live on PREFILL_NODE
#    decode            lives on DECODE_NODE
#    bench             lives on the current host (where you sourced this)
# ============================================================

# Pull in PREFILL_NODE / DECODE_NODE / ports / node_ip(), etc.
_DBG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${_DBG_DIR}/common.sh"

# ----- internal helpers -----------------------------------------------------

_dbg_user() { echo "${USER:-wunhuang}"; }

_dbg_container() {
  case "$1" in
    prefill) echo "mori-prefill-no-umbp-$(_dbg_user)" ;;
    decode)  echo "mori-decode-no-umbp-$(_dbg_user)" ;;
    router)  echo "mori-router-no-umbp-$(_dbg_user)" ;;
    bench)   echo "${BENCH_CONTAINER}" ;;
    *) echo "ERROR: unknown role '$1' (use prefill|decode|router|bench)" >&2; return 1 ;;
  esac
}

_dbg_logfile() {
  case "$1" in
    prefill) echo "/tmp/prefill_no_umbp.log" ;;
    decode)  echo "/tmp/decode_no_umbp.log" ;;
    router)  echo "/tmp/router_no_umbp.log" ;;
    *) echo "ERROR: no logfile for role '$1'" >&2; return 1 ;;
  esac
}

_dbg_node_for_role() {
  case "$1" in
    prefill|router) echo "${PREFILL_NODE}" ;;
    decode)         echo "${DECODE_NODE}" ;;
    bench)          echo "" ;;   # local
    *) echo "ERROR: unknown role '$1'" >&2; return 1 ;;
  esac
}

# Run a command on a target node. Empty node => run locally.
# Auto-detects if target is already the current host.
_dbg_on() {
  local node=$1; shift
  if [[ -z "$node" ]]; then
    bash -c "$*"
    return $?
  fi
  local short
  short="$(hostname -s 2>/dev/null)"
  if [[ -n "$short" && "$node" == "$short"* ]]; then
    bash -c "$*"
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" "$*"
  fi
}

# Same as _dbg_on but allocates TTY (for interactive `docker exec -it ...`).
_dbg_on_tty() {
  local node=$1; shift
  if [[ -z "$node" ]]; then
    bash -c "$*"
    return $?
  fi
  local short
  short="$(hostname -s 2>/dev/null)"
  if [[ -n "$short" && "$node" == "$short"* ]]; then
    bash -c "$*"
  else
    ssh -t -o BatchMode=yes -o ConnectTimeout=10 "$node" "$*"
  fi
}

# ----- top-level debug commands --------------------------------------------

dbg_help() {
  cat <<'EOF'
dbg_helpers — 1p1d no-UMBP debug toolbox

Quick status:
  dbg_status                snapshot: containers + health + GPU
  dbg_health                curl /health on prefill, decode, router
  dbg_test                  send tiny chat request through the router

Logs:
  dbg_logs <role> [N]       tail last N lines from container log
  dbg_follow <role>         tail -f (Ctrl-C to leave)
  dbg_bench_log [N]         tail latest benchmark.log

Process / container state:
  dbg_proc <role>           ps for sglang processes inside container
  dbg_inspect <role>        docker inspect summary (status, exit, OOM)
  dbg_shell <role>          interactive bash inside the container

GPU & kernel:
  dbg_gpu [node|both]       rocm-smi util/mem/pids
  dbg_dmesg [node] [N]      tail kernel log (looks for amdgpu / gpu errors)
  dbg_amdgpu [node]         grep amdgpu/kfd/mce in dmesg
  dbg_cores [node]          core_pattern + list cores under ${COREDUMP_DIR:-/home/wunhuang/workspace/cores}
  dbg_gdb_latest <role>     open the newest core in gdb (inside container)

RDMA / network:
  dbg_rdma [node|both]      ibstat summary (port state, rate)
  dbg_ports [node]          listening ports we care about

Cleanup:
  dbg_cleanup [--yes]       rm -f all four mori-*-no-umbp-$USER containers

Role names: prefill | decode | router | bench
Most commands accept a node arg too; default uses common.sh values:
  PREFILL_NODE=${PREFILL_NODE}
  DECODE_NODE =${DECODE_NODE}
EOF
}

dbg_status() {
  echo "=== Containers ==="
  for node in "${PREFILL_NODE}" "${DECODE_NODE}"; do
    echo "--- $node ---"
    _dbg_on "$node" "docker ps -a --filter name=mori-.*-no-umbp- --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}'" 2>&1
  done
  echo "--- local ($(hostname -s)) bench ---"
  docker ps -a --filter "name=${BENCH_CONTAINER}" --format 'table {{.Names}}\t{{.Status}}' 2>&1
  echo
  echo "=== Health ==="
  dbg_health
  echo
  echo "=== GPU ==="
  for node in "${PREFILL_NODE}" "${DECODE_NODE}"; do
    echo "--- $node ---"
    _dbg_on "$node" "rocm-smi --showuse 2>/dev/null | grep -E 'GPU\[' | head -8" 2>&1
  done
}

dbg_health() {
  local prefill_ip decode_ip
  prefill_ip="${PREFILL_IP:-$(node_ip "${PREFILL_NODE}" 2>/dev/null)}"
  decode_ip="${DECODE_IP:-$(node_ip "${DECODE_NODE}" 2>/dev/null)}"
  printf "  prefill (%s:%s)  -> " "$prefill_ip" "$PREFILL_PORT"
  curl -sf -m 3 -o /dev/null -w '%{http_code}\n' "http://${prefill_ip}:${PREFILL_PORT}/health" 2>&1 || echo "FAIL"
  printf "  decode  (%s:%s)  -> " "$decode_ip" "$DECODE_PORT"
  curl -sf -m 3 -o /dev/null -w '%{http_code}\n' "http://${decode_ip}:${DECODE_PORT}/health" 2>&1 || echo "FAIL"
  printf "  router  (%s:%s)   -> " "$prefill_ip" "$ROUTER_PORT"
  curl -sf -m 3 -o /dev/null -w '%{http_code}\n' "http://${prefill_ip}:${ROUTER_PORT}/health" 2>&1 || echo "FAIL"
}

dbg_test() {
  local prefill_ip
  prefill_ip="${PREFILL_IP:-$(node_ip "${PREFILL_NODE}")}"
  local endpoint="http://${prefill_ip}:${ROUTER_PORT}/v1/chat/completions"
  echo "POST ${endpoint}"
  curl -sS -X POST "$endpoint" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"say hi in 5 words\"}],\"max_tokens\":32,\"stream\":false}" \
    | { python3 -m json.tool 2>/dev/null || cat; } | head -40
}

dbg_logs() {
  local role=${1:?usage: dbg_logs <prefill|decode|router> [lines]}
  local lines=${2:-200}
  local node container logfile
  node="$(_dbg_node_for_role "$role")" || return 1
  container="$(_dbg_container "$role")" || return 1
  logfile="$(_dbg_logfile "$role")" || return 1
  _dbg_on "$node" "docker exec $container tail -n $lines $logfile 2>&1"
}

dbg_follow() {
  local role=${1:?usage: dbg_follow <prefill|decode|router>}
  local node container logfile
  node="$(_dbg_node_for_role "$role")" || return 1
  container="$(_dbg_container "$role")" || return 1
  logfile="$(_dbg_logfile "$role")" || return 1
  _dbg_on_tty "$node" "docker exec $container tail -F $logfile"
}

dbg_bench_log() {
  local lines=${1:-200}
  local latest
  latest="$(ls -t "${REPO_DIR}"/outputs/trace_replay/*/benchmark.log 2>/dev/null | head -1)"
  if [[ -z "$latest" ]]; then
    echo "No benchmark.log found under ${REPO_DIR}/outputs/trace_replay/" >&2
    return 1
  fi
  echo "=== $latest ==="
  tail -n "$lines" "$latest"
}

dbg_proc() {
  local role=${1:?usage: dbg_proc <prefill|decode|router>}
  local node container
  node="$(_dbg_node_for_role "$role")" || return 1
  container="$(_dbg_container "$role")" || return 1
  _dbg_on "$node" "
    if ! docker inspect $container >/dev/null 2>&1; then
      echo 'container $container does not exist'
      exit 1
    fi
    docker exec $container bash -c 'ps -eo pid,ppid,etime,rss,stat,cmd | grep -E sglang.launch_server | grep -v grep'
  "
}

dbg_inspect() {
  local role=${1:?usage: dbg_inspect <prefill|decode|router>}
  local node container
  node="$(_dbg_node_for_role "$role")" || return 1
  container="$(_dbg_container "$role")" || return 1
  _dbg_on "$node" "
    docker inspect $container --format '
status:       {{.State.Status}}
running:      {{.State.Running}}
exit_code:    {{.State.ExitCode}}
oom_killed:   {{.State.OOMKilled}}
started_at:   {{.State.StartedAt}}
finished_at:  {{.State.FinishedAt}}
pid:          {{.State.Pid}}
error:        {{.State.Error}}
' 2>&1
  "
}

dbg_shell() {
  local role=${1:?usage: dbg_shell <prefill|decode|router|bench>}
  local node container
  node="$(_dbg_node_for_role "$role")" || return 1
  container="$(_dbg_container "$role")" || return 1
  if [[ -z "$node" ]]; then
    docker exec -it "$container" bash
  else
    ssh -t -o BatchMode=yes "$node" "docker exec -it $container bash"
  fi
}

dbg_gpu() {
  local target=${1:-both}
  local cmd="rocm-smi --showuse --showmemuse --showpids 2>&1 | head -50"
  if [[ "$target" == "both" ]]; then
    for n in "${PREFILL_NODE}" "${DECODE_NODE}"; do
      echo "--- $n ---"
      _dbg_on "$n" "$cmd"
    done
  elif [[ "$target" == "prefill" ]]; then
    _dbg_on "${PREFILL_NODE}" "$cmd"
  elif [[ "$target" == "decode" ]]; then
    _dbg_on "${DECODE_NODE}" "$cmd"
  else
    _dbg_on "$target" "$cmd"
  fi
}

dbg_dmesg() {
  local node=${1:-${PREFILL_NODE}}
  local lines=${2:-100}
  # dmesg without sudo works on most cluster nodes (kernel.dmesg_restrict=0).
  _dbg_on "$node" "dmesg -T 2>/dev/null | tail -n $lines || sudo -n dmesg -T 2>/dev/null | tail -n $lines"
}

dbg_amdgpu() {
  local node=${1:-${PREFILL_NODE}}
  _dbg_on "$node" "
    echo '=== amdgpu / kfd / mce / gpu fault from dmesg -T ==='
    dmesg -T 2>/dev/null | grep -E -i 'amdgpu|kfd|gpu.?(fault|hang|reset)|mce|out of memory|oom-kill' | tail -60
    if command -v rocm-smi >/dev/null 2>&1; then
      echo
      echo '=== rocm-smi --showxgmierr ==='
      rocm-smi --showxgmierr 2>&1 | head -40
      echo
      echo '=== rocm-smi --showserial --showhw ==='
      rocm-smi --showserial 2>&1 | head -20
    fi
  "
}

dbg_cores() {
  local node=${1:-${PREFILL_NODE}}
  local coredump_dir="${COREDUMP_DIR:-/var/tmp/cores}"
  _dbg_on "$node" "
    echo '=== /proc/sys/kernel/core_pattern ==='
    cat /proc/sys/kernel/core_pattern 2>&1
    echo '=== /proc/sys/fs/suid_dumpable ==='
    cat /proc/sys/fs/suid_dumpable 2>&1
    echo '=== ulimit -c (in host shell) ==='
    ulimit -c
    echo '=== ${coredump_dir} on \$(hostname -s) ==='
    ls -lath '${coredump_dir}/' 2>/dev/null | head -20
    echo '=== /var/crash (apport) ==='
    ls -lath /var/crash/ 2>/dev/null | head -10
    echo '=== /tmp/core.* (container fallback) ==='
    ls -lath /tmp/core.* 2>/dev/null | head -5
    echo '=== Recent journalctl -p err (last 1h) ==='
    journalctl --since='1 hour ago' -p err --no-pager 2>/dev/null | tail -20
  "
}

# Open the latest core file in gdb inside the container that produced it.
# Usage: dbg_gdb_latest <prefill|decode>
dbg_gdb_latest() {
  local role=${1:?usage: dbg_gdb_latest <prefill|decode>}
  local node container coredump_dir latest
  node="$(_dbg_node_for_role "$role")" || return 1
  container="$(_dbg_container "$role")" || return 1
  coredump_dir="${COREDUMP_DIR:-/var/tmp/cores}"
  latest="$(_dbg_on "$node" "ls -t ${coredump_dir}/core.* 2>/dev/null | head -1")"
  if [[ -z "$latest" ]]; then
    echo "No core file under ${coredump_dir}/ on $node" >&2
    return 1
  fi
  echo "core: $latest (on $node)"
  _dbg_on_tty "$node" "docker exec -it $container bash -c \"gdb -ex 'set pagination off' -ex 'bt full' python3 '$latest'\""
}

dbg_rdma() {
  local target=${1:-both}
  local cmd="ibstat 2>/dev/null | grep -E '^(CA|\s+(State|Rate|Port|Link))' | head -60"
  if [[ "$target" == "both" ]]; then
    for n in "${PREFILL_NODE}" "${DECODE_NODE}"; do
      echo "--- $n ---"
      _dbg_on "$n" "$cmd"
    done
  else
    _dbg_on "$target" "$cmd"
  fi
}

dbg_ports() {
  local node=${1:-${PREFILL_NODE}}
  local pat="${PREFILL_PORT}|${DECODE_PORT}|${ROUTER_PORT}|8998|19100|20000|20020|20040|21000|21020|21040|18000|18100"
  _dbg_on "$node" "ss -tlnp 2>/dev/null | awk 'NR==1 || /:(${pat}) /'"
}

dbg_cleanup() {
  if [[ "${1:-}" != "--yes" ]]; then
    cat <<EOF
About to remove on each node:
  on ${PREFILL_NODE}:
    docker rm -f $(_dbg_container prefill) $(_dbg_container router)
  on ${DECODE_NODE}:
    docker rm -f $(_dbg_container decode)

Bench container ${BENCH_CONTAINER} on $(hostname -s) is NOT touched.

Re-run with --yes to confirm.
EOF
    return 1
  fi
  echo ">>> Cleaning prefill+router on ${PREFILL_NODE}..."
  _dbg_on "${PREFILL_NODE}" "docker rm -f $(_dbg_container prefill) $(_dbg_container router) 2>/dev/null; true"
  echo ">>> Cleaning decode on ${DECODE_NODE}..."
  _dbg_on "${DECODE_NODE}" "docker rm -f $(_dbg_container decode) 2>/dev/null; true"
  echo "Done."
}

# ----- side effect on source: print one-line hint --------------------------
echo "dbg_helpers loaded. Run 'dbg_help' for the full list." >&2
echo "  PREFILL_NODE=${PREFILL_NODE}" >&2
echo "  DECODE_NODE =${DECODE_NODE}" >&2
