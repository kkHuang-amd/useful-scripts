#!/usr/bin/env bash
# ============================================================
#  Enable kernel core dumps on this host for the 1p1d debug run.
#
#  Run on EVERY node that hosts a prefill/decode container
#  (defaults: 08-29 and 08-33). Idempotent — running twice is OK.
#
#  Usage:
#    bash /home/wunhuang/dbg-1p1d-gpu-fault/setup_coredumps.sh
#
#  Or remotely from 08-25:
#    for h in smci355-ccs-aus-n08-29 smci355-ccs-aus-n08-33; do
#      ssh "${h}.prov.aus.ccs.cpe.ice.amd.com" \
#        bash /home/wunhuang/dbg-1p1d-gpu-fault/setup_coredumps.sh
#    done
#
#  What it does:
#    * Backs up the current /proc/sys/kernel/core_pattern (apport by default)
#      into /var/tmp/dbg-1p1d-gpu-fault/core_pattern.<host>.original.
#    * Sets core_pattern to write into ${COREDUMP_DIR}/core.<exe>.<pid>.<ts>.<sig>
#      on LOCAL disk (see comment near COREDUMP_DIR for why not NFS).
#    * Sets fs.suid_dumpable=2 so privileged container processes still dump.
#    * Creates the destination dir mode 1777 so anyone can write.
#
#  The launch scripts already use --ulimit core=-1 and bind-mount COREDUMP_DIR
#  into the container, so once core_pattern is set the next crash will produce
#  a core file there.
#
#  Run restore_coredumps.sh to put apport back.
# ============================================================
set -euo pipefail

# COREDUMP_DIR must live on LOCAL disk on each node, not NFS:
#   1. The kernel writes the core file in the crashing process's mount
#      namespace (the container's). The path must therefore exist inside
#      the container — we handle that by mounting the same dir.
#   2. NFS root_squash mangles the kernel's root-credential writes into
#      0-byte files. Local disk avoids the squash entirely.
# Each launch_*_no_umbp.sh adds `-v ${COREDUMP_DIR}:${COREDUMP_DIR}` so the
# in-container path matches the host path.
COREDUMP_DIR="${COREDUMP_DIR:-/var/tmp/cores}"
# Backup of original /proc/sys/kernel/core_pattern, also local.
BACKUP_DIR="${BACKUP_DIR:-/var/tmp/dbg-1p1d-gpu-fault}"
HOSTNAME_SHORT="$(hostname -s)"

if [[ "$EUID" -ne 0 ]]; then
  if sudo -n true 2>/dev/null; then
    echo "[setup_coredumps] re-exec under sudo on ${HOSTNAME_SHORT}"
    exec sudo -E bash "$0" "$@"
  else
    echo "ERROR: need passwordless sudo on ${HOSTNAME_SHORT}" >&2
    exit 1
  fi
fi

# --- Backup current pattern (once per host) ---------------------------------
# Stored on LOCAL /var/tmp (not NFS) because NFS root-squashes sudo'd writes.
mkdir -p "$BACKUP_DIR"
chmod 0755 "$BACKUP_DIR"
BACKUP_FILE="${BACKUP_DIR}/core_pattern.${HOSTNAME_SHORT}.original"
if [[ ! -f "$BACKUP_FILE" ]]; then
  cp /proc/sys/kernel/core_pattern "$BACKUP_FILE"
  echo "[setup_coredumps] backed up original core_pattern -> ${BACKUP_FILE}"
  cat "$BACKUP_FILE"
else
  echo "[setup_coredumps] backup already exists at ${BACKUP_FILE} (not overwriting)"
fi

# --- Create destination dir (LOCAL, world-writable) -------------------------
mkdir -p "$COREDUMP_DIR"
chmod 1777 "$COREDUMP_DIR"

# --- Apply new core_pattern -------------------------------------------------
# %e=exe %p=pid %t=timestamp %s=signal
NEW_PATTERN="${COREDUMP_DIR}/core.%e.%p.%t.%s"
echo "$NEW_PATTERN" > /proc/sys/kernel/core_pattern
echo 2 > /proc/sys/fs/suid_dumpable
echo 1 > /proc/sys/kernel/core_uses_pid

echo "[setup_coredumps] core_pattern  -> $(cat /proc/sys/kernel/core_pattern)"
echo "[setup_coredumps] suid_dumpable -> $(cat /proc/sys/fs/suid_dumpable)"

# --- Sanity test ------------------------------------------------------------
TEST_FILE="${COREDUMP_DIR}/.write_test.$$"
if touch "$TEST_FILE" 2>/dev/null; then
  rm -f "$TEST_FILE"
  echo "[setup_coredumps] write test OK at ${COREDUMP_DIR}"
else
  echo "WARNING: cannot write to ${COREDUMP_DIR}; cores may fail." >&2
fi

cat <<EOF

============================================================
  core dumps enabled on ${HOSTNAME_SHORT}
  pattern: $(cat /proc/sys/kernel/core_pattern)
  size limit (host shell): $(ulimit -c)
  container side: launch_*_no_umbp.sh already passes --ulimit core=-1
============================================================
EOF
