#!/usr/bin/env bash
# ============================================================
#  Restore the original /proc/sys/kernel/core_pattern (apport)
#  saved by setup_coredumps.sh. Run on the same node(s).
# ============================================================
set -euo pipefail

BACKUP_DIR="${BACKUP_DIR:-/var/tmp/dbg-1p1d-gpu-fault}"
HOSTNAME_SHORT="$(hostname -s)"
BACKUP_FILE="${BACKUP_DIR}/core_pattern.${HOSTNAME_SHORT}.original"

if [[ "$EUID" -ne 0 ]]; then
  if sudo -n true 2>/dev/null; then
    exec sudo -E bash "$0" "$@"
  else
    echo "ERROR: need passwordless sudo on ${HOSTNAME_SHORT}" >&2
    exit 1
  fi
fi

if [[ ! -f "$BACKUP_FILE" ]]; then
  echo "ERROR: no backup at $BACKUP_FILE; nothing to restore." >&2
  echo "       Falling back to apport default..." >&2
  echo '|/usr/share/apport/apport -p%p -s%s -c%c -d%d -P%P -u%u -g%g -F%F -- %E' > /proc/sys/kernel/core_pattern
else
  cat "$BACKUP_FILE" > /proc/sys/kernel/core_pattern
fi

echo "[restore_coredumps] core_pattern -> $(cat /proc/sys/kernel/core_pattern)"
