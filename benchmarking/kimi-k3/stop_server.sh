#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/dockerx/data/Kimi-K3}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-30}"

mapfile -t pids < <(
  pgrep -f "sglang serve.*--model-path ${MODEL_PATH}" || true
)

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No SGLang server found for $MODEL_PATH"
  exit 0
fi

echo "Sending SIGINT to SGLang PID(s): ${pids[*]}"
kill -INT "${pids[@]}"

deadline=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  alive=()
  for pid in "${pids[@]}"; do
    kill -0 "$pid" 2>/dev/null && alive+=("$pid")
  done
  [[ "${#alive[@]}" -eq 0 ]] && {
    echo "Server stopped."
    exit 0
  }
  sleep 1
done

echo "Graceful shutdown timed out; sending SIGTERM to: ${alive[*]}" >&2
kill -TERM "${alive[@]}"

