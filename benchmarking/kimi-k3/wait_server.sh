#!/usr/bin/env bash
set -euo pipefail

SERVER_LOG="${SERVER_LOG:-/tmp/kimi-k3-server.log}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5}"
READY_TEXT="The server is fired up and ready to roll!"

echo "Waiting up to ${TIMEOUT_SECONDS}s for full warmup in ${SERVER_LOG}"

deadline=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < deadline )); do
  if [[ -f "$SERVER_LOG" ]] && rg -q "$READY_TEXT" "$SERVER_LOG"; then
    echo "Kimi-K3 server is fully warmed up."
    exit 0
  fi

  if [[ -f "$SERVER_LOG" ]] && rg -q \
    "Initialization failed|Traceback|Received sigquit from a child process|Killed.*sglang serve" "$SERVER_LOG"; then
    echo "error: server startup failed; inspect ${SERVER_LOG}" >&2
    exit 1
  fi

  sleep "$INTERVAL_SECONDS"
done

echo "error: timed out waiting for full server warmup" >&2
exit 1

