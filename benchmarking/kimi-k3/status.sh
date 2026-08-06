#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL_PATH="${MODEL_PATH:-/dockerx/data/models/Kimi-K3}"

echo "== Processes =="
pgrep -af "sglang serve|eval.py (ocrbench|mmmu|kimi_toolcall)|beam_generate.py" \
  || echo "No Kimi-K3 server/benchmark process found."

echo
echo "== Endpoint =="
if curl -fsS --max-time 5 "$BASE_URL/models"; then
  echo
else
  echo "Endpoint unavailable: $BASE_URL" >&2
fi

echo
echo "== Weights =="
if [[ -d "$MODEL_PATH" ]]; then
  du -sh "$MODEL_PATH"
  count="$(printf '%s\n' "$MODEL_PATH"/model-*-of-*.safetensors | wc -l)"
  echo "weight_shards=$count"
else
  echo "Missing: $MODEL_PATH" >&2
fi

