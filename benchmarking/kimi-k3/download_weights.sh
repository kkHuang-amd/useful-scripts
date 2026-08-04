#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-moonshotai/Kimi-K3}"
MODEL_PATH="${MODEL_PATH:-/dockerx/data/Kimi-K3}"

command -v hf >/dev/null 2>&1 || {
  echo "error: Hugging Face CLI 'hf' is not installed" >&2
  exit 1
}

mkdir -p "$(dirname "$MODEL_PATH")"

echo "Downloading ${REPO_ID} to ${MODEL_PATH}"
echo "Existing partial files will be resumed."
hf download "$REPO_ID" --local-dir "$MODEL_PATH"

test -f "$MODEL_PATH/config.json"
test -f "$MODEL_PATH/model.safetensors.index.json"

shard_count="$(printf '%s\n' "$MODEL_PATH"/model-*-of-*.safetensors | wc -l)"
if [[ "$shard_count" -ne 96 ]]; then
  echo "error: expected 96 weight shards, found ${shard_count}" >&2
  exit 1
fi

echo "Download verified: ${shard_count}/96 shards"

