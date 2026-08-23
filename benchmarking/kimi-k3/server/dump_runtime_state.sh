#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SGLANG_REPO="${SGLANG_REPO:-/sgl-workspace/sglang}"
AITER_REPO="${AITER_REPO:-/sgl-workspace/aiter}"

mkdir -p "$OUTPUT_DIR"

{
  date --iso-8601=seconds
  uname -a
  python --version
} >"$OUTPUT_DIR/runtime.txt" 2>&1

env | LC_ALL=C sort | rg \
  '^(AITER_|SGLANG_|HIP_|ROCM_|TRITON_|PYTORCH_|CUDA_|HSA_)' \
  >"$OUTPUT_DIR/accelerator-env.txt" || true

python -m pip freeze >"$OUTPUT_DIR/pip-freeze.txt"
rocm-smi --showproductname --showdriverversion --showmeminfo vram \
  >"$OUTPUT_DIR/rocm-smi.txt" 2>&1 || true

for item in "sglang:$SGLANG_REPO" "aiter:$AITER_REPO"; do
  name=${item%%:*}
  path=${item#*:}
  if git -C "$path" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    {
      git -C "$path" rev-parse HEAD
      git -C "$path" status --short --branch --untracked-files=all
      git -C "$path" diff --stat
    } >"$OUTPUT_DIR/${name}-git.txt"
  fi
done

curl --silent --show-error --max-time 5 "$BASE_URL/v1/models" \
  >"$OUTPUT_DIR/models.json" 2>"$OUTPUT_DIR/models-error.txt" || true

sha256sum "$OUTPUT_DIR"/* >"$OUTPUT_DIR/SHA256SUMS"
echo "RUNTIME_STATE_DUMPED dir=$OUTPUT_DIR"
