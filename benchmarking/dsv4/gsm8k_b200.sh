#!/usr/bin/env bash
# GSM-8K correctness check against a running SGLang server (B200).
# Uses the upstream in-tree few_shot_gsm8k (native /generate endpoint), so it
# does NOT require lm_eval to be installed.
#
#   PORT     = server port            (default 8000)
#   NQ       = num questions          (default 1319 = full test set)
#   PARALLEL = concurrent requests    (default 64)
#   MAXTOK   = max new tokens         (default 512)
#
# Example:
#   bash gsm8k_b200.sh
#   NQ=400 bash gsm8k_b200.sh
set -euo pipefail

export PYTHONPATH=/sgl-workspace/sglang-upstream/python:${PYTHONPATH:-}

PORT="${PORT:-8000}"
NQ="${NQ:-1319}"
PARALLEL="${PARALLEL:-64}"
MAXTOK="${MAXTOK:-512}"

set -x
exec python3 -m sglang.test.few_shot_gsm8k \
    --host http://127.0.0.1 \
    --port "${PORT}" \
    --num-questions "${NQ}" \
    --num-shots 5 \
    --parallel "${PARALLEL}" \
    --max-new-tokens "${MAXTOK}"
