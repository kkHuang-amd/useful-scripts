#!/usr/bin/env bash
set -euo pipefail

BENCH_DIR="${BENCH_DIR:-/sgl-workspace/kvv-bench/kvv-k3-0727-update}"
BEAM_DIR="$BENCH_DIR/beam"
ANSWERS="${ANSWERS:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.3}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-16384}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-}"
CONCURRENCY="${CONCURRENCY:-32}"

if [[ -z "$ANSWERS" || -z "$JUDGE_MODEL" || -z "$JUDGE_BASE_URL" ]]; then
  echo "usage: ANSWERS=/path/answers.jsonl JUDGE_MODEL=<id> \\" >&2
  echo "       JUDGE_BASE_URL=https://host/v1 JUDGE_API_KEY=<key> $0" >&2
  exit 2
fi

test -f "$ANSWERS" || {
  echo "error: answers file not found: $ANSWERS" >&2
  exit 1
}

OUTPUT="${OUTPUT:-${ANSWERS%.jsonl}_scores.jsonl}"
args=(
  --answers "$ANSWERS"
  --judge-model "$JUDGE_MODEL"
  --judge-base-url "$JUDGE_BASE_URL"
  --judge-api-key "$JUDGE_API_KEY"
  --judge-temperature "$JUDGE_TEMPERATURE"
  --judge-max-tokens "$JUDGE_MAX_TOKENS"
  --concurrency "$CONCURRENCY"
  --output "$OUTPUT"
)

if [[ -n "$JUDGE_REASONING_EFFORT" ]]; then
  args+=(--judge-reasoning-effort "$JUDGE_REASONING_EFFORT")
fi

cd "$BEAM_DIR"
uv run python beam_judge.py "${args[@]}"

