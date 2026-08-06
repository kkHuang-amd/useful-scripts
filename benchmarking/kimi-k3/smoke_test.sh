#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL_PATH="${MODEL_PATH:-/dockerx/data/models/Kimi-K3}"

command -v curl >/dev/null 2>&1 || {
  echo "error: curl is required" >&2
  exit 1
}
command -v jq >/dev/null 2>&1 || {
  echo "error: jq is required" >&2
  exit 1
}

models="$(curl -fsS --max-time 10 "$BASE_URL/models")"
served_id="$(jq -r '.data[0].id // empty' <<<"$models")"

if [[ "$served_id" != "$MODEL_PATH" ]]; then
  echo "error: expected model '$MODEL_PATH', endpoint serves '$served_id'" >&2
  exit 1
fi

payload="$(jq -nc --arg model "$served_id" '{
  model: $model,
  messages: [{role: "user", content: "Reply with OK only."}],
  max_tokens: 32,
  chat_template_kwargs: {thinking: false}
}')"

response="$(curl -fsS --max-time 120 \
  "$BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  --data "$payload")"

content="$(jq -r '.choices[0].message.content // empty' <<<"$response")"
if [[ -z "$content" ]]; then
  echo "error: smoke response has no assistant content" >&2
  jq . <<<"$response" >&2
  exit 1
fi

echo "served_model=$served_id"
echo "assistant_content=$content"
echo "Smoke test passed."

