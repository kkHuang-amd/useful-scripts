#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_common_profile_case.sh ENGINE CONCURRENCY CASE_DIR MANIFEST

Run one production-graph Kimi-K3 decode-profile case.

Arguments:
  ENGINE       sglang or atom
  CONCURRENCY  2 or 64
  CASE_DIR     fresh output directory for this case
  MANIFEST     persisted common-client prompt manifest (.jsonl.gz)

Optional environment:
  PROFILE_SECONDS       Profile duration in seconds (default: 2)
  TRANSITION_OUTPUT_LEN Unprofiled graph/dispatch warmup output length (default: 64)
  DECODE_OUTPUT_LEN     Profiled output length (default: 256 for C2, 1024 for C64)
  SGLANG_DECODE_ATTENTION_BACKEND
                        SGLang decode backend: triton or aiter (default: triton)
  SGLANG_MEM_FRACTION_STATIC
                        SGLang launch memory fraction (default: 0.85)

The runner starts a production-configured TP8 server, runs one unprofiled
manifest wave, profiles one decode wave from its first streamed token, validates
all eight rank traces, analyzes them, stops all process groups, and waits for
VRAM to return to its pre-launch level. It never enables eager execution.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 4 ]]; then
  usage >&2
  exit 2
fi

engine=$1
concurrency=$2
case_dir=$3
manifest=$4

case "$engine" in
  sglang | atom) ;;
  *)
    echo "ERROR invalid engine: $engine (expected sglang or atom)" >&2
    exit 2
    ;;
esac
case "$concurrency" in
  2 | 64) ;;
  *)
    echo "ERROR invalid concurrency: $concurrency (expected 2 or 64)" >&2
    exit 2
    ;;
esac

profile_seconds=${PROFILE_SECONDS:-2}
transition_output_len=${TRANSITION_OUTPUT_LEN:-64}
decode_attention_backend=${SGLANG_DECODE_ATTENTION_BACKEND:-triton}
mem_fraction_static=${SGLANG_MEM_FRACTION_STATIC:-0.85}
if [[ -n "${DECODE_OUTPUT_LEN:-}" ]]; then
  decode_output_len=$DECODE_OUTPUT_LEN
elif [[ "$concurrency" == 2 ]]; then
  decode_output_len=256
else
  decode_output_len=1024
fi

if [[ ! "$profile_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  ! python3 - "$profile_seconds" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)
PY
then
  echo "ERROR PROFILE_SECONDS must be greater than zero: $profile_seconds" >&2
  exit 2
fi
for value_name in transition_output_len decode_output_len; do
  value=${!value_name}
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR $value_name must be a positive integer: $value" >&2
    exit 2
  fi
done
if [[ "$decode_attention_backend" != triton && "$decode_attention_backend" != aiter ]]; then
  echo "ERROR SGLANG_DECODE_ATTENTION_BACKEND must be triton or aiter: $decode_attention_backend" >&2
  exit 2
fi
if [[ ! "$mem_fraction_static" =~ ^(0|1|0?[.][0-9]+|1[.]0+)$ ]] ||
  ! python3 - "$mem_fraction_static" <<'PY'
import sys
value = float(sys.argv[1])
raise SystemExit(0 if 0 < value <= 1 else 1)
PY
then
  echo "ERROR SGLANG_MEM_FRACTION_STATIC must be in (0, 1]: $mem_fraction_static" >&2
  exit 2
fi
if [[ ! -f "$manifest" ]]; then
  echo "ERROR manifest does not exist: $manifest" >&2
  exit 2
fi

model=/shared_nfs/models/Kimi-K3
client=/workspace/useful-scripts/benchmarking/common_oai_benchmark.py
analyzer=/workspace/useful-scripts/benchmarking/kimi-k3/analysis/analyze_chrome_trace.py
sglang_root=/sgl-workspace/sglang-k3-triton37
atom_root=/sgl-workspace/ATOM
aiter_root=/sgl-workspace/aiter-atom-current
expected_aiter_sha=dc4bdf1c142181ad90b7f6948564126df4c05fde
expected_flydsl=0.3.1
expected_ranks=8
max_trace_bytes=$((500 * 1024 * 1024))
server_pid=
client_pid=
server_started=0
gpu_release_checked=0

if [[ -e "$case_dir/server.log" || -e "$case_dir/decode-traces" ]]; then
  echo "ERROR CASE_DIR already contains server.log or decode-traces: $case_dir" >&2
  echo "Use a fresh case directory; existing evidence is never deleted." >&2
  exit 2
fi

mkdir -p "$case_dir" "$case_dir/decode-traces" "$case_dir/analysis"
shopt -s nullglob globstar
case_dir=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$case_dir")
manifest=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$manifest")
trace_dir="$case_dir/decode-traces"
server_log="$case_dir/server.log"

stop_group() {
  local pid=${1:-}
  [[ -n "$pid" ]] || return 0
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 -- "-$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
    fi
  fi
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_group "$client_pid"
  client_pid=
  stop_group "$server_pid"
  server_pid=
  if (( server_started == 1 && gpu_release_checked == 0 )) &&
    [[ -f "$case_dir/gpu-before.json" ]]; then
    wait_for_gpu_release "$case_dir/gpu-before.json" || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

snapshot_gpu_memory() {
  local output=$1
  if command -v amd-smi >/dev/null 2>&1; then
    timeout 10 amd-smi metric --mem-usage --json >"$output" 2>"${output%.json}.log"
  else
    printf '{"status":"amd-smi-unavailable"}\n' >"$output"
  fi
}

wait_for_gpu_release() {
  local before=$1
  local current="$case_dir/gpu-after.json"
  if ! command -v amd-smi >/dev/null 2>&1; then
    echo "GPU_RELEASE_SKIP amd-smi_unavailable"
    return 0
  fi
  for _ in $(seq 1 180); do
    if timeout 10 amd-smi metric --mem-usage --json >"$current.tmp" 2>>"$case_dir/gpu-after.log" &&
      python3 - "$before" "$current.tmp" <<'PY'
import json
import sys

def used(path):
    payload = json.load(open(path))
    return {
        int(item["gpu"]): float(item["mem_usage"]["used_vram"]["value"])
        for item in payload["gpu_data"]
    }

baseline = used(sys.argv[1])
current = used(sys.argv[2])
ok = baseline.keys() == current.keys() and all(
    current[gpu] <= baseline[gpu] + 512 for gpu in baseline
)
raise SystemExit(0 if ok else 1)
PY
    then
      mv "$current.tmp" "$current"
      echo "GPU_RELEASED baseline_plus_mib=512"
      return 0
    fi
    rm -f "$current.tmp"
    sleep 2
  done
  echo "ERROR GPU memory did not return to baseline within 360 seconds" >&2
  return 1
}

snapshot_gpu_memory "$case_dir/gpu-before.json"

PYTHONPATH="$aiter_root:$sglang_root/python:$atom_root" \
  python3 - "$case_dir/runtime-metadata.json" "$manifest" "$engine" \
  "$concurrency" "$profile_seconds" "$transition_output_len" \
  "$decode_output_len" "$decode_attention_backend" "$mem_fraction_static" \
  "$expected_aiter_sha" "$expected_flydsl" <<'PY'
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiter
import flydsl
import torch

(
    output_path,
    manifest_path,
    engine,
    concurrency,
    profile_seconds,
    transition_output_len,
    decode_output_len,
    decode_attention_backend,
    mem_fraction_static,
    expected_aiter_sha,
    expected_flydsl,
) = sys.argv[1:]

roots = {
    "sglang": "/sgl-workspace/sglang-k3-triton37",
    "atom": "/sgl-workspace/ATOM",
    "aiter": "/sgl-workspace/aiter-atom-current",
}

def git_sha(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()

def git_status(path):
    return subprocess.check_output(
        ["git", "-C", path, "status", "--short"], text=True
    ).splitlines()

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

entries = []
logical_digest = hashlib.sha256()
with gzip.open(manifest_path, "rt") as handle:
    for line in handle:
        entry = json.loads(line)
        entries.append(entry)
        logical_digest.update(entry["prompt"].encode())

required = int(concurrency)
if len(entries) < required:
    raise SystemExit(f"manifest has {len(entries)} prompts, needs {required}")
for index, entry in enumerate(entries[:required]):
    if entry.get("request_id") != index:
        raise SystemExit(
            f"manifest request_id at index {index} is {entry.get('request_id')!r}"
        )
    if entry.get("prompt_tokens") != 8192:
        raise SystemExit(
            f"manifest prompt_tokens at index {index} is "
            f"{entry.get('prompt_tokens')!r}, expected 8192"
        )

shas = {name: git_sha(path) for name, path in roots.items()}
if shas["aiter"] != expected_aiter_sha:
    raise SystemExit(
        f"AITER SHA is {shas['aiter']}, expected {expected_aiter_sha}"
    )
aiter_file = str(Path(aiter.__file__).resolve())
if not aiter_file.startswith(str(Path(roots["aiter"]).resolve()) + os.sep):
    raise SystemExit(f"wrong AITER import: {aiter_file}")
flydsl_version = (
    getattr(flydsl, "__version__", None)
    or importlib.metadata.version("flydsl")
)
if flydsl_version != expected_flydsl:
    raise SystemExit(
        f"FlyDSL version is {flydsl_version}, expected {expected_flydsl}"
    )

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "engine": engine,
    "concurrency": int(concurrency),
    "model": "/shared_nfs/models/Kimi-K3",
    "config": {
        "tp_size": 8,
        "kv_cache_dtype": "fp8",
        "graphs": "full",
        "profile_seconds": float(profile_seconds),
        "transition_input_len": 8192,
        "transition_output_len": int(float(transition_output_len)),
        "decode_input_len": 8192,
        "decode_output_len": int(float(decode_output_len)),
        "decode_attention_backend": (
            decode_attention_backend if engine == "sglang" else "atom-native"
        ),
        "mem_fraction_static_launch": float(mem_fraction_static),
        "profile_activities": ["CPU", "GPU"],
        "profile_with_stack": False,
        "profile_record_shapes": False,
    },
    "git": {
        name: {
            "path": roots[name],
            "sha": sha,
            "status_short": git_status(roots[name]),
        }
        for name, sha in shas.items()
    },
    "runtime": {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "flydsl": flydsl_version,
        "aiter_file": aiter_file,
    },
    "manifest": {
        "path": str(Path(manifest_path).resolve()),
        "file_sha256": sha256_file(manifest_path),
        "logical_prompt_sha256": logical_digest.hexdigest(),
        "count": len(entries),
        "selected_request_ids": [entry["request_id"] for entry in entries[:required]],
    },
}
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
print(
    f"RUNTIME_OK engine={engine} c={concurrency} "
    f"aiter={shas['aiter'][:8]} flydsl={flydsl_version} "
    f"decode_backend={decode_attention_backend if engine == 'sglang' else 'atom-native'}"
)
PY

if [[ "$engine" == sglang ]]; then
  port=30000
  server_pythonpath="$sglang_root/python:$aiter_root"
  setsid env -u KINETO_CONFIG \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$server_pythonpath" \
    AITER_JIT_DIR=/tmp/aiter-jit-common-oai-sglang-same-aiter \
    SGLANG_CACHE_DIR=/tmp/sglang-cache-common-oai-sglang-same-aiter \
    SGLANG_K3_FLYDSL_SOURCE=sglang \
    SGLANG_K3_AITER_M16384_PROFILE=1 \
    SGLANG_USE_AITER=1 \
    SGLANG_AITER_K3_OPT=1 \
    AITER_FLYDSL_FORCE=1 \
    AITER_SITUV2_A8W4=1 \
    AITER_SITUV2_A4W4=0 \
    AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1 \
    SGLANG_K3_FLYDSL_AR_NORM=1 \
    SGLANG_K3_KDA_FUSED_BACKEND=aiter \
    SGLANG_K3_AITER_MLA_GATE=1 \
    SGLANG_K3_AITER_KDA_GROUP64=1 \
    SGLANG_K3_AITER_MOE_PREROUTE_FP8=1 \
    SGLANG_K3_PREROUTE_PREACTIVATED_SHARED=1 \
    SGLANG_K3_AITER_LATENT_TAIL_FP8=0 \
    SGLANG_K3_AITER_B2_FUSIONS=1 \
    SGLANG_K3_AITER_MLA_Q_CACHE_FUSION=1 \
    SGLANG_K3_RADIX4_TOPK=1 \
    SGLANG_AITER_FP8_PREFILL_ATTN=0 \
    SGLANG_MLA_DECODE_TUNE=1 \
    SGLANG_TRITON_37_EXTEND_LQ576_N32=0 \
    SGLANG_ROCM_K3_FUSE_KDA_INPROJ=1 \
    SGLANG_K3_AITER_TUNED_MOE_FRONT=1 \
    SGLANG_K3_AITER_TUNED_MOE_FRONT_MIN_TOKENS=48 \
    SGLANG_K3_AITER_TUNED_MOE_FRONT_MAX_TOKENS=192 \
    SGLANG_K3_MOE_LATENT_MXFP4=0 \
    SGLANG_K3_MOE_LATENT_DOWN_MXFP4=0 \
    SGLANG_K3_MOE_LATENT_UP_MXFP4=1 \
    SGLANG_K3_MOE_LATENT_MXFP4_MIN_TOKENS=2048 \
    SGLANG_PROFILE_V2=0 \
    SGLANG_PROFILE_WITH_STACK=false \
    SGLANG_PROFILE_RECORD_SHAPES=false \
    python -m sglang.launch_server \
      --model-path "$model" \
      --trust-remote-code \
      --host 127.0.0.1 \
      --port "$port" \
      --tp-size 8 \
      --kv-cache-dtype fp8_e4m3 \
      --mem-fraction-static "$mem_fraction_static" \
      --max-running-requests 256 \
      --chunked-prefill-size 16384 \
      --max-prefill-tokens 16384 \
      --disable-radix-cache \
      --attention-backend triton \
      --prefill-attention-backend aiter \
      --decode-attention-backend "$decode_attention_backend" \
      --sampling-backend pytorch \
      --cuda-graph-max-bs-decode 256 \
      >"$server_log" 2>&1 &
else
  port=8000
  server_pythonpath="$aiter_root:$atom_root"
  setsid env -u KINETO_CONFIG \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$server_pythonpath" \
    ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0 \
    ATOM_PROFILER_MORE=0 \
    ATOM_ENABLE_DETAILED_ANNOTATION=1 \
    ATOM_PROFILER_TIMEOUT=600 \
    python -m atom.entrypoints.openai_server \
      --model "$model" \
      --kv_cache_dtype fp8 \
      -tp 8 \
      --trust-remote-code \
      --max-model-len 16384 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 16384 \
      --gpu-memory-utilization 0.93 \
      --block-size 128 \
      --no-enable_prefix_caching \
      --torch-profiler-dir "$trace_dir" \
      --online_quant_config \
        '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}' \
      >"$server_log" 2>&1 &
fi
server_pid=$!
server_started=1

ready=0
if [[ "$engine" == sglang ]]; then
  ready_checks=360
else
  ready_checks=180
fi
for _ in $(seq 1 "$ready_checks"); do
  if [[ "$engine" == sglang ]]; then
    if rg -q "ready to roll" "$server_log" 2>/dev/null; then
      ready=1
      break
    fi
  elif curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    break
  fi
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  echo "ERROR server did not become ready" >&2
  rg "Initialization failed|Traceback|ERROR|Error|RuntimeError|Killed|OutOfMemory" \
    "$server_log" || true
  exit 1
fi

if [[ "$engine" == sglang ]]; then
  full_count=$(rg -c "Capture target decode CUDA graph begin\. backend=full" "$server_log" || true)
  if (( full_count < expected_ranks )); then
    echo "ERROR SGLang did not capture FULL decode graphs on all ranks" >&2
    exit 1
  fi
  rg "server_args=|Capture target decode CUDA graph (begin|end)" \
    "$server_log" >"$case_dir/graph-mode.log"
else
  if ! rg -q "'enforce_eager': False" "$server_log"; then
    echo "ERROR ATOM effective args do not show enforce_eager=False" >&2
    exit 1
  fi
  if ! python3 - "$server_log" "$concurrency" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="replace").read()
expected = int(sys.argv[2])
for line in text.splitlines():
    if "Engine Core: cudagraph capture[" not in line:
        continue
    match = re.search(r"cudagraph capture\[([0-9, ]+)\]", line)
    if match and expected in {int(value) for value in match.group(1).split(",")}:
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    echo "ERROR ATOM graph capture does not include concurrency $concurrency" >&2
    exit 1
  fi
  rg "Engine kwargs:|cudagraph capture sizes|CUDA graph capture memory|Engine Core: cudagraph capture" \
    "$server_log" >"$case_dir/graph-mode.log"
fi
echo "READY engine=$engine graph=full port=$port"

if compgen -G "$trace_dir/**/*.json*" >/dev/null; then
  echo "ERROR profiler produced traces before the measured wave" >&2
  exit 1
fi

run_client_wave() {
  local label=$1
  local output_len=$2
  local profile=$3
  local output_dir="$case_dir/${label}-client"
  local log="$case_dir/${label}-client.log"
  local -a args=(
    "$client"
    --base-url "http://127.0.0.1:$port"
    --model "$model"
    --tokenizer "$model"
    --trust-remote-code
    --input-len 8192
    --output-len "$output_len"
    --num-prompts "$concurrency"
    --warmup-requests 0
    --max-concurrency "$concurrency"
    --seed 42
    --prompt-manifest "$manifest"
    --output-dir "$output_dir"
  )
  if [[ "$profile" == 1 ]]; then
    args+=(
      --profile-on-first-token
      --profile-engine "$engine"
      --profile-seconds "$profile_seconds"
      --profile-after-first-tokens "$concurrency"
      --profile-base-url "http://127.0.0.1:$port"
    )
    if [[ "$engine" == sglang ]]; then
      args+=(--profile-output-dir "$trace_dir")
    fi
  fi
  setsid python3 "${args[@]}" >"$log" 2>&1 &
  client_pid=$!
  local status
  if wait "$client_pid"; then
    status=0
  else
    status=$?
  fi
  client_pid=
  if (( status != 0 )); then
    echo "ERROR $label client failed; see $log" >&2
    return "$status"
  fi
  python3 - "$output_dir/summary.json" "$concurrency" "$output_len" "$label" <<'PY'
import json
import sys

path, concurrency, output_len, label = sys.argv[1:]
summary = json.load(open(path))
expected = int(concurrency)
if summary["successful_requests"] != expected or summary["failed_requests"] != 0:
    raise SystemExit(f"{label} request contract failed: {summary}")
if summary["config"]["max_concurrency"] != expected:
    raise SystemExit(f"{label} concurrency contract failed")
if summary["config"]["output_len"] != int(output_len):
    raise SystemExit(f"{label} output length contract failed")
print(
    f"{label.upper()}_OK requests={summary['successful_requests']} "
    f"output_len={output_len}"
)
PY
}

run_client_wave warmup "$transition_output_len" 0
if [[ "$engine" == sglang ]] &&
  ! rg -q "Decode batch.*cuda graph: True" "$server_log"; then
  echo "ERROR SGLang warmup wave did not replay a decode graph" >&2
  exit 1
fi

run_client_wave decode "$decode_output_len" 1
python3 - "$case_dir/decode-client/summary.json" \
  "$case_dir/profile-http-results.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.load(open(sys.argv[1]))
profile = summary.get("profile")
if not profile:
    raise SystemExit("decode summary has no profile result")
result = profile["result"]
for key in ("triggered", "success"):
    if not result.get(key):
        raise SystemExit(f"profile {key} gate failed: {result}")
for key in ("start_http_status", "stop_http_status"):
    if result.get(key) != 200:
        raise SystemExit(f"profile HTTP gate failed for {key}: {result.get(key)}")
Path(sys.argv[2]).write_text(json.dumps(profile, indent=2) + "\n")
print(
    f"PROFILE_HTTP_OK start={result['start_http_status']} "
    f"stop={result['stop_http_status']}"
)
PY

for _ in $(seq 1 600); do
  if [[ "$engine" == sglang ]]; then
    traces=("$trace_dir"/*.trace.json.gz)
  else
    traces=("$trace_dir"/rank_*/*.pt.trace.json.gz)
  fi
  if (( ${#traces[@]} >= expected_ranks )); then
    break
  fi
  sleep 1
done
if [[ "$engine" == sglang ]]; then
  traces=("$trace_dir"/*.trace.json.gz)
else
  traces=("$trace_dir"/rank_*/*.pt.trace.json.gz)
fi
all_gzip=("$trace_dir"/**/*.gz)
if (( ${#all_gzip[@]} != expected_ranks )); then
  echo "ERROR expected exactly $expected_ranks gzip files, found ${#all_gzip[@]}" >&2
  exit 1
fi
if (( ${#traces[@]} != expected_ranks )); then
  echo "ERROR expected exactly $expected_ranks $engine rank traces, found ${#traces[@]}" >&2
  exit 1
fi
orphans=("$trace_dir"/**/*.json)
if (( ${#orphans[@]} != 0 )); then
  echo "ERROR found ${#orphans[@]} orphan uncompressed JSON traces" >&2
  printf '%s\n' "${orphans[@]}" >&2
  exit 1
fi

declare -A ranks_seen=()
for trace in "${traces[@]}"; do
  gzip -t "$trace"
  size=$(stat -c %s "$trace")
  if (( size > max_trace_bytes )); then
    echo "ERROR trace exceeds 500 MiB: $trace ($size bytes)" >&2
    exit 1
  fi
  if [[ "$engine" == sglang && "$(basename "$trace")" =~ TP-([0-7]) ]]; then
    rank=${BASH_REMATCH[1]}
  elif [[ "$engine" == atom && "$(basename "$(dirname "$trace")")" =~ ^rank_([0-7])$ ]]; then
    rank=${BASH_REMATCH[1]}
  else
    echo "ERROR cannot identify TP rank for trace: $trace" >&2
    exit 1
  fi
  if [[ -n "${ranks_seen[$rank]:-}" ]]; then
    echo "ERROR duplicate trace for rank $rank" >&2
    exit 1
  fi
  ranks_seen[$rank]=1
done
for rank in $(seq 0 7); do
  if [[ -z "${ranks_seen[$rank]:-}" ]]; then
    echo "ERROR missing trace for rank $rank" >&2
    exit 1
  fi
done
echo "TRACE_EXPORT_OK engine=$engine traces=${#traces[@]} max_mib=500"

stop_group "$server_pid"
server_pid=
gpu_release_checked=1
wait_for_gpu_release "$case_dir/gpu-before.json"

: >"$case_dir/analysis/analyze.log"
for trace in "${traces[@]}"; do
  if [[ "$engine" == sglang && "$(basename "$trace")" =~ TP-([0-7]) ]]; then
    rank=${BASH_REMATCH[1]}
  else
    [[ "$(basename "$(dirname "$trace")")" =~ ^rank_([0-7])$ ]]
    rank=${BASH_REMATCH[1]}
  fi
  summary="$case_dir/analysis/rank-${rank}-summary.json"
  caller_map="$case_dir/analysis/rank-${rank}-caller-map.jsonl"
  python3 "$analyzer" "$trace" \
    --output "$summary" \
    --decode-only \
    --emit-caller-map "$caller_map" \
    >>"$case_dir/analysis/analyze.log" 2>&1
  python3 - "$summary" "$rank" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
rank = sys.argv[2]
gates = summary["validation_gates"]
counts = gates["category_counts"]
required = {
    "decode_kernel": summary["kernel_count"] > 0,
    "cpu_op": counts["cpu_op"] > 0,
    "annotation": counts["user_annotation"] + counts["gpu_user_annotation"] > 0,
    "cuda_runtime": counts["cuda_runtime"] > 0,
    "correlation_or_flow": gates["has_correlation_or_flow"],
    "decode_window": gates["decode_window_found"],
    "graph_replay": summary["graph_replay"]["detected"],
}
failed = [name for name, passed in required.items() if not passed]
if failed:
    raise SystemExit(f"rank {rank} trace gates failed: {failed}; counts={counts}")
print(
    f"TRACE_GATES_OK rank={rank} kernels={summary['kernel_count']} "
    f"cpu_op={counts['cpu_op']} annotation="
    f"{counts['user_annotation'] + counts['gpu_user_annotation']} "
    f"cuda_runtime={counts['cuda_runtime']}"
)
PY
done

python3 - "$case_dir/trace-inventory.json" "${traces[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

items = []
for value in sys.argv[2:]:
    path = Path(value).resolve()
    items.append(
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    )
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "count": len(items),
    "traces": sorted(items, key=lambda item: item["path"]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
print(f"INVENTORY_OK traces={len(items)}")
PY

rg '"successful_requests"|"failed_requests"|"duration_s"|"median_ttft_ms"|"median_tpot_ms"' \
  "$case_dir/warmup-client/summary.json" \
  "$case_dir/decode-client/summary.json" || true
echo "CASE_COMPLETE engine=$engine concurrency=$concurrency case_dir=$case_dir"
