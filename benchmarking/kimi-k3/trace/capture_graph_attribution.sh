#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  capture_graph_attribution.sh ENGINE OUTPUT_DIR

Capture diagnostic Kimi-K3 production graph-construction traces.

Arguments:
  ENGINE      sglang or atom
  OUTPUT_DIR  fresh evidence directory (must not already exist)

The server keeps FULL/non-eager production graphs enabled and captures only
decode batch sizes 2 and 64. SGLang emits one combined warmup/construction
trace per TP rank (8 files). ATOM emits BS2/q1 and BS64/q1 traces for every TP
rank (16 files). The server is stopped as soon as initialization is ready; no
endpoint workload is run.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 2 ]]; then
  usage >&2
  exit 2
fi

engine=$1
output_dir=$2
case "$engine" in
  sglang | atom) ;;
  *)
    echo "ERROR invalid engine: $engine (expected sglang or atom)" >&2
    exit 2
    ;;
esac

if [[ -e "$output_dir" ]]; then
  echo "ERROR OUTPUT_DIR already exists: $output_dir" >&2
  echo "Use a fresh path; existing evidence is never deleted." >&2
  exit 2
fi

model=/shared_nfs/models/Kimi-K3
sglang_root=/sgl-workspace/sglang-k3-triton37
atom_root=/sgl-workspace/ATOM
aiter_root=/sgl-workspace/aiter-atom-current
tool_root=/workspace/useful-scripts/benchmarking/kimi-k3
runtime_dumper="$tool_root/server/dump_runtime_state.sh"
analyzer="$tool_root/analysis/analyze_chrome_trace.py"
expected_aiter_sha=dc4bdf1c142181ad90b7f6948564126df4c05fde
expected_flydsl=0.3.1
expected_ranks=8
max_trace_bytes=$((500 * 1024 * 1024))
server_pid=
server_started=0
gpu_release_checked=0

mkdir -p "$output_dir/traces" "$output_dir/analysis" "$output_dir/runtime"
output_dir=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$output_dir")
trace_dir="$output_dir/traces"
server_log="$output_dir/server.log"
shopt -s nullglob globstar

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
  local current="$output_dir/gpu-after.json"
  if ! command -v amd-smi >/dev/null 2>&1; then
    echo "GPU_RELEASE_SKIP amd-smi_unavailable"
    return 0
  fi
  for _ in $(seq 1 180); do
    if timeout 10 amd-smi metric --mem-usage --json >"$current.tmp" \
      2>>"$output_dir/gpu-after.log" &&
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

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  stop_group "$server_pid"
  server_pid=
  if (( server_started == 1 && gpu_release_checked == 0 )) &&
    [[ -f "$output_dir/gpu-before.json" ]]; then
    wait_for_gpu_release "$output_dir/gpu-before.json" || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

snapshot_gpu_memory "$output_dir/gpu-before.json"

PYTHONPATH="$aiter_root:$sglang_root/python:$atom_root" \
  python3 - "$output_dir/runtime-contract.json" "$engine" \
  "$expected_aiter_sha" "$expected_flydsl" <<'PY'
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

output_path, engine, expected_aiter_sha, expected_flydsl = sys.argv[1:]
roots = {
    "sglang": "/sgl-workspace/sglang-k3-triton37",
    "atom": "/sgl-workspace/ATOM",
    "aiter": "/sgl-workspace/aiter-atom-current",
}

def git(path, *args):
    return subprocess.check_output(["git", "-C", path, *args], text=True).strip()

shas = {name: git(path, "rev-parse", "HEAD") for name, path in roots.items()}
if shas["aiter"] != expected_aiter_sha:
    raise SystemExit(f"AITER SHA is {shas['aiter']}, expected {expected_aiter_sha}")
aiter_file = str(Path(aiter.__file__).resolve())
if not aiter_file.startswith(str(Path(roots["aiter"]).resolve()) + os.sep):
    raise SystemExit(f"wrong AITER import: {aiter_file}")
flydsl_version = (
    getattr(flydsl, "__version__", None) or importlib.metadata.version("flydsl")
)
if flydsl_version != expected_flydsl:
    raise SystemExit(
        f"FlyDSL version is {flydsl_version}, expected {expected_flydsl}"
    )

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "purpose": "diagnostic production graph construction attribution",
    "engine": engine,
    "model": "/shared_nfs/models/Kimi-K3",
    "config": {
        "tp_size": 8,
        "kv_cache_dtype": "fp8",
        "graph_mode": "FULL",
        "graph_batch_sizes": [2, 64],
        "eager": False,
        "profile_activities": ["CPU", "GPU"],
    },
    "git": {
        name: {
            "path": roots[name],
            "sha": shas[name],
            "status_short": git(roots[name], "status", "--short").splitlines(),
        }
        for name in roots
    },
    "runtime": {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_hip": torch.version.hip,
        "flydsl": flydsl_version,
        "aiter_file": aiter_file,
    },
}
Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
print(
    f"RUNTIME_OK engine={engine} aiter={shas['aiter'][:8]} "
    f"flydsl={flydsl_version}"
)
PY

common_sglang_env=(
  PYTHONUNBUFFERED=1
  "PYTHONPATH=$sglang_root/python:$aiter_root"
  AITER_JIT_DIR=/tmp/aiter-jit-common-oai-sglang-same-aiter
  SGLANG_CACHE_DIR=/tmp/sglang-cache-common-oai-sglang-same-aiter
  SGLANG_K3_FLYDSL_SOURCE=sglang
  SGLANG_K3_AITER_M16384_PROFILE=1
  SGLANG_USE_AITER=1
  SGLANG_AITER_K3_OPT=1
  AITER_FLYDSL_FORCE=1
  AITER_SITUV2_A8W4=1
  AITER_SITUV2_A4W4=0
  AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1
  SGLANG_K3_FLYDSL_AR_NORM=1
  SGLANG_K3_KDA_FUSED_BACKEND=aiter
  SGLANG_K3_AITER_MLA_GATE=1
  SGLANG_K3_AITER_KDA_GROUP64=1
  SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
  SGLANG_K3_PREROUTE_PREACTIVATED_SHARED=1
  SGLANG_K3_AITER_LATENT_TAIL_FP8=0
  SGLANG_K3_AITER_B2_FUSIONS=1
  SGLANG_K3_AITER_MLA_Q_CACHE_FUSION=1
  SGLANG_K3_RADIX4_TOPK=1
  SGLANG_AITER_FP8_PREFILL_ATTN=0
  SGLANG_MLA_DECODE_TUNE=1
  SGLANG_TRITON_37_EXTEND_LQ576_N32=0
  SGLANG_ROCM_K3_FUSE_KDA_INPROJ=1
  SGLANG_K3_AITER_TUNED_MOE_FRONT=1
  SGLANG_K3_AITER_TUNED_MOE_FRONT_MIN_TOKENS=48
  SGLANG_K3_AITER_TUNED_MOE_FRONT_MAX_TOKENS=192
  SGLANG_K3_MOE_LATENT_MXFP4=0
  SGLANG_K3_MOE_LATENT_DOWN_MXFP4=0
  SGLANG_K3_MOE_LATENT_UP_MXFP4=1
  SGLANG_K3_MOE_LATENT_MXFP4_MIN_TOKENS=2048
  SGLANG_PROFILE_V2=0
  SGLANG_PROFILE_WITH_STACK=false
  SGLANG_PROFILE_RECORD_SHAPES=false
  SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE=1
  "SGLANG_TORCH_PROFILER_DIR=$trace_dir"
)

if [[ "$engine" == sglang ]]; then
  port=30000
  launch_env=("${common_sglang_env[@]}")
  launch_cmd=(
    python -m sglang.launch_server
    --model-path "$model"
    --trust-remote-code
    --host 127.0.0.1
    --port "$port"
    --tp-size 8
    --kv-cache-dtype fp8_e4m3
    --mem-fraction-static 0.85
    --max-running-requests 256
    --chunked-prefill-size 16384
    --max-prefill-tokens 16384
    --disable-radix-cache
    --attention-backend triton
    --prefill-attention-backend aiter
    --decode-attention-backend triton
    --sampling-backend pytorch
    --cuda-graph-backend-decode full
    --cuda-graph-bs-decode 2 64
    --enable-profile-cuda-graph
  )
else
  port=8000
  launch_env=(
    PYTHONUNBUFFERED=1
    "PYTHONPATH=$aiter_root:$atom_root"
    ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0
    ATOM_PROFILER_MORE=0
    ATOM_ENABLE_DETAILED_ANNOTATION=1
    ATOM_PROFILER_TIMEOUT=600
  )
  launch_cmd=(
    python -m atom.entrypoints.openai_server
    --model "$model"
    --kv_cache_dtype fp8
    -tp 8
    --trust-remote-code
    --max-model-len 16384
    --max-num-seqs 64
    --max-num-batched-tokens 16384
    --gpu-memory-utilization 0.93
    --block-size 128
    --no-enable_prefix_caching
    --cudagraph-mode FULL
    --cudagraph-capture-sizes '[2,64]'
    --mark-trace
    --torch-profiler-dir "$trace_dir"
    --online_quant_config
    '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}'
  )
fi

printf '%s\n' "${launch_env[@]}" >"$output_dir/launch-env.txt"
printf '%q ' "${launch_cmd[@]}" >"$output_dir/launch-command.sh"
printf '\n' >>"$output_dir/launch-command.sh"

(
  cd "$output_dir"
  exec setsid env -u KINETO_CONFIG "${launch_env[@]}" \
    "${launch_cmd[@]}" >"$server_log" 2>&1
) &
server_pid=$!
server_started=1

if [[ "$engine" == sglang ]]; then
  ready_checks=360
else
  ready_checks=180
fi
ready=0
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
  full_count=$(rg -c "Capture target decode CUDA graph begin\. backend=full" \
    "$server_log" || true)
  if (( full_count < expected_ranks )); then
    echo "ERROR SGLang did not construct FULL decode graphs on all ranks" >&2
    exit 1
  fi
  if ! rg -q "cuda_graph_bs_decode.*(2.*64|\\[2, 64\\])" "$server_log"; then
    echo "ERROR SGLang effective args do not show decode graph BS2 and BS64" >&2
    exit 1
  fi
  rg "server_args=|Capture target decode CUDA graph (begin|end)|CUDA graph capture trace saved" \
    "$server_log" >"$output_dir/graph-mode.log"
else
  if ! rg -q "'enforce_eager': False" "$server_log"; then
    echo "ERROR ATOM effective args do not show enforce_eager=False" >&2
    exit 1
  fi
  if ! rg -q "cudagraph_mode=<CUDAGraphMode\\.FULL:" "$server_log"; then
    echo "ERROR ATOM effective args do not show cudagraph_mode=FULL" >&2
    exit 1
  fi
  if ! rg -q "Engine Core: cudagraph capture\\[(2, 64|64, 2)\\]" "$server_log"; then
    echo "ERROR ATOM graph capture is not restricted to BS2 and BS64" >&2
    exit 1
  fi
  rg "Engine kwargs:|cudagraph capture sizes|CUDA graph capture memory|Engine Core: cudagraph capture|Saved capture trace" \
    "$server_log" >"$output_dir/graph-mode.log"
fi
echo "READY engine=$engine graph=full sizes=2,64 port=$port"

OUTPUT_DIR="$output_dir/runtime" \
BASE_URL="http://127.0.0.1:$port" \
SGLANG_REPO="$sglang_root" \
AITER_REPO="$aiter_root" \
  bash "$runtime_dumper" >"$output_dir/runtime-dump.log" 2>&1

stop_group "$server_pid"
server_pid=
gpu_release_checked=1
wait_for_gpu_release "$output_dir/gpu-before.json"

for _ in $(seq 1 60); do
  if [[ "$engine" == sglang ]]; then
    candidates=(
      "$trace_dir"/graph_capture_profile/cuda_graph_capture-DecodeCudaGraphRunner-TP-*.json.gz
    )
    (( ${#candidates[@]} >= 8 )) && break
  else
    candidates=("$trace_dir"/rank_*/capture_traces/bs_{2,64}_q_1_rank*.json.gz)
    (( ${#candidates[@]} >= 16 )) && break
  fi
  sleep 1
done

if [[ "$engine" == sglang ]]; then
  targets=(
    "$trace_dir"/graph_capture_profile/cuda_graph_capture-DecodeCudaGraphRunner-TP-*.json.gz
  )
  all_combined=("$trace_dir"/graph_capture_profile/cuda_graph_capture-*.json.gz)
  if (( ${#all_combined[@]} != expected_ranks )); then
    echo "ERROR expected exactly 8 SGLang combined capture traces, found ${#all_combined[@]}" >&2
    exit 1
  fi
  if (( ${#targets[@]} != expected_ranks )); then
    echo "ERROR expected exactly 8 DecodeCudaGraphRunner traces, found ${#targets[@]}" >&2
    exit 1
  fi
else
  targets=("$trace_dir"/rank_*/capture_traces/bs_{2,64}_q_1_rank*.json.gz)
  if (( ${#targets[@]} != 16 )); then
    echo "ERROR expected exactly 16 ATOM BS2/BS64 q1 traces, found ${#targets[@]}" >&2
    exit 1
  fi
fi

declare -A target_keys=()
for trace in "${targets[@]}"; do
  gzip -t "$trace"
  size=$(stat -c %s "$trace")
  if (( size > max_trace_bytes )); then
    echo "ERROR trace exceeds 500 MiB: $trace ($size bytes)" >&2
    exit 1
  fi
  base=$(basename "$trace")
  if [[ "$engine" == sglang &&
    "$base" =~ ^cuda_graph_capture-DecodeCudaGraphRunner-TP-([0-7])\.json\.gz$ ]]; then
    key="rank-${BASH_REMATCH[1]}"
  elif [[ "$engine" == atom &&
    "$base" =~ ^bs_(2|64)_q_1_rank([0-7])\.json\.gz$ ]]; then
    key="bs-${BASH_REMATCH[1]}-rank-${BASH_REMATCH[2]}"
  else
    echo "ERROR unexpected target trace name: $trace" >&2
    exit 1
  fi
  if [[ -n "${target_keys[$key]:-}" ]]; then
    echo "ERROR duplicate target trace key $key: $trace" >&2
    exit 1
  fi
  target_keys[$key]=$trace
done
for rank in $(seq 0 7); do
  if [[ "$engine" == sglang ]]; then
    required_keys=("rank-$rank")
  else
    required_keys=("bs-2-rank-$rank" "bs-64-rank-$rank")
  fi
  for key in "${required_keys[@]}"; do
    if [[ -z "${target_keys[$key]:-}" ]]; then
      echo "ERROR missing target trace $key" >&2
      exit 1
    fi
  done
done
echo "TRACE_EXPORT_OK engine=$engine targets=${#targets[@]} max_mib=500"

: >"$output_dir/analysis/analyze.log"
for trace in "${targets[@]}"; do
  base=$(basename "$trace" .json.gz)
  summary="$output_dir/analysis/$base-summary.json"
  python3 "$analyzer" "$trace" --output "$summary" \
    >>"$output_dir/analysis/analyze.log" 2>&1
  python3 - "$summary" "$engine" "$base" <<'PY'
import json
import sys

summary_path, engine, label = sys.argv[1:]
summary = json.load(open(summary_path))
gates = summary["validation_gates"]
counts = gates["category_counts"]
required = {
    "gpu_kernel": gates["has_kernel"],
    "cpu_op": gates["has_cpu_op"],
    "annotation": gates["has_annotation"],
}
if engine == "atom":
    required.update(
        {
            "runtime": gates["has_host_api"],
            "correlation": gates["has_correlation_or_flow"],
        }
    )
failed = [name for name, passed in required.items() if not passed]
if failed:
    raise SystemExit(f"{label} trace gates failed: {failed}; counts={counts}")
print(
    f"TRACE_GATES_OK label={label} kernels={counts['kernel']} "
    f"cpu_op={counts['cpu_op']} annotation="
    f"{counts['user_annotation'] + counts['gpu_user_annotation']} "
    f"runtime={counts['cuda_runtime'] + counts['cuda_driver']} "
    f"correlation={int(gates['has_correlation_or_flow'])}"
)
PY
done

python3 - "$output_dir/trace-inventory.json" "$output_dir" "$engine" \
  "${targets[@]}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

inventory_path = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
engine = sys.argv[3]
targets = {Path(value).resolve() for value in sys.argv[4:]}

def describe(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }

all_files = {
    path.resolve()
    for path in root.rglob("*")
    if path.is_file() and path.resolve() != inventory_path
}
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "engine": engine,
    "target_contract": (
        "8 combined DecodeCudaGraphRunner rank traces"
        if engine == "sglang"
        else "BS2/q1 and BS64/q1 for all 8 ranks (16 traces)"
    ),
    "target_count": len(targets),
    "target_traces": [describe(path) for path in sorted(targets)],
    "additional_artifacts": [
        describe(path) for path in sorted(all_files - targets)
    ],
}
inventory_path.write_text(json.dumps(payload, indent=2) + "\n")
print(
    f"INVENTORY_OK targets={len(targets)} "
    f"additional={len(all_files - targets)}"
)
PY

echo "CAPTURE_COMPLETE engine=$engine output_dir=$output_dir"
