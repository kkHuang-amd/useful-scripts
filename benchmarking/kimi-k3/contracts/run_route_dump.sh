#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_route_dump.sh ENGINE OUTPUT_DIR MANIFEST

Run one eager-only Kimi-K3 current-route contract diagnostic.

Arguments:
  ENGINE      sglang or atom
  OUTPUT_DIR  fresh directory for retained logs, metadata, client output, dumps
  MANIFEST    common C64 prompt manifest (.jsonl.gz; first 64 are exact 8192 tokens)

The server uses the current AITER/FlyDSL and the production model, precision,
attention, and MoE flags from the matched common-client campaign. Execution is
explicitly eager only for this diagnostic: SGLang uses --disable-cuda-graph;
ATOM uses --enforce-eager and ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0.

The single measured contract wave is fixed at input 8192, output 128,
concurrency 64, 64 prompts, seed 42, and zero warmups. This script does not
produce or claim valid eager timing comparisons.

The route wrapper remains disarmed throughout startup. The runner atomically
creates K3_ROUTE_DUMP_ARM_FILE immediately before the client, then removes the
active path and retains a timestamped route-arm-record.json after the client.

Batch example:
  run_route_dump.sh sglang /runs/routes/sglang "$MANIFEST"
  run_route_dump.sh atom   /runs/routes/atom   "$MANIFEST"
  python analyze_route_dumps.py /runs/routes/sglang/route-dumps \
    /runs/routes/atom/route-dumps --output-dir /runs/routes/analysis
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 3 ]]; then
  usage >&2
  exit 2
fi

engine=$1
output_dir=$2
manifest=$3
case "$engine" in
  sglang | atom) ;;
  *)
    echo "ERROR invalid engine: $engine (expected sglang or atom)" >&2
    exit 2
    ;;
esac
if [[ ! -f "$manifest" ]]; then
  echo "ERROR manifest does not exist: $manifest" >&2
  exit 2
fi
if [[ -e "$output_dir" ]]; then
  echo "ERROR OUTPUT_DIR already exists; evidence is never overwritten: $output_dir" >&2
  exit 2
fi

model=/shared_nfs/models/Kimi-K3
client=/workspace/useful-scripts/benchmarking/common_oai_benchmark.py
contracts=/workspace/useful-scripts/benchmarking/kimi-k3/contracts
runtime_dump=/workspace/useful-scripts/benchmarking/kimi-k3/server/dump_runtime_state.sh
sglang_root=/sgl-workspace/sglang-k3-triton37
atom_root=/sgl-workspace/ATOM
aiter_root=/sgl-workspace/aiter-atom-current
expected_aiter_sha=dc4bdf1c142181ad90b7f6948564126df4c05fde
expected_flydsl=0.3.1
expected_ranks=8
expected_moe_layers=92
concurrency=64
output_len=128

mkdir -p "$output_dir/route-dumps" "$output_dir/runtime"
output_dir=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$output_dir")
manifest=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$manifest")
route_dir="$output_dir/route-dumps"
arm_file="$output_dir/route-dump.arm"
arm_record="$output_dir/route-arm-record.json"
server_log="$output_dir/server.log"
client_dir="$output_dir/client"
client_log="$output_dir/client.log"
server_pid=
client_pid=
server_started=0
gpu_release_checked=0

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

disarm_route_dump() {
  local reason=${1:-cleanup}
  if [[ -e "$arm_file" ]]; then
    mv "$arm_file" "$arm_record"
    python3 - "$arm_record" "$reason" <<'PY'
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload.update({
    "armed": False,
    "was_armed": True,
    "disarmed_reason": sys.argv[2],
    "disarmed_at_utc": datetime.now(timezone.utc).isoformat(),
    "disarmed_time_ns": time.time_ns(),
    "disarmed_monotonic_ns": time.monotonic_ns(),
})
fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
    echo "ROUTE_DUMP_DISARMED reason=$reason record=$arm_record"
  fi
}

snapshot_gpu_memory() {
  local destination=$1
  if command -v amd-smi >/dev/null 2>&1; then
    timeout 10 amd-smi metric --mem-usage --json \
      >"$destination" 2>"${destination%.json}.log"
  else
    printf '{"status":"amd-smi-unavailable"}\n' >"$destination"
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
  disarm_route_dump "cleanup-status-$status" || true
  stop_group "$client_pid"
  client_pid=
  stop_group "$server_pid"
  server_pid=
  if (( server_started == 1 && gpu_release_checked == 0 )) &&
    [[ -f "$output_dir/gpu-before.json" ]]; then
    wait_for_gpu_release "$output_dir/gpu-before.json" || true
  fi
  rm -rf /tmp/aiter-jit-k3-route-dump-"$engine" \
    /tmp/sglang-cache-k3-route-dump-"$engine"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

snapshot_gpu_memory "$output_dir/gpu-before.json"

env -u K3_ROUTE_DUMP_DIR -u K3_ROUTE_DUMP_ARM_FILE \
  PYTHONPATH="$aiter_root:$sglang_root/python:$atom_root" \
  python3 - "$output_dir/contract-metadata.json" "$manifest" "$engine" \
  "$expected_aiter_sha" "$expected_flydsl" "$model/config.json" <<'PY'
import gzip
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiter
import flydsl

output, manifest, engine, expected_aiter, expected_flydsl, model_config = sys.argv[1:]
roots = {
    "sglang": "/sgl-workspace/sglang-k3-triton37",
    "atom": "/sgl-workspace/ATOM",
    "aiter": "/sgl-workspace/aiter-atom-current",
}

def git(command, path):
    return subprocess.check_output(["git", "-C", path, *command], text=True).strip()

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

entries = []
logical = hashlib.sha256()
with gzip.open(manifest, "rt") as handle:
    for line in handle:
        entry = json.loads(line)
        entries.append(entry)
        logical.update(entry["prompt"].encode())
if len(entries) < 64:
    raise SystemExit(f"manifest has {len(entries)} prompts, needs 64")
for index, entry in enumerate(entries[:64]):
    if entry.get("request_id") != index or entry.get("prompt_tokens") != 8192:
        raise SystemExit(f"manifest contract failed at entry {index}: {entry}")

shas = {name: git(["rev-parse", "HEAD"], root) for name, root in roots.items()}
if shas["aiter"] != expected_aiter:
    raise SystemExit(f"AITER SHA {shas['aiter']} != expected {expected_aiter}")
aiter_file = str(Path(aiter.__file__).resolve())
if not aiter_file.startswith(str(Path(roots["aiter"]).resolve()) + os.sep):
    raise SystemExit(f"wrong AITER import: {aiter_file}")
flydsl_version = getattr(flydsl, "__version__", None) or importlib.metadata.version("flydsl")
if flydsl_version != expected_flydsl:
    raise SystemExit(f"FlyDSL {flydsl_version} != expected {expected_flydsl}")
config = json.load(open(model_config))
hidden_layers = int(config.get("text_config", config)["num_hidden_layers"])
moe_layers = hidden_layers - 1
if moe_layers != 92:
    raise SystemExit(f"model reports {moe_layers} MoE layers, expected 92")

payload = {
    "schema": "k3-route-run-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "diagnostic_only": True,
    "timings_valid": False,
    "engine": engine,
    "execution": "eager",
    "expected_ranks": 8,
    "expected_moe_layers": moe_layers,
    "request_contract": {
        "input_len": 8192,
        "output_len": 128,
        "num_prompts": 64,
        "max_concurrency": 64,
        "warmup_requests": 0,
        "seed": 42,
    },
    "git": {
        name: {
            "root": root,
            "sha": shas[name],
            "status_short": git(["status", "--short"], root).splitlines(),
        }
        for name, root in roots.items()
    },
    "runtime": {"aiter_file": aiter_file, "flydsl": flydsl_version},
    "manifest": {
        "path": str(Path(manifest).resolve()),
        "sha256": sha256(manifest),
        "logical_prompt_sha256": logical.hexdigest(),
        "count": len(entries),
        "selected_request_ids": list(range(64)),
    },
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"CONTRACT_OK engine={engine} moe_layers={moe_layers} aiter={shas['aiter'][:8]}")
PY

if [[ "$engine" == sglang ]]; then
  port=30000
  route_mode=sglang-a8w4
  setsid env -u KINETO_CONFIG \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$contracts:$sglang_root/python:$aiter_root" \
    K3_ROUTE_DUMP_DIR="$route_dir" \
    K3_ROUTE_DUMP_ARM_FILE="$arm_file" \
    K3_ROUTE_DUMP_MAX_CALLS="$expected_moe_layers" \
    K3_ROUTE_ENV_MODE="$route_mode" \
    AITER_JIT_DIR=/tmp/aiter-jit-k3-route-dump-sglang \
    SGLANG_CACHE_DIR=/tmp/sglang-cache-k3-route-dump-sglang \
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
    python -m sglang.launch_server \
      --model-path "$model" \
      --trust-remote-code \
      --host 127.0.0.1 \
      --port "$port" \
      --tp-size 8 \
      --kv-cache-dtype fp8_e4m3 \
      --mem-fraction-static 0.85 \
      --max-running-requests 256 \
      --chunked-prefill-size 16384 \
      --max-prefill-tokens 16384 \
      --disable-radix-cache \
      --attention-backend triton \
      --prefill-attention-backend aiter \
      --decode-attention-backend triton \
      --sampling-backend pytorch \
      --disable-cuda-graph \
      >"$server_log" 2>&1 &
else
  port=8000
  route_mode=atom-a16w4
  setsid env -u KINETO_CONFIG \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$contracts:$aiter_root:$atom_root" \
    K3_ROUTE_DUMP_DIR="$route_dir" \
    K3_ROUTE_DUMP_ARM_FILE="$arm_file" \
    K3_ROUTE_DUMP_MAX_CALLS="$expected_moe_layers" \
    K3_ROUTE_ENV_MODE="$route_mode" \
    AITER_JIT_DIR=/tmp/aiter-jit-k3-route-dump-atom \
    ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0 \
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
      --enforce-eager \
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
  if [[ "$engine" == sglang ]] && rg -q "ready to roll" "$server_log" 2>/dev/null; then
    ready=1
    break
  elif [[ "$engine" == atom ]] &&
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
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
  if ! rg -q "disable_cuda_graph=True|'disable_cuda_graph': True" "$server_log"; then
    echo "ERROR SGLang effective args do not confirm eager mode" >&2
    exit 1
  fi
else
  if ! rg -q "'enforce_eager': True|enforce_eager=True" "$server_log"; then
    echo "ERROR ATOM effective args do not confirm eager mode" >&2
    exit 1
  fi
fi
printf 'engine=%s\nexecution=eager\nroute_mode=%s\noutput_len=%s\nconcurrency=%s\n' \
  "$engine" "$route_mode" "$output_len" "$concurrency" >"$output_dir/effective-contract.env"
echo "READY engine=$engine execution=eager diagnostic_only=true port=$port"

OUTPUT_DIR="$output_dir/runtime" \
  BASE_URL="http://127.0.0.1:$port" \
  SGLANG_REPO="$sglang_root" \
  AITER_REPO="$aiter_root" \
  bash "$runtime_dump" >"$output_dir/runtime-dump.log" 2>&1

python3 - "$arm_file" "$engine" <<'PY'
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    raise SystemExit(f"arm file unexpectedly exists before client: {path}")
payload = {
    "schema": "k3-route-arm-v1",
    "armed": True,
    "engine": sys.argv[2],
    "armed_at_utc": datetime.now(timezone.utc).isoformat(),
    "armed_time_ns": time.time_ns(),
    "armed_monotonic_ns": time.monotonic_ns(),
}
fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
print(
    f"ROUTE_DUMP_ARMED timestamp={payload['armed_at_utc']} "
    f"monotonic_ns={payload['armed_monotonic_ns']}"
)
PY

setsid python3 "$client" \
  --base-url "http://127.0.0.1:$port" \
  --model "$model" \
  --tokenizer "$model" \
  --trust-remote-code \
  --input-len 8192 \
  --output-len "$output_len" \
  --num-prompts "$concurrency" \
  --warmup-requests 0 \
  --max-concurrency "$concurrency" \
  --seed 42 \
  --prompt-manifest "$manifest" \
  --output-dir "$client_dir" \
  >"$client_log" 2>&1 &
client_pid=$!
if wait "$client_pid"; then
  status=0
else
  status=$?
  client_pid=
  echo "ERROR route client failed with status $status; see $client_log" >&2
  exit "$status"
fi
client_pid=
disarm_route_dump client-complete

python3 - "$client_dir/summary.json" "$concurrency" "$output_len" <<'PY'
import json
import sys

path, concurrency, output_len = sys.argv[1:]
summary = json.load(open(path))
expected = int(concurrency)
required = {
    "successful_requests": expected,
    "failed_requests": 0,
}
for key, value in required.items():
    if summary.get(key) != value:
        raise SystemExit(f"request contract failed: {key}={summary.get(key)!r}, expected {value}")
config = summary["config"]
for key, value in {
    "input_len": 8192,
    "output_len": int(output_len),
    "num_prompts": expected,
    "warmup_requests": 0,
    "max_concurrency": expected,
    "seed": 42,
}.items():
    if config.get(key) != value:
        raise SystemExit(f"request config failed: {key}={config.get(key)!r}, expected {value}")
print(f"REQUEST_CONTRACT_OK requests={expected} output_len={output_len} warmups=0")
PY

python3 - "$route_dir" "$expected_ranks" "$expected_moe_layers" \
  "$output_dir/route-inventory.json" "$arm_record" "$arm_file" <<'PY'
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

root, expected_ranks, expected_calls, output, arm_record, arm_file = sys.argv[1:]
expected_ranks = int(expected_ranks)
expected_calls = int(expected_calls)
arm = json.load(open(arm_record))
if arm.get("armed") is not False or arm.get("was_armed") is not True:
    raise SystemExit(f"arm record is not disarmed evidence: {arm}")
if "disarmed_time_ns" not in arm or "disarmed_monotonic_ns" not in arm:
    raise SystemExit(f"arm record lacks disarm timestamps: {arm}")
arm_file = str(Path(arm_file).resolve())
by_rank = defaultdict(list)
inventory = []
for path in sorted(Path(root).glob("*.json")):
    payload = json.loads(path.read_text())
    if payload.get("schema") != "k3-route-dump-v1":
        continue
    rank = payload.get("rank")
    if rank is None:
        rank = payload.get("device_index")
    rank = int(rank)
    by_rank[rank].append(payload)
    inventory.append({
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
if set(by_rank) != set(range(expected_ranks)):
    raise SystemExit(f"rank gate failed: observed {sorted(by_rank)}, expected 0..{expected_ranks - 1}")
for rank, rows in sorted(by_rank.items()):
    indices = sorted(row["call_index"] for row in rows)
    if indices != list(range(expected_calls)):
        raise SystemExit(f"rank {rank} call gate failed: count={len(rows)} indices={indices}")
    for row in rows:
        if row["topk_shape"] != [64, 16] or row["total_routes"] != 1024:
            raise SystemExit(f"rank {rank} malformed call {row['call_index']}")
        if row.get("armed") is not True:
            raise SystemExit(f"rank {rank} call {row['call_index']} is not armed")
        if row.get("arm_file") != arm_file:
            raise SystemExit(f"rank {rank} call {row['call_index']} arm path mismatch")
        if row.get("arm_time_ns") != arm["armed_time_ns"]:
            raise SystemExit(f"rank {rank} call {row['call_index']} arm time mismatch")
        if row.get("arm_monotonic_ns") != arm["armed_monotonic_ns"]:
            raise SystemExit(f"rank {rank} call {row['call_index']} arm monotonic mismatch")
        if row.get("dump_time_ns", -1) < arm["armed_time_ns"]:
            raise SystemExit(f"rank {rank} call {row['call_index']} predates arming")
        if row.get("dump_monotonic_ns", -1) < arm["armed_monotonic_ns"]:
            raise SystemExit(f"rank {rank} call {row['call_index']} monotonic predates arming")
        if row["dump_time_ns"] > arm["disarmed_time_ns"]:
            raise SystemExit(f"rank {rank} call {row['call_index']} follows disarming")
        if row["dump_monotonic_ns"] > arm["disarmed_monotonic_ns"]:
            raise SystemExit(f"rank {rank} call {row['call_index']} monotonic follows disarming")
        if not row.get("arm_timestamp_utc") or not row.get("dump_timestamp_utc"):
            raise SystemExit(f"rank {rank} call {row['call_index']} lacks UTC timestamps")
payload = {
    "schema": "k3-route-inventory-v1",
    "expected_ranks": expected_ranks,
    "expected_calls_per_rank": expected_calls,
    "arming": arm,
    "observed_calls_per_rank": {
        str(rank): len(rows) for rank, rows in sorted(by_rank.items())
    },
    "count": len(inventory),
    "files": inventory,
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"ROUTE_DUMP_GATE_OK ranks={expected_ranks} calls_per_rank={expected_calls}")
PY

stop_group "$server_pid"
server_pid=
gpu_release_checked=1
wait_for_gpu_release "$output_dir/gpu-before.json"
rg '"successful_requests"|"failed_requests"|"output_len"|"warmup_requests"' \
  "$client_dir/summary.json" || true
echo "ROUTE_DUMP_COMPLETE engine=$engine diagnostic_only=true timings_valid=false output_dir=$output_dir"
