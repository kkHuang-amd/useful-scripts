#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_common_prefill_case.sh ENGINE CONCURRENCY CASE_DIR MANIFEST

Run one production Kimi-K3 prefill-step profile case.

Arguments:
  ENGINE       sglang or atom
  CONCURRENCY  2 or 64
  CASE_DIR     output directory; it must not already exist
  MANIFEST     persisted common-client prompt manifest (.jsonl.gz)

Optional environment:
  PREFILL_PROFILE_SECONDS  Profile duration in seconds (default: 2)

The runner starts a production-configured TP8 server with FULL decode graphs,
starts profiling immediately before one exact 8192/2 manifest wave, stops
profiling while the wave is active, validates eight rank traces, analyzes each
full trace, and waits for VRAM release. It does not change attention backends
or enable eager decode.
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
case "$engine" in sglang | atom) ;; *) echo "ERROR invalid engine: $engine" >&2; exit 2 ;; esac
case "$concurrency" in 2 | 64) ;; *) echo "ERROR invalid concurrency: $concurrency" >&2; exit 2 ;; esac
profile_seconds=${PREFILL_PROFILE_SECONDS:-2}
if [[ ! "$profile_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
  ! python3 -c 'import sys; raise SystemExit(float(sys.argv[1]) <= 0)' "$profile_seconds"; then
  echo "ERROR PREFILL_PROFILE_SECONDS must be greater than zero" >&2
  exit 2
fi
if [[ ! -f "$manifest" ]]; then
  echo "ERROR manifest does not exist: $manifest" >&2
  exit 2
fi
if [[ -e "$case_dir" ]]; then
  echo "ERROR CASE_DIR must not already exist: $case_dir" >&2
  exit 2
fi

model=/shared_nfs/models/Kimi-K3
client=/workspace/useful-scripts/benchmarking/common_oai_benchmark.py
analyzer=/workspace/useful-scripts/benchmarking/kimi-k3/analysis/analyze_chrome_trace.py
runtime_dump=/workspace/useful-scripts/benchmarking/kimi-k3/server/dump_runtime_state.sh
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

mkdir -p "$case_dir/prefill-traces" "$case_dir/analysis" "$case_dir/runtime"
case_dir=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$case_dir")
manifest=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$manifest")
trace_dir="$case_dir/prefill-traces"
server_log="$case_dir/server.log"
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
    kill -KILL -- "-$pid" 2>/dev/null || true
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
  local before=$1 current="$case_dir/gpu-after.json"
  if ! command -v amd-smi >/dev/null 2>&1; then
    echo "GPU_RELEASE_SKIP amd-smi_unavailable"
    return 0
  fi
  for _ in $(seq 1 180); do
    if timeout 10 amd-smi metric --mem-usage --json >"$current.tmp" 2>>"$case_dir/gpu-after.log" &&
      python3 - "$before" "$current.tmp" <<'PY'
import json, sys
def used(path):
    return {int(x["gpu"]): float(x["mem_usage"]["used_vram"]["value"])
            for x in json.load(open(path))["gpu_data"]}
before, after = used(sys.argv[1]), used(sys.argv[2])
raise SystemExit(0 if before.keys() == after.keys() and
                 all(after[g] <= before[g] + 512 for g in before) else 1)
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
  stop_group "$client_pid"
  stop_group "$server_pid"
  if (( server_started == 1 && gpu_release_checked == 0 )) &&
    [[ -f "$case_dir/gpu-before.json" ]]; then
    wait_for_gpu_release "$case_dir/gpu-before.json" || true
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

OUTPUT_DIR="$case_dir/runtime" BASE_URL=http://127.0.0.1:1 \
  SGLANG_REPO="$sglang_root" AITER_REPO="$aiter_root" \
  bash "$runtime_dump" >"$case_dir/runtime-dump.log"
snapshot_gpu_memory "$case_dir/gpu-before.json"

PYTHONPATH="$aiter_root:$sglang_root/python:$atom_root" \
  python3 - "$case_dir/runtime-metadata.json" "$manifest" "$engine" \
  "$concurrency" "$profile_seconds" "$expected_aiter_sha" "$expected_flydsl" <<'PY'
import gzip, hashlib, importlib.metadata, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import aiter, flydsl, torch

out, manifest, engine, concurrency, seconds, expected_sha, expected_flydsl = sys.argv[1:]
roots = {"sglang": "/sgl-workspace/sglang-k3-triton37",
         "atom": "/sgl-workspace/ATOM",
         "aiter": "/sgl-workspace/aiter-atom-current"}
git = lambda *args: subprocess.check_output(["git", *args], text=True).strip()
shas = {name: git("-C", path, "rev-parse", "HEAD") for name, path in roots.items()}
if shas["aiter"] != expected_sha:
    raise SystemExit(f"AITER SHA is {shas['aiter']}, expected {expected_sha}")
aiter_file = str(Path(aiter.__file__).resolve())
if not aiter_file.startswith(str(Path(roots["aiter"]).resolve()) + os.sep):
    raise SystemExit(f"wrong AITER import: {aiter_file}")
flydsl_version = getattr(flydsl, "__version__", None) or importlib.metadata.version("flydsl")
if flydsl_version != expected_flydsl:
    raise SystemExit(f"FlyDSL version is {flydsl_version}, expected {expected_flydsl}")

file_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
logical_hash = hashlib.sha256()
with gzip.open(manifest, "rt") as handle:
    entries = [json.loads(line) for line in handle]
required = int(concurrency)
if len(entries) < required:
    raise SystemExit(f"manifest has {len(entries)} prompts, needs {required}")
for index, entry in enumerate(entries[:required]):
    logical_hash.update(entry["prompt"].encode())
    if entry.get("request_id") != index or entry.get("prompt_tokens") != 8192:
        raise SystemExit(f"manifest contract failed at index {index}: {entry}")

payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "engine": engine,
    "concurrency": required,
    "model": "/shared_nfs/models/Kimi-K3",
    "config": {"tp_size": 8, "kv_cache_dtype": "fp8", "graphs": "full",
               "input_len": 8192, "output_len": 2, "warmups": 0,
               "profile_seconds": float(seconds),
               "profile_activities": ["CPU", "GPU"],
               "profile_with_stack": False, "profile_record_shapes": False,
               "prefill_backend": "aiter", "decode_backend": "triton"},
    "git": {name: {"path": roots[name], "sha": sha,
                   "status_short": git("-C", roots[name], "status", "--short").splitlines()}
            for name, sha in shas.items()},
    "runtime": {"python": sys.version, "platform": platform.platform(),
                "torch": torch.__version__, "torch_hip": torch.version.hip,
                "flydsl": flydsl_version, "aiter_file": aiter_file},
    "manifest": {"path": str(Path(manifest).resolve()), "file_sha256": file_hash,
                 "selected_logical_prompt_sha256": logical_hash.hexdigest(),
                 "count": len(entries),
                 "selected_request_ids": [x["request_id"] for x in entries[:required]]},
}
Path(out).write_text(json.dumps(payload, indent=2) + "\n")
print(f"RUNTIME_OK engine={engine} c={required} aiter={shas['aiter'][:8]} flydsl={flydsl_version}")
PY

if [[ "$engine" == sglang ]]; then
  port=30000
  setsid env -u KINETO_CONFIG PYTHONUNBUFFERED=1 \
    PYTHONPATH="$sglang_root/python:$aiter_root" \
    AITER_JIT_DIR=/tmp/aiter-jit-common-oai-sglang-same-aiter \
    SGLANG_CACHE_DIR=/tmp/sglang-cache-common-oai-sglang-same-aiter \
    SGLANG_K3_FLYDSL_SOURCE=sglang SGLANG_K3_AITER_M16384_PROFILE=1 \
    SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 \
    AITER_SITUV2_A8W4=1 AITER_SITUV2_A4W4=0 \
    AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1 SGLANG_K3_FLYDSL_AR_NORM=1 \
    SGLANG_K3_KDA_FUSED_BACKEND=aiter SGLANG_K3_AITER_MLA_GATE=1 \
    SGLANG_K3_AITER_KDA_GROUP64=1 SGLANG_K3_AITER_MOE_PREROUTE_FP8=1 \
    SGLANG_K3_PREROUTE_PREACTIVATED_SHARED=1 SGLANG_K3_AITER_LATENT_TAIL_FP8=0 \
    SGLANG_K3_AITER_B2_FUSIONS=1 SGLANG_K3_AITER_MLA_Q_CACHE_FUSION=1 \
    SGLANG_K3_RADIX4_TOPK=1 SGLANG_AITER_FP8_PREFILL_ATTN=0 \
    SGLANG_MLA_DECODE_TUNE=1 SGLANG_TRITON_37_EXTEND_LQ576_N32=0 \
    SGLANG_ROCM_K3_FUSE_KDA_INPROJ=1 SGLANG_K3_AITER_TUNED_MOE_FRONT=1 \
    SGLANG_K3_AITER_TUNED_MOE_FRONT_MIN_TOKENS=48 \
    SGLANG_K3_AITER_TUNED_MOE_FRONT_MAX_TOKENS=192 \
    SGLANG_K3_MOE_LATENT_MXFP4=0 SGLANG_K3_MOE_LATENT_DOWN_MXFP4=0 \
    SGLANG_K3_MOE_LATENT_UP_MXFP4=1 SGLANG_K3_MOE_LATENT_MXFP4_MIN_TOKENS=2048 \
    SGLANG_PROFILE_V2=0 SGLANG_PROFILE_WITH_STACK=false \
    SGLANG_PROFILE_RECORD_SHAPES=false \
    python -m sglang.launch_server --model-path "$model" --trust-remote-code \
      --host 127.0.0.1 --port "$port" --tp-size 8 --kv-cache-dtype fp8_e4m3 \
      --mem-fraction-static 0.85 --max-running-requests 256 \
      --chunked-prefill-size 16384 --max-prefill-tokens 16384 \
      --disable-radix-cache --attention-backend triton \
      --prefill-attention-backend aiter --decode-attention-backend triton \
      --sampling-backend pytorch --cuda-graph-max-bs-decode 256 \
      >"$server_log" 2>&1 &
else
  port=8000
  setsid env -u KINETO_CONFIG PYTHONUNBUFFERED=1 \
    PYTHONPATH="$aiter_root:$atom_root" ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0 \
    ATOM_PROFILER_MORE=0 ATOM_ENABLE_DETAILED_ANNOTATION=1 ATOM_PROFILER_TIMEOUT=600 \
    python -m atom.entrypoints.openai_server --model "$model" \
      --kv_cache_dtype fp8 -tp 8 --trust-remote-code --max-model-len 16384 \
      --max-num-seqs 64 --max-num-batched-tokens 16384 \
      --gpu-memory-utilization 0.93 --block-size 128 --no-enable_prefix_caching \
      --torch-profiler-dir "$trace_dir" --online_quant_config \
      '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*self_attn.[qkv]_conv1d*","*block_sparse_moe.experts*","*block_sparse_moe.routed_expert_*","*vision_tower*","*mm_projector*"]}' \
      >"$server_log" 2>&1 &
fi
server_pid=$!
server_started=1

ready=0
[[ "$engine" == sglang ]] && ready_checks=360 || ready_checks=180
for _ in $(seq 1 "$ready_checks"); do
  if [[ "$engine" == sglang ]] && rg -q "ready to roll" "$server_log" 2>/dev/null; then
    ready=1; break
  elif [[ "$engine" == atom ]] && curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    ready=1; break
  fi
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 5
done
if [[ "$ready" != 1 ]]; then
  echo "ERROR server did not become ready" >&2
  rg "Initialization failed|Traceback|ERROR|RuntimeError|Killed|OutOfMemory" "$server_log" || true
  exit 1
fi

if [[ "$engine" == sglang ]]; then
  (( $(rg -c "Capture target decode CUDA graph begin\. backend=full" "$server_log" || true) >= expected_ranks )) ||
    { echo "ERROR SGLang did not capture FULL graphs on all ranks" >&2; exit 1; }
  rg "server_args=|Capture target decode CUDA graph (begin|end)" "$server_log" >"$case_dir/effective-args.log"
  rg -q "prefill_attention_backend='aiter'" "$case_dir/effective-args.log" &&
    rg -q "decode_attention_backend='triton'" "$case_dir/effective-args.log" ||
    { echo "ERROR SGLang effective production attention backends not verified" >&2; exit 1; }
else
  rg -q "'enforce_eager': False" "$server_log" ||
    { echo "ERROR ATOM effective args do not show enforce_eager=False" >&2; exit 1; }
  python3 - "$server_log" "$concurrency" <<'PY'
import re, sys
text, expected = open(sys.argv[1], errors="replace").read(), int(sys.argv[2])
sizes = set()
for line in text.splitlines():
    match = re.search(r"cudagraph capture\[([0-9, ]+)\]", line)
    if match:
        sizes.update(int(x) for x in match.group(1).split(","))
raise SystemExit(0 if expected in sizes else 1)
PY
  rg "Engine kwargs:|cudagraph capture sizes|Engine Core: cudagraph capture" \
    "$server_log" >"$case_dir/effective-args.log"
fi
echo "READY engine=$engine graph=full port=$port"

if compgen -G "$trace_dir/**/*.json*" >/dev/null; then
  echo "ERROR profiler produced traces before the measured wave" >&2
  exit 1
fi

client_dir="$case_dir/prefill-client"
client_log="$case_dir/prefill-client.log"
args=(
  "$client" --base-url "http://127.0.0.1:$port" --model "$model"
  --tokenizer "$model" --trust-remote-code --input-len 8192 --output-len 2
  --num-prompts "$concurrency" --warmup-requests 0
  --max-concurrency "$concurrency" --seed 42 --prompt-manifest "$manifest"
  --output-dir "$client_dir" --profile-before-wave --profile-engine "$engine"
  --profile-seconds "$profile_seconds" --profile-stop-after-wave
  --profile-base-url "http://127.0.0.1:$port"
)
[[ "$engine" == sglang ]] && args+=(--profile-output-dir "$trace_dir")
setsid python3 "${args[@]}" >"$client_log" 2>&1 &
client_pid=$!
if wait "$client_pid"; then
  status=0
else
  status=$?
fi
client_pid=
if (( status != 0 )); then
  echo "ERROR prefill client/profile control failed; see $client_log" >&2
  exit "$status"
fi

python3 - "$client_dir/summary.json" "$case_dir/profile-http-results.json" \
  "$concurrency" "$profile_seconds" <<'PY'
import json, sys
from pathlib import Path
summary = json.load(open(sys.argv[1]))
expected, seconds = int(sys.argv[3]), float(sys.argv[4])
if summary["successful_requests"] != expected or summary["failed_requests"] != 0:
    raise SystemExit(f"request contract failed: {summary}")
config = summary["config"]
for key, value in {"input_len": 8192, "output_len": 2, "num_prompts": expected,
                   "warmup_requests": 0, "max_concurrency": expected}.items():
    if config[key] != value:
        raise SystemExit(f"client config mismatch for {key}: {config[key]} != {value}")
profile = summary.get("profile")
if not profile:
    raise SystemExit("summary has no profile result")
result = profile["result"]
if not result.get("triggered") or not result.get("success"):
    raise SystemExit(f"profile control failed: {result}")
if result.get("start_http_status") != 200 or result.get("stop_http_status") != 200:
    raise SystemExit(f"profile HTTP status failed: {result}")
if result.get("profile_window_elapsed_s", 0) < seconds:
    raise SystemExit(f"profile window was short: {result}")
Path(sys.argv[2]).write_text(json.dumps(profile, indent=2) + "\n")
print(
    f"PREFILL_CLIENT_OK requests={expected} output_len=2 "
    f"profile_seconds={seconds} stop_while_wave_active="
    f"{bool(result.get('stop_while_wave_active'))}"
)
PY

for _ in $(seq 1 600); do
  if [[ "$engine" == sglang ]]; then
    traces=("$trace_dir"/*.trace.json.gz)
  else
    traces=("$trace_dir"/rank_*/*.pt.trace.json.gz)
  fi
  (( ${#traces[@]} >= expected_ranks )) && break
  sleep 1
done
[[ "$engine" == sglang ]] && traces=("$trace_dir"/*.trace.json.gz) ||
  traces=("$trace_dir"/rank_*/*.pt.trace.json.gz)
all_gzip=("$trace_dir"/**/*.gz)
orphans=("$trace_dir"/**/*.json)
(( ${#all_gzip[@]} == expected_ranks && ${#traces[@]} == expected_ranks && ${#orphans[@]} == 0 )) ||
  { echo "ERROR expected exactly 8 compressed rank traces and no orphan JSON" >&2; exit 1; }

declare -A ranks_seen=()
for trace in "${traces[@]}"; do
  gzip -t "$trace"
  (( $(stat -c %s "$trace") <= max_trace_bytes )) ||
    { echo "ERROR trace exceeds 500 MiB: $trace" >&2; exit 1; }
  if [[ "$engine" == sglang && "$(basename "$trace")" =~ TP-([0-7]) ]]; then
    rank=${BASH_REMATCH[1]}
  elif [[ "$engine" == atom && "$(basename "$(dirname "$trace")")" =~ ^rank_([0-7])$ ]]; then
    rank=${BASH_REMATCH[1]}
  else
    echo "ERROR cannot identify rank for $trace" >&2; exit 1
  fi
  [[ -z "${ranks_seen[$rank]:-}" ]] || { echo "ERROR duplicate rank $rank" >&2; exit 1; }
  ranks_seen[$rank]=$trace
done
for rank in $(seq 0 7); do
  [[ -n "${ranks_seen[$rank]:-}" ]] || { echo "ERROR missing rank $rank" >&2; exit 1; }
done
echo "TRACE_EXPORT_OK traces=8 max_mib=500"

stop_group "$server_pid"
server_pid=
gpu_release_checked=1
wait_for_gpu_release "$case_dir/gpu-before.json"
if rg -q "OutOfMemory|out of memory|disconnected|worker.*(died|failed)|Retract(ing|ed) [1-9]" "$server_log"; then
  echo "ERROR server log contains OOM, retraction, disconnect, or worker failure" >&2
  exit 1
fi

: >"$case_dir/analysis/analyze.log"
for rank in $(seq 0 7); do
  trace=${ranks_seen[$rank]}
  summary="$case_dir/analysis/rank-${rank}-summary.json"
  caller_map="$case_dir/analysis/rank-${rank}-caller-map.jsonl"
  python3 "$analyzer" "$trace" --output "$summary" \
    --emit-caller-map "$caller_map" >>"$case_dir/analysis/analyze.log" 2>&1
  python3 - "$summary" "$rank" "$engine" <<'PY'
import json, sys
summary, rank, engine = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
gates = summary["validation_gates"]
required = {name: gates[name] for name in
            ("has_kernel", "has_cpu_op", "has_host_api", "has_annotation",
             "has_correlation_or_flow")}
names = [item["name"] for item in summary["top_annotations"]]
if engine == "sglang":
    prefill = any(name.startswith("step[EXTEND") or
                  "sglang.vlm.language_model_prefill" in name for name in names)
else:
    prefill = any(name.lower().startswith("prefill[") for name in names)
required["prefill_annotation"] = prefill
failed = [name for name, passed in required.items() if not passed]
if failed:
    raise SystemExit(f"rank {rank} trace gates failed: {failed}; "
                     f"counts={gates['category_counts']}")
print(f"TRACE_GATES_OK rank={rank} kernels={summary['kernel_count']} "
      f"prefill_annotation=1")
PY
done

python3 - "$case_dir/trace-inventory.json" "${traces[@]}" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
items = [{"path": str(Path(x).resolve()), "size_bytes": Path(x).stat().st_size,
          "sha256": sha(x)} for x in sys.argv[2:]]
Path(sys.argv[1]).write_text(json.dumps(
    {"created_at": datetime.now(timezone.utc).isoformat(), "count": len(items),
     "traces": sorted(items, key=lambda x: x["path"])}, indent=2) + "\n")
PY

python3 - "$case_dir" <<'PY'
import hashlib, os, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for base, _, files in os.walk(root):
    for name in files:
        path = Path(base) / name
        if path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(f"{digest.hexdigest()}  {path.relative_to(root)}")
(root / "SHA256SUMS").write_text("\n".join(sorted(rows)) + "\n")
print(f"INVENTORY_OK files={len(rows)} traces=8")
PY

echo "CASE_COMPLETE engine=$engine concurrency=$concurrency case_dir=$case_dir"
