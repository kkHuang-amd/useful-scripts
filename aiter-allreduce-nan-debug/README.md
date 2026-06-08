# aiter custom all-reduce NaN / GPU ASSERT_TRAP debug toolkit

Scripts and instrumentation used to root-cause an intermittent decode crash in
SGLang P/D-disaggregated serving on AMD ROCm:

> Decode dies with `HSA_STATUS_ERROR_EXCEPTION 0x1016` from
> `at::native::_assert_async_cuda_kernel<bool>` during sampling. Root cause:
> aiter's `cross_device_reduce_1stage` all-reduce kernel is missing its exit
> `end_sync` barrier, so under a captured CUDA graph a fast rank's reused input
> slot is overwritten while a slow rank still reads it via IPC → NaN in the AR
> output → trips a torch device assert in the sampler.
> Fixed upstream by **ROCm/aiter PR #3514** (issue #3515).

Full write-up: `ROOT_CAUSE_aiter_custom_all_reduce_nan.md`.
Methodology skill: `kkHuang-amd/claude-skills` → `aiter-custom-allreduce-nan-crash`.

Validated on: DeepSeek-R1-0528-MXFP4, TP=8, 1P1D no-UMBP, MI355X (gfx950), SGLang+aiter.

## TL;DR workaround (no rebuild)

```
SGLANG_USE_AITER_AR=0          # use sglang-native custom AR  (validated 12/12 clean)
# or
--disable-custom-all-reduce    # use RCCL                     (validated 12/12 clean)
```

## Files

| File | Purpose |
|---|---|
| `common.sh` | Shared config (nodes, ports, model, IPs) + `node_ip()` resolver. Sourced by step1-4 / repro_loop. **Edit node hostnames here first.** |
| `step1.sh` | Launch no-UMBP **prefill** SGLang (run ON the prefill node) |
| `step2.sh` | Launch no-UMBP **decode** SGLang (run ON the decode node) |
| `step3.sh` | Launch **mori-sched router** (run ON the prefill node, after 1+2 healthy) |
| `step4.sh` | Run AgentX v0.3 trace replay against the router (run ON the bench host) |
| `repro_loop.sh` | Re-run step4 until a worker `/health` fails (auto-detects the crash). `REPRO_RUNS`, `BENCH_CONC`, `BENCH_DURATION`, `SKIP_DEP_INSTALL` |
| `dbg_helpers.sh` | `source` it → `dbg_status/health/logs/proc/inspect/shell/gpu/dmesg/amdgpu/cores/gdb_latest/cleanup` |
| `setup_coredumps.sh` / `restore_coredumps.sh` | Enable/restore kernel core dumps on a node (local `/var/tmp/cores`, avoids NFS root_squash 0-byte cores) |
| `wait_gpu_free.sh` | Poll a node until a named container is gone and GPUs are idle |
| `run_benchmark_container.sh` | Launch the bench client container |
| `aiter_ar_test.py` | Standalone 8-rank correctness test for aiter `custom_all_reduce` vs RCCL (eager + cuda-graph), with an `AR_IMPL=aiter|sglang` control |
| `aiter_fused_ar_rms_test.py` | Standalone correctness test for aiter `custom_fused_ar_rms` (fused AR+RMSNorm, 1stage/2stage) |
| `ar_nan_patch/sitecustomize.py` | `AR_NAN_CHECK=1` monkeypatch: log the first aiter AR call that outputs NaN/Inf + whether input was already bad (see pitfalls below) |
| `ROOT_CAUSE_aiter_custom_all_reduce_nan.md` | Full investigation report |

> NOTE: launching prefill/decode/router assumes the `mori-scheduler`
> `scripts/multi_node/launch_*_no_umbp.sh` scripts with the debug hooks added in
> this investigation (kkHuang-amd fork branch). Those hooks:
> `EXTRA_SGLANG_ARGS`, `SGLANG_USE_AITER_AR`, `LOG_LEVEL`, rocm-debug-agent
> (`HSA_TOOLS_LIB`/`HSA_ENABLE_DEBUG` + `ROCM_DEBUG_SAVE_CODE_OBJECTS`),
> `DECODE_CORE_LIMIT_KB`, `AR_DEBUG_PATCH_DIR`/`AR_NAN_CHECK`, and `AITER_PATCH_SO`.

## Typical flow

```bash
# 0. edit common.sh: PREFILL_NODE / DECODE_NODE / WORKSPACE_DIR / MODEL_NAME
# 1. enable core dumps on the nodes you'll use (one-time per boot)
for h in <prefill_node> <decode_node>; do ssh $h bash $PWD/setup_coredumps.sh; done

# 2. bring up 1P1D  (run each ON the right node)
ssh <prefill_node> "bash $PWD/step1.sh"
ssh <decode_node>  "bash $PWD/step2.sh"
ssh <prefill_node> "bash $PWD/step3.sh"

# 3. reproduce: loop the benchmark until a worker dies
ssh <bench_host> "REPRO_RUNS=12 SKIP_DEP_INSTALL=1 bash $PWD/repro_loop.sh"

# 4. after a crash, diagnose
source dbg_helpers.sh
dbg_status; dbg_logs decode 500; dbg_amdgpu decode; dbg_cores decode
```

## Bisection matrix (localizes the bug fast)

Run `repro_loop.sh` under each; the one that stops crashing tells you the layer:

| Variant | Decode launch flag/env |
|---|---|
| no custom AR (RCCL) | `EXTRA_SGLANG_ARGS=--disable-custom-all-reduce` |
| sglang-native AR | `SGLANG_USE_AITER_AR=0` |
| fused AR+RMSNorm | `EXTRA_SGLANG_ARGS=--enable-aiter-allreduce-fusion` |
| no cuda graph | `EXTRA_SGLANG_ARGS=--disable-cuda-graph` |

## NaN-source instrumentation — and its pitfalls

```bash
# decode launch with:
AR_DEBUG_PATCH_DIR=<dir containing ar_nan_patch/> AR_NAN_CHECK=1 EXTRA_SGLANG_ARGS=--disable-cuda-graph
```

- The Python wrapper is **blind under CUDA graph** (sync illegal during capture;
  Python not run during replay) — must use `--disable-cuda-graph` to observe.
- It only sees the **local rank's input**; a multi-rank reduce can output NaN with
  a clean local input when a *peer's* input is already NaN. `INPUT_ALREADY_BAD=
  False` ≠ "this AR is the source".
- To truly pin a 1stage source, instrument device-side or use a deterministic
  probe (each rank writes `1.0`; correct AR sum = world_size).

## Standalone AR correctness tests (no full model)

```bash
torchrun --nproc_per_node=8 aiter_ar_test.py            # AR_IMPL=aiter|sglang, USE_GRAPH=0|1, MAG=...
torchrun --nproc_per_node=8 aiter_fused_ar_rms_test.py  # STAGE=auto|1|2, MAG=...
```
NOTE: aiter AR is numerically correct in isolation — the bug needs the real
concurrent workload (cuda graph + graph_pool slot reuse), so these are mainly
to *rule out* simple math/precision errors.

## Rebuild aiter with a kernel patch + bind-mount (no new image)

```bash
# in a throwaway container with the same image (hipcc compile is CPU-only):
#  1. edit /sgl-workspace/aiter/csrc/include/custom_all_reduce.cuh  (e.g. PR #3514: add
#     `end_sync<ngpus, true>(sg, self_sg, rank);` at the end of cross_device_reduce_1stage)
#  2. ln -sfn /sgl-workspace/aiter /sgl-workspace/aiter/aiter_meta
#     mkdir -p /sgl-workspace/aiter/aiter/jit/build/module_custom_all_reduce/blob
#  3. cd /sgl-workspace/aiter/aiter/jit/build/module_custom_all_reduce/build
#     rm -f custom_all_reduce.cuda.o module_custom_all_reduce.so && ninja
#  4. cp module_custom_all_reduce.so /sgl-workspace/aiter/aiter/jit/module_custom_all_reduce.so
#  5. docker cp it out, then launch decode with:
#     AITER_PATCH_SO=<patched.so>   (bind-mounts it over the image's module, ro)
#  6. verify: docker inspect mount + roc-obj-ls/llvm-objdump diff the code object
```
```
buffer_inv sc1        -> DEVICE-scope acquire (original)
buffer_inv sc0 sc1    -> SYSTEM-scope (proof a scope patch reached the binary)
```
