# Root Cause: aiter `custom_all_reduce` emits NaN under concurrent decode load

**Date:** 2026-06-08
**Reporter:** wunhuang (debug session)
**Severity:** P/D disaggregated decode crashes (GPU ASSERT_TRAP) under agentic load

---

## 1. One-line summary

aiter's `custom_all_reduce` (2-stage / `use_new=True`, bf16, TP/world_size=8,
small decode batches) **intermittently produces NaN output from valid, non-NaN
input** under the real concurrent decode workload. The NaN propagates into the
PyTorch sampling backend, where a torch-internal / `torch.compile`-inserted
`_assert_async` guard traps on the GPU (`HSA_STATUS_ERROR_EXCEPTION 0x1016`),
killing the decode engine.

**Workaround (validated):** `SGLANG_USE_AITER_AR=0` (use sglang's native custom
all-reduce) or `--disable-custom-all-reduce` (use RCCL). Both are 100% stable.

---

## 2. Environment

| Item | Value |
|---|---|
| Model | DeepSeek-R1-0528-MXFP4-th (671B MoE, TP=8, EP=8) |
| Deployment | 1P1D disaggregated, no-UMBP (`scripts/multi_node/*_no_umbp.sh`) |
| Image | `rocm/pytorch-private:sglang-v0.5.12.post1-rocm720-mi35x-20260529-mori-0603-umbp-20260603-v1` |
| GPU arch | gfx950 (MI355X) |
| aiter commit | `46e6c92b3eb33f64823aaa1ff39a14586b059ef5` |
| Nodes | prefill+router = 08-21 (10.235.192.81), decode = 08-25 (10.235.192.85) |
| Workload | AgentX v0.3 trace replay (`scripts/run_trace_replay_v0.3.sh`), conc=8, dur=300 |
| KV transfer | mori |
| Sampling backend | `pytorch` (ROCm default) |

---

## 3. Crash signature

Decode engine dies intermittently (observed at benchmark run 3–6 of a repeat loop):

```
:0:rocdevice.cpp :3586:  Queue ... aborting with error :
    HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception. code: 0x1016
Fatal Python error: Aborted
Subprocess scheduler_N crashed with exit code -6 (SIGABRT)
```

- Faulting GPU kernel (rocm-debug-agent): `at::native::_assert_async_cuda_kernel<bool>(bool const*, at::native::Msg)`
- Trap reason: `ASSERT_TRAP`, `trapsts=0x80000000`
- Faulting wavefront registers full of `0xffc00000` (= bf16/fp32 NaN bit pattern)
- Python traceback surfaces (async) at:
  `event_loop_overlap_disagg_decode → process_batch_result_decode → result.copy_done.synchronize()`
  i.e. the error is reported at the next stream sync, not at the real failing op.

---

## 4. Variant matrix (what crashes vs what is stable)

Each row = a 12-run repeat loop (`repro_loop.sh`, conc=8 dur=300). Baseline
crashes deterministically by run 3–6; alternatives ran 12/12 clean.

| All-reduce impl | Fusion | cuda graph | Result |
|---|---|---|---|
| **aiter `custom_all_reduce`** (default) | off | on | ❌ crash run 3/3/6 |
| **aiter** | **on** (`--enable-aiter-allreduce-fusion`) | on | ❌ crash run 3 (same signature) |
| **aiter** | off | **off** (`--disable-cuda-graph`) | ❌ crash (NaN caught directly, see §6) |
| sglang native `CustomAllreduce` (`SGLANG_USE_AITER_AR=0`) | off | on | ✅ 12/12 clean |
| RCCL (`--disable-custom-all-reduce`) | off | on | ✅ 12/12 clean |

Conclusions from the matrix:
- The only variable that matters is the **all-reduce implementation**. RMSNorm
  was never fused in the crashing baseline (`enable_aiter_allreduce_fusion`
  defaults False and was not set), so fusion is **not** the cause — enabling it
  just crashes the same way.
- The bug is **not** cuda-graph-specific: it reproduces with `--disable-cuda-graph`.

---

## 5. Isolation tests (aiter AR is numerically correct in isolation)

8-rank standalone correctness harness (`aiter_ar_test.py`,
`aiter_fused_ar_rms_test.py`) comparing aiter AR vs RCCL reference:

| Path | Result |
|---|---|
| plain `custom_all_reduce`, eager, moderate data | ✅ matches RCCL (bf16 rounding only) |
| plain `custom_all_reduce`, **cuda graph** (proper `capture()` flow) | ✅ matches RCCL (identical to sglang-native control) |
| fused AR+RMSNorm, 1-stage (≤80 tok) and 2-stage | ✅ matches reference |
| large magnitude (MAG=50/200/1000) | ✅ no overflow/NaN |

So the kernel math is correct in isolation. The bug only appears under the
**real concurrent decode workload** (MoE FP4 all-to-all, attention, mori KV
transfer running alongside the AR) → strongly indicates a **race / memory
corruption / synchronization** issue, not a numerical one.

(Note: an earlier "graph repro" was a harness bug — missing
`flush_graph_buffers` from the `capture()` context — and was retracted.)

---

## 6. Definitive evidence: aiter AR is the NaN source

Direction-1 instrumentation: a sitecustomize monkeypatch
(`ar_nan_patch/sitecustomize.py`, `AR_NAN_CHECK=1`) wraps
`CustomAllreduce.all_reduce` and, after each call, checks the OUTPUT for NaN/Inf
and whether the INPUT was already bad.

Because the production decode AR runs inside CUDA graphs (Python wrappers don't
execute during graph replay, and `.item()` is illegal during capture), the run
was repeated with `--disable-cuda-graph` so the wrapper observes every eager AR.

Result on the eager run (decode crashed):

```
[AR_NAN] all_reduce: OUTPUT NaN/Inf  shape=(7, 7168)  dtype=torch.bfloat16  INPUT_ALREADY_BAD=False
```

| `INPUT_ALREADY_BAD` | count | meaning |
|---|---|---|
| **False** | **50** | clean input, NaN output → **aiter AR created the NaN** |
| True | 110 | NaN already present → downstream propagation to later AR calls |

The 50 clean-input→NaN-output cases are direct proof that aiter
`custom_all_reduce` itself manufactures NaN. Failing tensor:
**[7, 7168] bf16** (a small decode batch, ~98 KB), 2-stage path (`use_new=True`),
world_size=8.

---

## 7. Full causal chain

```
aiter custom_all_reduce(clean bf16 [7,7168])  ->  NaN output   (race/corruption under concurrent load)
  -> NaN propagates through hidden states / subsequent AR calls
  -> NaN enters the pytorch sampling backend (torch.multinomial / @torch.compile sampling)
  -> torch.compile/inductor-inserted _assert_async guard detects it -> s_trap (ASSERT_TRAP)
  -> HSA_STATUS_ERROR_EXCEPTION 0x1016 -> HIP unspecified launch failure
  -> decode scheduler aborts (SIGABRT), surfaces at next copy_done.synchronize()
```

Note: the trapping assert is **torch-internal** (pytorch sampling backend +
`torch.compile` guards), NOT sglang's `maybe_detect_nan` /
`SGLANG_SPEC_NAN_DETECTION` / `--enable-nan-detection` (all confirmed OFF), nor
sglang's sampler `enable_nan_detection` (OFF), nor any aiter device assert
(aiter has none). It fires regardless of sglang flags, which is why the crash
happens without any NaN-detection enabled.

---

## 8. Workaround

Validated stable (12/12 clean each):
- `SGLANG_USE_AITER_AR=0`  → sglang native custom all-reduce (keeps custom AR)
- `--disable-custom-all-reduce` → RCCL all-reduce

---

## 9. Reproduction / tooling (in `/home/wunhuang/dbg-1p1d-gpu-fault/`)

- `common.sh`, `step1-4.sh` — bring up 1P1D (prefill/decode/router) + benchmark
- `repro_loop.sh` — repeat the benchmark until a worker dies (auto-detect crash)
- `dbg_helpers.sh` — post-crash diagnostics (logs, dmesg/amdgpu, cores, rocgdb)
- `setup_coredumps.sh` / `restore_coredumps.sh` — core dump enable/restore
- `aiter_ar_test.py` / `aiter_fused_ar_rms_test.py` — isolated AR correctness harness
- `ar_nan_patch/sitecustomize.py` — `AR_NAN_CHECK=1` NaN-source instrumentation

Decode-launch hooks added (`scripts/multi_node/launch_decode_no_umbp.sh`):
`EXTRA_SGLANG_ARGS`, `SGLANG_USE_AITER_AR`, `LOG_LEVEL`, rocm-debug-agent
(`HSA_TOOLS_LIB`/`HSA_ENABLE_DEBUG`, save-code-objects), `DECODE_CORE_LIMIT_KB`,
and `AR_DEBUG_PATCH_DIR` + `AR_NAN_CHECK` for the instrumentation.

---

## 10. Kernel-level root cause (aiter `custom_all_reduce.cuh`)

### Which kernel runs in our case
Dispatch (`allreduce(...)`, `use_new=true`, `custom_all_reduce.cuh:~2850`):
- world_size=8, full xGMI, failing tensor [7,7168] bf16 = ~98 KB → `bytes >= 80KB`
  → **`call_2stage`** (not 1-stage).
- `use_write_mode` is gated on `arch.find("gfx942")`; we are **gfx950 (MI355X)**,
  so write-mode is OFF → the kernel used is **`cross_device_reduce_2stage`**
  (`custom_all_reduce.cuh:473`).

### The data dependency
`cross_device_reduce_2stage`:
1. `start_sync`
2. stage 1 (reduce-scatter): read peers' **input** buffers, reduce, write the
   partial to **own** tmp buffer `tmp_out[...]`.
3. `end_sync`
4. stage 2 (all-gather): `result[...] = tmps[warp_id][...]` → **reads PEER ranks'
   tmp buffers** (the partials each peer wrote in its stage 1).

Correctness of stage 2 thus requires `end_sync` to guarantee that **peer ranks'
stage-1 writes to their tmp buffers are globally (cross-GPU) visible** before
this rank reads them.

### The bug: acquire/release memory-scope mismatch in `end_sync`
`end_sync` ROCm path (`custom_all_reduce.cuh:205-229`):
```cpp
// publish flag to peers — SYSTEM scope (correct)
__scoped_atomic_store_n(&sg.signals[t]->end[blk][rank], flag,
                        final ? RELAXED : __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
// wait on own slot — DEVICE scope  <-- BUG
while (__scoped_atomic_load_n(&self_sg->end[blk][t],
                              final ? RELAXED : __ATOMIC_ACQUIRE,
                              __MEMORY_SCOPE_DEVICE) < flag) ;
```
- The **release-store is `__MEMORY_SCOPE_SYSTEM`** but the **acquire-load that
  pairs with it is `__MEMORY_SCOPE_DEVICE`**.
- For a cross-GPU producer→consumer hand-off, the acquire must also be SYSTEM
  scope. A DEVICE-scope acquire only orders memory ops within the consumer's own
  device; it does **not** guarantee the producer GPU's tmp-buffer data writes are
  visible after the flag is observed.
- The ROCm `end_sync` also has **no `__threadfence_system()`** (the non-ROCm path
  has one at line 237); it relies solely on this acquire/release pair.
- `start_sync` has the same DEVICE-scope acquire (`:171`) — lower risk, but the
  same class of issue.

The author's own comment in `end_sync` is the tell:
> "eliminate the case that prior writes are not visible after signals become
> visible. Note that I did not manage to make this happen through a lot of
> testing. Might be the case that hardware provides stronger guarantee than the
> memory model."

i.e. the visibility hole is known/theoretical and was never hit in (low-traffic)
testing — exactly matching our data: clean in isolation, NaN only under real
concurrent xGMI/HBM contention (MoE FP4 all-to-all + mori KV transfer), where the
window for a stale/partial peer-tmp read widens. The consumer observes the flag
but reads not-yet-arrived peer tmp data → garbage → NaN.

### Proposed fix (to validate)
In `end_sync` (and `start_sync`) ROCm path, make the acquire-load SYSTEM scope to
match the release-store, and/or add an explicit system fence:
```cpp
// option A: fix the scope
... __scoped_atomic_load_n(..., __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM) ...
// option B (belt-and-suspenders): after the wait loop, before reading peer data
__threadfence_system();
```
Validation: rebuild aiter with the fix, rerun `repro_loop.sh` (baseline aiter AR,
default config) for >=12 runs; expect 0 crashes and AR_NAN_CHECK=1 (eager) to
report 0 clean-input→NaN cases.

### Workaround until fixed
`SGLANG_USE_AITER_AR=0` or `--disable-custom-all-reduce` (both validated stable).

---

## 11. RESOLUTION — aiter PR #3514 (validated fix)

The actual upstream fix is **ROCm/aiter PR #3514** ("add missing end_sync barrier
in cross_device_reduce_1stage", merged Jun 4 2026), which addresses issue #3515
(identical symptoms: Qwen3.5-FP8 on SGLang, HSA 0x1016 from
`_assert_async_cuda_kernel`, sampling-only crash).

### Real root cause (corrected)
`cross_device_reduce_1stage` is the ONLY AR kernel that did **not** call
`end_sync` at kernel exit. Without an exit barrier:
1. a fast rank A finishes its peer IPC reads and exits the kernel;
2. A's next graph kernel runs and (via PyTorch graph_pool slot reuse in a
   captured CUDA graph) overwrites A's AR input slot;
3. a slow rank B is still inside the 1-stage kernel reading A's input slot over
   IPC, so B reads the overwritten data (GEMM intermediate / NaN / Inf);
4. B's AR output is corrupted → NaN propagates → sampling (`multinomial`) trips
   the torch device assert.

### The fix
```cpp
        buf = next_buf;
    }
+   end_sync<ngpus, true>(sg, self_sg, rank);   // match every other AR kernel
}
```

### Why my two earlier fixes failed (corrected analysis)
- I had concluded the bug was the SYSTEM-vs-DEVICE acquire scope inside
  `end_sync`, used by `cross_device_reduce_2stage`. Both my fixes (acquire scope,
  + producer write-side `__threadfence_system`) were built, disassembly-verified
  (`buffer_inv sc0 sc1`, `buffer_wbl2`), and loaded — yet crashed at run 4 / run 6.
- They failed because **the buggy kernel (`cross_device_reduce_1stage`) never
  calls `end_sync` at all** — so strengthening `end_sync` had no effect on it.
- My "2stage is the source" conclusion was a misread of the AR_NAN
  instrumentation: it only checks the LOCAL rank's input. A 2-stage AR that sums
  all ranks can output NaN with a clean LOCAL input when a PEER's input is
  already NaN (from the upstream 1-stage corruption). `INPUT_ALREADY_BAD=False`
  meant "my local input is clean", NOT "all peers clean". The [7,7168] (2-stage,
  >80KB) calls I caught were propagation; the true source was small-batch
  (<80KB, <=5 tokens) `cross_device_reduce_1stage` calls — invisible to the
  Python wrapper because the production bug only triggers under CUDA-graph
  capture, where the wrapper cannot run (sync illegal during capture; Python not
  executed during graph replay).

### Validation on our setup (DeepSeek-R1-MXFP4, TP=8, gfx950 MI355X, 1P1D)
Repeat-loop (conc=8, dur=300), aiter AR enabled, cuda graph ON (production):

| Variant | Result |
|---|---|
| baseline (no fix) | crash at run 3 / 3 / 6 |
| my fix v1 (end_sync acquire -> SYSTEM scope) | crash run 4 |
| my fix v2 (+ producer-side `__threadfence_system`) | crash run 6 |
| **aiter PR #3514 (end_sync in 1stage)** | **12 / 12 clean, 0 ASSERT_TRAP** |

=> PR #3514 is the correct fix and resolves the decode crash on this setup.
Recommendation: upgrade aiter to include PR #3514 (commit 3895df5). Until the
runtime image carries it, either bind-mount a rebuilt patched
`module_custom_all_reduce.so` or use the `SGLANG_USE_AITER_AR=0` workaround.
