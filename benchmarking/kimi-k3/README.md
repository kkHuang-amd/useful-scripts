# Kimi-K3 deployment and accuracy benchmark scripts

This directory is the canonical reusable toolkit. Dated experiment folders
under `/workspace/kimi-k3-runs/` remain immutable evidence; promote reusable
logic here instead of copying another run-specific wrapper.

## Functional layout

```text
kimi-k3/
├── server/     runtime/config dumps; root launch/wait/stop scripts stay compatible
├── endpoint/   serving workload matrices against an already-running endpoint
├── accuracy/   GSM8K, retrieval, and paired long-context logprob collection
├── trace/      compact and matched SGLang/ATOM profile capture
├── analysis/   endpoint, Chrome-trace, rocprof DB, and output comparisons
├── contracts/  tensor ABI dump and cross-runtime comparison
└── micro/      shared HIP/CUDA graph timing and numerical helpers
```

The existing root scripts are retained to avoid breaking established command
lines. New reusable tools live in the categorized directories:

```bash
# Dump reproducibility metadata without dumping credentials.
OUTPUT_DIR=/path/to/run/runtime server/dump_runtime_state.sh

# Run standard 8K and 68K matrices against an existing server.
RESULT_DIR=/path/to/run/endpoint endpoint/run_serving_matrix.sh

# Capture a compact TP8 trace from an existing server.
SERVER_LOG=/path/server.log RESULT_DIR=/path/to/run/trace \
  trace/capture_compact_profile.sh

# Run one matched common-client SGLang/ATOM profile case.
trace/run_common_profile_case.sh sglang 2 /path/to/sglang-c2 \
  /workspace/kimi-k3-runs/common-oai-sglang-atom-c2-2026-08-21/prompt-manifest-c2-8192.jsonl.gz

# Capture one production prefill-step trace from the common manifest.
trace/run_common_prefill_case.sh sglang 2 /path/to/fresh/sglang-c2-prefill \
  /workspace/kimi-k3-runs/common-oai-sglang-atom-c2-2026-08-21/prompt-manifest-c2-8192.jsonl.gz

# Capture diagnostic FULL graph-construction attribution for BS2 and BS64.
trace/capture_graph_attribution.sh sglang /path/to/fresh/sglang-graph-capture
trace/capture_graph_attribution.sh atom /path/to/fresh/atom-graph-capture

# Summarize endpoint logs and compare two variant directories.
python analysis/summarize_endpoint_logs.py /path/to/run \
  --csv /path/to/summary.csv \
  --baseline aiter --candidate triton_fp8 \
  --comparison-json /path/to/comparison.json

# Analyze SGLang Chrome traces or rocprof SQLite data.
python analysis/analyze_chrome_trace.py rank0.trace.json.gz \
  --output rank0-summary.json --tail-seconds 2
python analysis/analyze_chrome_trace.py rank0.trace.json.gz \
  --output rank0-decode-summary.json --decode-only \
  --emit-caller-map rank0-callers.jsonl --min-confidence inferred
python analysis/analyze_chrome_trace.py rank0.trace.json.gz \
  --output rank0-decode-summary.json --decode-only \
  --select-graph-step middle \
  --selected-step-output rank0-selected-step.json
python analysis/analyze_chrome_trace.py graph-capture.json.gz \
  --output graph-warmup-summary.json \
  --annotation-name ProfilerStep#1 \
  --annotation-category gpu_user_annotation \
  --annotation-occurrence 0
python analysis/analyze_rocprof_db.py results.db \
  --duration 2 --output rocprof-summary.json

# Collect and compare paired 68K output top-k probabilities.
python accuracy/run_long_context_logprobs.py --output baseline.json
python analysis/compare_generate_outputs.py \
  --baseline baseline.json --candidate candidate.json \
  --output comparison.json

# Capture and compare eager-only C64 fused-MoE route contracts.
contracts/run_route_dump.sh sglang /path/to/routes/sglang /path/to/c64-manifest.jsonl.gz
contracts/run_route_dump.sh atom /path/to/routes/atom /path/to/c64-manifest.jsonl.gz
python contracts/analyze_route_dumps.py \
  /path/to/routes/sglang/route-dumps /path/to/routes/atom/route-dumps \
  --output-dir /path/to/routes/analysis
```

Consolidation rules:

- Server launch remains separate from benchmark clients so one server can feed
  endpoint, accuracy, and trace runs.
- `endpoint/run_serving_matrix.sh` replaces repeated client loops; experiment
  wrappers should only define server environments and call it.
- Chrome and rocprof analyzers share `analysis/kernel_families.py`.
- The Chrome analyzer correlates timed CPU ops and CUDA/HIP launch APIs to GPU
  kernels using correlation/external IDs, with `ac2g` flow and annotation
  containment as lower-confidence fallbacks. Its v2 summary adds validation
  gates, decode capture windows, caller rollups, and graph-replay opacity while
  retaining the original aggregate keys.
- `--decode-only` recognizes SGLang `step[DECODE...` and ATOM `decode[...]`
  scopes, preferring GPU annotations and falling back to CPU annotations.
  `--emit-caller-map` writes grouped JSONL by caller, annotation, host API,
  kernel, family, stream, and confidence.
- `--select-graph-step {first,middle,last}` with `--selected-step-output`
  extracts one decode replay using exact graph-launch correlation IDs. Replay
  ordering uses the earliest correlated GPU-kernel timestamp because CPU
  runtime and GPU-projected events can use different clocks. `middle` excludes
  the first and last eligible replay and therefore requires at least three.
  Selection fails when direct per-step correlation is unavailable; the
  analyzer never infers steps by splitting a trace into equal chunks. The full
  summary remains unchanged except for a pointer/metadata block.
- `--annotation-name NAME` selects exactly one complete named annotation
  interval for graph warmup/capture attribution. Use
  `--annotation-category user_annotation|gpu_user_annotation` when needed and
  `--annotation-occurrence N` to choose a repeated occurrence in timestamp
  order. The interval intersects `--tail-seconds` and `--decode-only` windows;
  missing occurrences and same-timestamp category ambiguity fail explicitly.
  Selection details are recorded under `capture_window.annotation_selection`.
- `trace/run_common_profile_case.sh` owns one server lifetime for the matched
  four-case campaign. Its interface is `ENGINE CONCURRENCY CASE_DIR MANIFEST`,
  where engines are `sglang|atom` and concurrency is `2|64`. It keeps FULL
  production graphs enabled, runs an unprofiled exact-manifest 8192/64 wave,
  profiles the decode wave from its first streamed token, requires eight valid
  rank traces, and emits per-rank decode summaries and caller maps. Defaults
  are `PROFILE_SECONDS=2`, `TRANSITION_OUTPUT_LEN=64`, and
  `DECODE_OUTPUT_LEN=256` for C2 or `1024` for C64. Profiling starts only
  after all requests in the wave have produced their first non-empty token,
  so the fixed window samples full-batch decode rather than overlapping
  residual prefill work.
  `SGLANG_DECODE_ATTENTION_BACKEND=aiter` enables a controlled SGLang-only
  backend A/B while retaining the same production profile; the default is
  `triton`, accepted values are `triton|aiter`, and the effective backend is
  saved in runtime metadata. `SGLANG_MEM_FRACTION_STATIC` can override the
  SGLang launch fraction for backend workspace experiments (default `0.85`);
  it is also persisted and any non-default value must be treated as a capacity
  difference rather than a matched endpoint comparison.
- `trace/run_common_prefill_case.sh` owns one server lifetime for a single
  production prefill-step capture. Its interface is
  `ENGINE CONCURRENCY CASE_DIR MANIFEST`, with engines `sglang|atom` and
  concurrency `2|64`; `CASE_DIR` must not exist. It uses the exact production
  server profiles from `run_common_profile_case.sh`, including AITER prefill
  and FULL Triton decode graphs. After client/tokenizer setup, the common
  client starts profiling immediately before an exact 8192/2 measured wave
  with no warmups, stops after `PREFILL_PROFILE_SECONDS` (default 2) while the
  wave remains active, and preserves profile HTTP responses and timestamps.
  The runner requires eight gzip-valid rank exports of at most 500 MiB each,
  analyzes each complete trace (not decode-only), emits caller maps, gates
  CPU/GPU categories and correlation, and verifies `step[EXTEND` or
  `sglang.vlm.language_model_prefill` for SGLang and `prefill[` for ATOM.
- `trace/capture_graph_attribution.sh` owns one diagnostic server lifetime and
  accepts `ENGINE OUTPUT_DIR`, where the engine is `sglang|atom` and the output
  path must not exist. It uses the same current AITER/FlyDSL and production
  Kimi-K3 profiles as `run_common_profile_case.sh`, keeps FULL/non-eager graphs,
  and restricts construction to BS2 and BS64. SGLang records one combined
  warmup/construction trace per TP rank (8 targets); ATOM records BS2/q1 and
  BS64/q1 per rank (16 targets). It waits for readiness, stops without running
  endpoint requests, validates gzip/size/category coverage, inventories target
  traces separately from retained metadata, and verifies GPU-memory release.
- Graph replay is intentionally treated as opaque below
  `hipGraphLaunch`/`cudaGraphLaunch`. Kernel durations observed during replay
  may be unreliable. `--capture-trace` is a placeholder only: graph-node
  matching is not implemented or claimed, and future caller mapping for
  runtime graphs will use separate warmup/capture traces.
- Kernel micros should import `micro/microbench_common.py` instead of copying
  graph timing and relative-L2 helpers.
- `micro/benchmark_dense_crossover.py` compares complete BF16, PTPC FP8, and
  MXFP4 projection chains at M2-M64. Weight preparation is outside timing;
  activation quantization and per-call conversions are inside graph replay. Each
  captured case must pass an input-change replay gate before timing: the tool
  overwrites the same static activation tensor, rejects stale/equal or non-finite
  output, records output delta/hash diagnostics, restores the original input,
  and then uses the shared replay timing helper.
  `merged_front` and `kda_inproj` are labelled SGLang-only, non-like-for-like
  context. Use `--shapes`, `--modes`, and `--m-values` to narrow a run. These
  micro results never replace matched common-client endpoint workloads or
  paired GSM8K and long-context correctness gates.
  Projection metadata records the Kimi-K3 multiplicity assumptions: 92 MoE,
  69 KDA, and 24 MLA layers. The common `[7168,1536]` TP8-local output
  representation covers all 93 attention layers. Regenerate model-weighted
  storage artifacts from an existing JSON without importing GPU modules:

  ```bash
  python micro/benchmark_dense_crossover.py \
    --storage-input-json /path/to/dense-crossover.json \
    --storage-output-json /path/to/layer-weighted-storage.json \
    --storage-output-csv /path/to/layer-weighted-storage.csv
  ```

  The storage report keeps one-shape `representative_shape_storage` separate
  from `model_layer_weighted_storage`, retains BF16 in all policy costs, and
  does not add mutually exclusive prepared modes.
- `micro/benchmark_mla_shared_ptpc.py` graph-times the Kimi-K3 MLA input
  boundary at configurable M values. Its baseline includes BF16 QKV-A, BF16
  g_proj, and SGLang's fused sigmoid/multiply output gate. The candidate
  performs one runtime per-token FP8 quant and reuses its activation and scale
  for both preshuffled A8W8 projections before the same output gate. It also
  compares current RMSNorm plus the baseline boundary against AITER's fused
  RMSNorm+per-token quant plus the shared candidate. Weight preparation is
  excluded; all runtime conversions, changed-input graph replay, numerical
  gates, and combined 24-layer storage are recorded.
- `micro/benchmark_route_modes.py` reconstructs deterministic `[64,16]`
  assignments from the exact armed rank-0 expert counts in the SGLang and ATOM
  route dumps. It fails unless all 92 calls are present, all 1024 routes are
  preserved, and every token has 16 distinct experts. It compares the current
  AITER Kimi `E896/H3584/I384/topk16` SiTUv2 (`beta=4`,
  `linear_beta=25`) A8W4 and A16W4 paths over the cross matrix of both route
  sources and both activation modes. Full fused-MoE HIP graph replay includes
  sorting, A8 activation quantization, both stages, and caller-owned output;
  isolated stages use AITER's opt-in `kernel_bench_callable` closures and are
  explicitly skipped if current dispatch does not expose them. Weight
  quantization and the separate current A8/A16 preshuffles are outside timing.
  JSON retains every replay sample; CSV contains per-layer summaries; Markdown
  contains aggregate 92-layer sums/medians and latency regressions versus BM32
  blocks. The GPU campaign fails immediately on NaN/Inf, unchanged graph output
  after an activation update, A8/A16 relative L2 above `0.25`, or A8/A16 cosine
  below `0.95`; both numerical thresholds are configurable. These are
  current-kernel micros only, never endpoint claims.

  Validate routes without importing GPU modules:

  ```bash
  python micro/benchmark_route_modes.py \
    --validate-routes-only \
    --output-dir /tmp/k3-route-mode-validation
  ```

  Exact GPU command (not run during CPU-only development):

  ```bash
  AITER_FLYDSL_FORCE=1 \
  AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1 \
  python micro/benchmark_route_modes.py \
    --result-root /workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/route-validation \
    --warmup 10 --iterations 100 \
    --output-dir /workspace/kimi-k3-runs/matched-route-current-kernel-2026-08-23
  ```

  Allow approximately **10 GiB peak device memory**: two prepared MXFP4
  weight/scale layouts are about 2 GiB each, and preparation temporarily holds
  canonical BF16/quantized tensors; outputs, sort/stage scratch, graph pools,
  and JIT need additional headroom. The implementation releases canonical
  preparation tensors before timing.
- Contract instrumentation should import
  `contracts/tensor_contract_dump.py`; compare dumps with
  `contracts/compare_tensor_contracts.py`.
- `contracts/run_route_dump.sh` owns one eager-only TP8 diagnostic server
  lifetime and exact common-manifest 8192/128 C64 wave. Its isolated
  `sitecustomize.py` overlay records only armed `[64,16]` fused-MoE route
  counts and BM32 padding, capped at the model's 92 MoE layers per rank. The
  runner creates `K3_ROUTE_DUMP_ARM_FILE` only immediately before the common
  client and disarms after it. Analyze paired roots with
  `contracts/analyze_route_dumps.py`; exact rank/call alignment and valid
  post-arm timestamps are required. Legacy unarmed dumps are invalid. These
  eager artifacts are not timing evidence.

Defaults:

```text
MODEL_PATH=/dockerx/data/models/Kimi-K3
BENCH_DIR=/sgl-workspace/kvv-bench/kvv-k3-0727-update
BASE_URL=http://localhost:8000/v1
PORT=8000
```

Typical sequence:

```bash
# Optional: enter the replacement ROCm/SGLang image
IMAGE=<image@sha256:digest> ./run_docker.sh

# 1. Optional: download/resume weights
./download_weights.sh

# 2. Launch in a dedicated terminal or container process
./launch_server.sh 2>&1 | tee /tmp/kimi-k3-server.log

# 3. Wait for full SGLang warmup, then test
SERVER_LOG=/tmp/kimi-k3-server.log ./wait_server.sh
./smoke_test.sh

# 4. DCP text-accuracy regression
./run_gsm8k.sh

# 5. KVV accuracy suite (run one at a time)
./run_kvv.sh ocrbench
./run_kvv.sh mmmu
./run_kvv.sh toolcall

# 6. Long-context generation
RUN_TAG=<image-or-build-name> ./run_beam.sh

# Additional benchmark reproductions
./run_aiperf_68k_sweep.sh  # Shared-prefix ~68k aiperf sweep
./run_standard_8k1k.sh     # Fixed 8192/1024 c32 serving
./run_context_accuracy.py --mode <BF16-or-FP8> --output <results.json>  # One BF16/FP8 retrieval endpoint

# Inspect or stop the active deployment
./status.sh
./stop_server.sh
```

All settings can be overridden with environment variables. Run scripts under
`bash`; they use strict mode and fail on command errors.

The launcher defaults to TP8, decode DCP8 with the AITER backend, 8 concurrent
requests, and a decode CUDA Graph batch size of 8. Override the request and
graph limits together, for
example:

```bash
MAX_RUNNING_REQUESTS=32 CUDA_GRAPH_MAX_BS_DECODE=32 ./launch_server.sh
```

Prefill context parallelism is disabled by default. The 2026-08-04 A/B showed a
74-82% throughput regression for `ISL≈68k`, `OSL≈350`, because true CP raised
per-GPU weight memory and serialized requests. Use `ENABLE_PREFILL_CP=0` for
normal serving. Add `RADIX_CACHE=1` when the workload reuses prefixes.

Decode DCP is a separate feature. The default profile is equivalent to:

```bash
DCP_SIZE=8 ATTENTION_BACKEND=aiter ENABLE_PREFILL_CP=0 ./launch_server.sh
```

Use `DCP_SIZE=1` for the non-DCP TP8 baseline.

The local DSpark checkpoint can be enabled for smoke testing with:

```bash
DSPARK_DRAFT_MODEL_PATH=/dockerx/data/models/Kimi-K3-DSpark \
  ./launch_server.sh
```

This automatically selects static ragged verification and decode-mode
speculative attention. The DCP acceptance gate runs GSM8K without DSpark.

Important:

- Keep Radix Cache enabled for BEAM by launching with
  `ENABLE_PREFILL_CP=0 RADIX_CACHE=1`.
- Never start BEAM before `wait_server.sh` succeeds.
- `run_beam.sh` uses the model tokenizer to prevent K2.6/K3 token-count drift.
- Use a unique `RUN_TAG` for each image and weight build.
- BEAM judging is intentionally separate because it needs judge credentials.

