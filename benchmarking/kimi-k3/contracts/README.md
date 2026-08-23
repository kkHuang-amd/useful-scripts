# Kimi-K3 contract diagnostics

## Current-route A8W4/A16W4 diagnostic

`sitecustomize.py` is an isolated, opt-in import overlay. It wraps
`aiter.fused_moe.fused_moe` before SGLang or ATOM import it, without changing
either engine or AITER. It is a no-op unless `K3_ROUTE_DUMP_DIR` is set.
Even when installed, it does not increment its call index or dump unless the
required `K3_ROUTE_DUMP_ARM_FILE` exists and contains valid `armed=true` JSON.

For non-capturing calls whose `topk_ids` shape is exactly `[64,16]`, the
wrapper writes atomic, bounded JSON records. Each record contains route counts,
active experts, BM32 padding, top experts, tensor metadata, rank/process
identity, quantization mode, and a fixed allowlist of routing-related
environment values. It never stores hidden activation values or other large
tensors. Each dump records the shared arm event and UTC/wall/monotonic dump
timestamps. `K3_ROUTE_DUMP_MAX_CALLS` defaults to 92.

**Validity boundary:** route dumps produced before the arm-file contract was
added are unarmed and invalid, including the prior 2026-08-22 SGLang/ATOM
route results. The analyzer rejects those legacy dumps rather than silently
combining startup/warmup calls with the common-client wave.

Run each engine separately with the same common C64 manifest:

```bash
MANIFEST=/workspace/kimi-k3-runs/common-oai-sglang-atom-c64-2026-08-21/prompt-manifest-c64-8192.jsonl.gz

bash contracts/run_route_dump.sh \
  sglang /workspace/kimi-k3-runs/current-route-contract/sglang "$MANIFEST"
bash contracts/run_route_dump.sh \
  atom /workspace/kimi-k3-runs/current-route-contract/atom "$MANIFEST"
```

The runner owns one server lifetime. It preserves the current AITER/FlyDSL and
matched production model, precision, attention, and MoE settings, but
explicitly disables graphs for this contract diagnostic. SGLang uses
`--disable-cuda-graph`; ATOM uses `--enforce-eager` with dual-stream MoE
disabled. The client contract is exact 8192/128, concurrency 64, 64 requests,
seed 42, and zero warmups. The model config reports 93 hidden layers, with the
first dense layer followed by 92 MoE layers, so the runner requires calls
`0..91` on each of ranks `0..7`. It retains logs, runtime metadata, request
outputs, route checksums, and VRAM snapshots, then cleans its named JIT caches.
The arm path remains absent throughout server startup and readiness checks.
Immediately before the common client starts, the runner atomically creates the
arm file and records its UTC, wall-clock, and monotonic timestamps. Immediately
after the client exits it atomically moves that file to
`route-arm-record.json`, adds disarm timestamps, and validates that every dump
falls within the armed interval. The evidence record is retained.

Eager execution changes serving behavior and invalidates timing comparisons.
These runs are route-contract evidence only; do not report their latency or
throughput as engine performance.

Analyze and exactly align both roots:

```bash
python contracts/analyze_route_dumps.py \
  /workspace/kimi-k3-runs/current-route-contract/sglang/route-dumps \
  /workspace/kimi-k3-runs/current-route-contract/atom/route-dumps \
  --left-name sglang-a8w4 \
  --right-name atom-a16w4 \
  --output-dir /workspace/kimi-k3-runs/current-route-contract/analysis
```

The analyzer writes `route-analysis.json`, `route-layers.csv`,
`route-deltas.csv`, and `route-analysis.md`. Missing or duplicate rank/call
keys fail by default. Every input must declare `armed=true`, share one arm
event per engine, and timestamp after that event; old unarmed inputs are hard
errors. `--allow-incomplete` is available only for exploratory partial
captures and does not bypass the arming contract.

## Tests and help

```bash
python -m unittest discover -s contracts -p 'test_*.py'
python contracts/analyze_route_dumps.py --help
bash contracts/run_route_dump.sh --help
```

The existing `tensor_contract_dump.py` and `compare_tensor_contracts.py`
remain the generic tensor ABI helpers.
