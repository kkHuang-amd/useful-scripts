# DeepSeek-V4-Pro @ 8x B200 (SGLang) — Usage

Scripts in this dir (`/dockerx/home/wunhuang/useful-scripts/benchmarking/dsv4/b200/`):
- `run_sgl_dsv4_pro_b200.sh` — launch the SGLang server (dp8 + megamoe, optional HiCache)
- `run_agentic_replay.sh`    — drive the InferenceX agentic trace-replay (aiperf)
- `_dl_dsv4.sh`              — resumable model download helper
- `run_sgl_dsv4_b200.sh`, `run_sgl_dsv4_70k_b200.sh`, `gsm8k_b200.sh`, `lm-eval-b200.sh` — other B200 recipes

---

## 1. Launch the server — `run_sgl_dsv4_pro_b200.sh`

All config via env vars (defaults in parentheses):

| var | meaning | default |
|-----|---------|---------|
| `MODE` | `dp8` (dp-attn + megamoe + deepep) or `tp8` (TP-only baseline) | `dp8` |
| `TP` / `EP_SIZE` | tensor- / expert-parallel size | `8` / `8` |
| `CONC` | sizes `cuda-graph-max-bs` (cap 64) and `max-running-requests` (2*CONC) | `256` |
| `MEM` | `--mem-fraction-static` | `0.88` |
| `HICACHE` | `on` = DRAM KV offload tier; `off` = none | `off` |
| `HICACHE_RATIO` | host/device token ratio (<=8) | `8` |
| `PORT` | server port | `8000` |
| `MODEL` | model path | `/dockerx/mnt/models/deepseek-ai/DeepSeek-V4-Pro` |
| `SGL_EXTRA_ARGS` | extra flags appended verbatim | — |

Examples:

```bash
cd /dockerx/home/wunhuang/useful-scripts/benchmarking/dsv4/b200

# default high-throughput dp8
bash run_sgl_dsv4_pro_b200.sh

# TP-only baseline
MODE=tp8 bash run_sgl_dsv4_pro_b200.sh

# dp8 + HiCache DRAM offload (needed for agentic prefix reuse), higher concurrency
MODE=dp8 CONC=256 HICACHE=on HICACHE_RATIO=8 bash run_sgl_dsv4_pro_b200.sh

# A/B an extra env/flag
SGLANG_ROCM_USE_MULTI_STREAM=0 MODE=dp8 CONC=256 HICACHE=on bash run_sgl_dsv4_pro_b200.sh
```

---

## 2. Lane A — random ISL/OSL throughput (`sglang.bench_serving`)

Matches the InferenceX B200 "Inference Performance" 8k/1k lane. Server must be up first.
Convention: `num-prompts = conc*8`, `warmup = conc*2`, `ratio=1.0`.

```bash
# conc 128
python3 -m sglang.bench_serving \
  --backend sglang --base-url http://localhost:8000 --model DeepSeek-V4-Pro \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 1024 --max-concurrency 128 --warmup-requests 256 \
  --output-file bench_results/laneA_c128.jsonl

# conc 256
python3 -m sglang.bench_serving \
  --backend sglang --base-url http://localhost:8000 --model DeepSeek-V4-Pro \
  --dataset-name random --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 2048 --max-concurrency 256 --warmup-requests 512 \
  --output-file bench_results/laneA_c256.jsonl
```

---

## 3. Lane B — agentic trace replay (`run_agentic_replay.sh`)

Reuses InferenceX `benchmark_lib.sh` (aiperf, `inferencex-agentx-mvp` scenario).
Requires the InferenceX repo + built aiperf venv + the DSv4 trace dataset.
All paths overridable via env vars (defaults in parentheses):

| var | meaning | default |
|-----|---------|---------|
| `INFMAX_CONTAINER_WORKSPACE` | InferenceX repo root | `/dockerx/home/wunhuang/InferenceX` |
| `AIPERF_VENV` | isolated aiperf venv | `/tmp/inferencex-agentic-venv` |
| `AGENTIC_DIR` / `AIPERF_DIR` | derived from workspace unless set | — |
| `MODEL` | served-model-name (must match server) | `DeepSeek-V4-Pro` |
| `MODEL_PATH` | local weights = tokenizer source | `/dockerx/mnt/models/deepseek-ai/DeepSeek-V4-Pro` |
| `MODEL_PREFIX` | corpus family selector | `dsv4` |
| `KV_OFFLOADING` | `none` or `dram` | `none` |
| `CONC` / `DURATION` / `PORT` | concurrency / seconds / port | `128` / `900` / `8000` |
| `RESULT_DIR` | output dir | `<script_dir>/agentic_results/conc<CONC>_dur<DURATION>` |

One-time setup (clone repo + aiperf submodule, build venv, fetch dataset):

```bash
cd /dockerx/home/wunhuang
git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX.git
cd InferenceX && git submodule update --init --depth 1 utils/aiperf

WS=/dockerx/home/wunhuang/InferenceX VENV=/tmp/inferencex-agentic-venv
uv venv --python "$(command -v python3)" "$VENV"
uv pip install --python "$VENV/bin/python" \
  -r "$WS/utils/agentic-benchmark/requirements.txt" -e "$WS/utils/aiperf" \
  "datasets>=4.7.0" "huggingface_hub[cli]>=0.25.0" urllib3 requests

"$VENV/bin/hf" download --repo-type dataset semianalysisai/cc-traces-weka-062126
```

Run (server must already be up; use HICACHE=on for the prefix-reuse config):

```bash
cd /dockerx/home/wunhuang/useful-scripts/benchmarking/dsv4/b200

# conc 128, canonical 900s window
CONC=128 DURATION=900 bash run_agentic_replay.sh

# conc 256
CONC=256 DURATION=900 bash run_agentic_replay.sh

# quick smoke (short window auto-adds --unsafe-override; expect 0 profiled completions)
CONC=128 DURATION=120 bash run_agentic_replay.sh

# override paths / dram offload
INFMAX_CONTAINER_WORKSPACE=/path/to/InferenceX AIPERF_VENV=/path/to/venv \
MODEL_PATH=/path/to/DeepSeek-V4-Pro KV_OFFLOADING=dram CONC=256 bash run_agentic_replay.sh
```

Notes:
- Warmup at conc 128 takes ~15 min before the profiling window starts.
- Agentic runs are prefill-heavy (median ISL ~100k, OSL ~1); TTFT/E2E dominate,
  and HiCache is what enables prefix reuse (`--enable-hierarchical-cache`).

---

## 4. Model download — `_dl_dsv4.sh`

Resumable HF download of `deepseek-ai/DeepSeek-V4-Pro` onto the NFS mount (auto-retries
on transient failures; uses Xet high-performance transfer):

```bash
bash _dl_dsv4.sh    # writes to /dockerx/mnt/models/deepseek-ai/DeepSeek-V4-Pro, logs to dsv4_download.log
```
