# Kimi-K3 deployment and accuracy benchmark scripts

Defaults:

```text
MODEL_PATH=/dockerx/data/Kimi-K3
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

# 4. Accuracy suite (run one at a time)
./run_kvv.sh ocrbench
./run_kvv.sh mmmu
./run_kvv.sh toolcall

# 5. Long-context generation
RUN_TAG=<image-or-build-name> ./run_beam.sh

# Inspect or stop the active deployment
./status.sh
./stop_server.sh
```

All settings can be overridden with environment variables. Run scripts under
`bash`; they use strict mode and fail on command errors.

The long-context CP launcher defaults to 8 concurrent requests and a decode
CUDA Graph batch size of 8. Override both together, for
example:

```bash
MAX_RUNNING_REQUESTS=32 CUDA_GRAPH_MAX_BS_DECODE=32 ./launch_server.sh
```

Prefill context parallelism is enabled by default with the `zigzag` strategy,
attention CP size 8, and Radix Cache disabled. The 2026-08-04 A/B showed a
74-82% throughput regression for `ISL≈68k`, `OSL≈350`, because true CP raised
per-GPU weight memory and serialized requests. Use `ENABLE_PREFILL_CP=0` for
normal serving. Add `RADIX_CACHE=1` when the workload reuses prefixes.

Important:

- Keep Radix Cache enabled for BEAM by launching with
  `ENABLE_PREFILL_CP=0 RADIX_CACHE=1`.
- Never start BEAM before `wait_server.sh` succeeds.
- `run_beam.sh` uses the model tokenizer to prevent K2.6/K3 token-count drift.
- Use a unique `RUN_TAG` for each image and weight build.
- BEAM judging is intentionally separate because it needs judge credentials.

