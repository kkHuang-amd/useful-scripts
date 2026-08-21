# Common OpenAI-Compatible Benchmark

`common_oai_benchmark.py` benchmarks any server implementing streaming
`POST /v1/completions`. It uses one request lifecycle, prompt manifest, and
TTFT/TPOT calculation across engines.

Dependencies:

```bash
pip install aiohttp numpy transformers
```

Example:

```bash
python3 common_oai_benchmark.py \
  --base-url http://127.0.0.1:30000 \
  --model /path/to/model \
  --input-len 8192 \
  --output-len 1024 \
  --num-prompts 5120 \
  --warmup-requests 1024 \
  --max-concurrency 512 \
  --prompt-manifest /tmp/common_prompt_c512.jsonl.gz \
  --output-dir /tmp/common_oai_result
```

The client validates HTTP status, SSE `[DONE]`, non-empty streamed text,
server-reported prompt length, and exact completion length. It writes
per-request records to `requests.jsonl` and aggregate metrics to
`summary.json`.

Use the same persisted prompt manifest when comparing engines or revisions.
Pass `--prompt-source sharegpt --sharegpt-path <dataset.json>` to create a
ShareGPT-derived exact-length manifest when the requested manifest does not
already exist.

Run the unit tests with:

```bash
python3 -m unittest -v test_common_oai_benchmark.py
```
