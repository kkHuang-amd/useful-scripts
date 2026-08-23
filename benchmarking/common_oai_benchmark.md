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
  --trust-remote-code \
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

Use `--trust-remote-code` for models such as Kimi-K3 whose tokenizer
implementation is supplied by the model repository.

To start one two-second SGLang profile after the second unique measured
request produces a non-empty first token:

```bash
python3 common_oai_benchmark.py \
  --base-url http://127.0.0.1:30000 \
  --model /path/to/model \
  --prompt-manifest /tmp/common_prompt.jsonl.gz \
  --output-dir /tmp/common_oai_result \
  --profile-on-first-token \
  --profile-engine sglang \
  --profile-output-dir /tmp/common_oai_result/traces \
  --profile-after-first-tokens 2 \
  --profile-seconds 2
```

For ATOM, select `--profile-engine atom`; its trace directory is configured on
the server, so `--profile-output-dir` is not required:

```bash
python3 common_oai_benchmark.py \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/model \
  --prompt-manifest /tmp/common_prompt.jsonl.gz \
  --output-dir /tmp/common_oai_atom \
  --profile-on-first-token \
  --profile-engine atom
```

For a profile that covers the complete measured wave, start before the wave
and stop only after every measured request has returned a success or failure
record:

```bash
python3 common_oai_benchmark.py \
  --base-url http://127.0.0.1:30000 \
  --model /path/to/model \
  --prompt-manifest /tmp/common_prompt.jsonl.gz \
  --output-dir /tmp/common_oai_wave \
  --profile-before-wave \
  --profile-stop-after-wave \
  --profile-engine sglang \
  --profile-output-dir /tmp/common_oai_wave/traces
```

Without `--profile-stop-after-wave`, `--profile-before-wave` retains timed
behavior and stops after `--profile-seconds`. The wave-stop option is valid
only with `--profile-before-wave`; it ignores the timer and uses an independent
control connection for both profile requests.

`--profile-after-first-tokens N` defaults to `1`, preserving first-token
behavior. It counts each measured request ID once, even if duplicate callbacks
or chunks are observed, and must be between 1 and `--num-prompts`.
`--profile-base-url` defaults to `--base-url` and can point profile control at
a separate listener. Warmup requests never count. The requested threshold,
unique observed count, triggering request, first-token observations, capture
mode, wave-complete timestamp, control responses, elapsed profile duration,
and errors are stored under `profile` in `summary.json`. The command fails if
the threshold is not reached or either profile control request fails.

Run the unit tests with:

```bash
python3 -m unittest -v test_common_oai_benchmark.py
```
