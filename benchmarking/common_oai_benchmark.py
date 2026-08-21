#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import aiohttp
import numpy as np
from transformers import AutoTokenizer


@dataclass
class RequestResult:
    request_id: int
    prompt_tokens: int
    completion_tokens: int = 0
    launch_ns: int = 0
    first_token_ns: int = 0
    last_token_ns: int = 0
    done_ns: int = 0
    text_chunks: int = 0
    done_seen: bool = False
    success: bool = False
    error: str = ""

    @property
    def ttft_s(self):
        return (self.first_token_ns - self.launch_ns) / 1e9

    @property
    def e2e_s(self):
        return (self.last_token_ns - self.launch_ns) / 1e9

    @property
    def tpot_s(self):
        return (
            (self.last_token_ns - self.first_token_ns)
            / 1e9
            / (self.completion_tokens - 1)
        )


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def summarize(results, duration_s):
    valid = [result for result in results if result.success]
    ttft = [result.ttft_s * 1000 for result in valid]
    tpot = [result.tpot_s * 1000 for result in valid]
    e2e = [result.e2e_s * 1000 for result in valid]
    prompt_tokens = sum(result.prompt_tokens for result in valid)
    completion_tokens = sum(result.completion_tokens for result in valid)
    metrics = {
        "successful_requests": len(valid),
        "failed_requests": len(results) - len(valid),
        "duration_s": duration_s,
        "request_throughput_req_s": len(valid) / duration_s,
        "input_token_throughput_tok_s": prompt_tokens / duration_s,
        "output_token_throughput_tok_s": completion_tokens / duration_s,
        "total_token_throughput_tok_s": (
            prompt_tokens + completion_tokens
        ) / duration_s,
    }
    for name, values in (("ttft", ttft), ("tpot", tpot), ("e2e", e2e)):
        metrics[f"mean_{name}_ms"] = float(np.mean(values))
        metrics[f"median_{name}_ms"] = percentile(values, 50)
        metrics[f"p95_{name}_ms"] = percentile(values, 95)
        metrics[f"p99_{name}_ms"] = percentile(values, 99)
    return metrics


def generate_prompts(tokenizer, count, length, seed):
    rng = np.random.default_rng(seed)
    prompts = []
    for request_id in range(count):
        offset = int(rng.integers(0, tokenizer.vocab_size))
        ids = [(offset + request_id + j) % tokenizer.vocab_size for j in range(length)]
        prompt = tokenizer.decode(ids)
        for _ in range(10):
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            if len(encoded) == length:
                break
            if len(encoded) < length:
                encoded.extend(
                    rng.integers(
                        0, tokenizer.vocab_size, size=length - len(encoded)
                    ).tolist()
                )
            else:
                encoded = encoded[:length]
            prompt = tokenizer.decode(encoded)
        actual = len(tokenizer.encode(prompt, add_special_tokens=False))
        if actual != length:
            raise RuntimeError(
                f"prompt {request_id} has {actual} tokens, expected {length}"
            )
        prompts.append(prompt)
    return prompts


def generate_sharegpt_prompts(tokenizer, count, length, seed, dataset_path):
    dataset = json.loads(Path(dataset_path).read_text())
    pairs = [
        item.get("conversations", item.get("conversation", []))
        for item in dataset
    ]
    prompts = [conversation[0]["value"] for conversation in pairs if len(conversation) >= 2]
    random.Random(seed).shuffle(prompts)
    content_len = length - int(tokenizer.num_special_tokens_to_add())
    rng = np.random.default_rng(seed)
    output = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            continue
        ids = (ids * ((content_len + len(ids) - 1) // len(ids)))[:content_len]
        prompt = tokenizer.decode(ids)
        for _ in range(10):
            actual = len(tokenizer.encode(prompt))
            if actual == length:
                break
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            target_content = max(1, len(ids) + length - actual)
            if len(ids) < target_content:
                ids.extend(
                    rng.integers(
                        0, tokenizer.vocab_size, size=target_content - len(ids)
                    ).tolist()
                )
            else:
                ids = ids[:target_content]
            prompt = tokenizer.decode(ids)
        if len(tokenizer.encode(prompt)) != length:
            continue
        output.append(prompt)
        if len(output) == count:
            break
    if len(output) != count:
        raise RuntimeError(f"ShareGPT yielded {len(output)} prompts, expected {count}")
    return output


def save_manifest(path, prompts, tokenizer, input_len, seed):
    digest = hashlib.sha256()
    with gzip.open(path, "wt") as handle:
        for request_id, prompt in enumerate(prompts):
            digest.update(prompt.encode())
            handle.write(
                json.dumps(
                    {
                        "request_id": request_id,
                        "prompt": prompt,
                        "prompt_tokens": input_len,
                    }
                )
                + "\n"
            )
    return {
        "sha256": digest.hexdigest(),
        "tokenizer": tokenizer.name_or_path,
        "input_len": input_len,
        "seed": seed,
        "count": len(prompts),
    }


def load_manifest(path):
    prompts = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            prompts.append(json.loads(line)["prompt"])
    return prompts


async def request_one(session, semaphore, url, model, prompt, prompt_len, output_len, request_id):
    result = RequestResult(request_id=request_id, prompt_tokens=prompt_len)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with semaphore:
            result.launch_ns = time.perf_counter_ns()
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {(await response.text())[:500]}"
                    return result
                buffer = b""
                async for chunk in response.content.iter_any():
                    buffer += chunk
                    while b"\n\n" in buffer:
                        raw, buffer = buffer.split(b"\n\n", 1)
                        line = raw.decode(errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        now = time.perf_counter_ns()
                        if data == "[DONE]":
                            result.done_seen = True
                            result.done_ns = now
                            continue
                        event = json.loads(data)
                        usage = event.get("usage")
                        if usage:
                            result.completion_tokens = int(
                                usage.get("completion_tokens", 0)
                            )
                            server_prompt_tokens = int(usage.get("prompt_tokens", 0))
                            if server_prompt_tokens != prompt_len:
                                raise RuntimeError(
                                    f"usage prompt_tokens={server_prompt_tokens}, expected={prompt_len}"
                                )
                        choices = event.get("choices") or []
                        if choices and choices[0].get("text"):
                            if not result.first_token_ns:
                                result.first_token_ns = now
                            result.last_token_ns = now
                            result.text_chunks += 1
        if not result.done_seen:
            raise RuntimeError("stream ended without [DONE]")
        if not result.first_token_ns:
            raise RuntimeError("stream contained no non-empty text")
        if result.completion_tokens != output_len:
            raise RuntimeError(
                f"completion_tokens={result.completion_tokens}, expected={output_len}"
            )
        result.success = True
    except Exception as exc:
        result.error = repr(exc)
    return result


async def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    total = args.warmup_requests + args.num_prompts
    manifest_path = Path(args.prompt_manifest)
    if manifest_path.exists():
        prompts = load_manifest(manifest_path)
        if len(prompts) < total:
            raise RuntimeError(f"manifest has {len(prompts)} prompts, requires {total}")
        manifest_meta = {"path": str(manifest_path), "count": len(prompts)}
    else:
        if args.prompt_source == "sharegpt":
            if not args.sharegpt_path:
                raise RuntimeError("--sharegpt-path is required for ShareGPT prompts")
            prompts = generate_sharegpt_prompts(
                tokenizer, total, args.input_len, args.seed, args.sharegpt_path
            )
        else:
            prompts = generate_prompts(tokenizer, total, args.input_len, args.seed)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_meta = save_manifest(
            manifest_path, prompts, tokenizer, args.input_len, args.seed
        )
    connector = aiohttp.TCPConnector(
        limit=args.max_concurrency,
        limit_per_host=args.max_concurrency,
        enable_cleanup_closed=True,
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    url = args.base_url.rstrip("/") + "/v1/completions"
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        read_bufsize=10 * 1024**2,
        trust_env=True,
    ) as session:
        warmup = [
            request_one(
                session,
                semaphore,
                url,
                args.model,
                prompts[index],
                args.input_len,
                args.output_len,
                -index - 1,
            )
            for index in range(args.warmup_requests)
        ]
        warmup_results = await asyncio.gather(*warmup)
        if not all(result.success for result in warmup_results):
            failures = [
                asdict(result) for result in warmup_results if not result.success
            ]
            (output_dir / "warmup_failures.json").write_text(
                json.dumps(failures, indent=2)
            )
            raise RuntimeError(
                f"warmup failed: {len(failures)} failures; first={failures[0]['error']}"
            )
        measured_prompts = prompts[args.warmup_requests : total]
        start = time.perf_counter()
        results = await asyncio.gather(
            *[
                request_one(
                    session,
                    semaphore,
                    url,
                    args.model,
                    prompt,
                    args.input_len,
                    args.output_len,
                    request_id,
                )
                for request_id, prompt in enumerate(measured_prompts)
            ]
        )
        duration = time.perf_counter() - start
    with (output_dir / "requests.jsonl").open("w") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result)) + "\n")
    summary = summarize(results, duration)
    summary["config"] = vars(args)
    summary["prompt_manifest"] = manifest_meta
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["failed_requests"]:
        raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--input-len", type=int, default=8192)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--num-prompts", type=int, default=5120)
    parser.add_argument("--warmup-requests", type=int, default=1024)
    parser.add_argument("--max-concurrency", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-source", choices=("synthetic", "sharegpt"), default="synthetic")
    parser.add_argument("--sharegpt-path")
    parser.add_argument("--timeout-s", type=int, default=21600)
    parser.add_argument("--prompt-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
