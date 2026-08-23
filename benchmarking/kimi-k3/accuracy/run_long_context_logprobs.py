#!/usr/bin/env python3
"""Send deterministic long-context requests and retain output top-k logprobs."""

import argparse
import asyncio
import json
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer


async def send(session, url, input_ids, index, semaphore, max_new_tokens, topk):
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
        },
        "return_logprob": True,
        "logprob_start_len": len(input_ids),
        "top_logprobs_num": topk,
    }
    async with semaphore:
        async with session.post(url, json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"request {index}: HTTP {response.status}: {body}")
            return {"index": index, "response": json.loads(body)}


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    phrase = tokenizer.encode(
        " The quick brown fox jumps over the lazy dog. "
        "Kimi reasons carefully about this context.",
        add_special_tokens=False,
    )
    base = (
        phrase * ((args.input_tokens + len(phrase) - 1) // len(phrase))
    )[: args.input_tokens - 8]
    prompts = []
    for index in range(args.requests):
        suffix = tokenizer.encode(f" Question {index}:", add_special_tokens=False)
        prompts.append((base + suffix)[-args.input_tokens :])

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(
            *[
                send(
                    session,
                    f"{args.base_url}/generate",
                    input_ids,
                    index,
                    semaphore,
                    args.max_new_tokens,
                    args.topk,
                )
                for index, input_ids in enumerate(prompts)
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results) + "\n")
    print(
        f"requests={len(results)} input_tokens={args.input_tokens} "
        f"output={args.output}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/shared_nfs/models/Kimi-K3")
    parser.add_argument("--input-tokens", type=int, default=68000)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
