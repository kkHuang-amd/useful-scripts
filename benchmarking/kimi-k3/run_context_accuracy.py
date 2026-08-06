#!/usr/bin/env python3
"""Deterministic long-context retrieval validation for Kimi-K3 KV dtypes."""

import argparse
import json
import statistics
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


MODEL_PATH = "/dockerx/data/models/Kimi-K3"
TARGET_LENGTHS = (8192, 32768, 68000, 120000)
POSITIONS = (0.1, 0.5, 0.9)
FILLER = (
    "Archive record: the observatory measured ordinary weather patterns, "
    "catalogued routine instruments, and reported no exceptional events. "
)


def build_cases(tokenizer):
    filler_ids = tokenizer.encode(FILLER, add_special_tokens=False)
    cases = []
    for target in TARGET_LENGTHS:
        doc_budget = target - 256
        repeats = (doc_budget // len(filler_ids)) + 2
        doc_ids = (filler_ids * repeats)[:doc_budget]
        for position in POSITIONS:
            split = int(len(doc_ids) * position)
            code = f"K3-{target}-{int(position * 100):02d}-7391"
            document = (
                tokenizer.decode(doc_ids[:split])
                + f"\nSECRET CODE: {code}\n"
                + tokenizer.decode(doc_ids[split:])
            )
            prompt = (
                "Read the entire document. It contains exactly one line beginning "
                "with 'SECRET CODE:'. Return only the code after that label, with "
                "no explanation.\n\n"
                f"{document}\n\nWhat is the secret code?"
            )
            cases.append(
                {
                    "target_tokens": target,
                    "position": position,
                    "code": code,
                    "prompt": prompt,
                    "local_prompt_tokens": len(
                        tokenizer.encode(prompt, add_special_tokens=False)
                    ),
                }
            )
    return cases


def extract_logprobs(choice):
    content = (choice.get("logprobs") or {}).get("content") or []
    values = [item.get("logprob") for item in content if item.get("logprob") is not None]
    return {
        "token_count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True
    )
    cases = build_cases(tokenizer)
    requests.get(f"{args.base_url}/flush_cache", timeout=30).raise_for_status()

    results = []
    for index, case in enumerate(cases):
        payload = {
            "model": MODEL_PATH,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0,
            "max_tokens": 64,
            "logprobs": True,
            "top_logprobs": 1,
            "chat_template_kwargs": {"thinking": False},
        }
        started = time.perf_counter()
        response = requests.post(
            f"{args.base_url}/v1/chat/completions", json=payload, timeout=900
        )
        latency = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        expected = case["code"]
        result = {
            **{k: v for k, v in case.items() if k != "prompt"},
            "response": text,
            "exact_match": text == expected,
            "contains_match": expected in text,
            "latency_seconds": latency,
            "server_prompt_tokens": (data.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
            "logprobs": extract_logprobs(choice),
        }
        results.append(result)
        print(
            f"[{index + 1:02d}/{len(cases)}] mode={args.mode} "
            f"target={case['target_tokens']} pos={case['position']:.1f} "
            f"prompt={result['server_prompt_tokens']} exact={result['exact_match']} "
            f"latency={latency:.2f}s",
            flush=True,
        )

    payload = {
        "mode": args.mode,
        "model": MODEL_PATH,
        "case_count": len(results),
        "exact_matches": sum(item["exact_match"] for item in results),
        "contains_matches": sum(item["contains_match"] for item in results),
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"completed mode={args.mode} exact={payload['exact_matches']}/{len(results)} "
        f"contains={payload['contains_matches']}/{len(results)}"
    )


if __name__ == "__main__":
    main()
