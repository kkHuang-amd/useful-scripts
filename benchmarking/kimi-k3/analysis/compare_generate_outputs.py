#!/usr/bin/env python3
"""Compare paired SGLang /generate responses and exposed top-k probabilities."""

import argparse
import json
import math
from pathlib import Path


def load(path):
    return {
        row["index"]: row.get("response", row)
        for row in json.loads(path.read_text())
    }


def topk_cosine(left, right):
    left_map = {int(token): math.exp(float(score)) for score, token, *_ in left}
    right_map = {int(token): math.exp(float(score)) for score, token, *_ in right}
    tokens = left_map.keys() | right_map.keys()
    dot = sum(left_map.get(token, 0.0) * right_map.get(token, 0.0) for token in tokens)
    left_norm = math.sqrt(sum(value * value for value in left_map.values()))
    right_norm = math.sqrt(sum(value * value for value in right_map.values()))
    return dot / (left_norm * right_norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    shared = sorted(baseline.keys() & candidate.keys())
    if not shared:
        raise SystemExit("no paired request indices")

    first_token_matches = 0
    all_token_matches = 0
    all_token_count = 0
    first_token_cosines = []
    retractions = []
    for index in shared:
        left = baseline[index]
        right = candidate[index]
        left_ids = left["output_ids"]
        right_ids = right["output_ids"]
        first_token_matches += int(left_ids[0] == right_ids[0])
        all_token_matches += sum(a == b for a, b in zip(left_ids, right_ids))
        all_token_count += min(len(left_ids), len(right_ids))
        first_token_cosines.append(
            topk_cosine(
                left["meta_info"]["output_top_logprobs"][0],
                right["meta_info"]["output_top_logprobs"][0],
            )
        )
        retractions.extend(
            (
                left["meta_info"].get("num_retractions", 0),
                right["meta_info"].get("num_retractions", 0),
            )
        )

    summary = {
        "requests": len(shared),
        "first_token_top1_match": first_token_matches / len(shared),
        "all_generated_token_match": all_token_matches / all_token_count,
        "first_token_topk_probability_cosine_min": min(first_token_cosines),
        "first_token_topk_probability_cosine_mean": (
            sum(first_token_cosines) / len(first_token_cosines)
        ),
        "max_num_retractions": max(retractions),
        "note": (
            "Cosine uses the union of endpoint-exposed top-k probabilities, "
            "not full logits."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
