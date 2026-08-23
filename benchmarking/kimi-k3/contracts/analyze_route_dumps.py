#!/usr/bin/env python3
"""Aggregate and exactly align two Kimi-K3 fused-MoE route dump roots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_root(root: Path, engine: str) -> dict[tuple[str, int], dict]:
    records: dict[tuple[str, int], dict] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("schema") != "k3-route-dump-v1":
            continue
        required_arming = (
            "arm_file",
            "arm_timestamp_utc",
            "arm_time_ns",
            "arm_monotonic_ns",
            "dump_timestamp_utc",
            "dump_time_ns",
            "dump_monotonic_ns",
        )
        if payload.get("armed") is not True or any(
            field not in payload for field in required_arming
        ):
            raise ValueError(
                f"{engine}: old or unarmed route dump is invalid: {path}"
            )
        if payload["dump_time_ns"] < payload["arm_time_ns"]:
            raise ValueError(f"{engine}: dump wall-clock predates arming: {path}")
        if payload["dump_monotonic_ns"] < payload["arm_monotonic_ns"]:
            raise ValueError(f"{engine}: dump monotonic time predates arming: {path}")
        rank_value = payload.get("rank")
        if rank_value is None:
            rank_value = payload.get("device_index")
        rank = str(rank_value)
        call_index = int(payload["call_index"])
        key = (rank, call_index)
        if key in records:
            raise ValueError(
                f"{engine}: duplicate rank/call {key}: "
                f"{records[key]['_path']} and {path}"
            )
        payload["_path"] = str(path)
        payload["_engine"] = engine
        records[key] = payload
    if not records:
        raise ValueError(f"{engine}: no k3-route-dump-v1 JSON files under {root}")
    return records


def arming_metadata(records: dict[tuple[str, int], dict], engine: str) -> dict:
    identities = {
        (
            record["arm_file"],
            record["arm_timestamp_utc"],
            int(record["arm_time_ns"]),
            int(record["arm_monotonic_ns"]),
        )
        for record in records.values()
    }
    if len(identities) != 1:
        raise ValueError(
            f"{engine}: expected one shared arming event, found {len(identities)}"
        )
    arm_file, timestamp, time_ns, monotonic_ns = identities.pop()
    return {
        "armed": True,
        "arm_file": arm_file,
        "armed_at_utc": timestamp,
        "armed_time_ns": time_ns,
        "armed_monotonic_ns": monotonic_ns,
        "validated_dump_count": len(records),
        "all_dumps_after_arm": True,
    }


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def aggregate(records: dict[tuple[str, int], dict]) -> dict[str, dict]:
    by_rank: dict[str, list[dict]] = defaultdict(list)
    for (rank, _), record in records.items():
        by_rank[rank].append(record)
    result = {}
    for rank, rows in sorted(by_rank.items()):
        rows.sort(key=lambda row: row["call_index"])
        active_values = [row["unique_active_experts"] for row in rows]
        block_values = [row["bm32_padded_blocks"] for row in rows]
        result[rank] = {
            "call_count": len(rows),
            "call_indices": [row["call_index"] for row in rows],
            "active_experts": numeric_summary(active_values),
            "active_experts_histogram": {
                str(value): active_values.count(value)
                for value in sorted(set(active_values))
            },
            "bm32_padded_blocks": numeric_summary(block_values),
            "bm32_padded_blocks_histogram": {
                str(value): block_values.count(value)
                for value in sorted(set(block_values))
            },
            "mean_nonzero_routes_per_expert": numeric_summary(
                [row["nonzero_routes"]["mean"] for row in rows]
            ),
            "total_routes": numeric_summary([row["total_routes"] for row in rows]),
        }
    return result


def dense_counts(record: dict) -> list[int]:
    if "full_bincount" in record:
        return [int(value) for value in record["full_bincount"]]
    size = int(record["expert_count_size"])
    result = [0] * size
    for item in record["expert_counts"]:
        result[int(item["expert"])] = int(item["routes"])
    return result


def layer_row(engine: str, rank: str, layer: int, record: dict) -> dict:
    routes = record["nonzero_routes"]
    return {
        "engine": engine,
        "rank": rank,
        "layer_call_index": layer,
        "total_routes": record["total_routes"],
        "active_experts": record["unique_active_experts"],
        "routes_per_active_expert_min": routes["min"],
        "routes_per_active_expert_max": routes["max"],
        "routes_per_active_expert_mean": routes["mean"],
        "bm32_padded_blocks": record["bm32_padded_blocks"],
        "bm32_padded_routes": record["bm32_padded_routes"],
        "env_mode": record.get("env_mode"),
        "quant_type": record.get("quant_type"),
        "armed": record["armed"],
        "arm_timestamp_utc": record["arm_timestamp_utc"],
        "dump_timestamp_utc": record["dump_timestamp_utc"],
        "dump_monotonic_ns": record["dump_monotonic_ns"],
        "source": record["_path"],
    }


def compare_pair(left: dict, right: dict, rank: str, layer: int) -> dict:
    left_counts = dense_counts(left)
    right_counts = dense_counts(right)
    width = max(len(left_counts), len(right_counts))
    left_counts.extend([0] * (width - len(left_counts)))
    right_counts.extend([0] * (width - len(right_counts)))
    differences = [right_counts[i] - left_counts[i] for i in range(width)]
    return {
        "rank": rank,
        "layer_call_index": layer,
        "total_routes_equal": left["total_routes"] == right["total_routes"],
        "expert_counts_exact_match": left_counts == right_counts,
        "active_experts_delta": (
            right["unique_active_experts"] - left["unique_active_experts"]
        ),
        "bm32_padded_blocks_delta": (
            right["bm32_padded_blocks"] - left["bm32_padded_blocks"]
        ),
        "mean_routes_per_active_expert_delta": (
            right["nonzero_routes"]["mean"] - left["nonzero_routes"]["mean"]
        ),
        "expert_count_l1_delta": sum(abs(value) for value in differences),
        "expert_count_max_abs_delta": max(map(abs, differences), default=0),
    }


def markdown_report(payload: dict) -> str:
    alignment = payload["alignment"]
    lines = [
        "# Kimi-K3 current-route diagnostic",
        "",
        "> Eager contract diagnostic only. These artifacts contain no valid timing comparison.",
        "",
        "## Alignment",
        "",
        "- Armed route contract: `True` (all legacy unarmed dumps rejected)",
        f"- {payload['engines']['left']} armed at: "
        f"`{payload['arming'][payload['engines']['left']]['armed_at_utc']}`",
        f"- {payload['engines']['right']} armed at: "
        f"`{payload['arming'][payload['engines']['right']]['armed_at_utc']}`",
        f"- Exact rank/call alignment: `{alignment['exact']}`",
        f"- Aligned calls: `{alignment['aligned_count']}`",
        f"- Left-only calls: `{len(alignment['left_only'])}`",
        f"- Right-only calls: `{len(alignment['right_only'])}`",
        "",
        "## Per-rank summary",
        "",
        "| Engine | Rank | Calls | Active experts mean | BM32 blocks mean | Routes/active expert mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for engine, ranks in payload["aggregates"].items():
        for rank, values in ranks.items():
            lines.append(
                f"| {engine} | {rank} | {values['call_count']} | "
                f"{values['active_experts']['mean']:.3f} | "
                f"{values['bm32_padded_blocks']['mean']:.3f} | "
                f"{values['mean_nonzero_routes_per_expert']['mean']:.3f} |"
            )
    cross = payload["cross_engine"]
    lines.extend(
        [
            "",
            "## Cross-engine deltas",
            "",
            f"- Layers with identical expert counts: "
            f"`{cross['exact_expert_count_matches']}/{alignment['aligned_count']}`",
            f"- Mean active-expert delta (right-left): "
            f"`{cross['active_experts_delta']['mean']}`",
            f"- Mean BM32 padded-block delta (right-left): "
            f"`{cross['bm32_padded_blocks_delta']['mean']}`",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate two route-dump roots and align every rank/call exactly. "
            "Writes JSON, CSV, and Markdown into --output-dir."
        )
    )
    parser.add_argument("left_root", type=Path)
    parser.add_argument("right_root", type=Path)
    parser.add_argument("--left-name", default="sglang")
    parser.add_argument("--right-name", default="atom")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write reports and exit zero even when rank/call alignment is incomplete",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        left = load_root(args.left_root, args.left_name)
        right = load_root(args.right_root, args.right_name)
        left_arming = arming_metadata(left, args.left_name)
        right_arming = arming_metadata(right, args.right_name)
    except ValueError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2

    left_keys = set(left)
    right_keys = set(right)
    aligned_keys = sorted(left_keys & right_keys, key=lambda item: (item[0], item[1]))
    left_only = sorted(left_keys - right_keys, key=lambda item: (item[0], item[1]))
    right_only = sorted(right_keys - left_keys, key=lambda item: (item[0], item[1]))
    comparisons = [
        compare_pair(left[key], right[key], key[0], key[1]) for key in aligned_keys
    ]
    payload = {
        "schema": "k3-route-analysis-v2",
        "engines": {
            "left": args.left_name,
            "right": args.right_name,
        },
        "roots": {
            args.left_name: str(args.left_root.resolve()),
            args.right_name: str(args.right_root.resolve()),
        },
        "arming": {
            args.left_name: left_arming,
            args.right_name: right_arming,
        },
        "alignment": {
            "exact": not left_only and not right_only,
            "aligned_count": len(aligned_keys),
            "left_only": [
                {"rank": rank, "layer_call_index": layer}
                for rank, layer in left_only
            ],
            "right_only": [
                {"rank": rank, "layer_call_index": layer}
                for rank, layer in right_only
            ],
        },
        "aggregates": {
            args.left_name: aggregate(left),
            args.right_name: aggregate(right),
        },
        "per_engine_rank_layer": {
            args.left_name: [
                layer_row(args.left_name, rank, layer, record)
                for (rank, layer), record in sorted(left.items())
            ],
            args.right_name: [
                layer_row(args.right_name, rank, layer, record)
                for (rank, layer), record in sorted(right.items())
            ],
        },
        "cross_engine": {
            "exact_expert_count_matches": sum(
                row["expert_counts_exact_match"] for row in comparisons
            ),
            "active_experts_delta": numeric_summary(
                [row["active_experts_delta"] for row in comparisons]
            ),
            "bm32_padded_blocks_delta": numeric_summary(
                [row["bm32_padded_blocks_delta"] for row in comparisons]
            ),
            "mean_routes_per_active_expert_delta": numeric_summary(
                [row["mean_routes_per_active_expert_delta"] for row in comparisons]
            ),
            "per_layer": comparisons,
        },
    }
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "route-analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    rows = [
        layer_row(args.left_name, rank, layer, record)
        for (rank, layer), record in sorted(left.items())
    ] + [
        layer_row(args.right_name, rank, layer, record)
        for (rank, layer), record in sorted(right.items())
    ]
    with (output_dir / "route-layers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if comparisons:
        with (output_dir / "route-deltas.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    (output_dir / "route-analysis.md").write_text(markdown_report(payload))
    print(
        f"ROUTE_ANALYSIS_OK aligned={len(aligned_keys)} "
        f"exact={payload['alignment']['exact']} output_dir={output_dir}"
    )
    if not payload["alignment"]["exact"] and not args.allow_incomplete:
        print("ERROR rank/call alignment is incomplete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
