#!/usr/bin/env python3
"""Parse SGLang bench_serving logs and optionally compare two variants."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


PATTERNS = {
    "successful": r"Successful requests:\s+(\d+)",
    "throughput": r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "ttft_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
    "tpot_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
}


def parse_rows(root: Path, filename_regex: str):
    pattern = re.compile(filename_regex)
    rows = []
    for path in root.rglob("*.log"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        values = match.groupdict()
        variant = values.get("variant") or path.parent.name
        text = path.read_text(errors="replace")
        row = {
            "variant": variant,
            "workload": values.get("workload", ""),
            "concurrency": int(values["concurrency"]),
            "path": str(path),
        }
        for key, metric_pattern in PATTERNS.items():
            found = re.search(metric_pattern, text)
            row[key] = float(found.group(1)) if found else None
        rows.append(row)
    return sorted(
        rows, key=lambda row: (row["variant"], row["workload"], row["concurrency"])
    )


def geomean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def compare(rows, baseline_name, candidate_name):
    result = {}
    workloads = sorted({row["workload"] for row in rows})
    for workload in workloads:
        baseline = {
            row["concurrency"]: row
            for row in rows
            if row["variant"] == baseline_name and row["workload"] == workload
        }
        candidate = {
            row["concurrency"]: row
            for row in rows
            if row["variant"] == candidate_name and row["workload"] == workload
        }
        shared = sorted(baseline.keys() & candidate.keys())
        if not shared:
            continue
        throughput_ratios = {
            str(concurrency): candidate[concurrency]["throughput"]
            / baseline[concurrency]["throughput"]
            for concurrency in shared
        }
        result[workload] = {
            "throughput_ratios": throughput_ratios,
            "throughput_geomean_ratio": geomean(throughput_ratios.values()),
            "ttft_ratios": {
                str(concurrency): candidate[concurrency]["ttft_ms"]
                / baseline[concurrency]["ttft_ms"]
                for concurrency in shared
            },
            "tpot_ratios": {
                str(concurrency): candidate[concurrency]["tpot_ms"]
                / baseline[concurrency]["tpot_ms"]
                for concurrency in shared
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--comparison-json", type=Path)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    parser.add_argument(
        "--filename-regex",
        default=r"(?P<workload>.+)-c(?P<concurrency>\d+)\.log",
        help="Must provide workload/concurrency named groups; variant defaults to parent.",
    )
    args = parser.parse_args()

    rows = parse_rows(args.root, args.filename_regex)
    if not rows:
        raise SystemExit(f"no matching benchmark logs under {args.root}")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    comparisons = {}
    if args.baseline or args.candidate:
        if not args.baseline or not args.candidate or not args.comparison_json:
            parser.error(
                "--baseline, --candidate, and --comparison-json are required together"
            )
        comparisons = compare(rows, args.baseline, args.candidate)
        args.comparison_json.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_json.write_text(json.dumps(comparisons, indent=2) + "\n")

    print(json.dumps({"rows": len(rows), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
