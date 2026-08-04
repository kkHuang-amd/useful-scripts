#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


METRICS = (
    "input_throughput",
    "total_throughput",
    "median_ttft_ms",
    "median_tpot_ms",
    "median_e2e_latency_ms",
)


def load_results(path: Path) -> dict[int, dict]:
    results = {}
    for result_path in sorted(path.glob("*.jsonl")):
        for line in result_path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            results[int(record["max_concurrency"])] = record
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    cp = load_results(args.cp)
    concurrencies = sorted(set(baseline) & set(cp))
    if not concurrencies:
        raise SystemExit("no matching baseline/CP concurrency results")

    lines = [
        "# Kimi-K3 Prefill CP A/B summary",
        "",
        "| Concurrency | Metric | Baseline | Prefill CP | Delta |",
        "|---:|:---|---:|---:|---:|",
    ]
    for concurrency in concurrencies:
        for metric in METRICS:
            base_value = float(baseline[concurrency][metric])
            cp_value = float(cp[concurrency][metric])
            delta = ((cp_value / base_value) - 1) * 100 if base_value else float("nan")
            lines.append(
                f"| {concurrency} | `{metric}` | {base_value:.3f} | "
                f"{cp_value:.3f} | {delta:+.2f}% |"
            )

    lines.extend(
        [
            "",
            f"- Baseline artifacts: `{args.baseline}`",
            f"- Prefill CP artifacts: `{args.cp}`",
            "",
        ]
    )
    args.output.write_text("\n".join(lines))
    print(args.output)


if __name__ == "__main__":
    main()
