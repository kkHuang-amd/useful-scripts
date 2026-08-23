#!/usr/bin/env python3
"""Summarize a tail window from a rocprofiler SQLite database."""

import argparse
import collections
import json
import sqlite3
from pathlib import Path

from kernel_families import classify_kernel, merge_intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-kernels", type=int, default=100)
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    end_ns = connection.execute("select max(end) from rocpd_op").fetchone()[0]
    start_ns = int(end_ns - args.duration * 1e9)
    rows = connection.execute(
        """
        select o.start, o.end, o.queueId, s.string
        from rocpd_op o
        join rocpd_string s on s.id = o.description_id
        where o.end > ? and o.start < ?
        """,
        (start_ns, end_ns),
    )

    intervals = []
    queue_intervals = collections.defaultdict(list)
    names = collections.defaultdict(lambda: [0, 0])
    families = collections.defaultdict(lambda: [0, 0])
    for raw_start, raw_end, queue, name in rows:
        start = max(raw_start, start_ns)
        end = min(raw_end, end_ns)
        if end <= start:
            continue
        duration = end - start
        intervals.append((start, end))
        queue_intervals[str(queue)].append((start, end))
        names[name][0] += duration
        names[name][1] += 1
        family = classify_kernel(name)
        families[family][0] += duration
        families[family][1] += 1
    connection.close()

    kernel_sum_ns = sum(value[0] for value in names.values())
    busy_union_ns = merge_intervals(intervals)
    result = {
        "database": str(args.database),
        "window_seconds": args.duration,
        "window_start_ns": start_ns,
        "window_end_ns": end_ns,
        "kernel_count": sum(value[1] for value in names.values()),
        "kernel_sum_ms": kernel_sum_ns / 1e6,
        "busy_union_ms": busy_union_ns / 1e6,
        "overlap_ms": (kernel_sum_ns - busy_union_ns) / 1e6,
        "overlap_fraction_of_sum": (
            (kernel_sum_ns - busy_union_ns) / kernel_sum_ns if kernel_sum_ns else 0
        ),
        "active_queue_count": len(queue_intervals),
        "queues": [
            {
                "queue": queue,
                "count": len(values),
                "sum_ms": sum(end - start for start, end in values) / 1e6,
                "busy_union_ms": merge_intervals(values) / 1e6,
            }
            for queue, values in sorted(
                queue_intervals.items(),
                key=lambda item: sum(end - start for start, end in item[1]),
                reverse=True,
            )
        ],
        "families": [
            {
                "family": family,
                "duration_ms": values[0] / 1e6,
                "count": values[1],
                "pct_kernel_sum": (
                    100 * values[0] / kernel_sum_ns if kernel_sum_ns else 0
                ),
            }
            for family, values in sorted(
                families.items(), key=lambda item: item[1][0], reverse=True
            )
        ],
        "top_kernels": [
            {"name": name, "duration_ms": values[0] / 1e6, "count": values[1]}
            for name, values in sorted(
                names.items(), key=lambda item: item[1][0], reverse=True
            )[: args.top_kernels]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"kernels={result['kernel_count']} "
        f"sum_ms={result['kernel_sum_ms']:.3f} "
        f"busy_union_ms={result['busy_union_ms']:.3f} "
        f"queues={result['active_queue_count']}"
    )


if __name__ == "__main__":
    main()
