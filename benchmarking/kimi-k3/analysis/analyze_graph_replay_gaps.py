#!/usr/bin/env python3
"""Analyze CPU launch cadence and graph-external gaps in decode replay traces.

The host launch cadence and selected replay GPU span come from different
profiler timing domains. Their difference is a reconciliation diagnostic, not
a directly measured idle interval or evidence of CPU/GPU overlap.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import re
import statistics
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from trace_annotations import category_tokens, collect_annotations, decode_annotations
from trace_correlation import collect_timed_events, graph_launch_events


CASES = ("sglang-c2", "sglang-c64", "atom-c2", "atom-c64")
HOST_GROUPS = (
    "sync_query",
    "memcpy",
    "graph_launch",
    "scheduler_user_annotation",
    "cpu_op",
    "other_host_api",
)
TIMING_DOMAIN_CAVEAT = (
    "Launch cadence is measured from CPU hipGraphLaunch/cudaGraphLaunch "
    "timestamps, while graph span is the correlated GPU activity span from "
    "rank-N-selected-step.json. Cadence minus graph span is a cross-domain "
    "reconciliation residual, not a directly observed idle interval."
)
OVERLAP_CAVEAT = (
    "Host category and name durations are clipped unions or sums within "
    "launch-to-launch intervals. Categories may nest or overlap; they must not "
    "be added to infer GPU overlap or a critical path."
)


class AnalysisError(ValueError):
    """Raised for incomplete or ambiguous replay evidence."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure decode graph-launch cadence for four TP8 production cases "
            "and reconcile it with selected GPU graph spans and endpoint TPOT."
        )
    )
    parser.add_argument("result_root", type=Path, help="four-case trace result root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: RESULT_ROOT/graph-external)",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=CASES,
        dest="cases",
        help="case to analyze; repeat as needed (default: all four)",
    )
    parser.add_argument(
        "--endpoint-summary",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help=(
            "common-client summary JSON; repeat paths for multi-round cases "
            "and take the median of median_tpot_ms"
        ),
    )
    return parser.parse_args(argv)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise AnalysisError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[float], *, include_p95: bool = True) -> dict:
    numbers = [float(value) for value in values]
    if not numbers:
        return {"count": 0, "median": None, "min": None, "max": None, "p95": None}
    result = {
        "count": len(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }
    if include_p95:
        result["p95"] = percentile(numbers, 0.95)
    return result


def tp8_distribution(values: Iterable[float]) -> dict:
    numbers = [float(value) for value in values]
    result = distribution(numbers, include_p95=False)
    mean = statistics.fmean(numbers) if numbers else 0.0
    result["cv"] = statistics.pstdev(numbers) / abs(mean) if mean else None
    return result


def load_trace(path: Path) -> list[Mapping]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise AnalysisError(f"{path}: expected a Chrome trace event list")
    return events


def _interval(event: Mapping) -> tuple[float, float] | None:
    if event.get("ph") != "X" or not event.get("dur"):
        return None
    start = float(event.get("ts", 0.0))
    return start, start + float(event["dur"])


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _clip(
    start: float, end: float, window_start: float, window_end: float
) -> tuple[float, float] | None:
    clipped = max(start, window_start), min(end, window_end)
    return clipped if clipped[1] > clipped[0] else None


def classify_host_event(event: Mapping) -> str | None:
    tokens = category_tokens(event)
    name = str(event.get("name", ""))
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    if "user_annotation" in tokens:
        return "scheduler_user_annotation"
    if "cpu_op" in tokens:
        return "cpu_op"
    if not tokens & {"cuda_runtime", "cuda_driver"}:
        return None
    if normalized.startswith(("hipgraphlaunch", "cudagraphlaunch")):
        return "graph_launch"
    if any(
        marker in normalized
        for marker in (
            "streamsynchronize",
            "eventsynchronize",
            "devicesynchronize",
            "streamquery",
            "eventquery",
            "streamwaitevent",
        )
    ):
        return "sync_query"
    if "memcpy" in normalized:
        return "memcpy"
    return "other_host_api"


def _is_d2h(event: Mapping) -> bool:
    text = " ".join(
        [str(event.get("name", ""))]
        + [f"{key}={value}" for key, value in (event.get("args", {}) or {}).items()]
    ).lower()
    normalized = re.sub(r"[^a-z0-9]", "", text)
    return any(marker in normalized for marker in ("devicetohost", "dtoh", "d2h"))


def select_decode_launches(
    events: Sequence[Mapping],
) -> tuple[list, list, str]:
    annotations = collect_annotations(events)
    decode, source = decode_annotations(annotations)
    cpu_decode = [item for item in decode if item.category == "user_annotation"]
    # decode_annotations prefers projected GPU annotations. Host containment
    # must explicitly use the CPU copies when both clock domains are present.
    if not cpu_decode:
        cpu_decode = [
            item
            for item in annotations
            if item.category == "user_annotation"
            and item.name.startswith(("step[DECODE", "decode["))
        ]
        source = "user_annotation" if cpu_decode else source
    launches = graph_launch_events(collect_timed_events(events))
    selected = []
    owners = []
    for launch in launches:
        matching = [
            annotation
            for annotation in cpu_decode
            if annotation.start <= launch.start <= annotation.end
            and (
                annotation.pid is None
                or launch.pid is None
                or annotation.pid == launch.pid
            )
            and (
                annotation.tid is None
                or launch.tid is None
                or annotation.tid == launch.tid
            )
        ]
        if not matching:
            continue
        owner = min(matching, key=lambda item: (item.duration, item.start, item.name))
        selected.append(launch)
        owners.append(owner)
    ordered = sorted(zip(selected, owners), key=lambda item: item[0].start)
    if len(ordered) < 2:
        raise AnalysisError(
            "fewer than two hipGraphLaunch/cudaGraphLaunch events were found "
            "inside CPU decode annotations"
        )
    return (
        [item[0] for item in ordered],
        [item[1] for item in ordered],
        source or "user_annotation",
    )


def aggregate_host_intervals(
    events: Sequence[Mapping], launches: Sequence
) -> tuple[dict, list[dict]]:
    timed = []
    for event in events:
        interval = _interval(event)
        group = classify_host_event(event)
        if interval is not None and group is not None:
            timed.append((event, interval, group))

    interval_rows = []
    all_names: dict[str, collections.Counter] = {
        group: collections.Counter() for group in HOST_GROUPS
    }
    for index, (left, right) in enumerate(zip(launches, launches[1:])):
        window_start, window_end = left.start, right.start
        grouped_intervals = {group: [] for group in HOST_GROUPS}
        grouped_sums = collections.Counter()
        active_intervals = []
        d2h_intervals = []
        for event, (start, end), group in timed:
            clipped = _clip(start, end, window_start, window_end)
            if clipped is None:
                continue
            duration = clipped[1] - clipped[0]
            grouped_intervals[group].append(clipped)
            grouped_sums[group] += duration
            all_names[group][str(event.get("name", ""))] += duration
            if group != "scheduler_user_annotation":
                active_intervals.append(clipped)
            if group == "memcpy" and _is_d2h(event):
                d2h_intervals.append(clipped)
        cadence = window_end - window_start
        row = {
            "index": index,
            "start_us": window_start,
            "end_us": window_end,
            "cadence_us": cadence,
            "groups": {
                group: {
                    "union_us": _union_duration(grouped_intervals[group]),
                    "duration_sum_us": grouped_sums[group],
                }
                for group in HOST_GROUPS
            },
            "d2h_union_us": _union_duration(d2h_intervals),
            "idle_unattributed_us": max(0.0, cadence - _union_duration(active_intervals)),
        }
        interval_rows.append(row)

    aggregate = {}
    for group in HOST_GROUPS:
        aggregate[group] = {
            "union_us_per_interval": distribution(
                [row["groups"][group]["union_us"] for row in interval_rows]
            ),
            "duration_sum_us_per_interval": distribution(
                [row["groups"][group]["duration_sum_us"] for row in interval_rows]
            ),
            "top_names_by_clipped_duration": [
                {"name": name, "duration_sum_us": duration}
                for name, duration in all_names[group].most_common(20)
            ],
        }
    aggregate["d2h_union_us_per_interval"] = distribution(
        [row["d2h_union_us"] for row in interval_rows]
    )
    aggregate["idle_unattributed_us_per_interval"] = distribution(
        [row["idle_unattributed_us"] for row in interval_rows]
    )
    return aggregate, interval_rows


def annotation_consistency(launches: Sequence, owners: Sequence) -> dict:
    unique = []
    seen = set()
    for owner in owners:
        key = (owner.start, owner.end, owner.name, owner.pid, owner.tid)
        if key not in seen:
            seen.add(key)
            unique.append(owner)
    unique.sort(key=lambda item: item.start)
    launch_cadences = [
        right.start - left.start for left, right in zip(launches, launches[1:])
    ]
    annotation_cadences = [
        right.start - left.start for left, right in zip(unique, unique[1:])
    ]
    paired_deltas = []
    owner_by_launch = list(zip(launches, owners))
    for (left_launch, left_owner), (right_launch, right_owner) in zip(
        owner_by_launch, owner_by_launch[1:]
    ):
        if left_owner != right_owner:
            paired_deltas.append(
                (right_owner.start - left_owner.start)
                - (right_launch.start - left_launch.start)
            )
    launch_median = statistics.median(launch_cadences)
    annotation_median = (
        statistics.median(annotation_cadences) if annotation_cadences else None
    )
    relative_delta = (
        abs(annotation_median - launch_median) / launch_median
        if annotation_median is not None and launch_median
        else None
    )
    return {
        "launch_count": len(launches),
        "unique_decode_annotation_count": len(unique),
        "launch_annotation_coverage_pct": 100.0 * len(owners) / len(launches),
        "annotation_name_patterns": sorted(
            {
                re.sub(r"\d+", "N", owner.name)
                for owner in unique
            }
        ),
        "annotation_start_cadence_us": distribution(annotation_cadences),
        "annotation_minus_launch_cadence_us": distribution(paired_deltas),
        "median_relative_difference_pct": (
            100.0 * relative_delta if relative_delta is not None else None
        ),
        "consistent_with_launch_cadence": (
            len(unique) == len(launches)
            and relative_delta is not None
            and relative_delta <= 0.05
        ),
    }


def analyze_rank(
    events: Sequence[Mapping], selected_step: Mapping, *, rank: int
) -> dict:
    launches, owners, source = select_decode_launches(events)
    cadences = [
        right.start - left.start for left, right in zip(launches, launches[1:])
    ]
    graph_span = float(selected_step["gpu_timestamp_span"]["duration_us"])
    external_gaps = [cadence - graph_span for cadence in cadences]
    host_aggregate, interval_rows = aggregate_host_intervals(events, launches)
    return {
        "rank": rank,
        "trace": selected_step.get("trace"),
        "selected_step": {
            "graph_launch_api": selected_step.get("graph_launch_api"),
            "graph_launch_correlation": selected_step.get(
                "graph_launch_correlation"
            ),
            "gpu_graph_span_us": graph_span,
        },
        "decode_annotation_source": source,
        "launch_count": len(launches),
        "launch_interval_count": len(cadences),
        "cadence_us": distribution(cadences),
        "graph_external_gap_us": distribution(external_gaps),
        "annotation_consistency": annotation_consistency(launches, owners),
        "host_interval_aggregate": host_aggregate,
        "intervals": interval_rows,
        "timing_domain_caveat": TIMING_DOMAIN_CAVEAT,
        "overlap_caveat": OVERLAP_CAVEAT,
    }


def aggregate_case(
    case: str, rank_reports: Sequence[Mapping], endpoint_tpot_ms: float | None
) -> dict:
    if len(rank_reports) != 8 or sorted(item["rank"] for item in rank_reports) != list(
        range(8)
    ):
        raise AnalysisError(f"{case}: expected exactly ranks 0-7")
    fields = {
        "cadence_us": [item["cadence_us"]["median"] for item in rank_reports],
        "graph_span_us": [
            item["selected_step"]["gpu_graph_span_us"] for item in rank_reports
        ],
        "graph_external_gap_us": [
            item["graph_external_gap_us"]["median"] for item in rank_reports
        ],
        "launch_count": [item["launch_count"] for item in rank_reports],
        "launch_interval_count": [
            item["launch_interval_count"] for item in rank_reports
        ],
        "idle_unattributed_us": [
            item["host_interval_aggregate"][
                "idle_unattributed_us_per_interval"
            ]["median"]
            for item in rank_reports
        ],
        "d2h_us": [
            item["host_interval_aggregate"]["d2h_union_us_per_interval"]["median"]
            for item in rank_reports
        ],
    }
    tp8 = {name: tp8_distribution(values) for name, values in fields.items()}
    host = {}
    for group in HOST_GROUPS:
        host[group] = {
            "union_us": tp8_distribution(
                [
                    item["host_interval_aggregate"][group][
                        "union_us_per_interval"
                    ]["median"]
                    for item in rank_reports
                ]
            ),
            "duration_sum_us": tp8_distribution(
                [
                    item["host_interval_aggregate"][group][
                        "duration_sum_us_per_interval"
                    ]["median"]
                    for item in rank_reports
                ]
            ),
        }
    cadence_ms = tp8["cadence_us"]["median"] / 1000.0
    graph_ms = tp8["graph_span_us"]["median"] / 1000.0
    gap_ms = tp8["graph_external_gap_us"]["median"] / 1000.0
    reconciliation = {
        "endpoint_median_tpot_ms": endpoint_tpot_ms,
        "tp8_median_launch_cadence_ms": cadence_ms,
        "tp8_median_gpu_graph_span_ms": graph_ms,
        "tp8_median_cross_domain_graph_external_gap_ms": gap_ms,
        "cadence_minus_endpoint_tpot_ms": (
            cadence_ms - endpoint_tpot_ms
            if endpoint_tpot_ms is not None
            else None
        ),
        "cadence_to_endpoint_tpot_ratio": (
            cadence_ms / endpoint_tpot_ms
            if endpoint_tpot_ms
            else None
        ),
        "interpretation": (
            "Endpoint TPOT and launch cadence are compared once as independent "
            "step-rate measurements. GPU graph span is not added to cadence, "
            "and no CPU/GPU overlap is inferred."
        ),
    }
    return {
        "schema_version": 1,
        "case": case,
        "rank_count": len(rank_reports),
        "tp8_rank_aggregate": tp8,
        "host_category_tp8_rank_aggregate": host,
        "host_category_definitions": {
            "memcpy": "all CUDA/HIP memcpy APIs",
            "d2h_us": "memcpy subset explicitly identified as device-to-host",
            "idle_unattributed_us": (
                "cadence not covered by timed CPU op or CUDA/HIP API events; "
                "user-annotation scopes are excluded from the active union"
            ),
        },
        "annotation_consistency": {
            "all_ranks_consistent": all(
                item["annotation_consistency"]["consistent_with_launch_cadence"]
                for item in rank_reports
            ),
            "per_rank": [
                {
                    "rank": item["rank"],
                    **item["annotation_consistency"],
                }
                for item in rank_reports
            ],
        },
        "endpoint_reconciliation": reconciliation,
        "ranks": list(rank_reports),
        "timing_domain_caveat": TIMING_DOMAIN_CAVEAT,
        "overlap_caveat": OVERLAP_CAVEAT,
    }


def compare_cases(cases: Mapping[str, Mapping]) -> dict:
    rows = {}
    for concurrency in (2, 64):
        left = cases[f"sglang-c{concurrency}"]
        right = cases[f"atom-c{concurrency}"]
        left_recon = left["endpoint_reconciliation"]
        right_recon = right["endpoint_reconciliation"]
        rows[f"c{concurrency}"] = {
            "sglang": left_recon,
            "atom": right_recon,
            "atom_minus_sglang": {
                "launch_cadence_ms": (
                    right_recon["tp8_median_launch_cadence_ms"]
                    - left_recon["tp8_median_launch_cadence_ms"]
                ),
                "gpu_graph_span_ms": (
                    right_recon["tp8_median_gpu_graph_span_ms"]
                    - left_recon["tp8_median_gpu_graph_span_ms"]
                ),
                "cross_domain_graph_external_gap_ms": (
                    right_recon[
                        "tp8_median_cross_domain_graph_external_gap_ms"
                    ]
                    - left_recon[
                        "tp8_median_cross_domain_graph_external_gap_ms"
                    ]
                ),
                "endpoint_median_tpot_ms": (
                    right_recon["endpoint_median_tpot_ms"]
                    - left_recon["endpoint_median_tpot_ms"]
                    if right_recon["endpoint_median_tpot_ms"] is not None
                    and left_recon["endpoint_median_tpot_ms"] is not None
                    else None
                ),
            },
        }
    return {
        "schema_version": 1,
        "comparisons": rows,
        "timing_domain_caveat": TIMING_DOMAIN_CAVEAT,
        "interpretation": (
            "Deltas compare like metrics only. Graph span and graph-external "
            "residual are a decomposition of cadence for reconciliation and "
            "are not added to endpoint TPOT."
        ),
    }


def _parse_endpoint_summaries(
    entries: Sequence[str],
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = collections.defaultdict(list)
    for entry in entries:
        case, separator, raw_path = entry.partition("=")
        if not separator or case not in CASES or not raw_path:
            raise AnalysisError(
                f"invalid --endpoint-summary {entry!r}; expected CASE=PATH"
            )
        result[case].append(Path(raw_path))
    return result


def _endpoint_tpot(paths: Sequence[Path]) -> float | None:
    if not paths:
        return None
    values = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values.append(float(payload["median_tpot_ms"]))
    return statistics.median(values)


def _write_csv(path: Path, cases: Mapping[str, Mapping]) -> None:
    fields = [
        "case",
        "metric",
        "median",
        "min",
        "max",
        "cv",
        "unit",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, result in cases.items():
            for metric in ("cadence_us", "graph_span_us", "graph_external_gap_us"):
                values = result["tp8_rank_aggregate"][metric]
                writer.writerow(
                    {
                        "case": case,
                        "metric": metric,
                        "median": values["median"],
                        "min": values["min"],
                        "max": values["max"],
                        "cv": values["cv"],
                        "unit": "us",
                    }
                )
            for group, values in result[
                "host_category_tp8_rank_aggregate"
            ].items():
                item = values["union_us"]
                writer.writerow(
                    {
                        "case": case,
                        "metric": f"host_{group}_union_us",
                        "median": item["median"],
                        "min": item["min"],
                        "max": item["max"],
                        "cv": item["cv"],
                        "unit": "us",
                    }
                )
            for metric in ("idle_unattributed_us", "d2h_us"):
                item = result["tp8_rank_aggregate"][metric]
                writer.writerow(
                    {
                        "case": case,
                        "metric": metric,
                        "median": item["median"],
                        "min": item["min"],
                        "max": item["max"],
                        "cv": item["cv"],
                        "unit": "us",
                    }
                )
            endpoint_tpot = result["endpoint_reconciliation"][
                "endpoint_median_tpot_ms"
            ]
            writer.writerow(
                {
                    "case": case,
                    "metric": "endpoint_median_tpot_ms",
                    "median": endpoint_tpot,
                    "min": endpoint_tpot,
                    "max": endpoint_tpot,
                    "cv": 0.0 if endpoint_tpot is not None else None,
                    "unit": "ms",
                }
            )


def _write_markdown(path: Path, cases: Mapping[str, Mapping], comparison: Mapping) -> None:
    lines = [
        "# Production graph replay cadence and external-gap analysis",
        "",
        "| case | cadence ms | GPU graph span ms | external residual ms | endpoint TPOT ms | cadence−TPOT ms | annotation consistent |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for case, result in cases.items():
        recon = result["endpoint_reconciliation"]
        endpoint = recon["endpoint_median_tpot_ms"]
        delta = recon["cadence_minus_endpoint_tpot_ms"]
        lines.append(
            "| {case} | {cadence:.3f} | {graph:.3f} | {gap:.3f} | "
            "{endpoint} | {delta} | {consistent} |".format(
                case=case,
                cadence=recon["tp8_median_launch_cadence_ms"],
                graph=recon["tp8_median_gpu_graph_span_ms"],
                gap=recon["tp8_median_cross_domain_graph_external_gap_ms"],
                endpoint=f"{endpoint:.3f}" if endpoint is not None else "n/a",
                delta=f"{delta:.3f}" if delta is not None else "n/a",
                consistent=(
                    "yes"
                    if result["annotation_consistency"]["all_ranks_consistent"]
                    else "no"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Host intervals (TP8 median of rank medians)",
            "",
            "| case | sync/query ms | memcpy ms | explicit D2H ms | graph-launch API ms | CPU-op union ms | idle/unattributed ms |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case, result in cases.items():
        host = result["host_category_tp8_rank_aggregate"]
        rank = result["tp8_rank_aggregate"]
        lines.append(
            f"| {case} | {host['sync_query']['union_us']['median'] / 1000:.3f} "
            f"| {host['memcpy']['union_us']['median'] / 1000:.3f} "
            f"| {rank['d2h_us']['median'] / 1000:.3f} "
            f"| {host['graph_launch']['union_us']['median'] / 1000:.3f} "
            f"| {host['cpu_op']['union_us']['median'] / 1000:.3f} "
            f"| {rank['idle_unattributed_us']['median'] / 1000:.3f} |"
        )
    lines.extend(
        [
            "",
            "Explicit D2H is zero when profiler API names/arguments do not encode "
            "copy direction; this does not prove that no device-to-host copy occurred.",
            "",
            "## Engine deltas (ATOM − SGLang)",
            "",
        ]
    )
    for concurrency, values in comparison["comparisons"].items():
        delta = values["atom_minus_sglang"]
        lines.append(
            f"- {concurrency.upper()}: cadence {delta['launch_cadence_ms']:+.3f} ms; "
            f"GPU graph span {delta['gpu_graph_span_ms']:+.3f} ms; "
            "cross-domain external residual "
            f"{delta['cross_domain_graph_external_gap_ms']:+.3f} ms; "
            f"endpoint TPOT {delta['endpoint_median_tpot_ms']:+.3f} ms."
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            f"- {TIMING_DOMAIN_CAVEAT}",
            f"- {OVERLAP_CAVEAT}",
            "- Endpoint TPOT is reconciled against cadence once; graph span is not "
            "added again, and GPU overlap is not inferred.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result_root = args.result_root.resolve()
    output_dir = (args.output_dir or result_root / "graph-external").resolve()
    selected_cases = tuple(args.cases or CASES)
    endpoints = _parse_endpoint_summaries(args.endpoint_summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    comparisons_dir = output_dir / "comparisons"
    cases_dir.mkdir(exist_ok=True)
    comparisons_dir.mkdir(exist_ok=True)

    case_results = {}
    for case in selected_cases:
        rank_reports = []
        for rank in range(8):
            selected_path = result_root / case / "analysis" / (
                f"rank-{rank}-selected-step.json"
            )
            if not selected_path.is_file():
                raise AnalysisError(f"{case}: missing {selected_path}")
            selected = json.loads(selected_path.read_text(encoding="utf-8"))
            trace_path = Path(selected["trace"])
            rank_reports.append(
                analyze_rank(load_trace(trace_path), selected, rank=rank)
            )
        result = aggregate_case(case, rank_reports, _endpoint_tpot(endpoints[case]))
        case_results[case] = result
        (cases_dir / f"{case}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    if set(selected_cases) != set(CASES):
        print(
            f"analyzed {len(selected_cases)} case(s); full comparison requires all four"
        )
        return 0

    comparison = compare_cases(case_results)
    comparison_path = comparisons_dir / "graph-external-c2-c64.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "graph-external-summary.csv", case_results)
    _write_markdown(
        output_dir / "graph-external-summary.md", case_results, comparison
    )
    print(
        f"analyzed 4 cases x 8 ranks; outputs={output_dir}; "
        f"comparison={comparison_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
