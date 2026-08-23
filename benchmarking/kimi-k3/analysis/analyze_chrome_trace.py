#!/usr/bin/env python3
"""Summarize and attribute Kimi-K3 PyTorch Chrome traces."""

import argparse
import collections
import gzip
import json
from pathlib import Path

from kernel_families import classify_kernel, merge_intervals
from trace_annotations import (
    AnnotationSelectionError,
    annotation_windows,
    category_tokens,
    collect_annotations,
    select_annotation,
    timed_interval,
    windows_duration,
)
from trace_correlation import (
    CONFIDENCE_RANK,
    caller_group_key,
    collect_timed_events,
    correlate_gpu_activities,
    graph_launch_events,
)
from trace_graph_steps import GraphStepSelectionError, select_graph_step


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-seconds", type=float)
    parser.add_argument("--top-kernels", type=int, default=100)
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help="restrict GPU aggregates to SGLang/ATOM decode annotations",
    )
    parser.add_argument(
        "--emit-caller-map",
        nargs="?",
        const="",
        metavar="JSONL",
        help="write grouped caller attribution JSONL (default: beside output)",
    )
    parser.add_argument(
        "--min-confidence",
        choices=("unmatched", "inferred", "direct"),
        default="unmatched",
        help="minimum confidence included in caller rollups/map",
    )
    parser.add_argument(
        "--capture-trace",
        type=Path,
        help="reserved capture trace for future graph-node matching",
    )
    parser.add_argument(
        "--select-graph-step",
        choices=("first", "middle", "last"),
        help="select one directly correlated decode graph replay",
    )
    parser.add_argument(
        "--selected-step-output",
        type=Path,
        help="write the selected graph replay JSON",
    )
    parser.add_argument(
        "--annotation-name",
        help="restrict analysis to one complete, exactly named annotation",
    )
    parser.add_argument(
        "--annotation-category",
        choices=("user_annotation", "gpu_user_annotation"),
        help="category filter for --annotation-name (default: either)",
    )
    parser.add_argument(
        "--annotation-occurrence",
        type=int,
        default=0,
        help="0-based timestamp-ordered annotation occurrence (default: 0)",
    )
    args = parser.parse_args(argv)
    if args.select_graph_step and not args.decode_only:
        parser.error("--select-graph-step requires --decode-only")
    if args.select_graph_step and args.selected_step_output is None:
        parser.error("--select-graph-step requires --selected-step-output")
    if args.selected_step_output is not None and not args.select_graph_step:
        parser.error("--selected-step-output requires --select-graph-step")
    if args.annotation_category and not args.annotation_name:
        parser.error("--annotation-category requires --annotation-name")
    if args.annotation_occurrence and not args.annotation_name:
        parser.error("--annotation-occurrence requires --annotation-name")
    if args.annotation_occurrence < 0:
        parser.error("--annotation-occurrence must be non-negative")
    return args


def load_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        payload = json.load(handle)
    return payload["traceEvents"] if isinstance(payload, dict) else payload


def _intersect_windows(left, right):
    if not left:
        return list(right)
    if not right:
        return list(left)
    intersections = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start, end = max(left_start, right_start), min(left_end, right_end)
            if end > start:
                intersections.append((start, end))
    return intersections


def _event_segments(start, end, windows):
    if not windows:
        return [(start, end)]
    return [
        (max(start, window_start), min(end, window_end))
        for window_start, window_end in windows
        if min(end, window_end) > max(start, window_start)
    ]


def _rollups(grouped, total_duration, limit=10):
    return [
        {
            "caller": key[0] or None,
            "annotation": key[1] or None,
            "host_api": key[2] or None,
            "confidence": key[6],
            "duration_us": values[0],
            "count": values[1],
            "pct_duration": (
                100.0 * values[0] / total_duration if total_duration else 0.0
            ),
        }
        for key, values in sorted(
            grouped.items(), key=lambda item: item[1][0], reverse=True
        )[:limit]
    ]


def analyze(events, args):
    annotations_list = collect_annotations(events)
    annotation_name = getattr(args, "annotation_name", None)
    annotation_category = getattr(args, "annotation_category", None)
    annotation_occurrence = getattr(args, "annotation_occurrence", 0)

    kernels = []
    annotations = collections.defaultdict(lambda: [0.0, 0])
    categories = collections.Counter()
    parsed_categories = collections.Counter()
    metadata_count = 0
    for index, event in enumerate(events):
        category = str(event.get("cat", ""))
        categories[category] += 1
        tokens = category_tokens(event)
        for token in (
            "kernel",
            "cpu_op",
            "cuda_runtime",
            "cuda_driver",
            "user_annotation",
            "gpu_user_annotation",
            "ac2g",
            "gpu_memcpy",
            "gpu_memset",
        ):
            if token in tokens:
                parsed_categories[token] += 1
        if event.get("ph") == "M":
            metadata_count += 1
        interval = timed_interval(event)
        if interval is None:
            continue
        duration = interval[1] - interval[0]
        name = str(event.get("name", ""))
        if "kernel" in tokens:
            stream = str(event.get("args", {}).get("stream", event.get("tid", "")))
            kernels.append((interval[0], interval[1], name, stream, index))
        elif tokens & {"user_annotation", "gpu_user_annotation"}:
            annotations[name][0] += duration
            annotations[name][1] += 1

    windows = []
    window_mode = "full"
    if args.tail_seconds and kernels:
        window_end = max(end for _, end, _, _, _ in kernels)
        window_start = window_end - args.tail_seconds * 1e6
        windows = [(window_start, window_end)]
        window_mode = "tail"

    decode_source = None
    decode_annotation_count = 0
    decode_windows = []
    if args.decode_only:
        decode_windows, decode_source, decode_annotation_count = annotation_windows(
            annotations_list
        )
        windows = _intersect_windows(windows, decode_windows)
        if not decode_windows:
            windows = [(0.0, 0.0)]
        window_mode = "decode_tail" if args.tail_seconds else "decode"

    annotation_selection = None
    if annotation_name:
        selected_annotation, matching_count = select_annotation(
            annotations_list,
            annotation_name,
            annotation_category,
            annotation_occurrence,
        )
        selected_window = [
            (selected_annotation.start, selected_annotation.end)
        ]
        previous_filter_active = bool(args.tail_seconds) or bool(args.decode_only)
        if previous_filter_active:
            windows = _intersect_windows(windows, selected_window)
            if not windows:
                windows = [(0.0, 0.0)]
        else:
            windows = selected_window
        window_mode = f"{window_mode}_annotation" if window_mode != "full" else "annotation"
        annotation_selection = {
            "name": selected_annotation.name,
            "requested_category": annotation_category,
            "category": selected_annotation.category,
            "occurrence": annotation_occurrence,
            "matching_occurrence_count": matching_count,
            "start_us": selected_annotation.start,
            "end_us": selected_annotation.end,
            "duration_us": selected_annotation.duration,
        }

    gpu_activities, attribution = correlate_gpu_activities(events, annotations_list)
    gpu_by_index = {event.index: event for event in gpu_activities}
    selected_kernels = []
    for start, end, name, stream, index in kernels:
        segments = _event_segments(start, end, windows)
        duration = sum(segment_end - segment_start for segment_start, segment_end in segments)
        if duration > 0:
            selected_kernels.append(
                (segments, duration, name, stream, index)
            )

    by_name = collections.defaultdict(lambda: [0.0, 0])
    by_family = collections.defaultdict(lambda: [0.0, 0])
    by_stream = collections.defaultdict(list)
    intervals = []
    caller_groups = collections.defaultdict(lambda: [0.0, 0])
    family_callers = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0])
    )
    kernel_callers = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0])
    )
    confidence_counts = collections.Counter()
    confidence_duration = collections.Counter()
    graph_kernel_count = 0
    graph_kernel_duration = 0.0
    for segments, duration, name, stream, index in selected_kernels:
        by_name[name][0] += duration
        by_name[name][1] += 1
        family = classify_kernel(name)
        by_family[family][0] += duration
        by_family[family][1] += 1
        intervals.extend(segments)
        by_stream[stream].extend(segments)
        item = attribution[index]
        confidence_counts[item.confidence] += 1
        confidence_duration[item.confidence] += duration
        if item.graph_replay_opaque:
            graph_kernel_count += 1
            graph_kernel_duration += duration
        if CONFIDENCE_RANK[item.confidence] >= CONFIDENCE_RANK[args.min_confidence]:
            key = caller_group_key(gpu_by_index[index], item)
            caller_groups[key][0] += duration
            caller_groups[key][1] += 1
            family_callers[family][key][0] += duration
            family_callers[family][key][1] += 1
            kernel_callers[name][key][0] += duration
            kernel_callers[name][key][1] += 1

    total_kernel_us = sum(item[1] for item in selected_kernels)
    span_us = (
        max(end for _, end in intervals) - min(start for start, _ in intervals)
        if intervals
        else 0.0
    )
    all_graph_launches = graph_launch_events(collect_timed_events(events))
    graph_launches = [
        event
        for event in all_graph_launches
        if _event_segments(event.start, event.end, windows)
    ]
    # Kineto CPU runtime events and GPU-projected annotation/kernel events can
    # use different clock domains. Direct correlation may therefore prove that
    # selected decode kernels belong to hipGraphLaunch even when the CPU launch
    # timestamp does not intersect the GPU decode window. The active profiler
    # session is already clipped to this request wave, so retain all graph
    # launches whenever correlated selected kernels mark replay opacity.
    if graph_kernel_count and not graph_launches:
        graph_launches = all_graph_launches
    capture_trace = {
        "provided": args.capture_trace is not None,
        "path": str(args.capture_trace) if args.capture_trace is not None else None,
        "matching_status": "not_implemented",
        "graph_map_applied": False,
    }
    valid_windows = [(start, end) for start, end in windows if end > start]
    capture_intervals = valid_windows or intervals
    capture_start = min((start for start, _ in capture_intervals), default=None)
    capture_end = max((end for _, end in capture_intervals), default=None)
    capture_duration = (
        windows_duration(valid_windows)
        if valid_windows
        else (span_us if not windows else 0.0)
    )
    required_counts = {
        key: parsed_categories.get(key, 0)
        for key in (
            "kernel",
            "cpu_op",
            "cuda_runtime",
            "cuda_driver",
            "user_annotation",
            "gpu_user_annotation",
            "ac2g",
            "gpu_memcpy",
            "gpu_memset",
        )
    }
    result = {
        "schema_version": 2,
        "trace": str(args.trace),
        "event_count": len(events),
        "category_counts": dict(categories),
        "kernel_count": len(selected_kernels),
        "kernel_sum_us": total_kernel_us,
        "kernel_busy_union_us": merge_intervals(intervals),
        "kernel_span_us": span_us,
        "active_stream_count": len(by_stream),
        "streams": [
            {
                "stream": stream,
                "count": len(values),
                "sum_us": sum(end - start for start, end in values),
                "busy_union_us": merge_intervals(values),
            }
            for stream, values in sorted(
                by_stream.items(),
                key=lambda item: sum(end - start for start, end in item[1]),
                reverse=True,
            )
        ],
        "families": [
            {
                "family": family,
                "duration_us": values[0],
                "count": values[1],
                "pct_kernel_sum": (
                    100.0 * values[0] / total_kernel_us if total_kernel_us else 0.0
                ),
                "callers": _rollups(family_callers[family], values[0]),
            }
            for family, values in sorted(
                by_family.items(), key=lambda item: item[1][0], reverse=True
            )
        ],
        "top_kernels": [
            {
                "name": name,
                "duration_us": values[0],
                "count": values[1],
                "callers": _rollups(kernel_callers[name], values[0]),
            }
            for name, values in sorted(
                by_name.items(), key=lambda item: item[1][0], reverse=True
            )[: args.top_kernels]
        ],
        "top_annotations": [
            {"name": name, "duration_us": values[0], "count": values[1]}
            for name, values in sorted(
                annotations.items(), key=lambda item: item[1][0], reverse=True
            )[:80]
        ],
        "validation_gates": {
            "category_counts": required_counts,
            "metadata_event_count": metadata_count,
            "has_kernel": required_counts["kernel"] > 0,
            "has_cpu_op": required_counts["cpu_op"] > 0,
            "has_host_api": (
                required_counts["cuda_runtime"] + required_counts["cuda_driver"] > 0
            ),
            "has_annotation": (
                required_counts["user_annotation"]
                + required_counts["gpu_user_annotation"]
                > 0
            ),
            "has_correlation_or_flow": (
                confidence_counts["direct"] > 0 or required_counts["ac2g"] > 0
            ),
            "decode_window_found": (
                decode_annotation_count > 0 if args.decode_only else None
            ),
        },
        "capture_window": {
            "mode": window_mode,
            "start_us": capture_start,
            "end_us": capture_end,
            "duration_us": capture_duration,
            "decode_annotation_source": decode_source,
            "decode_annotation_count": decode_annotation_count,
            **(
                {"annotation_selection": annotation_selection}
                if annotation_selection is not None
                else {}
            ),
        },
        "caller_map_stats": {
            "min_confidence": args.min_confidence,
            "kernel_counts_by_confidence": dict(confidence_counts),
            "kernel_duration_us_by_confidence": dict(confidence_duration),
            "group_count": len(caller_groups),
        },
        "graph_replay": {
            "detected": bool(graph_launches),
            "launch_count": len(graph_launches),
            "opaque_kernel_count": graph_kernel_count,
            "opaque_kernel_duration_us": graph_kernel_duration,
            "timing_trust": (
                "Graph-replay kernel duration may be unreliable; production "
                "caller mapping will later use separate capture traces."
            ),
            "capture_trace": capture_trace,
        },
    }
    return result, caller_groups


def write_caller_map(path, grouped):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for key, values in sorted(
            grouped.items(), key=lambda item: item[1][0], reverse=True
        ):
            row = {
                "caller": key[0] or None,
                "annotation": key[1] or None,
                "host_api": key[2] or None,
                "kernel": key[3],
                "family": key[4],
                "stream": key[5],
                "confidence": key[6],
                "count": values[1],
                "duration_us": values[0],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv=None):
    args = parse_args(argv)
    events = load_trace(args.trace)
    try:
        result, caller_groups = analyze(events, args)
    except AnnotationSelectionError as error:
        raise SystemExit(f"error: {error}") from error
    if args.select_graph_step:
        annotations = collect_annotations(events)
        decode_windows, _, _ = annotation_windows(annotations)
        if args.tail_seconds:
            kernel_ends = [
                timed_interval(event)[1]
                for event in events
                if "kernel" in category_tokens(event)
                and timed_interval(event) is not None
            ]
            if kernel_ends:
                tail_end = max(kernel_ends)
                decode_windows = _intersect_windows(
                    decode_windows,
                    [(tail_end - args.tail_seconds * 1e6, tail_end)],
                )
        try:
            selected_step, selected_metadata = select_graph_step(
                events,
                annotations,
                decode_windows,
                args.select_graph_step,
                str(args.trace),
                args.top_kernels,
            )
        except GraphStepSelectionError as error:
            raise SystemExit(f"error: {error}") from error
        args.selected_step_output.parent.mkdir(parents=True, exist_ok=True)
        args.selected_step_output.write_text(
            json.dumps(selected_step, indent=2) + "\n"
        )
        result["selected_graph_step"] = {
            "output": str(args.selected_step_output),
            **selected_metadata,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.emit_caller_map is not None:
        caller_map_path = (
            Path(args.emit_caller_map)
            if args.emit_caller_map
            else args.output.with_name(f"{args.output.stem}-caller-map.jsonl")
        )
        write_caller_map(caller_map_path, caller_groups)
    print(
        f"kernels={result['kernel_count']} "
        f"sum_ms={result['kernel_sum_us'] / 1000:.3f} "
        f"busy_union_ms={result['kernel_busy_union_us'] / 1000:.3f} "
        f"span_ms={result['kernel_span_us'] / 1000:.3f}"
    )


if __name__ == "__main__":
    main()
