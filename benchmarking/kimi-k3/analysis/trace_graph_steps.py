"""Select one directly correlated graph replay from a decode trace."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Mapping, Sequence

from kernel_families import classify_kernel
from trace_annotations import Annotation
from trace_correlation import (
    GPU_ACTIVITY_CATEGORIES,
    TimedEvent,
    collect_timed_events,
    correlation_id,
    graph_launch_events,
)


TIMING_TRUST_WARNING = (
    "Graph-replay kernel duration may be unreliable; production caller mapping "
    "will later use separate capture traces."
)


class GraphStepSelectionError(ValueError):
    """Raised when a graph step cannot be selected from direct evidence."""


@dataclass(frozen=True)
class GraphStep:
    launch: TimedEvent
    correlation: str
    activities: tuple[TimedEvent, ...]
    kernels: tuple[TimedEvent, ...]

    @property
    def first_kernel_timestamp(self) -> float:
        return min(event.start for event in self.kernels)


def _overlaps(
    start: float, end: float, windows: Sequence[tuple[float, float]]
) -> bool:
    return any(
        min(end, window_end) > max(start, window_start)
        for window_start, window_end in windows
    )


def _clipped_duration(
    event: TimedEvent, windows: Sequence[tuple[float, float]]
) -> float:
    return sum(
        max(0.0, min(event.end, end) - max(event.start, start))
        for start, end in windows
    )


def eligible_graph_steps(
    events: Sequence[Mapping],
    decode_windows: Sequence[tuple[float, float]],
) -> list[GraphStep]:
    """Build replay groups using only exact launch/activity correlation IDs."""
    if not decode_windows:
        raise GraphStepSelectionError(
            "graph-step selection requires a non-empty decode annotation window"
        )

    timed = collect_timed_events(events)
    launches_by_correlation = {}
    for launch in graph_launch_events(timed):
        value = correlation_id(launch)
        if value is not None:
            if value in launches_by_correlation:
                raise GraphStepSelectionError(
                    f"duplicate graph-launch correlation ID {value}; "
                    "cannot distinguish replay instances"
                )
            launches_by_correlation[value] = launch

    grouped = collections.defaultdict(list)
    for activity in timed:
        if activity.category not in GPU_ACTIVITY_CATEGORIES:
            continue
        if not _overlaps(activity.start, activity.end, decode_windows):
            continue
        value = correlation_id(activity)
        if value in launches_by_correlation:
            grouped[value].append(activity)

    steps = []
    for value, activities in grouped.items():
        kernels = tuple(
            event for event in activities if event.category == "kernel"
        )
        if not kernels:
            continue
        steps.append(
            GraphStep(
                launch=launches_by_correlation[value],
                correlation=value,
                activities=tuple(activities),
                kernels=kernels,
            )
        )
    steps.sort(key=lambda step: (step.first_kernel_timestamp, step.correlation))
    if not steps:
        raise GraphStepSelectionError(
            "no decode graph replay has kernels directly correlated to a "
            "hipGraphLaunch/cudaGraphLaunch; refusing equal-chunk inference"
        )
    return steps


def _choose_step(
    steps: Sequence[GraphStep], mode: str
) -> tuple[GraphStep, str, int]:
    if mode == "first":
        return (
            steps[0],
            "selected earliest eligible replay by correlated GPU kernel timestamp",
            0,
        )
    if mode == "last":
        return (
            steps[-1],
            "selected latest eligible replay by correlated GPU kernel timestamp",
            len(steps) - 1,
        )
    if len(steps) < 3:
        raise GraphStepSelectionError(
            "middle graph-step selection requires at least 3 eligible replays "
            "so first and last can be excluded"
        )
    interior = steps[1:-1]
    interior_index = len(interior) // 2
    return (
        interior[interior_index],
        "selected middle eligible replay after excluding first and last",
        interior_index + 1,
    )


def _annotation_details(
    step: GraphStep, annotations: Sequence[Annotation]
) -> tuple[list[str], list[str]]:
    span_start = min(event.start for event in step.activities)
    span_end = max(event.end for event in step.activities)
    overlapping = sorted(
        {
            annotation.name
            for annotation in annotations
            if min(span_end, annotation.end) > max(span_start, annotation.start)
        }
    )
    prefill_extend = [
        name
        for name in overlapping
        if name.lower().startswith("prefill[")
        or name.startswith("step[EXTEND")
        or "EXTEND" in name
    ]
    return overlapping, prefill_extend


def select_graph_step(
    events: Sequence[Mapping],
    annotations: Sequence[Annotation],
    decode_windows: Sequence[tuple[float, float]],
    mode: str,
    trace: str,
    top_kernels: int = 100,
) -> tuple[dict, dict]:
    steps = eligible_graph_steps(events, decode_windows)
    step, reason, selected_index = _choose_step(steps, mode)

    by_family = collections.defaultdict(lambda: [0.0, 0])
    by_kernel = collections.defaultdict(lambda: [0.0, 0])
    for kernel in step.kernels:
        duration = _clipped_duration(kernel, decode_windows)
        family = classify_kernel(kernel.name)
        by_family[family][0] += duration
        by_family[family][1] += 1
        by_kernel[kernel.name][0] += duration
        by_kernel[kernel.name][1] += 1

    annotation_names, prefill_extend_names = _annotation_details(step, annotations)
    streams = sorted(
        {
            str(
                event.args.get(
                    "stream", event.tid if event.tid is not None else ""
                )
            )
            for event in step.activities
        }
    )
    span_start = min(event.start for event in step.activities)
    span_end = max(event.end for event in step.activities)
    payload = {
        "trace": trace,
        "selection_mode": mode,
        "selection_reason": reason,
        "eligible_step_count": len(steps),
        "selected_step_index": selected_index,
        "graph_launch_api": step.launch.name,
        "graph_launch_correlation": step.correlation,
        "gpu_timestamp_span": {
            "start_us": span_start,
            "end_us": span_end,
            "duration_us": span_end - span_start,
        },
        "kernel_count": len(step.kernels),
        "memcpy_count": sum(
            event.category == "gpu_memcpy" for event in step.activities
        ),
        "memset_count": sum(
            event.category == "gpu_memset" for event in step.activities
        ),
        "streams": streams,
        "families": [
            {
                "family": family,
                "count": values[1],
                "duration_us": values[0],
            }
            for family, values in sorted(
                by_family.items(), key=lambda item: item[1][0], reverse=True
            )
        ],
        "top_kernels": [
            {"name": name, "count": values[1], "duration_us": values[0]}
            for name, values in sorted(
                by_kernel.items(), key=lambda item: item[1][0], reverse=True
            )[:top_kernels]
        ],
        "annotation_names": annotation_names,
        "timing_trust_warning": TIMING_TRUST_WARNING,
        "prefill_extend_annotation_overlap": bool(prefill_extend_names),
        "prefill_extend_annotation_names": prefill_extend_names,
    }
    metadata = {
        "selection_mode": mode,
        "selection_reason": reason,
        "eligible_step_count": len(steps),
        "selected_step_index": selected_index,
        "graph_launch_api": step.launch.name,
        "graph_launch_correlation": step.correlation,
    }
    return payload, metadata
