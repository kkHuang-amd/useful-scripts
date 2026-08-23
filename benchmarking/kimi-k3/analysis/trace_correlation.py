"""Reusable CPU/API-to-GPU correlation for PyTorch Chrome traces."""

from __future__ import annotations

import collections
import bisect
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from kernel_families import classify_kernel
from trace_annotations import (
    Annotation,
    category_tokens,
    innermost_annotation,
    timed_interval,
)


RUNTIME_CATEGORIES = {"cuda_runtime", "cuda_driver"}
GPU_ACTIVITY_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
GRAPH_LAUNCH_NAMES = {"hipgraphlaunch", "cudagraphlaunch"}
CONFIDENCE_RANK = {"unmatched": 0, "inferred": 1, "direct": 2}


@dataclass(frozen=True)
class TimedEvent:
    index: int
    name: str
    category: str
    start: float
    end: float
    pid: object
    tid: object
    args: Mapping


@dataclass(frozen=True)
class Attribution:
    kernel_index: int
    caller: Optional[str]
    annotation: Optional[str]
    host_api: Optional[str]
    confidence: str
    source: str
    graph_replay_opaque: bool


def _normalized_args(args: Mapping) -> dict[str, object]:
    return {
        re.sub(r"[^a-z0-9]", "", str(key).lower()): value
        for key, value in args.items()
    }


def _identifier(args: Mapping, *names: str) -> Optional[str]:
    normalized = _normalized_args(args)
    for name in names:
        value = normalized.get(re.sub(r"[^a-z0-9]", "", name.lower()))
        if value is not None and value != "":
            return str(value)
    return None


def correlation_id(event: TimedEvent) -> Optional[str]:
    return _identifier(event.args, "correlation", "correlation id")


def external_id(event: TimedEvent) -> Optional[str]:
    return _identifier(event.args, "External id", "external_id")


def _is_graph_launch(name: str) -> bool:
    normalized = name.lower().replace(" ", "")
    return any(normalized.startswith(marker) for marker in GRAPH_LAUNCH_NAMES)


def collect_timed_events(events: Iterable[Mapping]) -> list[TimedEvent]:
    timed = []
    for index, event in enumerate(events):
        interval = timed_interval(event)
        if interval is None:
            continue
        categories = category_tokens(event)
        category = next(
            (
                value
                for value in (
                    "kernel",
                    "cpu_op",
                    "cuda_runtime",
                    "cuda_driver",
                    "gpu_memcpy",
                    "gpu_memset",
                )
                if value in categories
            ),
            None,
        )
        if category is None:
            continue
        timed.append(
            TimedEvent(
                index=index,
                name=str(event.get("name", "")),
                category=category,
                start=interval[0],
                end=interval[1],
                pid=event.get("pid"),
                tid=event.get("tid"),
                args=event.get("args", {}) or {},
            )
        )
    return timed


def _event_at(
    timestamp: float,
    events: Sequence[TimedEvent],
    *,
    pid: object = None,
    tid: object = None,
) -> Optional[TimedEvent]:
    compatible = [
        event
        for event in events
        if (pid is None or event.pid is None or event.pid == pid)
        and (tid is None or event.tid is None or event.tid == tid)
    ]
    containing = [
        event for event in compatible if event.start <= timestamp <= event.end
    ]
    if containing:
        return min(containing, key=lambda event: (event.end - event.start, event.index))
    if not compatible:
        return None
    nearest = min(
        compatible,
        key=lambda event: min(
            abs(timestamp - event.start), abs(timestamp - event.end)
        ),
    )
    distance = min(abs(timestamp - nearest.start), abs(timestamp - nearest.end))
    return nearest if distance <= 1.0 else None


def _ac2g_links(
    raw_events: Sequence[Mapping],
    launches: Sequence[TimedEvent],
    gpu_activities: Sequence[TimedEvent],
) -> dict[int, TimedEvent]:
    starts: dict[str, list[Mapping]] = collections.defaultdict(list)
    pairs = []
    for event in raw_events:
        if "ac2g" not in category_tokens(event):
            continue
        flow_id = event.get("id", event.get("bind_id"))
        if flow_id is None:
            continue
        key = str(flow_id)
        phase = event.get("ph")
        if phase == "s":
            starts[key].append(event)
        elif phase == "f" and starts[key]:
            pairs.append((starts[key].pop(0), event))

    launch_matches = _match_flow_events(
        [pair[0] for pair in pairs], launches
    )
    gpu_matches = _match_flow_events(
        [pair[1] for pair in pairs], gpu_activities
    )
    links = {}
    for launch, gpu_event in zip(launch_matches, gpu_matches):
        if launch is not None and gpu_event is not None:
            links[gpu_event.index] = launch
    return links


def _match_flow_events(
    flow_events: Sequence[Mapping], candidates: Sequence[TimedEvent]
) -> list[Optional[TimedEvent]]:
    """Batch timestamp containment lookups by process/thread."""
    candidate_groups = collections.defaultdict(list)
    for candidate in candidates:
        candidate_groups[(candidate.pid, candidate.tid)].append(candidate)
    query_groups = collections.defaultdict(list)
    for index, event in enumerate(flow_events):
        query_groups[(event.get("pid"), event.get("tid"))].append(
            (float(event.get("ts", 0.0)), index)
        )

    result: list[Optional[TimedEvent]] = [None] * len(flow_events)
    for key, queries in query_groups.items():
        group = candidate_groups.get(key)
        if not group:
            for timestamp, index in queries:
                result[index] = _event_at(
                    timestamp,
                    candidates,
                    pid=key[0],
                    tid=key[1],
                )
            continue

        ordered_events = sorted(group, key=lambda event: event.start)
        ordered_queries = sorted(queries)
        active = []
        event_index = 0
        endpoints = sorted(
            (value, event.index, event)
            for event in ordered_events
            for value in (event.start, event.end)
        )
        endpoint_values = [item[0] for item in endpoints]
        for timestamp, index in ordered_queries:
            while (
                event_index < len(ordered_events)
                and ordered_events[event_index].start <= timestamp
            ):
                active.append(ordered_events[event_index])
                event_index += 1
            active = [event for event in active if event.end >= timestamp]
            if active:
                result[index] = min(
                    active,
                    key=lambda event: (event.end - event.start, event.index),
                )
                continue
            insertion = bisect.bisect_left(endpoint_values, timestamp)
            nearby = endpoints[max(0, insertion - 1) : insertion + 1]
            if nearby:
                distance, _, nearest = min(
                    (
                        abs(timestamp - value),
                        event.index,
                        event,
                    )
                    for value, _, event in nearby
                )
                if distance <= 1.0:
                    result[index] = nearest
    return result


def correlate_gpu_activities(
    raw_events: Sequence[Mapping],
    annotations: Sequence[Annotation],
) -> tuple[list[TimedEvent], dict[int, Attribution]]:
    """Attribute GPU activities using correlation IDs, then ac2g flow fallback."""
    timed = collect_timed_events(raw_events)
    cpu_ops = [event for event in timed if event.category == "cpu_op"]
    launches = [event for event in timed if event.category in RUNTIME_CATEGORIES]
    gpu_activities = [
        event for event in timed if event.category in GPU_ACTIVITY_CATEGORIES
    ]

    cpu_by_external = {}
    for cpu_op in cpu_ops:
        value = external_id(cpu_op)
        if value is not None:
            cpu_by_external[value] = cpu_op

    launch_by_correlation = {}
    for launch in launches:
        value = correlation_id(launch)
        if value is not None:
            launch_by_correlation[value] = launch

    flow_links = _ac2g_links(raw_events, launches, gpu_activities)
    result = {}
    for gpu_event in gpu_activities:
        launch = None
        source = "none"
        confidence = "unmatched"
        value = correlation_id(gpu_event)
        if value is not None:
            launch = launch_by_correlation.get(value)
            if launch is not None:
                source = "correlation"
                confidence = "direct"
        if launch is None and gpu_event.index in flow_links:
            launch = flow_links[gpu_event.index]
            source = "ac2g"
            confidence = "inferred"

        cpu_op = (
            cpu_by_external.get(external_id(launch))
            if launch is not None and external_id(launch) is not None
            else None
        )
        anchor = cpu_op or launch
        gpu_annotation = innermost_annotation(
            gpu_event.start,
            gpu_event.end,
            [
                item
                for item in annotations
                if item.category == "gpu_user_annotation"
            ],
            pid=gpu_event.pid,
            tid=gpu_event.tid,
        )
        host_annotation = (
            innermost_annotation(
                anchor.start,
                anchor.end,
                [
                    item
                    for item in annotations
                    if item.category == "user_annotation"
                ],
                pid=anchor.pid,
                tid=anchor.tid,
                prefer_gpu=False,
            )
            if anchor is not None
            else None
        )
        annotation = min(
            [
                item
                for item in (gpu_annotation, host_annotation)
                if item is not None
            ],
            key=lambda item: (
                item.duration,
                0 if item.category == "gpu_user_annotation" else 1,
            ),
            default=None,
        )
        if launch is None and annotation is not None:
            source = "annotation_containment"
            confidence = "inferred"
        graph_opaque = (
            launch is not None and _is_graph_launch(launch.name)
        )
        result[gpu_event.index] = Attribution(
            kernel_index=gpu_event.index,
            caller=cpu_op.name if cpu_op is not None else None,
            annotation=annotation.name if annotation is not None else None,
            host_api=launch.name if launch is not None else None,
            confidence=confidence,
            source=source,
            graph_replay_opaque=graph_opaque,
        )
    return gpu_activities, result


def graph_launch_events(events: Sequence[TimedEvent]) -> list[TimedEvent]:
    return [
        event
        for event in events
        if event.category in RUNTIME_CATEGORIES
        and _is_graph_launch(event.name)
    ]


def caller_group_key(
    event: TimedEvent, attribution: Attribution
) -> tuple[str, str, str, str, str, str, str]:
    stream = str(event.args.get("stream", event.tid if event.tid is not None else ""))
    return (
        attribution.caller or "",
        attribution.annotation or "",
        attribution.host_api or "",
        event.name,
        classify_kernel(event.name),
        stream,
        attribution.confidence,
    )
