"""Annotation interval selection and containment for Chrome traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from kernel_families import merge_intervals


ANNOTATION_CATEGORIES = {"user_annotation", "gpu_user_annotation"}


class AnnotationSelectionError(ValueError):
    """Raised when an exact annotation occurrence cannot be selected."""


@dataclass(frozen=True)
class Annotation:
    name: str
    start: float
    end: float
    category: str
    pid: object = None
    tid: object = None

    @property
    def duration(self) -> float:
        return self.end - self.start


def category_tokens(event: Mapping) -> set[str]:
    """Return normalized Chrome category tokens without losing compound cats."""
    return {
        token.strip()
        for token in str(event.get("cat", "")).split(",")
        if token.strip()
    }


def timed_interval(event: Mapping) -> Optional[tuple[float, float]]:
    if event.get("ph") != "X" or not event.get("dur"):
        return None
    start = float(event.get("ts", 0.0))
    return start, start + float(event["dur"])


def collect_annotations(events: Iterable[Mapping]) -> list[Annotation]:
    annotations = []
    for event in events:
        interval = timed_interval(event)
        categories = category_tokens(event)
        category = next(
            (value for value in ANNOTATION_CATEGORIES if value in categories), None
        )
        if interval is None or category is None:
            continue
        annotations.append(
            Annotation(
                name=str(event.get("name", "")),
                start=interval[0],
                end=interval[1],
                category=category,
                pid=event.get("pid"),
                tid=event.get("tid"),
            )
        )
    return annotations


def decode_annotations(
    annotations: Sequence[Annotation],
) -> tuple[list[Annotation], Optional[str]]:
    """Select engine decode scopes, preferring GPU annotations."""

    def is_decode(annotation: Annotation) -> bool:
        return annotation.name.startswith("step[DECODE") or annotation.name.startswith(
            "decode["
        )

    matching = [annotation for annotation in annotations if is_decode(annotation)]
    gpu = [
        annotation
        for annotation in matching
        if annotation.category == "gpu_user_annotation"
    ]
    if gpu:
        return gpu, "gpu_user_annotation"
    cpu = [
        annotation for annotation in matching if annotation.category == "user_annotation"
    ]
    return cpu, "user_annotation" if cpu else None


def annotation_windows(
    annotations: Sequence[Annotation],
) -> tuple[list[tuple[float, float]], Optional[str], int]:
    selected, source = decode_annotations(annotations)
    windows = [(annotation.start, annotation.end) for annotation in selected]
    if not windows:
        return [], source, 0
    # merge_intervals returns a duration, so keep a local interval merger here.
    ordered = sorted(windows)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged], source, len(selected)


def select_annotation(
    annotations: Sequence[Annotation],
    name: str,
    category: Optional[str] = None,
    occurrence: int = 0,
) -> tuple[Annotation, int]:
    """Select one complete named occurrence in timestamp order."""
    if occurrence < 0:
        raise AnnotationSelectionError(
            "annotation occurrence must be a non-negative 0-based index"
        )
    matching = [
        annotation
        for annotation in annotations
        if annotation.name == name
        and (category is None or annotation.category == category)
    ]
    matching.sort(
        key=lambda annotation: (
            annotation.start,
            annotation.end,
            annotation.category,
        )
    )
    category_description = category or "either annotation category"
    if occurrence >= len(matching):
        raise AnnotationSelectionError(
            f"annotation {name!r} occurrence {occurrence} not found in "
            f"{category_description}; found {len(matching)} occurrence(s)"
        )
    selected = matching[occurrence]
    same_timestamp = [
        annotation
        for annotation in matching
        if annotation.start == selected.start
    ]
    if len(same_timestamp) > 1:
        categories = sorted({annotation.category for annotation in same_timestamp})
        raise AnnotationSelectionError(
            f"annotation {name!r} occurrence {occurrence} is ambiguous at "
            f"timestamp {selected.start}; matching categories={categories}. "
            "Use --annotation-category to disambiguate."
        )
    return selected, len(matching)


def clipped_duration(
    start: float, end: float, windows: Sequence[tuple[float, float]]
) -> float:
    if not windows:
        return end - start
    return sum(
        max(0.0, min(end, window_end) - max(start, window_start))
        for window_start, window_end in windows
    )


def innermost_annotation(
    start: float,
    end: float,
    annotations: Sequence[Annotation],
    *,
    pid: object = None,
    tid: object = None,
    prefer_gpu: bool = True,
) -> Optional[Annotation]:
    """Find the narrowest annotation containing an event interval."""
    candidates = []
    for annotation in annotations:
        if annotation.start > start or annotation.end < end:
            continue
        if pid is not None and annotation.pid is not None and annotation.pid != pid:
            continue
        if (
            annotation.category == "user_annotation"
            and tid is not None
            and annotation.tid is not None
            and annotation.tid != tid
        ):
            continue
        candidates.append(annotation)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda annotation: (
            annotation.duration,
            0
            if prefer_gpu and annotation.category == "gpu_user_annotation"
            else 1,
        ),
    )


def windows_duration(windows: Sequence[tuple[float, float]]) -> float:
    return float(merge_intervals(windows))
