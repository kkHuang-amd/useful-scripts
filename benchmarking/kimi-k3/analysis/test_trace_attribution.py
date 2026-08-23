import argparse
import collections
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_chrome_trace
from kernel_families import classify_kernel, merge_intervals
from trace_annotations import AnnotationSelectionError, collect_annotations
from trace_correlation import correlate_gpu_activities
from trace_graph_steps import GraphStepSelectionError, select_graph_step


def event(name, cat, ts, dur=None, *, ph="X", pid=1, tid=1, args=None, **extra):
    value = {
        "name": name,
        "cat": cat,
        "ph": ph,
        "ts": ts,
        "pid": pid,
        "tid": tid,
        "args": args or {},
    }
    if dur is not None:
        value["dur"] = dur
    value.update(extra)
    return value


def args(**overrides):
    defaults = dict(
        trace=Path("synthetic.trace.json.gz"),
        output=Path("summary.json"),
        tail_seconds=None,
        top_kernels=100,
        decode_only=False,
        emit_caller_map=None,
        min_confidence="unmatched",
        capture_trace=None,
        annotation_name=None,
        annotation_category=None,
        annotation_occurrence=0,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def graph_steps(count=5):
    events = [
        event(
            "decode[bs=2]",
            "gpu_user_annotation",
            50,
            count * 100 + 100,
            pid=2,
            tid=7,
        )
    ]
    for index in range(count):
        correlation = index + 1
        # Deliberately reverse CPU launch timestamps relative to GPU execution.
        events.extend(
            [
                event(
                    "hipGraphLaunch",
                    "cuda_runtime",
                    9000 - index * 100,
                    2,
                    args={"correlation": correlation},
                ),
                event(
                    f"gemm_step_{correlation}",
                    "kernel",
                    100 + index * 100,
                    10,
                    pid=2,
                    tid=7,
                    args={"correlation": correlation, "stream": 3},
                ),
            ]
        )
    return events


class TraceAttributionTest(unittest.TestCase):
    def test_direct_correlation_and_innermost_host_annotation(self):
        events = [
            event("outer", "user_annotation", 0, 100),
            event("inner", "user_annotation", 10, 20),
            event("aten::mm", "cpu_op", 12, 10, args={"External id": 7}),
            event(
                "hipLaunchKernel",
                "cuda_runtime",
                15,
                2,
                args={"correlation": 99, "External id": 7},
            ),
            event("gemm_kernel", "kernel", 40, 5, args={"correlation": 99}),
        ]
        gpu, mapped = correlate_gpu_activities(events, collect_annotations(events))
        attribution = mapped[gpu[0].index]
        self.assertEqual(attribution.caller, "aten::mm")
        self.assertEqual(attribution.host_api, "hipLaunchKernel")
        self.assertEqual(attribution.annotation, "inner")
        self.assertEqual(attribution.confidence, "direct")

    def test_ac2g_flow_fallback(self):
        events = [
            event("hipLaunchKernel", "cuda_runtime", 10, 3, pid=1, tid=2),
            event("flow", "ac2g", 11, ph="s", pid=1, tid=2, id=42),
            event("flow", "ac2g", 21, ph="f", pid=2, tid=9, id=42),
            event("fallback_kernel", "kernel", 20, 4, pid=2, tid=9),
        ]
        gpu, mapped = correlate_gpu_activities(events, [])
        attribution = mapped[gpu[0].index]
        self.assertEqual(attribution.host_api, "hipLaunchKernel")
        self.assertEqual(attribution.confidence, "inferred")
        self.assertEqual(attribution.source, "ac2g")

    def test_gpu_annotation_containment_infers_caller_scope(self):
        events = [
            event("decode[layer]", "gpu_user_annotation", 100, 20, pid=2, tid=7),
            event("orphan_kernel", "kernel", 105, 4, pid=2, tid=7),
        ]
        gpu, mapped = correlate_gpu_activities(events, collect_annotations(events))
        attribution = mapped[gpu[0].index]
        self.assertEqual(attribution.annotation, "decode[layer]")
        self.assertEqual(attribution.confidence, "inferred")
        self.assertEqual(attribution.source, "annotation_containment")

    def test_decode_clipping_prefers_gpu_annotation(self):
        events = [
            event("decode[cpu]", "user_annotation", 90, 40),
            event("decode[gpu]", "gpu_user_annotation", 100, 20, pid=2, tid=7),
            event("left_kernel", "kernel", 95, 10, pid=2, tid=7),
            event("right_kernel", "kernel", 115, 10, pid=2, tid=7),
            event("outside_kernel", "kernel", 125, 5, pid=2, tid=7),
        ]
        result, _ = analyze_chrome_trace.analyze(
            events, args(decode_only=True)
        )
        self.assertEqual(result["kernel_count"], 2)
        self.assertEqual(result["kernel_sum_us"], 10)
        self.assertEqual(
            result["capture_window"]["decode_annotation_source"],
            "gpu_user_annotation",
        )

    def test_repeated_gpu_annotation_occurrence_selection(self):
        events = [
            event("capture", "gpu_user_annotation", 10, 20, pid=2, tid=7),
            event("first_kernel", "kernel", 15, 5, pid=2, tid=7),
            event("capture", "gpu_user_annotation", 100, 30, pid=2, tid=7),
            event("second_kernel", "kernel", 110, 7, pid=2, tid=7),
        ]
        result, _ = analyze_chrome_trace.analyze(
            events,
            args(
                annotation_name="capture",
                annotation_category="gpu_user_annotation",
                annotation_occurrence=1,
            ),
        )
        self.assertEqual(result["kernel_count"], 1)
        self.assertEqual(result["top_kernels"][0]["name"], "second_kernel")
        selection = result["capture_window"]["annotation_selection"]
        self.assertEqual(selection["occurrence"], 1)
        self.assertEqual(selection["matching_occurrence_count"], 2)
        self.assertEqual(selection["start_us"], 100)

    def test_annotation_category_filter(self):
        events = [
            event("capture", "user_annotation", 10, 20),
            event("cpu_window_kernel", "kernel", 15, 5),
            event("capture", "gpu_user_annotation", 100, 20, pid=2, tid=7),
            event("gpu_window_kernel", "kernel", 105, 5, pid=2, tid=7),
        ]
        result, _ = analyze_chrome_trace.analyze(
            events,
            args(
                annotation_name="capture",
                annotation_category="gpu_user_annotation",
            ),
        )
        self.assertEqual(result["kernel_count"], 1)
        self.assertEqual(result["top_kernels"][0]["name"], "gpu_window_kernel")
        self.assertEqual(
            result["capture_window"]["annotation_selection"]["category"],
            "gpu_user_annotation",
        )

    def test_annotation_intersects_decode_and_tail_windows(self):
        events = [
            event("decode[bs=2]", "gpu_user_annotation", 0, 100, pid=2, tid=7),
            event("capture", "gpu_user_annotation", 40, 40, pid=2, tid=7),
            event("left_kernel", "kernel", 45, 10, pid=2, tid=7),
            event("right_kernel", "kernel", 75, 10, pid=2, tid=7),
            event("tail_anchor", "kernel", 99, 1, pid=2, tid=7),
        ]
        result, _ = analyze_chrome_trace.analyze(
            events,
            args(
                decode_only=True,
                tail_seconds=0.00005,
                annotation_name="capture",
                annotation_category="gpu_user_annotation",
            ),
        )
        self.assertEqual(result["kernel_count"], 2)
        self.assertEqual(result["kernel_sum_us"], 10)
        self.assertEqual(result["capture_window"]["start_us"], 50)
        self.assertEqual(result["capture_window"]["end_us"], 80)
        self.assertEqual(
            result["capture_window"]["mode"], "decode_tail_annotation"
        )

    def test_missing_annotation_occurrence_fails(self):
        events = [event("capture", "gpu_user_annotation", 10, 20)]
        with self.assertRaisesRegex(
            AnnotationSelectionError, "occurrence 1 not found"
        ):
            analyze_chrome_trace.analyze(
                events,
                args(
                    annotation_name="capture",
                    annotation_category="gpu_user_annotation",
                    annotation_occurrence=1,
                ),
            )

    def test_same_timestamp_annotation_requires_category_filter(self):
        events = [
            event("capture", "user_annotation", 10, 20),
            event("capture", "gpu_user_annotation", 10, 20, pid=2, tid=7),
        ]
        with self.assertRaisesRegex(AnnotationSelectionError, "ambiguous"):
            analyze_chrome_trace.analyze(
                events, args(annotation_name="capture")
            )

    def test_graph_launch_is_detected_and_opaque(self):
        events = [
            event(
                "hipGraphLaunch",
                "cuda_runtime",
                10,
                2,
                args={"correlation": 3},
            ),
            event("graph_kernel", "kernel", 20, 4, args={"correlation": 3}),
        ]
        result, _ = analyze_chrome_trace.analyze(
            events, args(capture_trace=Path("capture.json.gz"))
        )
        self.assertTrue(result["graph_replay"]["detected"])
        self.assertEqual(result["graph_replay"]["opaque_kernel_count"], 1)
        self.assertFalse(
            result["graph_replay"]["capture_trace"]["graph_map_applied"]
        )
        self.assertEqual(
            result["graph_replay"]["capture_trace"]["matching_status"],
            "not_implemented",
        )

    def test_graph_launch_detection_handles_cpu_gpu_clock_skew(self):
        events = [
            event(
                "hipGraphLaunch",
                "cuda_runtime",
                10,
                2,
                args={"correlation": 3},
            ),
            event(
                "decode[bs=2]",
                "gpu_user_annotation",
                100,
                20,
                pid=2,
                tid=7,
            ),
            event(
                "graph_kernel",
                "kernel",
                110,
                4,
                pid=2,
                tid=7,
                args={"correlation": 3},
            ),
        ]
        result, _ = analyze_chrome_trace.analyze(
            events, args(decode_only=True)
        )
        self.assertTrue(result["graph_replay"]["detected"])
        self.assertEqual(result["graph_replay"]["launch_count"], 1)
        self.assertEqual(result["graph_replay"]["opaque_kernel_count"], 1)

    def test_unmatched_kernel(self):
        events = [event("orphan_kernel", "kernel", 1, 3)]
        result, groups = analyze_chrome_trace.analyze(events, args())
        self.assertEqual(
            result["caller_map_stats"]["kernel_counts_by_confidence"]["unmatched"],
            1,
        )
        self.assertEqual(len(groups), 1)

    def test_legacy_aggregates_are_unchanged(self):
        events = [
            event("gemm", "kernel", 0, 10, tid=4, args={"stream": 2}),
            event("copy_kernel", "kernel", 5, 5, tid=4, args={"stream": 2}),
            event("scope", "user_annotation", 0, 30),
            event("process_name", "", 0, ph="M", args={"name": "worker"}),
        ]
        result, _ = analyze_chrome_trace.analyze(events, args())

        kernels = [(0, 10, "gemm", "2"), (5, 10, "copy_kernel", "2")]
        by_name = collections.defaultdict(lambda: [0.0, 0])
        by_family = collections.defaultdict(lambda: [0.0, 0])
        for start, end, name, _ in kernels:
            by_name[name][0] += end - start
            by_name[name][1] += 1
            family = classify_kernel(name)
            by_family[family][0] += end - start
            by_family[family][1] += 1

        self.assertEqual(result["event_count"], len(events))
        self.assertEqual(result["kernel_count"], 2)
        self.assertEqual(result["kernel_sum_us"], 15)
        self.assertEqual(result["kernel_busy_union_us"], merge_intervals([(0, 10), (5, 10)]))
        self.assertEqual(result["kernel_span_us"], 10)
        self.assertEqual(result["active_stream_count"], 1)
        self.assertEqual(
            [(item["name"], item["duration_us"], item["count"]) for item in result["top_kernels"]],
            [
                (name, values[0], values[1])
                for name, values in sorted(
                    by_name.items(), key=lambda item: item[1][0], reverse=True
                )
            ],
        )
        self.assertEqual(result["top_annotations"][0]["duration_us"], 30)

    def test_cli_writes_grouped_caller_map(self):
        events = [
            event(
                "hipLaunchKernel",
                "cuda_runtime",
                5,
                1,
                args={"correlation": 8},
            ),
            event("kernel", "kernel", 10, 2, args={"correlation": 8, "stream": 3}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trace = directory / "trace.json.gz"
            output = directory / "summary.json"
            caller_map = directory / "callers.jsonl"
            with gzip.open(trace, "wt") as handle:
                json.dump({"traceEvents": events}, handle)
            analyze_chrome_trace.main(
                [
                    str(trace),
                    "--output",
                    str(output),
                    "--emit-caller-map",
                    str(caller_map),
                ]
            )
            row = json.loads(caller_map.read_text().strip())
            self.assertEqual(row["host_api"], "hipLaunchKernel")
            self.assertEqual(row["count"], 1)
            self.assertEqual(row["duration_us"], 2)

    def test_graph_steps_order_by_gpu_time_despite_clock_skew(self):
        events = graph_steps()
        annotations = collect_annotations(events)
        windows = [(50, 650)]
        first, _ = select_graph_step(
            events, annotations, windows, "first", "synthetic"
        )
        last, _ = select_graph_step(
            events, annotations, windows, "last", "synthetic"
        )
        self.assertEqual(first["graph_launch_correlation"], "1")
        self.assertEqual(last["graph_launch_correlation"], "5")
        self.assertEqual(first["gpu_timestamp_span"]["start_us"], 100)
        self.assertEqual(last["gpu_timestamp_span"]["start_us"], 500)

    def test_middle_graph_step_excludes_first_and_last(self):
        events = graph_steps()
        selected, metadata = select_graph_step(
            events,
            collect_annotations(events),
            [(50, 650)],
            "middle",
            "synthetic",
        )
        self.assertEqual(selected["graph_launch_correlation"], "3")
        self.assertNotIn(selected["graph_launch_correlation"], {"1", "5"})
        self.assertEqual(metadata["selected_step_index"], 2)
        self.assertIn("excluding first and last", selected["selection_reason"])

    def test_graph_step_fails_without_direct_correlation(self):
        events = [
            event("decode[bs=2]", "gpu_user_annotation", 50, 100),
            event("hipGraphLaunch", "cuda_runtime", 9000, 2),
            event("uncorrelated_kernel", "kernel", 100, 5, pid=2, tid=7),
        ]
        with self.assertRaisesRegex(
            GraphStepSelectionError, "refusing equal-chunk inference"
        ):
            select_graph_step(
                events,
                collect_annotations(events),
                [(50, 150)],
                "first",
                "synthetic",
            )

    def test_selected_graph_step_family_counts_and_memcpy(self):
        events = graph_steps(3)
        events.extend(
            [
                event(
                    "recompute_w_u_kernel",
                    "kernel",
                    305,
                    7,
                    pid=2,
                    tid=8,
                    args={"correlation": 3, "stream": 4},
                ),
                event(
                    "device_to_device",
                    "gpu_memcpy",
                    313,
                    2,
                    pid=2,
                    tid=8,
                    args={"correlation": 3, "stream": 4},
                ),
            ]
        )
        selected, _ = select_graph_step(
            events,
            collect_annotations(events),
            [(50, 450)],
            "last",
            "synthetic",
        )
        families = {
            item["family"]: item for item in selected["families"]
        }
        self.assertEqual(selected["kernel_count"], 2)
        self.assertEqual(selected["memcpy_count"], 1)
        self.assertEqual(families["dense_gemm"]["count"], 1)
        self.assertEqual(families["kda"]["count"], 1)
        self.assertEqual(set(selected["streams"]), {"3", "4"})

    def test_cli_writes_selected_step_and_summary_pointer(self):
        events = graph_steps(3)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            trace = directory / "trace.json.gz"
            output = directory / "summary.json"
            step_output = directory / "selected-step.json"
            with gzip.open(trace, "wt") as handle:
                json.dump({"traceEvents": events}, handle)
            analyze_chrome_trace.main(
                [
                    str(trace),
                    "--output",
                    str(output),
                    "--decode-only",
                    "--select-graph-step",
                    "middle",
                    "--selected-step-output",
                    str(step_output),
                ]
            )
            selected = json.loads(step_output.read_text())
            summary = json.loads(output.read_text())
            self.assertEqual(selected["graph_launch_correlation"], "2")
            self.assertEqual(
                summary["selected_graph_step"]["output"], str(step_output)
            )
            self.assertEqual(
                summary["selected_graph_step"]["graph_launch_correlation"], "2"
            )


if __name__ == "__main__":
    unittest.main()
