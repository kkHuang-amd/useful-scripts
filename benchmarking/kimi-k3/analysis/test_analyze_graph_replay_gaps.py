import copy
import unittest

from analyze_graph_replay_gaps import (
    AnalysisError,
    aggregate_case,
    analyze_rank,
    classify_host_event,
    select_decode_launches,
)


def event(name, category, ts, dur, *, pid=1, tid=1, args=None):
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": args or {},
    }


def synthetic_events():
    events = []
    for index, (annotation_ts, launch_ts) in enumerate(
        ((90.0, 100.0), (190.0, 200.0), (340.0, 350.0))
    ):
        events.append(
            event(
                f"decode[bs=2 tok=2 d=2 sk={index}]",
                "user_annotation",
                annotation_ts,
                90.0,
            )
        )
        events.append(
            event(
                f"decode[bs=2 tok=2 d=2 sk={index}]",
                "gpu_user_annotation",
                10_000.0 + index * 80.0,
                70.0,
                pid=2,
                tid=7,
            )
        )
        events.append(
            event(
                "hipGraphLaunch",
                "cuda_runtime",
                launch_ts,
                5.0,
                args={"correlation": 7},
            )
        )
        # Reused correlations and a disjoint GPU clock must not influence
        # CPU-domain launch selection or cadence.
        events.append(
            event(
                "graph_kernel",
                "kernel",
                10_010.0 + index * 80.0,
                60.0,
                pid=2,
                tid=7,
                args={"correlation": 7},
            )
        )
    events.extend(
        [
            event("hipEventSynchronize", "cuda_runtime", 110.0, 10.0),
            event("hipEventQuery", "cuda_runtime", 122.0, 3.0),
            event("hipMemcpyAsync", "cuda_runtime", 130.0, 5.0),
            event("aten::copy_", "cpu_op", 140.0, 20.0),
        ]
    )
    return events


class GraphReplayGapTest(unittest.TestCase):
    def test_graph_cadence_and_cpu_gpu_clock_skew(self):
        selected = {
            "trace": "synthetic.json",
            "graph_launch_api": "hipGraphLaunch",
            "graph_launch_correlation": "7",
            "gpu_timestamp_span": {
                "start_us": 10_010.0,
                "end_us": 10_090.0,
                "duration_us": 80.0,
            },
        }
        result = analyze_rank(synthetic_events(), selected, rank=0)
        self.assertEqual(result["launch_count"], 3)
        self.assertEqual(result["cadence_us"]["median"], 125.0)
        self.assertEqual(result["cadence_us"]["min"], 100.0)
        self.assertEqual(result["cadence_us"]["max"], 150.0)
        self.assertEqual(result["cadence_us"]["p95"], 147.5)
        self.assertEqual(result["graph_external_gap_us"]["median"], 45.0)
        self.assertTrue(
            result["annotation_consistency"]["consistent_with_launch_cadence"]
        )

    def test_sync_api_grouping(self):
        for name in (
            "hipStreamSynchronize",
            "hipEventSynchronize",
            "hipEventQuery",
            "hipStreamQuery",
            "hipStreamWaitEvent",
        ):
            self.assertEqual(
                classify_host_event(event(name, "cuda_runtime", 0.0, 1.0)),
                "sync_query",
            )
        self.assertEqual(
            classify_host_event(
                event("hipMemcpyDtoHAsync", "cuda_runtime", 0.0, 1.0)
            ),
            "memcpy",
        )

    def test_no_launch_failure(self):
        events = [
            event("decode[bs=2]", "user_annotation", 0.0, 100.0),
            event("aten::copy_", "cpu_op", 10.0, 5.0),
        ]
        with self.assertRaisesRegex(AnalysisError, "fewer than two"):
            select_decode_launches(events)

    def test_aggregate_case_uses_all_eight_ranks(self):
        selected = {
            "trace": "synthetic.json",
            "graph_launch_api": "hipGraphLaunch",
            "graph_launch_correlation": "7",
            "gpu_timestamp_span": {"duration_us": 80.0},
        }
        base = analyze_rank(synthetic_events(), selected, rank=0)
        reports = []
        for rank in range(8):
            report = copy.deepcopy(base)
            report["rank"] = rank
            report["cadence_us"]["median"] = 100.0 + rank
            report["graph_external_gap_us"]["median"] = 20.0 + rank
            reports.append(report)
        result = aggregate_case("atom-c2", reports, endpoint_tpot_ms=0.1035)
        aggregate = result["tp8_rank_aggregate"]
        self.assertEqual(aggregate["cadence_us"]["median"], 103.5)
        self.assertEqual(aggregate["cadence_us"]["min"], 100.0)
        self.assertEqual(aggregate["cadence_us"]["max"], 107.0)
        self.assertGreater(aggregate["cadence_us"]["cv"], 0.0)
        self.assertAlmostEqual(
            result["endpoint_reconciliation"]["cadence_minus_endpoint_tpot_ms"],
            0.0,
        )

    def test_missing_rank_rejected(self):
        with self.assertRaisesRegex(AnalysisError, "ranks 0-7"):
            aggregate_case("atom-c2", [], endpoint_tpot_ms=None)


if __name__ == "__main__":
    unittest.main()
