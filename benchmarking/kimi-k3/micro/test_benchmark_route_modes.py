"""CPU-only contracts for benchmark_route_modes.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import benchmark_route_modes as bench


def balanced_counts() -> list[int]:
    counts = [0] * bench.EXPERTS
    for expert in range(bench.TOKENS * bench.TOPK):
        counts[expert % bench.EXPERTS] += 1
    return counts


def payload(call_index: int, counts: list[int]) -> dict:
    active = sum(value > 0 for value in counts)
    return {
        "schema": "k3-route-dump-v1",
        "armed": True,
        "arm_time_ns": 10,
        "arm_monotonic_ns": 20,
        "dump_time_ns": 11,
        "dump_monotonic_ns": 21,
        "call_index": call_index,
        "rank": 0,
        "topk_shape": [bench.TOKENS, bench.TOPK],
        "total_routes": bench.TOKENS * bench.TOPK,
        "expert_count_size": bench.EXPERTS,
        "full_bincount": counts,
        "unique_active_experts": active,
        "bm32_padded_blocks": bench.bm32_blocks(counts),
    }


class ReconstructionTest(unittest.TestCase):
    def test_preserves_exact_ids_counts_and_unique_rows(self):
        counts = balanced_counts()
        rows = bench.reconstruct_topk_ids(counts)
        observed = [0] * bench.EXPERTS
        self.assertEqual(len(rows), bench.TOKENS)
        for row in rows:
            self.assertEqual(len(row), bench.TOPK)
            self.assertEqual(len(set(row)), bench.TOPK)
            for expert in row:
                observed[expert] += 1
        self.assertEqual(observed, counts)

    def test_skewed_valid_sequence_is_realized(self):
        counts = [0] * bench.EXPERTS
        for expert in range(bench.TOPK):
            counts[expert] = bench.TOKENS
        rows = bench.reconstruct_topk_ids(counts)
        self.assertTrue(all(row == list(range(bench.TOPK)) for row in rows))

    def test_impossible_duplicate_requirement_fails(self):
        counts = balanced_counts()
        needed = bench.TOKENS + 1 - counts[0]
        counts[0] += needed
        for expert in range(1, len(counts)):
            take = min(needed, counts[expert])
            counts[expert] -= take
            needed -= take
            if not needed:
                break
        with self.assertRaisesRegex(bench.ContractError, "one-per-token"):
            bench.reconstruct_topk_ids(counts)

    def test_wrong_route_total_fails(self):
        counts = balanced_counts()
        counts[0] -= 1
        with self.assertRaisesRegex(bench.ContractError, "tokens\\*topk"):
            bench.reconstruct_topk_ids(counts)


class LoadingAndAlignmentTest(unittest.TestCase):
    def test_requires_exact_92_rank0_calls(self):
        counts = balanced_counts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for layer in range(92):
                (root / f"call-{layer:03d}.json").write_text(
                    json.dumps(payload(layer, counts))
                )
            routes = bench.load_rank0_routes(root, "mock")
            self.assertEqual([case.layer for case in routes], list(range(92)))
            self.assertTrue(all(case.counts == tuple(counts) for case in routes))

            (root / "call-091.json").unlink()
            with self.assertRaisesRegex(bench.ContractError, "missing=\\[91\\]"):
                bench.load_rank0_routes(root, "mock")

    def test_unarmed_dump_is_rejected(self):
        counts = balanced_counts()
        record = payload(0, counts)
        record["armed"] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(bench.ContractError, "unarmed"):
                bench.load_rank0_routes(path.parent, "mock")

    def test_bm32_metadata_is_recomputed(self):
        counts = balanced_counts()
        record = payload(0, counts)
        record["bm32_padded_blocks"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(bench.ContractError, "BM32 metadata"):
                bench.load_rank0_routes(path.parent, "mock")


class SchemaAndAnalysisTest(unittest.TestCase):
    def test_fixed_weights_are_deterministic_and_normalized(self):
        first = bench.fixed_topk_weights()
        second = bench.fixed_topk_weights()
        self.assertEqual(first, second)
        self.assertEqual(len(first), bench.TOKENS)
        self.assertAlmostEqual(sum(first[0]), 1.0)
        self.assertTrue(all(row == first[0] for row in first))

    def test_regression_and_aggregate_schema(self):
        case0 = bench.RouteCase("sglang", 0, tuple(), "x", 10, 100)
        case1 = bench.RouteCase("sglang", 1, tuple(), "y", 11, 120)
        rows = []
        for case, latency in ((case0, 1.0), (case1, 2.0)):
            row = bench.base_row(case, "a8w4", "full")
            row.update(
                status="ok", sample_count=2, min_ms=latency,
                p50_ms=latency, p90_ms=latency, max_ms=latency,
                mean_ms=latency, samples_ms=[latency, latency],
            )
            rows.append(row)
        fit = bench.regression(rows)
        self.assertAlmostEqual(fit["slope_ms_per_block"], 0.05)
        aggregates = bench.aggregate_rows(rows)
        selected = next(
            item for item in aggregates
            if (item["source"], item["mode"], item["scope"])
            == ("sglang", "a8w4", "full")
        )
        self.assertEqual(selected["ok_layers"], 2)
        self.assertIsNone(selected["sum_92_layer_p50_ms"])
        self.assertEqual(set(selected["representative"]), {"min", "median", "max"})

    def test_report_schema_and_reasoned_skips(self):
        case = bench.RouteCase("sglang", 0, tuple(), "x", 1, 1)
        row = bench.skipped_row(case, "a8w4", "stage1", "hook unavailable")
        report = {
            "schema_version": bench.SCHEMA_VERSION,
            "created_at": "now",
            "benchmark": "test",
            "configuration": {},
            "runtime": {},
            "routes": {},
            "rows": [row],
            "numerical_comparisons": [],
            "aggregates": [],
            "claims": {"endpoint_claim": False},
        }
        bench.validate_report(report)
        row["skip_reason"] = None
        with self.assertRaisesRegex(bench.ContractError, "without reason"):
            bench.validate_report(report)


if __name__ == "__main__":
    unittest.main()
