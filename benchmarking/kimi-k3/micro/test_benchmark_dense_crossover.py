"""CPU-only contract tests for benchmark_dense_crossover.py."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

import benchmark_dense_crossover as bench


class CaseGenerationTest(unittest.TestCase):
    def test_default_matrix_and_context_labels(self):
        cases = bench.generate_cases()
        # Six M values x (three modes for seven shapes + one latent-only mode).
        self.assertEqual(len(cases), 6 * (7 * 3 + 1))
        context = [case for case in cases if case.projection.context_only]
        self.assertEqual(
            {case.projection.name for case in context},
            {"merged_front", "kda_inproj"},
        )
        self.assertFalse(
            any(
                case.mode == "rmsnorm_mxfp4"
                and case.projection.name != "latent_up"
                for case in cases
            )
        )

    def test_shape_mode_and_m_filters(self):
        cases = bench.generate_cases(
            shape_filter={"shared_down"},
            mode_filter={"bf16", "ptpc_fp8"},
            m_values=(2, 64),
        )
        self.assertEqual(
            [case.case_id for case in cases],
            [
                "shared_down:m2:bf16",
                "shared_down:m2:ptpc_fp8",
                "shared_down:m64:bf16",
                "shared_down:m64:ptpc_fp8",
            ],
        )

    def test_unknown_filter_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown shapes"):
            bench.generate_cases(shape_filter={"not_a_projection"})
        with self.assertRaisesRegex(ValueError, "unknown modes"):
            bench.generate_cases(mode_filter={"gemm_only"})


class StorageTest(unittest.TestCase):
    def test_storage_formulas(self):
        n, k = 7168, 3584
        self.assertEqual(bench.nominal_prepared_weight_bytes(n, k, "bf16"), n * k * 2)
        self.assertEqual(
            bench.nominal_prepared_weight_bytes(n, k, "ptpc_fp8"),
            n * k + n * 4,
        )
        self.assertEqual(
            bench.nominal_prepared_weight_bytes(n, k, "mxfp4"),
            n * k // 2 + n * k // 32,
        )

    def test_dual_storage_uses_actual_prepared_bytes(self):
        record = bench.storage_record(
            16, 32, "mxfp4", actual_prepared_bytes=999
        )
        self.assertEqual(record["bf16_weight_bytes"], 1024)
        self.assertEqual(record["prepared_weight_bytes"], 999)
        self.assertEqual(record["incremental_dual_storage_bytes"], 999)
        self.assertEqual(record["dual_storage_total_bytes"], 2023)

    def test_bf16_has_no_incremental_dual_copy(self):
        record = bench.storage_record(16, 32, "bf16")
        self.assertEqual(record["incremental_dual_storage_bytes"], 0)
        self.assertEqual(
            record["dual_storage_total_bytes"], record["bf16_weight_bytes"]
        )

    def test_projection_layer_counts_and_output_representation(self):
        counts = {
            projection.name: projection.layer_count
            for projection in bench.PROJECTIONS
        }
        self.assertEqual(counts["latent_up"], 92)
        self.assertEqual(counts["shared_down"], 92)
        self.assertEqual(counts["merged_front"], 92)
        self.assertEqual(counts["kda_inproj"], 69)
        self.assertEqual(counts["mla_qkv_a"], 24)
        self.assertEqual(counts["mla_gate"], 24)
        self.assertEqual(counts["kda_mla_output"], 93)
        self.assertIn(
            "same TP8-local",
            bench.PROJECTION_BY_NAME["kda_mla_output"].layer_count_assumption,
        )

    def test_layer_weighted_recommended_policy_cost(self):
        rows = []
        for projection in bench.PROJECTIONS:
            for mode in ("bf16", "ptpc_fp8", "mxfp4"):
                case = bench.Case(projection, 2, mode)
                rows.append(bench.base_result(case))
        report = bench.build_report(rows)
        storage = bench.build_layer_weighted_storage_report(report)
        representative = storage["representative_shape_storage"]
        weighted = storage["model_layer_weighted_storage"]["recommended_policy"]
        estimate = storage["estimated_token_capacity_impact"]

        self.assertEqual(representative["incremental_prepared_bytes"], 52_739_840)
        self.assertEqual(
            weighted["incremental_prepared_bytes_per_gpu"], 3_271_323_136
        )
        self.assertAlmostEqual(
            weighted["incremental_prepared_gib_per_gpu"], 3.046657, places=6
        )
        self.assertEqual(estimate["estimated_token_loss"], 371_925)
        self.assertIn("not_a_measurement", estimate["label"])

    def test_layer_weighted_report_uses_recorded_actual_bytes(self):
        rows = []
        for projection in bench.PROJECTIONS:
            for mode in ("bf16", "ptpc_fp8", "mxfp4"):
                case = bench.Case(projection, 2, mode)
                row = bench.base_result(case)
                if projection.name == "kda_inproj" and mode == "mxfp4":
                    row["storage"] = bench.storage_record(
                        projection.n,
                        projection.k,
                        mode,
                        actual_prepared_bytes=23_969_792,
                    )
                rows.append(row)
        storage = bench.build_layer_weighted_storage_report(
            bench.build_report(rows)
        )
        policy = storage["model_layer_weighted_storage"]["recommended_policy"]
        contribution = next(
            item
            for item in policy["contributions"]
            if item["projection"] == "kda_inproj"
        )
        self.assertEqual(
            contribution["incremental_prepared_bytes_per_gpu"], 1_653_915_648
        )


class InputChangeReplayGateTest(unittest.TestCase):
    def test_gate_passes_for_finite_changed_output(self):
        output_a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        output_b = torch.tensor([[-1.0, -2.0]], dtype=torch.float32)
        metrics = bench.assess_input_change_replay(torch, output_a, output_b)
        self.assertTrue(metrics["input_change_replay_passed"])
        self.assertGreater(metrics["input_change_output_delta_norm"], 0.0)
        self.assertNotEqual(
            metrics["input_change_output_a_sha256"],
            metrics["input_change_output_b_sha256"],
        )

    def test_gate_fails_for_stale_equal_output(self):
        output_a = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        metrics = bench.assess_input_change_replay(
            torch, output_a, output_a.clone()
        )
        self.assertFalse(metrics["input_change_replay_passed"])
        self.assertEqual(metrics["input_change_output_delta_norm"], 0.0)
        self.assertEqual(
            metrics["input_change_output_a_sha256"],
            metrics["input_change_output_b_sha256"],
        )

    def test_case_failure_preserves_gate_diagnostics(self):
        case = bench.generate_cases(
            shape_filter={"latent_up"},
            mode_filter={"bf16"},
            m_values=(2,),
        )[0]
        diagnostics = {
            "input_change_replay_passed": False,
            "input_change_output_delta_norm": 0.0,
            "input_change_output_a_sha256": "same",
            "input_change_output_b_sha256": "same",
        }

        def stale(_case):
            raise bench.CaseFailure("stale replay", diagnostics)

        row = bench.run_cases([case], stale)[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["skip_reason"], "stale replay")
        self.assertFalse(row["input_change_replay_passed"])
        self.assertEqual(row["input_change_output_a_sha256"], "same")


class SchemaAndSkipTest(unittest.TestCase):
    def setUp(self):
        self.case = bench.generate_cases(
            shape_filter={"latent_up"},
            mode_filter={"bf16"},
            m_values=(2,),
        )[0]

    def test_mock_unsupported_mode_becomes_reasoned_skip(self):
        def unsupported(_case):
            raise bench.UnsupportedMode("mock API absent")

        rows = bench.run_cases([self.case], unsupported)
        self.assertEqual(rows[0]["status"], "skipped")
        self.assertEqual(rows[0]["skip_reason"], "mock API absent")
        self.assertIsNone(rows[0]["graph_ms"])

    def test_unexpected_mock_error_is_failed_not_equivalent(self):
        def broken(_case):
            raise TypeError("mock signature drift")

        rows = bench.run_cases([self.case], broken)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertIn("mock signature drift", rows[0]["skip_reason"])

    def test_json_and_csv_output_schema(self):
        row = bench.base_result(self.case)
        row.update(
            {
                "status": "ok",
                "graph_ms": 0.01,
                "rel_l2": 0.0,
                "cosine": 1.0,
                "finite": True,
                "input_change_replay_passed": True,
                "input_change_output_delta_norm": 2.0,
                "input_change_output_a_sha256": "a",
                "input_change_output_b_sha256": "b",
                "input_change_reference_rel_l2": 0.0,
                "input_change_reference_cosine": 1.0,
                "dispatch": {
                    "api": "mock",
                    "config": {"libtype": "mock"},
                    "weight_layout": "bf16_nk",
                },
            }
        )
        args = argparse.Namespace(
            seed=7, warmup=1, iterations=2, eager=False
        )
        report = bench.build_report([row], args=args, runtime_metadata={"gfx": "mock"})
        bench.validate_report(report)
        self.assertEqual(report["schema_version"], bench.SCHEMA_VERSION)
        self.assertIn("paired GSM8K correctness", report["promotion_requirements"])

        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "result.json"
            csv_path = Path(directory) / "result.csv"
            bench.write_json(json_path, report)
            bench.write_csv(csv_path, [row])
            loaded = json.loads(json_path.read_text())
            self.assertEqual(loaded["cases"][0]["case_id"], self.case.case_id)
            with csv_path.open(newline="") as stream:
                csv_row = next(csv.DictReader(stream))
            self.assertEqual(csv_row["dispatch_api"], "mock")
            self.assertEqual(csv_row["projection"], "latent_up")
            self.assertEqual(csv_row["input_change_replay_passed"], "True")

    def test_skipped_schema_requires_reason(self):
        row = bench.base_result(self.case)
        row["status"] = "skipped"
        report = bench.build_report([row])
        with self.assertRaisesRegex(ValueError, "without a reason"):
            bench.validate_report(report)


if __name__ == "__main__":
    unittest.main()
