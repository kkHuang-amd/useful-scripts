import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aggregate_trace_attribution import (
    OPS,
    aggregate_case,
    classify_operation,
    compare_cases,
    distribution,
    map_kernel,
    operation_allocations,
)
from kernel_families import classify_kernel


class AggregateTraceAttributionTest(unittest.TestCase):
    def test_distribution_uses_population_cv(self):
        result = distribution([1.0, 2.0, 3.0])
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["min"], 1.0)
        self.assertEqual(result["max"], 3.0)
        self.assertAlmostEqual(result["cv"], (2.0 / 3.0) ** 0.5 / 2.0)

    def test_moe_silu_is_not_norm(self):
        name = "mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x64x256"
        self.assertEqual(classify_kernel(name), "moe_gemm")
        self.assertEqual(classify_operation(name), "routed_moe_stage1")
        self.assertEqual(
            classify_kernel("gemm2_a16w4_port_ne896_h3584_i384_bm32_tn128"),
            "moe_gemm",
        )

    def test_mapping_confidence_distinguishes_annotation_quality(self):
        detailed = [
            {
                "count": 2,
                "confidence": "direct",
                "caller": "aiter::fused_moe_",
                "annotation": "language_model.model.layers.1.block_sparse_moe",
            }
        ]
        self.assertEqual(map_kernel("atom", 2, detailed)["confidence"], "graph-map")
        self.assertEqual(map_kernel("sglang", 2, detailed)["confidence"], "inferred")
        self.assertEqual(map_kernel("atom", 3, detailed)["confidence"], "inferred")
        self.assertEqual(
            map_kernel("sglang", 2, detailed, phase="prefill")["confidence"],
            "direct",
        )

    def test_reused_kernel_is_split_by_warmup_occurrences(self):
        kernel = "_gemm_kernel"
        entries = [
            {
                "count": 3,
                "caller": "aten::mm",
                "annotation": "apply_attn_res",
            },
            {
                "count": 1,
                "caller": "aten::mm",
                "annotation": "shared_experts.down_proj",
            },
        ]
        allocations = {
            item["operation"]: item
            for item in operation_allocations(kernel, 8, 4.0, entries)
        }
        self.assertEqual(allocations["attention_residual_add3"]["count"], 6.0)
        self.assertEqual(allocations["attention_residual_add3"]["duration_us"], 3.0)
        self.assertEqual(allocations["shared_expert_down"]["count"], 2.0)
        self.assertEqual(allocations["shared_expert_down"]["duration_us"], 1.0)
        self.assertEqual(
            classify_operation(
                "kernel_gemm_0",
                [{"annotation": "language_model.model.layers.6.self_attn.in_proj"}],
            ),
            "kda_inproj",
        )
        self.assertEqual(
            classify_operation(
                "aiter::f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128",
                [{"annotation": "block_sparse_moe.routed_expert_down_proj"}],
            ),
            "routed_moe_stage2",
        )
        self.assertEqual(
            classify_operation(
                "kernel_gemm_0",
                [{"annotation": "model.block_sparse_moe.shared_experts.gate_up_proj"}],
            ),
            "moe_front_merged",
        )
        self.assertEqual(
            classify_operation(
                "kernel_gemm_0",
                [{"annotation": "model.block_sparse_moe.shared_experts.down_proj"}],
            ),
            "shared_expert_down",
        )
        self.assertEqual(
            classify_operation(
                "kernel_gemm_0",
                [{"annotation": "model.self_attn.fused_qkv_a_proj"}],
            ),
            "mla_qkv_a",
        )

    def test_sglang_shape_and_count_semantics(self):
        split = operation_allocations(
            "hgemm_bf16_32x64x128x4_SPK1_W2x2x1_BLDS1_TN_AS1_0",
            93,
            930.0,
            [],
            engine="sglang",
            concurrency=64,
        )
        self.assertEqual(
            [item["operation"] for item in split],
            ["kda_output_projection", "mla_output_projection"],
        )
        self.assertEqual([item["count"] for item in split], [69.0, 24.0])
        self.assertEqual([item["duration_us"] for item in split], [690.0, 240.0])
        front = operation_allocations(
            "hgemm_bf16_64x64x128x3_SPK2_W2x2x2_BLDS1_TN_AS1_0",
            92,
            1000.0,
            [],
            engine="sglang",
            concurrency=64,
        )
        self.assertEqual(front[0]["operation"], "moe_front_merged")
        self.assertEqual(front[0]["semantic_confidence"], "direct_shape")
        prefill = operation_allocations(
            "hgemm_bf16_64x64x128x3_SPK2_W2x2x2_BLDS1_TN_AS1_0",
            92,
            1000.0,
            [],
            engine="sglang",
            concurrency=64,
            phase="prefill",
        )
        self.assertEqual(prefill[0]["operation"], "other_dense_gemms")

    def test_atom_annotation_semantic_confidence(self):
        allocated = operation_allocations(
            "kernel_gemm_0",
            24,
            240.0,
            [
                {
                    "count": 24,
                    "annotation": "model.self_attn.fused_qkv_a_proj",
                }
            ],
            engine="atom",
            concurrency=64,
        )
        self.assertEqual(allocated[0]["operation"], "mla_qkv_a")
        self.assertEqual(
            allocated[0]["semantic_confidence"],
            "direct_annotation_known_shape",
        )

    def test_aggregate_case_and_comparison(self):
        endpoint = {
            "median_tpot_ms": 2.5,
            "median_ttft_ms": 10.0,
            "median_e2e_ms": 30.0,
            "total_token_throughput_tok_s": 100.0,
        }
        kernel = "mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x64x256"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(8):
                analysis = root / "atom-c2" / "analysis"
                analysis.mkdir(parents=True, exist_ok=True)
                selected = {
                    "gpu_timestamp_span": {"duration_us": 2000.0 + rank},
                    "kernel_count": 2,
                    "streams": ["3"],
                    "top_kernels": [
                        {"name": kernel, "count": 2, "duration_us": 1000.0 + rank}
                    ],
                }
                (analysis / f"rank-{rank}-selected-step.json").write_text(
                    json.dumps(selected), encoding="utf-8"
                )
                warm = (
                    root
                    / "graph-attribution/atom/analysis/bs2"
                    / f"rank{rank}-caller-map.jsonl"
                )
                warm.parent.mkdir(parents=True, exist_ok=True)
                warm.write_text(
                    json.dumps(
                        {
                            "kernel": kernel,
                            "count": 2,
                            "confidence": "direct",
                            "caller": "aiter::fused_moe_",
                            "annotation": (
                                "language_model.model.layers.1."
                                "block_sparse_moe.experts.fused_moe"
                            ),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                prefill = (
                    root
                    / "prefill-steps/atom-c2/analysis/selected-prefill"
                )
                prefill.mkdir(parents=True, exist_ok=True)
                prefill_summary = {
                    "capture_window": {
                        "annotation_selection": {
                            "name": (
                                "prefill[bs=1 tok=8192 ctx=8192 "
                                "sqsq=67108864 sqsk=67108864 sk=8192]"
                            ),
                            "duration_us": 4000.0 + rank,
                        }
                    },
                    "kernel_count": 2,
                    "kernel_busy_union_us": 1000.0 + rank,
                    "active_stream_count": 1,
                    "top_kernels": [
                        {"name": kernel, "count": 2, "duration_us": 1000.0 + rank}
                    ],
                }
                (prefill / f"rank-{rank}-summary.json").write_text(
                    json.dumps(prefill_summary), encoding="utf-8"
                )
                (prefill / f"rank-{rank}-caller-map.jsonl").write_text(
                    json.dumps(
                        {
                            "kernel": kernel,
                            "count": 2,
                            "confidence": "direct",
                            "host_api": "hipModuleLaunchKernel",
                            "caller": "aiter::fused_moe_",
                            "annotation": (
                                "language_model.model.layers.1."
                                "block_sparse_moe.experts.fused_moe"
                            ),
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            result = aggregate_case(root, "atom-c2", endpoint)
            prefill_result = aggregate_case(
                root, "atom-c2", endpoint, phase="prefill"
            )

        self.assertTrue(
            result["rank_consistency"]["kernel_name_count_signatures_identical"]
        )
        self.assertEqual(result["span_us"]["median"], 2003.5)
        self.assertEqual(result["kernel_count"]["median"], 2.0)
        self.assertEqual(
            result["mapping_coverage"]["mapped_kernel_occurrence_pct"], 100.0
        )
        op = {
            item["operation"]: item for item in result["operations"]
        }["routed_moe_stage1"]
        self.assertEqual(op["duration_us"]["median"], 1003.5)

        right = json.loads(json.dumps(result))
        right["case"] = "atom-c64"
        right["span_us"]["median"] = 3000.0
        right["endpoint"]["median_tpot_ms"] = 3.5
        right["endpoint"]["median_ttft_ms"] = 8.0
        right["endpoint"]["total_token_throughput_tok_s"] = 200.0
        for item in right["operations"]:
            if item["operation"] == "routed_moe_stage1":
                item["duration_us"]["median"] = 1500.0
        comparison = compare_cases(result, right, "concurrency_scale")
        self.assertAlmostEqual(comparison["span"]["delta_ms"], 0.9965)
        self.assertEqual(comparison["endpoint"]["throughput_delta_pct"], 100.0)
        stage1 = next(
            item
            for item in comparison["operations"]
            if item["operation"] == "routed_moe_stage1"
        )
        self.assertAlmostEqual(stage1["delta_ms"], 0.4965)
        self.assertEqual({item["operation"] for item in result["operations"]}, set(OPS))
        self.assertEqual(prefill_result["phase"], "prefill")
        self.assertEqual(prefill_result["input_tokens"], 8192)
        self.assertEqual(prefill_result["span_us"]["median"], 4003.5)
        self.assertTrue(
            prefill_result["validation_gates"]["no_decode_overlap_all_ranks"]
        )
        self.assertEqual(
            prefill_result["mapping_coverage"]["mapped_duration_pct"], 100.0
        )


if __name__ == "__main__":
    unittest.main()
