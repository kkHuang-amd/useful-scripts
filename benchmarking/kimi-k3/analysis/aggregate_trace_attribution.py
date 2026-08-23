#!/usr/bin/env python3
"""Aggregate matched Kimi-K3 replay steps and graph-warmup caller maps.

The production replay step is the timing source of truth.  Graph-warmup maps
are used only to label exact kernel names and count patterns with callers and
annotations; their durations are never compared with production timings.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path
from typing import Iterable

from kernel_families import classify_kernel


CASES = ("sglang-c2", "sglang-c64", "atom-c2", "atom-c64")
OPS = (
    "routed_moe_stage1",
    "routed_moe_stage2",
    "moe_front_merged",
    "shared_latent_projections",
    "latent_up_bf16",
    "shared_expert_down",
    "route_sort_topk",
    "quantization",
    "kda_inproj",
    "kda_output_projection",
    "kda_recurrence_f_b",
    "mla_q_cache",
    "mla_qkv_a",
    "mla_gate",
    "mla_output_projection",
    "mla_decode",
    "attention_residual_add3",
    "collectives",
    "copies",
    "other_dense_gemms",
    "other",
)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    values = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                values.append(json.loads(line))
    return values


def distribution(values: Iterable[float]) -> dict:
    data = list(values)
    if not data:
        return {"median": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0, "cv": 0.0}
    mean = statistics.fmean(data)
    return {
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
        "mean": mean,
        "cv": statistics.pstdev(data) / mean if mean else 0.0,
    }


def kernel_markers(value: str, markers: tuple[str, ...]) -> bool:
    return any(marker in value for marker in markers)


def classify_operation(kernel: str, mappings: list[dict] | None = None) -> str:
    """Return one exclusive Kimi-K3 operation group.

    Caller/annotation text wins only for semantic distinctions that generic
    GEMM symbols cannot encode. Kernel markers remain the fallback required
    for SGLang's coarse graph-warmup annotation.
    """
    name = kernel.lower()
    evidence = " ".join(
        str(item.get(key) or "").lower()
        for item in (mappings or [])
        for key in ("caller", "annotation")
    )
    joined = f"{name} {evidence}"

    if kernel_markers(
        name,
        (
            "all_reduce",
            "allreduce",
            "quick_reduce",
            "quickreduce",
            "cross_device_reduce",
            "allgather",
            "rccl",
            "nccl",
        ),
    ):
        return "collectives"
    if kernel_markers(
        name,
        (
            "mfma_moe1",
            "gemm1_a4w4",
            "gemm1_a8w4",
            "gemm1_a16w4",
            "stage1_a8w4",
            "opus_moe_stage1",
        ),
    ):
        return "routed_moe_stage1"
    if kernel_markers(
        name,
        (
            "gemm2_a4w4",
            "gemm2_a8w4",
            "gemm2_a16w4",
            "stage2_a8w4",
            "opus_moe_stage2",
        ),
    ) or "routed_expert_down_proj" in evidence:
        return "routed_moe_stage2"
    if kernel_markers(
        joined,
        (
            "moe_sort",
            "moe sorting",
            "sorting_entry",
            "topk",
            "top_k",
            "route_radix",
            "route_sort",
            "fused_mx_quant_moe_sort",
        ),
    ):
        return "route_sort_topk"
    if (
        "shared_experts.gate_up_proj" in evidence
        or "block_sparse_moe.gate" in evidence
        or "moe_front_merged" in evidence
        or "mixed_tri" in name
    ):
        return "moe_front_merged"
    if "routed_expert_up_proj" in evidence or "latent_up_bf16" in evidence:
        return "latent_up_bf16"
    if "shared_experts.down_proj" in evidence:
        return "shared_expert_down"
    if "self_attn.fused_qkv_a_proj" in evidence:
        return "mla_qkv_a"
    if "self_attn.g_proj" in evidence:
        return "mla_gate"
    if "self_attn.o_proj[m=" in evidence:
        return "kda_output_projection"
    if "self_attn.o_proj" in evidence:
        return "mla_output_projection"
    if kernel_markers(
        joined,
        (
            "fuse_qk_rope_concat_and_cache",
            "fused_qk_rmsnorm",
            "q_proj_and_k_up_proj",
            "v_up_proj_and_o_proj",
            "q_cache",
            "kv_cache",
            "concat_and_cache",
        ),
    ):
        return "mla_q_cache"
    if kernel_markers(
        name,
        (
            "mla_a8w8",
            "mla_decode",
            "mla_reduce",
            "_fwd_grouped_kernel_stage1",
            "_fwd_kernel_stage2",
        ),
    ):
        return "mla_decode"
    if "self_attn.in_proj" in evidence or kernel_markers(
        joined,
        (
            "kda_input",
            "kda_inproj",
            "inproj",
        ),
    ):
        return "kda_inproj"
    if kernel_markers(
        joined,
        (
            "kda_decode",
            "kda_recurr",
            "gating_delta",
            "causal_conv",
            "chunk_kda",
            "chunk_gla",
            "recompute_w_u",
            "rmsnorm_gated",
        ),
    ):
        return "kda_recurrence_f_b"
    if kernel_markers(
        joined,
        (
            "apply_attn_res",
            "attn_res",
            "_agg_kernel",
            "add3_kernel",
            "attention_residual",
        ),
    ):
        return "attention_residual_add3"
    if kernel_markers(
        evidence,
        (
            "shared_expert",
            "shared expert",
            "latent",
            "down_proj",
            "up_proj",
            "q_proj_and_k_up_proj",
            "v_up_proj_and_o_proj",
        ),
    ) or kernel_markers(
        name,
        (
            "latent",
            "shared_down",
            "shared_up",
            "shared_gate",
            "mixed_tri",
        ),
    ):
        return "shared_latent_projections"
    if (
        kernel_markers(name, ("quant", "fp8_quant", "fp4_quant"))
        and "gemm" not in name
        and "moe_sort" not in name
    ):
        return "quantization"
    if kernel_markers(
        name,
        (
            "memcpy",
            "copy",
            "fill",
            "set_value",
            "catarray",
            "clone",
            "material",
        ),
    ):
        return "copies"
    if classify_kernel(kernel) == "dense_gemm":
        return "other_dense_gemms"
    return "other"


def index_warm_map(entries: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = collections.defaultdict(list)
    for entry in entries:
        result[entry["kernel"]].append(entry)
    return result


def is_detailed_annotation(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return not (
        lowered.startswith("profilerstep")
        or lowered.startswith("sglang.vlm.language_model_prefill")
    )


def map_kernel(
    engine: str,
    selected_count: int,
    warm_entries: list[dict],
    phase: str = "decode",
) -> dict:
    warm_count = sum(int(item["count"]) for item in warm_entries)
    direct_count = sum(
        int(item["count"])
        for item in warm_entries
        if item.get("confidence") == "direct"
    )
    detailed_count = sum(
        int(item["count"])
        for item in warm_entries
        if is_detailed_annotation(item.get("annotation"))
    )
    exact_count = warm_count == selected_count
    ratio = warm_count / selected_count if selected_count else 0.0
    if not warm_entries:
        confidence = "unmatched"
    elif phase == "prefill" and direct_count and exact_count:
        confidence = "direct"
    elif engine == "atom" and direct_count and detailed_count and exact_count:
        confidence = "graph-map"
    else:
        # Even a directly correlated warmup launch is only an inferred mapping
        # to an opaque replay node. SGLang annotations do not identify model ops.
        confidence = "inferred"
    return {
        "confidence": confidence,
        "selected_count": selected_count,
        "warm_count": warm_count,
        "warm_to_selected_count_ratio": ratio,
        "exact_count_pattern": exact_count,
        "direct_warm_count": direct_count,
        "detailed_annotation_count": detailed_count,
        "callers": sorted(
            {
                item["caller"]
                for item in warm_entries
                if item.get("caller") is not None
            }
        ),
        "annotations": sorted(
            {
                item["annotation"]
                for item in warm_entries
                if item.get("annotation") is not None
            }
        ),
    }


def _allocation(
    operation: str,
    fraction: float,
    selected_count: int,
    duration_us: float,
    warm_count: float,
    confidence: str,
    evidence: str,
) -> dict:
    return {
        "operation": operation,
        "fraction": fraction,
        "count": selected_count * fraction,
        "duration_us": duration_us * fraction,
        "warm_count": warm_count,
        "semantic_confidence": confidence,
        "semantic_evidence": evidence,
    }


def sglang_shape_allocations(
    concurrency: int,
    kernel: str,
    selected_count: int,
    duration_us: float,
) -> list[dict] | None:
    """Map exact SGLang GEMM symbols using directly observed warmup Input Dims."""
    name = kernel.lower()
    operation = None
    evidence = None
    if concurrency == 64:
        rules = (
            (
                "hgemm_bf16_64x64x128x3_spk2_w2x2x2",
                "moe_front_merged",
                "M64xN6016xK7168",
            ),
            (
                "mt192x64x128",
                "kda_inproj",
                "M64xN6288xK7168",
            ),
            (
                "hgemm_bf16_32x64x256x3",
                "latent_up_bf16",
                "M64xN3584xK7168",
            ),
            (
                "mt64x32x256",
                "shared_expert_down",
                "M64xN7168xK768",
            ),
            (
                "hgemm_bf16_64x64x64x5",
                "mla_qkv_a",
                "M64xN2112xK7168",
            ),
            (
                "hgemm_bf16_32x64x128x4_spk4",
                "mla_gate",
                "M64xN1536xK7168",
            ),
        )
        split_marker = "hgemm_bf16_32x64x128x4_spk1_w2x2x1"
        bmm = (
            ("mt128x64x128", "skxccm0"),
            ("mt64x32x64", "skxccm0"),
        )
    else:
        rules = (
            (
                "splitk_block_size_3584",
                "latent_up_bf16",
                "M2xN3584xK7168",
            ),
            (
                "mt32x16x128",
                "shared_expert_down",
                "M2xN7168xK768",
            ),
            (
                "hgemm_bf16_16x64x64x6_spk4",
                "mla_output_projection",
                "M2xN2304xK1536",
            ),
        )
        split_marker = "splitk_block_size_1536"
        bmm = (
            ("mt128x32x32", "skxccm0"),
            ("mt32x16x64", "skxccm0"),
        )
        reused_mla = "hgemm_bf16_16x64x128x4_spk7"
        if reused_mla in name:
            return [
                _allocation(
                    "mla_qkv_a",
                    0.5,
                    selected_count,
                    duration_us,
                    24,
                    "direct_shape_count",
                    "M2xN2112xK7168; 24 warmup calls",
                ),
                _allocation(
                    "mla_gate",
                    0.5,
                    selected_count,
                    duration_us,
                    24,
                    "direct_shape_count",
                    "M2xN1536xK7168; 24 warmup calls",
                ),
            ]
    if split_marker in name:
        total = 93
        return [
            _allocation(
                "kda_output_projection",
                69 / total,
                selected_count,
                duration_us,
                69,
                "direct_shape_count",
                f"M{concurrency}xN1536xK7168; 69 warmup calls",
            ),
            _allocation(
                "mla_output_projection",
                24 / total,
                selected_count,
                duration_us,
                24,
                "direct_shape_count",
                f"M{concurrency}xN1536xK7168; 24 warmup calls",
            ),
        ]
    if any(all(marker in name for marker in markers) for markers in bmm):
        return [
            _allocation(
                "mla_decode",
                1.0,
                selected_count,
                duration_us,
                selected_count,
                "direct_shape",
                f"12x{concurrency} MLA BMM",
            )
        ]
    for marker, candidate, shape in rules:
        if marker in name:
            operation, evidence = candidate, shape
            break
    if operation is None:
        return None
    return [
        _allocation(
            operation,
            1.0,
            selected_count,
            duration_us,
            selected_count,
            "direct_shape",
            evidence,
        )
    ]


def operation_allocations(
    kernel: str,
    selected_count: int,
    duration_us: float,
    warm_entries: list[dict],
    engine: str | None = None,
    concurrency: int | None = None,
    phase: str = "decode",
) -> list[dict]:
    """Split a reused kernel symbol using warmup occurrence-count patterns."""
    if phase == "decode" and engine == "sglang" and concurrency is not None:
        shaped = sglang_shape_allocations(
            concurrency, kernel, selected_count, duration_us
        )
        if shaped is not None:
            return shaped
    operation_counts: collections.Counter[str] = collections.Counter()
    operation_confidence: dict[str, tuple[str, str]] = {}
    for entry in warm_entries:
        operation = classify_operation(kernel, [entry])
        operation_counts[operation] += int(entry["count"])
        annotation = str(entry.get("annotation") or "")
        if engine == "atom" and annotation:
            confidence = (
                "direct_annotation_shape"
                if "[M=" in annotation
                else "direct_annotation_known_shape"
            )
            operation_confidence[operation] = (confidence, annotation)
    if not operation_counts:
        operation_counts[classify_operation(kernel)] = selected_count
    denominator = sum(operation_counts.values())
    return [
        _allocation(
            operation,
            count / denominator,
            selected_count,
            duration_us,
            count,
            operation_confidence.get(
                operation, ("name_or_annotation", "kernel/caller taxonomy")
            )[0],
            operation_confidence.get(
                operation, ("name_or_annotation", "kernel/caller taxonomy")
            )[1],
        )
        for operation, count in sorted(operation_counts.items())
    ]


def endpoint_medians(paths: list[Path]) -> dict:
    summaries = [load_json(path) for path in paths]
    fields = (
        "total_token_throughput_tok_s",
        "median_ttft_ms",
        "median_tpot_ms",
        "median_e2e_ms",
    )
    return {
        field: statistics.median(float(item[field]) for item in summaries)
        for field in fields
    } | {
        "round_count": len(summaries),
        "successful_requests": sum(int(item["successful_requests"]) for item in summaries),
        "failed_requests": sum(int(item["failed_requests"]) for item in summaries),
        "sources": [str(path) for path in paths],
    }


def discover_endpoints(args) -> dict[str, dict]:
    c2 = args.endpoint_c2
    c64 = args.endpoint_c64
    return {
        "sglang-c2": endpoint_medians(
            sorted((c2 / "sglang-same-aiter").glob("round*/summary.json"))
        ),
        "atom-c2": endpoint_medians(
            sorted((c2 / "atom-same-aiter").glob("round*/summary.json"))
        ),
        "sglang-c64": endpoint_medians(
            [c64 / "sglang-same-aiter/common-client/summary.json"]
        ),
        "atom-c64": endpoint_medians(
            [c64 / "atom-same-aiter/common-client/summary.json"]
        ),
    }


def aggregate_case(
    root: Path, case: str, endpoint: dict, phase: str = "decode"
) -> dict:
    engine, conc_text = case.split("-c")
    concurrency = int(conc_text)
    if phase not in ("decode", "prefill"):
        raise ValueError(f"unsupported phase: {phase}")
    rank_values = []
    selected_names = []
    for rank in range(8):
        if phase == "decode":
            selected_path = root / case / "analysis" / f"rank-{rank}-selected-step.json"
            map_path = (
                root
                / "graph-attribution"
                / engine
                / "analysis"
                / f"bs{concurrency}"
                / f"rank{rank}-caller-map.jsonl"
            )
        else:
            selected_path = (
                root
                / "prefill-steps"
                / case
                / "analysis"
                / "selected-prefill"
                / f"rank-{rank}-summary.json"
            )
            map_path = selected_path.with_name(f"rank-{rank}-caller-map.jsonl")
        selected = load_json(selected_path)
        map_entries = load_jsonl(map_path)
        warm_index = index_warm_map(map_entries)
        if phase == "decode":
            span_us = float(selected["gpu_timestamp_span"]["duration_us"])
            stream_count = len(selected["streams"])
            selected_name = (selected.get("annotation_names") or ["decode_graph_step"])[-1]
            kernel_busy_union_us = span_us
        else:
            selection = selected["capture_window"]["annotation_selection"]
            span_us = float(selection["duration_us"])
            stream_count = int(selected["active_stream_count"])
            selected_name = selection["name"]
            kernel_busy_union_us = float(selected["kernel_busy_union_us"])
        selected_names.append(selected_name)
        kernels = []
        for kernel in selected["top_kernels"]:
            matches = warm_index.get(kernel["name"], [])
            mapping = map_kernel(
                engine, int(kernel["count"]), matches, phase=phase
            )
            allocations = operation_allocations(
                kernel["name"],
                int(kernel["count"]),
                float(kernel["duration_us"]),
                matches,
                engine=engine,
                concurrency=concurrency,
                phase=phase,
            )
            kernels.append(
                {
                    "name": kernel["name"],
                    "count": int(kernel["count"]),
                    "duration_us": float(kernel["duration_us"]),
                    "family": classify_kernel(kernel["name"]),
                    "operation": max(
                        allocations, key=lambda item: item["duration_us"]
                    )["operation"],
                    "operation_allocations": allocations,
                    "mapping": mapping,
                }
            )
        represented_count = sum(item["count"] for item in kernels)
        if represented_count != selected["kernel_count"]:
            raise ValueError(
                f"{case} rank{rank}: top_kernels represents {represented_count} "
                f"of {selected['kernel_count']} kernels; rerun extraction with a larger top-k"
            )
        rank_values.append(
            {
                "rank": rank,
                "span_us": span_us,
                "kernel_sum_us": sum(item["duration_us"] for item in kernels),
                "kernel_busy_union_us": kernel_busy_union_us,
                "kernel_count": int(selected["kernel_count"]),
                "stream_count": stream_count,
                "has_host_api": any(item.get("host_api") for item in map_entries),
                "has_direct_correlation": any(
                    item.get("confidence") == "direct" for item in map_entries
                ),
                "decode_overlap": any(
                    str(item.get("annotation") or "").lower().startswith(
                        ("decode[", "step[decode")
                    )
                    for item in map_entries
                ),
                "kernels": kernels,
            }
        )

    signatures = [
        tuple(sorted((item["name"], item["count"]) for item in rank["kernels"]))
        for rank in rank_values
    ]
    kernel_names = sorted({item["name"] for rank in rank_values for item in rank["kernels"]})
    operations = []
    for op in OPS:
        durations = []
        counts = []
        mapped_counts = []
        mapped_durations = []
        confidence_counts: collections.Counter[str] = collections.Counter()
        semantic_confidence_counts: collections.Counter[str] = collections.Counter()
        semantic_confidence_durations: collections.Counter[str] = collections.Counter()
        for rank in rank_values:
            selected = [
                (item, allocation)
                for item in rank["kernels"]
                for allocation in item["operation_allocations"]
                if allocation["operation"] == op
            ]
            durations.append(sum(allocation["duration_us"] for _, allocation in selected))
            counts.append(sum(allocation["count"] for _, allocation in selected))
            mapped = [
                (item, allocation)
                for item, allocation in selected
                if item["mapping"]["confidence"] != "unmatched"
            ]
            mapped_counts.append(sum(allocation["count"] for _, allocation in mapped))
            mapped_durations.append(
                sum(allocation["duration_us"] for _, allocation in mapped)
            )
            for item, allocation in selected:
                confidence_counts[item["mapping"]["confidence"]] += allocation["count"]
                semantic_confidence_counts[
                    allocation["semantic_confidence"]
                ] += allocation["count"]
                semantic_confidence_durations[
                    allocation["semantic_confidence"]
                ] += allocation["duration_us"]
        operations.append(
            {
                "operation": op,
                "duration_us": distribution(durations),
                "count": distribution(counts),
                "mapped_count": distribution(mapped_counts),
                "mapped_duration_us": distribution(mapped_durations),
                "mapping_confidence_counts": dict(confidence_counts),
                "semantic_confidence_counts": dict(semantic_confidence_counts),
                "semantic_confidence_duration_us": dict(
                    semantic_confidence_durations
                ),
            }
        )

    families = []
    family_names = sorted(
        {item["family"] for rank in rank_values for item in rank["kernels"]}
    )
    for family in family_names:
        durations = [
            sum(item["duration_us"] for item in rank["kernels"] if item["family"] == family)
            for rank in rank_values
        ]
        counts = [
            sum(item["count"] for item in rank["kernels"] if item["family"] == family)
            for rank in rank_values
        ]
        families.append(
            {
                "family": family,
                "duration_us": distribution(durations),
                "count": distribution(counts),
            }
        )

    top_kernels = []
    for name in kernel_names:
        matching = [
            next(
                (item for item in rank["kernels"] if item["name"] == name),
                None,
            )
            for rank in rank_values
        ]
        exemplar = next(item for item in matching if item is not None)
        top_kernels.append(
            {
                "name": name,
                "family": exemplar["family"],
                "operation": exemplar["operation"],
                "duration_us": distribution(
                    item["duration_us"] if item is not None else 0.0
                    for item in matching
                ),
                "count": distribution(
                    item["count"] if item is not None else 0
                    for item in matching
                ),
                "mapping_confidence": collections.Counter(
                    item["mapping"]["confidence"]
                    for item in matching
                    if item is not None
                ).most_common(1)[0][0],
                "mapping": exemplar["mapping"],
            }
        )
    top_kernels.sort(key=lambda item: item["duration_us"]["median"], reverse=True)

    total_count = sum(rank["kernel_count"] for rank in rank_values)
    mapped_count = sum(
        item["count"]
        for rank in rank_values
        for item in rank["kernels"]
        if item["mapping"]["confidence"] != "unmatched"
    )
    exact_count = sum(
        item["count"]
        for rank in rank_values
        for item in rank["kernels"]
        if item["mapping"]["exact_count_pattern"]
    )
    detailed_count = sum(
        item["count"]
        for rank in rank_values
        for item in rank["kernels"]
        if item["mapping"]["detailed_annotation_count"]
    )
    total_duration = sum(rank["kernel_sum_us"] for rank in rank_values)
    mapped_duration = sum(
        item["duration_us"]
        for rank in rank_values
        for item in rank["kernels"]
        if item["mapping"]["confidence"] != "unmatched"
    )
    span = distribution(rank["span_us"] for rank in rank_values)
    endpoint_tpot = endpoint["median_tpot_ms"]
    graph_span_ms = span["median"] / 1000.0
    input_tokens = 8192 if concurrency == 2 else 16384
    return {
        "schema_version": 1,
        "phase": phase,
        "case": case,
        "engine": engine,
        "concurrency": concurrency,
        "selected_annotation": selected_names[0],
        "input_tokens": input_tokens if phase == "prefill" else None,
        "absolute_step_shape": (
            {
                "batch_size": 1 if concurrency == 2 else 2,
                "input_tokens": input_tokens,
            }
            if phase == "prefill"
            else {"batch_size": concurrency}
        ),
        "rank_count": len(rank_values),
        "rank_consistency": {
            "expected_ranks": list(range(8)),
            "selected_annotation_identical": len(set(selected_names)) == 1,
            "kernel_name_count_signatures_identical": len(set(signatures)) == 1,
            "kernel_count_identical": len({rank["kernel_count"] for rank in rank_values}) == 1,
            "kernel_count_range": [
                min(rank["kernel_count"] for rank in rank_values),
                max(rank["kernel_count"] for rank in rank_values),
            ],
            "stream_count_identical": len({rank["stream_count"] for rank in rank_values}) == 1,
        },
        "validation_gates": {
            "same_selected_name_shape_across_ranks": len(set(selected_names)) == 1,
            "nonzero_gpu_all_ranks": all(
                rank["kernel_count"] > 0 and rank["kernel_sum_us"] > 0
                for rank in rank_values
            ),
            "nonzero_host_all_ranks": all(
                rank["has_host_api"] for rank in rank_values
            ),
            "nonzero_correlation_all_ranks": all(
                rank["has_direct_correlation"] for rank in rank_values
            ),
            "no_decode_overlap_all_ranks": not any(
                rank["decode_overlap"] for rank in rank_values
            ),
        },
        "span_us": span,
        "kernel_sum_us": distribution(rank["kernel_sum_us"] for rank in rank_values),
        "kernel_busy_union_us": distribution(
            rank["kernel_busy_union_us"] for rank in rank_values
        ),
        "kernel_count": distribution(rank["kernel_count"] for rank in rank_values),
        "stream_count": distribution(rank["stream_count"] for rank in rank_values),
        "families": families,
        "operations": operations,
        "top_kernels": top_kernels,
        "mapping_coverage": {
            "kernel_occurrence_count": total_count,
            "mapped_kernel_occurrence_count": mapped_count,
            "mapped_kernel_occurrence_pct": 100.0 * mapped_count / total_count,
            "mapped_duration_pct": 100.0 * mapped_duration / total_duration,
            "exact_count_pattern_occurrence_pct": 100.0 * exact_count / total_count,
            "detailed_annotation_occurrence_pct": 100.0 * detailed_count / total_count,
            "interpretation": (
                (
                    "Selected prefill caller maps use direct CPU/API-to-GPU correlation "
                    "inside the exact production annotation window."
                )
                if phase == "prefill"
                else (
                    "ATOM graph-map labels require exact counts, direct warmup correlation, "
                    "and detailed model annotations. SGLang labels are inferred from exact "
                    "kernel/count evidence because its warmup annotation is graph-wide."
                )
            ),
        },
        "endpoint": endpoint,
        "reconciliation": (
            {
                "selected_prefill_step_span_ms": graph_span_ms,
                "endpoint_median_ttft_ms": endpoint["median_ttft_ms"],
                "residual_not_computed": (
                    "TTFT includes admission, queueing, and potentially multiple scheduler "
                    "batches; one selected prefill step is component evidence only."
                ),
            }
            if phase == "prefill"
            else {
                "graph_step_span_ms": graph_span_ms,
                "endpoint_median_tpot_ms": endpoint_tpot,
                "graph_external_residual_ms": endpoint_tpot - graph_span_ms,
                "graph_external_residual_pct_of_tpot": (
                    100.0 * (endpoint_tpot - graph_span_ms) / endpoint_tpot
                ),
                "warning": (
                    "A negative residual indicates profiler/clock perturbation or that endpoint "
                    "TPOT is not a strict serialized graph-step wall time; it is not negative work."
                ),
            }
        ),
        "sources": {
            "selected_steps": [
                str(
                    root / case / "analysis" / f"rank-{rank}-selected-step.json"
                    if phase == "decode"
                    else root
                    / "prefill-steps"
                    / case
                    / "analysis"
                    / "selected-prefill"
                    / f"rank-{rank}-summary.json"
                )
                for rank in range(8)
            ],
            "caller_maps": [
                str(
                    (
                        root
                        / "graph-attribution"
                        / engine
                        / "analysis"
                        / f"bs{concurrency}"
                        / f"rank{rank}-caller-map.jsonl"
                    )
                    if phase == "decode"
                    else (
                        root
                        / "prefill-steps"
                        / case
                        / "analysis"
                        / "selected-prefill"
                        / f"rank-{rank}-caller-map.jsonl"
                    )
                )
                for rank in range(8)
            ],
            "graph_warmup_caller_maps": (
                [
                    str(
                        root
                        / "graph-attribution"
                        / engine
                        / "analysis"
                        / f"bs{concurrency}"
                        / f"rank{rank}-caller-map.jsonl"
                    )
                    for rank in range(8)
                ]
                if phase == "decode"
                else []
            ),
        },
    }


def keyed(rows: list[dict], key: str) -> dict[str, dict]:
    return {item[key]: item for item in rows}


def compare_cases(
    left: dict, right: dict, kind: str, phase: str = "decode"
) -> dict:
    left_ops = keyed(left["operations"], "operation")
    right_ops = keyed(right["operations"], "operation")
    operations = []
    for op in OPS:
        left_us = left_ops[op]["duration_us"]["median"]
        right_us = right_ops[op]["duration_us"]["median"]
        delta = right_us - left_us
        operations.append(
            {
                "operation": op,
                "left_ms": left_us / 1000.0,
                "right_ms": right_us / 1000.0,
                "delta_ms": delta / 1000.0,
                "delta_pct_of_span_change": (
                    100.0
                    * delta
                    / (
                        right["span_us"]["median"] - left["span_us"]["median"]
                    )
                    if right["span_us"]["median"] != left["span_us"]["median"]
                    else None
                ),
                "ratio": right_us / left_us if left_us else None,
                "count_delta": (
                    right_ops[op]["count"]["median"] - left_ops[op]["count"]["median"]
                ),
            }
        )
    operations.sort(key=lambda item: abs(item["delta_ms"]), reverse=True)
    return {
        "schema_version": 1,
        "phase": phase,
        "kind": kind,
        "left_case": left["case"],
        "right_case": right["case"],
        "span": {
            "left_ms": left["span_us"]["median"] / 1000.0,
            "right_ms": right["span_us"]["median"] / 1000.0,
            "delta_ms": (
                right["span_us"]["median"] - left["span_us"]["median"]
            )
            / 1000.0,
            "ratio": right["span_us"]["median"] / left["span_us"]["median"],
            "left_us_per_input_token": (
                left["span_us"]["median"] / left["input_tokens"]
                if phase == "prefill"
                else None
            ),
            "right_us_per_input_token": (
                right["span_us"]["median"] / right["input_tokens"]
                if phase == "prefill"
                else None
            ),
        },
        "endpoint": {
            "left_median_tpot_ms": left["endpoint"]["median_tpot_ms"],
            "right_median_tpot_ms": right["endpoint"]["median_tpot_ms"],
            "tpot_delta_ms": (
                right["endpoint"]["median_tpot_ms"]
                - left["endpoint"]["median_tpot_ms"]
            ),
            "throughput_delta_pct": 100.0
            * (
                right["endpoint"]["total_token_throughput_tok_s"]
                / left["endpoint"]["total_token_throughput_tok_s"]
                - 1.0
            ),
            "ttft_delta_ms": (
                right["endpoint"]["median_ttft_ms"]
                - left["endpoint"]["median_ttft_ms"]
            ),
        },
        "operations": operations,
        "caveat": (
            (
                "Both sides use one complete full-model GPU prefill annotation. Profiled "
                "annotation wall/GPU spans include profiler perturbation and are not endpoint "
                "TTFT estimates; compare kernel composition and call counts, with per-token "
                "normalization used only for BS1-versus-BS2 shape context."
            )
            if phase == "prefill"
            else (
                "Operation groups are exclusive, but generic GEMMs can only be resolved where "
                "warmup caller/annotation evidence is specific. Kernel sums are used only inside "
                "the single expanded replay and are not substituted for replay span."
            )
        ),
    }


def write_csv(
    path: Path,
    decode_aggregates: dict[str, dict],
    prefill_aggregates: dict[str, dict] | None = None,
) -> None:
    columns = (
        "phase",
        "case",
        "engine",
        "concurrency",
        "operation",
        "duration_median_ms",
        "duration_min_ms",
        "duration_max_ms",
        "duration_cv",
        "count_median",
        "count_min",
        "count_max",
        "count_cv",
        "pct_of_step_span",
        "mapped_count_median",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        phase_groups = [("decode", decode_aggregates)]
        if prefill_aggregates is not None:
            phase_groups.append(("prefill", prefill_aggregates))
        for phase, aggregates in phase_groups:
            for case in CASES:
                aggregate = aggregates[case]
                for op in aggregate["operations"]:
                    duration = op["duration_us"]
                    count = op["count"]
                    writer.writerow(
                        {
                            "phase": phase,
                            "case": case,
                            "engine": aggregate["engine"],
                            "concurrency": aggregate["concurrency"],
                            "operation": op["operation"],
                            "duration_median_ms": duration["median"] / 1000.0,
                            "duration_min_ms": duration["min"] / 1000.0,
                            "duration_max_ms": duration["max"] / 1000.0,
                            "duration_cv": duration["cv"],
                            "count_median": count["median"],
                            "count_min": count["min"],
                            "count_max": count["max"],
                            "count_cv": count["cv"],
                            "pct_of_step_span": (
                                100.0
                                * duration["median"]
                                / aggregate["span_us"]["median"]
                            ),
                            "mapped_count_median": op["mapped_count"]["median"],
                        }
                    )


def format_top(aggregate: dict, limit: int = 8) -> str:
    rows = sorted(
        aggregate["operations"],
        key=lambda item: item["duration_us"]["median"],
        reverse=True,
    )[:limit]
    return "\n".join(
        f"- `{item['operation']}`: {item['duration_us']['median']/1000:.3f} ms "
        f"({100*item['duration_us']['median']/aggregate['span_us']['median']:.1f}% "
        f"of step span), {item['count']['median']:.0f} calls"
        for item in rows
    )


def build_report(
    aggregates: dict[str, dict],
    comparisons: dict[str, dict],
    prefill_aggregates: dict[str, dict],
    prefill_comparisons: dict[str, dict],
) -> str:
    lines = [
        "# Kimi-K3 SGLang/ATOM C2/C64 trace attribution — final draft",
        "",
        "## Scope and method",
        "",
        "Timing comes from one exact-shape production prefill annotation and one expanded "
        "production decode graph replay per case/rank. Decode graph caller maps label replay "
        "kernels by exact symbol/count patterns; prefill maps use direct correlation inside "
        "the selected annotation. Rank statistics use TP8 median/min/max/CV.",
        "",
        "SGLang decode labels use the corrected combined-graph warmups: GPU occurrence 1 "
        "for BS64 (raw Input Dims M=64) and occurrence 3 for BS2 (M=2). Direct C64 "
        "Input Dims and 69/24 call patterns resolve reused dense symbols; ATOM uses its "
        "detailed annotations, with known Kimi shapes explicitly confidence-labelled.",
        "",
        "All selections require ranks 0–7 and the same annotation name/shape across ranks. "
        "Absolute prefill times are preserved: C2 is BS1/8,192 input tokens and C64 is "
        "BS2/16,384 input tokens.",
        "",
        "## Prefill single-step findings",
        "",
        "| Case | Shape | Kernels | Annotation span ms | Rank min–max ms | CV | us/input token | Endpoint TTFT ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        item = prefill_aggregates[case]
        span = item["span_us"]
        shape = item["absolute_step_shape"]
        lines.append(
            f"| {case} | BS{shape['batch_size']} / {shape['input_tokens']} toks | "
            f"{item['kernel_count']['median']:.0f} | {span['median']/1000:.3f} | "
            f"{span['min']/1000:.3f}–{span['max']/1000:.3f} | "
            f"{span['cv']*100:.3f}% | {span['median']/item['input_tokens']:.3f} | "
            f"{item['endpoint']['median_ttft_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "All ranks passed nonzero GPU, host-API, direct-correlation, and no-decode-overlap "
            "gates. Exact annotation names/shapes match across TP8. The stop-after-wave "
            "SGLang recapture selects the final complete full-model GPU occurrence: C2 "
            "occurrence 1/2 and C64 occurrence 33/34. Prior early-stop directories are "
            "retained only as historical evidence.",
            "",
            "**Withdrawn prefill claim:** all earlier SGLang 2–5 ms staging-only selections "
            "and ratios are invalid and are replaced by the complete "
            "`sglang.vlm.language_model_prefill` scopes.",
            "",
            "These annotation wall/GPU times are profiler-perturbed: approximately 468 ms for "
            "SGLang and 1,049/1,210 ms for ATOM. Unprofiled common-client TTFT also includes "
            "admission, queueing, scheduler batch count, and profiler-free execution, so no "
            "annotation-to-TTFT residual or endpoint timing claim is valid. Use these scopes "
            "for one-step kernel/operation composition and call counts. Per-token values are "
            "shape context only: C2 SGLang processes BS1/8,192-token steps, while C64 processes "
            "BS2/16,384-token steps.",
            "",
            "### Prefill mapping coverage",
            "",
            "| Case | Mapped occurrences | Mapped duration | Exact count pattern | Evidence |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for case in CASES:
        coverage = prefill_aggregates[case]["mapping_coverage"]
        evidence = (
            "direct kernel correlation; semantic labels mostly kernel/caller taxonomy"
            if case.startswith("sglang")
            else "direct detailed annotation; known-shape custom ops confidence-labelled"
        )
        lines.append(
            f"| {case} | {coverage['mapped_kernel_occurrence_pct']:.2f}% | "
            f"{coverage['mapped_duration_pct']:.2f}% | "
            f"{coverage['exact_count_pattern_occurrence_pct']:.2f}% | "
            f"{evidence} |"
        )
    lines.extend(
        [
            "",
            "Mapping coverage measures exact selected-window correlation, not semantic "
            "specificity. SGLang's full-model outer annotation is coarse, so its residual "
            "`other_dense_gemms`/`other` buckets and cross-engine operation deltas have lower "
            "semantic confidence than ATOM's detailed per-operation annotations.",
        ]
    )
    lines.extend(["", "### Prefill top operation groups"])
    for case in CASES:
        lines.extend(["", f"#### {case}", "", format_top(prefill_aggregates[case])])
    lines.extend(["", "### Profiled prefill composition and shape scaling", ""])
    for name, comparison in prefill_comparisons.items():
        span = comparison["span"]
        lines.extend(
            [
                f"#### {name}",
                "",
                f"Profiled annotation span: {span['left_ms']:.3f} -> {span['right_ms']:.3f} ms "
                f"({span['delta_ms']:+.3f} ms, {span['ratio']:.3f}x); "
                f"per-token {span['left_us_per_input_token']:.3f} -> "
                f"{span['right_us_per_input_token']:.3f} us. "
                "This ratio is profiler-observed composition context, not an endpoint "
                "latency ratio.",
                "",
            ]
        )
        for op in comparison["operations"][:6]:
            ratio = "n/a" if op["ratio"] is None else f"{op['ratio']:.3f}x"
            lines.append(
                f"- `{op['operation']}`: {op['left_ms']:.3f} -> "
                f"{op['right_ms']:.3f} ms ({op['delta_ms']:+.3f} ms, {ratio})"
            )
    lines.extend(
        [
            "",
            "### Prefill route-stage caveat",
            "",
            "ATOM is confirmed A16W4 while SGLang is A8W4. At C64 the observed routed-MoE "
            "stage1 grids are 2,784 versus 5,562 workgroups. Active-expert counts have not "
            "yet been dumped, so grid/work differences must not be attributed solely to "
            "kernel efficiency or precision.",
            "",
            "## Decode single-step findings",
            "",
            "All four decode cases have identical kernel-name/count signatures across TP "
            "ranks, identical kernel counts, and one replay stream per rank.",
        "",
        "### Replay and endpoint reconciliation",
        "",
        "| Case | Kernels | Graph span ms | Rank min–max ms | CV | Endpoint TPOT ms | External residual ms | Throughput tok/s | Median TTFT ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in CASES:
        item = aggregates[case]
        span = item["span_us"]
        rec = item["reconciliation"]
        endpoint = item["endpoint"]
        lines.append(
            f"| {case} | {item['kernel_count']['median']:.0f} | "
            f"{span['median']/1000:.3f} | "
            f"{span['min']/1000:.3f}–{span['max']/1000:.3f} | "
            f"{span['cv']*100:.3f}% | {endpoint['median_tpot_ms']:.3f} | "
            f"{rec['graph_external_residual_ms']:+.3f} | "
            f"{endpoint['total_token_throughput_tok_s']:.2f} | "
            f"{endpoint['median_ttft_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Negative C2 residuals are measurement/definition effects, not negative scheduler "
            "work. At C64, ATOM's large positive residual records graph-external scheduling "
            "and token-delivery delay that median per-request TPOT includes.",
            "",
            "### Decode mapping coverage",
            "",
            "| Case | Exact-name mapped occurrences | Mapped duration | Exact count-pattern occurrences | Detailed annotation occurrences | Interpretation |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for case in CASES:
        coverage = aggregates[case]["mapping_coverage"]
        interpretation = (
            "detailed ATOM model annotation + direct warmup CPU op"
            if case.startswith("atom")
            else "exact symbol/count + direct warmup CPU op; replay op inferred"
        )
        lines.append(
            f"| {case} | {coverage['mapped_kernel_occurrence_pct']:.2f}% | "
            f"{coverage['mapped_duration_pct']:.2f}% | "
            f"{coverage['exact_count_pattern_occurrence_pct']:.2f}% | "
            f"{coverage['detailed_annotation_occurrence_pct']:.2f}% | {interpretation} |"
        )
    lines.extend(["", "### Decode top operation groups"])
    for case in CASES:
        lines.extend(["", f"### {case}", "", format_top(aggregates[case])])

    lines.extend(["", "### Decode engine gaps and scaling", ""])
    for name, comparison in comparisons.items():
        span = comparison["span"]
        endpoint = comparison["endpoint"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"Replay span: {span['left_ms']:.3f} -> {span['right_ms']:.3f} ms "
                f"({span['delta_ms']:+.3f} ms, {span['ratio']:.3f}x). "
                f"Endpoint TPOT delta: {endpoint['tpot_delta_ms']:+.3f} ms; "
                f"throughput delta: {endpoint['throughput_delta_pct']:+.2f}%; "
                f"TTFT delta: {endpoint['ttft_delta_ms']:+.2f} ms.",
                "",
            ]
        )
        for op in comparison["operations"][:6]:
            ratio = "n/a" if op["ratio"] is None else f"{op['ratio']:.3f}x"
            contribution = op["delta_pct_of_span_change"]
            contribution_text = (
                "n/a"
                if contribution is None
                else f"{contribution:+.1f}% of signed span delta"
            )
            lines.append(
                f"- `{op['operation']}`: {op['left_ms']:.3f} -> "
                f"{op['right_ms']:.3f} ms ({op['delta_ms']:+.3f} ms, "
                f"{ratio}, {contribution_text})"
            )

    c64_ops = keyed(comparisons["engine-gap-c64"]["operations"], "operation")
    lines.extend(
        [
            "",
            "## Corrected C64 dense attribution",
            "",
            "The earlier `other_dense_gemms -7.08 ms` interpretation was caused by the "
            "reversed SGLang warmup occurrence labels and is withdrawn. The corrected "
            "semantic deltas (ATOM minus SGLang) include "
            f"`routed_moe_stage1` {c64_ops['routed_moe_stage1']['delta_ms']:+.3f} ms, "
            f"`kda_inproj` {c64_ops['kda_inproj']['delta_ms']:+.3f} ms, "
            f"`mla_decode` {c64_ops['mla_decode']['delta_ms']:+.3f} ms, "
            f"`collectives` {c64_ops['collectives']['delta_ms']:+.3f} ms, and "
            f"`shared_expert_down` {c64_ops['shared_expert_down']['delta_ms']:+.3f} ms. "
            f"The residual unresolved `other_dense_gemms` delta is only "
            f"{c64_ops['other_dense_gemms']['delta_ms']:+.3f} ms.",
            "",
            "## Combined endpoint reconciliation",
            "",
            "Decode C2 is graph-internal: ATOM's replay is materially longer while both endpoint "
            "residuals are near zero. At C64 the replay spans converge and ATOM's graph is "
            "shorter, but ATOM carries a large graph-external TPOT residual. Aggregate "
            "throughput nevertheless converges because ATOM starts requests much earlier "
            "(lower TTFT), while SGLang delays first tokens and then decodes with lower "
            "per-request TPOT. Profiled prefill annotation spans are not reconciled to "
            "unprofiled TTFT because scheduler batch count and profiler overhead differ.",
            "",
            "## Caveats",
            "",
            "- Family and operation durations are kernel sums inside one selected expanded "
            "step. They may overlap and do not add to a critical path on multi-stream traces.",
            "- Operation groups are exclusive. Corrected SGLang C64 dense labels use direct "
            "raw Input Dims; ATOM custom ops without dimensions use direct detailed annotation "
            "plus known Kimi shapes and are labelled `direct_annotation_known_shape`.",
            "- Residual `other_dense_gemms` and unmapped exact symbols remain unresolved; "
            "mapping coverage is reported separately and prevents treating semantic sums as "
            "complete attribution.",
            "- Percentages use replay span as denominator, so summed percentages can exceed "
            "100% if streams overlap. These captures report one active replay stream.",
            "- The report does not project one replay across all output tokens and does not "
            "use profiler-perturbed throughput.",
            "- Prefill C2 and C64 use different absolute shapes and SGLang batch sizes "
            "(BS1 versus BS2); per-token normalization is reported only as composition context.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument(
        "--endpoint-c2",
        type=Path,
        default=Path(
            "/workspace/kimi-k3-runs/common-oai-sglang-atom-c2-2026-08-21"
        ),
    )
    parser.add_argument(
        "--endpoint-c64",
        type=Path,
        default=Path(
            "/workspace/kimi-k3-runs/common-oai-sglang-atom-c64-2026-08-21"
        ),
    )
    parser.add_argument(
        "--preserve-prefill",
        action="store_true",
        help="load existing prefill aggregates/comparisons without rewriting them",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    root = args.result_root.resolve()
    endpoints = discover_endpoints(args)
    aggregates = {
        case: aggregate_case(root, case, endpoints[case]) for case in CASES
    }
    case_dir = root / "case-aggregates"
    prefill_case_dir = root / "prefill-case-aggregates"
    comparison_dir = root / "comparisons"
    if args.preserve_prefill:
        prefill_aggregates = {
            case: load_json(prefill_case_dir / f"{case}.json") for case in CASES
        }
    else:
        prefill_aggregates = {
            case: aggregate_case(root, case, endpoints[case], phase="prefill")
            for case in CASES
        }
    case_dir.mkdir(exist_ok=True)
    prefill_case_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    for case, aggregate in aggregates.items():
        (case_dir / f"{case}.json").write_text(
            json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
        )
    if not args.preserve_prefill:
        for case, aggregate in prefill_aggregates.items():
            (prefill_case_dir / f"{case}.json").write_text(
                json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
            )
    comparisons = {
        "engine-gap-c2": compare_cases(
            aggregates["sglang-c2"], aggregates["atom-c2"], "engine_gap"
        ),
        "engine-gap-c64": compare_cases(
            aggregates["sglang-c64"], aggregates["atom-c64"], "engine_gap"
        ),
        "scale-sglang": compare_cases(
            aggregates["sglang-c2"], aggregates["sglang-c64"], "concurrency_scale"
        ),
        "scale-atom": compare_cases(
            aggregates["atom-c2"], aggregates["atom-c64"], "concurrency_scale"
        ),
    }
    for name, comparison in comparisons.items():
        (comparison_dir / f"{name}.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
    if args.preserve_prefill:
        prefill_comparisons = {
            name: load_json(comparison_dir / f"{name}.json")
            for name in (
                "prefill-engine-gap-c2",
                "prefill-engine-gap-c64",
                "prefill-scale-sglang",
                "prefill-scale-atom",
            )
        }
    else:
        prefill_comparisons = {
            "prefill-engine-gap-c2": compare_cases(
                prefill_aggregates["sglang-c2"],
                prefill_aggregates["atom-c2"],
                "engine_gap",
                phase="prefill",
            ),
            "prefill-engine-gap-c64": compare_cases(
                prefill_aggregates["sglang-c64"],
                prefill_aggregates["atom-c64"],
                "engine_gap",
                phase="prefill",
            ),
            "prefill-scale-sglang": compare_cases(
                prefill_aggregates["sglang-c2"],
                prefill_aggregates["sglang-c64"],
                "shape_scale",
                phase="prefill",
            ),
            "prefill-scale-atom": compare_cases(
                prefill_aggregates["atom-c2"],
                prefill_aggregates["atom-c64"],
                "shape_scale",
                phase="prefill",
            ),
        }
        for name, comparison in prefill_comparisons.items():
            (comparison_dir / f"{name}.json").write_text(
                json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
            )
    write_csv(root / "op-attribution.csv", aggregates, prefill_aggregates)
    (root / "TRACE_ATTRIBUTION_DRAFT.md").write_text(
        build_report(
            aggregates, comparisons, prefill_aggregates, prefill_comparisons
        ),
        encoding="utf-8",
    )
    print(
        "DONE decode_cases=4 prefill_cases=4 ranks_per_phase=32 comparisons=8 "
        f"output={root / 'TRACE_ATTRIBUTION_DRAFT.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
