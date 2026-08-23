#!/usr/bin/env python3
"""Graph-time the complete Kimi-K3 MLA shared-input PTPC boundary.

Weight quantization and preshuffling are setup-only. Every runtime conversion,
the output sigmoid/multiply, and (for norm-inclusive cases) RMSNorm are timed.
The candidate quantizes one activation exactly once and reuses the resulting
``(fp8, per-token scale)`` for both QKV-A and g_proj.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from microbench_common import benchmark_graph_replay, capture_graph, rel_l2

SCHEMA_VERSION = "kimi-k3-mla-shared-ptpc-v1"
QKV_N = 2112
GATE_N = 1536
HIDDEN = 7168
MLA_LAYERS = 24


def tensor_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().float().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def cosine(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            candidate.float().reshape(1, -1), reference.float().reshape(1, -1)
        ).item()
    )


def summarize_us(samples_ms: list[float]) -> dict[str, Any]:
    values = [1000.0 * value for value in samples_ms]
    ordered = sorted(values)
    return {
        "samples": len(values),
        "min_us": ordered[0],
        "p50_us": statistics.median(ordered),
        "p90_us": ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)],
        "max_us": ordered[-1],
        "mean_us": statistics.fmean(ordered),
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


class BoundaryBench:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        for root in (args.aiter_root, args.sglang_root / "python"):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        self.aiter = importlib.import_module("aiter")
        self.tgemm = importlib.import_module("aiter.tuned_gemm")
        self.quant_mod = importlib.import_module("aiter.ops.quant")
        self.shuffle_mod = importlib.import_module("aiter.ops.shuffle")
        self.chip_mod = importlib.import_module("aiter.jit.utils.chip_info")
        self.output_gate = importlib.import_module(
            "sglang.kernels.ops.kimi_k3.mla_output_gate"
        )
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("a HIP PyTorch device is required")
        self.gfx = self.chip_mod.get_gfx_runtime()
        if self.gfx != "gfx950":
            raise RuntimeError(f"exact benchmark requires gfx950, got {self.gfx}")
        self.quant = self.quant_mod.get_hip_quant(self.aiter.QuantType.per_Token)

    def prepare_weight(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight_q, weight_scale = self.quant(
            weight, quant_dtype=self.aiter.dtypes.fp8
        )
        weight_q = self.shuffle_mod.shuffle_weight(
            weight_q.contiguous(), layout=(16, 16)
        )
        return weight_q, weight_scale

    def bf16_gemm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return self.tgemm.tgemm.mm(x, weight, None, otype=torch.bfloat16)

    def fp8_gemm(
        self,
        xq: torch.Tensor,
        weight_q: torch.Tensor,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return self.aiter.gemm_a8w8_bpreshuffle(
            xq,
            weight_q,
            x_scale,
            weight_scale,
            None,
            torch.bfloat16,
        )

    def fused_rms_quant(
        self, x: torch.Tensor, norm_weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x, dtype=self.aiter.dtypes.fp8)
        scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        self.aiter.rmsnorm_quant(
            out, x, scale, norm_weight, self.args.eps, 0, False
        )
        return out, scale

    def time(self, run: Callable[[], Any]) -> dict[str, Any]:
        graph = capture_graph(run, warmup=self.args.warmup)
        samples: list[float] = []
        for _ in range(self.args.repeats):
            samples.extend(
                benchmark_graph_replay(
                    graph,
                    warmup=self.args.warmup,
                    iterations=self.args.iterations,
                )
            )
        return summarize_us(samples)

    def freshness(
        self,
        run: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        x: torch.Tensor,
        attention: torch.Tensor,
    ) -> dict[str, Any]:
        holder: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        def captured_run():
            holder["outputs"] = run()
            return holder["outputs"]

        graph = capture_graph(captured_run, warmup=self.args.warmup)
        graph.replay()
        torch.cuda.synchronize()
        first = tuple(value.clone() for value in holder["outputs"])
        x_original = x.clone()
        attention_original = attention.clone()
        x.copy_(x_original.flip(0) * 0.9375 + 0.03125)
        attention.copy_(attention_original.flip(0) * 1.0625 - 0.015625)
        graph.replay()
        torch.cuda.synchronize()
        second = tuple(value.clone() for value in holder["outputs"])
        x.copy_(x_original)
        attention.copy_(attention_original)
        changed = [
            bool(torch.isfinite(value).all().item())
            and not torch.equal(value, original)
            for value, original in zip(second, first)
        ]
        if not all(changed):
            raise RuntimeError(f"input-change graph replay failed: {changed}")
        return {
            "passed": True,
            "outputs_changed": changed,
            "first_hashes": [digest(value) for value in first],
            "second_hashes": [digest(value) for value in second],
        }

    def execute(self, m: int) -> dict[str, Any]:
        seed = self.args.seed + m
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x = torch.randn((m, HIDDEN), device="cuda", dtype=torch.bfloat16)
        attention = torch.randn(
            (m, GATE_N), device="cuda", dtype=torch.bfloat16
        )
        norm_weight = (
            1.0
            + 0.05
            * torch.randn((HIDDEN,), device="cuda", dtype=torch.bfloat16)
        ).contiguous()
        qkv_weight = (
            torch.randn(
                (QKV_N, HIDDEN), device="cuda", dtype=torch.bfloat16
            )
            / math.sqrt(HIDDEN)
        ).contiguous()
        gate_weight = (
            torch.randn(
                (GATE_N, HIDDEN), device="cuda", dtype=torch.bfloat16
            )
            / math.sqrt(HIDDEN)
        ).contiguous()
        qkv_q, qkv_scale = self.prepare_weight(qkv_weight)
        gate_q, gate_scale = self.prepare_weight(gate_weight)
        xq_static, x_scale_static = self.quant(
            x, quant_dtype=self.aiter.dtypes.fp8
        )
        qkv_bf_static = self.bf16_gemm(x, qkv_weight)
        gate_bf_static = self.bf16_gemm(x, gate_weight)

        def output_gate(attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
            if not self.output_gate.covered(attn, gate):
                raise RuntimeError("SGLang fused output gate rejected the tensors")
            return self.output_gate.kimi_k3_mla_output_gate(attn, gate)

        def baseline_total():
            qkv = self.bf16_gemm(x, qkv_weight)
            gate = self.bf16_gemm(x, gate_weight)
            return qkv, output_gate(attention, gate)

        def baseline_gate_chain():
            gate = self.bf16_gemm(x, gate_weight)
            return output_gate(attention, gate)

        def candidate_gate_chain_prequantized():
            gate = self.fp8_gemm(
                xq_static, gate_q, x_scale_static, gate_scale
            )
            return output_gate(attention, gate)

        def candidate_total():
            xq, x_scale = self.quant(x, quant_dtype=self.aiter.dtypes.fp8)
            qkv = self.fp8_gemm(xq, qkv_q, x_scale, qkv_scale)
            gate = self.fp8_gemm(xq, gate_q, x_scale, gate_scale)
            return qkv, output_gate(attention, gate)

        def baseline_norm_total():
            normed = self.aiter.rmsnorm2d_fwd(x, norm_weight, self.args.eps)
            qkv = self.bf16_gemm(normed, qkv_weight)
            gate = self.bf16_gemm(normed, gate_weight)
            return qkv, output_gate(attention, gate)

        def candidate_norm_total():
            xq, x_scale = self.fused_rms_quant(x, norm_weight)
            qkv = self.fp8_gemm(xq, qkv_q, x_scale, qkv_scale)
            gate = self.fp8_gemm(xq, gate_q, x_scale, gate_scale)
            return qkv, output_gate(attention, gate)

        baseline = baseline_total()
        candidate = candidate_total()
        norm_baseline = baseline_norm_total()
        norm_candidate = candidate_norm_total()
        for label, outputs in (
            ("baseline", baseline),
            ("candidate", candidate),
            ("norm_baseline", norm_baseline),
            ("norm_candidate", norm_candidate),
        ):
            if not all(torch.isfinite(value).all().item() for value in outputs):
                raise RuntimeError(f"{label} produced NaN/Inf")

        timings = {
            "bf16_qkv": self.time(lambda: self.bf16_gemm(x, qkv_weight)),
            "bf16_gate_projection": self.time(
                lambda: self.bf16_gemm(x, gate_weight)
            ),
            "shared_ptpc_quant": self.time(
                lambda: self.quant(x, quant_dtype=self.aiter.dtypes.fp8)
            ),
            "fp8_qkv_prequantized": self.time(
                lambda: self.fp8_gemm(
                    xq_static, qkv_q, x_scale_static, qkv_scale
                )
            ),
            "fp8_gate_projection_prequantized": self.time(
                lambda: self.fp8_gemm(
                    xq_static, gate_q, x_scale_static, gate_scale
                )
            ),
            "fused_output_gate": self.time(
                lambda: output_gate(attention, gate_bf_static)
            ),
            "baseline_gate_chain": self.time(baseline_gate_chain),
            "candidate_gate_chain_prequantized": self.time(
                candidate_gate_chain_prequantized
            ),
            "baseline_total": self.time(baseline_total),
            "candidate_total": self.time(candidate_total),
            "current_rmsnorm": self.time(
                lambda: self.aiter.rmsnorm2d_fwd(
                    x, norm_weight, self.args.eps
                )
            ),
            "fused_rmsnorm_ptpc_quant": self.time(
                lambda: self.fused_rms_quant(x, norm_weight)
            ),
            "baseline_rmsnorm_total": self.time(baseline_norm_total),
            "candidate_fused_rmsnorm_total": self.time(candidate_norm_total),
        }
        timings["candidate_speedup"] = (
            timings["baseline_total"]["p50_us"]
            / timings["candidate_total"]["p50_us"]
        )
        timings["candidate_delta_us"] = (
            timings["candidate_total"]["p50_us"]
            - timings["baseline_total"]["p50_us"]
        )
        timings["candidate_fused_rmsnorm_speedup"] = (
            timings["baseline_rmsnorm_total"]["p50_us"]
            / timings["candidate_fused_rmsnorm_total"]["p50_us"]
        )
        timings["candidate_fused_rmsnorm_delta_us"] = (
            timings["candidate_fused_rmsnorm_total"]["p50_us"]
            - timings["baseline_rmsnorm_total"]["p50_us"]
        )

        numerical = {
            "qkv_rel_l2": rel_l2(candidate[0], baseline[0]),
            "qkv_cosine": cosine(candidate[0], baseline[0]),
            "gated_attention_rel_l2": rel_l2(candidate[1], baseline[1]),
            "gated_attention_cosine": cosine(candidate[1], baseline[1]),
            "fused_norm_qkv_rel_l2": rel_l2(
                norm_candidate[0], norm_baseline[0]
            ),
            "fused_norm_qkv_cosine": cosine(
                norm_candidate[0], norm_baseline[0]
            ),
            "fused_norm_gated_attention_rel_l2": rel_l2(
                norm_candidate[1], norm_baseline[1]
            ),
            "fused_norm_gated_attention_cosine": cosine(
                norm_candidate[1], norm_baseline[1]
            ),
        }
        if (
            max(
                numerical["qkv_rel_l2"],
                numerical["gated_attention_rel_l2"],
                numerical["fused_norm_qkv_rel_l2"],
                numerical["fused_norm_gated_attention_rel_l2"],
            )
            > self.args.max_rel_l2
            or min(
                numerical["qkv_cosine"],
                numerical["gated_attention_cosine"],
                numerical["fused_norm_qkv_cosine"],
                numerical["fused_norm_gated_attention_cosine"],
            )
            < self.args.min_cosine
        ):
            raise RuntimeError(f"numerical gate failed: {numerical}")

        return {
            "m": m,
            "timings": timings,
            "numerical": numerical,
            "input_change": {
                "baseline": self.freshness(baseline_total, x, attention),
                "candidate": self.freshness(candidate_total, x, attention),
                "baseline_rmsnorm": self.freshness(
                    baseline_norm_total, x, attention
                ),
                "candidate_fused_rmsnorm": self.freshness(
                    candidate_norm_total, x, attention
                ),
            },
        }


def storage() -> dict[str, Any]:
    qkv_bf16 = QKV_N * HIDDEN * 2
    gate_bf16 = GATE_N * HIDDEN * 2
    qkv_prepared = QKV_N * HIDDEN + QKV_N * 4
    gate_prepared = GATE_N * HIDDEN + GATE_N * 4
    incremental = qkv_prepared + gate_prepared
    retained = qkv_bf16 + gate_bf16
    return {
        "layers": MLA_LAYERS,
        "per_layer": {
            "qkv_bf16_bytes": qkv_bf16,
            "gate_bf16_bytes": gate_bf16,
            "qkv_fp8_scale_bytes": qkv_prepared,
            "gate_fp8_scale_bytes": gate_prepared,
            "incremental_prepared_bytes": incremental,
            "dual_total_bytes": retained + incremental,
        },
        "model": {
            "incremental_prepared_bytes": incremental * MLA_LAYERS,
            "incremental_prepared_gib": incremental * MLA_LAYERS / 2**30,
            "dual_total_bytes": (retained + incremental) * MLA_LAYERS,
            "dual_total_gib": (retained + incremental) * MLA_LAYERS / 2**30,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m-values", default="32,64")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--max-rel-l2", type=float, default=0.05)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument(
        "--aiter-root",
        type=Path,
        default=Path("/sgl-workspace/aiter-atom-current"),
    )
    parser.add_argument(
        "--sglang-root",
        type=Path,
        default=Path("/sgl-workspace/sglang-k3-triton37"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    m_values = [int(value) for value in args.m_values.split(",") if value]
    if not m_values or any(value <= 0 for value in m_values):
        raise ValueError("--m-values must contain positive integers")
    bench = BoundaryBench(args)
    result = {
        "schema": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "metadata": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_hip": torch.version.hip,
            "device": torch.cuda.get_device_name(),
            "gfx": bench.gfx,
            "aiter_module": str(Path(bench.aiter.__file__).resolve()),
            "aiter_revision": git_revision(args.aiter_root),
            "sglang_revision": git_revision(args.sglang_root),
            "weight_prep_timed": False,
            "runtime_conversions_timed": True,
            "output_gate_api": (
                "sglang.kernels.ops.kimi_k3.mla_output_gate."
                "kimi_k3_mla_output_gate"
            ),
            "fused_rmsnorm_quant_api": "aiter.rmsnorm_quant",
        },
        "storage": storage(),
        "cases": [],
    }
    for m in m_values:
        case = bench.execute(m)
        result["cases"].append(case)
        timings = case["timings"]
        print(
            f"M{m} baseline={timings['baseline_total']['p50_us']:.3f}us "
            f"candidate={timings['candidate_total']['p50_us']:.3f}us "
            f"speedup={timings['candidate_speedup']:.4f}x "
            f"rms_speedup={timings['candidate_fused_rmsnorm_speedup']:.4f}x"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
