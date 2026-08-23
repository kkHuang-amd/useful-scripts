#!/usr/bin/env python3
"""Production-faithful Kimi-K3 dense projection crossover microbenchmark.

The primary metric is HIP/CUDA graph replay of the complete per-call chain.
Weight quantization and preshuffling are setup work and are deliberately outside
the timed region. PTPC FP8 includes per-token activation quantization on every
replay; MXFP4 includes per-1x32 activation quantization on every replay.

The two SGLang-only merged shapes are emitted as ``context_only=true`` and are
never treated as equivalent to projections with a different N. A microbenchmark
result is not sufficient for production promotion: matched common-client
endpoint workloads and paired GSM8K/long-context checks are still required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCHEMA_VERSION = "kimi-k3-dense-crossover-v3"
STORAGE_SCHEMA_VERSION = "kimi-k3-dense-crossover-storage-v1"
DEFAULT_M = (2, 4, 8, 16, 32, 64)
DEFAULT_MODES = ("bf16", "ptpc_fp8", "mxfp4", "rmsnorm_mxfp4")
DEFAULT_AITER_ROOT = Path("/sgl-workspace/aiter-atom-current")
DEFAULT_SGLANG_ROOT = Path("/sgl-workspace/sglang-k3-triton37")
LATENT_CAPACITY_CALIBRATION = {
    "source": (
        "/workspace/claude-skills/kimi-k3/aiter-optimization-tracker/"
        "MOE_LATENT_SPLIT_2026-08-20.md"
    ),
    "projection": "latent_up",
    "mode": "mxfp4",
    "layer_count": 92,
    "measured_token_loss": 142_753,
    "baseline_tokens": 1_519_705,
    "calibrated_profile_tokens": 1_376_952,
}
RECOMMENDED_POLICY = (
    {
        "projection": "latent_up",
        "mode": "mxfp4",
        "activation": "all tested M buckets",
    },
    {
        "projection": "kda_inproj",
        "mode": "mxfp4",
        "activation": "M >= 32",
    },
    {
        "projection": "mla_qkv_a",
        "mode": "ptpc_fp8",
        "activation": "M >= 64",
    },
)


@dataclass(frozen=True)
class Projection:
    name: str
    n: int
    k: int
    layer_count: int
    layer_count_assumption: str
    context_only: bool = False


@dataclass(frozen=True)
class Case:
    projection: Projection
    m: int
    mode: str

    @property
    def case_id(self) -> str:
        return f"{self.projection.name}:m{self.m}:{self.mode}"


PROJECTIONS = (
    Projection(
        "latent_up",
        7168,
        3584,
        92,
        "one projection on each MoE layer; layer 0 is dense",
    ),
    Projection(
        "shared_down",
        7168,
        768,
        92,
        "one shared-expert down projection on each MoE layer",
    ),
    Projection(
        "kda_mla_output",
        7168,
        1536,
        93,
        (
            "69 KDA plus 24 MLA output projections; both are the same TP8-local "
            "[7168,1536] BF16 o_proj representation and use the same prepared formats"
        ),
    ),
    Projection(
        "mla_qkv_a",
        2112,
        7168,
        24,
        "one QKV-A representative projection on each MLA layer",
    ),
    Projection(
        "mla_gate",
        1536,
        7168,
        24,
        "one output-gate projection on each MLA layer",
    ),
    Projection(
        "merged_front",
        6016,
        7168,
        92,
        "one SGLang-only merged front on each MoE layer",
        context_only=True,
    ),
    Projection(
        "kda_inproj",
        6288,
        7168,
        69,
        "one SGLang-only merged input projection on each KDA layer",
        context_only=True,
    ),
)
PROJECTION_BY_NAME = {projection.name: projection for projection in PROJECTIONS}


class UnsupportedMode(RuntimeError):
    """A fail-closed mode/case skip with a user-readable reason."""


class CaseFailure(RuntimeError):
    """A measured case failure carrying fields that must survive reporting."""

    def __init__(self, reason: str, result_updates: dict[str, Any] | None = None):
        super().__init__(reason)
        self.result_updates = result_updates or {}


def parse_csv_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def generate_cases(
    *,
    shape_filter: set[str] | None = None,
    mode_filter: set[str] | None = None,
    m_values: Sequence[int] = DEFAULT_M,
) -> list[Case]:
    known_shapes = {projection.name for projection in PROJECTIONS}
    unknown_shapes = (shape_filter or set()) - known_shapes
    if unknown_shapes:
        raise ValueError(f"unknown shapes: {', '.join(sorted(unknown_shapes))}")
    unknown_modes = (mode_filter or set()) - set(DEFAULT_MODES)
    if unknown_modes:
        raise ValueError(f"unknown modes: {', '.join(sorted(unknown_modes))}")
    if any(m <= 0 for m in m_values):
        raise ValueError("all M values must be positive")

    cases: list[Case] = []
    for projection in PROJECTIONS:
        if shape_filter is not None and projection.name not in shape_filter:
            continue
        for m in m_values:
            for mode in DEFAULT_MODES:
                if mode_filter is not None and mode not in mode_filter:
                    continue
                if mode == "rmsnorm_mxfp4" and projection.name != "latent_up":
                    continue
                cases.append(Case(projection, int(m), mode))
    return cases


def nominal_prepared_weight_bytes(n: int, k: int, mode: str) -> int:
    if mode == "bf16":
        return n * k * 2
    if mode == "ptpc_fp8":
        return n * k + n * 4
    if mode in {"mxfp4", "rmsnorm_mxfp4"}:
        if k % 32:
            raise ValueError("MXFP4 K must be divisible by 32")
        return n * k // 2 + n * (k // 32)
    raise ValueError(f"unknown mode: {mode}")


def storage_record(
    n: int,
    k: int,
    mode: str,
    *,
    actual_prepared_bytes: int | None = None,
) -> dict[str, int]:
    bf16_bytes = n * k * 2
    nominal = nominal_prepared_weight_bytes(n, k, mode)
    prepared = nominal if actual_prepared_bytes is None else actual_prepared_bytes
    incremental = 0 if mode == "bf16" else prepared
    return {
        "bf16_weight_bytes": bf16_bytes,
        "prepared_weight_bytes": prepared,
        "nominal_prepared_weight_bytes": nominal,
        "incremental_dual_storage_bytes": incremental,
        "dual_storage_total_bytes": bf16_bytes + incremental,
    }


def base_result(case: Case) -> dict[str, Any]:
    projection = case.projection
    return {
        "case_id": case.case_id,
        "projection": projection.name,
        "m": case.m,
        "n": projection.n,
        "k": projection.k,
        "projection_layer_count": projection.layer_count,
        "projection_layer_count_assumption": projection.layer_count_assumption,
        "mode": case.mode,
        "context_only": projection.context_only,
        "comparison_scope": (
            "sglang_only_non_like_for_like_context"
            if projection.context_only
            else "representative_kimi_projection"
        ),
        "status": "pending",
        "skip_reason": None,
        "graph_ms": None,
        "eager_ms": None,
        "rel_l2": None,
        "cosine": None,
        "finite": None,
        "input_change_replay_passed": None,
        "input_change_output_delta_norm": None,
        "input_change_output_a_sha256": None,
        "input_change_output_b_sha256": None,
        "input_change_reference_rel_l2": None,
        "input_change_reference_cosine": None,
        "dispatch": {},
        "storage": storage_record(projection.n, projection.k, case.mode),
    }


def skipped_result(case: Case, reason: str) -> dict[str, Any]:
    row = base_result(case)
    row["status"] = "skipped"
    row["skip_reason"] = reason
    return row


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(
    rows: Sequence[dict[str, Any]],
    *,
    args: argparse.Namespace | None = None,
    runtime_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    configuration = {
        "seed": getattr(args, "seed", None),
        "warmup": getattr(args, "warmup", None),
        "iterations": getattr(args, "iterations", None),
        "eager": bool(getattr(args, "eager", False)),
        "graph_primary": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "kimi_k3_dense_projection_crossover",
        "promotion_requirements": [
            "matched common-client endpoint workloads",
            "paired GSM8K correctness",
            "paired long-context output/logprob correctness",
        ],
        "configuration": configuration,
        "runtime": runtime_metadata or {},
        "cases": list(rows),
    }


def validate_report(report: dict[str, Any]) -> None:
    required_top = {
        "schema_version",
        "created_at",
        "benchmark",
        "promotion_requirements",
        "configuration",
        "runtime",
        "cases",
    }
    missing = required_top - report.keys()
    if missing:
        raise ValueError(f"report missing keys: {sorted(missing)}")
    required_case = set(base_result(Case(PROJECTIONS[0], 2, "bf16")))
    for index, row in enumerate(report["cases"]):
        missing_case = required_case - row.keys()
        if missing_case:
            raise ValueError(f"case {index} missing keys: {sorted(missing_case)}")
        if row["status"] not in {"ok", "skipped", "failed"}:
            raise ValueError(f"case {index} has invalid status {row['status']!r}")
        if row["status"] == "skipped" and not row["skip_reason"]:
            raise ValueError(f"case {index} skipped without a reason")
        if row["status"] == "ok" and row["input_change_replay_passed"] is not True:
            raise ValueError(f"case {index} passed without the input-change replay gate")


def run_cases(
    cases: Iterable[Case],
    execute: Callable[[Case], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        try:
            row = execute(case)
        except UnsupportedMode as exc:
            row = skipped_result(case, str(exc))
        except CaseFailure as exc:
            row = base_result(case)
            row["status"] = "failed"
            row["skip_reason"] = str(exc)
            row.update(exc.result_updates)
        except Exception as exc:  # fail closed per case and retain the campaign
            row = base_result(case)
            row["status"] = "failed"
            row["skip_reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def tensor_bytes(*tensors: Any) -> int:
    return sum(int(t.numel()) * int(t.element_size()) for t in tensors)


def tensor_sha256(torch_module: Any, tensor: Any) -> str:
    raw = (
        tensor.detach()
        .contiguous()
        .view(torch_module.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw).hexdigest()


def assess_input_change_replay(torch_module: Any, output_a: Any, output_b: Any):
    finite = bool(
        torch_module.isfinite(output_a).all().item()
        and torch_module.isfinite(output_b).all().item()
    )
    equal = bool(torch_module.equal(output_a, output_b))
    delta_norm = float((output_b.float() - output_a.float()).norm().item())
    hash_a = tensor_sha256(torch_module, output_a)
    hash_b = tensor_sha256(torch_module, output_b)
    passed = finite and not equal and delta_norm > 0.0 and hash_a != hash_b
    return {
        "input_change_replay_passed": passed,
        "input_change_output_delta_norm": delta_norm,
        "input_change_output_a_sha256": hash_a,
        "input_change_output_b_sha256": hash_b,
    }


class AiterRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        aiter_root = Path(args.aiter_root).resolve()
        sglang_root = Path(args.sglang_root).resolve()
        for path in (str(aiter_root), str(sglang_root / "python")):
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            self.torch = importlib.import_module("torch")
            self.aiter = importlib.import_module("aiter")
            self.quant_mod = importlib.import_module("aiter.ops.quant")
            self.shuffle_mod = importlib.import_module("aiter.ops.shuffle")
            self.tuned_mod = importlib.import_module("aiter.tuned_gemm")
            self.a8w8_mod = importlib.import_module("aiter.ops.gemm_op_a8w8")
            self.a4w4_mod = importlib.import_module("aiter.ops.gemm_op_a4w4")
            self.common = importlib.import_module("microbench_common")
        except (ImportError, ModuleNotFoundError) as exc:
            raise UnsupportedMode(f"required AITER runtime import failed: {exc}") from exc

        if not self.torch.cuda.is_available():
            raise UnsupportedMode("HIP/CUDA device is unavailable")
        if getattr(self.torch.version, "hip", None) is None:
            raise UnsupportedMode("PyTorch is not a HIP build")

        self.aiter_root = aiter_root
        self.sglang_root = sglang_root
        try:
            self.gfx = importlib.import_module(
                "aiter.jit.utils.chip_info"
            ).get_gfx_runtime()
        except Exception as exc:
            raise UnsupportedMode(f"cannot determine AITER GPU architecture: {exc}") from exc

    def metadata(self) -> dict[str, Any]:
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": self.torch.__version__,
            "torch_hip": self.torch.version.hip,
            "aiter_module": str(Path(self.aiter.__file__).resolve()),
            "aiter_revision": _git_revision(self.aiter_root),
            "sglang_revision": _git_revision(self.sglang_root),
            "gfx": self.gfx,
            "device": self.torch.cuda.get_device_name(),
            "timing_helper": str(Path(self.common.__file__).resolve()),
        }

    def _seed(self, case: Case) -> int:
        stable_name = sum((index + 1) * ord(char) for index, char in enumerate(case.case_id))
        return int(self.args.seed) + stable_name

    def _inputs(self, case: Case):
        torch = self.torch
        torch.manual_seed(self._seed(case))
        torch.cuda.manual_seed_all(self._seed(case))
        x = torch.randn(
            (case.m, case.projection.k), device="cuda", dtype=torch.bfloat16
        )
        weight = torch.randn(
            (case.projection.n, case.projection.k),
            device="cuda",
            dtype=torch.bfloat16,
        ) / math.sqrt(case.projection.k)
        return x, weight.contiguous()

    def _bf16_reference(self, x, weight, norm_weight=None, epsilon=1e-6):
        torch = self.torch
        if norm_weight is not None:
            x = (
                x.float()
                * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + epsilon)
                * norm_weight.float()
            ).to(torch.bfloat16)
        return torch.nn.functional.linear(x, weight)

    def _dispatch_metadata(self, case: Case, prepared: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        try:
            if case.mode == "bf16":
                config = self.tuned_mod.get_GEMM_A16W16_config(
                    case.m,
                    case.projection.n,
                    case.projection.k,
                    False,
                    str(torch.bfloat16),
                    str(torch.bfloat16),
                )
                api = "aiter.tuned_gemm.tgemm.mm"
            elif case.mode == "ptpc_fp8":
                config = self.a8w8_mod.get_GEMM_config_with_quant_type(
                    case.m,
                    case.projection.n,
                    case.projection.k,
                    self.aiter.dtypes.fp8,
                )
                api = "aiter.gemm_a8w8_bpreshuffle"
            else:
                config = self.a4w4_mod.get_GEMM_config(
                    case.m, case.projection.n, case.projection.k
                )
                api = (
                    "sglang.latent_mxfp4_aiter_hip.run_norm_quant"
                    if case.mode == "rmsnorm_mxfp4"
                    else "aiter.gemm_a4w4"
                )
        except Exception as exc:
            config = {"metadata_error": f"{type(exc).__name__}: {exc}"}
            api = "unknown"
        return {
            "api": api,
            "config": _jsonable(config),
            "weight_layout": prepared["weight_layout"],
            "activation_quant_in_timed_chain": prepared["activation_quant"],
            "weight_prep_timed": False,
        }

    def _prepare(self, case: Case, weight) -> dict[str, Any]:
        if case.mode == "bf16":
            return {
                "weight": weight,
                "prepared_bytes": tensor_bytes(weight),
                "weight_layout": "bf16_nk",
                "activation_quant": "none",
            }
        if case.mode == "ptpc_fp8":
            quant = self.quant_mod.get_hip_quant(self.aiter.QuantType.per_Token)
            weight_q, weight_scale = quant(
                weight, quant_dtype=self.aiter.dtypes.fp8
            )
            weight_q = self.shuffle_mod.shuffle_weight(
                weight_q.contiguous(), layout=(16, 16)
            )
            return {
                "weight": weight_q,
                "scale": weight_scale,
                "prepared_bytes": tensor_bytes(weight_q, weight_scale),
                "weight_layout": "aiter_preshuffled_16x16_fp8_nk",
                "activation_quant": "aiter_per_token_fp8",
            }
        if case.mode in {"mxfp4", "rmsnorm_mxfp4"}:
            quant = self.quant_mod.get_hip_quant(self.aiter.QuantType.per_1x32)
            weight_q, weight_scale = quant(
                weight, quant_dtype=self.aiter.dtypes.fp4x2, shuffle=True
            )
            weight_q = self.shuffle_mod.shuffle_weight(
                weight_q, layout=(16, 16)
            )
            return {
                "weight": weight_q,
                "scale": weight_scale,
                "prepared_bytes": tensor_bytes(weight_q, weight_scale),
                "weight_layout": "aiter_preshuffled_16x16_mxfp4_nk",
                "activation_quant": "aiter_per_1x32_mxfp4",
            }
        raise UnsupportedMode(f"unknown mode {case.mode}")

    def _runner(self, case: Case, x, prepared, norm_weight=None):
        torch = self.torch
        if case.mode == "bf16":
            return lambda: self.tuned_mod.tgemm.mm(
                x, prepared["weight"], None, otype=x.dtype
            )
        if case.mode == "ptpc_fp8":
            quant = self.quant_mod.get_hip_quant(self.aiter.QuantType.per_Token)

            def run_fp8():
                xq, x_scale = quant(x, quant_dtype=self.aiter.dtypes.fp8)
                return self.aiter.gemm_a8w8_bpreshuffle(
                    xq,
                    prepared["weight"],
                    x_scale,
                    prepared["scale"],
                    None,
                    torch.bfloat16,
                )

            return run_fp8
        if case.mode == "mxfp4":
            quant = self.quant_mod.get_hip_quant(self.aiter.QuantType.per_1x32)

            def run_mxfp4():
                xq, x_scale = quant(
                    x, quant_dtype=self.aiter.dtypes.fp4x2, shuffle=True
                )
                return self.aiter.gemm_a4w4(
                    xq,
                    prepared["weight"],
                    x_scale,
                    prepared["scale"],
                    dtype=torch.bfloat16,
                    bpreshuffle=True,
                )[: case.m]

            return run_mxfp4
        if case.mode == "rmsnorm_mxfp4":
            if case.projection.name != "latent_up":
                raise UnsupportedMode("RMSNorm+MXFP4 is only defined for latent_up")
            os.environ["SGLANG_K3_MOE_LATENT_NORM_QUANT_MXFP4"] = "1"
            os.environ["SGLANG_K3_MOE_LATENT_UP_MXFP4_ALL_TOKENS"] = "1"
            try:
                adapter = importlib.import_module(
                    "sglang.kernels.ops.kimi_k3.latent_mxfp4_aiter_hip"
                )
            except (ImportError, ModuleNotFoundError) as exc:
                raise UnsupportedMode(f"SGLang latent MXFP4 adapter unavailable: {exc}") from exc
            if not adapter.supported():
                raise UnsupportedMode("SGLang latent RMSNorm+MXFP4 adapter is unsupported")
            if not adapter.norm_quant_covered(
                x, norm_weight, prepared["weight"], prepared["scale"]
            ):
                raise UnsupportedMode(
                    "current SGLang adapter rejects this RMSNorm+MXFP4 case"
                )
            return lambda: adapter.run_norm_quant(
                x,
                norm_weight,
                1e-6,
                prepared["weight"],
                prepared["scale"],
            )
        raise UnsupportedMode(f"unknown mode {case.mode}")

    def execute(self, case: Case) -> dict[str, Any]:
        if case.mode in {"mxfp4", "rmsnorm_mxfp4"} and self.gfx != "gfx950":
            raise UnsupportedMode(f"current Kimi MXFP4 production path requires gfx950, got {self.gfx}")

        torch = self.torch
        x, weight = self._inputs(case)
        norm_weight = None
        if case.mode == "rmsnorm_mxfp4":
            norm_weight = torch.randn(
                (case.projection.k,), device="cuda", dtype=torch.bfloat16
            )
        reference = self._bf16_reference(x, weight, norm_weight)
        prepared = self._prepare(case, weight)
        run = self._runner(case, x, prepared, norm_weight)

        try:
            candidate = run()
            torch.cuda.synchronize()
        except Exception as exc:
            raise UnsupportedMode(
                f"eager API smoke failed: {type(exc).__name__}: {exc}"
            ) from exc

        finite = bool(torch.isfinite(candidate).all().item())
        if not finite:
            raise RuntimeError("candidate produced NaN/Inf")
        rel_l2 = float(self.common.rel_l2(candidate, reference))
        cosine = float(
            torch.nn.functional.cosine_similarity(
                candidate.float().reshape(1, -1),
                reference.float().reshape(1, -1),
            ).item()
        )

        captured: dict[str, Any] = {}

        def capture_run():
            output = run()
            captured["output"] = output
            return output

        try:
            graph = self.common.capture_graph(
                capture_run,
                warmup=self.args.warmup,
            )
        except Exception as exc:
            raise UnsupportedMode(
                f"HIP graph capture unsupported: {type(exc).__name__}: {exc}"
            ) from exc

        original_x = x.clone()
        gate_metrics: dict[str, Any] = {}
        try:
            graph.replay()
            torch.cuda.synchronize()
            output_a = captured["output"].clone()
            torch.cuda.synchronize()

            # Negation is deterministic, changes the same static input storage,
            # and produces a strong signal for linear and RMSNorm-linear chains.
            x.copy_(original_x.neg())
            graph.replay()
            torch.cuda.synchronize()
            output_b = captured["output"].clone()
            torch.cuda.synchronize()

            gate_metrics = assess_input_change_replay(torch, output_a, output_b)
            changed_reference = self._bf16_reference(x, weight, norm_weight)
            gate_metrics["input_change_reference_rel_l2"] = float(
                self.common.rel_l2(output_b, changed_reference)
            )
            gate_metrics["input_change_reference_cosine"] = float(
                torch.nn.functional.cosine_similarity(
                    output_b.float().reshape(1, -1),
                    changed_reference.float().reshape(1, -1),
                ).item()
            )
        finally:
            # Restore the original values in the same static allocation. Replay
            # once before benchmark warmups so timing always starts from the
            # production input, even when the freshness gate fails.
            x.copy_(original_x)
            graph.replay()
            torch.cuda.synchronize()

        if not gate_metrics.get("input_change_replay_passed", False):
            raise CaseFailure(
                "stale HIP graph replay: output did not change after static input mutation",
                gate_metrics,
            )

        try:
            samples = self.common.benchmark_graph_replay(
                graph,
                warmup=self.args.warmup,
                iterations=self.args.iterations,
            )
            graph_ms = float(self.common.latency_summary(samples)["mean_ms"])
        except Exception as exc:
            raise UnsupportedMode(
                f"HIP graph replay timing unsupported: {type(exc).__name__}: {exc}"
            ) from exc

        eager_ms = None
        if self.args.eager:
            eager_ms = float(
                self.common.eager_bench(
                    run,
                    warmup=self.args.warmup,
                    iterations=self.args.iterations,
                )
            )

        row = base_result(case)
        row.update(
            {
                "status": "ok",
                "graph_ms": graph_ms,
                "eager_ms": eager_ms,
                "rel_l2": rel_l2,
                "cosine": cosine,
                "finite": finite,
                **gate_metrics,
                "dispatch": self._dispatch_metadata(case, prepared),
                "storage": storage_record(
                    case.projection.n,
                    case.projection.k,
                    case.mode,
                    actual_prepared_bytes=prepared["prepared_bytes"],
                ),
            }
        )
        del reference, weight, prepared, x
        torch.cuda.empty_cache()
        return row


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_id",
        "projection",
        "m",
        "n",
        "k",
        "projection_layer_count",
        "projection_layer_count_assumption",
        "mode",
        "context_only",
        "comparison_scope",
        "status",
        "skip_reason",
        "graph_ms",
        "eager_ms",
        "rel_l2",
        "cosine",
        "finite",
        "input_change_replay_passed",
        "input_change_output_delta_norm",
        "input_change_output_a_sha256",
        "input_change_output_b_sha256",
        "input_change_reference_rel_l2",
        "input_change_reference_cosine",
        "dispatch_api",
        "dispatch_config",
        "weight_layout",
        "bf16_weight_bytes",
        "prepared_weight_bytes",
        "nominal_prepared_weight_bytes",
        "incremental_dual_storage_bytes",
        "dual_storage_total_bytes",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            dispatch = row["dispatch"]
            storage = row["storage"]
            writer.writerow(
                {
                    **{key: row.get(key) for key in fieldnames if key in row},
                    "dispatch_api": dispatch.get("api"),
                    "dispatch_config": json.dumps(
                        _jsonable(dispatch.get("config", {})), sort_keys=True
                    ),
                    "weight_layout": dispatch.get("weight_layout"),
                    **storage,
                }
            )


def _storage_by_projection_mode(
    benchmark_report: dict[str, Any],
) -> dict[tuple[str, str], dict[str, int]]:
    storage_by_key: dict[tuple[str, str], dict[str, int]] = {}
    for row in benchmark_report["cases"]:
        key = (row["projection"], row["mode"])
        storage = {
            name: int(value) for name, value in row["storage"].items()
        }
        previous = storage_by_key.setdefault(key, storage)
        if previous != storage:
            raise ValueError(
                f"inconsistent storage across M buckets for {key[0]} {key[1]}"
            )
    return storage_by_key


def build_layer_weighted_storage_report(
    benchmark_report: dict[str, Any],
    *,
    source_path: Path | None = None,
) -> dict[str, Any]:
    """Expand per-shape storage from an existing benchmark over Kimi-K3 layers."""
    storage_by_key = _storage_by_projection_mode(benchmark_report)
    policy_keys = {
        (selection["projection"], selection["mode"]) for selection in RECOMMENDED_POLICY
    }
    rows: list[dict[str, Any]] = []
    for (projection_name, mode), storage in sorted(storage_by_key.items()):
        projection = PROJECTION_BY_NAME[projection_name]
        layer_count = projection.layer_count
        prepared = storage["prepared_weight_bytes"]
        bf16 = storage["bf16_weight_bytes"]
        selected = (projection_name, mode) in policy_keys
        rows.append(
            {
                "projection": projection_name,
                "mode": mode,
                "n": projection.n,
                "k": projection.k,
                "layer_count": layer_count,
                "layer_count_assumption": projection.layer_count_assumption,
                "representative_bf16_weight_bytes": bf16,
                "representative_prepared_weight_bytes": prepared,
                "model_layer_weighted_bf16_bytes": bf16 * layer_count,
                "model_layer_weighted_prepared_bytes": prepared * layer_count,
                "model_layer_weighted_dual_storage_total_bytes": (
                    bf16 * layer_count
                    + (0 if mode == "bf16" else prepared * layer_count)
                ),
                "recommended_policy_selected": selected,
                "recommended_policy_incremental_prepared_bytes": (
                    prepared * layer_count if selected else 0
                ),
            }
        )

    policy_contributions = []
    for selection in RECOMMENDED_POLICY:
        key = (selection["projection"], selection["mode"])
        storage = storage_by_key[key]
        projection = PROJECTION_BY_NAME[key[0]]
        contribution = storage["prepared_weight_bytes"] * projection.layer_count
        policy_contributions.append(
            {
                **selection,
                "layer_count": projection.layer_count,
                "prepared_bytes_per_layer_per_gpu": storage["prepared_weight_bytes"],
                "incremental_prepared_bytes_per_gpu": contribution,
                "incremental_prepared_mib_per_gpu": contribution / 2**20,
                "incremental_prepared_gib_per_gpu": contribution / 2**30,
            }
        )

    representative_total = sum(
        storage_by_key[(item["projection"], item["mode"])][
            "prepared_weight_bytes"
        ]
        for item in RECOMMENDED_POLICY
    )
    weighted_total = sum(
        item["incremental_prepared_bytes_per_gpu"] for item in policy_contributions
    )
    retained_bf16_total = sum(
        storage_by_key[(item["projection"], item["mode"])][
            "bf16_weight_bytes"
        ]
        * PROJECTION_BY_NAME[item["projection"]].layer_count
        for item in RECOMMENDED_POLICY
    )
    calibration = dict(LATENT_CAPACITY_CALIBRATION)
    calibration_prepared_per_layer = storage_by_key[
        (calibration["projection"], calibration["mode"])
    ]["prepared_weight_bytes"]
    calibration_total_bytes = (
        calibration_prepared_per_layer * calibration["layer_count"]
    )
    estimated_token_loss = round(
        weighted_total
        * calibration["measured_token_loss"]
        / calibration_total_bytes
    )
    calibration.update(
        {
            "prepared_bytes_per_layer_per_gpu": calibration_prepared_per_layer,
            "prepared_bytes_per_gpu": calibration_total_bytes,
            "bytes_per_token_loss": (
                calibration_total_bytes / calibration["measured_token_loss"]
            ),
        }
    )

    source = None
    if source_path is not None:
        source = {
            "path": str(source_path),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
    return {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_benchmark": source,
        "scope": "per_gpu_tp8_checkpoint_local_weights",
        "assumptions": {
            projection.name: {
                "layer_count": projection.layer_count,
                "reason": projection.layer_count_assumption,
            }
            for projection in PROJECTIONS
        },
        "representative_shape_storage": {
            "description": (
                "Unweighted one-copy total across the selected representative shapes; "
                "this is not model storage and replaces the former +50.32 MiB label."
            ),
            "incremental_prepared_bytes": representative_total,
            "incremental_prepared_mib": representative_total / 2**20,
        },
        "model_layer_weighted_storage": {
            "description": (
                "Prepared copies multiplied by actual projection layer counts, with "
                "the checkpoint's BF16 weights retained."
            ),
            "rows": rows,
            "recommended_policy": {
                "contributions": policy_contributions,
                "retained_bf16_bytes_per_gpu": retained_bf16_total,
                "retained_bf16_gib_per_gpu": retained_bf16_total / 2**30,
                "incremental_prepared_bytes_per_gpu": weighted_total,
                "incremental_prepared_mib_per_gpu": weighted_total / 2**20,
                "incremental_prepared_gib_per_gpu": weighted_total / 2**30,
                "dual_storage_total_bytes_per_gpu": (
                    retained_bf16_total + weighted_total
                ),
                "dual_storage_total_gib_per_gpu": (
                    retained_bf16_total + weighted_total
                )
                / 2**30,
            },
        },
        "estimated_token_capacity_impact": {
            "label": "calibrated_linear_estimate_not_a_measurement",
            "calibration": calibration,
            "estimated_token_loss": estimated_token_loss,
            "estimated_remaining_tokens_from_calibration_baseline": (
                calibration["baseline_tokens"] - estimated_token_loss
            ),
            "method": (
                "Scale the measured latent-up-only token loss linearly by total "
                "incremental prepared bytes per GPU. This assumes the same bytes/token "
                "slope and allocator behavior; no server was run."
            ),
        },
        "mutual_exclusivity": (
            "Each selected projection contributes exactly one prepared representation. "
            "MXFP4 and RMSNorm+MXFP4 share weights and are never added together; "
            "threshold-dependent modes do not allocate duplicate copies per M bucket."
        ),
    }


def write_layer_weighted_storage_csv(
    path: Path, storage_report: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = storage_report["model_layer_weighted_storage"]["rows"]
    fieldnames = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark complete Kimi-K3 dense projection chains with HIP graph replay.",
        epilog=(
            "Micro results do not promote a mode: common-client endpoint and paired "
            "GSM8K/long-context validation remain mandatory."
        ),
    )
    parser.add_argument(
        "--shapes",
        help="Comma-separated projection names (default: all, including context-only shapes).",
    )
    parser.add_argument(
        "--modes",
        help=f"Comma-separated modes from {','.join(DEFAULT_MODES)} (default: all).",
    )
    parser.add_argument(
        "--m-values",
        default=",".join(map(str, DEFAULT_M)),
        help="Comma-separated M values (default: 2,4,8,16,32,64).",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--eager", action="store_true", help="Also record eager timing.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--storage-input-json",
        type=Path,
        help="Build layer-weighted storage artifacts from an existing benchmark JSON; no GPU is used.",
    )
    parser.add_argument("--storage-output-json", type=Path)
    parser.add_argument("--storage-output-csv", type=Path)
    parser.add_argument("--aiter-root", type=Path, default=DEFAULT_AITER_ROOT)
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.storage_input_json:
        benchmark_report = json.loads(args.storage_input_json.read_text())
        storage_report = build_layer_weighted_storage_report(
            benchmark_report, source_path=args.storage_input_json
        )
        if args.storage_output_json:
            write_json(args.storage_output_json, storage_report)
        if args.storage_output_csv:
            write_layer_weighted_storage_csv(
                args.storage_output_csv, storage_report
            )
        if not args.storage_output_json:
            print(json.dumps(_jsonable(storage_report), indent=2, sort_keys=True))
        return 0
    if args.storage_output_json or args.storage_output_csv:
        parser.error("--storage-output-json/--storage-output-csv require --storage-input-json")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations must be positive")
    try:
        m_values = tuple(int(value) for value in args.m_values.split(","))
        cases = generate_cases(
            shape_filter=parse_csv_set(args.shapes),
            mode_filter=parse_csv_set(args.modes),
            m_values=m_values,
        )
    except ValueError as exc:
        parser.error(str(exc))

    runtime_metadata: dict[str, Any] = {}
    try:
        runtime = AiterRuntime(args)
        runtime_metadata = runtime.metadata()
        rows = run_cases(cases, runtime.execute)
    except UnsupportedMode as exc:
        runtime_metadata = {"unavailable_reason": str(exc)}
        rows = [skipped_result(case, str(exc)) for case in cases]

    report = build_report(rows, args=args, runtime_metadata=runtime_metadata)
    validate_report(report)
    if args.output_json:
        write_json(args.output_json, report)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if not args.output_json:
        print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
