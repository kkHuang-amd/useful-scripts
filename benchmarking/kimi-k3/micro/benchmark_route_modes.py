#!/usr/bin/env python3
"""Matched-route current-kernel Kimi-K3 fused-MoE microbenchmark.

The input artifacts are armed rank-0 route-count dumps, not timing evidence.
This tool deterministically realizes each count vector as a simple [64, 16]
token/expert assignment, then benchmarks the current AITER fused call with HIP
graph replay. Weight preparation is deliberately outside every timed region.

Micro results are kernel attribution only. They are not endpoint or promotion
evidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "kimi-k3-matched-route-current-kernel-v1"
TOKENS, TOPK, EXPERTS = 64, 16, 896
HIDDEN, INTER = 3584, 384
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0
BLOCK_M = 32
DEFAULT_RESULT_ROOT = Path(
    "/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/"
    "route-validation"
)
DEFAULT_AITER_ROOT = Path("/sgl-workspace/aiter-atom-current")
DEFAULT_SGLANG_ROOT = Path("/sgl-workspace/sglang-k3-triton37")
MODES = ("a8w4", "a16w4")
SOURCES = ("sglang", "atom")
SCOPES = ("full", "stage1", "stage2")


class ContractError(ValueError):
    """An invalid route artifact or impossible assignment."""


class UnsupportedRuntime(RuntimeError):
    """A fail-closed GPU/runtime/API skip."""


@dataclass(frozen=True)
class RouteCase:
    source: str
    layer: int
    counts: tuple[int, ...]
    source_path: str
    active_experts: int
    bm32_blocks: int


def dense_counts(payload: dict[str, Any], experts: int = EXPERTS) -> list[int]:
    if "full_bincount" in payload:
        result = [int(value) for value in payload["full_bincount"]]
    else:
        size = int(payload.get("expert_count_size", experts))
        result = [0] * size
        for item in payload.get("expert_counts", []):
            expert = int(item["expert"])
            if not 0 <= expert < size:
                raise ContractError(f"expert id {expert} outside [0, {size})")
            if result[expert]:
                raise ContractError(f"duplicate expert count entry {expert}")
            result[expert] = int(item["routes"])
    if len(result) != experts:
        raise ContractError(f"expected {experts} expert counts, got {len(result)}")
    if any(value < 0 for value in result):
        raise ContractError("expert route counts must be non-negative")
    return result


def validate_armed_dump(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema") != "k3-route-dump-v1" or payload.get("armed") is not True:
        raise ContractError(f"old or unarmed route dump is invalid: {path}")
    required = (
        "arm_time_ns",
        "arm_monotonic_ns",
        "dump_time_ns",
        "dump_monotonic_ns",
        "call_index",
        "rank",
        "topk_shape",
        "total_routes",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ContractError(f"{path}: missing armed fields {missing}")
    if int(payload["dump_time_ns"]) < int(payload["arm_time_ns"]):
        raise ContractError(f"{path}: dump wall clock predates arming")
    if int(payload["dump_monotonic_ns"]) < int(payload["arm_monotonic_ns"]):
        raise ContractError(f"{path}: dump monotonic clock predates arming")
    if [int(v) for v in payload["topk_shape"]] != [TOKENS, TOPK]:
        raise ContractError(f"{path}: expected topk shape [{TOKENS}, {TOPK}]")
    if int(payload["total_routes"]) != TOKENS * TOPK:
        raise ContractError(f"{path}: expected {TOKENS * TOPK} routes")


def reconstruct_topk_ids(
    counts: Sequence[int], *, tokens: int = TOKENS, topk: int = TOPK
) -> list[list[int]]:
    """Realize expert degrees against equal token degrees via Havel-Hakimi.

    Each expert is assigned to distinct tokens with the greatest remaining
    capacity. This preserves every expert id/count and prevents duplicate
    experts within a token. Impossible degree sequences fail explicitly.
    """
    values = [int(value) for value in counts]
    if any(value < 0 for value in values):
        raise ContractError("expert route counts must be non-negative")
    if sum(values) != tokens * topk:
        raise ContractError(
            f"route count sum {sum(values)} != tokens*topk {tokens * topk}"
        )
    if max(values, default=0) > tokens:
        raise ContractError(
            f"expert count {max(values)} exceeds one-per-token limit {tokens}"
        )
    remaining = [topk] * tokens
    rows: list[list[int]] = [[] for _ in range(tokens)]
    for expert, degree in sorted(
        enumerate(values), key=lambda item: (-item[1], item[0])
    ):
        if not degree:
            continue
        candidates = sorted(range(tokens), key=lambda token: (-remaining[token], token))
        selected = candidates[:degree]
        if len(selected) != degree or any(remaining[token] <= 0 for token in selected):
            raise ContractError(
                f"cannot assign expert {expert} degree {degree} without duplicates"
            )
        for token in selected:
            rows[token].append(expert)
            remaining[token] -= 1
    if any(remaining):
        raise ContractError(f"incomplete token capacities after assignment: {remaining}")
    for token, row in enumerate(rows):
        row.sort()
        if len(row) != topk or len(set(row)) != topk:
            raise ContractError(f"token {token} does not contain {topk} unique experts")
    observed = [0] * len(values)
    for row in rows:
        for expert in row:
            observed[expert] += 1
    if observed != values:
        raise ContractError("reconstructed assignment did not preserve expert counts")
    return rows


def bm32_blocks(counts: Sequence[int]) -> int:
    return sum((int(value) + BLOCK_M - 1) // BLOCK_M for value in counts if value)


def load_rank0_routes(root: Path, source: str) -> list[RouteCase]:
    records: dict[int, RouteCase] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema") != "k3-route-dump-v1":
            continue
        rank = payload.get("rank", payload.get("device_index"))
        if int(rank) != 0:
            continue
        validate_armed_dump(payload, path)
        layer = int(payload["call_index"])
        if layer in records:
            raise ContractError(f"{source}: duplicate rank0 call {layer}")
        counts = dense_counts(payload)
        reconstruct_topk_ids(counts)
        active = sum(value > 0 for value in counts)
        blocks = bm32_blocks(counts)
        if int(payload["unique_active_experts"]) != active:
            raise ContractError(f"{path}: active-expert metadata mismatch")
        if int(payload["bm32_padded_blocks"]) != blocks:
            raise ContractError(f"{path}: BM32 metadata mismatch")
        records[layer] = RouteCase(
            source, layer, tuple(counts), str(path), active, blocks
        )
    expected = set(range(92))
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise ContractError(
            f"{source}: expected rank0 calls 0..91; missing={missing}, extra={extra}"
        )
    return [records[layer] for layer in range(92)]


def fixed_topk_weights() -> list[list[float]]:
    raw = [float(TOPK - index) for index in range(TOPK)]
    total = sum(raw)
    row = [value / total for value in raw]
    return [list(row) for _ in range(TOKENS)]


def regression(rows: Sequence[dict[str, Any]], latency_key: str = "p50_ms") -> dict:
    pairs = [
        (float(row["bm32_blocks"]), float(row[latency_key]))
        for row in rows
        if row.get("status") == "ok" and row.get(latency_key) is not None
    ]
    if len(pairs) < 2:
        return {"count": len(pairs), "slope_ms_per_block": None, "intercept_ms": None,
                "r2": None, "pearson": None}
    xs, ys = zip(*pairs)
    xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - xbar) ** 2 for x in xs)
    syy = sum((y - ybar) ** 2 for y in ys)
    sxy = sum((x - xbar) * (y - ybar) for x, y in pairs)
    if sxx == 0:
        return {"count": len(pairs), "slope_ms_per_block": None, "intercept_ms": ybar,
                "r2": None, "pearson": None}
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    pearson = sxy / math.sqrt(sxx * syy) if syy else 0.0
    return {
        "count": len(pairs),
        "slope_ms_per_block": slope,
        "intercept_ms": intercept,
        "r2": pearson * pearson,
        "pearson": pearson,
    }


def aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for source in SOURCES:
        for mode in MODES:
            for scope in SCOPES:
                group = [
                    row for row in rows
                    if row["source"] == source and row["mode"] == mode
                    and row["scope"] == scope and row["status"] == "ok"
                ]
                latencies = [float(row["p50_ms"]) for row in group]
                ordered = sorted(group, key=lambda row: row["p50_ms"])
                result.append(
                    {
                        "source": source,
                        "mode": mode,
                        "scope": scope,
                        "ok_layers": len(group),
                        "sum_92_layer_p50_ms": sum(latencies) if len(group) == 92 else None,
                        "median_layer_p50_ms": (
                            statistics.median(latencies) if latencies else None
                        ),
                        "representative": {
                            label: (
                                {
                                    "layer": row["layer"],
                                    "p50_ms": row["p50_ms"],
                                    "active_experts": row["active_experts"],
                                    "bm32_blocks": row["bm32_blocks"],
                                }
                                if ordered else None
                            )
                            for label, row in (
                                ("min", ordered[0] if ordered else {}),
                                ("median", ordered[len(ordered) // 2] if ordered else {}),
                                ("max", ordered[-1] if ordered else {}),
                            )
                        },
                        "regression_vs_bm32_blocks": regression(group),
                    }
                )
    return result


def base_row(case: RouteCase, mode: str, scope: str) -> dict[str, Any]:
    return {
        "source": case.source,
        "layer": case.layer,
        "mode": mode,
        "scope": scope,
        "status": "pending",
        "skip_reason": None,
        "active_experts": case.active_experts,
        "bm32_blocks": case.bm32_blocks,
        "sample_count": 0,
        "min_ms": None,
        "p50_ms": None,
        "p90_ms": None,
        "max_ms": None,
        "mean_ms": None,
        "samples_ms": [],
        "finite": None,
        "input_change_detected": None,
        "weight_layout": None,
        "caller_output": True,
        "stage1_scratch_reuse": True,
    }


def skipped_row(case: RouteCase, mode: str, scope: str, reason: str) -> dict:
    row = base_row(case, mode, scope)
    row.update(status="skipped", skip_reason=reason)
    return row


def _git_revision(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@contextmanager
def activation_mode(mode: str):
    old = os.environ.get("AITER_SITUV2_A8W4")
    os.environ["AITER_SITUV2_A8W4"] = "1" if mode == "a8w4" else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("AITER_SITUV2_A8W4", None)
        else:
            os.environ["AITER_SITUV2_A8W4"] = old


class AiterRuntime:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.aiter_root = Path(args.aiter_root).resolve()
        self.sglang_root = Path(args.sglang_root).resolve()
        for path in (str(self.aiter_root), str(self.sglang_root / "python")):
            if path not in sys.path:
                sys.path.insert(0, path)
        self.torch = importlib.import_module("torch")
        self.aiter = importlib.import_module("aiter")
        self.fused_module = importlib.import_module("aiter.fused_moe")
        self.quant = importlib.import_module("aiter.ops.quant")
        self.shuffle = importlib.import_module("aiter.ops.shuffle")
        self.gate = importlib.import_module("aiter.ops.flydsl.moe_common").GateMode
        self.common = importlib.import_module("microbench_common")
        gfx = importlib.import_module("aiter.jit.utils.chip_info").get_gfx_runtime()
        if not self.torch.cuda.is_available() or self.torch.version.hip is None:
            raise UnsupportedRuntime("ROCm PyTorch GPU runtime is unavailable")
        if gfx != "gfx950":
            raise UnsupportedRuntime(f"current Kimi path requires gfx950, got {gfx}")
        self.gfx = gfx
        os.environ.setdefault("AITER_FLYDSL_FORCE", "1")
        os.environ.setdefault("AITER_SITUV2_A4W4", "0")
        os.environ.setdefault("AITER_FLYDSL_STAGE1_SCRATCH_REUSE", "1")
        self._prepare_fixed_data()

    def metadata(self) -> dict[str, Any]:
        return {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": self.torch.__version__,
            "torch_hip": self.torch.version.hip,
            "device": self.torch.cuda.get_device_name(),
            "gfx": self.gfx,
            "aiter_module": str(Path(self.aiter.__file__).resolve()),
            "aiter_revision": _git_revision(self.aiter_root),
            "sglang_revision": _git_revision(self.sglang_root),
            "stage_hook": "aiter.fused_moe.kernel_bench_callable",
        }

    def _prepare_fixed_data(self) -> None:
        torch, aiter = self.torch, self.aiter
        seed = int(self.args.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.hidden = torch.randn(
            (TOKENS, HIDDEN), dtype=torch.bfloat16, device="cuda"
        ) / math.sqrt(HIDDEN)
        tq = aiter.get_torch_quant(aiter.QuantType.per_1x32)
        # Quantize canonical separated GGUU weights once. The current AITER
        # op-test production contracts then derive distinct A8 and A16 layouts.
        w1 = torch.randn(
            (EXPERTS, INTER * 2, HIDDEN), dtype=torch.bfloat16, device="cuda"
        ) / math.sqrt(HIDDEN)
        w1q, w1s = tq(w1, quant_dtype=aiter.dtypes.fp4x2)
        del w1
        w1q = w1q.view(EXPERTS, INTER * 2, HIDDEN // 2)
        w2 = torch.randn(
            (EXPERTS, HIDDEN, INTER), dtype=torch.bfloat16, device="cuda"
        ) / math.sqrt(INTER)
        w2q, w2s = tq(w2, quant_dtype=aiter.dtypes.fp4x2)
        del w2
        w2q = w2q.view(EXPERTS, HIDDEN, INTER // 2)
        self.weights = {}
        for mode, gate_up in (("a8w4", True), ("a16w4", False)):
            pw1 = self.shuffle.shuffle_weight_a16w4(w1q, 16, gate_up)
            ps1 = self.shuffle.shuffle_scale_a16w4(w1s, EXPERTS, gate_up)
            pw2 = self.shuffle.shuffle_weight_a16w4(w2q, 16, False)
            ps2 = self.shuffle.shuffle_scale_a16w4(w2s, EXPERTS, False)
            pw1.is_shuffled = pw2.is_shuffled = True
            self.weights[mode] = {
                "w1": pw1, "w1_scale": ps1, "w2": pw2, "w2_scale": ps2,
                "layout": (
                    "shuffle_weight_a16w4(gate_up=True)_GUGU"
                    if gate_up else
                    "shuffle_weight_a16w4(gate_up=False)_GGUU"
                ),
            }
        del w1q, w1s, w2q, w2s
        self.topk_weights = torch.tensor(
            fixed_topk_weights(), dtype=torch.float32, device="cuda"
        )
        self.outputs = {
            mode: torch.empty((TOKENS, HIDDEN), dtype=torch.bfloat16, device="cuda")
            for mode in MODES
        }
        torch.cuda.empty_cache()

    def _call(self, mode: str, ids):
        data = self.weights[mode]
        return self.fused_module.fused_moe(
            self.hidden,
            data["w1"],
            data["w2"],
            self.topk_weights,
            ids,
            w1_scale=data["w1_scale"],
            w2_scale=data["w2_scale"],
            quant_type=self.aiter.QuantType.per_1x32,
            activation=self.aiter.ActivationType.Situv2,
            doweight_stage1=False,
            gate_mode=self.gate.SEPARATED.value,
            beta=SITU_BETA,
            linear_beta=SITU_LINEAR_BETA,
            output=self.outputs[mode],
        )

    def _capture_stage_calls(self, mode: str, ids) -> tuple[Any, list]:
        captured: list = []
        self.fused_module.kernel_bench_callable = captured
        try:
            output = self._call(mode, ids)
            self.torch.cuda.synchronize()
        finally:
            self.fused_module.kernel_bench_callable = None
        return output, captured

    def _time(self, run) -> tuple[list[float], Any]:
        graph = self.common.capture_graph(run, warmup=self.args.warmup)
        samples = self.common.benchmark_graph_replay(
            graph, warmup=self.args.warmup, iterations=self.args.iterations
        )
        return samples, graph

    def run_case(self, case: RouteCase, mode: str) -> tuple[list[dict], Any]:
        torch = self.torch
        ids = torch.tensor(
            reconstruct_topk_ids(case.counts), dtype=torch.int32, device="cuda"
        )
        rows: list[dict] = []
        with activation_mode(mode):
            output, captured = self._capture_stage_calls(mode, ids)
            finite = bool(torch.isfinite(output).all().item())
            if not finite:
                raise RuntimeError(f"{case.source} layer {case.layer} {mode}: NaN/Inf")
            full = base_row(case, mode, "full")
            samples, graph = self._time(lambda: self._call(mode, ids))
            baseline = output.clone()
            delta = torch.full_like(self.hidden, 0.03125)
            self.hidden.add_(delta)
            graph.replay()
            torch.cuda.synchronize()
            changed = not torch.equal(baseline, output)
            self.hidden.sub_(delta)
            graph.replay()
            torch.cuda.synchronize()
            if not changed:
                raise RuntimeError("captured fused_moe output did not change with input")
            full.update(
                status="ok", finite=True, input_change_detected=True,
                weight_layout=self.weights[mode]["layout"],
                samples_ms=samples, **self.common.latency_summary(samples),
            )
            rows.append(full)
            by_name = {name: call for name, call in captured}
            for scope in ("stage1", "stage2"):
                call = by_name.get(scope)
                if call is None:
                    rows.append(
                        skipped_row(
                            case, mode, scope,
                            "current dispatch did not expose this stage through "
                            "aiter.fused_moe.kernel_bench_callable",
                        )
                    )
                    continue
                stage = base_row(case, mode, scope)
                stage_samples, _ = self._time(call)
                stage.update(
                    status="ok", finite=True,
                    weight_layout=self.weights[mode]["layout"],
                    samples_ms=stage_samples,
                    **self.common.latency_summary(stage_samples),
                )
                rows.append(stage)
        return rows, output.detach().clone()


def numerical_comparison(a8, a16, torch_module) -> dict[str, Any]:
    finite = bool(torch_module.isfinite(a8).all() and torch_module.isfinite(a16).all())
    denominator = a16.float().norm()
    rel_l2 = float((a8.float() - a16.float()).norm() / denominator) if denominator else 0.0
    cosine = float(
        torch_module.nn.functional.cosine_similarity(
            a8.float().reshape(1, -1), a16.float().reshape(1, -1)
        ).item()
    )
    return {"finite": finite, "rel_l2_a8_vs_a16": rel_l2, "cosine_a8_vs_a16": cosine}


def validate_report(report: dict[str, Any]) -> None:
    required = {
        "schema_version", "created_at", "benchmark", "configuration", "runtime",
        "routes", "rows", "numerical_comparisons", "aggregates", "claims",
    }
    missing = required - report.keys()
    if missing:
        raise ContractError(f"report missing keys: {sorted(missing)}")
    for index, row in enumerate(report["rows"]):
        if row["status"] not in {"ok", "skipped", "failed"}:
            raise ContractError(f"row {index}: invalid status")
        if row["status"] == "skipped" and not row["skip_reason"]:
            raise ContractError(f"row {index}: skipped without reason")
        if row["status"] == "ok" and row["sample_count"] <= 0:
            raise ContractError(f"row {index}: ok without samples")


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Kimi-K3 matched-route current-kernel microbenchmark", "",
        "> HIP graph replay kernel micro only. No endpoint or promotion claim.", "",
        "## Configuration", "",
        f"- Shape: `M64 E{EXPERTS} H{HIDDEN} I{INTER} topk{TOPK}`",
        f"- SiTUv2: `beta={SITU_BETA:g}`, `linear_beta={SITU_LINEAR_BETA:g}`",
        f"- Warmup/iterations: `{report['configuration']['warmup']}` / "
        f"`{report['configuration']['iterations']}`",
        "- Weight prep is outside timing; full includes sorting, activation "
        "quantization, both stages, and caller-output handling.", "",
        "## 92-layer aggregates", "",
        "| Routes | Mode | Scope | OK | Sum of layer p50 (ms) | Median layer p50 (ms) | "
        "Slope (ms/BM32 block) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["aggregates"]:
        def fmt(value):
            return "n/a" if value is None else f"{value:.6f}"
        lines.append(
            f"| {row['source']} | {row['mode']} | {row['scope']} | "
            f"{row['ok_layers']} | {fmt(row['sum_92_layer_p50_ms'])} | "
            f"{fmt(row['median_layer_p50_ms'])} | "
            f"{fmt(row['regression_vs_bm32_blocks']['slope_ms_per_block'])} |"
        )
    lines.extend([
        "", "## Representative layer latencies", "",
        "| Routes | Mode | Scope | Point | Layer | p50 (ms) | Active experts | BM32 blocks |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for aggregate in report["aggregates"]:
        for label in ("min", "median", "max"):
            point = aggregate["representative"][label]
            if point is None:
                continue
            lines.append(
                f"| {aggregate['source']} | {aggregate['mode']} | "
                f"{aggregate['scope']} | {label} | {point['layer']} | "
                f"{point['p50_ms']:.6f} | {point['active_experts']} | "
                f"{point['bm32_blocks']} |"
            )
    lines.extend(["", "## Correctness", ""])
    comparisons = report["numerical_comparisons"]
    if comparisons:
        lines.append(
            f"- A8/A16 comparisons: `{len(comparisons)}` layers; all outputs finite: "
            f"`{all(item['finite'] for item in comparisons)}`."
        )
        lines.append(
            f"- Worst A8/A16 relative L2: "
            f"`{max(item['rel_l2_a8_vs_a16'] for item in comparisons):.6f}`; "
            f"minimum cosine: `{min(item['cosine_a8_vs_a16'] for item in comparisons):.6f}`."
        )
    lines.extend(["", "Representative min/median/max cases and full latency samples are "
                  "stored in the JSON; per-layer summaries are in the CSV.", ""])
    return "\n".join(lines)


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "route-mode-results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    fields = list(base_row(RouteCase("x", 0, tuple(), "", 0, 0), "a8w4", "full"))
    with (output_dir / "route-mode-layers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            serial = dict(row)
            serial["samples_ms"] = json.dumps(serial["samples_ms"])
            writer.writerow(serial)
    (output_dir / "route-mode-report.md").write_text(markdown_report(report))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark current AITER Kimi-K3 A8W4/A16W4 fused-MoE kernels with "
            "deterministically reconstructed armed rank-0 SGLang/ATOM routes."
        ),
        epilog=(
            "GPU command: python micro/benchmark_route_modes.py --output-dir "
            "/workspace/kimi-k3-runs/<experiment>/matched-route-micro . "
            "Estimated peak device memory: about 10 GiB (two prepared layouts, "
            "temporary preparation tensors, outputs/scratch/JIT headroom)."
        ),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--sglang-routes", type=Path)
    parser.add_argument("--atom-routes", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--max-a8-a16-rel-l2", type=float, default=0.25,
        help="fail GPU campaign above this A8/A16 relative-L2 threshold (default: 0.25)",
    )
    parser.add_argument(
        "--min-a8-a16-cosine", type=float, default=0.95,
        help="fail GPU campaign below this A8/A16 cosine threshold (default: 0.95)",
    )
    parser.add_argument("--aiter-root", type=Path, default=DEFAULT_AITER_ROOT)
    parser.add_argument("--sglang-root", type=Path, default=DEFAULT_SGLANG_ROOT)
    parser.add_argument(
        "--validate-routes-only", action="store_true",
        help="validate/reconstruct all routes and write schema without importing GPU modules",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be non-negative and --iterations positive")
    if args.max_a8_a16_rel_l2 < 0 or not -1 <= args.min_a8_a16_cosine <= 1:
        parser.error("invalid A8/A16 numerical thresholds")
    route_roots = {
        "sglang": args.sglang_routes or args.result_root / "sglang" / "route-dumps",
        "atom": args.atom_routes or args.result_root / "atom" / "route-dumps",
    }
    try:
        routes = {
            source: load_rank0_routes(Path(route_roots[source]), source)
            for source in SOURCES
        }
    except ContractError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    runtime_metadata: dict[str, Any] = {}
    if args.validate_routes_only:
        reason = "route-validation-only mode; GPU runtime intentionally not imported"
        for source in SOURCES:
            for case in routes[source]:
                for mode in MODES:
                    for scope in SCOPES:
                        rows.append(skipped_row(case, mode, scope, reason))
    else:
        try:
            runtime = AiterRuntime(args)
            runtime_metadata = runtime.metadata()
        except (ImportError, ModuleNotFoundError, UnsupportedRuntime) as error:
            print(f"ERROR {error}", file=sys.stderr)
            return 3
        for source in SOURCES:
            for case in routes[source]:
                outputs = {}
                for mode in MODES:
                    mode_rows, outputs[mode] = runtime.run_case(case, mode)
                    rows.extend(mode_rows)
                if set(outputs) == set(MODES):
                    comparison = numerical_comparison(
                        outputs["a8w4"], outputs["a16w4"], runtime.torch
                    )
                    comparison.update(source=source, layer=case.layer)
                    comparison["thresholds_pass"] = bool(
                        comparison["finite"]
                        and comparison["rel_l2_a8_vs_a16"]
                        <= args.max_a8_a16_rel_l2
                        and comparison["cosine_a8_vs_a16"]
                        >= args.min_a8_a16_cosine
                    )
                    if not comparison["thresholds_pass"]:
                        raise RuntimeError(
                            f"{source} layer {case.layer}: A8/A16 numerical gate failed: "
                            f"rel_l2={comparison['rel_l2_a8_vs_a16']:.6f} "
                            f"(max {args.max_a8_a16_rel_l2}), "
                            f"cosine={comparison['cosine_a8_vs_a16']:.6f} "
                            f"(min {args.min_a8_a16_cosine})"
                        )
                    comparisons.append(comparison)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "kimi_k3_matched_route_current_kernel",
        "configuration": {
            "tokens": TOKENS, "topk": TOPK, "experts": EXPERTS,
            "hidden": HIDDEN, "intermediate": INTER, "block_m": BLOCK_M,
            "activation": "SiTUv2", "beta": SITU_BETA,
            "linear_beta": SITU_LINEAR_BETA, "seed": args.seed,
            "max_a8_a16_rel_l2": args.max_a8_a16_rel_l2,
            "min_a8_a16_cosine": args.min_a8_a16_cosine,
            "warmup": args.warmup, "iterations": args.iterations,
            "weight_prep_timed": False, "graph_replay": True,
            "caller_output": True, "stage1_scratch_reuse": True,
        },
        "runtime": runtime_metadata,
        "routes": {
            source: {
                "root": str(Path(route_roots[source]).resolve()),
                "rank": 0, "layer_count": len(routes[source]),
                "all_reconstructed": True, "counts_preserved": True,
                "no_per_token_duplicates": True,
            }
            for source in SOURCES
        },
        "rows": rows,
        "numerical_comparisons": comparisons,
        "aggregates": aggregate_rows(rows),
        "claims": {
            "endpoint_claim": False,
            "promotion_evidence": False,
            "scope": "current-kernel graph-replay microbenchmark only",
        },
    }
    validate_report(report)
    write_outputs(args.output_dir, report)
    print(f"ROUTE_MODE_BENCH_OK output_dir={args.output_dir} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
