"""Opt-in Kimi-K3 fused-MoE route diagnostics.

Python imports ``sitecustomize`` during interpreter startup.  Put this
directory first on ``PYTHONPATH`` and set ``K3_ROUTE_DUMP_DIR`` to install the
wrapper before SGLang or ATOM bind ``aiter.fused_moe.fused_moe`` locally.
Without that environment variable this module deliberately does nothing.
Dumping additionally requires ``K3_ROUTE_DUMP_ARM_FILE`` to name an existing,
valid armed-state JSON file.
"""

from __future__ import annotations

import os


if os.getenv("K3_ROUTE_DUMP_DIR"):
    import functools
    import importlib
    import json
    import math
    import tempfile
    import threading
    import time
    import warnings
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    _TARGET_SHAPE = (64, 16)
    _WRAPPER_MARKER = "_k3_route_dump_wrapper_v1"
    _ENV_KEYS = (
        "K3_ROUTE_ENV_MODE",
        "AITER_SITUV2_A8W4",
        "AITER_SITUV2_A4W4",
        "AITER_FLYDSL_FORCE",
        "SGLANG_K3_FLYDSL_SOURCE",
        "SGLANG_K3_MOE_LATENT_MXFP4",
        "SGLANG_K3_MOE_LATENT_DOWN_MXFP4",
        "SGLANG_K3_MOE_LATENT_UP_MXFP4",
        "ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD",
    )
    _lock = threading.Lock()
    _call_index = 0

    def _positive_int_env(name: str, default: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            return default
        return value if value > 0 else default

    def _is_stream_capturing(tensor: torch.Tensor) -> bool:
        if tensor.device.type not in ("cuda", "hip") or not torch.cuda.is_available():
            return False
        try:
            return bool(torch.cuda.is_current_stream_capturing())
        except RuntimeError:
            # Some ROCm/PyTorch combinations expose the API but reject it before
            # runtime initialization. Such a call cannot be in graph capture.
            return False

    def _rank_metadata(tensor: torch.Tensor) -> dict[str, object]:
        distributed_rank = None
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                distributed_rank = torch.distributed.get_rank()
        except RuntimeError:
            pass
        env_rank = next(
            (
                os.environ[name]
                for name in ("RANK", "LOCAL_RANK", "TP_RANK")
                if name in os.environ
            ),
            None,
        )
        device_index = tensor.device.index
        rank = distributed_rank
        if rank is None and env_rank is not None:
            try:
                rank = int(env_rank)
            except ValueError:
                rank = env_rank
        if rank is None:
            rank = device_index
        return {
            "rank": rank,
            "distributed_rank": distributed_rank,
            "env_rank": env_rank,
            "local_rank": os.getenv("LOCAL_RANK"),
            "device": str(tensor.device),
            "device_index": device_index,
            "pid": os.getpid(),
        }

    def _layout(tensor: torch.Tensor | None) -> dict[str, object] | None:
        if tensor is None:
            return None
        return {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "contiguous": tensor.is_contiguous(),
        }

    def _route_payload(
        call_index: int,
        topk_ids: torch.Tensor,
        hidden_states: torch.Tensor | None,
        w1: torch.Tensor | None,
        quant_type: object,
        arm_state: dict[str, object],
        arm_file: Path,
    ) -> dict[str, object]:
        ids = topk_ids.detach().reshape(-1).to(dtype=torch.int64, device="cpu")
        if ids.numel() and int(ids.min()) < 0:
            raise ValueError("topk_ids contains a negative expert id")
        inferred_experts = int(w1.shape[0]) if w1 is not None and w1.ndim else 0
        count_size = max(inferred_experts, int(ids.max()) + 1 if ids.numel() else 0)
        counts = torch.bincount(ids, minlength=count_size)
        active = torch.nonzero(counts, as_tuple=False).reshape(-1)
        compact_counts = [
            {"expert": int(expert), "routes": int(counts[expert])}
            for expert in active.tolist()
        ]
        nonzero = [item["routes"] for item in compact_counts]
        padded_blocks = sum(math.ceil(count / 32) for count in nonzero)
        top_experts = sorted(
            compact_counts,
            key=lambda item: (-item["routes"], item["expert"]),
        )[:16]
        dump_time_ns = time.time_ns()
        dump_monotonic_ns = time.monotonic_ns()
        payload = {
            "schema": "k3-route-dump-v1",
            "armed": True,
            "arm_file": str(arm_file.resolve()),
            "arm_timestamp_utc": arm_state["armed_at_utc"],
            "arm_time_ns": int(arm_state["armed_time_ns"]),
            "arm_monotonic_ns": int(arm_state["armed_monotonic_ns"]),
            "dump_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dump_time_ns": dump_time_ns,
            "dump_monotonic_ns": dump_monotonic_ns,
            "call_index": call_index,
            **_rank_metadata(topk_ids),
            "topk_shape": list(topk_ids.shape),
            "total_routes": int(ids.numel()),
            "unique_active_experts": len(compact_counts),
            "nonzero_routes": {
                "min": min(nonzero) if nonzero else 0,
                "max": max(nonzero) if nonzero else 0,
                "mean": (sum(nonzero) / len(nonzero)) if nonzero else 0.0,
            },
            "expert_count_size": int(counts.numel()),
            "expert_counts": compact_counts,
            "bm32_padded_blocks": padded_blocks,
            "bm32_padded_routes": padded_blocks * 32,
            "top_experts": top_experts,
            "topk_ids": _layout(topk_ids),
            "hidden_states": _layout(hidden_states),
            "quant_type": str(quant_type),
            "env_mode": os.getenv("K3_ROUTE_ENV_MODE", "unspecified"),
            "route_environment": {
                key: os.environ[key] for key in _ENV_KEYS if key in os.environ
            },
        }
        # A full bincount is convenient for small expert sets; compact counts
        # remain the complete representation when the inferred range is large.
        if counts.numel() <= 4096:
            payload["full_bincount"] = [int(value) for value in counts.tolist()]
        return payload

    def _read_arm_state() -> tuple[Path, dict[str, object]] | None:
        value = os.getenv("K3_ROUTE_DUMP_ARM_FILE")
        if not value:
            return None
        path = Path(value)
        try:
            state = json.loads(path.read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        required = ("armed_at_utc", "armed_time_ns", "armed_monotonic_ns")
        if state.get("armed") is not True or any(key not in state for key in required):
            return None
        return path, state

    def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def _install_wrapper() -> None:
        module = importlib.import_module("aiter.fused_moe")
        original = module.fused_moe
        if getattr(original, _WRAPPER_MARKER, False):
            return

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            topk_ids = kwargs.get("topk_ids")
            if topk_ids is None and len(args) > 4:
                topk_ids = args[4]
            if (
                isinstance(topk_ids, torch.Tensor)
                and tuple(topk_ids.shape) == _TARGET_SHAPE
                and not _is_stream_capturing(topk_ids)
            ):
                arm = _read_arm_state()
                if arm is not None:
                    arm_file, arm_state = arm
                    global _call_index
                    max_calls = _positive_int_env("K3_ROUTE_DUMP_MAX_CALLS", 92)
                    with _lock:
                        if _call_index < max_calls:
                            call_index = _call_index
                            _call_index += 1
                        else:
                            call_index = None
                else:
                    call_index = None
                if call_index is not None and arm is not None:
                    try:
                        hidden_states = kwargs.get("hidden_states")
                        if hidden_states is None and args:
                            hidden_states = args[0]
                        w1 = kwargs.get("w1")
                        if w1 is None and len(args) > 1:
                            w1 = args[1]
                        quant_type = kwargs.get("quant_type")
                        if quant_type is None and len(args) > 7:
                            quant_type = args[7]
                        payload = _route_payload(
                            call_index,
                            topk_ids,
                            hidden_states,
                            w1,
                            quant_type,
                            arm_state,
                            arm_file,
                        )
                        identity = (
                            f"rank-{payload['rank']}-device-{payload['device_index']}"
                            f"-pid-{payload['pid']}-call-{call_index:03d}.json"
                        )
                        _atomic_json_write(
                            Path(os.environ["K3_ROUTE_DUMP_DIR"]) / identity,
                            payload,
                        )
                    except Exception as error:
                        # Diagnostics must never alter fused_moe return behavior.
                        warnings.warn(
                            f"K3 route dump call {call_index} failed: {error}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            return original(*args, **kwargs)

        setattr(wrapped, _WRAPPER_MARKER, True)
        module.fused_moe = wrapped

    _install_wrapper()
