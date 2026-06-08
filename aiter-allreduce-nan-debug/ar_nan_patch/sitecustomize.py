"""
Direction-1 instrumentation: catch the exact aiter custom all-reduce call that
first produces NaN/Inf in the real decode workload.

Auto-imported by Python when this dir is on PYTHONPATH (sitecustomize). Active
only when AR_NAN_CHECK=1.

Monkeypatches aiter CustomAllreduce.all_reduce / fused_ar_rms to, after each
call, check the OUTPUT for NaN/Inf and whether the INPUT was already bad, then
log shape/dtype/input_already_bad + a Python stack (which layer). Logged to
stderr -> decode log.

  INPUT_ALREADY_BAD=False + output NaN  => aiter AR is the NaN source
  INPUT_ALREADY_BAD=True                => NaN came from upstream (look earlier)
"""
import os
import sys
import traceback

if os.environ.get("AR_NAN_CHECK", "0") == "1":
    # NOTE: do NOT import torch at top level here. sitecustomize runs at
    # interpreter startup for every process (incl. spawned TP workers); importing
    # torch that early perturbs sglang's ROCm/torch init ordering and stalls
    # startup. torch is always already imported by the time AR runs, so we import
    # it lazily inside the functions (cheap, cached).
    _state = {"calls": 0, "fired": 0}
    _MAX = int(os.environ.get("AR_NAN_MAX_REPORTS", "20"))

    def _check(tag, out_tensor, in_tensor):
        import torch
        _state["calls"] += 1
        try:
            o = out_tensor
            if not torch.is_tensor(o):
                return
            if not bool(torch.isnan(o).any().item() or torch.isinf(o).any().item()):
                return
            in_bad = torch.is_tensor(in_tensor) and bool(
                torch.isnan(in_tensor).any().item() or torch.isinf(in_tensor).any().item()
            )
            if _state["fired"] < _MAX:
                _state["fired"] += 1
                shp = tuple(in_tensor.shape) if torch.is_tensor(in_tensor) else "?"
                dt = getattr(in_tensor, "dtype", "?")
                sys.stderr.write(
                    f"\n[AR_NAN] {tag}: OUTPUT NaN/Inf  call#={_state['calls']}  "
                    f"shape={shp}  dtype={dt}  INPUT_ALREADY_BAD={in_bad}\n"
                    "[AR_NAN] caller stack (most recent last):\n"
                )
                traceback.print_stack(file=sys.stderr)
                sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"[AR_NAN] check error: {e}\n")

    def _install():
        from aiter.dist.device_communicators import custom_all_reduce as m
        CA = m.CustomAllreduce
        if getattr(CA, "_ar_nan_patched", False):
            return True
        CA._ar_nan_patched = True

        _orig = CA.all_reduce

        def all_reduce(self, inp, **kw):
            out = _orig(self, inp, **kw)
            _check("all_reduce", out, inp)
            return out

        CA.all_reduce = all_reduce

        if hasattr(CA, "fused_ar_rms"):
            _orig_f = CA.fused_ar_rms

            def fused_ar_rms(self, inp, res_inp, **kw):
                res = _orig_f(self, inp, res_inp, **kw)
                out = res[0] if isinstance(res, tuple) else res
                _check("fused_ar_rms", out, inp)
                return res

            CA.fused_ar_rms = fused_ar_rms

        sys.stderr.write("[AR_NAN] aiter CustomAllreduce instrumented (AR_NAN_CHECK=1)\n")
        sys.stderr.flush()
        return True

    # IMPORTANT: do NOT call _install() eagerly here -- that would import aiter at
    # interpreter startup (before sglang's ROCm/HIP init), which hangs. Instead we
    # ONLY patch lazily, after sglang itself has imported the aiter AR module. We
    # detect that via an O(1) sys.modules check inside a thin __import__ wrapper
    # (no early aiter import, no expensive failing imports).
    import builtins
    _MODNAME = "aiter.dist.device_communicators.custom_all_reduce"
    _orig_import = builtins.__import__

    def _patched_import(name, *a, **k):
        mod = _orig_import(name, *a, **k)
        if (not _state.get("installed")) and (_MODNAME in sys.modules):
            try:
                if _install():
                    _state["installed"] = True
                    builtins.__import__ = _orig_import
            except Exception:
                pass
        return mod

    builtins.__import__ = _patched_import
