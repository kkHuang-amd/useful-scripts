#!/usr/bin/env python3
"""Thin launcher for sglang.bench_serving with shims so it can drive an ATOM
OpenAI-compatible server cleanly. Used by sweep_dsv4_sglang_client.sh.

Two shims, both non-invasive (no edits to the sglang repo at runtime):

Shim 1 (import, NOW OPTIONAL): the local sglang repo's
``srt/configs/cohere2_moe.py`` used to apply huggingface_hub's ``@strict`` to a
non-@dataclass class, which raised StrictDataclassDefinitionError on import.
This has since been fixed in the repo (cohere2_moe.py: drop @strict, add
@dataclass). The identity-``strict`` shim below is kept only as a defensive
fallback for environments that still have the unpatched file; it is a no-op
once the repo is patched.

Shim 2 (runtime, STILL REQUIRED for ATOM): ATOM's streaming /v1/completions
ends with a usage-only SSE chunk that has NO ``choices`` key, while
bench_serving assumes ``data["choices"][0]["text"]`` always exists -> KeyError.
We wrap json.loads to inject an empty ``choices`` into such usage/summary
chunks; bench_serving's existing falsy-text guard then skips them (output
length still comes from --random-output-len with ignore_eos on). This is an
ATOM server quirk, not an sglang issue, so keep it for ATOM targets. Against an
SGLang server you can call ``python -m sglang.bench_serving`` directly.

Usage: python3 bench_dsv4.py <all bench_serving args...>
"""
import sys
import json
import runpy

import huggingface_hub.dataclasses as _hf_dc


def _identity_strict(cls=None, **_kwargs):
    if cls is not None:
        return cls
    return lambda c: c


_hf_dc.strict = _identity_strict

_orig_loads = json.loads


def _patched_loads(s, *args, **kwargs):
    obj = _orig_loads(s, *args, **kwargs)
    if (
        isinstance(obj, dict)
        and "choices" not in obj
        and (obj.get("object") == "text_completion" or "usage" in obj)
    ):
        obj["choices"] = [{"text": "", "index": 0, "finish_reason": None}]
    return obj


json.loads = _patched_loads

runpy.run_module("sglang.bench_serving", run_name="__main__")
