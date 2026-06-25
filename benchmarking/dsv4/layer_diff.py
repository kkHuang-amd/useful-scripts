#!/usr/bin/env python3
"""Per-LAYER kernel diff between two DSV4 prefill traces (e.g. SGLang vs ATOM).

WHY: in DSV4 the decoder layers alternate compress_ratio 128 / 4 (config.json
`compress_ratios`), so the MLA attention kernel `pa_prefill` has alternating
duration. Using `pa_prefill` as the LAYER BOUNDARY, one "window" = pa_prefill[i] ->
pa_prefill[i+1] = one decoder layer's worth of kernels (attn tail + MoE + next
layer's pre-attn). Comparing the SAME compress-ratio layer across engines (match by
pa_prefill duration) isolates real per-layer differences.

Reliability rule (Exp 19/48): per-kernel `dur` is UNRELIABLE (host-sync inflated,
esp. ATOM). The trustworthy metrics are:
  - the WINDOW span (ts of next pa_prefill - ts of this pa_prefill) = per-layer wall.
  - the GPU-active UNION within the window (wall time with >=1 kernel running).
The per-kernel dur is shown only for op-sequence/identification, not as ground truth.

USAGE:
  # overview: list pa_prefill (=layer) windows with dur + GPU-active union
  python3 layer_diff.py overview <trace.json.gz>
  # dump one layer's ordered kernel sequence (pick by ratio 4 or 128, mid-trace)
  python3 layer_diff.py seq <trace.json.gz> <ratio:4|128>
  # side-by-side two traces for a ratio (the main debug view)
  python3 layer_diff.py cmp <sgl_trace.json.gz> <atom_trace.json.gz> <ratio:4|128>

Traces: SGLang dumps per-rank `*TP-0-DP-0.trace.json.gz` (via /start_profile with
output_dir + num_steps); ATOM dumps `dp0_tp0/*.pt.trace.json.gz` (via /start_profile
+ /stop_profile when launched with ATOM_TORCH_PROFILER_DIR). Use rank 0. For a clean
apples-to-apples compare run BOTH single-stream (SGLang aligned is single-stream;
ATOM with ATOM_DISABLE_SIDE_STREAMS=1) and pure-prefill (OSL=1, same chunk/rank).
"""
import gzip, json, sys
from collections import defaultdict


def load_kern(path):
    op = gzip.open if path.endswith(".gz") else open
    ev = json.load(op(path, "rt")).get("traceEvents", [])
    return sorted(
        [e for e in ev if e.get("cat") == "kernel" and "dur" in e and "ts" in e],
        key=lambda e: e["ts"],
    )


def pa_indices(kern):
    return [i for i, e in enumerate(kern) if "pa_prefill" in e["name"].lower()]


def union_us(kern_slice):
    iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in kern_slice)
    if not iv:
        return 0.0
    u = 0.0
    cs, ce = iv[0]
    for s, en in iv[1:]:
        if s > ce:
            u += ce - cs
            cs, ce = s, en
        else:
            ce = max(ce, en)
    return u + (ce - cs)


def shorten(n):
    nl = n.lower()
    table = [
        ("nccl", "comm(nccl)"),
        ("moe1_silu", "moe1_up_gate(silu)"),
        ("moe2", "moe2_down") if "cshuffle" in nl else (None, None),
        ("moe_reduction", "moe_reduce"),
        ("moe_sorting", "moe_sort"),
        ("opus_moe_sorting", "moe_sort"),
        ("pa_prefill", ">>> pa_prefill(MLA attn)"),
        ("compressor_states", "compressor_update"),
        ("hca_norm_rope_scatter", "hca_norm_rope_scatter(fused)"),
        ("hca_compress_forward", "hca_compress_forward"),
        ("gemm_a8w8_blockscale_kernel", "GEMM:triton_a8w8_blockscale"),
        ("quantgemm", "GEMM:ck_tile_quant"),
        ("kernel_gemm_xdl_cshuffle", "GEMM:ck_xdl_blockscale"),
        ("cijk", "GEMM:cijk(rocblas)"),
        ("mhc_post", "mhc_post"),
        ("mhc_pre", "mhc_pre"),
        ("sqrsum", "mhc_pre"),
        ("apply_rotary_emb_triton_kernel_batched", "rope_batched"),
        ("apply_rotary_emb", "rope(per-token)"),
        ("inverse_rope", "rope_inverse_gptj"),
        ("qk_norm_rope", "qk_norm_rope_fused"),
        ("fused_q_norm_rope", "qk_norm_rope_fused"),
        ("act_and_mul", "act_and_mul"),
        ("clamp_silu_mul", "silu_mul"),
        ("quant", "fp8_quant"),
        ("catarray", "cat/copy"),
        ("elementwise", "elementwise"),
        ("rocprim", "rocprim(sort/scan)"),
        ("fill", "fill"),
    ]
    for key, name in table:
        if key and key in nl:
            return name
    return n[:46]


def dump_seq(kern, i0, i1, label):
    print("--- %s | window=%.0f us  GPU-active(union)=%.0f us ---"
          % (label, kern[i1]["ts"] - kern[i0]["ts"], union_us(kern[i0:i1])))
    out = []
    for e in kern[i0:i1]:
        s = shorten(e["name"])
        if out and out[-1][0] == s:
            out[-1][1] += e["dur"]; out[-1][2] += 1
        else:
            out.append([s, e["dur"], 1])
    for s, d, c in out:
        print("   %-32s %8.1f us  x%d" % (s, d, c))


def pick_window(kern, pis, ratio):
    target = 4600 if str(ratio) == "4" else 1950
    for k in range(len(pis) // 3, 2 * len(pis) // 3):
        if abs(kern[pis[k]]["dur"] - target) < 800:
            return k
    return len(pis) // 2


def main():
    cmd = sys.argv[1]
    if cmd == "overview":
        kern = load_kern(sys.argv[2]); pis = pa_indices(kern)
        print("pa_prefill windows: %d (=layers x steps)" % (len(pis) - 1))
        for k in range(min(len(pis) - 1, 64)):
            i0, i1 = pis[k], pis[k + 1]
            print("  win%2d pa_dur=%5.0f  window=%6.0f us  GPU-active=%6.0f us"
                  % (k, kern[i0]["dur"], kern[i1]["ts"] - kern[i0]["ts"], union_us(kern[i0:i1])))
    elif cmd == "seq":
        kern = load_kern(sys.argv[2]); pis = pa_indices(kern)
        k = pick_window(kern, pis, sys.argv[3])
        dump_seq(kern, pis[k], pis[k + 1], "ratio=%s pa_dur=%.0f" % (sys.argv[3], kern[pis[k]]["dur"]))
    elif cmd == "cmp":
        for tag, p in [("A: " + sys.argv[2], sys.argv[2]), ("B: " + sys.argv[3], sys.argv[3])]:
            kern = load_kern(p); pis = pa_indices(kern)
            k = pick_window(kern, pis, sys.argv[4])
            dump_seq(kern, pis[k], pis[k + 1], "%s ratio=%s pa_dur=%.0f"
                     % (tag, sys.argv[4], kern[pis[k]]["dur"]))
            print()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
