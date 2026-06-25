#!/usr/bin/env python3
"""Per-launch GPU time for moe1/moe2 kernels (and all-reduce), grouped by tile
variant, from a trace. Lets us compare the MoE kernel itself across engines."""
import gzip, json, sys, glob
from collections import defaultdict

def load(p):
    op = gzip.open if p.endswith(".gz") else open
    return json.load(op(p, "rt")).get("traceEvents", [])

def summ(path):
    ev = load(path)
    agg = defaultdict(lambda: [0.0, 0])
    for e in ev:
        if e.get("cat") != "kernel" or "dur" not in e:
            continue
        n = e["name"]
        key = None
        if "mfma_moe1_silu_mul" in n:
            key = "moe1 " + n.split("mfma_moe1_silu_mul_afp8_wfp4_fp8_")[-1].split("_pm1")[0]
        elif "mfma_moe2" in n:
            key = "moe2 " + n.split("mfma_moe2_afp8_wfp4_bf16_cshuffle_")[-1].split("_vscale")[0]
        elif "quickreduce" in n:
            key = "AR quickreduce_twoshot"
        elif "cross_device_reduce_2stage" in n:
            key = "AR cross_device_reduce_2stage"
        elif "allgather_vec" in n:
            key = "AR allgather_vec"
        if key:
            agg[key][0] += e["dur"]; agg[key][1] += 1
    print(f"\n### {path}")
    print(f"  {'kernel':40s} {'total ms':>9s} {'launches':>8s} {'us/launch':>10s}")
    for k,(d,c) in sorted(agg.items()):
        print(f"  {k:40s} {d/1000:>9.1f} {c:>8d} {d/c:>10.1f}")

for a in sys.argv[1:]:
    for p in sorted(glob.glob(a)):
        summ(p)
