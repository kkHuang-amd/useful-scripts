#!/usr/bin/env python3
"""Summarize SGLang bench_serving (sglangClient_*) result JSONLs into a table.

Usage:
  python3 summarize_sgl_dsv4.py LABEL=/path/to/result_dir [LABEL2=/dir2 ...] [--gpus N]

Scans each dir for sglangClient_dsv4_isl*_osl*_c*.jsonl (last JSON line used).
Columns: total tok/s, tok/s/gpu, output tok/s, median TTFT/TPOT/ITL,
interactivity (=1000/median_TPOT_ms), median E2E. tok/s/gpu assumes TP=8.
"""
import json
import sys
import glob
import os
import re

GPUS = 8
args = []
for a in sys.argv[1:]:
    if a.startswith("--gpus"):
        GPUS = int(a.split("=")[1])
    elif "=" in a:
        args.append(a)

if not args:
    print(__doc__)
    sys.exit(1)

PAT = re.compile(r"sglangClient_dsv4_isl(\d+)_osl(\d+)_c(\d+)\.jsonl$")


def load_last(p):
    lines = [l for l in open(p) if l.strip()]
    return json.loads(lines[-1]) if lines else None


def load_dir(d):
    rows = []
    for p in glob.glob(os.path.join(d, "sglangClient_dsv4_isl*_osl*_c*.jsonl")):
        m = PAT.search(os.path.basename(p))
        if not m:
            continue
        isl, osl, conc = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            j = load_last(p)
        except Exception as e:
            print(f"  (skip {p}: {e})")
            continue
        if not j:
            continue
        tpot = j.get("median_tpot_ms", 0.0)
        rows.append({
            "isl": isl, "osl": osl, "conc": conc,
            "total": j.get("total_throughput", 0.0),
            "out": j.get("output_throughput", 0.0),
            "ttft": j.get("median_ttft_ms", 0.0),
            "tpot": tpot,
            "itl": j.get("median_itl_ms", 0.0),
            "interactivity": (1000.0 / tpot) if tpot else 0.0,
            "e2e": j.get("median_e2e_latency_ms", 0.0),
            "completed": j.get("completed", 0),
        })
    rows.sort(key=lambda r: (r["isl"], r["osl"], r["conc"]))
    return rows


hdr = (f"{'workload':>9s} {'conc':>5s} | {'total tok/s':>11s} {'tok/s/gpu':>9s} "
       f"{'out tok/s':>9s} | {'TTFT ms':>9s} {'TPOT ms':>8s} {'ITL ms':>8s} "
       f"{'interact':>8s} {'E2E ms':>10s} | {'done':>6s}")

for a in args:
    label, d = a.split("=", 1)
    rows = load_dir(d)
    print(f"\n========== {label}  (TP/GPUs={GPUS})  dir={d} ==========")
    print(hdr)
    print("-" * len(hdr))
    last_wl = None
    for r in rows:
        wl = f"{r['isl']//1024}k/{r['osl']//1024}k" if r['osl'] >= 1024 else f"{r['isl']}/{r['osl']}"
        if last_wl is not None and wl != last_wl:
            print()
        last_wl = wl
        print(f"{wl:>9s} {r['conc']:>5d} | {r['total']:>11.0f} {r['total']/GPUS:>9.0f} "
              f"{r['out']:>9.0f} | {r['ttft']:>9.1f} {r['tpot']:>8.2f} {r['itl']:>8.2f} "
              f"{r['interactivity']:>8.1f} {r['e2e']:>10.0f} | {r['completed']:>6d}")
print()
