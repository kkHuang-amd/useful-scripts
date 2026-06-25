#!/usr/bin/env python3
"""Summarize a torch/chrome trace: top GPU kernels, GPU busy vs idle, step gaps."""
import gzip, json, sys, glob, os
from collections import defaultdict

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d if isinstance(d, list) else [])

def analyze(path):
    ev = load(path)
    # GPU kernel events: cat == 'kernel'. Also collect their stream/pid.
    kern = [e for e in ev if e.get("cat") == "kernel" and "dur" in e and "ts" in e]
    if not kern:
        # some traces label differently
        kern = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("gpu_op","Kernel") and "dur" in e]
    by_name = defaultdict(lambda: [0.0, 0])
    for e in kern:
        by_name[e["name"]][0] += e["dur"]
        by_name[e["name"]][1] += 1
    total_k = sum(v[0] for v in by_name.values())
    tmin = min(e["ts"] for e in kern)
    tmax = max(e["ts"] + e["dur"] for e in kern)
    span = tmax - tmin
    # busy time = union of kernel intervals (approx: sum dur if non-overlapping per stream)
    # compute per-stream (tid) union
    by_tid = defaultdict(list)
    for e in kern:
        by_tid[e.get("tid")].append((e["ts"], e["ts"]+e["dur"]))
    busy = 0.0
    for tid, ivs in by_tid.items():
        ivs.sort()
        cur_s, cur_e = ivs[0]
        for s,en in ivs[1:]:
            if s > cur_e:
                busy += cur_e-cur_s; cur_s, cur_e = s, en
            else:
                cur_e = max(cur_e, en)
        busy += cur_e-cur_s
    # the dominant stream busy (max over tids) approximates GPU occupancy
    stream_busy = {}
    for tid, ivs in by_tid.items():
        ivs.sort(); b=0.0; cs,ce=ivs[0]
        for s,en in ivs[1:]:
            if s>ce: b+=ce-cs; cs,ce=s,en
            else: ce=max(ce,en)
        b+=ce-cs; stream_busy[tid]=b
    main_busy = max(stream_busy.values()) if stream_busy else 0
    print(f"\n### {path}")
    print(f"  #kernel events: {len(kern)}   span: {span/1000:.1f} ms   total kernel-time: {total_k/1000:.1f} ms")
    print(f"  busiest-stream busy: {main_busy/1000:.1f} ms  ({100*main_busy/span:.0f}% of span)  idle on that stream: {(span-main_busy)/1000:.1f} ms")
    print(f"  top GPU kernels by total time:")
    for name,(dur,cnt) in sorted(by_name.items(), key=lambda x:-x[1][0])[:18]:
        print(f"    {dur/1000:8.1f} ms  x{cnt:5d}  {name[:90]}")

for arg in sys.argv[1:]:
    paths = sorted(glob.glob(arg))
    for p in paths:
        analyze(p)
