#!/usr/bin/env python3
"""Per-forward-step wall time from a trace, using the routed-MoE gemm1 kernel as
a once-per-forward marker. Classifies prefill (large token count -> long kernel)
vs decode (tiny) and reports wall-time spent in each."""
import gzip, json, sys, glob

MOE1_SUBSTR = "mfma_moe1_silu_mul"   # routed expert gemm1, 1x per forward w/ MoE

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d if isinstance(d, list) else [])

def analyze(path):
    ev = load(path)
    moe1 = sorted([e for e in ev if e.get("cat")=="kernel" and MOE1_SUBSTR in e.get("name","")],
                  key=lambda e:e["ts"])
    if len(moe1) < 3:
        print(f"\n### {path}\n  (only {len(moe1)} moe1 kernels; skip)"); return
    starts = [e["ts"] for e in moe1]
    durs   = [e["dur"] for e in moe1]
    # classify by kernel duration: decode kernels are small, prefill large
    med = sorted(durs)[len(durs)//2]
    thr = max(med*3, 200)  # us
    deltas = [(starts[i+1]-starts[i], durs[i], durs[i]>thr) for i in range(len(starts)-1)]
    dec = [d for d,k,p in deltas if not p]
    pre = [d for d,k,p in deltas if p]
    span = starts[-1]-starts[0]
    print(f"\n### {path}")
    print(f"  forwards(moe1): {len(moe1)}   span(first->last moe1): {span/1000:.1f} ms")
    print(f"  moe1 dur: median {med:.0f} us  (prefill threshold {thr:.0f} us)")
    print(f"  PREFILL steps: {len(pre):3d}   total wall {sum(pre)/1000:7.1f} ms   avg step {sum(pre)/len(pre)/1000 if pre else 0:6.2f} ms")
    print(f"  DECODE  steps: {len(dec):3d}   total wall {sum(dec)/1000:7.1f} ms   avg step {sum(dec)/len(dec)/1000 if dec else 0:6.2f} ms")
    if dec:
        sd=sorted(dec); 
        print(f"  decode step wall: p50 {sd[len(sd)//2]/1000:.2f} ms  p90 {sd[int(len(sd)*0.9)]/1000:.2f} ms  max {sd[-1]/1000:.2f} ms")
    frac_pre = sum(pre)/(sum(pre)+sum(dec))*100 if (pre or dec) else 0
    print(f"  => wall fraction spent in PREFILL steps: {frac_pre:.0f}%")

for arg in sys.argv[1:]:
    for p in sorted(glob.glob(arg)):
        analyze(p)
