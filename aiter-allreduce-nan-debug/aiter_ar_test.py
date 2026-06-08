#!/usr/bin/env python3
"""
Isolated correctness test for custom all-reduce in a CUDA graph (Phase 3, v3).

This faithfully mimics sglang's capture flow:
    with ca.capture():            # sets _IS_CAPTURING; flush_graph_buffers on exit
        with torch.cuda.graph(g):
            x   = static_inp * 1   # graph-internal intermediate (like a fwd activation)
            out = ca.custom_all_reduce(x)

and runs the SAME harness against two implementations so the harness itself is
controlled:
    AR_IMPL=aiter   -> aiter.dist...CustomAllreduce   (suspected buggy)
    AR_IMPL=sglang  -> sglang.srt...CustomAllreduce   (control / reference impl)

If only AR_IMPL=aiter diverges/NaNs while AR_IMPL=sglang is clean through the
identical harness, the bug is in aiter's custom all-reduce (cuda-graph path).

Launch on an 8-GPU node:
    AR_IMPL=aiter  torchrun --nproc_per_node=8 aiter_ar_test.py
    AR_IMPL=sglang torchrun --nproc_per_node=8 aiter_ar_test.py

Env: ITERS(100) HIDDEN(7168) USE_NEW(1) MAG(2.0) AR_IMPL(aiter)
"""
import os
import sys

import torch
import torch.distributed as dist


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def get_ar_class(impl):
    if impl == "aiter":
        from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
        return CustomAllreduce
    elif impl == "sglang":
        from sglang.srt.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        return CustomAllreduce
    raise ValueError(f"unknown AR_IMPL={impl}")


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    iters = int(os.environ.get("ITERS", "100"))
    hidden = int(os.environ.get("HIDDEN", "7168"))
    use_new = os.environ.get("USE_NEW", "1") == "1"
    mag = float(os.environ.get("MAG", "2.0"))
    impl = os.environ.get("AR_IMPL", "aiter")
    seed_base = int(os.environ.get("SEED_BASE", "1234"))
    max_size = int(os.environ.get("MAX_SIZE", str(1024 * 1024 * 1024)))

    dist.init_process_group(backend="nccl")
    gloo_group = dist.new_group(backend="gloo")

    log(rank, f"=== custom AR cuda-graph correctness test v3  [AR_IMPL={impl}] ===")
    log(rank, f"world_size={world_size} hidden={hidden} use_new={use_new} mag={mag} iters={iters}")

    ARClass = get_ar_class(impl)
    # sglang's class may not accept max_size kwarg identically; try then fallback.
    try:
        ca = ARClass(group=gloo_group, device=device, max_size=max_size)
    except TypeError:
        ca = ARClass(group=gloo_group, device=device)

    if getattr(ca, "disabled", True):
        log(rank, f"FATAL: {impl} CustomAllreduce DISABLED on this setup.")
        dist.destroy_process_group()
        sys.exit(3)

    # call helper: aiter uses custom_all_reduce(input, use_new=...); sglang uses
    # custom_all_reduce(input). Probe signature.
    def do_ar(x):
        try:
            return ca.custom_all_reduce(x, use_new=use_new)
        except TypeError:
            return ca.custom_all_reduce(x)

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    token_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    def metric(out, ref):
        of, rf = out.float(), ref.float()
        nan_bad = ((torch.isnan(of) & ~torch.isnan(rf)) |
                   (torch.isinf(of) & ~torch.isinf(rf))).any().item()
        diff = (of - rf).abs()
        gross = ((diff > 0.1) & (diff > 0.25 * rf.abs())).sum().item()
        return nan_bad, gross, diff.max().item()

    total_nan = 0
    total_gross = 0
    first_fail = None

    for dname, dt in dtypes.items():
        elem = torch.tensor([], dtype=dt).element_size()
        for tok in token_list:
            inp_size = tok * hidden * elem
            if inp_size % 16 != 0 or inp_size > 8192 * 8192:
                continue

            static_inp = torch.zeros(tok, hidden, dtype=dt, device=device)

            # Capture exactly like sglang: ca.capture() wraps the graph capture,
            # AR runs on a graph-internal intermediate (x), out is graph-internal.
            graph = torch.cuda.CUDAGraph()
            captured_out = [None]

            # warmup outside capture (required before CUDAGraph capture)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    xw = static_inp * 1
                    _ = do_ar(xw)
            torch.cuda.current_stream().wait_stream(s)
            dist.barrier()

            ok = True
            try:
                with ca.capture():
                    with torch.cuda.graph(graph):
                        x = static_inp * 1
                        o = do_ar(x)
                        captured_out[0] = o
            except Exception as e:
                log(rank, f"[{dname}] tokens={tok}: capture failed: {e}")
                ok = False

            if not ok or captured_out[0] is None:
                total_gross += 1
                continue

            out_t = captured_out[0]
            case_nan = 0
            case_gross = 0
            max_abs = 0.0

            for it in range(iters):
                g = torch.Generator(device=device)
                g.manual_seed(seed_base + it * 100003 + tok * 31 + rank * 7)
                data = torch.randn(tok, hidden, dtype=dt, device=device, generator=g) * mag
                ref = data.clone()
                dist.all_reduce(ref, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

                static_inp.copy_(data)
                graph.replay()
                torch.cuda.synchronize()

                nan_bad, gross, abs_max = metric(out_t, ref)
                max_abs = max(max_abs, abs_max)
                if nan_bad:
                    case_nan += 1
                    if first_fail is None:
                        first_fail = (dname, tok, it, "NaN/Inf")
                elif gross > 0:
                    case_gross += 1
                    if first_fail is None:
                        first_fail = (dname, tok, it, f"gross {gross} elems abs={abs_max:.3f}")

            del graph
            stats = torch.tensor([case_nan, case_gross], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            cn, cg = int(stats[0]), int(stats[1])
            total_nan += cn
            total_gross += cg
            flag = "OK" if (cn == 0 and cg == 0) else (f"NaN/Inf x{cn}" if cn else f"GROSS x{cg}")
            log(rank, f"[{dname}] tokens={tok:>5} size={inp_size:>9}B max_abs={max_abs:.4f} -> {flag}")

    log(rank, "")
    log(rank, "============================================================")
    log(rank, f"  AR_IMPL={impl}  total_nan_inf={total_nan}  total_gross={total_gross}")
    if first_fail:
        log(rank, f"  first: {first_fail}")
    if total_nan == 0 and total_gross == 0:
        log(rank, f"  RESULT[{impl}]: clean through this harness.")
    else:
        log(rank, f"  RESULT[{impl}]: DIVERGED / NaN.")
    log(rank, "============================================================")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
