#!/usr/bin/env python3
"""
Isolated correctness test for aiter's FUSED all-reduce + RMSNorm (Phase 3, v4).

This is the path the model actually uses with aiter AR enabled (sglang
parallel_state.fused_allreduce_rmsnorm -> ca_comm.custom_fused_ar_rms), which
has an aiter-specific 1-stage small-batch kernel (<=128KB, kMaxBlocks=80 tokens)
that sglang's native AR does NOT have. Plain custom_all_reduce already tested
clean, so this targets the fused path.

custom_fused_ar_rms(input, residual, weight, eps, use_1stage) -> (out, res_out)
  res_out = allreduce_sum(input) + residual
  out     = rmsnorm(res_out, weight, eps)

Reference: torch all_reduce(input) + residual, then rmsnorm.

Launch on 8-GPU node:
    torchrun --nproc_per_node=8 aiter_fused_ar_rms_test.py
Env: ITERS(100) HIDDEN(7168) STAGE(auto|1|2) MAG(2.0) EPS(1e-6)
"""
import os
import sys

import torch
import torch.distributed as dist


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def ref_rmsnorm(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.rsqrt(var + eps)
    return (xn * w.float())


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    iters = int(os.environ.get("ITERS", "100"))
    hidden = int(os.environ.get("HIDDEN", "7168"))
    stage_mode = os.environ.get("STAGE", "auto")  # auto|1|2
    mag = float(os.environ.get("MAG", "2.0"))
    eps = float(os.environ.get("EPS", "1e-6"))
    seed_base = int(os.environ.get("SEED_BASE", "1234"))
    max_size = int(os.environ.get("MAX_SIZE", str(1024 * 1024 * 1024)))

    dist.init_process_group(backend="nccl")
    gloo_group = dist.new_group(backend="gloo")

    log(rank, "=== aiter FUSED all-reduce + RMSNorm correctness test (v4) ===")
    log(rank, f"world_size={world_size} hidden={hidden} stage={stage_mode} mag={mag} eps={eps} iters={iters}")

    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    ca = CustomAllreduce(group=gloo_group, device=device, max_size=max_size)
    if getattr(ca, "disabled", True) or not hasattr(ca, "custom_fused_ar_rms"):
        log(rank, "FATAL: aiter custom_fused_ar_rms unavailable/disabled.")
        dist.destroy_process_group()
        sys.exit(3)

    dt = torch.bfloat16
    # 1-stage regime is small batches (<=80 tokens, <=128KB). Sweep around it.
    token_list = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 79, 80, 81, 96, 128]

    def stages_for(tok):
        total_bytes = tok * hidden * dt.itemsize if hasattr(dt, "itemsize") else tok * hidden * 2
        if stage_mode == "1":
            return [True]
        if stage_mode == "2":
            return [False]
        # auto: mimic sglang threshold (<=128KB -> 1-stage), but also try both
        return [True, False]

    total_nan = 0
    total_gross = 0
    first_fail = None

    for tok in token_list:
        inp_size = tok * hidden * 2
        if inp_size % 16 != 0 or inp_size > 8192 * 8192:
            continue
        for use_1stage in stages_for(tok):
            # 1-stage is documented capped at 80 tokens; record but still try.
            case_nan = 0
            case_gross = 0
            max_abs = 0.0
            errored = False

            for it in range(iters):
                gi = torch.Generator(device=device); gi.manual_seed(seed_base + it * 9173 + tok * 31 + rank * 7)
                gs = torch.Generator(device=device); gs.manual_seed(seed_base + it * 7919 + tok * 17)  # shared (rank-independent)
                inp = torch.randn(tok, hidden, dtype=dt, device=device, generator=gi) * mag
                # residual + weight are replicated across TP ranks -> rank-independent seed
                residual = torch.randn(tok, hidden, dtype=dt, device=device, generator=gs) * mag
                weight = torch.randn(hidden, dtype=dt, device=device, generator=gs) * 0.1 + 1.0

                # reference
                ar = inp.clone()
                dist.all_reduce(ar, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
                res_ref = (ar.float() + residual.float())
                out_ref = ref_rmsnorm(res_ref.to(dt), weight, eps)

                try:
                    res = ca.custom_fused_ar_rms(inp, residual, weight, eps, use_1stage)
                except Exception as e:
                    errored = True
                    if first_fail is None:
                        first_fail = (tok, use_1stage, it, f"exception: {e}")
                    break
                if res is None:
                    case_gross += 1
                    continue
                out, res_out = res
                torch.cuda.synchronize()

                of = out.float()
                nan_bad = (torch.isnan(of) | torch.isinf(of)).any().item() and not \
                          (torch.isnan(out_ref) | torch.isinf(out_ref)).any().item()
                diff = (of - out_ref).abs()
                # rmsnorm output is ~O(1)*weight; use absolute + relative guard
                gross = ((diff > 0.05) & (diff > 0.1 * out_ref.abs())).sum().item()
                max_abs = max(max_abs, diff.max().item())
                if nan_bad:
                    case_nan += 1
                    if first_fail is None:
                        first_fail = (tok, use_1stage, it, "NaN/Inf in fused output")
                elif gross > 0:
                    case_gross += 1
                    if first_fail is None:
                        first_fail = (tok, use_1stage, it, f"gross {gross} elems abs={diff.max().item():.3f}")

            ev = torch.tensor([case_nan, case_gross, 1 if errored else 0], device=device)
            dist.all_reduce(ev, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
            cn, cg, ce = int(ev[0]), int(ev[1]), int(ev[2])
            total_nan += cn
            total_gross += cg
            tag = f"1stage" if use_1stage else "2stage"
            flag = "OK"
            if ce:
                flag = f"EXCEPTION(x{ce})"
            elif cn:
                flag = f"NaN/Inf x{cn}"
            elif cg:
                flag = f"GROSS x{cg}"
            log(rank, f"tokens={tok:>4} {tag} size={inp_size:>8}B max_abs={max_abs:.4f} -> {flag}")

    log(rank, "")
    log(rank, "============================================================")
    log(rank, f"  total_nan_inf={total_nan}  total_gross={total_gross}")
    if first_fail:
        log(rank, f"  first failure: tokens={first_fail[0]} 1stage={first_fail[1]} iter={first_fail[2]} -> {first_fail[3]}")
    if total_nan == 0 and total_gross == 0:
        log(rank, "  RESULT: aiter fused AR+RMSNorm matched reference (no repro).")
    else:
        log(rank, "  RESULT: aiter fused AR+RMSNorm DIVERGED / NaN (repro!).")
    log(rank, "============================================================")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
