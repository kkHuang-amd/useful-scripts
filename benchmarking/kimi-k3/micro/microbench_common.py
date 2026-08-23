"""Common HIP/CUDA graph timing and numerical helpers for Kimi-K3 micros."""

import statistics
import time

import torch


def capture_graph(run, *, warmup=10):
    """Warm up and capture ``run``; setup and capture are never timed."""
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    return graph


def benchmark_graph_replay(graph, *, warmup=10, iterations=50):
    """Return one HIP/CUDA-event latency sample per graph replay."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]


def latency_summary(samples):
    """Summarize a non-empty sequence of millisecond samples."""
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("latency samples must not be empty")
    ordered = sorted(values)
    return {
        "sample_count": len(values),
        "min_ms": ordered[0],
        "p50_ms": statistics.median(ordered),
        "p90_ms": ordered[max(0, (9 * len(ordered) + 9) // 10 - 1)],
        "max_ms": ordered[-1],
        "mean_ms": statistics.fmean(ordered),
    }


def capture_and_bench(run, *, warmup=10, iterations=50):
    graph = capture_graph(run, warmup=warmup)
    return statistics.fmean(
        benchmark_graph_replay(graph, warmup=warmup, iterations=iterations)
    )


def eager_bench(run, *, warmup=10, iterations=50):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3 / iterations


def rel_l2(candidate, reference):
    return (
        (candidate.float() - reference.float()).norm()
        / reference.float().norm()
    ).item()
