"""Shared Kimi-K3 GPU-kernel family classification."""


def classify_kernel(name: str) -> str:
    value = name.lower()
    rules = (
        (
            "collective",
            (
                "all_reduce",
                "allreduce",
                "quick_reduce",
                "quickreduce",
                "cross_device_reduce",
                "allgather",
                "rccl",
                "nccl",
            ),
        ),
        ("route_sort_topk", ("moe_sort", "sorting", "topk", "top_k", "route")),
        (
            "moe_gemm",
            (
                "f4gemm",
                "a16w4",
                "grouped_gemm",
                "groupedgemm",
                "moe_gemm",
                "mfma_moe1",
                "gemm1_a4w4",
                "gemm1_a8w4",
                "gemm1_a16w4",
                "gemm2_a4w4",
                "gemm2_a8w4",
                "gemm2_a16w4",
                "opus_moe",
                "stage1_a8w4",
                "stage2_a8w4",
            ),
        ),
        (
            "kda",
            (
                "gating_delta",
                "causal_conv",
                "kda_",
                "chunk_gla",
                "chunk_kda",
                "recompute_w_u",
            ),
        ),
        (
            "mla_attention",
            (
                "mla",
                "attention",
                "paged_attn",
                "paged_attention",
                "gqa_",
                "_fwd_kernel",
            ),
        ),
        ("attention_residual", ("attn_res", "_agg_kernel", "add3_kernel")),
        (
            "norm_activation",
            (
                "rmsnorm",
                "layernorm",
                "situ",
                "activation",
                "sigmoid",
                "silu",
                "l2norm",
            ),
        ),
        ("copy_fill", ("memcpy", "copy", "fill", "set_value", "catarray")),
        ("sampling", ("sample", "softmax", "argmax", "multinomial")),
    )
    for family, markers in rules:
        if any(marker in value for marker in markers):
            return family
    if any(marker in value for marker in ("quant", "scaled_quant", "fp8", "fp4")):
        if "gemm" not in value:
            return "quantization"
    if "gemm" in value or "matmul" in value or "cijk_" in value:
        return "dense_gemm"
    return "other"


def merge_intervals(intervals):
    """Return the covered duration for possibly overlapping intervals."""
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start
