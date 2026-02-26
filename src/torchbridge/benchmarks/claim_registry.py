"""
Claim Registry — Concrete Benchmarks for TorchBridge Performance Claims

Each function builds a ClaimBenchmark that compares a vanilla PyTorch baseline
against the TorchBridge-optimized path. Claims that don't show ≥3% improvement
across runs are candidates for deletion per the Benchmark-or-Delete rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from torchbridge.benchmarks.claim_benchmarks import BenchmarkSuite, ClaimBenchmark


# ── Claim 1: NVIDIA Tensor Core Alignment ────────────────────────────────────


def build_tensor_core_alignment_benchmark() -> ClaimBenchmark:
    """Measure padded-to-multiple-of-16 Linear vs unaligned Linear."""
    from torchbridge.backends.nvidia.nvidia_backend import _TensorCoreAlignedLinear

    original = nn.Linear(127, 63)
    original.eval()
    aligned = _TensorCoreAlignedLinear(original, optimal_multiple=16)
    aligned.eval()
    x = torch.randn(32, 127)

    return ClaimBenchmark(
        name="tensor_core_alignment",
        baseline_fn=lambda: original(x),
        optimized_fn=lambda: aligned(x),
        warmup=10,
        runs=50,
        threshold_pct=3.0,
        requires_backend="cuda",
        notes=[
            "Measures GEMM throughput for aligned (pad to 16) vs unaligned Linear.",
            "Benefit requires NVIDIA tensor cores — padding adds overhead on CPU/MPS.",
            "Will be SKIPPED on non-CUDA hardware.",
        ],
    )


# ── Claim 2: channels_last Memory Layout ─────────────────────────────────────


def build_channels_last_benchmark() -> ClaimBenchmark:
    """Measure channels_last (NHWC) vs contiguous (NCHW) for Conv2d."""
    model_nchw = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
    )
    model_nchw.eval()

    model_nhwc = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
    )
    model_nhwc.eval()
    model_nhwc = model_nhwc.to(memory_format=torch.channels_last)

    x_nchw = torch.randn(16, 64, 56, 56)
    x_nhwc = x_nchw.to(memory_format=torch.channels_last)

    return ClaimBenchmark(
        name="channels_last_layout",
        baseline_fn=lambda: model_nchw(x_nchw),
        optimized_fn=lambda: model_nhwc(x_nhwc),
        warmup=5,
        runs=30,
        threshold_pct=3.0,
        requires_backend="cuda",
        notes=[
            "Measures NHWC vs NCHW for Conv2d workloads.",
            "Benefit is CUDA-specific (10-30%). CPU and MPS show near-zero or negative.",
            "Will be SKIPPED on non-CUDA hardware.",
        ],
    )


# ── Claim 3: Attention Dispatch Overhead ──────────────────────────────────────


def build_attention_dispatch_benchmark() -> ClaimBenchmark:
    """Measure dispatch overhead — should be negligible (<1ms)."""
    from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

    dispatcher = AttentionDispatcher(use_benchmark_cache=False)

    def baseline():
        """Direct SDPA call — no dispatch overhead."""
        q = torch.randn(4, 8, 512, 64)
        k = torch.randn(4, 8, 512, 64)
        v = torch.randn(4, 8, 512, 64)
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def optimized():
        """Dispatch + SDPA — overhead is the dispatch decision cost."""
        dispatcher.select_kernel(seq_length=512, num_heads=8, head_dim=64)
        q = torch.randn(4, 8, 512, 64)
        k = torch.randn(4, 8, 512, 64)
        v = torch.randn(4, 8, 512, 64)
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    return ClaimBenchmark(
        name="attention_dispatch_overhead",
        baseline_fn=baseline,
        optimized_fn=optimized,
        warmup=5,
        runs=30,
        threshold_pct=-5.0,  # Negative threshold: PASS if overhead < 5%
        notes=[
            "Measures dispatch decision overhead, NOT kernel speedup.",
            "The dispatch call should add <1ms overhead to the SDPA call.",
            "Kernel-level speedup (FlexAttention vs SDPA) requires GPU hardware.",
            "Negative threshold: passes if optimized is no more than 5% slower.",
        ],
    )


# ── Claim 4: Quantization Speedup (INT8 Dynamic) ─────────────────────────────


def build_quantization_speedup_benchmark() -> ClaimBenchmark:
    """Measure INT8 dynamic quantization speedup on CPU.

    Falls back to a no-op benchmark if the quantization engine (FBGEMM) is
    unavailable (e.g., on macOS without FBGEMM support).
    """
    model_fp32 = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    model_fp32.eval()

    x = torch.randn(32, 512)

    skip_reason: str | None = None
    try:
        model_int8 = torch.ao.quantization.quantize_dynamic(
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            ),
            {nn.Linear},
            dtype=torch.qint8,
        )
        model_int8.eval()
        optimized_fn: torch.nn.Module = model_int8
    except (RuntimeError, NotImplementedError):
        # FBGEMM not available (macOS, non-x86) — skip instead of running
        # identity functions that produce a misleading FAIL result.
        optimized_fn = model_fp32
        skip_reason = "FBGEMM not available on this platform (requires Linux x86_64)"

    return ClaimBenchmark(
        name="quantization_int8_dynamic",
        baseline_fn=lambda: model_fp32(x),
        optimized_fn=lambda: optimized_fn(x),
        warmup=10,
        runs=50,
        threshold_pct=3.0,
        skip_reason=skip_reason,
        notes=[
            "Measures INT8 dynamic quantization speedup on CPU (FBGEMM backend).",
            "This is a well-established PyTorch optimization — expected 10-40% speedup.",
            "TorchBridge's value: auto-selecting this format per backend.",
            "Requires FBGEMM (Linux x86_64) — SKIPPED on other platforms.",
        ],
    )


# ── Claim 5: AMD TunableOp ───────────────────────────────────────────────────


def build_tunableop_benchmark() -> ClaimBenchmark:
    """Placeholder — requires AMD hardware to measure."""
    # Cannot benchmark TunableOp on non-AMD hardware.
    # The benchmark is registered but will be skipped on non-AMD systems.
    model = nn.Linear(256, 128)
    model.eval()
    x = torch.randn(32, 256)

    return ClaimBenchmark(
        name="amd_tunableop",
        baseline_fn=lambda: model(x),
        optimized_fn=lambda: model(x),
        warmup=3,
        runs=10,
        threshold_pct=3.0,
        requires_backend="rocm",
        notes=[
            "Requires AMD ROCm hardware to measure TunableOp benefit.",
            "PYTORCH_TUNABLEOP_ENABLED=1 auto-tunes GEMM kernels on CDNA2+.",
            "Expected 5-15% speedup on MI300X after warmup tuning.",
            "Will be SKIPPED on non-AMD systems.",
        ],
    )


# ── Registry ──────────────────────────────────────────────────────────────────


def get_all_claim_benchmarks() -> list[ClaimBenchmark]:
    """Return all registered claim benchmarks."""
    return [
        build_tensor_core_alignment_benchmark(),
        build_channels_last_benchmark(),
        build_attention_dispatch_benchmark(),
        build_quantization_speedup_benchmark(),
        build_tunableop_benchmark(),
    ]


def build_claim_suite() -> BenchmarkSuite:
    """Build a BenchmarkSuite with all registered claims."""
    suite = BenchmarkSuite()
    for bench in get_all_claim_benchmarks():
        suite.add(bench)
    return suite
