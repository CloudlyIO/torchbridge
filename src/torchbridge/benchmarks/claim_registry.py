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

# ── Claim 1: Attention Dispatch Overhead ──────────────────────────────────────


def build_attention_dispatch_benchmark() -> ClaimBenchmark:
    """Measure dispatch overhead — should be negligible (<5% vs direct SDPA call).

    Tensors are pre-created outside both timed functions so that tensor-creation
    noise (~5ms) does not swamp the dispatch overhead (<0.1ms) being measured.
    """
    from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

    dispatcher = AttentionDispatcher(use_benchmark_cache=False)
    # Pre-create tensors once — reused across all timed iterations
    q = torch.randn(4, 8, 512, 64)
    k = torch.randn(4, 8, 512, 64)
    v = torch.randn(4, 8, 512, 64)

    def baseline():
        """Direct SDPA call — no dispatch overhead."""
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def optimized():
        """Dispatch decision + SDPA — overhead is the select_kernel() call."""
        dispatcher.select_kernel(seq_length=512, num_heads=8, head_dim=64)
        torch.nn.functional.scaled_dot_product_attention(q, k, v)

    return ClaimBenchmark(
        name="attention_dispatch_overhead",
        baseline_fn=baseline,
        optimized_fn=optimized,
        warmup=5,
        runs=30,
        threshold_pct=-5.0,  # Negative threshold: PASS if overhead < 5%
        description=(
            "AttentionDispatcher kernel-selection adds <5% overhead vs direct "
            "SDPA call. Verifiable on CPU; kernel speedup requires GPU."
        ),
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
    # Use large Linear layers so INT8 GEMM savings dominate quantization overhead.
    # Small models (≤512 dim) show no speedup — overhead > savings.
    model_fp32 = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
    )
    model_fp32.eval()

    # Explicit CPU placement — INT8 dynamic quantization is a CPU-FBGEMM optimization.
    model_fp32 = model_fp32.cpu()
    x = torch.randn(128, 2048)  # batch=128, large input to stress GEMM

    skip_reason: str | None = None
    try:
        model_int8 = torch.ao.quantization.quantize_dynamic(
            nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
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
        description=(
            "INT8 dynamic quantization (FBGEMM) reduces Linear compute by 10-40% "
            "on x86 CPU. TorchBridge auto-selects this format per backend."
        ),
        notes=[
            "Measures INT8 dynamic quantization speedup on CPU (FBGEMM backend).",
            "This is a well-established PyTorch optimization — expected 10-40% speedup.",
            "TorchBridge's value: auto-selecting this format per backend.",
            "Requires FBGEMM (Linux x86_64) — SKIPPED on other platforms.",
        ],
    )


# NOTE: AMD TunableOp benchmark was removed (Benchmark-or-Delete rule).
# NOTE: tensor_core_alignment and channels_last benchmarks were removed — claimed
#       speedups were literature estimates, not measured TorchBridge results.
# NOTE: batch_throughput benchmark was removed — it measured vanilla HuggingFace
#       model.generate() batching, which is not a TorchBridge optimization.

# ── Registry ──────────────────────────────────────────────────────────────────


def get_all_claim_benchmarks() -> list[ClaimBenchmark]:
    """Return all registered claim benchmarks."""
    return [
        build_attention_dispatch_benchmark(),
        build_quantization_speedup_benchmark(),
    ]


def build_claim_suite() -> BenchmarkSuite:
    """Build a BenchmarkSuite with all registered claims."""
    suite = BenchmarkSuite()
    for bench in get_all_claim_benchmarks():
        suite._benchmarks.append(bench)
    return suite
