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
    """Measure padded-to-multiple-of-16 Linear vs unaligned Linear on CUDA."""
    from torchbridge.backends.nvidia.nvidia_backend import _TensorCoreAlignedLinear

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use intentionally misaligned dims (1023, 511) near a multiple-of-16 boundary
    # where alignment padding provides meaningful TC benefit. Batch 256 for memory
    # pressure typical of real inference.
    original = nn.Linear(1023, 511).to(device)
    original.eval()
    aligned = _TensorCoreAlignedLinear(
        nn.Linear(1023, 511).to(device), optimal_multiple=16
    )
    aligned.eval()
    x = torch.randn(256, 1023, device=device)

    return ClaimBenchmark(
        name="tensor_core_alignment",
        baseline_fn=lambda: original(x),
        optimized_fn=lambda: aligned(x),
        warmup=10,
        runs=50,
        threshold_pct=3.0,
        requires_backend="cuda",
        description=(
            "Padding Linear weight dims to multiples-of-16 increases GEMM "
            "throughput on NVIDIA tensor cores (5-25% speedup, GPU only)."
        ),
        notes=[
            "Measures GEMM throughput for aligned (pad to 16) vs unaligned Linear.",
            "Benefit requires NVIDIA tensor cores — padding adds overhead on CPU/MPS.",
            "Expected GPU speedup: 5-25% for weight sizes near a multiple-of-16 boundary.",
            "Literature: NVIDIA cuBLAS docs — GEMM throughput scales with alignment.",
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
        description=(
            "NHWC (channels_last) memory layout eliminates NCHW→NHWC transposes "
            "in cuDNN Conv2d kernels (10-30% speedup on Ampere/Ada, GPU only)."
        ),
        notes=[
            "Measures NHWC vs NCHW for Conv2d workloads.",
            "Benefit is CUDA-specific. CPU and MPS show near-zero or negative.",
            "Expected GPU speedup: 10-30% on Ampere/Ada for typical CNN workloads.",
            "Literature: NVIDIA cuDNN — NHWC is the native format; NCHW requires transposes.",
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
# PYTORCH_TUNABLEOP_ENABLED=1 takes effect at kernel selection time and requires
# a process restart to measure. In-process benchmarking is not feasible —
# baseline_fn and optimized_fn would run the same un-tuned kernels, producing
# ~0% delta regardless of hardware.

def build_batch_throughput_benchmark() -> ClaimBenchmark:
    """Measure throughput gain of batched vs sequential model.generate().

    Simulates 4 concurrent inference requests: sequential processes them one
    at a time (4 separate generate calls); batched packs all 4 into a single
    generate call with left-padded inputs.

    Requires CUDA — batching benefit is negligible on CPU.
    """
    # Small sequence lengths to keep the benchmark fast on any GPU.
    batch_size = 4
    max_new_tokens = 8

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", torch_dtype=torch.float16
        ).cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        prompts = ["The quick brown fox"] * batch_size
        inputs = [
            tokenizer(p, return_tensors="pt").input_ids.cuda() for p in prompts
        ]

        # Batch: pad left, single generate call
        max_len = max(x.size(1) for x in inputs)
        pad_id = tokenizer.pad_token_id
        padded = [
            torch.nn.functional.pad(x, (max_len - x.size(1), 0), value=pad_id)
            for x in inputs
        ]
        batch_ids = torch.cat(padded, dim=0)
        attn_mask = (batch_ids != pad_id).long()

        def sequential_fn():
            for x in inputs:
                model.generate(x, max_new_tokens=max_new_tokens, do_sample=False)

        def batched_fn():
            model.generate(
                batch_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        skip_reason = None
        baseline_fn = sequential_fn
        optimized_fn = batched_fn

    except Exception as e:
        # CUDA unavailable or model not cached — skip cleanly
        skip_reason = f"CUDA or model unavailable: {e}"
        dummy = torch.zeros(1)

        def baseline_fn():
            return dummy + 1

        def optimized_fn():
            return dummy + 1

    return ClaimBenchmark(
        name="batch_throughput",
        baseline_fn=baseline_fn,
        optimized_fn=optimized_fn,
        warmup=2,
        runs=10,
        threshold_pct=30.0,  # batching 4 requests should be ≥30% faster than sequential
        requires_backend="cuda",
        skip_reason=skip_reason,
        description=(
            "Dynamic batching: 4 concurrent requests in one model.generate() call "
            "vs 4 sequential calls. Expected ≥30% throughput improvement on GPU."
        ),
        notes=[
            "Measures wall-clock time for batch_size=4 sequential vs batched generation.",
            "Batching amortizes KV-cache setup and GPU kernel launch overhead.",
            "Requires CUDA — batch benefit is negligible on CPU.",
            "Benchmark uses left-padded inputs (correct for causal LMs).",
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
        build_batch_throughput_benchmark(),
    ]


def build_claim_suite() -> BenchmarkSuite:
    """Build a BenchmarkSuite with all registered claims."""
    suite = BenchmarkSuite()
    for bench in get_all_claim_benchmarks():
        suite.add(bench)
    return suite
