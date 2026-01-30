"""
E2E Test Configuration and Fixtures

Provides shared fixtures for end-to-end tests with real models.
These tests load actual HuggingFace/torchvision models and validate
that optimizations produce measurable speedups without accuracy loss.

Markers:
    - @pytest.mark.e2e: End-to-end test (>30 seconds)
    - @pytest.mark.slow: Long-running test
    - @pytest.mark.real_model: Uses real model weights
    - @pytest.mark.requires_transformers: Requires HuggingFace transformers
    - @pytest.mark.requires_torchvision: Requires torchvision
    - @pytest.mark.gpu: Requires CUDA GPU
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

# =============================================================================
# Skip Conditions
# =============================================================================

def _check_transformers():
    """Check if transformers is available."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _check_torchvision():
    """Check if torchvision is available."""
    try:
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def _check_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


# Skip decorators for conditional tests
requires_transformers = pytest.mark.skipif(
    not _check_transformers(),
    reason="Requires HuggingFace transformers library"
)

requires_torchvision = pytest.mark.skipif(
    not _check_torchvision(),
    reason="Requires torchvision library"
)

requires_cuda = pytest.mark.skipif(
    not _check_cuda(),
    reason="Requires CUDA GPU"
)


# =============================================================================
# Benchmark Utilities
# =============================================================================

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    num_runs: int
    warmup_runs: int

    @property
    def mean_time_s(self) -> float:
        return self.mean_time_ms / 1000.0

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(mean={self.mean_time_ms:.2f}ms, "
            f"std={self.std_time_ms:.2f}ms, "
            f"runs={self.num_runs})"
        )


def benchmark_function(
    fn: Callable,
    *args,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    sync_cuda: bool = True,
    **kwargs
) -> BenchmarkResult:
    """
    Benchmark a function with warmup and multiple runs.

    Args:
        fn: Function to benchmark
        *args: Positional arguments for the function
        warmup_runs: Number of warmup runs (not counted)
        benchmark_runs: Number of benchmark runs
        sync_cuda: Whether to synchronize CUDA before timing
        **kwargs: Keyword arguments for the function

    Returns:
        BenchmarkResult with timing statistics
    """
    import statistics

    # Warmup
    for _ in range(warmup_runs):
        fn(*args, **kwargs)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(benchmark_runs):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        fn(*args, **kwargs)

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return BenchmarkResult(
        mean_time_ms=statistics.mean(times),
        std_time_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        num_runs=benchmark_runs,
        warmup_runs=warmup_runs
    )


def calculate_speedup(
    baseline: BenchmarkResult,
    optimized: BenchmarkResult
) -> float:
    """
    Calculate speedup ratio.

    Args:
        baseline: Benchmark result for baseline model
        optimized: Benchmark result for optimized model

    Returns:
        Speedup ratio (>1 means optimized is faster)
    """
    return baseline.mean_time_ms / optimized.mean_time_ms


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def e2e_device():
    """Get the best available device for e2e testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Get CUDA device or skip test."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# =============================================================================
# Model Loading Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def bert_model_and_tokenizer():
    """Load real BERT model and tokenizer (session-scoped to avoid re-downloading)."""
    if not _check_transformers():
        pytest.skip("transformers not available")

    from transformers import AutoModel, AutoTokenizer

    model_name = "bert-base-uncased"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Failed to load BERT model: {e}")
    model.eval()

    return model, tokenizer


@pytest.fixture(scope="session")
def gpt2_model_and_tokenizer():
    """Load real GPT-2 model and tokenizer (session-scoped to avoid re-downloading)."""
    if not _check_transformers():
        pytest.skip("transformers not available")

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    model_name = "gpt2"
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Failed to load GPT-2 model: {e}")
    model.eval()

    return model, tokenizer


@pytest.fixture(scope="session")
def resnet50_model():
    """Load real ResNet-50 model (session-scoped)."""
    if not _check_torchvision():
        pytest.skip("torchvision not available")

    import ssl
    import urllib.error

    from torchvision.models import ResNet50_Weights, resnet50

    try:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.eval()
        return model
    except (urllib.error.URLError, ssl.SSLError, OSError) as e:
        pytest.skip(f"Cannot download ResNet-50 weights: {e}")


@pytest.fixture(scope="session")
def clip_model_and_processor():
    """Load real CLIP model and processor (session-scoped)."""
    if not _check_transformers():
        pytest.skip("transformers not available")

    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    try:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    except Exception as e:
        pytest.skip(f"Failed to load CLIP model: {e}")
    model.eval()

    return model, processor


# =============================================================================
# Sample Input Fixtures
# =============================================================================

@pytest.fixture
def sample_text_inputs():
    """Sample text inputs for NLP models."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of technology.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can learn complex patterns from data.",
    ]


@pytest.fixture
def sample_image_tensor():
    """Sample image tensor for vision models (batch of 4, 224x224 RGB)."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_pil_images():
    """Sample PIL images for multimodal models."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        pytest.skip("PIL not available")

    images = []
    for _ in range(4):
        # Create random RGB images
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr))

    return images


# =============================================================================
# Tolerance Fixtures
# =============================================================================

@pytest.fixture
def output_tolerance():
    """
    Default tolerance for output comparison.

    NOTE: These tolerances account for mixed precision (BF16/FP16) optimizations.
    Real-world models with BF16 can have differences of 0.1-0.5 in logits, which
    is expected and acceptable for inference quality.
    """
    return {
        "atol": 0.5,   # Absolute tolerance - realistic for BF16 mixed precision
        "rtol": 0.1,   # Relative tolerance - 10% relative difference allowed
    }


@pytest.fixture
def strict_tolerance():
    """Strict tolerance for FP32-only output comparison."""
    return {
        "atol": 1e-4,
        "rtol": 1e-4,
    }


@pytest.fixture
def fp32_tolerance():
    """Tolerance for FP32 precision comparisons (no mixed precision)."""
    return {
        "atol": 1e-3,
        "rtol": 1e-3,
    }


# =============================================================================
# Speedup Assertion Helpers
# =============================================================================

def assert_speedup(
    baseline: BenchmarkResult,
    optimized: BenchmarkResult,
    min_speedup: float = 1.0,
    message: str = ""
) -> float:
    """
    Assert that optimized version achieves minimum speedup.

    Args:
        baseline: Baseline benchmark result
        optimized: Optimized benchmark result
        min_speedup: Minimum required speedup (1.0 = no slowdown, 1.2 = 20% faster)
        message: Optional message for assertion

    Returns:
        Actual speedup achieved

    Raises:
        AssertionError: If speedup is below minimum
    """
    speedup = calculate_speedup(baseline, optimized)

    if speedup < min_speedup:
        raise AssertionError(
            f"{message}\n"
            f"Expected speedup >= {min_speedup}x, got {speedup:.2f}x\n"
            f"Baseline: {baseline.mean_time_ms:.2f}ms\n"
            f"Optimized: {optimized.mean_time_ms:.2f}ms"
        )

    return speedup


def assert_output_close(
    baseline_output: torch.Tensor,
    optimized_output: torch.Tensor,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    message: str = ""
) -> None:
    """
    Assert that optimized output is close to baseline.

    Args:
        baseline_output: Output from baseline model
        optimized_output: Output from optimized model
        atol: Absolute tolerance
        rtol: Relative tolerance
        message: Optional message for assertion

    Raises:
        AssertionError: If outputs differ beyond tolerance
    """
    # Convert to same dtype and device for comparison
    baseline = baseline_output.detach().float().cpu()
    optimized = optimized_output.detach().float().cpu()

    if not torch.allclose(baseline, optimized, atol=atol, rtol=rtol):
        max_diff = (baseline - optimized).abs().max().item()
        raise AssertionError(
            f"{message}\n"
            f"Outputs differ beyond tolerance (atol={atol}, rtol={rtol})\n"
            f"Max difference: {max_diff}"
        )
