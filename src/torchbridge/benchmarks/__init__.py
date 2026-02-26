"""
TorchBridge Benchmark Infrastructure

Provides claim-level benchmarking: measure every performance claim against
a vanilla PyTorch baseline, report speedup, and flag claims that don't meet
the ≥3% improvement threshold for deletion.
"""

from torchbridge.benchmarks.claim_benchmarks import (
    BenchmarkReport,
    BenchmarkSuite,
    ClaimBenchmark,
    ClaimResult,
)
from torchbridge.benchmarks.claim_registry import (
    build_claim_suite,
    get_all_claim_benchmarks,
)

__all__ = [
    "ClaimBenchmark",
    "ClaimResult",
    "BenchmarkSuite",
    "BenchmarkReport",
    "build_claim_suite",
    "get_all_claim_benchmarks",
]
