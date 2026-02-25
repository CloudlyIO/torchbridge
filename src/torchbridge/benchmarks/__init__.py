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

__all__ = [
    "ClaimBenchmark",
    "ClaimResult",
    "BenchmarkSuite",
    "BenchmarkReport",
]
