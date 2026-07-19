# SPDX-License-Identifier: Apache-2.0
"""
Backend-Aware Attention Kernel Dispatch

Auto-selects the optimal attention kernel for the detected hardware
with ordered fallback chains.
"""

from .benchmark_cache import BenchmarkEntry, BenchmarkFingerprint, KernelBenchmarkCache
from .compatibility import AttentionDispatchMatrix
from .dispatcher import AttentionDispatcher, AttentionDispatchResult
from .kernel_types import AttentionKernelType

__all__ = [
    "AttentionKernelType",
    "AttentionDispatchMatrix",
    "AttentionDispatcher",
    "AttentionDispatchResult",
    "KernelBenchmarkCache",
    "BenchmarkEntry",
    "BenchmarkFingerprint",
]
