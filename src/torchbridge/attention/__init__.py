"""
Attention Dispatch Framework

Backend-aware kernel selection for attention operations.
TorchBridge identifies the optimal kernel (FlashAttention-3, SDPA, Pallas, etc.)
for the given hardware; callers use the result to configure their own attention calls.
"""

from .dispatch import (
    AttentionDispatcher,
    AttentionDispatchMatrix,
    AttentionDispatchResult,
    AttentionKernelType,
    KernelBenchmarkCache,
)

__all__ = [
    "AttentionKernelType",
    "AttentionDispatchMatrix",
    "AttentionDispatcher",
    "AttentionDispatchResult",
    "KernelBenchmarkCache",
]
