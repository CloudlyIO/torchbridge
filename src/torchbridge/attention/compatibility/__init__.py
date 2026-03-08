"""
Attention backward-compatibility layer — removed in v0.5.57.

This module re-exported FlashAttention2, FlashAttention3, MemoryEfficientAttention,
and DynamicSparseAttention — all of which were Rule 1 violations deleted in v0.5.57.
The backward-compat layer has no surviving targets and is therefore removed.

For backend-aware kernel selection use:
  torchbridge.attention.AttentionDispatcher.select_kernel()
"""

__all__: list[str] = []
