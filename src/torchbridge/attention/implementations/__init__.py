"""
Attention implementations — removed in v0.5.57.

These wrapper classes (FlashAttention2, FlashAttention3, MemoryEfficientAttention,
ChunkedAttention, LongSequenceAttention, DynamicSparseAttention) violated Rule 1
(no-wrapper): each was a thin nn.Module around a single PyTorch or flash_attn call.

For attention computation use:
  torch.nn.functional.scaled_dot_product_attention  (built-in, hardware-optimised)
  flash_attn.flash_attn_func                        (if flash_attn is installed)

For backend-aware kernel selection use:
  torchbridge.attention.AttentionDispatcher.select_kernel()
"""

__all__: list[str] = []
