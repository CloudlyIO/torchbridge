"""
Attention core — removed in v0.5.57.

BaseAttention, AttentionModuleConfig, the attention registry, and attention_ops
wrapper functions violated Rule 1 (no-wrapper) and had no legitimate selection
logic separate from the implementations they served.

For backend-aware kernel selection use:
  torchbridge.attention.AttentionDispatcher.select_kernel()
"""

__all__: list[str] = []
