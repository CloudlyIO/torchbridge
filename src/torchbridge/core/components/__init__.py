"""
Core components removed in v0.5.56 (Rule 1 — No-Wrapper violations).

OptimizedLinear, OptimizedLayerNorm, FusedLinearActivation, JIT variants, etc.
all reduced to single-call wrappers around F.linear(), F.layer_norm(), etc.
Use nn.Linear, nn.LayerNorm, and torch.compile for fusion directly.
"""

__all__: list[str] = []
