"""
Kernel-Optimized PyTorch Components

This package provides neural network components optimized at different levels
to demonstrate the progression from basic PyTorch operations to custom CUDA kernels.
"""

from .basic_optimized import (
    FusedLinearActivation,
    OptimizedLayerNorm,
    OptimizedLinear,
    OptimizedMLP,
    OptimizedMultiHeadAttention,
    OptimizedTransformerBlock,
    PositionalEncoding,
    SimpleTransformer,
)
from .jit_optimized import (
    FullyJITTransformerBlock,
    JITOptimizedLayerNorm,
    JITOptimizedLinear,
    JITOptimizedMLP,
    JITOptimizedTransformer,
    JITRotaryAttention,
)

__all__ = [
    # Basic optimized components
    'OptimizedLinear',
    'FusedLinearActivation',
    'OptimizedLayerNorm',
    'OptimizedMultiHeadAttention',
    'OptimizedMLP',
    'OptimizedTransformerBlock',
    'PositionalEncoding',
    'SimpleTransformer',

    # JIT optimized components
    'JITOptimizedLinear',
    'JITOptimizedLayerNorm',
    'JITOptimizedMLP',
    'JITRotaryAttention',
    'FullyJITTransformerBlock',
    'JITOptimizedTransformer'
]
