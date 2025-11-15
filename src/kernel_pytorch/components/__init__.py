"""
Kernel-Optimized PyTorch Components

This package provides neural network components optimized at different levels
to demonstrate the progression from basic PyTorch operations to custom CUDA kernels.
"""

from .basic_optimized import (
    OptimizedLinear,
    FusedLinearActivation,
    OptimizedLayerNorm,
    OptimizedMultiHeadAttention,
    OptimizedMLP,
    OptimizedTransformerBlock,
    PositionalEncoding,
    SimpleTransformer
)

from .jit_optimized import (
    JITOptimizedLinear,
    JITOptimizedLayerNorm,
    JITOptimizedMLP,
    JITRotaryAttention,
    FullyJITTransformerBlock,
    JITOptimizedTransformer
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