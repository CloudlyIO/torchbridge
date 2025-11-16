"""
Compiler-Optimized PyTorch Components

This module contains neural network components specifically designed for maximum
GPU performance through PyTorch compiler optimization (torch.compile, TorchScript, etc.).

Key Design Principles:
- Tensor-native operations that map efficiently to GPU kernels
- Compiler-friendly patterns that enable automatic optimization
- Memory-efficient designs that minimize GPU memory bandwidth
- Validation frameworks to ensure correctness and measure performance

Components:
- attention_modules: Various attention implementations optimized for compilation
- normalization_layers: Layer normalization variants with kernel optimization
- activation_functions: Fused activation patterns for better performance
- linear_transformations: Optimized linear layers and projections
"""

from .attention_modules import (
    CompilerOptimizedMultiHeadAttention,
    FlashAttentionWrapper,
    MemoryEfficientAttention
)

from .normalization_layers import (
    OptimizedLayerNorm,
    OptimizedRMSNorm,
    FusedLayerNormActivation
)

__all__ = [
    "CompilerOptimizedMultiHeadAttention",
    "FlashAttentionWrapper",
    "MemoryEfficientAttention",
    "OptimizedLayerNorm",
    "OptimizedRMSNorm",
    "FusedLayerNormActivation"
]