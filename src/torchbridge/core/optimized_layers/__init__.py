"""
Compiler-Optimized PyTorch Components

Complete suite of neural network components specifically designed for maximum
GPU performance through PyTorch compiler optimization (torch.compile, TorchScript, etc.).

Key Design Principles:
- Tensor-native operations that map efficiently to GPU kernels
- Compiler-friendly patterns that enable automatic optimization
- Memory-efficient designs that minimize GPU memory bandwidth
- Validation frameworks to ensure correctness and measure performance

Complete Component Suite:
- attention_modules: Various attention implementations optimized for compilation
- normalization_layers: Layer normalization variants with kernel optimization
- activation_functions: Fused activation patterns for better performance
- linear_transformations: Optimized linear layers and projections
- embedding_layers: Efficient embeddings and positional encoding
"""

# Attention components
# Activation components
from .activation_functions import (
    FusedGELU,
    FusedReLU,
    FusedSwiGLU,
    create_optimized_activation,
)
from .attention_modules import (
    CompilerOptimizedMultiHeadAttention,
    FlashAttentionWrapper,
    MemoryEfficientAttention,
    benchmark_attention_implementations,
    validate_attention_correctness,
)

# Embedding components
from .embedding_layers import (
    FusedTokenPositionalEmbedding,
    LearnablePositionalEncoding,
    OptimizedEmbedding,
    RotaryPositionalEncoding,
    create_optimized_embedding,
)

# Linear transformation components
from .linear_transformations import (
    FusedLinearSequence,
    GroupedLinearTransformation,
    MemoryEfficientLinear,
    MultiHeadLinearProjection,
    create_optimized_linear,
)

# Normalization components
from .normalization_layers import (
    FusedLayerNormActivation,
    OptimizedLayerNorm,
    OptimizedRMSNorm,
)

__all__ = [
    # Attention components
    "CompilerOptimizedMultiHeadAttention",
    "FlashAttentionWrapper",
    "MemoryEfficientAttention",

    # Normalization components
    "OptimizedLayerNorm",
    "OptimizedRMSNorm",
    "FusedLayerNormActivation",

    # Activation components
    "FusedGELU",
    "FusedSwiGLU",
    "FusedReLU",
    "create_optimized_activation",

    # Linear transformation components
    "MultiHeadLinearProjection",
    "GroupedLinearTransformation",
    "MemoryEfficientLinear",
    "FusedLinearSequence",
    "create_optimized_linear",

    # Embedding components
    "OptimizedEmbedding",
    "RotaryPositionalEncoding",
    "LearnablePositionalEncoding",
    "FusedTokenPositionalEmbedding",
    "create_optimized_embedding",

    # Utilities
    "benchmark_attention_implementations",
    "validate_attention_correctness",
]
