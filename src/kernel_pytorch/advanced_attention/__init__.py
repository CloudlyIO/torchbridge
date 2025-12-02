"""
Advanced Attention Mechanisms (2025-2026)

This module implements cutting-edge attention optimizations based on the latest research:
- FlashAttention-3 with FP8 support and asynchronous operations
- FlexAttention API integration for diverse attention patterns
- FlashLight for complex attention mechanisms
- Advanced memory-efficient attention variants
- Ring Attention for 1M+ token sequences with linear memory complexity
- Dynamic Sparse Attention with 90% compute reduction
- Context Parallel Attention for multi-GPU coordination

Key Features:
- Up to 2x speedup over FlashAttention-2
- FP8 precision with 2.6x error reduction
- 75% GPU utilization on Hopper architecture
- Support for million-token sequences
- Linear memory complexity with Ring Attention
- Content-aware sparse patterns
- Multi-GPU distributed attention
"""

from .flashattention3 import FlashAttention3, FP8AttentionConfig
from .flex_attention import FlexAttentionAPI, AttentionPatterns
from .flashlight_attention import FlashLightAttention, DifferentialAttention
from .memory_efficient_attention import MemoryEfficientAttention, LongSequenceAttention

# New Phase 1 implementations
from .ring_attention import (
    RingAttentionLayer,
    RingAttentionBlock,
    RingAttentionConfig,
    create_ring_attention,
    estimate_memory_usage,
    validate_ring_attention_setup
)
from .sparse_attention import (
    DynamicSparseAttention,
    DynamicSparseConfig,
    SparsePattern,
    SparseAttentionMaskGenerator,
    create_sparse_attention,
    compute_attention_efficiency
)
from .context_parallel import (
    ContextParallelAttention,
    ContextParallelBlock,
    ContextParallelConfig,
    create_context_parallel_attention,
    estimate_context_parallel_efficiency
)

# Phase 2.2 implementations
from .unified_attention_fusion import (
    UnifiedAttentionFusion,
    FusionConfig,
    FusionStrategy,
    OptimizationLevel,
    FusionPerformanceStats,
    create_unified_attention_fusion,
    benchmark_fusion_performance,
    print_fusion_analysis,
    print_benchmark_results
)

__all__ = [
    # Existing implementations
    'FlashAttention3',
    'FP8AttentionConfig',
    'FlexAttentionAPI',
    'AttentionPatterns',
    'FlashLightAttention',
    'DifferentialAttention',
    'MemoryEfficientAttention',
    'LongSequenceAttention',

    # Phase 1 implementations - Ring Attention
    'RingAttentionLayer',
    'RingAttentionBlock',
    'RingAttentionConfig',
    'create_ring_attention',
    'estimate_memory_usage',
    'validate_ring_attention_setup',

    # Phase 1 implementations - Sparse Attention
    'DynamicSparseAttention',
    'DynamicSparseConfig',
    'SparsePattern',
    'SparseAttentionMaskGenerator',
    'create_sparse_attention',
    'compute_attention_efficiency',

    # Phase 1 implementations - Context Parallel
    'ContextParallelAttention',
    'ContextParallelBlock',
    'ContextParallelConfig',
    'create_context_parallel_attention',
    'estimate_context_parallel_efficiency',

    # Phase 2.2 implementations - Neural Operator Fusion
    'UnifiedAttentionFusion',
    'FusionConfig',
    'FusionStrategy',
    'OptimizationLevel',
    'FusionPerformanceStats',
    'create_unified_attention_fusion',
    'benchmark_fusion_performance',
    'print_fusion_analysis',
    'print_benchmark_results'
]