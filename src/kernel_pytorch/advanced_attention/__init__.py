"""
Advanced Attention Mechanisms (2024-2025)

This module implements cutting-edge attention optimizations based on the latest research:
- FlashAttention-3 with FP8 support and asynchronous operations
- FlexAttention API integration for diverse attention patterns
- FlashLight for complex attention mechanisms
- Advanced memory-efficient attention variants

Key Features:
- Up to 2x speedup over FlashAttention-2
- FP8 precision with 2.6x error reduction
- 75% GPU utilization on Hopper architecture
- Support for million-token sequences
"""

from .flashattention3 import FlashAttention3, FP8AttentionConfig
from .flex_attention import FlexAttentionAPI, AttentionPatterns
from .flashlight_attention import FlashLightAttention, DifferentialAttention
from .memory_efficient_attention import MemoryEfficientAttention, LongSequenceAttention

__all__ = [
    'FlashAttention3',
    'FP8AttentionConfig',
    'FlexAttentionAPI',
    'AttentionPatterns',
    'FlashLightAttention',
    'DifferentialAttention',
    'MemoryEfficientAttention',
    'LongSequenceAttention'
]