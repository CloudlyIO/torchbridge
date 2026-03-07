"""
TorchBridge Model Integration Module

Provides LLM inference utilities across NVIDIA, AMD, Trainium, and TPU backends.
"""

from .llm import (
    GenerationConfig,
    KVCacheManager,
    LLMConfig,
    LLMOptimizer,
    OptimizedLlama,
    OptimizedMistral,
    OptimizedPhi,
    PagedKVCache,
    QuantizationMode,
    SlidingWindowCache,
    create_optimized_llm,
)

__all__ = [
    "LLMOptimizer",
    "LLMConfig",
    "OptimizedLlama",
    "OptimizedMistral",
    "OptimizedPhi",
    "create_optimized_llm",
    "QuantizationMode",
    "GenerationConfig",
    "KVCacheManager",
    "PagedKVCache",
    "SlidingWindowCache",
]
