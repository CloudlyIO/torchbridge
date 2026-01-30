"""
LLM Optimization Module

Provides optimized wrappers for large language models (1B-13B parameters):
- Llama-2-7B, Llama-3-8B
- Mistral-7B, Mixtral-8x7B
- Phi-2, Phi-3

Features:
- KV-cache optimization for inference
- Quantization support (INT8, INT4, FP8)
- Flash Attention integration
- Memory-efficient generation
- Multi-GPU support (tensor parallelism)

Version: 0.4.12
"""

from .kv_cache import (
    KVCacheManager,
    PagedKVCache,
    SlidingWindowCache,
)
from .llm_optimizer import (
    GenerationConfig,
    LLMConfig,
    LLMOptimizer,
    OptimizedLlama,
    OptimizedMistral,
    OptimizedPhi,
    QuantizationMode,
    create_optimized_llm,
)

__all__ = [
    # Optimizer
    "LLMOptimizer",
    "LLMConfig",
    "OptimizedLlama",
    "OptimizedMistral",
    "OptimizedPhi",
    "create_optimized_llm",
    "QuantizationMode",
    "GenerationConfig",
    # KV Cache
    "KVCacheManager",
    "PagedKVCache",
    "SlidingWindowCache",
]
