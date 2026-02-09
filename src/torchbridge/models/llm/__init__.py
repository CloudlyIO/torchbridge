"""
LLM Cross-Backend Module

Provides HAL-aware wrappers for large language models (1B-13B parameters):
- DeepSeek R1 7B, Qwen 3 8B, Gemma 3 12B, Llama 4 Scout 17B
- Mistral-7B, Mixtral-8x7B
- Phi-2, Phi-3

Features:
- KV-cache management for inference
- Quantization support (INT8, INT4, FP8)
- Flash Attention integration
- Memory-efficient generation
- Multi-GPU support (tensor parallelism)

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
