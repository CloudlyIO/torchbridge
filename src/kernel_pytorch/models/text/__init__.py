"""
Text Model Optimization Module

Provides optimized wrappers for text models including:
- BERT (bert-base-uncased, bert-large-uncased)
- GPT-2 (gpt2, gpt2-medium)
- DistilBERT (distilbert-base-uncased)

Features:
- Automatic backend selection (NVIDIA, AMD, TPU, Intel, CPU)
- Mixed precision support (FP16, BF16, FP8)
- torch.compile optimization
- FlashAttention integration
- Memory-efficient inference

Version: 0.4.11
"""

from .text_model_optimizer import (
    TextModelOptimizer,
    OptimizedBERT,
    OptimizedGPT2,
    OptimizedDistilBERT,
    create_optimized_text_model,
)

__all__ = [
    "TextModelOptimizer",
    "OptimizedBERT",
    "OptimizedGPT2",
    "OptimizedDistilBERT",
    "create_optimized_text_model",
]
