"""
Multi-modal Model Optimization (v0.4.15)

This module provides optimized inference for multi-modal models:
- CLIP (Vision-Language): Image-text embedding and similarity
- LLaVA (Visual Instruction Following): Visual question answering
- Whisper (Speech Recognition): Audio transcription and translation

Features:
- Cross-modal attention optimization
- Precision optimization (FP16/BF16)
- Memory-efficient inference
- Batch processing
"""

# Base classes and configuration
from .base import (
    BaseMultiModalOptimizer,
    CrossModalAttention,
    ModalityType,
    MultiModalOptimizationConfig,
    MultiModalType,
    OptimizationLevel,
    count_parameters,
    estimate_model_memory,
)

# CLIP optimization
from .clip import (
    CLIPBenchmark,
    CLIPOptimizer,
    create_clip_optimizer,
    create_clip_vit_b_optimized,
    create_clip_vit_l_optimized,
)

# LLaVA optimization
from .llava import (
    LLaVABenchmark,
    LLaVAOptimizer,
    create_llava_7b_optimized,
    create_llava_13b_optimized,
    create_llava_optimizer,
)

# Whisper optimization
from .whisper import (
    WhisperBenchmark,
    WhisperOptimizer,
    create_whisper_base_optimized,
    create_whisper_large_optimized,
    create_whisper_optimizer,
    create_whisper_small_optimized,
)

__all__ = [
    # Base
    "BaseMultiModalOptimizer",
    "MultiModalOptimizationConfig",
    "MultiModalType",
    "OptimizationLevel",
    "ModalityType",
    "CrossModalAttention",
    "count_parameters",
    "estimate_model_memory",
    # CLIP
    "CLIPOptimizer",
    "CLIPBenchmark",
    "create_clip_optimizer",
    "create_clip_vit_b_optimized",
    "create_clip_vit_l_optimized",
    # LLaVA
    "LLaVAOptimizer",
    "LLaVABenchmark",
    "create_llava_optimizer",
    "create_llava_7b_optimized",
    "create_llava_13b_optimized",
    # Whisper
    "WhisperOptimizer",
    "WhisperBenchmark",
    "create_whisper_optimizer",
    "create_whisper_base_optimized",
    "create_whisper_small_optimized",
    "create_whisper_large_optimized",
]

__version__ = "0.4.15"
