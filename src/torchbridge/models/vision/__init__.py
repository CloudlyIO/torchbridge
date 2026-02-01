"""
Vision Model Optimization (v0.4.14)

This module provides optimized inference for computer vision models:
- ResNet (ResNet-50/152): Image classification
- ViT (Vision Transformer): Image classification
- Stable Diffusion: Image generation

Features:
- Operator fusion (Conv+BN+ReLU)
- Memory layout optimization (channels_last)
- Precision optimization (FP16/BF16)
- Attention slicing for memory efficiency
- VAE tiling for large images
- Batch inference optimization
"""

# Base classes and configuration
from .base import (
    BaseVisionOptimizer,
    OptimizationLevel,
    VisionModelType,
    VisionOptimizationConfig,
    count_parameters,
    estimate_model_memory,
)

# Stable Diffusion optimization
from .diffusion import (
    StableDiffusionBenchmark,
    StableDiffusionOptimizer,
    create_sd_1_5_optimized,
    create_sd_2_1_optimized,
    create_sdxl_optimized,
    create_stable_diffusion_optimizer,
)

# ResNet optimization
from .resnet import (
    ResNetBenchmark,
    ResNetOptimizer,
    create_resnet50_optimized,
    create_resnet152_optimized,
    create_resnet_optimizer,
)

# Vision Transformer optimization
from .vit import (
    ViTBenchmark,
    ViTOptimizer,
    create_vit_base_optimized,
    create_vit_large_optimized,
    create_vit_optimizer,
)

__all__ = [
    # Base
    "BaseVisionOptimizer",
    "VisionOptimizationConfig",
    "VisionModelType",
    "OptimizationLevel",
    "count_parameters",
    "estimate_model_memory",
    # ResNet
    "ResNetOptimizer",
    "ResNetBenchmark",
    "create_resnet_optimizer",
    "create_resnet50_optimized",
    "create_resnet152_optimized",
    # ViT
    "ViTOptimizer",
    "ViTBenchmark",
    "create_vit_optimizer",
    "create_vit_base_optimized",
    "create_vit_large_optimized",
    # Stable Diffusion
    "StableDiffusionOptimizer",
    "StableDiffusionBenchmark",
    "create_stable_diffusion_optimizer",
    "create_sd_1_5_optimized",
    "create_sd_2_1_optimized",
    "create_sdxl_optimized",
]

__version__ = "0.4.41"
