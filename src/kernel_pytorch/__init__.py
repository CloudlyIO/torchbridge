"""
Kernel-Optimized PyTorch Components

This package provides progressively optimized neural network components
that demonstrate how to align PyTorch models with GPU computation patterns.

Optimization Levels:
- Level 1: PyTorch native (cuDNN/cuBLAS optimized)
- Level 2: TorchScript JIT compilation
- Level 3: torch.compile (Inductor backend)
- Level 4: Triton kernels (Python-based CUDA)
- Level 5: Custom CUDA kernels (maximum control)
"""

__version__ = "0.4.3"

# Unified Configuration System
from .core.config import (
    KernelPyTorchConfig,
    PrecisionConfig,
    MemoryConfig,
    AttentionConfig,
    HardwareConfig,
    DistributedConfig,
    ValidationConfig,
    get_config,
    set_config,
    configure
)

# Core Components (explicit imports)
from .core.compilers.flashlight_compiler import FlashLightKernelCompiler
from .core.compilers.pygraph_optimizer import PyGraphCUDAOptimizer
from .core.optimized_layers.activation_functions import FusedGELU

# Unified Management System
from .core.management import UnifiedManager, get_manager

# Attention Mechanisms
from .attention.fusion.neural_operator import UnifiedAttentionFusion, FusionStrategy
from .attention.distributed.ring_attention import RingAttentionLayer
from .attention.core.base import BaseAttention as AttentionLayer

# Precision Optimization
from .precision.ultra_precision import UltraPrecisionModule, AdaptivePrecisionAllocator
from .precision.fp8_training_engine import FP8TrainingEngine

# Memory Optimization
from .advanced_memory.deep_optimizer_states import DeepOptimizerStates, CPUGPUHybridOptimizer
from .advanced_memory.advanced_checkpointing import SelectiveGradientCheckpointing

# Hardware Abstraction
from .hardware.abstraction.hal_core import HardwareAbstractionLayer

# Validation Framework
from .validation.unified_validator import UnifiedValidator

# Public API - explicitly defined exports
__all__ = [
    # Version
    "__version__",

    # Configuration
    "KernelPyTorchConfig", "PrecisionConfig", "MemoryConfig", "AttentionConfig",
    "HardwareConfig", "DistributedConfig", "ValidationConfig",
    "get_config", "set_config", "configure",

    # Core Components
    "FlashLightKernelCompiler", "PyGraphCUDAOptimizer", "FusedGELU",

    # Management System
    "UnifiedManager", "get_manager",

    # Attention
    "UnifiedAttentionFusion", "FusionStrategy", "RingAttentionLayer", "AttentionLayer",

    # Precision
    "UltraPrecisionModule", "AdaptivePrecisionAllocator", "FP8TrainingEngine",

    # Memory
    "DeepOptimizerStates", "CPUGPUHybridOptimizer", "SelectiveGradientCheckpointing",

    # Hardware
    "HardwareAbstractionLayer",

    # Validation
    "UnifiedValidator",
]

# Convenience functions for quick setup
def create_attention(d_model: int, num_heads: int, **kwargs):
    """Create optimized attention layer with automatic configuration."""
    config = get_config()
    return AttentionLayer(
        embed_dim=d_model,
        num_heads=num_heads,
        device=config.device,
        **kwargs
    )

def create_precision_module(model, **kwargs):
    """Create precision-optimized model wrapper."""
    config = get_config()
    return UltraPrecisionModule(
        model,
        config=config.precision,
        **kwargs
    )

def create_memory_optimizer(optimizer, model, **kwargs):
    """Create memory-optimized training setup."""
    config = get_config()
    return DeepOptimizerStates(
        optimizer=optimizer,
        model=model,
        memory_config=config.memory,
        **kwargs
    )

def optimize_model(model, **kwargs):
    """Apply unified optimization to model using global manager."""
    return get_manager().optimize(model, **kwargs)