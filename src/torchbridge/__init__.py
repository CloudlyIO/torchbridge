"""
TorchBridge — Hardware Abstraction Layer for PyTorch

Unified optimization across NVIDIA, AMD, Intel, and TPU backends.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchbridge")
except PackageNotFoundError:
    __version__ = "0.4.41"  # Fallback for development

# Unified Configuration System
from .advanced_memory.advanced_checkpointing import SelectiveGradientCheckpointing

# Memory Optimization
from .advanced_memory.deep_optimizer_states import (
    CPUGPUHybridOptimizer,
    DeepOptimizerStates,
)
from .attention.core.base import BaseAttention as AttentionLayer

# Core Components (explicit imports)
from .core.config import (
    AttentionConfig,
    DistributedConfig,
    HardwareConfig,
    MemoryConfig,
    PrecisionConfig,
    TorchBridgeConfig,
    ValidationConfig,
    configure,
    get_config,
    set_config,
)

# Unified Management System
from .core.management import UnifiedManager, get_manager
from .core.optimized_layers.activation_functions import FusedGELU

# Hardware Abstraction
from .hardware.abstraction.hal_core import HardwareAbstractionLayer

# Mixture of Experts
from .mixture_of_experts import (
    FeedForwardExpert,
    GLaMStyleMoE,
    LoadBalancer,
    MoEConfig,
    MoELayer,
    SparseMoELayer,
    SwitchRouter,
    SwitchTransformerMoE,
    TopKRouter,
    create_moe_layer,
)
from .precision.fp8_training_engine import FP8TrainingEngine

# Validation Framework
from .validation.unified_validator import UnifiedValidator

# Public API - explicitly defined exports
__all__ = [
    # Version
    "__version__",

    # Configuration
    "TorchBridgeConfig", "PrecisionConfig", "MemoryConfig", "AttentionConfig",
    "HardwareConfig", "DistributedConfig", "ValidationConfig",
    "get_config", "set_config", "configure",

    # Core Components
    "FusedGELU",

    # Management System
    "UnifiedManager", "get_manager",

    # Attention
    "AttentionLayer",

    # Precision
    "FP8TrainingEngine",

    # Memory
    "DeepOptimizerStates", "CPUGPUHybridOptimizer", "SelectiveGradientCheckpointing",

    # Hardware
    "HardwareAbstractionLayer",

    # Validation
    "UnifiedValidator",

    # Mixture of Experts
    "MoELayer", "SparseMoELayer", "SwitchTransformerMoE", "GLaMStyleMoE",
    "MoEConfig", "create_moe_layer", "create_moe",
    "TopKRouter", "SwitchRouter", "LoadBalancer", "FeedForwardExpert",
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

def create_moe(hidden_size: int, num_experts: int = 8, top_k: int = 2, moe_type: str = "standard", **kwargs):
    """Create Mixture of Experts layer with automatic configuration.

    Args:
        hidden_size: Hidden dimension size
        num_experts: Number of experts (default 8)
        top_k: Number of experts per token (default 2)
        moe_type: Type of MoE ("standard", "sparse", "switch", "glam", "adaptive")
        **kwargs: Additional configuration options

    Returns:
        MoE layer instance
    """
    return create_moe_layer(
        moe_type=moe_type,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        **kwargs
    )
