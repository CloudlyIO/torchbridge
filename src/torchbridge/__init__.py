"""
TorchBridge — Cross-backend validation and configuration intelligence for PyTorch.

Validates that outputs match across NVIDIA, AMD, Trainium, and TPU backends,
and generates optimal configurations per hardware target.
"""

# Suppress noisy platform-specific warnings before importing torch
# These are informational notes, not errors - hardware status is reported via CLI
import logging
import os
import warnings

# Suppress PyTorch distributed elastic logging on non-Linux (macOS/Windows)
logging.getLogger("torch.distributed.elastic").setLevel(logging.ERROR)

# Suppress pynvml deprecation warning (PyTorch issue, not ours)
os.environ.setdefault("PYTORCH_NVML_SUPPRESS_DEPRECATION_WARNING", "1")

# Filter specific warnings that are informational, not actionable errors
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

from typing import Any

import torch
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchbridge-ml")
except PackageNotFoundError:
    try:
        import re
        from pathlib import Path

        _pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        _match = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject.read_text(), re.MULTILINE)
        __version__ = _match.group(1) if _match else "0.0.0"
    except Exception:
        __version__ = "0.0.0"

from .attention.core.base import BaseAttention as AttentionLayer
from .attention.core.config import AttentionModuleConfig
from .attention.implementations.memory_efficient import MemoryEfficientAttention

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

    # Management System
    "UnifiedManager", "get_manager",

    # Attention
    "AttentionLayer",

    # Validation
    "UnifiedValidator",
]

# Convenience functions for quick setup
def create_attention(d_model: int, num_heads: int, **kwargs: Any) -> MemoryEfficientAttention:
    """Create backend-aware attention layer with automatic configuration."""
    attn_config = AttentionModuleConfig(
        embed_dim=d_model,
        num_heads=num_heads,
        **kwargs,
    )
    return MemoryEfficientAttention(attn_config)

def optimize_model(model: torch.nn.Module, **kwargs: Any) -> Any:
    """Apply unified backend abstraction to model using global manager."""
    return get_manager().optimize(model, **kwargs)
