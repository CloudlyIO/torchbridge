"""
Unified Configuration System for KernelPyTorch

This module consolidates all configuration classes into a unified system,
replacing the scattered 36+ config classes throughout the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import torch


class PrecisionFormat(Enum):
    """Supported precision formats."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"


class OptimizationLevel(Enum):
    """Optimization levels from conservative to aggressive."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class HardwareBackend(Enum):
    """Supported hardware backends."""
    CUDA = "cuda"
    CPU = "cpu"
    AMD = "amd"
    INTEL = "intel"
    CUSTOM = "custom"


@dataclass
class PrecisionConfig:
    """Unified precision configuration."""
    default_format: PrecisionFormat = PrecisionFormat.FP32
    adaptive_allocation: bool = True
    entropy_threshold: float = 0.5
    memory_budget: float = 0.5
    quality_target: float = 0.3

    # FP8 specific settings
    fp8_enabled: bool = False
    fp8_margin: int = 0
    fp8_interval: int = 1

    # Quantization settings
    quantization_enabled: bool = False
    calibration_samples: int = 1000


@dataclass
class MemoryConfig:
    """Unified memory optimization configuration."""
    deep_optimizer_states: bool = True
    gradient_checkpointing: bool = False
    memory_pool_enabled: bool = True
    offloading_enabled: bool = False

    # Memory thresholds
    max_memory_gb: Optional[float] = None
    memory_fraction: float = 0.8
    fragmentation_threshold: float = 0.1

    # Advanced settings
    long_sequence_optimization: bool = False
    sequence_length_threshold: int = 8192


@dataclass
class AttentionConfig:
    """Unified attention mechanism configuration."""
    mechanism: str = "flash_attention"
    sparse_enabled: bool = False
    sparsity_ratio: float = 0.5

    # Ring attention settings
    ring_enabled: bool = False
    max_sequence_length: int = 1000000

    # Fusion settings
    fusion_enabled: bool = True
    fusion_strategy: str = "attention_ffn"

    # Context parallel settings
    context_parallel_size: int = 1


@dataclass
class HardwareConfig:
    """Unified hardware optimization configuration."""
    backend: HardwareBackend = HardwareBackend.CUDA
    device_id: Optional[int] = None
    multi_gpu: bool = False

    # Tensor Core settings
    tensor_cores_enabled: bool = True
    mixed_precision: bool = True

    # Compilation settings
    torch_compile: bool = True
    triton_enabled: bool = True
    flashlight_enabled: bool = False

    # Performance settings
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED


@dataclass
class DistributedConfig:
    """Unified distributed training configuration."""
    enabled: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0

    # FSDP settings
    fsdp_enabled: bool = False
    sharding_strategy: str = "full_shard"
    cpu_offload: bool = False

    # Communication settings
    communication_backend: str = "nccl"
    timeout_minutes: int = 30


@dataclass
class ValidationConfig:
    """Unified validation and testing configuration."""
    enabled: bool = True
    strict_mode: bool = False
    performance_tracking: bool = True

    # Test thresholds
    accuracy_threshold: float = 0.95
    performance_threshold: float = 0.8
    memory_threshold_gb: float = 16.0

    # Benchmark settings
    benchmark_iterations: int = 10
    warmup_iterations: int = 3


@dataclass
class KernelPyTorchConfig:
    """
    Unified configuration for the entire KernelPyTorch framework.

    This replaces all scattered configuration classes throughout the codebase:
    - precision/ultra_precision.py:PrecisionConfig
    - attention/core/config.py:AttentionConfig
    - hardware/gpu/tensor_cores.py:TensorCoreConfig
    - distributed_scale/multi_node_training.py:FSDPConfig
    - And 30+ other configuration classes
    """

    # Core configurations
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Global settings
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    seed: int = 42
    debug: bool = False
    profile: bool = False

    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    experimental_features: bool = False

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Auto-detect device if not explicitly set
        if self.device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")

        # Adjust hardware config based on device
        if self.device.type == "cpu":
            self.hardware.backend = HardwareBackend.CPU
            self.hardware.tensor_cores_enabled = False
            self.precision.fp8_enabled = False

        # Validate memory settings
        if self.memory.max_memory_gb is None:
            if self.device.type == "cuda":
                self.memory.max_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    @classmethod
    def for_inference(cls) -> 'KernelPyTorchConfig':
        """Create optimized configuration for inference."""
        config = cls()
        config.memory.gradient_checkpointing = False
        config.memory.deep_optimizer_states = False
        config.validation.enabled = False
        config.optimization_level = OptimizationLevel.AGGRESSIVE
        return config

    @classmethod
    def for_training(cls) -> 'KernelPyTorchConfig':
        """Create optimized configuration for training."""
        config = cls()
        config.memory.gradient_checkpointing = True
        config.memory.deep_optimizer_states = True
        config.validation.enabled = True
        config.optimization_level = OptimizationLevel.BALANCED
        return config

    @classmethod
    def for_development(cls) -> 'KernelPyTorchConfig':
        """Create configuration for development with debugging enabled."""
        config = cls()
        config.debug = True
        config.profile = True
        config.validation.strict_mode = True
        config.optimization_level = OptimizationLevel.CONSERVATIVE
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result

    def update(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


# Global default configuration instance
default_config = KernelPyTorchConfig()


def get_config() -> KernelPyTorchConfig:
    """Get the global default configuration."""
    return default_config


def set_config(config: KernelPyTorchConfig) -> None:
    """Set the global default configuration."""
    global default_config
    default_config = config


def configure(**kwargs) -> KernelPyTorchConfig:
    """Configure KernelPyTorch with keyword arguments."""
    config = KernelPyTorchConfig()
    config.update(**kwargs)
    set_config(config)
    return config