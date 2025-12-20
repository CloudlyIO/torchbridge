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


class NVIDIAArchitecture(Enum):
    """NVIDIA GPU architectures."""
    AUTO = "auto"
    PASCAL = "pascal"     # GTX 1000 series
    VOLTA = "volta"       # V100
    TURING = "turing"     # RTX 2000 series
    AMPERE = "ampere"     # RTX 3000/A100
    ADA = "ada"           # RTX 4000 series
    HOPPER = "hopper"     # H100/H200
    BLACKWELL = "blackwell"  # B100/B200


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
class NVIDIAConfig:
    """NVIDIA-specific hardware configuration."""
    enabled: bool = True
    architecture: NVIDIAArchitecture = NVIDIAArchitecture.AUTO

    # FP8 settings for H100/Blackwell
    fp8_enabled: bool = True
    fp8_recipe: str = "DelayedScaling"

    # Tensor Core optimization
    tensor_core_version: int = 4  # Auto-detect based on architecture
    mixed_precision_enabled: bool = True

    # FlashAttention settings
    flash_attention_version: str = "3"
    flash_attention_enabled: bool = True

    # Memory optimization
    memory_pool_enabled: bool = True
    memory_fraction: float = 0.95

    # Kernel fusion settings
    kernel_fusion_enabled: bool = True
    cudnn_benchmark: bool = True

    def __post_init__(self):
        """Auto-configure based on detected architecture."""
        if self.architecture == NVIDIAArchitecture.AUTO:
            self.architecture = self._detect_architecture()

        # Configure FP8 based on architecture
        if self.architecture in [NVIDIAArchitecture.HOPPER, NVIDIAArchitecture.BLACKWELL]:
            self.fp8_enabled = True
            self.tensor_core_version = 4
        elif self.architecture == NVIDIAArchitecture.AMPERE:
            self.fp8_enabled = False  # A100 doesn't support FP8
            self.tensor_core_version = 3
        else:
            self.fp8_enabled = False
            self.tensor_core_version = 2

    def _detect_architecture(self) -> NVIDIAArchitecture:
        """Detect NVIDIA GPU architecture."""
        if not torch.cuda.is_available():
            return NVIDIAArchitecture.PASCAL

        try:
            device_props = torch.cuda.get_device_properties(0)
            device_name = device_props.name.upper()

            # H100/H200 (Hopper)
            if any(name in device_name for name in ["H100", "H200"]):
                return NVIDIAArchitecture.HOPPER

            # B100/B200 (Blackwell)
            if any(name in device_name for name in ["B100", "B200"]):
                return NVIDIAArchitecture.BLACKWELL

            # A100 (Ampere)
            if "A100" in device_name:
                return NVIDIAArchitecture.AMPERE

            # RTX 4000 series (Ada)
            if any(name in device_name for name in ["RTX 40", "RTX 4090", "RTX 4080"]):
                return NVIDIAArchitecture.ADA

            # RTX 3000/A40/A30 series (Ampere)
            if any(name in device_name for name in ["RTX 30", "A40", "A30"]):
                return NVIDIAArchitecture.AMPERE

            # RTX 2000 series (Turing)
            if any(name in device_name for name in ["RTX 20", "TITAN RTX"]):
                return NVIDIAArchitecture.TURING

            # V100 (Volta)
            if "V100" in device_name:
                return NVIDIAArchitecture.VOLTA

            # Fallback based on compute capability
            if device_props.major >= 9:
                return NVIDIAArchitecture.HOPPER
            elif device_props.major >= 8:
                return NVIDIAArchitecture.AMPERE
            elif device_props.major >= 7:
                return NVIDIAArchitecture.TURING if device_props.minor >= 5 else NVIDIAArchitecture.VOLTA
            else:
                return NVIDIAArchitecture.PASCAL

        except Exception:
            return NVIDIAArchitecture.PASCAL


@dataclass
class HardwareConfig:
    """Unified hardware optimization configuration."""
    backend: HardwareBackend = HardwareBackend.CUDA
    device_id: Optional[int] = None
    multi_gpu: bool = False

    # NVIDIA-specific configuration
    nvidia: NVIDIAConfig = field(default_factory=NVIDIAConfig)

    # Tensor Core settings (general)
    tensor_cores_enabled: bool = True
    mixed_precision: bool = True

    # Compilation settings
    torch_compile: bool = True
    triton_enabled: bool = True
    flashlight_enabled: bool = False

    # Performance settings
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED

    def __post_init__(self):
        """Auto-configure hardware settings based on detected capabilities."""
        # Auto-detect and configure NVIDIA settings
        if self.backend == HardwareBackend.CUDA and torch.cuda.is_available():
            self.nvidia.enabled = True
            # NVIDIA config will auto-detect architecture in its own __post_init__


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
        def _convert_value(value, visited=None):
            if visited is None:
                visited = set()

            # Prevent recursion by tracking object IDs
            if id(value) in visited:
                return f"<circular reference to {type(value).__name__}>"

            if hasattr(value, '__dict__'):
                visited.add(id(value))
                # Handle nested dataclass objects
                nested_dict = {}
                for nested_key, nested_value in value.__dict__.items():
                    nested_dict[nested_key] = _convert_value(nested_value, visited.copy())
                return nested_dict
            elif hasattr(value, 'value'):  # Handle Enum objects
                return value.value
            elif isinstance(value, torch.device):
                return str(value)
            else:
                return value

        result = {}
        for key, value in self.__dict__.items():
            result[key] = _convert_value(value)
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