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
    TPU = "tpu"
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


class TPUVersion(Enum):
    """TPU hardware versions."""
    AUTO = "auto"
    V4 = "v4"            # TPU v4
    V5E = "v5e"          # TPU v5e (cost-optimized)
    V5P = "v5p"          # TPU v5p (performance-optimized)
    V6E = "v6e"          # TPU v6e (next-gen cost-optimized)
    V7 = "v7"            # TPU v7 (future)


class TPUTopology(Enum):
    """TPU deployment topologies."""
    AUTO = "auto"
    SINGLE = "single"    # Single TPU chip
    POD = "pod"          # TPU Pod (multiple chips)
    SUPERPOD = "superpod"  # TPU Superpod (massive scale)


class TPUCompilationMode(Enum):
    """TPU compilation modes."""
    XLA = "xla"          # Standard XLA compilation
    PJIT = "pjit"        # JAX pjit compilation
    TORCH_XLA = "torch_xla"  # PyTorch/XLA compilation


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
class TPUConfig:
    """TPU-specific hardware configuration."""
    enabled: bool = True
    version: TPUVersion = TPUVersion.AUTO
    topology: TPUTopology = TPUTopology.AUTO

    # Compilation settings
    compilation_mode: TPUCompilationMode = TPUCompilationMode.TORCH_XLA
    xla_flags: Optional[str] = None

    # Performance settings
    precision: str = "bfloat16"
    mixed_precision: bool = True

    # Memory optimization
    memory_fraction: float = 0.90
    gradient_checkpointing: bool = True

    # XLA optimization flags
    xla_optimization_level: int = 2  # 0=debug, 1=basic, 2=aggressive
    enable_xla_dynamic_shapes: bool = True

    # JAX integration settings (if available)
    enable_jax_integration: bool = False
    jax_backend: str = "tpu"

    def __post_init__(self):
        """Auto-configure based on detected TPU environment."""
        if self.version == TPUVersion.AUTO:
            self.version = self._detect_tpu_version()

        if self.topology == TPUTopology.AUTO:
            self.topology = self._detect_tpu_topology()

        # Configure settings based on TPU version
        if self.version in [TPUVersion.V5P, TPUVersion.V6E, TPUVersion.V7]:
            # High-performance TPUs
            self.memory_fraction = 0.95
            self.xla_optimization_level = 2
        elif self.version == TPUVersion.V5E:
            # Cost-optimized TPUs
            self.memory_fraction = 0.90
            self.xla_optimization_level = 1

    def _detect_tpu_version(self) -> TPUVersion:
        """Detect TPU version from environment."""
        try:
            # Try to import XLA and detect TPU
            import torch_xla.core.xla_model as xm
            if xm.xla_device_hw(xm.xla_device()) == 'TPU':
                # Try to detect TPU version from environment
                import os
                tpu_type = os.environ.get('TPU_TYPE', '')

                if 'v5p' in tpu_type.lower():
                    return TPUVersion.V5P
                elif 'v5e' in tpu_type.lower():
                    return TPUVersion.V5E
                elif 'v6e' in tpu_type.lower():
                    return TPUVersion.V6E
                elif 'v4' in tpu_type.lower():
                    return TPUVersion.V4
                else:
                    # Default to v5e for unknown types
                    return TPUVersion.V5E
        except ImportError:
            # XLA not available
            pass
        except Exception:
            # Other detection errors
            pass

        return TPUVersion.V5E  # Default fallback

    def _detect_tpu_topology(self) -> TPUTopology:
        """Detect TPU topology from environment."""
        try:
            import torch_xla.core.xla_model as xm
            if xm.xla_device_hw(xm.xla_device()) == 'TPU':
                # Get number of TPU cores
                world_size = xm.xrt_world_size()

                if world_size == 1:
                    return TPUTopology.SINGLE
                elif world_size <= 8:
                    return TPUTopology.SINGLE  # Single node
                elif world_size <= 256:
                    return TPUTopology.POD
                else:
                    return TPUTopology.SUPERPOD
        except ImportError:
            pass
        except Exception:
            pass

        return TPUTopology.SINGLE  # Default fallback


@dataclass
class HardwareConfig:
    """Unified hardware optimization configuration."""
    backend: HardwareBackend = HardwareBackend.CUDA
    device_id: Optional[int] = None
    multi_gpu: bool = False

    # Hardware-specific configurations
    nvidia: NVIDIAConfig = field(default_factory=NVIDIAConfig)
    tpu: TPUConfig = field(default_factory=TPUConfig)

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
        # Auto-detect hardware backend if not explicitly set
        if self.backend == HardwareBackend.CUDA and not torch.cuda.is_available():
            # Try TPU detection if CUDA not available
            try:
                import torch_xla.core.xla_model as xm
                if xm.xla_device_hw(xm.xla_device()) == 'TPU':
                    self.backend = HardwareBackend.TPU
            except ImportError:
                # Fall back to CPU
                self.backend = HardwareBackend.CPU

        # Configure NVIDIA settings
        if self.backend == HardwareBackend.CUDA and torch.cuda.is_available():
            self.nvidia.enabled = True
            # NVIDIA config will auto-detect architecture in its own __post_init__
        else:
            self.nvidia.enabled = False

        # Configure TPU settings
        if self.backend == HardwareBackend.TPU:
            self.tpu.enabled = True
            # Disable incompatible settings for TPU
            self.tensor_cores_enabled = False  # TPU doesn't use Tensor Cores
            self.triton_enabled = False        # Triton is CUDA-specific
        else:
            self.tpu.enabled = False


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
class KernelConfig:
    """
    Custom CUDA kernel configuration for Phase 4A.

    Controls the behavior of custom CUDA kernels including FlashAttention-3,
    fused Linear+Activation, and other optimized operations.

    This configuration integrates with the KernelRegistry system to enable
    automatic kernel selection based on hardware capabilities and user preferences.
    """

    # Global kernel settings
    enabled: bool = True
    validate_kernels_on_load: bool = True
    auto_select_optimal: bool = True

    # FlashAttention settings
    flash_attention_enabled: bool = True
    flash_attention_version: str = "auto"  # "2", "3", or "auto"
    flash_attention_split_k: bool = True   # Enable Split-K for long sequences
    flash_attention_causal_default: bool = False

    # Fused Linear + Activation settings
    fuse_linear_activation: bool = True
    fused_gelu_enabled: bool = True
    fused_silu_enabled: bool = True
    fused_relu_enabled: bool = True

    # FP8 kernel settings (H100/Blackwell only)
    fp8_layernorm: bool = False  # Auto-enabled on H100+
    fp8_attention: bool = False  # Auto-enabled on H100+
    fp8_matmul: bool = False

    # Kernel fusion settings
    fusion_enabled: bool = True
    fusion_threshold: int = 2  # Minimum ops to fuse

    # Performance settings
    kernel_cache_enabled: bool = True
    benchmark_on_init: bool = False  # Benchmark kernels during initialization
    fallback_to_pytorch: bool = True  # Use PyTorch if kernel fails

    # Memory settings
    preallocate_kernel_memory: bool = False
    kernel_memory_pool_mb: int = 512

    # Debugging and profiling
    kernel_profiling: bool = False
    kernel_logging: bool = False
    save_kernel_stats: bool = False

    def __post_init__(self):
        """Auto-configure kernel settings based on hardware."""
        # Import here to avoid circular dependency
        try:
            import torch

            # Disable all kernels if CUDA not available
            if not torch.cuda.is_available():
                self.enabled = False
                self.flash_attention_enabled = False
                self.fuse_linear_activation = False
                return

            # Get compute capability
            compute_cap = torch.cuda.get_device_capability(0)

            # Enable FP8 kernels only on H100+ (compute capability 9.0+)
            if compute_cap >= (9, 0):
                self.fp8_layernorm = True
                self.fp8_attention = True
                self.fp8_matmul = True

                # Default to FlashAttention-3 on H100+
                if self.flash_attention_version == "auto":
                    self.flash_attention_version = "3"
            else:
                # Use FlashAttention-2 on older GPUs
                if self.flash_attention_version == "auto":
                    self.flash_attention_version = "2"

                # Disable FP8 on older GPUs
                self.fp8_layernorm = False
                self.fp8_attention = False
                self.fp8_matmul = False

            # Disable Split-K on older GPUs (requires compute 8.0+)
            if compute_cap < (8, 0):
                self.flash_attention_split_k = False

        except Exception:
            # If anything fails, use safe defaults
            self.flash_attention_version = "2"
            self.fp8_layernorm = False
            self.fp8_attention = False


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
    kernel: KernelConfig = field(default_factory=KernelConfig)

    # Global settings
    device: torch.device = field(default_factory=lambda: KernelPyTorchConfig._detect_device())
    seed: int = 42
    debug: bool = False
    profile: bool = False

    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    experimental_features: bool = False

    @staticmethod
    def _detect_device() -> torch.device:
        """Detect the best available device."""
        # Try CUDA first
        if torch.cuda.is_available():
            return torch.device("cuda")

        # Try TPU
        try:
            import torch_xla.core.xla_model as xm
            if xm.xla_device_hw(xm.xla_device()) == 'TPU':
                return xm.xla_device()
        except ImportError:
            pass
        except Exception:
            pass

        # Fall back to CPU
        return torch.device("cpu")

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Sync hardware backend with detected device
        if self.device.type == "cuda":
            self.hardware.backend = HardwareBackend.CUDA
        elif str(self.device).startswith('xla'):  # TPU device
            self.hardware.backend = HardwareBackend.TPU
        else:
            self.hardware.backend = HardwareBackend.CPU

        # Adjust hardware config based on device type
        if self.device.type == "cpu":
            self.hardware.tensor_cores_enabled = False
            self.precision.fp8_enabled = False
        elif str(self.device).startswith('xla'):  # TPU
            self.hardware.tensor_cores_enabled = False  # TPU doesn't use Tensor Cores
            self.precision.fp8_enabled = False          # TPU uses bfloat16, not FP8
            self.hardware.triton_enabled = False        # Triton is CUDA-specific

        # Validate memory settings
        if self.memory.max_memory_gb is None:
            if self.device.type == "cuda":
                self.memory.max_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            elif str(self.device).startswith('xla'):  # TPU
                # TPU memory varies by type, use reasonable default
                self.memory.max_memory_gb = 32.0  # Default for v5e

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