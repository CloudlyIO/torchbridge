"""
Unified Configuration System for TorchBridge

This module consolidates all configuration classes into a unified system,
replacing the scattered 36+ config classes throughout the codebase.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

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


class AMDArchitecture(Enum):
    """AMD GPU architectures."""
    AUTO = "auto"
    CDNA = "cdna"        # MI50, MI60 (1st gen)
    CDNA2 = "cdna2"      # MI200 series (MI210, MI250, MI250X)
    CDNA3 = "cdna3"      # MI300 series (MI300A, MI300X)
    RDNA2 = "rdna2"      # Consumer GPUs (RX 6000 series)
    RDNA3 = "rdna3"      # Consumer GPUs (RX 7000 series)


class IntelArchitecture(Enum):
    """Intel XPU architectures."""
    AUTO = "auto"
    PVC = "pvc"          # Ponte Vecchio (Data Center Max)
    ATS = "ats"          # Arctic Sound (older data center)
    DG2 = "dg2"          # Arc GPUs (A770, A750, A580)
    FLEX = "flex"        # Data Center Flex series
    INTEGRATED = "integrated"  # Integrated graphics (Iris Xe, etc.)


class AttentionPatterns(Enum):
    """Supported attention patterns - from attention module."""
    FULL = "full"                           # Standard full attention
    CAUSAL = "causal"                       # Causal/autoregressive attention
    SLIDING_WINDOW = "sliding_window"       # Local sliding window
    SPARSE = "sparse"                       # Sparse attention patterns
    RING = "ring"                           # Ring attention for long sequences
    LOCAL = "local"                         # Local attention (fixed window)
    GLOBAL = "global"                       # Global + local attention
    DIFFERENTIAL = "differential"           # Differential attention
    DYNAMIC_SPARSE = "dynamic_sparse"       # Dynamic sparse attention


@dataclass
class FP8AttentionConfig:
    """Enhanced FP8 configuration for attention mechanisms."""
    use_fp8: bool = False
    fp8_format: str = "e4m3"  # "e4m3" or "e5m2"
    async_compute: bool = True
    warp_specialization: bool = True
    tensor_core_utilization: float = 0.75
    sequence_length_threshold: int = 8192  # Use FP8 for sequences longer than this

    # Additional options
    gradient_checkpointing: bool = False
    mixed_precision: bool = True


@dataclass
class DynamicSparseConfig:
    """Configuration for dynamic sparse attention."""
    sparsity_threshold: float = 0.1
    adaptive_threshold: bool = True
    content_aware: bool = True
    efficiency_target: float = 0.8
    pattern_learning: bool = False
    min_sparsity: float = 0.05
    max_sparsity: float = 0.9


@dataclass
class RingAttentionConfig:
    """Configuration for ring attention."""
    segment_size: int = 2048
    communication_backend: str = "nccl"  # "nccl", "gloo", "mpi"
    overlap_communication: bool = True
    pipeline_parallel: bool = False
    memory_efficient: bool = True


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
    max_memory_gb: float | None = None
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
    xla_flags: str | None = None

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

    # Cache management settings
    cache_max_size: int = 100  # Maximum number of cached compilations/models
    compilation_timeout_seconds: int = 300  # XLA compilation timeout

    # Memory management settings
    allocation_history_retention_seconds: int = 3600  # Keep allocation history for 1 hour
    v6e_memory_gb: float | None = None  # Override TPU v6e memory capacity (default: 32.0)
    v7_memory_gb: float | None = None  # Override TPU v7 memory capacity (default: 128.0)

    # Validation settings
    enable_strict_validation: bool = False  # Raise errors instead of warnings for validation failures

    # Monitoring settings
    monitoring_interval_seconds: float = 1.0  # Memory monitoring interval
    monitoring_duration_seconds: float = 60.0  # Default monitoring duration

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
            # Check if we're on a TPU using compatible API
            if self._is_tpu_environment():
                # Try to detect TPU version from environment
                import os
                tpu_type = os.environ.get('TPU_TYPE', '')

                if 'v5p' in tpu_type.lower():
                    return TPUVersion.V5P
                elif 'v5e' in tpu_type.lower() or 'v5lite' in tpu_type.lower():
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
            if self._is_tpu_environment():
                # Get number of TPU cores using compatible API
                world_size = self._get_world_size()

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

    def _is_tpu_environment(self) -> bool:
        """Check if running in TPU environment (compatible with torch_xla 2.9+)."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # Get device using new API if available
            if hasattr(torch_xla, 'device'):
                device = torch_xla.device()
            else:
                device = xm.xla_device()

            if device.type != 'xla':
                return False

            # Check device hardware type
            if hasattr(xm, 'xla_device_hw'):
                return xm.xla_device_hw(device) == 'TPU'

            # Fallback: check environment variable
            import os
            return os.environ.get('PJRT_DEVICE', '').upper() == 'TPU'
        except Exception:
            return False

    def _get_world_size(self) -> int:
        """Get world size (compatible with torch_xla 2.9+)."""
        try:
            # Try new runtime API first (torch_xla 2.9+)
            import torch_xla
            if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'world_size'):
                return torch_xla.runtime.world_size()

            # Try older runtime API
            try:
                from torch_xla import runtime as xr
                if hasattr(xr, 'world_size'):
                    return xr.world_size()
            except ImportError:
                pass

            # Fall back to old xm API
            import torch_xla.core.xla_model as xm
            if hasattr(xm, 'xrt_world_size'):
                return xm.xrt_world_size()

            return 1
        except ImportError:
            return 1


@dataclass
class AMDConfig:
    """AMD ROCm-specific hardware configuration."""
    enabled: bool = True
    architecture: AMDArchitecture = AMDArchitecture.AUTO
    device_id: int = 0

    # ROCm settings
    rocm_version: str = "auto"  # ROCm version (e.g., "5.7", "6.0")
    hip_version: str = "auto"   # HIP version

    # Matrix Core settings (AMD's equivalent of Tensor Cores)
    enable_matrix_cores: bool = True
    matrix_core_precision: str = "auto"  # "fp16", "bf16", "fp32"

    # Performance optimization
    optimization_level: str = "balanced"  # "conservative", "balanced", "aggressive"

    # Memory settings
    enable_memory_pooling: bool = True
    memory_pool_size_gb: float = 8.0
    memory_pool_init_mb: int = 1024
    memory_growth_enabled: bool = True
    max_memory_fraction: float = 0.9

    # HIP kernel settings
    hip_kernel_cache_enabled: bool = True
    hip_kernel_cache_size: int = 100
    hip_compiler_cache_size: int = 100
    hip_compiler_cache_dir: str = "/tmp/hip_cache"

    # rocBLAS settings
    rocblas_enabled: bool = True
    rocblas_workspace_mb: int = 256
    enable_rocblas_tuning: bool = True

    # MIOpen settings (AMD's equivalent of cuDNN)
    miopen_enabled: bool = True
    miopen_find_mode: str = "NORMAL"  # "NORMAL", "FAST", "HYBRID"

    # Precision settings
    default_precision: str = "fp32"  # Default precision: "fp32", "fp16", "bf16"
    enable_mixed_precision: bool = True
    allow_fp16: bool = True
    allow_bf16: bool = True

    # Profiling and debugging
    enable_profiling: bool = False

    # Operator fusion
    enable_operator_fusion: bool = True

    # Validation and error handling
    enable_strict_validation: bool = False
    enable_oom_protection: bool = True

    def __post_init__(self):
        """Auto-configure based on detected architecture."""
        if self.architecture == AMDArchitecture.AUTO:
            self.architecture = self._detect_architecture()

        # Configure settings based on architecture
        if self.architecture == AMDArchitecture.CDNA3:
            # MI300 series - most advanced
            self.enable_matrix_cores = True
            self.matrix_core_precision = "bf16"
            self.allow_bf16 = True
        elif self.architecture == AMDArchitecture.CDNA2:
            # MI200 series - data center
            self.enable_matrix_cores = True
            self.matrix_core_precision = "fp16"
            self.allow_bf16 = True
        elif self.architecture == AMDArchitecture.CDNA:
            # MI50/MI60 - older data center
            self.enable_matrix_cores = False
            self.allow_bf16 = False
        else:
            # Consumer GPUs (RDNA2/RDNA3)
            self.enable_matrix_cores = False
            self.allow_bf16 = False

    def _detect_architecture(self) -> AMDArchitecture:
        """Detect AMD GPU architecture."""
        try:
            # Try to detect ROCm availability
            import torch
            if hasattr(torch, 'hip') and torch.hip.is_available():
                device_props = torch.hip.get_device_properties(0)
                device_name = device_props.name.upper()

                # MI300 series (CDNA3)
                if any(name in device_name for name in ["MI300", "MI3"]):
                    return AMDArchitecture.CDNA3

                # MI200 series (CDNA2)
                if any(name in device_name for name in ["MI210", "MI250", "MI2"]):
                    return AMDArchitecture.CDNA2

                # MI50/MI60 series (CDNA)
                if any(name in device_name for name in ["MI50", "MI60"]):
                    return AMDArchitecture.CDNA

                # RDNA3 (RX 7000 series)
                if any(name in device_name for name in ["RX 7", "RADEON 7"]):
                    return AMDArchitecture.RDNA3

                # RDNA2 (RX 6000 series)
                if any(name in device_name for name in ["RX 6", "RADEON 6"]):
                    return AMDArchitecture.RDNA2

                # Default to CDNA2 for unknown GPUs
                return AMDArchitecture.CDNA2
        except (ImportError, AttributeError, Exception):
            pass

        # Default to CDNA2 (most common data center GPU)
        return AMDArchitecture.CDNA2


@dataclass
class IntelConfig:
    """Intel XPU-specific hardware configuration."""
    enabled: bool = True
    architecture: IntelArchitecture = IntelArchitecture.AUTO
    device_id: int = 0

    # IPEX settings
    ipex_enabled: bool = True
    ipex_optimization_level: str = "O1"  # "O0", "O1"

    # oneDNN settings
    onednn_enabled: bool = True
    onednn_fusion_enabled: bool = True

    # Precision settings
    default_precision: str = "fp32"  # "fp32", "fp16", "bf16"
    enable_mixed_precision: bool = True
    allow_fp16: bool = True
    allow_bf16: bool = True

    # Memory settings
    enable_memory_pooling: bool = True
    max_memory_fraction: float = 0.9

    # Performance optimization
    optimization_level: str = "balanced"  # "conservative", "balanced", "aggressive"
    enable_amx: bool = True  # Enable AMX (Advanced Matrix Extensions) if available
    auto_kernel_selection: bool = True

    # Profiling and debugging
    enable_profiling: bool = False

    def __post_init__(self):
        """Auto-configure based on detected architecture."""
        if self.architecture == IntelArchitecture.AUTO:
            self.architecture = self._detect_architecture()

        # Configure settings based on architecture
        if self.architecture == IntelArchitecture.PVC:
            # Ponte Vecchio (Data Center Max) - most advanced
            self.allow_bf16 = True
            self.enable_amx = True
        elif self.architecture == IntelArchitecture.DG2:
            # Arc GPUs - consumer
            self.allow_bf16 = True
            self.enable_amx = False
        elif self.architecture == IntelArchitecture.FLEX:
            # Flex series - data center
            self.allow_bf16 = True
            self.enable_amx = False
        else:
            # Integrated graphics
            self.allow_bf16 = False
            self.enable_amx = False
            self.max_memory_fraction = 0.5  # Share memory with system

    def _detect_architecture(self) -> IntelArchitecture:
        """Detect Intel XPU architecture."""
        try:
            import torch
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device_props = torch.xpu.get_device_properties(0)
                device_name = device_props.name.upper()

                # Ponte Vecchio / Data Center Max
                if any(name in device_name for name in ["MAX", "PONTE VECCHIO", "PVC"]):
                    return IntelArchitecture.PVC

                # Arc GPUs (DG2)
                if any(name in device_name for name in ["ARC", "DG2", "A770", "A750", "A580"]):
                    return IntelArchitecture.DG2

                # Flex series
                if "FLEX" in device_name:
                    return IntelArchitecture.FLEX

                # Integrated graphics
                if any(name in device_name for name in ["IRIS", "UHD", "INTEGRATED"]):
                    return IntelArchitecture.INTEGRATED

                return IntelArchitecture.DG2  # Default to Arc for unknown
        except (ImportError, AttributeError, Exception):
            pass

        return IntelArchitecture.DG2  # Default fallback


@dataclass
class HardwareConfig:
    """Unified hardware optimization configuration."""
    backend: HardwareBackend = HardwareBackend.CUDA
    device_id: int | None = None
    multi_gpu: bool = False

    # Hardware-specific configurations
    nvidia: NVIDIAConfig = field(default_factory=NVIDIAConfig)
    tpu: TPUConfig = field(default_factory=TPUConfig)
    amd: AMDConfig = field(default_factory=AMDConfig)
    intel: IntelConfig = field(default_factory=IntelConfig)

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
            # Try Intel XPU detection first
            if self._detect_intel_xpu():
                self.backend = HardwareBackend.INTEL
            # Try AMD ROCm detection
            elif self._detect_amd_rocm():
                self.backend = HardwareBackend.AMD
            # Try TPU detection
            elif self._detect_tpu_environment():
                self.backend = HardwareBackend.TPU
            else:
                # Fall back to CPU
                self.backend = HardwareBackend.CPU

        # Configure NVIDIA settings
        if self.backend == HardwareBackend.CUDA and torch.cuda.is_available():
            self.nvidia.enabled = True
            # NVIDIA config will auto-detect architecture in its own __post_init__
        else:
            self.nvidia.enabled = False

        # Configure AMD settings
        if self.backend == HardwareBackend.AMD:
            self.amd.enabled = True
            # AMD config will auto-detect architecture in its own __post_init__
        else:
            self.amd.enabled = False

        # Configure Intel settings
        if self.backend == HardwareBackend.INTEL:
            self.intel.enabled = True
            # Disable incompatible settings for Intel XPU
            self.triton_enabled = False  # Triton is CUDA-specific
            # Intel config will auto-detect architecture in its own __post_init__
        else:
            self.intel.enabled = False

        # Configure TPU settings
        if self.backend == HardwareBackend.TPU:
            self.tpu.enabled = True
            # Disable incompatible settings for TPU
            self.tensor_cores_enabled = False  # TPU doesn't use Tensor Cores
            self.triton_enabled = False        # Triton is CUDA-specific
        else:
            self.tpu.enabled = False

    def _detect_intel_xpu(self) -> bool:
        """Check if Intel XPU is available."""
        try:
            import torch
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return True
        except Exception:
            pass
        return False

    def _detect_amd_rocm(self) -> bool:
        """Check if AMD ROCm is available."""
        try:
            import torch
            if hasattr(torch, 'hip') and torch.hip.is_available():
                return True
        except Exception:
            pass
        return False

    def _detect_tpu_environment(self) -> bool:
        """Check if running in TPU environment (compatible with torch_xla 2.9+)."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # Get device using new API if available
            if hasattr(torch_xla, 'device'):
                device = torch_xla.device()
            else:
                device = xm.xla_device()

            if device.type != 'xla':
                return False

            # Check device hardware type
            if hasattr(xm, 'xla_device_hw'):
                return xm.xla_device_hw(device) == 'TPU'

            # Fallback: check environment variable
            import os
            return os.environ.get('PJRT_DEVICE', '').upper() == 'TPU'
        except Exception:
            return False


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
class TorchBridgeConfig:
    """
    Unified configuration for the entire TorchBridge framework.

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
    device: torch.device = field(default_factory=lambda: TorchBridgeConfig._detect_device())
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

        # Try Intel XPU
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return torch.device("xpu")
        except Exception:
            pass

        # Try TPU (compatible with torch_xla 2.9+)
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # Get device using new API if available
            if hasattr(torch_xla, 'device'):
                device = torch_xla.device()
            else:
                device = xm.xla_device()

            if device.type == 'xla':
                # Check if it's actually a TPU
                if hasattr(xm, 'xla_device_hw'):
                    if xm.xla_device_hw(device) == 'TPU':
                        return device
                else:
                    # Assume it's a TPU if we got an XLA device
                    return device
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
        elif self.device.type == "xpu":
            self.hardware.backend = HardwareBackend.INTEL
        elif str(self.device).startswith('xla'):  # TPU device
            self.hardware.backend = HardwareBackend.TPU
        else:
            self.hardware.backend = HardwareBackend.CPU

        # Adjust hardware config based on device type
        if self.device.type == "cpu":
            self.hardware.tensor_cores_enabled = False
            self.precision.fp8_enabled = False
        elif self.device.type == "xpu":  # Intel XPU
            self.hardware.tensor_cores_enabled = False  # Intel uses Vector Engine, not Tensor Cores
            self.precision.fp8_enabled = False          # Intel XPU uses BF16/FP16, not FP8
            self.hardware.triton_enabled = False        # Triton is CUDA-specific
        elif str(self.device).startswith('xla'):  # TPU
            self.hardware.tensor_cores_enabled = False  # TPU doesn't use Tensor Cores
            self.precision.fp8_enabled = False          # TPU uses bfloat16, not FP8
            self.hardware.triton_enabled = False        # Triton is CUDA-specific

        # Validate memory settings
        if self.memory.max_memory_gb is None:
            if self.device.type == "cuda":
                self.memory.max_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            elif self.device.type == "xpu":
                # Intel XPU memory varies by device
                try:
                    props = torch.xpu.get_device_properties(0)
                    self.memory.max_memory_gb = props.total_memory / (1024**3)
                except Exception:
                    self.memory.max_memory_gb = 16.0  # Default for Arc GPUs
            elif str(self.device).startswith('xla'):  # TPU
                # TPU memory varies by type, use reasonable default
                self.memory.max_memory_gb = 32.0  # Default for v5e

    @classmethod
    def for_inference(cls) -> 'TorchBridgeConfig':
        """Create optimized configuration for inference."""
        config = cls()
        config.memory.gradient_checkpointing = False
        config.memory.deep_optimizer_states = False
        config.validation.enabled = False
        config.optimization_level = OptimizationLevel.AGGRESSIVE
        return config

    @classmethod
    def for_training(cls) -> 'TorchBridgeConfig':
        """Create optimized configuration for training."""
        config = cls()
        config.memory.gradient_checkpointing = True
        config.memory.deep_optimizer_states = True
        config.validation.enabled = True
        config.optimization_level = OptimizationLevel.BALANCED
        return config

    @classmethod
    def for_development(cls) -> 'TorchBridgeConfig':
        """Create configuration for development with debugging enabled."""
        config = cls()
        config.debug = True
        config.profile = True
        config.validation.strict_mode = True
        config.optimization_level = OptimizationLevel.CONSERVATIVE
        return config

    def to_dict(self) -> dict[str, Any]:
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
default_config = TorchBridgeConfig()


def get_config() -> TorchBridgeConfig:
    """Get the global default configuration."""
    return default_config


def set_config(config: TorchBridgeConfig) -> None:
    """Set the global default configuration."""
    global default_config
    default_config = config


def configure(**kwargs) -> TorchBridgeConfig:
    """Configure TorchBridge with keyword arguments."""
    config = TorchBridgeConfig()
    config.update(**kwargs)
    set_config(config)
    return config
