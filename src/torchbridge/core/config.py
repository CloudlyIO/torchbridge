# SPDX-License-Identifier: Apache-2.0
"""
Unified Configuration System for TorchBridge

This module consolidates all configuration classes into a unified system,
replacing the scattered 36+ config classes throughout the codebase.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

logger = logging.getLogger(__name__)


class PrecisionFormat(Enum):
    """Supported precision formats."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"


class OptimizationLevel(Enum):
    """
    Standardized optimization levels across all backends.

    O0: No optimizations (debug mode)
    O1: Conservative optimizations (safe, minimal impact)
    O2: Balanced optimizations (performance + stability)
    O3: Aggressive optimizations (maximum performance)
    """

    O0 = "O0"
    O1 = "O1"
    O2 = "O2"
    O3 = "O3"

    # Aliases for compatibility
    DEBUG = "O0"
    CONSERVATIVE = "O1"
    BALANCED = "O2"
    AGGRESSIVE = "O3"

    @classmethod
    def from_string(cls, level: str) -> "OptimizationLevel":
        """
        Convert string to OptimizationLevel.

        Args:
            level: String like "O0", "O1", "conservative", "balanced", etc.

        Returns:
            OptimizationLevel enum value
        """
        level_upper = level.upper()

        if level_upper in ("O0", "DEBUG"):
            return cls.O0
        elif level_upper in ("O1", "CONSERVATIVE"):
            return cls.O1
        elif level_upper in ("O2", "BALANCED"):
            return cls.O2
        elif level_upper in ("O3", "AGGRESSIVE"):
            return cls.O3
        else:
            logger.warning(
                f"Unknown optimization level '{level}', defaulting to O2 (balanced)"
            )
            return cls.O2


class HardwareBackend(Enum):
    """Supported hardware backends."""

    CUDA = "cuda"
    CPU = "cpu"
    TPU = "tpu"
    AMD = "amd"
    TRAINIUM = "trainium"
    CUSTOM = "custom"


class NVIDIAArchitecture(Enum):
    """NVIDIA GPU architectures."""

    AUTO = "auto"
    PASCAL = "pascal"  # GTX 1000 series
    VOLTA = "volta"  # V100
    TURING = "turing"  # RTX 2000 series
    AMPERE = "ampere"  # RTX 3000/A100
    ADA = "ada"  # RTX 4000 series
    HOPPER = "hopper"  # H100/H200
    BLACKWELL_DC = "blackwell_dc"  # B100/B200/GB200 (sm_100, cc 10.0)
    BLACKWELL_CONSUMER = "blackwell_consumer"  # RTX 5090/5080 (sm_120, cc 12.0)
    BLACKWELL_ULTRA = "blackwell_ultra"  # B300 (sm_103) — placeholder, H2 2026
    RUBIN = "rubin"  # R200/VR200 (sm_rubin) — placeholder, in production HPC


class TPUVersion(Enum):
    """TPU hardware versions."""

    AUTO = "auto"
    V4 = "v4"  # TPU v4
    V5E = "v5e"  # TPU v5e (cost-optimized)
    V5P = "v5p"  # TPU v5p (performance-optimized)
    V6E = "v6e"  # TPU v6e (next-gen cost-optimized)
    V7 = "v7"  # TPU v7 (future)


class TPUTopology(Enum):
    """TPU deployment topologies."""

    AUTO = "auto"
    SINGLE = "single"  # Single TPU chip
    POD = "pod"  # TPU Pod (multiple chips)
    SUPERPOD = "superpod"  # TPU Superpod (massive scale)


class TPUCompilationMode(Enum):
    """TPU compilation modes."""

    XLA = "xla"  # Standard XLA compilation
    PJIT = "pjit"  # JAX pjit compilation
    TORCH_XLA = "torch_xla"  # PyTorch/XLA compilation


class AMDArchitecture(Enum):
    """AMD GPU architectures."""

    AUTO = "auto"
    CDNA = "cdna"  # MI50, MI60 (1st gen)
    CDNA2 = "cdna2"  # MI200 series (MI210, MI250, MI250X)
    CDNA3 = "cdna3"  # MI300 series (MI300A, MI300X, MI325X)
    CDNA4 = "cdna4"  # MI350X, MI355X (gfx950)
    RDNA2 = "rdna2"  # Consumer GPUs (RX 6000 series)
    RDNA3 = "rdna3"  # Consumer GPUs (RX 7000 series)
    RDNA4 = "rdna4"  # Consumer GPUs (RX 9000 series, gfx1201)


class TrainiumArchitecture(Enum):
    """AWS Trainium chip generations."""

    AUTO = "auto"
    TRN1 = "trn1"  # Trainium1, NeuronCore v1, 32GB HBM
    TRN2 = "trn2"  # Trainium2, NeuronCore v3, 96GB HBM
    TRN3 = "trn3"  # Trainium3, NeuronCore v4, 144GB HBM3e
    INF2 = "inf2"  # Inferentia2 (inference-optimized)


class AttentionPatterns(Enum):
    """Supported attention patterns - from attention module."""

    FULL = "full"  # Standard full attention
    CAUSAL = "causal"  # Causal/autoregressive attention
    SLIDING_WINDOW = "sliding_window"  # Local sliding window
    SPARSE = "sparse"  # Sparse attention patterns
    RING = "ring"  # Ring attention for long sequences
    LOCAL = "local"  # Local attention (fixed window)
    GLOBAL = "global"  # Global + local attention
    DIFFERENTIAL = "differential"  # Differential attention
    DYNAMIC_SPARSE = "dynamic_sparse"  # Dynamic sparse attention


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

    def __post_init__(self):
        if not 0.0 <= self.sparsity_threshold <= 1.0:
            raise ValueError(
                f"sparsity_threshold must be in [0, 1], got {self.sparsity_threshold}"
            )
        if not 0.0 <= self.efficiency_target <= 1.0:
            raise ValueError(
                f"efficiency_target must be in [0, 1], got {self.efficiency_target}"
            )
        if not 0.0 <= self.min_sparsity <= 1.0:
            raise ValueError(f"min_sparsity must be in [0, 1], got {self.min_sparsity}")
        if not 0.0 <= self.max_sparsity <= 1.0:
            raise ValueError(f"max_sparsity must be in [0, 1], got {self.max_sparsity}")
        if self.min_sparsity > self.max_sparsity:
            raise ValueError(
                f"min_sparsity ({self.min_sparsity}) must be <= max_sparsity ({self.max_sparsity})"
            )


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

    def __post_init__(self):
        if not 0.0 <= self.entropy_threshold <= 1.0:
            raise ValueError(
                f"entropy_threshold must be in [0, 1], got {self.entropy_threshold}"
            )
        if not 0.0 <= self.memory_budget <= 1.0:
            raise ValueError(
                f"memory_budget must be in [0, 1], got {self.memory_budget}"
            )
        if not 0.0 <= self.quality_target <= 1.0:
            raise ValueError(
                f"quality_target must be in [0, 1], got {self.quality_target}"
            )
        if self.fp8_interval < 1:
            raise ValueError(f"fp8_interval must be >= 1, got {self.fp8_interval}")
        if self.calibration_samples < 1:
            raise ValueError(
                f"calibration_samples must be >= 1, got {self.calibration_samples}"
            )


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

    def __post_init__(self):
        if not 0.0 <= self.memory_fraction <= 1.0:
            raise ValueError(
                f"memory_fraction must be in [0, 1], got {self.memory_fraction}"
            )
        if not 0.0 <= self.fragmentation_threshold <= 1.0:
            raise ValueError(
                f"fragmentation_threshold must be in [0, 1], got {self.fragmentation_threshold}"
            )
        if self.max_memory_gb is not None and self.max_memory_gb <= 0:
            raise ValueError(
                f"max_memory_gb must be positive, got {self.max_memory_gb}"
            )
        if self.sequence_length_threshold < 1:
            raise ValueError(
                f"sequence_length_threshold must be >= 1, got {self.sequence_length_threshold}"
            )


@dataclass
class AttentionConfig:
    """Unified attention mechanism configuration."""

    mechanism: str = "flash_attention"
    sparse_enabled: bool = False
    sparsity_ratio: float = 0.5

    # Ring attention settings
    ring_enabled: bool = False
    max_sequence_length: int = 131072  # 128K; increase for ring attention workloads

    # Fusion settings
    fusion_enabled: bool = True
    fusion_strategy: str = "attention_ffn"

    # Context parallel settings
    context_parallel_size: int = 1

    def __post_init__(self):
        if not 0.0 <= self.sparsity_ratio <= 1.0:
            raise ValueError(
                f"sparsity_ratio must be in [0, 1], got {self.sparsity_ratio}"
            )
        if self.max_sequence_length < 1:
            raise ValueError(
                f"max_sequence_length must be >= 1, got {self.max_sequence_length}"
            )
        if self.context_parallel_size < 1:
            raise ValueError(
                f"context_parallel_size must be >= 1, got {self.context_parallel_size}"
            )


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
        if self.architecture in [
            NVIDIAArchitecture.HOPPER,
            NVIDIAArchitecture.BLACKWELL_DC,
            NVIDIAArchitecture.BLACKWELL_CONSUMER,
        ]:
            self.fp8_enabled = True
            self.tensor_core_version = (
                5
                if self.architecture
                in [
                    NVIDIAArchitecture.BLACKWELL_DC,
                    NVIDIAArchitecture.BLACKWELL_CONSUMER,
                ]
                else 4
            )
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

            # Blackwell Data Center (B100/B200/GB200/GB300)
            if any(name in device_name for name in ["B100", "B200", "GB200", "GB300"]):
                return NVIDIAArchitecture.BLACKWELL_DC

            # Blackwell Consumer (RTX 5090/5080)
            if any(name in device_name for name in ["RTX 50", "RTX 5090", "RTX 5080"]):
                return NVIDIAArchitecture.BLACKWELL_CONSUMER

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
            if device_props.major >= 12:
                return NVIDIAArchitecture.BLACKWELL_CONSUMER
            elif device_props.major >= 10:
                return NVIDIAArchitecture.BLACKWELL_DC
            elif device_props.major >= 9:
                return NVIDIAArchitecture.HOPPER
            elif device_props.major >= 8:
                return NVIDIAArchitecture.AMPERE
            elif device_props.major >= 7:
                return (
                    NVIDIAArchitecture.TURING
                    if device_props.minor >= 5
                    else NVIDIAArchitecture.VOLTA
                )
            else:
                return NVIDIAArchitecture.PASCAL

        except Exception:
            logger.debug("NVIDIA GPU architecture detection failed", exc_info=True)
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
    allocation_history_retention_seconds: int = (
        3600  # Keep allocation history for 1 hour
    )
    v6e_memory_gb: float | None = (
        None  # Override TPU v6e memory capacity (default: 32.0)
    )
    v7_memory_gb: float | None = (
        None  # Override TPU v7 memory capacity (default: 128.0)
    )

    # Validation settings
    enable_strict_validation: bool = (
        False  # Raise errors instead of warnings for validation failures
    )

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

                tpu_type = os.environ.get("TPU_TYPE", "")

                if "v5p" in tpu_type.lower():
                    return TPUVersion.V5P
                elif "v5e" in tpu_type.lower() or "v5lite" in tpu_type.lower():
                    return TPUVersion.V5E
                elif "v6e" in tpu_type.lower():
                    return TPUVersion.V6E
                elif "v4" in tpu_type.lower():
                    return TPUVersion.V4
                else:
                    # Default to v5e for unknown types
                    return TPUVersion.V5E
        except ImportError:
            # XLA not available
            pass
        except Exception:
            # Other detection errors
            logger.debug("TPU version detection failed", exc_info=True)
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
            logger.debug("TPU topology detection failed", exc_info=True)
            pass

        return TPUTopology.SINGLE  # Default fallback

    def _is_tpu_environment(self) -> bool:
        """Check if running in TPU environment (compatible with torch_xla 2.9+)."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # Get device using new API if available
            if hasattr(torch_xla, "device"):
                device = torch_xla.device()
            else:
                device = xm.xla_device()

            if device.type != "xla":
                return False

            # Check device hardware type
            if hasattr(xm, "xla_device_hw"):
                return xm.xla_device_hw(device) == "TPU"

            # Fallback: check environment variable
            import os

            return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
        except Exception:
            logger.debug("TPU environment check failed", exc_info=True)
            return False

    def _get_world_size(self) -> int:
        """Get world size (compatible with torch_xla 2.9+)."""
        try:
            # Try new runtime API first (torch_xla 2.9+)
            import torch_xla

            if hasattr(torch_xla, "runtime") and hasattr(
                torch_xla.runtime, "world_size"
            ):
                return torch_xla.runtime.world_size()

            # Try older runtime API
            try:
                from torch_xla import runtime as xr

                if hasattr(xr, "world_size"):
                    return xr.world_size()
            except ImportError:
                pass

            # Fall back to old xm API
            import torch_xla.core.xla_model as xm

            if hasattr(xm, "xrt_world_size"):
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
    hip_version: str = "auto"  # HIP version

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
    hip_compiler_cache_dir: str = os.path.join(tempfile.gettempdir(), "hip_cache")

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
        if self.architecture == AMDArchitecture.CDNA4:
            # MI350X/MI355X - latest data center (gfx950)
            self.enable_matrix_cores = True
            self.matrix_core_precision = "bf16"
            self.allow_bf16 = True
        elif self.architecture == AMDArchitecture.CDNA3:
            # MI300 series / MI325X - data center (gfx940/gfx942)
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

            if hasattr(torch, "hip") and torch.hip.is_available():
                device_props = torch.hip.get_device_properties(0)
                device_name = device_props.name.upper()

                # MI350X/MI355X (CDNA4, gfx950) — check before MI300 patterns
                if any(name in device_name for name in ["MI350", "MI355"]):
                    return AMDArchitecture.CDNA4

                # MI325X / MI300 series (CDNA3, gfx940/gfx942)
                if any(name in device_name for name in ["MI325", "MI300", "MI3"]):
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
            logger.debug("AMD GPU architecture detection failed", exc_info=True)

        # Default to CDNA2 (most common data center GPU)
        return AMDArchitecture.CDNA2


@dataclass
class TrainiumConfig:
    """AWS Trainium-specific configuration."""

    enabled: bool = True
    architecture: TrainiumArchitecture = TrainiumArchitecture.AUTO
    device_id: int = 0

    # Neuron compiler settings
    neuron_cc_flags: str = ""
    compilation_timeout_seconds: int = 600
    enable_graph_caching: bool = True

    # Precision
    precision: str = "bfloat16"  # bf16 is Trainium's native precision
    mixed_precision: bool = True
    enable_cfp8: bool = False  # Configurable FP8 (Trn1+)
    enable_mxfp8: bool = False  # Microscaling FP8 (Trn2+)
    enable_mxfp4: bool = False  # Microscaling FP4 (Trn3 only)

    # Memory
    memory_fraction: float = 0.90
    gradient_checkpointing: bool = True

    # Distributed
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1

    # Cache management
    cache_max_size: int = 100

    # Allocation tracking
    allocation_history_retention_seconds: int = 3600

    # Validation
    enable_strict_validation: bool = False

    # Monitoring
    monitoring_interval_seconds: float = 1.0
    monitoring_duration_seconds: float = 60.0

    def __post_init__(self):
        """Auto-detect architecture from environment."""
        if self.architecture == TrainiumArchitecture.AUTO:
            self.architecture = self._detect_architecture()

        # Configure precision based on architecture
        if self.architecture == TrainiumArchitecture.TRN3:
            # Trn3 supports MXFP4 and MXFP8
            pass  # Let user configure explicitly
        elif self.architecture == TrainiumArchitecture.TRN2:
            # Trn2 supports MXFP8 but not MXFP4
            self.enable_mxfp4 = False
        elif self.architecture in (
            TrainiumArchitecture.TRN1,
            TrainiumArchitecture.INF2,
        ):
            # Trn1/Inf2 support cFP8 only
            self.enable_mxfp8 = False
            self.enable_mxfp4 = False

    def _detect_architecture(self) -> TrainiumArchitecture:
        """Detect Trainium architecture from environment."""
        try:
            instance_type = os.environ.get("NEURON_INSTANCE_TYPE", "")
            if "trn1" in instance_type.lower():
                return TrainiumArchitecture.TRN1
            elif "trn2" in instance_type.lower():
                return TrainiumArchitecture.TRN2
            elif "trn3" in instance_type.lower():
                return TrainiumArchitecture.TRN3
            elif "inf2" in instance_type.lower():
                return TrainiumArchitecture.INF2

            # Fallback: try to detect from Neuron runtime
            neuron_cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
            if neuron_cores:
                # We're on a Neuron instance but don't know the type
                return TrainiumArchitecture.TRN2  # Default to most common
        except Exception:
            logger.debug("Trainium architecture detection failed", exc_info=True)
            pass

        return TrainiumArchitecture.TRN2  # Default fallback


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
    trainium: TrainiumConfig = field(default_factory=TrainiumConfig)

    # Tensor Core settings (general)
    tensor_cores_enabled: bool = True
    mixed_precision: bool = True

    # Compilation settings
    torch_compile: bool = True
    triton_enabled: bool = True
    flashlight_enabled: bool = False

    # Performance settings
    optimization_level: OptimizationLevel = OptimizationLevel.O2

    def __post_init__(self):
        """Auto-configure hardware settings based on detected capabilities."""
        # Auto-detect hardware backend if not explicitly set
        if self.backend == HardwareBackend.CUDA and not torch.cuda.is_available():
            # Try AMD ROCm detection
            if self._detect_amd_rocm():
                self.backend = HardwareBackend.AMD
            # Try Trainium detection (before TPU to avoid XLA misdetection)
            elif self._detect_trainium_environment():
                self.backend = HardwareBackend.TRAINIUM
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

        # Configure Trainium settings
        if self.backend == HardwareBackend.TRAINIUM:
            self.trainium.enabled = True
            # Disable incompatible settings for Trainium
            self.tensor_cores_enabled = (
                False  # Trainium uses NeuronCores, not Tensor Cores
            )
            self.triton_enabled = False  # Triton is CUDA-specific
        else:
            self.trainium.enabled = False

        # Configure TPU settings
        if self.backend == HardwareBackend.TPU:
            self.tpu.enabled = True
            # Disable incompatible settings for TPU
            self.tensor_cores_enabled = False  # TPU doesn't use Tensor Cores
            self.triton_enabled = False  # Triton is CUDA-specific
        else:
            self.tpu.enabled = False

    def _detect_amd_rocm(self) -> bool:
        """Check if AMD ROCm is available."""
        try:
            import torch

            if hasattr(torch, "hip") and torch.hip.is_available():
                return True
        except Exception:
            logger.debug("AMD ROCm detection failed", exc_info=True)
            pass
        return False

    def _detect_trainium_environment(self) -> bool:
        """Check if running on AWS Trainium/Inferentia2."""
        try:
            import torch_neuronx  # noqa: F401

            # Trainium uses XLA under the hood but is not a TPU
            pjrt = os.environ.get("PJRT_DEVICE", "").upper()
            if pjrt == "NEURON":
                return True
            if os.environ.get("NEURON_RT_VISIBLE_CORES"):
                return True
            return False
        except ImportError:
            return False

    def _detect_tpu_environment(self) -> bool:
        """Check if running in TPU environment (compatible with torch_xla 2.9+)."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # Get device using new API if available
            if hasattr(torch_xla, "device"):
                device = torch_xla.device()
            else:
                device = xm.xla_device()

            if device.type != "xla":
                return False

            # Check device hardware type
            if hasattr(xm, "xla_device_hw"):
                return xm.xla_device_hw(device) == "TPU"

            # Fallback: check environment variable
            import os

            return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
        except Exception:
            logger.debug("TPU environment detection failed", exc_info=True)
            return False


@dataclass
class QuantizationConfig:
    """Backend-aware quantization configuration."""

    enabled: bool = False
    strategy: str = "auto"  # "auto" or explicit format name
    format: str = "auto"  # QuantizationFormat value or "auto"
    calibration_samples: int = 512
    validate_after: bool = True
    in_place: bool = False

    def __post_init__(self):
        valid_strategies = {"auto", "manual"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {valid_strategies}, got '{self.strategy}'"
            )
        if self.calibration_samples < 1:
            raise ValueError(
                f"calibration_samples must be >= 1, got {self.calibration_samples}"
            )


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

    def __post_init__(self):
        if not 0.0 <= self.accuracy_threshold <= 1.0:
            raise ValueError(
                f"accuracy_threshold must be in [0, 1], got {self.accuracy_threshold}"
            )
        if not 0.0 <= self.performance_threshold <= 1.0:
            raise ValueError(
                f"performance_threshold must be in [0, 1], got {self.performance_threshold}"
            )
        if self.memory_threshold_gb <= 0:
            raise ValueError(
                f"memory_threshold_gb must be positive, got {self.memory_threshold_gb}"
            )
        if self.benchmark_iterations < 1:
            raise ValueError(
                f"benchmark_iterations must be >= 1, got {self.benchmark_iterations}"
            )
        if self.warmup_iterations < 0:
            raise ValueError(
                f"warmup_iterations must be >= 0, got {self.warmup_iterations}"
            )


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
    flash_attention_split_k: bool = True  # Enable Split-K for long sequences
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
            logger.debug(
                "Kernel config auto-detection failed, using safe defaults",
                exc_info=True,
            )
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
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Global settings
    device: torch.device = field(
        default_factory=lambda: TorchBridgeConfig._detect_device()
    )
    seed: int = 42
    debug: bool = False
    profile: bool = False

    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    experimental_features: bool = False

    @staticmethod
    def _detect_device() -> torch.device:
        """Detect the best available device."""
        # Try CUDA first
        if torch.cuda.is_available():
            return torch.device("cuda")

        # Try Trainium (before TPU — both use XLA)
        try:
            import torch_neuronx  # noqa: F401

            pjrt = os.environ.get("PJRT_DEVICE", "").upper()
            if pjrt == "NEURON" or os.environ.get("NEURON_RT_VISIBLE_CORES"):
                import torch_xla

                if hasattr(torch_xla, "device"):
                    return torch_xla.device()
                import torch_xla.core.xla_model as xm

                return xm.xla_device()
        except ImportError:
            pass
        except Exception:
            logger.debug("Trainium device detection failed", exc_info=True)
            pass

        # Try TPU (compatible with torch_xla 2.9+)
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # Get device using new API if available
            if hasattr(torch_xla, "device"):
                device = torch_xla.device()
            else:
                device = xm.xla_device()

            if device.type == "xla":
                # Check if it's actually a TPU
                if hasattr(xm, "xla_device_hw"):
                    if xm.xla_device_hw(device) == "TPU":
                        return device
                else:
                    # Assume it's a TPU if we got an XLA device
                    return device
        except ImportError:
            pass
        except Exception:
            logger.debug("TPU device detection failed", exc_info=True)
            pass

        # Fall back to CPU
        return torch.device("cpu")

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Sync hardware backend with detected device
        if self.device.type == "cuda":
            self.hardware.backend = HardwareBackend.CUDA
        elif str(self.device).startswith("xla"):
            # Distinguish Trainium from TPU — both use XLA
            if os.environ.get("PJRT_DEVICE", "").upper() == "NEURON" or os.environ.get(
                "NEURON_RT_VISIBLE_CORES"
            ):
                self.hardware.backend = HardwareBackend.TRAINIUM
            else:
                self.hardware.backend = HardwareBackend.TPU
        else:
            self.hardware.backend = HardwareBackend.CPU

        # Adjust hardware config based on device type
        if self.device.type == "cpu":
            self.hardware.tensor_cores_enabled = False
            self.precision.fp8_enabled = False
        elif str(self.device).startswith("xla"):  # TPU or Trainium
            self.hardware.tensor_cores_enabled = False  # Neither uses Tensor Cores
            self.precision.fp8_enabled = False  # Both use bfloat16 primarily
            self.hardware.triton_enabled = False  # Triton is CUDA-specific

        # Validate memory settings
        if self.memory.max_memory_gb is None:
            if self.device.type == "cuda":
                self.memory.max_memory_gb = torch.cuda.get_device_properties(
                    0
                ).total_memory / (1024**3)
            elif str(self.device).startswith("xla"):  # TPU
                # TPU memory varies by type, use reasonable default
                self.memory.max_memory_gb = 32.0  # Default for v5e

    @classmethod
    def for_inference(cls) -> "TorchBridgeConfig":
        """Create optimized configuration for inference."""
        config = cls()
        config.memory.gradient_checkpointing = False
        config.memory.deep_optimizer_states = False
        config.validation.enabled = False
        config.optimization_level = OptimizationLevel.O3
        return config

    @classmethod
    def for_training(cls) -> "TorchBridgeConfig":
        """Create optimized configuration for training."""
        config = cls()
        config.memory.gradient_checkpointing = True
        config.memory.deep_optimizer_states = True
        config.validation.enabled = True
        config.optimization_level = OptimizationLevel.O2
        return config

    @classmethod
    def for_development(cls) -> "TorchBridgeConfig":
        """Create configuration for development with debugging enabled."""
        config = cls()
        config.debug = True
        config.profile = True
        config.validation.strict_mode = True
        config.optimization_level = OptimizationLevel.O0
        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""

        def _convert_value(value, visited=None):
            if visited is None:
                visited = set()

            # Prevent recursion by tracking object IDs
            if id(value) in visited:
                return f"<circular reference to {type(value).__name__}>"

            if hasattr(value, "__dict__"):
                visited.add(id(value))
                # Handle nested dataclass objects
                nested_dict = {}
                for nested_key, nested_value in value.__dict__.items():
                    nested_dict[nested_key] = _convert_value(
                        nested_value, visited.copy()
                    )
                return nested_dict
            elif hasattr(value, "value"):  # Handle Enum objects
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
            if not hasattr(self, key):
                raise ValueError(f"Unknown configuration parameter: {key}")
            setattr(self, key, value)


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
