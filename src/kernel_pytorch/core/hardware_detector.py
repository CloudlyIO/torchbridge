"""
Hardware Detection Module

Automatically detects available hardware and provides capability profiles
for intelligent optimization selection.
"""

import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .config import NVIDIAArchitecture, TPUVersion


class HardwareType(Enum):
    """Available hardware types."""
    NVIDIA_GPU = "nvidia_gpu"
    TPU = "tpu"
    AMD_GPU = "amd_gpu"
    INTEL_GPU = "intel_gpu"
    CPU = "cpu"


class OptimizationCapability(Enum):
    """Optimization capabilities available on detected hardware."""
    FP8_TRAINING = "fp8_training"
    FLASH_ATTENTION_3 = "flash_attention_3"
    XLA_COMPILATION = "xla_compilation"
    TENSOR_CORES = "tensor_cores"
    MIXED_PRECISION = "mixed_precision"
    KERNEL_FUSION = "kernel_fusion"


@dataclass
class HardwareProfile:
    """Complete hardware profile with capabilities."""

    # Primary hardware type
    hardware_type: HardwareType
    device_name: str
    device_count: int

    # Specific architecture details
    nvidia_architecture: Optional[NVIDIAArchitecture] = None
    tpu_version: Optional[TPUVersion] = None
    compute_capability: Optional[tuple] = None

    # Available capabilities
    capabilities: List[OptimizationCapability] = None

    # Memory information
    total_memory_gb: float = 0.0

    # Additional metadata
    cuda_version: Optional[str] = None
    xla_available: bool = False

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []

    def has_capability(self, capability: OptimizationCapability) -> bool:
        """Check if hardware has specific capability."""
        return capability in self.capabilities

    def is_nvidia_h100_or_better(self) -> bool:
        """Check if NVIDIA H100 or Blackwell."""
        return self.nvidia_architecture in [
            NVIDIAArchitecture.HOPPER,
            NVIDIAArchitecture.BLACKWELL
        ]

    def is_high_end_tpu(self) -> bool:
        """Check if high-end TPU (v5p, v6e, v7)."""
        return self.tpu_version in [
            TPUVersion.V5P,
            TPUVersion.V6E,
            TPUVersion.V7
        ]

    def supports_advanced_optimization(self) -> bool:
        """Check if hardware supports advanced optimizations."""
        return (
            self.has_capability(OptimizationCapability.FP8_TRAINING) or
            self.has_capability(OptimizationCapability.XLA_COMPILATION) or
            self.is_nvidia_h100_or_better() or
            self.is_high_end_tpu()
        )


class HardwareDetector:
    """
    Automatic hardware detection and capability profiling.

    Detects available hardware and provides comprehensive profiles
    for intelligent optimization selection.
    """

    def __init__(self):
        """Initialize hardware detector."""
        self._cached_profile: Optional[HardwareProfile] = None

    def detect(self, force_redetect: bool = False) -> HardwareProfile:
        """
        Detect hardware and return profile.

        Args:
            force_redetect: Force re-detection even if cached

        Returns:
            HardwareProfile with detected capabilities
        """
        if self._cached_profile and not force_redetect:
            return self._cached_profile

        # Try detection in order of preference
        profile = (
            self._detect_nvidia_gpu() or
            self._detect_tpu() or
            self._detect_cpu()
        )

        self._cached_profile = profile
        return profile

    def _detect_nvidia_gpu(self) -> Optional[HardwareProfile]:
        """Detect NVIDIA GPU hardware."""
        if not torch.cuda.is_available():
            return None

        try:
            device_count = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)

            # Detect architecture
            architecture = self._detect_nvidia_architecture(props)

            # Determine capabilities
            capabilities = [OptimizationCapability.MIXED_PRECISION]

            # Tensor Cores (Volta+, compute capability 7.0+)
            if props.major >= 7:
                capabilities.append(OptimizationCapability.TENSOR_CORES)

            # FP8 training (Hopper+, compute capability 9.0+)
            if props.major >= 9:
                capabilities.append(OptimizationCapability.FP8_TRAINING)
                capabilities.append(OptimizationCapability.FLASH_ATTENTION_3)

            # Kernel fusion available on modern GPUs
            if props.major >= 7:
                capabilities.append(OptimizationCapability.KERNEL_FUSION)

            return HardwareProfile(
                hardware_type=HardwareType.NVIDIA_GPU,
                device_name=props.name,
                device_count=device_count,
                nvidia_architecture=architecture,
                compute_capability=(props.major, props.minor),
                capabilities=capabilities,
                total_memory_gb=props.total_memory / 1024**3,
                cuda_version=torch.version.cuda if hasattr(torch.version, 'cuda') else None
            )

        except Exception as e:
            # Detection failed, return None
            return None

    def _detect_nvidia_architecture(self, props) -> NVIDIAArchitecture:
        """Detect specific NVIDIA architecture."""
        name = props.name.lower()
        major, minor = props.major, props.minor

        # Blackwell (compute capability 10.0+)
        if major >= 10:
            return NVIDIAArchitecture.BLACKWELL

        # Hopper - H100 (compute capability 9.0)
        if major == 9:
            return NVIDIAArchitecture.HOPPER

        # Ampere - A100 (compute capability 8.0, 8.6)
        if major == 8:
            return NVIDIAArchitecture.AMPERE

        # Older architectures
        return NVIDIAArchitecture.PASCAL

    def _detect_tpu(self) -> Optional[HardwareProfile]:
        """Detect TPU hardware."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm

            # TPU is available - get device count (compatible with torch_xla 2.9+)
            device_count = 1
            if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'device_count'):
                device_count = torch_xla.runtime.device_count()
            elif hasattr(xm, 'xla_device_count'):
                device_count = xm.xla_device_count()

            # Detect TPU version from environment
            import os
            tpu_type = os.environ.get('TPU_TYPE', 'auto')
            tpu_version = self._parse_tpu_version(tpu_type)

            # TPU capabilities
            capabilities = [
                OptimizationCapability.XLA_COMPILATION,
                OptimizationCapability.MIXED_PRECISION,
            ]

            # High-end TPUs have additional capabilities
            if tpu_version in [TPUVersion.V5P, TPUVersion.V6E, TPUVersion.V7]:
                capabilities.append(OptimizationCapability.KERNEL_FUSION)

            return HardwareProfile(
                hardware_type=HardwareType.TPU,
                device_name=f"TPU {tpu_version.value}",
                device_count=device_count,
                tpu_version=tpu_version,
                capabilities=capabilities,
                xla_available=True
            )

        except ImportError:
            return None

    def _parse_tpu_version(self, tpu_type: str) -> TPUVersion:
        """Parse TPU version from type string."""
        tpu_type_lower = tpu_type.lower()

        if 'v7' in tpu_type_lower:
            return TPUVersion.V7
        elif 'v6e' in tpu_type_lower:
            return TPUVersion.V6E
        elif 'v5p' in tpu_type_lower:
            return TPUVersion.V5P
        elif 'v5e' in tpu_type_lower:
            return TPUVersion.V5E
        elif 'v4' in tpu_type_lower:
            return TPUVersion.V4
        else:
            return TPUVersion.AUTO

    def _detect_cpu(self) -> HardwareProfile:
        """Detect CPU as fallback."""
        return HardwareProfile(
            hardware_type=HardwareType.CPU,
            device_name="CPU",
            device_count=1,
            capabilities=[],  # Minimal capabilities on CPU
        )

    def get_optimal_backend(self, profile: Optional[HardwareProfile] = None) -> str:
        """
        Get optimal backend name based on hardware profile.

        Args:
            profile: Hardware profile (auto-detect if None)

        Returns:
            Backend name: 'nvidia', 'tpu', or 'cpu'
        """
        if profile is None:
            profile = self.detect()

        if profile.hardware_type == HardwareType.NVIDIA_GPU:
            return 'nvidia'
        elif profile.hardware_type == HardwareType.TPU:
            return 'tpu'
        else:
            return 'cpu'

    def get_recommended_optimization_level(
        self,
        profile: Optional[HardwareProfile] = None
    ) -> str:
        """
        Get recommended optimization level based on hardware.

        Args:
            profile: Hardware profile (auto-detect if None)

        Returns:
            Optimization level: 'conservative', 'balanced', or 'aggressive'
        """
        if profile is None:
            profile = self.detect()

        # Aggressive for high-end hardware
        if profile.is_nvidia_h100_or_better() or profile.is_high_end_tpu():
            return 'aggressive'

        # Balanced for modern GPUs/TPUs
        if profile.supports_advanced_optimization():
            return 'balanced'

        # Conservative for older hardware or CPU
        return 'conservative'


# Global detector instance
_global_detector: Optional[HardwareDetector] = None


def get_hardware_detector() -> HardwareDetector:
    """Get global hardware detector instance."""
    global _global_detector
    if _global_detector is None:
        _global_detector = HardwareDetector()
    return _global_detector


def detect_hardware() -> HardwareProfile:
    """Convenience function to detect hardware."""
    return get_hardware_detector().detect()


def get_optimal_backend() -> str:
    """Convenience function to get optimal backend."""
    return get_hardware_detector().get_optimal_backend()
