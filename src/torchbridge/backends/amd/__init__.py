"""
AMD ROCm Backend for TorchBridge

This module provides comprehensive AMD GPU support through ROCm/HIP,
targeting CDNA2 (MI200) and CDNA3 (MI300) architectures.

Architecture:
- AMDBackend: Main backend orchestrator for AMD GPUs
- AMDOptimizer: Multi-level optimization (conservative/balanced/aggressive)
- ROCmCompiler: HIP kernel compilation and optimization
- AMDMemoryManager: GPU memory management with HBM pooling
- HIPUtilities: Device coordination and profiling

Supported Hardware:
- AMD MI200 series (CDNA2): MI210, MI250, MI250X
- AMD MI300 series (CDNA3): MI300A, MI300X

Version: 0.4.19
"""

from .amd_backend import AMDBackend
from .amd_optimizer import AMDOptimizer
from .hip_utilities import HIPUtilities
from .memory_manager import AMDMemoryManager
from .rocm_compiler import ROCmCompiler

__all__ = [
    "AMDBackend",
    "AMDOptimizer",
    "ROCmCompiler",
    "AMDMemoryManager",
    "HIPUtilities",
]

__version__ = "0.4.42"
