"""
AMD ROCm Backend for TorchBridge

This module provides comprehensive AMD GPU support through ROCm/HIP,
targeting CDNA2 through CDNA4 data center and RDNA consumer architectures.

Architecture:
- AMDBackend: Main backend orchestrator for AMD GPUs
- AMDAdapter: Multi-level backend tuning (conservative/balanced/aggressive)
- ROCmCompiler: HIP kernel compilation and backend-aware generation
- AMDMemoryManager: GPU memory management with HBM pooling
- HIPUtilities: Device coordination and profiling

Supported Hardware:
- AMD MI200 series (CDNA2): MI210, MI250, MI250X
- AMD MI300 series (CDNA3): MI300A, MI300X, MI325X
- AMD MI350 series (CDNA4): MI350X, MI355X
- AMD RDNA consumer GPUs: RX 7000 series (RDNA3), RX 6000 series (RDNA2)

"""

from .amd_adapter import AMDAdapter
from .amd_backend import AMDBackend
from .hip_utilities import HIPUtilities
from .memory_manager import AMDMemoryManager
from .rocm_compiler import ROCmCompiler

__all__ = [
    "AMDBackend",
    "AMDAdapter",
    "ROCmCompiler",
    "AMDMemoryManager",
    "HIPUtilities",
]

