"""
AMD ROCm Backend for TorchBridge

AMD GPU support through ROCm/HIP, targeting CDNA2 through CDNA4
data center and RDNA consumer architectures.

Components:
- AMDBackend: Device detection, model preparation, optimization orchestration
- AMDAdapter: Multi-level optimization (conservative/balanced/aggressive)

Supported Hardware:
- AMD MI200 series (CDNA2): MI210, MI250, MI250X
- AMD MI300 series (CDNA3): MI300A, MI300X, MI325X
- AMD MI350 series (CDNA4): MI350X, MI355X
- AMD RDNA consumer GPUs: RX 7000 series (RDNA3), RX 6000 series (RDNA2)

"""

from .amd_adapter import AMDAdapter
from .amd_backend import AMDBackend

__all__ = [
    "AMDBackend",
    "AMDAdapter",
]
