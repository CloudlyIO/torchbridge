"""
Kernel-Optimized PyTorch Components

This package provides progressively optimized neural network components
that demonstrate how to align PyTorch models with GPU computation patterns.

Optimization Levels:
- Level 1: PyTorch native (cuDNN/cuBLAS optimized)
- Level 2: TorchScript JIT compilation
- Level 3: torch.compile (Inductor backend)
- Level 4: Triton kernels (Python-based CUDA)
- Level 5: Custom CUDA kernels (maximum control)
"""

__version__ = "0.1.0"

from .components import *
from .utils import *