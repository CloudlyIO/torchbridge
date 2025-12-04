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

__version__ = "0.1.55"

# Core optimization components (Phase 3 unified)
from .core import *

# Attention mechanisms (Phase 2 unified)
from .attention import *

# Specialized modules
from .precision import *
from .mixture_of_experts import *
from .advanced_memory import *
from .distributed_scale import *
from .testing_framework import *
from .utils import *

# Hardware optimization (Phase 3 unified)
from .hardware import *

# Optimization patterns (Phase 3 unified)
from .optimizations import *

# Backward compatibility is handled by individual modules through deprecation warnings
# No need to import non-existent modules here