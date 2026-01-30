"""
Compiler Integration Module for PyTorch Optimization

This module implements state-of-the-art compiler integration techniques for 2025-2026:
- FlashLight Compiler Framework for automatic kernel generation
- PyGraph CUDA Graphs optimization with cost-benefit analysis
- Enhanced TorchInductor fusion beyond standard boundaries

These implementations bridge current PyTorch optimization gaps while preparing
for next-generation computing paradigms.
"""

from .enhanced_fusion import (
    FusionBoundaryOptimizer,
    FusionPass,
    FusionStrategy,
    OptimizedFXGraph,
)
from .flashlight_compiler import (
    AttentionPattern,
    CompiledKernel,
    FlashLightKernelCompiler,
    KernelCache,
)
from .pygraph_optimizer import (
    CUDAGraphState,
    GraphDeploymentStrategy,
    PyGraphCUDAOptimizer,
    WorkloadAnalysis,
)
from .pygraph_optimizer import (
    CUDAGraphState as CUDAGraphManager,  # Backward compatibility alias
)

__all__ = [
    # FlashLight Compiler
    'FlashLightKernelCompiler',
    'AttentionPattern',
    'CompiledKernel',
    'KernelCache',

    # PyGraph Optimization
    'PyGraphCUDAOptimizer',
    'WorkloadAnalysis',
    'GraphDeploymentStrategy',
    'CUDAGraphState',
    'CUDAGraphManager',  # Backward compatibility alias

    # Enhanced Fusion
    'FusionBoundaryOptimizer',
    'FusionPass',
    'OptimizedFXGraph',
    'FusionStrategy'
]
