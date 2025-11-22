"""
Compiler Integration Module for PyTorch Optimization

This module implements state-of-the-art compiler integration techniques for 2025-2026:
- FlashLight Compiler Framework for automatic kernel generation
- PyGraph CUDA Graphs optimization with cost-benefit analysis
- Enhanced TorchInductor fusion beyond standard boundaries

These implementations bridge current PyTorch optimization gaps while preparing
for next-generation computing paradigms.
"""

from .flashlight_compiler import (
    FlashLightKernelCompiler,
    AttentionPattern,
    CompiledKernel,
    KernelCache
)

from .pygraph_optimizer import (
    PyGraphCUDAOptimizer,
    WorkloadAnalysis,
    GraphDeploymentStrategy,
    CUDAGraphManager
)

from .enhanced_fusion import (
    FusionBoundaryOptimizer,
    FusionPass,
    OptimizedFXGraph,
    FusionStrategy
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
    'CUDAGraphManager',

    # Enhanced Fusion
    'FusionBoundaryOptimizer',
    'FusionPass',
    'OptimizedFXGraph',
    'FusionStrategy'
]