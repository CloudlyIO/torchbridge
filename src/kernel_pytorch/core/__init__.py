"""
KernelPyTorch Core Module - Unified Core Optimization Components

This module consolidates all core PyTorch optimization functionality:
- Compiler integrations (FlashLight, PyGraph)
- Optimized layer implementations
- Basic optimized components
- Hardware detection and auto-optimization
- Performance tracking and regression detection

Note: This module unifies previously separate compiler and component modules
"""

# Compiler integrations
from .compilers.flashlight_compiler import (
    FlashLightKernelCompiler,
    AttentionPattern,
    CompiledKernel,
    KernelCache
)
from .compilers.pygraph_optimizer import (
    PyGraphCUDAOptimizer,
    WorkloadAnalysis,
    GraphDeploymentStrategy,
    CUDAGraphManager
)
from .compilers.enhanced_fusion import (
    FusionBoundaryOptimizer,
    FusionPass,
    OptimizedFXGraph,
    FusionStrategy
)

# Hardware detection and auto-optimization
from .hardware_detector import (
    HardwareDetector,
    HardwareProfile,
    HardwareType,
    OptimizationCapability,
    detect_hardware,
    get_optimal_backend,
    get_hardware_detector
)

# Performance tracking and regression detection
from .performance_tracker import (
    PerformanceTracker,
    PerformanceMetrics,
    RegressionResult,
    MetricType,
    RegressionSeverity,
    get_performance_tracker,
    track_performance,
    detect_regression
)

# Optimized layers
from .optimized_layers.activation_functions import FusedGELU, FusedSwiGLU, FusedReLU, create_optimized_activation
from .optimized_layers.linear_transformations import (
    MultiHeadLinearProjection,
    GroupedLinearTransformation,
    MemoryEfficientLinear,
    FusedLinearSequence,
    compiled_linear_gelu
)
# TODO: Complete implementation of all optimized layer components

# Basic components - using actual available exports
try:
    from .components.basic_optimized import *
    from .components.jit_optimized import *
except ImportError as e:
    import warnings
    warnings.warn(f"Some components not available: {e}")
    pass

__all__ = [
    # Compiler integrations
    'FlashLightKernelCompiler', 'AttentionPattern', 'CompiledKernel', 'KernelCache',
    'PyGraphCUDAOptimizer', 'WorkloadAnalysis', 'GraphDeploymentStrategy', 'CUDAGraphManager',
    'FusionBoundaryOptimizer', 'FusionPass', 'OptimizedFXGraph', 'FusionStrategy',

    # Hardware detection and auto-optimization
    'HardwareDetector', 'HardwareProfile', 'HardwareType', 'OptimizationCapability',
    'detect_hardware', 'get_optimal_backend', 'get_hardware_detector',

    # Performance tracking and regression detection
    'PerformanceTracker', 'PerformanceMetrics', 'RegressionResult',
    'MetricType', 'RegressionSeverity', 'get_performance_tracker',
    'track_performance', 'detect_regression',

    # Optimized layers - available implementations
    'FusedGELU', 'FusedSwiGLU', 'FusedReLU', 'create_optimized_activation',
    'MultiHeadLinearProjection', 'GroupedLinearTransformation', 'MemoryEfficientLinear', 'FusedLinearSequence',
]

# Add dynamically available components
import sys
current_module = sys.modules[__name__]
for attr_name in dir(current_module):
    if not attr_name.startswith('_') and attr_name not in __all__:
        try:
            attr = getattr(current_module, attr_name)
            if hasattr(attr, '__module__') and 'kernel_pytorch.core' in str(getattr(attr, '__module__', '')):
                __all__.append(attr_name)
        except:
            pass

# Backward compatibility
import warnings

def _deprecation_warning(old_path: str, new_path: str):
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Please use 'from kernel_pytorch.core import ...' instead of '{new_path}'.",
        DeprecationWarning,
        stacklevel=3
    )

# Legacy import support with deprecation warnings
class _LegacyImportHelper:
    def __getattr__(self, name):
        if name in ['compiler_integration', 'compiler_optimized', 'components']:
            _deprecation_warning(f'kernel_pytorch.{name}', 'kernel_pytorch.core')
            return self
        # Import from current module instead of recursion
        import kernel_pytorch.core as core_module
        if hasattr(core_module, name):
            return getattr(core_module, name)
        raise AttributeError(f"module has no attribute '{name}'")

import sys
sys.modules['kernel_pytorch.compiler_integration'] = _LegacyImportHelper()
sys.modules['kernel_pytorch.compiler_optimized'] = _LegacyImportHelper()
sys.modules['kernel_pytorch.components'] = _LegacyImportHelper()