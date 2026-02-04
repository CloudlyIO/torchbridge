"""
TorchBridge Core Module - Unified Core Optimization Components

This module consolidates all core PyTorch optimization functionality:
- Optimized layer implementations
- Basic optimized components
- Hardware detection and auto-optimization
- Performance tracking and regression detection
"""

# Configuration system (Phase 3)
from .config import (
    AMDArchitecture,
    AMDConfig,
    AttentionConfig,
    # Attention configs (centralized from attention module)
    AttentionPatterns,
    DistributedConfig,
    DynamicSparseConfig,
    FP8AttentionConfig,
    HardwareBackend,
    HardwareConfig,
    KernelConfig,
    MemoryConfig,
    NVIDIAArchitecture,
    NVIDIAConfig,
    OptimizationLevel,
    PrecisionConfig,
    # Enums
    PrecisionFormat,
    RingAttentionConfig,
    TorchBridgeConfig,
    TPUCompilationMode,
    TPUConfig,
    TPUTopology,
    TPUVersion,
    ValidationConfig,
    configure,
    # Helpers
    get_config,
    set_config,
)

# Error handling framework (v0.3.11)
from .errors import (
    CompilationError,
    ConfigValidationError,
    ContainerError,
    DeploymentError,
    ExportError,
    FusionError,
    HardwareCapabilityError,
    HardwareDetectionError,
    HardwareError,
    HardwareNotFoundError,
    HealthCheckError,
    InputValidationError,
    MetricsError,
    ModelValidationError,
    MonitoringError,
    OptimizationError,
    PrecisionError,
    ServingError,
    TorchBridgeError,
    ValidationError,
    format_error_chain,
    raise_or_warn,
)

# Hardware detection and auto-optimization
from .hardware_detector import (
    HardwareDetector,
    HardwareProfile,
    HardwareType,
    OptimizationCapability,
    detect_hardware,
    get_hardware_detector,
    get_optimal_backend,
)

# Optimized layers
from .optimized_layers.activation_functions import (
    FusedGELU,
    FusedReLU,
    FusedSwiGLU,
    create_optimized_activation,
)
from .optimized_layers.linear_transformations import (
    FusedLinearSequence,
    GroupedLinearTransformation,
    MemoryEfficientLinear,
    MultiHeadLinearProjection,
)

# Performance tracking and regression detection
from .performance_tracker import (
    MetricType,
    PerformanceMetrics,
    PerformanceTracker,
    RegressionResult,
    RegressionSeverity,
    detect_regression,
    get_performance_tracker,
    track_performance,
)

# NOTE: Core optimized layers are production-ready. Additional layer types can be
# added by extending the patterns in optimized_layers/ module.

# Basic components - using actual available exports
try:
    from .components.basic_optimized import *  # noqa: F403
    from .components.jit_optimized import *  # noqa: F403
except ImportError as e:
    import warnings
    warnings.warn(f"Some components not available: {e}", stacklevel=2)
    pass

__all__ = [
    # Hardware detection and auto-optimization
    'HardwareDetector', 'HardwareProfile', 'HardwareType', 'OptimizationCapability',
    'detect_hardware', 'get_optimal_backend', 'get_hardware_detector',

    # Performance tracking and regression detection
    'PerformanceTracker', 'PerformanceMetrics', 'RegressionResult',
    'MetricType', 'RegressionSeverity', 'get_performance_tracker',
    'track_performance', 'detect_regression',

    # Configuration system
    'TorchBridgeConfig', 'PrecisionConfig', 'MemoryConfig', 'AttentionConfig',
    'HardwareConfig', 'NVIDIAConfig', 'TPUConfig', 'AMDConfig',
    'DistributedConfig', 'ValidationConfig', 'KernelConfig',
    'PrecisionFormat', 'OptimizationLevel', 'HardwareBackend',
    'NVIDIAArchitecture', 'TPUVersion', 'TPUTopology', 'TPUCompilationMode', 'AMDArchitecture',
    'AttentionPatterns', 'FP8AttentionConfig', 'DynamicSparseConfig', 'RingAttentionConfig',
    'get_config', 'set_config', 'configure',

    # Error handling framework
    'TorchBridgeError', 'ValidationError', 'ConfigValidationError', 'InputValidationError',
    'ModelValidationError', 'HardwareError', 'HardwareDetectionError', 'HardwareNotFoundError',
    'HardwareCapabilityError', 'OptimizationError', 'CompilationError', 'FusionError',
    'PrecisionError', 'DeploymentError', 'ExportError', 'ServingError', 'ContainerError',
    'MonitoringError', 'MetricsError', 'HealthCheckError', 'raise_or_warn', 'format_error_chain',

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
            if hasattr(attr, '__module__') and 'torchbridge.core' in str(getattr(attr, '__module__', '')):
                __all__.append(attr_name)
        except Exception:
            pass

# Backward compatibility
import warnings  # noqa: E402


def _deprecation_warning(old_path: str, new_path: str):
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Please use 'from torchbridge.core import ...' instead of '{new_path}'.",
        DeprecationWarning,
        stacklevel=3
    )


# =============================================================================
# DEPRECATED: Legacy import support
#
# The following classes and sys.modules manipulation provide backward
# compatibility for old import paths:
#   - torchbridge.compiler_integration → torchbridge.core
#   - torchbridge.compiler_optimized → torchbridge.core
#   - torchbridge.components → torchbridge.core
#
# Migration guide: Use 'from torchbridge.core import ...' instead.
# =============================================================================

class _LegacyImportHelper:
    def __getattr__(self, name):
        if name in ['compiler_integration', 'compiler_optimized', 'components']:
            _deprecation_warning(f'torchbridge.{name}', 'torchbridge.core')
            return self
        # Import from current module instead of recursion
        import torchbridge.core as core_module
        if hasattr(core_module, name):
            return getattr(core_module, name)
        raise AttributeError(f"module has no attribute '{name}'")

import sys  # noqa: E402

sys.modules['torchbridge.compiler_integration'] = _LegacyImportHelper()
sys.modules['torchbridge.compiler_optimized'] = _LegacyImportHelper()
sys.modules['torchbridge.components'] = _LegacyImportHelper()
