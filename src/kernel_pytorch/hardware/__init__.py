"""
KernelPyTorch Unified Hardware Module

This module consolidates all hardware-specific optimizations:
- GPU integration and optimization patterns
- Hardware abstraction layer for multi-vendor support
- Custom CUDA kernels and kernel interfaces
- Performance profiling and optimization tools

Note: This module unifies hardware-specific optimization components
TODO: Complete CUDA kernel implementations for all GPU vendors
TODO: Implement comprehensive vendor adapter support (currently partial)
"""

# GPU integration and optimization - import what's available
try:
    from .gpu import *
except ImportError as e:
    import warnings
    warnings.warn(f"GPU module import failed: {e}")
    pass

# Hardware abstraction layer - import what's available
try:
    from .abstraction import *
except ImportError as e:
    import warnings
    warnings.warn(f"Hardware abstraction import failed: {e}")
    pass

# CUDA kernels and interfaces (legacy files)
# Note: .cu and .cpp files are kept as-is for compilation

__all__ = []

# Dynamically add available exports
import sys
current_module = sys.modules[__name__]
for attr_name in dir(current_module):
    if not attr_name.startswith('_') and attr_name not in ['warnings', 'sys']:
        try:
            attr = getattr(current_module, attr_name)
            if hasattr(attr, '__module__') and ('kernel_pytorch.hardware' in str(getattr(attr, '__module__', '')) or
                                                'kernel_pytorch.gpu_integration' in str(getattr(attr, '__module__', '')) or
                                                'kernel_pytorch.hardware_abstraction' in str(getattr(attr, '__module__', ''))):
                __all__.append(attr_name)
        except Exception:
            pass

# Backward compatibility support
import warnings
import sys

def _deprecation_warning(old_path: str):
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Please use 'from kernel_pytorch.hardware import ...' instead.",
        DeprecationWarning,
        stacklevel=3
    )

class _LegacyHardwareImportHelper:
    """Helper for backward compatibility with old hardware import paths"""
    def __getattr__(self, name):
        if 'gpu_integration' in str(self.__class__):
            _deprecation_warning('kernel_pytorch.gpu_integration')
        elif 'hardware_abstraction' in str(self.__class__):
            _deprecation_warning('kernel_pytorch.hardware_abstraction')
        elif 'cuda_kernels' in str(self.__class__):
            _deprecation_warning('kernel_pytorch.cuda_kernels')
        return getattr(self, name)

# Legacy import support
sys.modules['kernel_pytorch.gpu_integration'] = _LegacyHardwareImportHelper()
sys.modules['kernel_pytorch.hardware_abstraction'] = _LegacyHardwareImportHelper()
sys.modules['kernel_pytorch.cuda_kernels'] = _LegacyHardwareImportHelper()