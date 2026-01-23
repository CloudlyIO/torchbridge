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

def _deprecation_warning(old_path: str, new_path: str):
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Please use 'from {new_path} import ...' instead.",
        DeprecationWarning,
        stacklevel=3
    )

class _LegacyGPUIntegrationHelper:
    """Helper for backward compatibility with old gpu_integration import path"""
    def __getattr__(self, name):
        _deprecation_warning('kernel_pytorch.gpu_integration', 'kernel_pytorch.hardware.gpu')
        from kernel_pytorch.hardware import gpu
        return getattr(gpu, name)

class _LegacyHardwareAbstractionHelper:
    """Helper for backward compatibility with old hardware_abstraction import path"""
    def __getattr__(self, name):
        _deprecation_warning('kernel_pytorch.hardware_abstraction', 'kernel_pytorch.hardware.abstraction')
        from kernel_pytorch.hardware import abstraction
        return getattr(abstraction, name)

class _LegacyCUDAKernelsHelper:
    """Helper for backward compatibility with old cuda_kernels import path"""
    def __getattr__(self, name):
        _deprecation_warning('kernel_pytorch.cuda_kernels', 'kernel_pytorch.hardware.gpu')
        # cuda_kernels was removed - forward to gpu module
        from kernel_pytorch.hardware import gpu
        return getattr(gpu, name)

# Legacy import support - register parent modules
sys.modules['kernel_pytorch.gpu_integration'] = _LegacyGPUIntegrationHelper()
sys.modules['kernel_pytorch.hardware_abstraction'] = _LegacyHardwareAbstractionHelper()
sys.modules['kernel_pytorch.cuda_kernels'] = _LegacyCUDAKernelsHelper()

# Also register submodules for deep imports (e.g., kernel_pytorch.hardware_abstraction.privateuse1_integration)
try:
    from kernel_pytorch.hardware import abstraction
    sys.modules['kernel_pytorch.hardware_abstraction.hal_core'] = abstraction.hal_core if hasattr(abstraction, 'hal_core') else abstraction
    sys.modules['kernel_pytorch.hardware_abstraction.privateuse1_integration'] = __import__('kernel_pytorch.hardware.abstraction.privateuse1_integration', fromlist=[''])
    sys.modules['kernel_pytorch.hardware_abstraction.vendor_adapters'] = __import__('kernel_pytorch.hardware.abstraction.vendor_adapters', fromlist=[''])
except ImportError:
    pass  # Abstraction module may not be fully available