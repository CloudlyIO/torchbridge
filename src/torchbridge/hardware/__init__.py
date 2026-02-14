"""
TorchBridge Unified Hardware Module

This module consolidates all hardware-specific backend components:
- GPU integration and cross-backend patterns
- Hardware abstraction layer for multi-vendor support
- Custom CUDA kernels and kernel interfaces
- Performance profiling and measurement tools

Note: This module unifies hardware-specific backend components.

Development Status:
- CUDA kernel implementations: Educational patterns provided; production use should
  leverage Triton or torch.compile() for automatic kernel generation.
- Vendor adapter support: NVIDIA (full), AMD ROCm (full),
  TPU (full). ASIC/neuromorphic adapters are placeholder implementations.
"""

# GPU integration and backend support - import what's available
try:
    from .gpu import *  # noqa: F403
except ImportError as e:
    import warnings
    warnings.warn(f"GPU module import failed: {e}", stacklevel=2)
    pass

# Hardware abstraction layer - import what's available
try:
    from .abstraction import *  # noqa: F403
except ImportError as e:
    import warnings
    warnings.warn(f"Hardware abstraction import failed: {e}", stacklevel=2)
    pass

# CUDA kernels and interfaces (legacy files)
# Note: .cu and .cpp files are kept as-is for compilation

__all__ = []

# Dynamically add available exports
import logging
import sys

logger = logging.getLogger(__name__)

current_module = sys.modules[__name__]
for attr_name in dir(current_module):
    if not attr_name.startswith('_') and attr_name not in ['warnings', 'sys']:
        try:
            attr = getattr(current_module, attr_name)
            if hasattr(attr, '__module__') and ('torchbridge.hardware' in str(getattr(attr, '__module__', '')) or
                                                'torchbridge.gpu_integration' in str(getattr(attr, '__module__', '')) or
                                                'torchbridge.hardware_abstraction' in str(getattr(attr, '__module__', ''))):
                __all__.append(attr_name)
        except Exception:
            logger.debug("Failed to inspect attribute '%s' for __all__", attr_name, exc_info=True)
            pass

# Backward compatibility support
import sys  # noqa: E402
import warnings  # noqa: E402


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
        _deprecation_warning('torchbridge.gpu_integration', 'torchbridge.hardware.gpu')
        from torchbridge.hardware import gpu
        return getattr(gpu, name)

class _LegacyHardwareAbstractionHelper:
    """Helper for backward compatibility with old hardware_abstraction import path"""
    def __getattr__(self, name):
        _deprecation_warning('torchbridge.hardware_abstraction', 'torchbridge.hardware.abstraction')
        from torchbridge.hardware import abstraction
        return getattr(abstraction, name)

class _LegacyCUDAKernelsHelper:
    """Helper for backward compatibility with old cuda_kernels import path"""
    def __getattr__(self, name):
        _deprecation_warning('torchbridge.cuda_kernels', 'torchbridge.hardware.gpu')
        # cuda_kernels was removed - forward to gpu module
        from torchbridge.hardware import gpu
        return getattr(gpu, name)

# Legacy import support - register parent modules
sys.modules['torchbridge.gpu_integration'] = _LegacyGPUIntegrationHelper()  # type: ignore[assignment]
sys.modules['torchbridge.hardware_abstraction'] = _LegacyHardwareAbstractionHelper()  # type: ignore[assignment]
sys.modules['torchbridge.cuda_kernels'] = _LegacyCUDAKernelsHelper()  # type: ignore[assignment]

# Also register submodules for deep imports (e.g., torchbridge.hardware_abstraction.privateuse1_integration)
try:
    from torchbridge.hardware import abstraction
    sys.modules['torchbridge.hardware_abstraction.hal_core'] = abstraction.hal_core if hasattr(abstraction, 'hal_core') else abstraction
    sys.modules['torchbridge.hardware_abstraction.privateuse1_integration'] = __import__('torchbridge.hardware.abstraction.privateuse1_integration', fromlist=[''])
    sys.modules['torchbridge.hardware_abstraction.vendor_adapters'] = __import__('torchbridge.hardware.abstraction.vendor_adapters', fromlist=[''])
except ImportError:
    pass  # Abstraction module may not be fully available
