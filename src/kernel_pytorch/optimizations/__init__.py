"""
KernelPyTorch Unified Optimizations Module

This module consolidates all optimization strategies and patterns:
- Optimization patterns for common use cases
- Next-generation cutting-edge optimizations
- Advanced optimization strategies

Note: This module unifies optimization patterns and next-gen optimizations
"""

# Optimization patterns - import what's available
try:
    from .patterns import *  # noqa: F403
except ImportError as e:
    import warnings
    warnings.warn(f"Optimization patterns import failed: {e}", stacklevel=2)
    pass

# Next-generation optimizations - import what's available
try:
    from .next_gen import *  # noqa: F403
except ImportError as e:
    import warnings
    warnings.warn(f"Next-gen optimizations import failed: {e}", stacklevel=2)
    pass

__all__ = []

# Dynamically build __all__ from available imports
import sys

current_module = sys.modules[__name__]
for attr_name in dir(current_module):
    if not attr_name.startswith('_') and attr_name not in ['warnings', 'sys']:
        try:
            attr = getattr(current_module, attr_name)
            if hasattr(attr, '__module__') and ('kernel_pytorch.optimizations' in str(getattr(attr, '__module__', '')) or
                                                'kernel_pytorch.optimization_patterns' in str(getattr(attr, '__module__', '')) or
                                                'kernel_pytorch.next_gen_optimizations' in str(getattr(attr, '__module__', ''))):
                __all__.append(attr_name)
        except Exception:
            pass

# Backward compatibility
import sys  # noqa: E402
import warnings  # noqa: E402


def _deprecation_warning(old_path: str):
    warnings.warn(
        f"Importing from '{old_path}' is deprecated. "
        f"Please use 'from kernel_pytorch.optimizations import ...' instead.",
        DeprecationWarning,
        stacklevel=3
    )

class _LegacyOptimizationImportHelper:
    """Helper for backward compatibility with old import paths"""
    def __getattr__(self, name):
        _deprecation_warning('kernel_pytorch.optimization_patterns')
        # Import the actual module to avoid recursion
        try:
            from . import optimization_patterns
            return getattr(optimization_patterns, name)
        except (ImportError, AttributeError):
            raise AttributeError(f"module 'kernel_pytorch.optimization_patterns' has no attribute '{name}'") from None

class _LegacyNextGenImportHelper:
    """Helper for backward compatibility with old import paths"""
    def __getattr__(self, name):
        _deprecation_warning('kernel_pytorch.next_gen_optimizations')
        # Import the actual module to avoid recursion
        try:
            from . import next_gen
            return getattr(next_gen, name)
        except (ImportError, AttributeError):
            raise AttributeError(f"module 'kernel_pytorch.next_gen_optimizations' has no attribute '{name}'") from None

# Legacy import support
sys.modules['kernel_pytorch.optimization_patterns'] = _LegacyOptimizationImportHelper()
sys.modules['kernel_pytorch.next_gen_optimizations'] = _LegacyNextGenImportHelper()
