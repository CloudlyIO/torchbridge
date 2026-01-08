"""
XLA Compatibility Layer for torch_xla 2.9.0+

This module provides backward-compatible wrappers for deprecated
torch_xla APIs, supporting both old (2.x) and new (2.9+) versions.

Version: 0.3.7
"""

import warnings
from typing import Any, Optional

import torch


def get_xla_device() -> torch.device:
    """
    Get XLA device (compatible with torch_xla 2.9+).

    Returns:
        XLA device or CPU fallback
    """
    try:
        # Try new API first (torch_xla 2.9+)
        import torch_xla
        if hasattr(torch_xla, 'device'):
            return torch_xla.device()

        # Fall back to old API
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        return torch.device('cpu')


def get_world_size() -> int:
    """
    Get world size for distributed training (compatible with torch_xla 2.9+).

    Returns:
        World size (1 if not distributed)
    """
    try:
        # Try new runtime API first (torch_xla 2.9+)
        import torch_xla
        if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'world_size'):
            return torch_xla.runtime.world_size()

        # Try older runtime API
        try:
            from torch_xla import runtime as xr
            if hasattr(xr, 'world_size'):
                return xr.world_size()
        except ImportError:
            pass

        # Fall back to old xm API
        import torch_xla.core.xla_model as xm
        if hasattr(xm, 'xrt_world_size'):
            return xm.xrt_world_size()

        # Default to 1 if nothing works
        return 1
    except ImportError:
        return 1


def get_ordinal() -> int:
    """
    Get process ordinal/rank (compatible with torch_xla 2.9+).

    Returns:
        Process rank (0 if not distributed)
    """
    try:
        # Try new runtime API first (torch_xla 2.9+)
        import torch_xla
        if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'global_ordinal'):
            return torch_xla.runtime.global_ordinal()

        # Try older runtime API
        try:
            from torch_xla import runtime as xr
            if hasattr(xr, 'global_ordinal'):
                return xr.global_ordinal()
        except ImportError:
            pass

        # Fall back to old xm API
        import torch_xla.core.xla_model as xm
        if hasattr(xm, 'get_ordinal'):
            return xm.get_ordinal()

        return 0
    except ImportError:
        return 0


def sync() -> None:
    """
    Synchronize XLA operations (compatible with torch_xla 2.9+).

    Replaces xm.mark_step() with torch_xla.sync().
    """
    try:
        # Try new API first (torch_xla 2.9+)
        import torch_xla
        if hasattr(torch_xla, 'sync'):
            torch_xla.sync()
            return

        # Fall back to old API
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except ImportError:
        pass


def get_device_count() -> int:
    """
    Get number of XLA devices (compatible with torch_xla 2.9+).

    Returns:
        Number of XLA devices available
    """
    try:
        # Try new runtime API first
        import torch_xla
        if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'device_count'):
            return torch_xla.runtime.device_count()

        # Try older runtime API
        try:
            from torch_xla import runtime as xr
            if hasattr(xr, 'device_count'):
                return xr.device_count()
        except ImportError:
            pass

        # Fall back to old xm API
        import torch_xla.core.xla_model as xm
        if hasattr(xm, 'xla_device_count'):
            return xm.xla_device_count()

        return 1
    except ImportError:
        return 0


def rendezvous(tag: str) -> None:
    """
    Distributed rendezvous point (compatible with torch_xla 2.9+).

    Args:
        tag: Rendezvous tag for synchronization
    """
    try:
        import torch_xla.core.xla_model as xm
        if hasattr(xm, 'rendezvous'):
            xm.rendezvous(tag)
    except ImportError:
        pass


def is_xla_available() -> bool:
    """
    Check if XLA/TPU is available.

    Returns:
        True if XLA is available and working
    """
    try:
        import torch_xla
        device = get_xla_device()
        return device.type == 'xla'
    except ImportError:
        return False


def is_tpu_device() -> bool:
    """
    Check if the current XLA device is a TPU (compatible with torch_xla 2.9+).

    Returns:
        True if running on TPU, False otherwise
    """
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        # Get device using compatibility layer
        device = get_xla_device()
        if device.type != 'xla':
            return False

        # Check device hardware type
        if hasattr(xm, 'xla_device_hw'):
            # Suppress deprecation warning for xla_device() by using device directly
            try:
                # In torch_xla 2.9+, we can check the device type differently
                hw_type = xm.xla_device_hw(device)
                return hw_type == 'TPU'
            except Exception:
                pass

        # Fallback: check environment variable
        import os
        pjrt_device = os.environ.get('PJRT_DEVICE', '')
        if pjrt_device.upper() == 'TPU':
            return True

        # Check if libtpu is available (indicates TPU environment)
        try:
            from torch_xla._internal import tpu
            return True
        except ImportError:
            pass

        return False
    except ImportError:
        return False


def get_device_hw_type() -> str:
    """
    Get XLA device hardware type (compatible with torch_xla 2.9+).

    Returns:
        Hardware type string ('TPU', 'GPU', 'CPU', or 'UNKNOWN')
    """
    try:
        import torch_xla.core.xla_model as xm

        device = get_xla_device()
        if device.type != 'xla':
            return 'CPU'

        if hasattr(xm, 'xla_device_hw'):
            try:
                return xm.xla_device_hw(device)
            except Exception:
                pass

        # Fallback: check environment
        import os
        pjrt_device = os.environ.get('PJRT_DEVICE', '')
        if pjrt_device:
            return pjrt_device.upper()

        return 'UNKNOWN'
    except ImportError:
        return 'CPU'


# Version info
def get_torch_xla_version() -> str:
    """Get torch_xla version string."""
    try:
        import torch_xla
        return getattr(torch_xla, '__version__', 'unknown')
    except ImportError:
        return 'not installed'
