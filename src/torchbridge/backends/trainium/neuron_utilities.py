"""
Neuron Utilities for Trainium Backend

Provides detection, version checks, and XLA compatibility wrappers
for AWS Trainium/Inferentia2 via the Neuron SDK.

Trainium uses XLA under the hood (same as TPU), differentiated by
environment variables and torch_neuronx imports.
"""

import os

import torch


def get_xla_device() -> torch.device:
    """
    Get XLA device for Trainium (compatible with Neuron SDK 2.x).

    Returns:
        XLA device or CPU fallback
    """
    try:
        import torch_xla
        if hasattr(torch_xla, 'device'):
            return torch_xla.device()

        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except ImportError:
        return torch.device('cpu')


def get_world_size() -> int:
    """
    Get world size for distributed training on Trainium.

    Returns:
        World size (1 if not distributed)
    """
    try:
        import torch_xla
        if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'world_size'):
            return torch_xla.runtime.world_size()

        try:
            from torch_xla import runtime as xr
            if hasattr(xr, 'world_size'):
                return xr.world_size()
        except ImportError:
            pass

        import torch_xla.core.xla_model as xm
        if hasattr(xm, 'xrt_world_size'):
            return xm.xrt_world_size()

        return 1
    except ImportError:
        return 1


def get_ordinal() -> int:
    """
    Get process ordinal/rank on Trainium.

    Returns:
        Process rank (0 if not distributed)
    """
    try:
        import torch_xla
        if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'global_ordinal'):
            return torch_xla.runtime.global_ordinal()

        try:
            from torch_xla import runtime as xr
            if hasattr(xr, 'global_ordinal'):
                return xr.global_ordinal()
        except ImportError:
            pass

        import torch_xla.core.xla_model as xm
        if hasattr(xm, 'get_ordinal'):
            return xm.get_ordinal()

        return 0
    except ImportError:
        return 0


def sync() -> None:
    """
    Synchronize XLA operations on Trainium.

    Replaces xm.mark_step() with torch_xla.sync() for Neuron SDK 2.27+.
    """
    try:
        import torch_xla
        if hasattr(torch_xla, 'sync'):
            torch_xla.sync()
            return

        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except ImportError:
        pass


def get_device_count() -> int:
    """
    Get number of NeuronCores available.

    Returns:
        Number of XLA devices (NeuronCores) available
    """
    try:
        import torch_xla
        if hasattr(torch_xla, 'runtime') and hasattr(torch_xla.runtime, 'device_count'):
            return torch_xla.runtime.device_count()

        try:
            from torch_xla import runtime as xr
            if hasattr(xr, 'device_count'):
                return xr.device_count()
        except ImportError:
            pass

        import torch_xla.core.xla_model as xm
        if hasattr(xm, 'xla_device_count'):
            return xm.xla_device_count()

        return 1
    except ImportError:
        return 0


def rendezvous(tag: str) -> None:
    """
    Distributed rendezvous point on Trainium.

    Args:
        tag: Rendezvous tag for synchronization
    """
    try:
        import torch_xla.core.xla_model as xm
        if hasattr(xm, 'rendezvous'):
            xm.rendezvous(tag)
    except ImportError:
        pass


def is_neuron_available() -> bool:
    """
    Check if Neuron SDK (torch_neuronx) is available.

    Returns:
        True if Neuron SDK is installed and a Trainium device is detected
    """
    try:
        import torch_neuronx  # noqa: F401
        pjrt = os.environ.get('PJRT_DEVICE', '').upper()
        if pjrt == 'NEURON':
            return True
        if os.environ.get('NEURON_RT_VISIBLE_CORES'):
            return True
        return False
    except ImportError:
        return False


def get_neuron_sdk_version() -> str:
    """
    Get the Neuron SDK version string.

    Returns:
        Version string or 'not installed'
    """
    try:
        import torch_neuronx
        return getattr(torch_neuronx, '__version__', 'unknown')
    except ImportError:
        return 'not installed'


def get_torch_xla_version() -> str:
    """Get torch_xla version string."""
    try:
        import torch_xla
        return getattr(torch_xla, '__version__', 'unknown')
    except ImportError:
        return 'not installed'


def detect_instance_type() -> str:
    """
    Detect the AWS instance type from environment.

    Returns:
        Instance type string (e.g., 'trn1.2xlarge') or 'unknown'
    """
    # Check for explicit env var
    instance_type = os.environ.get('NEURON_INSTANCE_TYPE', '')
    if instance_type:
        return instance_type

    # Try EC2 metadata (only works on actual EC2 instances)
    try:
        import urllib.request
        req = urllib.request.Request(
            'http://169.254.169.254/latest/meta-data/instance-type',
            headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'}
        )
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.read().decode('utf-8')
    except Exception:
        pass

    return 'unknown'


def get_neuron_env_info() -> dict[str, str]:
    """Get Neuron-related environment information."""
    return {
        'NEURON_RT_VISIBLE_CORES': os.environ.get('NEURON_RT_VISIBLE_CORES', ''),
        'NEURON_CC_FLAGS': os.environ.get('NEURON_CC_FLAGS', ''),
        'NEURON_INSTANCE_TYPE': os.environ.get('NEURON_INSTANCE_TYPE', ''),
        'PJRT_DEVICE': os.environ.get('PJRT_DEVICE', ''),
        'neuron_sdk_version': get_neuron_sdk_version(),
        'torch_xla_version': get_torch_xla_version(),
        'neuron_available': str(is_neuron_available()),
    }
