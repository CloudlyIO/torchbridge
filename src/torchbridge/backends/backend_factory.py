"""
Backend Factory for Automatic Hardware Selection

This module provides a factory pattern for automatically selecting and
initializing the appropriate backend based on available hardware.

The factory supports:
- Automatic hardware detection and backend selection
- Priority-based backend ordering
- Fallback to CPU when no accelerator is available
- Backend registration for extensibility

"""

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch

from .base_adapter import BaseAdapter, CPUAdapter
from .base_backend import BaseBackend, CPUBackend

logger = logging.getLogger(__name__)

class BackendType(Enum):
    """
    Supported backend types.
    """
    AUTO = "auto"
    NVIDIA = "nvidia"
    AMD = "amd"
    TPU = "tpu"
    TRAINIUM = "trainium"
    CPU = "cpu"

    @classmethod
    def from_string(cls, name: str) -> "BackendType":
        """Convert string to BackendType."""
        name_lower = name.lower()
        for member in cls:
            if member.value == name_lower:
                return member
        # Handle aliases
        aliases = {
            'cuda': cls.NVIDIA,
            'rocm': cls.AMD,
            'hip': cls.AMD,
            'xla': cls.TPU,
            'neuron': cls.TRAINIUM,
            'trn': cls.TRAINIUM,
        }
        return aliases.get(name_lower, cls.CPU)

class BackendFactory:
    """
    Factory for creating and managing hardware backends.

    This class provides automatic hardware detection and backend selection,
    with support for priority ordering and fallback strategies.

    Example usage:
        # Automatic selection
        backend = BackendFactory.create()

        # Specific backend
        backend = BackendFactory.create(BackendType.NVIDIA)

        # With configuration
        backend = BackendFactory.create(BackendType.NVIDIA, config=nvidia_config)
    """

    # Registered backends (populated by register_backend)
    _backends: dict[BackendType, type[BaseBackend]] = {}

    # Registered optimizers
    _optimizers: dict[BackendType, type[BaseAdapter]] = {}

    # Backend priority for auto-selection (higher = preferred)
    _priority: dict[BackendType, int] = {
        BackendType.NVIDIA: 100,  # Highest priority
        BackendType.AMD: 90,
        BackendType.TRAINIUM: 88,
        BackendType.TPU: 85,
        BackendType.CPU: 0,  # Fallback
    }

    # Availability check functions
    _availability_checks: dict[BackendType, Callable[[], bool]] = {}

    @classmethod
    def register_backend(
        cls,
        backend_type: BackendType,
        backend_class: type[BaseBackend],
        optimizer_class: type[BaseAdapter] | None = None,
        availability_check: Callable[[], bool] | None = None,
        priority: int | None = None
    ) -> None:
        """
        Register a backend with the factory.

        Args:
            backend_type: Type of backend
            backend_class: Backend class to instantiate
            optimizer_class: Optional optimizer class for this backend
            availability_check: Optional function to check availability
            priority: Optional priority for auto-selection
        """
        cls._backends[backend_type] = backend_class

        if optimizer_class:
            cls._optimizers[backend_type] = optimizer_class

        if availability_check:
            cls._availability_checks[backend_type] = availability_check

        if priority is not None:
            cls._priority[backend_type] = priority

        logger.debug(f"Registered backend: {backend_type.value}")

    @classmethod
    def create(
        cls,
        backend_type: BackendType | str = BackendType.AUTO,
        config: Any = None,
        **kwargs
    ) -> BaseBackend:
        """
        Create a backend instance.

        Args:
            backend_type: Type of backend to create (or AUTO for automatic selection)
            config: Backend-specific configuration
            **kwargs: Additional arguments passed to backend constructor

        Returns:
            Initialized backend instance
        """
        # Convert string to BackendType if needed
        if isinstance(backend_type, str):
            backend_type = BackendType.from_string(backend_type)

        # Auto-select based on availability
        if backend_type == BackendType.AUTO:
            backend_type = cls._auto_select()

        logger.info(f"Creating backend: {backend_type.value}")

        # Get backend class
        backend_class = cls._get_backend_class(backend_type)

        # Create instance
        try:
            backend = backend_class(config=config, **kwargs)
            return backend
        except Exception as e:
            logger.error(f"Failed to create {backend_type.value} backend: {e}")
            logger.info("Falling back to CPU backend")
            return CPUBackend(config=config)

    @classmethod
    def create_optimizer(
        cls,
        backend_type: BackendType | str = BackendType.AUTO,
        config: Any = None,
        device: torch.device | None = None,
        **kwargs
    ) -> BaseAdapter:
        """
        Create an optimizer for the specified backend.

        Args:
            backend_type: Type of backend
            config: Backend-specific configuration
            device: Target device
            **kwargs: Additional arguments

        Returns:
            Initialized optimizer instance
        """
        # Convert string to BackendType if needed
        if isinstance(backend_type, str):
            backend_type = BackendType.from_string(backend_type)

        # Auto-select based on availability
        if backend_type == BackendType.AUTO:
            backend_type = cls._auto_select()

        # Get optimizer class
        optimizer_class = cls._optimizers.get(backend_type, CPUAdapter)

        try:
            return optimizer_class(config=config, device=device, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create {backend_type.value} optimizer: {e}")
            return CPUAdapter(config=config, device=device)

    @classmethod
    def _auto_select(cls) -> BackendType:
        """
        Automatically select the best available backend.

        Returns:
            BackendType for the best available backend
        """
        available = cls.get_available_backends()

        if not available:
            logger.warning("No accelerator backends available, using CPU")
            return BackendType.CPU

        # Sort by priority and return the highest priority available backend
        sorted_backends = sorted(
            available,
            key=lambda b: cls._priority.get(b, 0),
            reverse=True
        )

        selected = sorted_backends[0]
        logger.info(f"Auto-selected backend: {selected.value}")
        return selected

    @classmethod
    def _get_backend_class(cls, backend_type: BackendType) -> type[BaseBackend]:
        """Get the backend class for the given type."""
        if backend_type in cls._backends:
            return cls._backends[backend_type]

        # Lazy-load backends to avoid import cycles
        if backend_type == BackendType.NVIDIA:
            try:
                from .nvidia import NVIDIABackend
                cls._backends[BackendType.NVIDIA] = NVIDIABackend
                return NVIDIABackend
            except ImportError:
                logger.warning("NVIDIA backend not available")

        elif backend_type == BackendType.AMD:
            try:
                from .amd import AMDBackend
                cls._backends[BackendType.AMD] = AMDBackend
                return AMDBackend
            except ImportError:
                logger.warning("AMD backend not available")

        elif backend_type == BackendType.TRAINIUM:
            try:
                from .trainium import TrainiumBackend
                cls._backends[BackendType.TRAINIUM] = TrainiumBackend
                return TrainiumBackend
            except ImportError:
                logger.warning("Trainium backend not available")

        elif backend_type == BackendType.TPU:
            try:
                from .tpu import TPUBackend
                cls._backends[BackendType.TPU] = TPUBackend
                return TPUBackend
            except ImportError:
                logger.warning("TPU backend not available")

        # Default to CPU
        return CPUBackend

    @classmethod
    def get_available_backends(cls) -> list[BackendType]:
        """
        Get list of available backends.

        Returns:
            List of available BackendTypes
        """
        available = []

        # Check NVIDIA/CUDA
        if cls._check_nvidia_available():
            available.append(BackendType.NVIDIA)

        # Check AMD/ROCm
        if cls._check_amd_available():
            available.append(BackendType.AMD)

        # Check Trainium (before TPU to avoid XLA misdetection)
        if cls._check_trainium_available():
            available.append(BackendType.TRAINIUM)

        # Check TPU/XLA
        if cls._check_tpu_available():
            available.append(BackendType.TPU)

        # CPU is always available
        available.append(BackendType.CPU)

        return available

    @classmethod
    def _check_nvidia_available(cls) -> bool:
        """Check if NVIDIA CUDA is available."""
        if BackendType.NVIDIA in cls._availability_checks:
            return cls._availability_checks[BackendType.NVIDIA]()

        try:
            return torch.cuda.is_available() and torch.version.cuda is not None
        except Exception:
            logger.debug("NVIDIA CUDA availability check failed", exc_info=True)
            return False

    @classmethod
    def _check_amd_available(cls) -> bool:
        """Check if AMD ROCm is available."""
        if BackendType.AMD in cls._availability_checks:
            return cls._availability_checks[BackendType.AMD]()

        try:
            # ROCm uses the CUDA API via HIP
            if torch.cuda.is_available():
                # Check if it's actually ROCm, not CUDA
                if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                    return True
                # Alternative check
                if 'rocm' in str(torch.__config__.show()).lower():
                    return True
            return False
        except Exception:
            logger.debug("AMD ROCm availability check failed", exc_info=True)
            return False

    @classmethod
    def _check_trainium_available(cls) -> bool:
        """Check if AWS Trainium/Inferentia2 is available."""
        if BackendType.TRAINIUM in cls._availability_checks:
            return cls._availability_checks[BackendType.TRAINIUM]()

        try:
            import os

            import torch_neuronx  # noqa: F401

            # Trainium uses XLA but is not a TPU
            pjrt = os.environ.get('PJRT_DEVICE', '').upper()
            if pjrt == 'NEURON':
                return True
            # Also check if neuron runtime cores are visible
            if os.environ.get('NEURON_RT_VISIBLE_CORES'):
                return True
            return False
        except ImportError:
            return False

    @classmethod
    def _check_tpu_available(cls) -> bool:
        """Check if TPU/XLA is available.

        Uses a 5-second timeout to avoid hanging on machines where
        torch_xla is installed but no TPU device is present.
        """
        if BackendType.TPU in cls._availability_checks:
            return cls._availability_checks[BackendType.TPU]()

        try:
            import torch_xla.core.xla_model as xm
        except ImportError:
            return False

        # Wrap xla_device() in a timeout: it can block 30+ seconds on
        # machines where XLA is installed but no TPU hardware is present.
        import signal as _signal

        if hasattr(_signal, "SIGALRM"):
            # POSIX systems (Linux, macOS)
            def _timeout_handler(signum, frame):  # noqa: ARG001
                raise TimeoutError("TPU device enumeration timed out")

            old_handler = _signal.signal(_signal.SIGALRM, _timeout_handler)
            _signal.alarm(5)
            try:
                xm.xla_device()
                _signal.alarm(0)
                _signal.signal(_signal.SIGALRM, old_handler)
                return True
            except TimeoutError:
                logger.debug("TPU device enumeration timed out after 5s — no TPU detected")
                _signal.signal(_signal.SIGALRM, old_handler)
                return False
            except Exception:
                _signal.alarm(0)
                _signal.signal(_signal.SIGALRM, old_handler)
                logger.debug("TPU/XLA availability check failed", exc_info=True)
                return False
        else:
            # Windows fallback: threading-based timeout
            import threading

            _result: list[bool] = [False]

            def _check() -> None:
                try:
                    xm.xla_device()
                    _result[0] = True
                except Exception:
                    pass

            t = threading.Thread(target=_check, daemon=True)
            t.start()
            t.join(timeout=5.0)
            if t.is_alive():
                logger.debug("TPU device enumeration timed out after 5s — no TPU detected")
            return _result[0]

    @classmethod
    def get_backend_info(cls, backend_type: BackendType) -> dict[str, Any]:
        """
        Get information about a specific backend.

        Args:
            backend_type: Type of backend

        Returns:
            Dictionary with backend information
        """
        info: dict[str, Any] = {
            'type': backend_type.value,
            'available': False,
            'priority': cls._priority.get(backend_type, 0),
            'registered': backend_type in cls._backends,
        }

        # Check availability
        if backend_type == BackendType.NVIDIA:
            info['available'] = cls._check_nvidia_available()
            if info['available']:
                info['cuda_version'] = torch.version.cuda
                info['device_count'] = torch.cuda.device_count()
        elif backend_type == BackendType.AMD:
            info['available'] = cls._check_amd_available()
            if info['available']:
                info['hip_version'] = getattr(torch.version, 'hip', None)
        elif backend_type == BackendType.TRAINIUM:
            info['available'] = cls._check_trainium_available()
        elif backend_type == BackendType.TPU:
            info['available'] = cls._check_tpu_available()
        elif backend_type == BackendType.CPU:
            info['available'] = True
            import platform
            info['platform'] = platform.processor()

        return info

    @classmethod
    def get_all_backend_info(cls) -> dict[str, dict[str, Any]]:
        """
        Get information about all backends.

        Returns:
            Dictionary mapping backend names to their info
        """
        return {
            bt.value: cls.get_backend_info(bt)
            for bt in BackendType
            if bt != BackendType.AUTO
        }

    @classmethod
    def print_status(cls) -> None:
        """Print status of all backends."""
        print("=" * 60)
        print(" TorchBridge Backend Status")
        print("=" * 60)

        available = cls.get_available_backends()

        for bt in BackendType:
            if bt == BackendType.AUTO:
                continue

            info = cls.get_backend_info(bt)
            status = "AVAILABLE" if info['available'] else "NOT AVAILABLE"
            priority = f"[Priority: {info['priority']}]" if info['available'] else ""

            print(f"\n{bt.value.upper():10} {status:15} {priority}")

            if info['available']:
                if 'cuda_version' in info:
                    print(f"           CUDA: {info['cuda_version']}, Devices: {info['device_count']}")
                if 'hip_version' in info:
                    print(f"           HIP: {info['hip_version']}")
                if bt == BackendType.TRAINIUM:
                    print("           Neuron SDK detected")

        print("\n" + "=" * 60)
        if available and available[0] != BackendType.CPU:
            print(f"Auto-selected: {cls._auto_select().value.upper()}")
        else:
            print("Auto-selected: CPU (no accelerators available)")
        print("=" * 60)

def get_backend(
    backend_type: BackendType | str = BackendType.AUTO,
    config: Any = None,
    **kwargs
) -> BaseBackend:
    """
    Convenience function to get a backend instance.

    Args:
        backend_type: Type of backend (or "auto" for automatic)
        config: Backend-specific configuration
        **kwargs: Additional arguments

    Returns:
        Backend instance
    """
    return BackendFactory.create(backend_type, config, **kwargs)

def get_optimizer(
    backend_type: BackendType | str = BackendType.AUTO,
    config: Any = None,
    device: torch.device | None = None,
    **kwargs
) -> BaseAdapter:
    """
    Convenience function to get an optimizer instance.

    Args:
        backend_type: Type of backend
        config: Configuration
        device: Target device
        **kwargs: Additional arguments

    Returns:
        Optimizer instance
    """
    return BackendFactory.create_optimizer(backend_type, config, device, **kwargs)

def detect_best_backend() -> BackendType:
    """
    Detect the best available backend.

    Returns:
        BackendType for the best available backend
    """
    return BackendFactory._auto_select()

def list_available_backends() -> list[str]:
    """
    List available backend names.

    Returns:
        List of available backend names
    """
    return [bt.value for bt in BackendFactory.get_available_backends()]

__all__ = [
    'BackendFactory',
    'BackendType',
    'get_backend',
    'get_optimizer',
    'detect_best_backend',
    'list_available_backends',
]
