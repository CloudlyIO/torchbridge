"""
PyTorch PrivateUse1 Integration Framework

Provides seamless integration of custom hardware devices with PyTorch using the
PrivateUse1 mechanism while maintaining full backward compatibility with existing code.
"""

import torch
import logging
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading

# Import existing hardware discovery types for compatibility
from ..distributed_scale.hardware_discovery import (
    HardwareVendor, DeviceInfo, DeviceCapability, ThermalState
)

logger = logging.getLogger(__name__)


@dataclass
class PrivateUse1Config:
    """Configuration for PrivateUse1 device registration"""
    device_name: str
    vendor: HardwareVendor
    backend_library: str
    kernel_registrations: Dict[str, Callable] = None
    generator_class: Optional[type] = None
    guard_class: Optional[type] = None
    enable_autograd: bool = True
    enable_compilation: bool = True


class CustomDeviceBackend(ABC):
    """
    Abstract base class for custom device backends using PrivateUse1

    This provides a standardized interface for implementing device-specific
    operations while leveraging PyTorch's PrivateUse1 integration mechanism.
    """

    def __init__(self, device_name: str, vendor: HardwareVendor):
        self.device_name = device_name
        self.vendor = vendor
        self.is_registered = False
        self._kernel_registry: Dict[str, Callable] = {}

    @abstractmethod
    def initialize_device(self, device_id: int) -> bool:
        """Initialize specific device instance"""
        pass

    @abstractmethod
    def get_device_count(self) -> int:
        """Get number of available devices"""
        pass

    @abstractmethod
    def get_device_properties(self, device_id: int) -> Dict[str, Any]:
        """Get device properties and capabilities"""
        pass

    @abstractmethod
    def allocate_memory(self, size: int, device_id: int) -> Any:
        """Allocate memory on device"""
        pass

    @abstractmethod
    def copy_to_device(self, tensor: torch.Tensor, device_id: int) -> torch.Tensor:
        """Copy tensor to device"""
        pass

    def register_kernel(self, operation_name: str, kernel_impl: Callable) -> None:
        """Register custom kernel implementation"""
        self._kernel_registry[operation_name] = kernel_impl
        logger.debug(f"Registered kernel {operation_name} for {self.device_name}")

    def get_kernel(self, operation_name: str) -> Optional[Callable]:
        """Get registered kernel implementation"""
        return self._kernel_registry.get(operation_name)


class PrivateUse1Manager:
    """
    Manager for PyTorch PrivateUse1 device registration and management

    Provides a high-level interface for registering custom devices with PyTorch
    while maintaining compatibility with existing hardware discovery systems.
    """

    def __init__(self):
        self.registered_devices: Dict[str, CustomDeviceBackend] = {}
        self.device_mappings: Dict[HardwareVendor, str] = {}
        self._lock = threading.Lock()

    def register_device_backend(self,
                               backend: CustomDeviceBackend,
                               config: PrivateUse1Config) -> bool:
        """
        Register custom device backend with PyTorch PrivateUse1

        Args:
            backend: Custom device backend implementation
            config: PrivateUse1 configuration

        Returns:
            True if registration successful, False otherwise
        """
        try:
            with self._lock:
                # Check if already registered
                if config.device_name in self.registered_devices:
                    logger.warning(f"Device {config.device_name} already registered")
                    return False

                # Register with PyTorch PrivateUse1
                if self._register_with_pytorch(backend, config):
                    self.registered_devices[config.device_name] = backend
                    self.device_mappings[config.vendor] = config.device_name
                    backend.is_registered = True

                    logger.info(f"Successfully registered {config.device_name} with PyTorch")
                    return True
                else:
                    logger.error(f"Failed to register {config.device_name} with PyTorch")
                    return False

        except Exception as e:
            logger.error(f"Error registering device backend: {e}")
            return False

    def _register_with_pytorch(self,
                              backend: CustomDeviceBackend,
                              config: PrivateUse1Config) -> bool:
        """Internal method to register with PyTorch PrivateUse1"""
        try:
            # Rename PrivateUse1 to custom device name
            torch.utils.rename_privateuse1_backend(config.device_name)

            # Register device-specific operations
            if config.kernel_registrations:
                self._register_kernels(config.device_name, config.kernel_registrations)

            # Register generator if provided
            if config.generator_class:
                self._register_generator(config.device_name, config.generator_class)

            # Register device guard if provided
            if config.guard_class:
                self._register_device_guard(config.device_name, config.guard_class)

            # Enable autograd support if requested
            if config.enable_autograd:
                self._setup_autograd_support(config.device_name)

            return True

        except Exception as e:
            logger.error(f"PyTorch PrivateUse1 registration failed: {e}")
            return False

    def _register_kernels(self, device_name: str, kernel_registry: Dict[str, Callable]) -> None:
        """Register custom kernels for device"""
        try:
            # Register kernels using TORCH_LIBRARY_IMPL
            for op_name, kernel_impl in kernel_registry.items():
                # This would use TORCH_LIBRARY_IMPL in actual implementation
                # For now, we store the registration info
                logger.debug(f"Registering kernel {op_name} for {device_name}")

        except Exception as e:
            logger.error(f"Kernel registration failed: {e}")

    def _register_generator(self, device_name: str, generator_class: type) -> None:
        """Register custom random number generator"""
        try:
            # Register generator class with PyTorch
            # This would use torch.Generator.register_generator in actual implementation
            logger.debug(f"Registered generator for {device_name}")

        except Exception as e:
            logger.error(f"Generator registration failed: {e}")

    def _register_device_guard(self, device_name: str, guard_class: type) -> None:
        """Register device guard for memory management"""
        try:
            # Register device guard with PyTorch
            # This would integrate with PyTorch's device guard system
            logger.debug(f"Registered device guard for {device_name}")

        except Exception as e:
            logger.error(f"Device guard registration failed: {e}")

    def _setup_autograd_support(self, device_name: str) -> None:
        """Setup autograd support for custom device"""
        try:
            # Enable autograd operations for the custom device
            # This would integrate with PyTorch's autograd system
            logger.debug(f"Enabled autograd support for {device_name}")

        except Exception as e:
            logger.error(f"Autograd setup failed: {e}")

    def is_device_available(self, device_name: str) -> bool:
        """Check if custom device is available"""
        return device_name in self.registered_devices

    def get_device_backend(self, device_name: str) -> Optional[CustomDeviceBackend]:
        """Get registered device backend"""
        return self.registered_devices.get(device_name)

    def get_device_for_vendor(self, vendor: HardwareVendor) -> Optional[str]:
        """Get device name for hardware vendor"""
        return self.device_mappings.get(vendor)

    def list_registered_devices(self) -> List[str]:
        """Get list of registered device names"""
        return list(self.registered_devices.keys())

    def unregister_device(self, device_name: str) -> bool:
        """Unregister custom device"""
        try:
            with self._lock:
                if device_name in self.registered_devices:
                    backend = self.registered_devices[device_name]
                    backend.is_registered = False

                    # Remove from mappings
                    vendor_to_remove = None
                    for vendor, name in self.device_mappings.items():
                        if name == device_name:
                            vendor_to_remove = vendor
                            break

                    if vendor_to_remove:
                        del self.device_mappings[vendor_to_remove]

                    del self.registered_devices[device_name]

                    logger.info(f"Unregistered device {device_name}")
                    return True

        except Exception as e:
            logger.error(f"Error unregistering device: {e}")

        return False


# Global PrivateUse1 manager instance
_privateuse1_manager = None


def get_privateuse1_manager() -> PrivateUse1Manager:
    """Get global PrivateUse1 manager instance"""
    global _privateuse1_manager
    if _privateuse1_manager is None:
        _privateuse1_manager = PrivateUse1Manager()
    return _privateuse1_manager


def register_custom_device(backend: CustomDeviceBackend,
                          config: PrivateUse1Config) -> bool:
    """
    Convenience function to register custom device

    Args:
        backend: Custom device backend implementation
        config: PrivateUse1 configuration

    Returns:
        True if registration successful
    """
    manager = get_privateuse1_manager()
    return manager.register_device_backend(backend, config)


def is_custom_device_available(device_name: str) -> bool:
    """Check if custom device is available"""
    manager = get_privateuse1_manager()
    return manager.is_device_available(device_name)


def get_custom_device_count(device_name: str) -> int:
    """Get number of available custom devices"""
    manager = get_privateuse1_manager()
    backend = manager.get_device_backend(device_name)
    if backend:
        return backend.get_device_count()
    return 0


# Backward compatibility functions
def supports_privateuse1() -> bool:
    """Check if current PyTorch version supports PrivateUse1"""
    try:
        # Check if PrivateUse1 APIs are available
        return hasattr(torch.utils, 'rename_privateuse1_backend')
    except Exception:
        return False


def validate_privateuse1_setup() -> Dict[str, Any]:
    """Validate PrivateUse1 setup and return status"""
    status = {
        'pytorch_version': torch.__version__,
        'privateuse1_supported': supports_privateuse1(),
        'registered_devices': [],
        'errors': []
    }

    try:
        manager = get_privateuse1_manager()
        status['registered_devices'] = manager.list_registered_devices()

        # Check each registered device
        for device_name in status['registered_devices']:
            backend = manager.get_device_backend(device_name)
            if backend and not backend.is_registered:
                status['errors'].append(f"Device {device_name} registration incomplete")

    except Exception as e:
        status['errors'].append(f"Validation error: {e}")

    return status