"""
Intel XPU Utilities

Provides device management, synchronization, and utility functions
for Intel XPU devices using Intel Extension for PyTorch (IPEX).
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import torch

from .intel_exceptions import (
    XPUNotAvailableError,
    IPEXNotInstalledError,
    XPUDeviceError,
)

logger = logging.getLogger(__name__)

# Try to import Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False
    ipex = None

# Check for XPU availability
def _check_xpu_available() -> bool:
    """Check if XPU is available."""
    if not IPEX_AVAILABLE:
        return False
    try:
        return torch.xpu.is_available()
    except AttributeError:
        return False

XPU_AVAILABLE = _check_xpu_available()


@dataclass
class XPUDeviceInfo:
    """Information about an Intel XPU device."""
    device_id: int
    name: str
    total_memory: int  # bytes
    driver_version: str
    compute_capability: Tuple[int, int]
    supports_amx: bool
    supports_fp16: bool
    supports_bf16: bool
    max_compute_units: int
    device_type: str  # "data_center", "consumer", "integrated"


class XPUDeviceManager:
    """
    Manages Intel XPU devices and provides coordination for multi-GPU setups.
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize XPU device manager.

        Args:
            device_id: Default device ID to use
        """
        self.device_id = device_id
        self._device_count = 0
        self._current_device = None
        self._device_infos: Dict[int, XPUDeviceInfo] = {}

        self._initialize()

    def _initialize(self):
        """Initialize device detection and configuration."""
        if not XPU_AVAILABLE:
            logger.warning("Intel XPU not available - device manager running in fallback mode")
            return

        try:
            self._device_count = torch.xpu.device_count()
            if self._device_count > 0:
                self._current_device = torch.device("xpu", self.device_id)
                logger.info(f"Detected {self._device_count} Intel XPU device(s)")

                # Cache device info for all devices
                for i in range(self._device_count):
                    self._device_infos[i] = self._get_device_info(i)
        except Exception as e:
            logger.warning(f"Failed to initialize XPU devices: {e}")
            self._device_count = 0

    def _get_device_info(self, device_id: int) -> XPUDeviceInfo:
        """Get detailed information about a specific XPU device."""
        if not XPU_AVAILABLE:
            raise XPUNotAvailableError()

        try:
            props = torch.xpu.get_device_properties(device_id)

            # Determine device type based on name
            name = props.name.upper()
            if any(x in name for x in ["MAX", "PONTE VECCHIO", "PVC"]):
                device_type = "data_center"
            elif any(x in name for x in ["ARC", "DG2", "A770", "A750"]):
                device_type = "consumer"
            else:
                device_type = "integrated"

            return XPUDeviceInfo(
                device_id=device_id,
                name=props.name,
                total_memory=props.total_memory,
                driver_version=getattr(props, 'driver_version', 'unknown'),
                compute_capability=(
                    getattr(props, 'major', 1),
                    getattr(props, 'minor', 0)
                ),
                supports_amx=self._check_amx_support(props),
                supports_fp16=True,  # All Intel XPUs support FP16
                supports_bf16=self._check_bf16_support(props),
                max_compute_units=getattr(props, 'max_compute_units', 0),
                device_type=device_type,
            )
        except Exception as e:
            logger.warning(f"Failed to get device info for XPU:{device_id}: {e}")
            return XPUDeviceInfo(
                device_id=device_id,
                name="Unknown Intel XPU",
                total_memory=0,
                driver_version="unknown",
                compute_capability=(1, 0),
                supports_amx=False,
                supports_fp16=True,
                supports_bf16=False,
                max_compute_units=0,
                device_type="unknown",
            )

    def _check_amx_support(self, props) -> bool:
        """Check if device supports Intel AMX (Advanced Matrix Extensions)."""
        # PVC (Ponte Vecchio / Data Center Max) supports AMX-like matrix operations
        name = props.name.upper()
        return any(x in name for x in ["MAX", "PONTE VECCHIO", "PVC"])

    def _check_bf16_support(self, props) -> bool:
        """Check if device supports BF16."""
        # Most modern Intel XPUs support BF16
        name = props.name.upper()
        return any(x in name for x in ["MAX", "PVC", "ARC", "DG2", "A770", "A750", "A580"])

    @property
    def device_count(self) -> int:
        """Get number of available XPU devices."""
        return self._device_count

    @property
    def current_device(self) -> Optional[torch.device]:
        """Get current XPU device."""
        return self._current_device

    def get_device(self, device_id: int = None) -> torch.device:
        """
        Get a torch device for the specified XPU.

        Args:
            device_id: Device ID (default: use initialized device)

        Returns:
            torch.device for the XPU
        """
        if device_id is None:
            device_id = self.device_id

        if not XPU_AVAILABLE:
            logger.warning("XPU not available, falling back to CPU")
            return torch.device("cpu")

        if device_id >= self._device_count:
            raise XPUDeviceError(
                f"Device {device_id} not available. Only {self._device_count} devices found.",
                device_id=device_id
            )

        return torch.device("xpu", device_id)

    def get_device_info(self, device_id: int = None) -> XPUDeviceInfo:
        """Get information about a specific device."""
        if device_id is None:
            device_id = self.device_id

        if device_id not in self._device_infos:
            if XPU_AVAILABLE:
                self._device_infos[device_id] = self._get_device_info(device_id)
            else:
                raise XPUNotAvailableError()

        return self._device_infos[device_id]

    def set_device(self, device_id: int):
        """Set the current XPU device."""
        if not XPU_AVAILABLE:
            raise XPUNotAvailableError()

        if device_id >= self._device_count:
            raise XPUDeviceError(
                f"Cannot set device {device_id}. Only {self._device_count} devices available.",
                device_id=device_id
            )

        torch.xpu.set_device(device_id)
        self.device_id = device_id
        self._current_device = torch.device("xpu", device_id)

    def synchronize(self, device_id: int = None):
        """Synchronize the specified or current XPU device."""
        if not XPU_AVAILABLE:
            return

        if device_id is not None:
            torch.xpu.synchronize(device_id)
        else:
            torch.xpu.synchronize()

    def empty_cache(self):
        """Empty the XPU memory cache."""
        if not XPU_AVAILABLE:
            return

        torch.xpu.empty_cache()


class XPUOptimizations:
    """
    Intel XPU-specific optimizations and configurations.
    """

    def __init__(self, device_manager: XPUDeviceManager = None):
        """
        Initialize XPU optimizations.

        Args:
            device_manager: XPU device manager instance
        """
        self.device_manager = device_manager or XPUDeviceManager()

    def optimize_model_for_inference(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype = torch.float32,
        sample_input: Optional[torch.Tensor] = None
    ) -> torch.nn.Module:
        """
        Optimize model for inference using IPEX.

        Args:
            model: Model to optimize
            dtype: Data type to use (torch.float32, torch.float16, torch.bfloat16)
            sample_input: Optional sample input for tracing

        Returns:
            Optimized model
        """
        if not IPEX_AVAILABLE:
            logger.warning("IPEX not available, returning unoptimized model")
            return model

        try:
            # Move model to XPU if available
            if XPU_AVAILABLE:
                model = model.to("xpu")
                if sample_input is not None:
                    sample_input = sample_input.to("xpu")

            # Apply IPEX optimization
            model = ipex.optimize(
                model,
                dtype=dtype,
                level="O1",  # Standard optimization level
                auto_kernel_selection=True,
            )

            logger.info(f"Model optimized with IPEX for inference (dtype={dtype})")
            return model

        except Exception as e:
            logger.warning(f"IPEX optimization failed: {e}. Returning original model.")
            return model

    def optimize_model_for_training(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dtype: torch.dtype = torch.float32
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """
        Optimize model and optimizer for training using IPEX.

        Args:
            model: Model to optimize
            optimizer: Optimizer to optimize
            dtype: Data type to use

        Returns:
            Tuple of (optimized_model, optimized_optimizer)
        """
        if not IPEX_AVAILABLE:
            logger.warning("IPEX not available, returning unoptimized model/optimizer")
            return model, optimizer

        try:
            # Move model to XPU if available
            if XPU_AVAILABLE:
                model = model.to("xpu")

            # Apply IPEX optimization for training
            model, optimizer = ipex.optimize(
                model,
                optimizer=optimizer,
                dtype=dtype,
                level="O1",
            )

            logger.info(f"Model and optimizer optimized with IPEX for training")
            return model, optimizer

        except Exception as e:
            logger.warning(f"IPEX training optimization failed: {e}")
            return model, optimizer

    def enable_onednn_fusion(self, enabled: bool = True):
        """Enable or disable oneDNN operator fusion."""
        if not IPEX_AVAILABLE:
            return

        try:
            if hasattr(ipex, 'enable_onednn_fusion'):
                ipex.enable_onednn_fusion(enabled)
                logger.debug(f"oneDNN fusion {'enabled' if enabled else 'disabled'}")
        except Exception as e:
            logger.warning(f"Failed to configure oneDNN fusion: {e}")

    def get_optimal_dtype(self) -> torch.dtype:
        """Get the optimal dtype for the current XPU device."""
        if not XPU_AVAILABLE:
            return torch.float32

        device_info = self.device_manager.get_device_info()

        # Data center GPUs (PVC/Max) work best with BF16
        if device_info.device_type == "data_center" and device_info.supports_bf16:
            return torch.bfloat16
        # Consumer GPUs work well with FP16
        elif device_info.supports_fp16:
            return torch.float16
        else:
            return torch.float32


def get_xpu_device_count() -> int:
    """Get the number of available XPU devices."""
    if not XPU_AVAILABLE:
        return 0
    try:
        return torch.xpu.device_count()
    except Exception:
        return 0


def is_xpu_available() -> bool:
    """Check if Intel XPU is available."""
    return XPU_AVAILABLE


def is_ipex_available() -> bool:
    """Check if Intel Extension for PyTorch is available."""
    return IPEX_AVAILABLE


def get_ipex_version() -> Optional[str]:
    """Get the IPEX version if available."""
    if not IPEX_AVAILABLE:
        return None
    return getattr(ipex, '__version__', 'unknown')


def xpu_synchronize(device_id: int = None):
    """Synchronize XPU device(s)."""
    if not XPU_AVAILABLE:
        return
    if device_id is not None:
        torch.xpu.synchronize(device_id)
    else:
        torch.xpu.synchronize()


def xpu_empty_cache():
    """Empty XPU memory cache."""
    if not XPU_AVAILABLE:
        return
    torch.xpu.empty_cache()


__all__ = [
    'XPUDeviceManager',
    'XPUDeviceInfo',
    'XPUOptimizations',
    'get_xpu_device_count',
    'is_xpu_available',
    'is_ipex_available',
    'get_ipex_version',
    'xpu_synchronize',
    'xpu_empty_cache',
    'IPEX_AVAILABLE',
    'XPU_AVAILABLE',
]
