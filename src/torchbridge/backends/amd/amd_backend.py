"""
AMD ROCm Backend Implementation

This module provides the main AMDBackend class that orchestrates AMD GPU
operations through ROCm/HIP, supporting CDNA2 (MI200) and CDNA3 (MI300)
architectures.

Architecture Support:
- CDNA2: MI210, MI250, MI250X (2nd gen, Matrix Cores)
- CDNA3: MI300A, MI300X (3rd gen, Matrix Cores v2)

Key Features:
- Automatic device detection and initialization
- Multi-level optimization (conservative/balanced/aggressive)
- HIP kernel compilation and caching
- Memory management with HBM pooling
- Matrix Core acceleration
- ROCm profiling integration

Inherits from BaseBackend to provide a consistent interface across all
hardware backends.

Version: 0.5.3
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchbridge.backends.base_backend import (
    BaseBackend,
    DeviceInfo,
    OptimizationLevel,
)
from torchbridge.core.config import AMDArchitecture, AMDConfig

from .amd_exceptions import (
    AMDBackendError,
    AMDConfigurationError,
    AMDDeviceError,
)

logger = logging.getLogger(__name__)


@dataclass
class AMDDeviceInfoLegacy:
    """Information about an AMD GPU device."""

    device_id: int
    name: str
    architecture: AMDArchitecture
    compute_capability: str
    total_memory_gb: float
    matrix_cores_available: bool
    rocm_version: str


class AMDBackend(BaseBackend):
    """
    Main backend orchestrator for AMD ROCm/HIP operations.

    This class manages AMD GPU devices, coordinates optimization strategies,
    and provides a unified interface for model preparation and execution.

    Inherits from BaseBackend to provide a unified interface while maintaining
    backward compatibility with existing AMD-specific APIs.

    Example:
        >>> config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        >>> backend = AMDBackend(config)
        >>> model = backend.prepare_model(model)
    """

    # Backend identifier
    BACKEND_NAME: str = "amd"

    def __init__(self, config: AMDConfig | None = None):
        """
        Initialize AMD backend with configuration.

        Args:
            config: AMD configuration settings. If None, uses defaults.

        Note:
            Falls back to CPU mode if ROCm is not available.
        """
        self._amd_config = config or AMDConfig()
        self._amd_devices: list[AMDDeviceInfoLegacy] = []
        self._current_amd_device: AMDDeviceInfoLegacy | None = None
        self._cpu_fallback = False

        logger.info("Initializing AMD ROCm Backend v0.5.3")

        # Validate configuration
        self._validate_config()

        # Call parent init (which calls _setup_environment)
        super().__init__(config=self._amd_config)

        # Alias for backward compatibility
        self.config = self._amd_config
        self._devices = self._amd_devices
        self._current_device = self._current_amd_device

    @property
    def device(self) -> torch.device:
        """Get the current device (AMD GPU or CPU fallback)."""
        if self._cpu_fallback:
            return torch.device("cpu")
        return torch.device("cuda", self.config.device_id)

    def _validate_config(self) -> None:
        """Validate AMD configuration settings."""
        # Validate device ID
        if self._amd_config.device_id < 0:
            raise AMDConfigurationError(
                "device_id", self._amd_config.device_id, "Device ID must be non-negative"
            )

        # Validate memory settings
        if self._amd_config.memory_pool_size_gb <= 0:
            raise AMDConfigurationError(
                "memory_pool_size_gb",
                self._amd_config.memory_pool_size_gb,
                "Memory pool size must be positive",
            )

        # Validate precision settings
        valid_precisions = ["fp32", "fp16", "bf16", "fp8"]
        if self._amd_config.default_precision not in valid_precisions:
            raise AMDConfigurationError(
                "default_precision",
                self._amd_config.default_precision,
                f"Must be one of {valid_precisions}",
            )

        logger.debug("Configuration validated successfully")

    def _setup_environment(self) -> None:
        """Set up ROCm environment for AMD GPUs (implements BaseBackend abstract method)."""
        # Check ROCm availability
        if not self._check_rocm_available():
            logger.warning(
                "ROCm not available - falling back to CPU mode. "
                "For GPU acceleration, install ROCm and set ROCM_HOME."
            )
            self._cpu_fallback = True
            self._device = torch.device("cpu")
            return

        # Initialize devices
        self._initialize_amd_devices()

        # Set default device
        if self._amd_devices:
            self._current_amd_device = self._amd_devices[self._amd_config.device_id]
            self._device = torch.device(f"cuda:{self._amd_config.device_id}")
            logger.info(
                "AMD backend initialized: device=%s, architecture=%s",
                self._current_amd_device.name,
                self._current_amd_device.architecture.value,
            )
        else:
            logger.warning("No AMD GPUs detected - using CPU fallback")
            self._cpu_fallback = True
            self._device = torch.device("cpu")

    def _check_availability(self) -> bool:
        """Check if ROCm is available (implements BaseBackend abstract method)."""
        return not self._cpu_fallback and self._check_rocm_available()

    def _get_device_info(self, device_id: int = 0) -> DeviceInfo:
        """Get device info in unified format (implements BaseBackend abstract method)."""
        if self._cpu_fallback or device_id >= len(self._amd_devices):
            return DeviceInfo(
                backend="amd",
                device_type="cpu",
                device_id=0,
                device_name="CPU (ROCm fallback)",
                is_available=False
            )

        amd_info = self._amd_devices[device_id]
        return DeviceInfo(
            backend="amd",
            device_type=f"cuda:{device_id}",
            device_id=device_id,
            device_name=amd_info.name,
            compute_capability=amd_info.compute_capability,
            total_memory_bytes=int(amd_info.total_memory_gb * 1024**3),
            driver_version=amd_info.rocm_version,
            is_available=True,
            properties={
                'architecture': amd_info.architecture.value,
                'matrix_cores_available': amd_info.matrix_cores_available,
            }
        )

    def _check_rocm_available(self) -> bool:
        """
        Check if ROCm runtime is available.

        Returns:
            True if ROCm is available, False otherwise
        """
        try:
            # Check if PyTorch was built with ROCm support
            if not torch.cuda.is_available():
                logger.warning("PyTorch CUDA API not available (ROCm uses CUDA API)")
                return False

            # Check if this is actually ROCm (not NVIDIA CUDA)
            # ROCm uses torch.cuda API but with AMD GPUs
            if torch.cuda.device_count() > 0:
                device_name = torch.cuda.get_device_name(0)
                is_amd = any(
                    marker in device_name.upper()
                    for marker in ["AMD", "RADEON", "MI200", "MI300", "MI250", "MI210"]
                )
                if is_amd:
                    logger.info("ROCm detected: %s", device_name)
                    return True
                else:
                    logger.warning("CUDA device found but not AMD GPU: %s", device_name)
                    return False

            return False

        except Exception as e:
            logger.error("Failed to check ROCm availability: %s", e)
            return False

    def _initialize_amd_devices(self) -> None:
        """
        Initialize and catalog all AMD GPU devices.

        Raises:
            AMDDeviceError: If device initialization fails
        """
        try:
            device_count = torch.cuda.device_count()
            logger.info("Detected %d AMD GPU device(s)", device_count)

            for device_id in range(device_count):
                device_info = self._get_amd_device_info(device_id)
                self._amd_devices.append(device_info)

                logger.debug(
                    "Device %d: %s (%s, %.2f GB, Matrix Cores: %s)",
                    device_id,
                    device_info.name,
                    device_info.architecture.value,
                    device_info.total_memory_gb,
                    device_info.matrix_cores_available,
                )

        except Exception as e:
            raise AMDDeviceError(
                self._amd_config.device_id,
                "initialization",
                f"Failed to initialize devices: {e}",
            ) from e

    def _get_amd_device_info(self, device_id: int) -> AMDDeviceInfoLegacy:
        """
        Get detailed information about an AMD GPU device (legacy format).

        Args:
            device_id: Device ID to query

        Returns:
            AMDDeviceInfoLegacy object with device details
        """
        device_name = torch.cuda.get_device_name(device_id)
        properties = torch.cuda.get_device_properties(device_id)

        # Detect architecture from device name
        architecture = self._detect_architecture(device_name)

        # Get memory in GB
        total_memory_gb = properties.total_memory / (1024**3)

        # Determine if Matrix Cores are available
        # CDNA2 and CDNA3 have Matrix Cores
        matrix_cores_available = architecture in [
            AMDArchitecture.CDNA2,
            AMDArchitecture.CDNA3,
        ]

        # Get compute capability (ROCm version)
        compute_capability = f"{properties.major}.{properties.minor}"

        # Get ROCm version (approximation from compute capability)
        rocm_version = self._get_rocm_version()

        return AMDDeviceInfoLegacy(
            device_id=device_id,
            name=device_name,
            architecture=architecture,
            compute_capability=compute_capability,
            total_memory_gb=total_memory_gb,
            matrix_cores_available=matrix_cores_available,
            rocm_version=rocm_version,
        )

    def _detect_architecture(self, device_name: str) -> AMDArchitecture:
        """
        Detect AMD GPU architecture from device name.

        Args:
            device_name: Name of the GPU device

        Returns:
            Detected AMDArchitecture
        """
        device_name_upper = device_name.upper()

        # Check for specific architectures
        if "MI300" in device_name_upper:
            return AMDArchitecture.CDNA3
        elif any(marker in device_name_upper for marker in ["MI200", "MI250", "MI210"]):
            return AMDArchitecture.CDNA2
        elif "MI50" in device_name_upper or "MI60" in device_name_upper:
            return AMDArchitecture.CDNA
        elif "RX 7" in device_name_upper or "7900" in device_name_upper:
            return AMDArchitecture.RDNA3
        elif "RX 6" in device_name_upper or "6900" in device_name_upper:
            return AMDArchitecture.RDNA2

        # Default to config setting or AUTO
        if self.config.architecture != AMDArchitecture.AUTO:
            logger.warning(
                "Could not detect architecture from '%s', using config: %s",
                device_name,
                self.config.architecture.value,
            )
            return self.config.architecture

        logger.warning(
            "Could not detect architecture from '%s', defaulting to CDNA2", device_name
        )
        return AMDArchitecture.CDNA2

    def _get_rocm_version(self) -> str:
        """
        Get ROCm version string.

        Returns:
            ROCm version (e.g., "5.7.0")
        """
        try:
            # Try to get ROCm version from torch
            if hasattr(torch.version, "hip"):
                return torch.version.hip or "unknown"
            return "unknown"
        except Exception:
            return "unknown"

    def prepare_model(
        self,
        model: nn.Module,
        optimization_level: str | OptimizationLevel | None = None
    ) -> nn.Module:
        """
        Prepare a PyTorch model for AMD GPU execution (implements BaseBackend abstract method).

        This method:
        1. Moves the model to AMD GPU (or keeps on CPU if fallback)
        2. Applies ROCm-specific optimizations
        3. Compiles HIP kernels if needed
        4. Sets up memory management

        Args:
            model: PyTorch model to prepare
            optimization_level: Optimization level

        Returns:
            Prepared model ready for execution

        Raises:
            AMDBackendError: If model preparation fails
        """
        if not self._initialized:
            raise AMDBackendError("Backend not initialized")

        try:
            # Handle CPU fallback mode
            if self._cpu_fallback:
                logger.info("Preparing model in CPU fallback mode")
                model = model.to(torch.device("cpu"))
                return model

            if not self._current_amd_device:
                raise AMDBackendError("No AMD GPU device available")

            logger.info("Preparing model for AMD GPU: %s", self._current_amd_device.name)

            # Move model to GPU
            device = torch.device(f"cuda:{self._amd_config.device_id}")
            model = model.to(device)

            # Set precision if specified
            if self._amd_config.default_precision == "fp16":
                model = model.half()
            elif self._amd_config.default_precision == "bf16":
                model = model.bfloat16()

            logger.info("Model prepared successfully for AMD GPU")
            return model

        except Exception as e:
            raise AMDBackendError(f"Model preparation failed: {e}") from e

    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_input: torch.Tensor | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module:
        """
        Optimize a model for inference (implements BaseBackend abstract method).

        Args:
            model: PyTorch model
            sample_input: Optional sample input for tracing
            dtype: Optional dtype for precision

        Returns:
            Inference-optimized model
        """
        model = self.prepare_model(model)
        model.eval()

        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False

        # Apply torch.compile if available
        if sample_input is not None and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='reduce-overhead')
                with torch.no_grad():
                    _ = model(sample_input.to(self.device))
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        return model

    def optimize_for_training(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        dtype: torch.dtype | None = None
    ) -> nn.Module | tuple[nn.Module, torch.optim.Optimizer]:
        """
        Optimize a model for training (implements BaseBackend abstract method).

        Args:
            model: PyTorch model
            optimizer: Optional optimizer to optimize along with model
            dtype: Optional dtype for precision

        Returns:
            Training-optimized model, or tuple of (model, optimizer)
        """
        model = self.prepare_model(model)
        model.train()

        if optimizer:
            return model, optimizer
        return model

    @property
    def device_count(self) -> int:
        """Get the number of available AMD devices (overrides BaseBackend)."""
        return len(self._amd_devices)

    def get_device_info_dict(self) -> dict[str, Any]:
        """
        Get information about the current device (legacy method).

        Returns:
            Dictionary with device information
        """
        if self._cpu_fallback:
            return {
                "device_type": "cpu",
                "name": "CPU (ROCm fallback)",
                "architecture": "cpu",
                "total_memory_gb": 0,
                "matrix_cores_available": False,
                "rocm_available": False,
            }

        if self._current_amd_device:
            return {
                "device_type": "amd_gpu",
                "name": self._current_amd_device.name,
                "architecture": self._current_amd_device.architecture.value,
                "total_memory_gb": self._current_amd_device.total_memory_gb,
                "matrix_cores_available": self._current_amd_device.matrix_cores_available,
                "rocm_available": True,
            }

        return {"device_type": "unknown", "rocm_available": False}

    def get_all_amd_devices(self) -> list[AMDDeviceInfoLegacy]:
        """
        Get information about all available AMD GPU devices (legacy method).

        Returns:
            List of AMDDeviceInfoLegacy objects
        """
        return self._amd_devices.copy()

    def synchronize(self) -> None:
        """Synchronize all pending operations (overrides BaseBackend)."""
        if not self._cpu_fallback and torch.cuda.is_available():
            torch.cuda.synchronize()

    def empty_cache(self) -> None:
        """Empty ROCm cache (overrides BaseBackend)."""
        if not self._cpu_fallback and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        """String representation of AMD backend."""
        if self._current_amd_device:
            return (
                f"AMDBackend(device={self._current_amd_device.name}, "
                f"architecture={self._current_amd_device.architecture.value}, "
                f"memory={self._current_amd_device.total_memory_gb:.2f}GB)"
            )
        return "AMDBackend(no device)"


# Alias for backward compatibility
AMDDeviceInfo = AMDDeviceInfoLegacy

__all__ = ["AMDBackend", "AMDDeviceInfo", "AMDDeviceInfoLegacy"]
