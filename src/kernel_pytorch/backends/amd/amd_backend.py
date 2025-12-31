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

Version: 0.3.4
"""

import logging
import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from kernel_pytorch.core.config import AMDConfig, AMDArchitecture
from .amd_exceptions import (
    AMDBackendError,
    ROCmNotAvailableError,
    AMDDeviceError,
    AMDConfigurationError,
)

logger = logging.getLogger(__name__)


@dataclass
class AMDDeviceInfo:
    """Information about an AMD GPU device."""

    device_id: int
    name: str
    architecture: AMDArchitecture
    compute_capability: str
    total_memory_gb: float
    matrix_cores_available: bool
    rocm_version: str


class AMDBackend:
    """
    Main backend orchestrator for AMD ROCm/HIP operations.

    This class manages AMD GPU devices, coordinates optimization strategies,
    and provides a unified interface for model preparation and execution.

    Example:
        >>> config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        >>> backend = AMDBackend(config)
        >>> model = backend.prepare_model(model)
    """

    def __init__(self, config: Optional[AMDConfig] = None):
        """
        Initialize AMD backend with configuration.

        Args:
            config: AMD configuration settings. If None, uses defaults.

        Raises:
            ROCmNotAvailableError: If ROCm runtime is not available
            AMDConfigurationError: If configuration is invalid
        """
        self.config = config or AMDConfig()
        self._devices: List[AMDDeviceInfo] = []
        self._current_device: Optional[AMDDeviceInfo] = None
        self._initialized = False

        logger.info("Initializing AMD ROCm Backend v0.3.4")

        # Validate configuration
        self._validate_config()

        # Check ROCm availability
        if not self._check_rocm_available():
            raise ROCmNotAvailableError(
                "ROCm runtime not available. Please install ROCm and set ROCM_HOME."
            )

        # Initialize devices
        self._initialize_devices()

        # Set default device
        if self._devices:
            self._current_device = self._devices[self.config.device_id]
            logger.info(
                "AMD backend initialized: device=%s, architecture=%s",
                self._current_device.name,
                self._current_device.architecture.value,
            )
        else:
            logger.warning("No AMD GPUs detected")

        self._initialized = True

    def _validate_config(self) -> None:
        """Validate AMD configuration settings."""
        # Validate device ID
        if self.config.device_id < 0:
            raise AMDConfigurationError(
                "device_id", self.config.device_id, "Device ID must be non-negative"
            )

        # Validate memory settings
        if self.config.memory_pool_size_gb <= 0:
            raise AMDConfigurationError(
                "memory_pool_size_gb",
                self.config.memory_pool_size_gb,
                "Memory pool size must be positive",
            )

        # Validate precision settings
        valid_precisions = ["fp32", "fp16", "bf16", "fp8"]
        if self.config.default_precision not in valid_precisions:
            raise AMDConfigurationError(
                "default_precision",
                self.config.default_precision,
                f"Must be one of {valid_precisions}",
            )

        logger.debug("Configuration validated successfully")

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

    def _initialize_devices(self) -> None:
        """
        Initialize and catalog all AMD GPU devices.

        Raises:
            AMDDeviceError: If device initialization fails
        """
        try:
            device_count = torch.cuda.device_count()
            logger.info("Detected %d AMD GPU device(s)", device_count)

            for device_id in range(device_count):
                device_info = self._get_device_info(device_id)
                self._devices.append(device_info)

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
                self.config.device_id,
                "initialization",
                f"Failed to initialize devices: {e}",
            )

    def _get_device_info(self, device_id: int) -> AMDDeviceInfo:
        """
        Get detailed information about an AMD GPU device.

        Args:
            device_id: Device ID to query

        Returns:
            AMDDeviceInfo object with device details
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

        return AMDDeviceInfo(
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
        self, model: torch.nn.Module, optimization_level: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Prepare a PyTorch model for AMD GPU execution.

        This method:
        1. Moves the model to AMD GPU
        2. Applies ROCm-specific optimizations
        3. Compiles HIP kernels if needed
        4. Sets up memory management

        Args:
            model: PyTorch model to prepare
            optimization_level: Optimization level (conservative/balanced/aggressive)
                              If None, uses config setting

        Returns:
            Prepared model ready for AMD GPU execution

        Raises:
            AMDBackendError: If model preparation fails
        """
        if not self._initialized:
            raise AMDBackendError("Backend not initialized")

        if not self._current_device:
            raise AMDBackendError("No AMD GPU device available")

        try:
            logger.info("Preparing model for AMD GPU: %s", self._current_device.name)

            # Move model to GPU
            device = torch.device(f"cuda:{self.config.device_id}")
            model = model.to(device)

            # Set precision if specified
            if self.config.default_precision == "fp16":
                model = model.half()
            elif self.config.default_precision == "bf16":
                model = model.bfloat16()

            # TODO: Apply optimizations via AMDOptimizer
            # TODO: Compile HIP kernels via ROCmCompiler
            # TODO: Set up memory management via AMDMemoryManager

            logger.info("Model prepared successfully for AMD GPU")
            return model

        except Exception as e:
            raise AMDBackendError(f"Model preparation failed: {e}")

    def get_device_info(self) -> Optional[AMDDeviceInfo]:
        """
        Get information about the current AMD GPU device.

        Returns:
            AMDDeviceInfo for current device, or None if no device
        """
        return self._current_device

    def get_all_devices(self) -> List[AMDDeviceInfo]:
        """
        Get information about all available AMD GPU devices.

        Returns:
            List of AMDDeviceInfo objects
        """
        return self._devices.copy()

    def is_available(self) -> bool:
        """
        Check if AMD backend is available and initialized.

        Returns:
            True if backend is ready, False otherwise
        """
        return self._initialized and self._current_device is not None

    def cleanup(self) -> None:
        """Clean up AMD backend resources."""
        if self._initialized:
            logger.info("Cleaning up AMD backend resources")
            # TODO: Clean up memory pools
            # TODO: Clear compilation caches
            self._initialized = False
            logger.info("AMD backend cleanup complete")

    def __repr__(self) -> str:
        """String representation of AMD backend."""
        if self._current_device:
            return (
                f"AMDBackend(device={self._current_device.name}, "
                f"architecture={self._current_device.architecture.value}, "
                f"memory={self._current_device.total_memory_gb:.2f}GB)"
            )
        return "AMDBackend(no device)"


__all__ = ["AMDBackend", "AMDDeviceInfo"]
