"""
HIP Utilities for AMD GPU Operations

This module provides utility functions for HIP (Heterogeneous-compute Interface
for Portability) operations, device management, and profiling on AMD GPUs.

HIP is AMD's programming interface that provides a portable way to write
GPU code that can run on both AMD and NVIDIA hardware.

Key Features:
- Device coordination and synchronization
- Stream management for concurrent operations
- Event-based timing and profiling
- Memory transfer utilities
- Multi-GPU coordination

Version: 0.3.6
"""

import logging
import torch
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import time

from kernel_pytorch.core.config import AMDConfig, AMDArchitecture
from .amd_exceptions import AMDDeviceError, HIPKernelError

logger = logging.getLogger(__name__)


@dataclass
class HIPEvent:
    """Represents a HIP event for timing and synchronization."""

    name: str
    device_id: int
    _event: Optional[torch.cuda.Event] = None

    def __post_init__(self):
        """Initialize the CUDA/HIP event."""
        if torch.cuda.is_available():
            self._event = torch.cuda.Event(enable_timing=True)

    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Record the event on a stream."""
        if self._event is not None:
            self._event.record(stream)

    def synchronize(self) -> None:
        """Wait for the event to complete."""
        if self._event is not None:
            self._event.synchronize()

    def elapsed_time(self, end_event: "HIPEvent") -> float:
        """
        Get elapsed time between this event and another.

        Args:
            end_event: The end event

        Returns:
            Elapsed time in milliseconds
        """
        if self._event is not None and end_event._event is not None:
            return self._event.elapsed_time(end_event._event)
        return 0.0


@dataclass
class HIPStream:
    """Represents a HIP stream for concurrent operations."""

    name: str
    device_id: int
    priority: int = 0
    _stream: Optional[torch.cuda.Stream] = None

    def __post_init__(self):
        """Initialize the CUDA/HIP stream."""
        if torch.cuda.is_available():
            with torch.cuda.device(self.device_id):
                self._stream = torch.cuda.Stream(
                    device=self.device_id,
                    priority=self.priority,
                )

    def synchronize(self) -> None:
        """Wait for all operations on this stream to complete."""
        if self._stream is not None:
            self._stream.synchronize()

    @property
    def stream(self) -> Optional[torch.cuda.Stream]:
        """Get the underlying CUDA/HIP stream."""
        return self._stream


class HIPUtilities:
    """
    Utility class for HIP operations on AMD GPUs.

    Provides device management, stream coordination, profiling,
    and memory transfer utilities.

    Example:
        >>> config = AMDConfig()
        >>> utils = HIPUtilities(config)
        >>> with utils.profile_region("forward_pass"):
        ...     output = model(input)
    """

    def __init__(self, config: AMDConfig):
        """
        Initialize HIP utilities.

        Args:
            config: AMD configuration
        """
        self.config = config
        self._streams: Dict[str, HIPStream] = {}
        self._events: Dict[str, HIPEvent] = {}
        self._profiling_enabled = config.enable_profiling
        self._profiling_data: List[Dict[str, Any]] = []

        logger.info("HIPUtilities initialized for device %d", config.device_id)

    def create_stream(
        self, name: str, priority: int = 0, device_id: Optional[int] = None
    ) -> HIPStream:
        """
        Create a named HIP stream.

        Args:
            name: Stream name for identification
            priority: Stream priority (-1 = high, 0 = default)
            device_id: Device ID (defaults to config device)

        Returns:
            Created HIPStream
        """
        device = device_id if device_id is not None else self.config.device_id

        stream = HIPStream(name=name, device_id=device, priority=priority)
        self._streams[name] = stream

        logger.debug("Created stream '%s' on device %d", name, device)
        return stream

    def get_stream(self, name: str) -> Optional[HIPStream]:
        """
        Get a named stream.

        Args:
            name: Stream name

        Returns:
            HIPStream or None if not found
        """
        return self._streams.get(name)

    def create_event(
        self, name: str, device_id: Optional[int] = None
    ) -> HIPEvent:
        """
        Create a named HIP event.

        Args:
            name: Event name for identification
            device_id: Device ID (defaults to config device)

        Returns:
            Created HIPEvent
        """
        device = device_id if device_id is not None else self.config.device_id

        event = HIPEvent(name=name, device_id=device)
        self._events[name] = event

        logger.debug("Created event '%s' on device %d", name, device)
        return event

    def synchronize_device(self, device_id: Optional[int] = None) -> None:
        """
        Synchronize all operations on a device.

        Args:
            device_id: Device to synchronize (defaults to config device)
        """
        device = device_id if device_id is not None else self.config.device_id

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            logger.debug("Device %d synchronized", device)

    def synchronize_all_devices(self) -> None:
        """Synchronize all available AMD GPU devices."""
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                torch.cuda.synchronize(device_id)
            logger.debug("All %d devices synchronized", torch.cuda.device_count())

    @contextmanager
    def stream_context(self, stream_name: str):
        """
        Context manager for executing operations on a specific stream.

        Args:
            stream_name: Name of the stream to use

        Yields:
            The stream object
        """
        stream = self._streams.get(stream_name)
        if stream is None:
            stream = self.create_stream(stream_name)

        if stream._stream is not None:
            with torch.cuda.stream(stream._stream):
                yield stream
        else:
            yield stream

    @contextmanager
    def profile_region(self, name: str):
        """
        Context manager for profiling a code region.

        Args:
            name: Name of the region for identification

        Yields:
            None

        Example:
            >>> with utils.profile_region("attention"):
            ...     output = attention(query, key, value)
        """
        if not self._profiling_enabled:
            yield
            return

        start_event = self.create_event(f"{name}_start")
        end_event = self.create_event(f"{name}_end")

        start_event.record()
        start_time = time.perf_counter()

        try:
            yield
        finally:
            end_event.record()
            end_event.synchronize()

            end_time = time.perf_counter()
            gpu_time = start_event.elapsed_time(end_event)
            cpu_time = (end_time - start_time) * 1000  # Convert to ms

            profile_entry = {
                "name": name,
                "gpu_time_ms": gpu_time,
                "cpu_time_ms": cpu_time,
                "timestamp": time.time(),
            }

            self._profiling_data.append(profile_entry)
            logger.debug(
                "Profile '%s': GPU=%.3fms, CPU=%.3fms",
                name,
                gpu_time,
                cpu_time,
            )

    def get_profiling_data(self) -> List[Dict[str, Any]]:
        """
        Get collected profiling data.

        Returns:
            List of profiling entries
        """
        return self._profiling_data.copy()

    def clear_profiling_data(self) -> None:
        """Clear collected profiling data."""
        self._profiling_data.clear()
        logger.debug("Profiling data cleared")

    def get_profiling_summary(self) -> Dict[str, Any]:
        """
        Get summary of profiling data.

        Returns:
            Dictionary with profiling statistics
        """
        if not self._profiling_data:
            return {"total_regions": 0}

        # Group by region name
        regions: Dict[str, List[float]] = {}
        for entry in self._profiling_data:
            name = entry["name"]
            if name not in regions:
                regions[name] = []
            regions[name].append(entry["gpu_time_ms"])

        summary = {
            "total_regions": len(self._profiling_data),
            "unique_regions": len(regions),
            "regions": {},
        }

        for name, times in regions.items():
            summary["regions"][name] = {
                "count": len(times),
                "total_ms": sum(times),
                "avg_ms": sum(times) / len(times),
                "min_ms": min(times),
                "max_ms": max(times),
            }

        return summary

    def memory_copy_async(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        stream: Optional[HIPStream] = None,
    ) -> None:
        """
        Asynchronous memory copy between tensors.

        Args:
            dst: Destination tensor
            src: Source tensor
            stream: Stream to use (optional)
        """
        cuda_stream = stream._stream if stream else None

        with torch.cuda.stream(cuda_stream) if cuda_stream else nullcontext():
            dst.copy_(src, non_blocking=True)

    def get_device_properties(
        self, device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get properties of an AMD GPU device.

        Args:
            device_id: Device ID (defaults to config device)

        Returns:
            Dictionary with device properties
        """
        device = device_id if device_id is not None else self.config.device_id

        if not torch.cuda.is_available():
            return {"available": False}

        props = torch.cuda.get_device_properties(device)

        return {
            "available": True,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024**3),
            "multi_processor_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}",
            "max_threads_per_block": props.max_threads_per_block,
            "max_threads_per_multi_processor": props.max_threads_per_multi_processor,
            "warp_size": props.warp_size,
        }

    def get_memory_info(
        self, device_id: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Get memory information for a device.

        Args:
            device_id: Device ID (defaults to config device)

        Returns:
            Tuple of (free_memory, total_memory) in bytes
        """
        device = device_id if device_id is not None else self.config.device_id

        if torch.cuda.is_available():
            return torch.cuda.mem_get_info(device)
        return (0, 0)

    def set_device(self, device_id: int) -> None:
        """
        Set the current CUDA/HIP device.

        Args:
            device_id: Device ID to set as current
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            logger.debug("Current device set to %d", device_id)

    @contextmanager
    def device_context(self, device_id: int):
        """
        Context manager for temporarily switching devices.

        Args:
            device_id: Device ID to use

        Yields:
            None
        """
        if torch.cuda.is_available():
            with torch.cuda.device(device_id):
                yield
        else:
            yield

    def cleanup(self) -> None:
        """Clean up all streams and events."""
        # Synchronize all streams
        for stream in self._streams.values():
            stream.synchronize()

        # Clear collections
        self._streams.clear()
        self._events.clear()
        self._profiling_data.clear()

        logger.info("HIPUtilities cleanup complete")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HIPUtilities(device={self.config.device_id}, "
            f"streams={len(self._streams)}, "
            f"events={len(self._events)}, "
            f"profiling={self._profiling_enabled})"
        )


# Null context manager for when CUDA is not available
class nullcontext:
    """Null context manager for fallback."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        pass


__all__ = ["HIPUtilities", "HIPEvent", "HIPStream"]
