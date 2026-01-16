"""
Health Monitoring for KernelPyTorch

This module provides health monitoring capabilities for KernelPyTorch
deployments, including component health checks and system status.

Features:
- Component health tracking
- Kubernetes-compatible health probes
- Custom health checks
- Automatic recovery suggestions

Example:
    ```python
    from kernel_pytorch.monitoring import HealthMonitor, create_health_monitor

    # Create monitor
    monitor = create_health_monitor(model=model)

    # Check health
    status = monitor.check_health()
    print(f"Status: {status.overall_status}")

    # Get liveness/readiness for K8s
    if monitor.is_ready():
        print("Service is ready")
    ```

Version: 0.3.10
"""

import torch
import torch.nn as nn
import time
import threading
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    last_check: Optional[datetime] = None
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass
class HealthCheck:
    """Health check result."""

    overall_status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0

    def is_healthy(self) -> bool:
        """Check if overall status is healthy."""
        return self.overall_status == HealthStatus.HEALTHY

    def is_ready(self) -> bool:
        """Check if service is ready (healthy or degraded)."""
        return self.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": [c.to_dict() for c in self.components],
        }


class HealthMonitor:
    """
    Health monitor for KernelPyTorch deployments.

    Tracks health of various components including:
    - Model (loaded, functional)
    - GPU (available, memory)
    - Inference (latency, errors)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_name: str = "model",
    ):
        """
        Initialize health monitor.

        Args:
            model: PyTorch model to monitor
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
        self._start_time = time.time()
        self._custom_checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._last_inference_time: Optional[float] = None
        self._inference_errors: int = 0
        self._lock = threading.Lock()

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register a custom health check.

        Args:
            name: Name of the check
            check_fn: Function that returns ComponentHealth
        """
        self._custom_checks[name] = check_fn

    def _check_model(self) -> ComponentHealth:
        """Check model health."""
        start = time.time()

        if self.model is None:
            return ComponentHealth(
                name="model",
                status=HealthStatus.UNHEALTHY,
                message="Model not loaded",
                last_check=datetime.now(),
            )

        try:
            # Check model is in eval mode
            is_eval = not self.model.training

            # Try a simple forward pass with dummy input
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype

            # Get first layer input size
            first_param = next(self.model.parameters())
            if len(first_param.shape) >= 2:
                input_size = first_param.shape[1]
            else:
                input_size = first_param.shape[0]

            dummy_input = torch.randn(1, input_size, device=device, dtype=dtype)

            with torch.no_grad():
                _ = self.model(dummy_input)

            latency = (time.time() - start) * 1000

            return ComponentHealth(
                name="model",
                status=HealthStatus.HEALTHY,
                message="Model loaded and functional",
                last_check=datetime.now(),
                latency_ms=latency,
                details={
                    "model_name": self.model_name,
                    "device": str(device),
                    "dtype": str(dtype),
                    "eval_mode": is_eval,
                },
            )

        except Exception as e:
            return ComponentHealth(
                name="model",
                status=HealthStatus.UNHEALTHY,
                message=f"Model check failed: {str(e)}",
                last_check=datetime.now(),
                latency_ms=(time.time() - start) * 1000,
            )

    def _check_gpu(self) -> ComponentHealth:
        """Check GPU health."""
        start = time.time()

        if not torch.cuda.is_available():
            return ComponentHealth(
                name="gpu",
                status=HealthStatus.HEALTHY,
                message="GPU not available (CPU mode)",
                last_check=datetime.now(),
                details={"mode": "cpu"},
            )

        try:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)

            memory_allocated = torch.cuda.memory_allocated(current_device)
            memory_total = torch.cuda.get_device_properties(current_device).total_memory
            memory_percent = (memory_allocated / memory_total) * 100

            # Determine status based on memory usage
            if memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "GPU memory critically low"
            elif memory_percent > 85:
                status = HealthStatus.DEGRADED
                message = "GPU memory usage high"
            else:
                status = HealthStatus.HEALTHY
                message = "GPU healthy"

            return ComponentHealth(
                name="gpu",
                status=status,
                message=message,
                last_check=datetime.now(),
                latency_ms=(time.time() - start) * 1000,
                details={
                    "device_count": device_count,
                    "current_device": current_device,
                    "device_name": device_name,
                    "memory_allocated_mb": memory_allocated / (1024 * 1024),
                    "memory_total_mb": memory_total / (1024 * 1024),
                    "memory_percent": memory_percent,
                },
            )

        except Exception as e:
            return ComponentHealth(
                name="gpu",
                status=HealthStatus.UNHEALTHY,
                message=f"GPU check failed: {str(e)}",
                last_check=datetime.now(),
            )

    def _check_inference(self) -> ComponentHealth:
        """Check inference health based on recent activity."""
        with self._lock:
            if self._last_inference_time is None:
                return ComponentHealth(
                    name="inference",
                    status=HealthStatus.HEALTHY,
                    message="No inference activity yet",
                    last_check=datetime.now(),
                )

            time_since_inference = time.time() - self._last_inference_time
            error_rate = self._inference_errors

            if error_rate > 10:
                status = HealthStatus.UNHEALTHY
                message = f"High error rate: {error_rate} errors"
            elif error_rate > 0:
                status = HealthStatus.DEGRADED
                message = f"Some errors detected: {error_rate} errors"
            else:
                status = HealthStatus.HEALTHY
                message = "Inference healthy"

            return ComponentHealth(
                name="inference",
                status=status,
                message=message,
                last_check=datetime.now(),
                details={
                    "time_since_last_inference_seconds": time_since_inference,
                    "error_count": error_rate,
                },
            )

    def record_inference(self, success: bool = True) -> None:
        """Record an inference event."""
        with self._lock:
            self._last_inference_time = time.time()
            if not success:
                self._inference_errors += 1

    def reset_errors(self) -> None:
        """Reset error counters."""
        with self._lock:
            self._inference_errors = 0

    def check_health(self) -> HealthCheck:
        """
        Perform full health check.

        Returns:
            HealthCheck with overall status and component details
        """
        components = []

        # Built-in checks
        components.append(self._check_model())
        components.append(self._check_gpu())
        components.append(self._check_inference())

        # Custom checks
        for name, check_fn in self._custom_checks.items():
            try:
                result = check_fn()
                components.append(result)
            except Exception as e:
                components.append(ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.now(),
                ))

        # Determine overall status
        statuses = [c.status for c in components]

        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        elif HealthStatus.UNKNOWN in statuses:
            overall = HealthStatus.UNKNOWN
        else:
            overall = HealthStatus.HEALTHY

        return HealthCheck(
            overall_status=overall,
            components=components,
            timestamp=datetime.now(),
            uptime_seconds=time.time() - self._start_time,
        )

    def is_live(self) -> bool:
        """Kubernetes liveness probe."""
        # Basic liveness - just check we can run
        return True

    def is_ready(self) -> bool:
        """Kubernetes readiness probe."""
        health = self.check_health()
        return health.is_ready()

    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self._start_time


def create_health_monitor(
    model: Optional[nn.Module] = None,
    model_name: str = "model",
) -> HealthMonitor:
    """
    Create a health monitor.

    Args:
        model: Model to monitor
        model_name: Name of the model

    Returns:
        Configured HealthMonitor
    """
    return HealthMonitor(model=model, model_name=model_name)
