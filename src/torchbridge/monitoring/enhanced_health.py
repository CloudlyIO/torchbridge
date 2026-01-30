"""
Enhanced Health Monitoring for TorchBridge

Extends the base health monitoring with:
- Actual liveness verification (not just return True)
- System resource checks (CPU, memory, disk)
- Network connectivity checks
- Predictive health monitoring with trend analysis
- Health history tracking
- Auto-recovery recommendations
- Integration with structured logging

Version: 0.4.32

Example:
    ```python
    from torchbridge.monitoring import EnhancedHealthMonitor

    # Create enhanced monitor
    monitor = EnhancedHealthMonitor(model=model)

    # Add resource thresholds
    monitor.set_thresholds(
        cpu_percent=80,
        memory_percent=85,
        disk_percent=90
    )

    # Full health check
    health = monitor.check_health()

    # Kubernetes probes with real checks
    if monitor.is_live():  # Actually verifies system health
        serve_traffic()
    ```
"""

from __future__ import annotations

import gc
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

from .health_monitor import (
    ComponentHealth,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
)
from .structured_logging import correlation_context, get_logger

logger = get_logger(__name__)


class ResourceType(Enum):
    """Types of system resources to monitor."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    GPU_MEMORY = "gpu_memory"
    NETWORK = "network"


class HealthTrend(Enum):
    """Health trend indicators."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    CRITICAL = "critical"


@dataclass
class ResourceThresholds:
    """Thresholds for resource monitoring."""
    cpu_warning_percent: float = 70.0
    cpu_critical_percent: float = 90.0
    memory_warning_percent: float = 75.0
    memory_critical_percent: float = 90.0
    disk_warning_percent: float = 80.0
    disk_critical_percent: float = 95.0
    gpu_memory_warning_percent: float = 85.0
    gpu_memory_critical_percent: float = 95.0
    response_time_warning_ms: float = 100.0
    response_time_critical_ms: float = 500.0


@dataclass
class HealthHistoryEntry:
    """Single entry in health history."""
    timestamp: datetime
    status: HealthStatus
    components: dict[str, HealthStatus]
    metrics: dict[str, float]


@dataclass
class PredictiveHealthReport:
    """Predictive health analysis report."""
    current_status: HealthStatus
    predicted_status: HealthStatus
    trend: HealthTrend
    time_to_degradation: timedelta | None
    recommendations: list[str]
    confidence: float
    analysis_window_minutes: int
    data_points: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_status": self.current_status.value,
            "predicted_status": self.predicted_status.value,
            "trend": self.trend.value,
            "time_to_degradation": str(self.time_to_degradation) if self.time_to_degradation else None,
            "recommendations": self.recommendations,
            "confidence": self.confidence,
            "analysis_window_minutes": self.analysis_window_minutes,
            "data_points": self.data_points,
        }


class SystemResourceMonitor:
    """Monitors system resources (CPU, memory, disk)."""

    def __init__(self):
        self._psutil_available = False
        try:
            import psutil
            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            logger.warning("psutil not available, system resource monitoring limited")

    def get_cpu_percent(self, interval: float = 0.1) -> float | None:
        """Get CPU usage percentage."""
        if not self._psutil_available:
            return None
        try:
            return self._psutil.cpu_percent(interval=interval)
        except Exception as e:
            logger.error(f"Failed to get CPU usage: {e}")
            return None

    def get_memory_info(self) -> dict[str, float] | None:
        """Get memory usage information."""
        if not self._psutil_available:
            return None
        try:
            mem = self._psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "percent": mem.percent,
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return None

    def get_disk_info(self, path: str = "/") -> dict[str, float] | None:
        """Get disk usage information."""
        if not self._psutil_available:
            return None
        try:
            disk = self._psutil.disk_usage(path)
            return {
                "total_gb": disk.total / (1024**3),
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent": disk.percent,
            }
        except Exception as e:
            logger.error(f"Failed to get disk info: {e}")
            return None

    def get_gpu_memory_info(self) -> list[dict[str, float]] | None:
        """Get GPU memory information for all devices."""
        if not torch.cuda.is_available():
            return None

        try:
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
                gpu_info.append({
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "allocated_gb": allocated / (1024**3),
                    "reserved_gb": reserved / (1024**3),
                    "total_gb": total / (1024**3),
                    "percent": (allocated / total) * 100 if total > 0 else 0,
                })
            return gpu_info
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return None

    def get_process_info(self) -> dict[str, Any] | None:
        """Get current process information."""
        if not self._psutil_available:
            return None
        try:
            process = self._psutil.Process()
            return {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "memory_mb": process.memory_info().rss / (1024**2),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
            }
        except Exception as e:
            logger.error(f"Failed to get process info: {e}")
            return None


class EnhancedHealthMonitor(HealthMonitor):
    """
    Enhanced health monitor with real liveness checks and predictive monitoring.

    Extends the base HealthMonitor with:
    - Actual liveness verification
    - System resource monitoring
    - Health history tracking
    - Predictive health analysis
    - Auto-recovery recommendations
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        model_name: str = "model",
        history_size: int = 1000,
        enable_predictive: bool = True,
    ):
        """
        Initialize enhanced health monitor.

        Args:
            model: PyTorch model to monitor
            model_name: Name of the model
            history_size: Number of health history entries to keep
            enable_predictive: Enable predictive health analysis
        """
        super().__init__(model=model, model_name=model_name)

        self._thresholds = ResourceThresholds()
        self._resource_monitor = SystemResourceMonitor()
        self._history: deque[HealthHistoryEntry] = deque(maxlen=history_size)
        self._enable_predictive = enable_predictive
        self._liveness_checks: list[Callable[[], bool]] = []
        self._last_liveness_check: float | None = None
        self._liveness_cache_seconds = 5.0
        self._cached_liveness: bool = True

        # Register default liveness checks
        self._register_default_liveness_checks()

    def _register_default_liveness_checks(self) -> None:
        """Register default liveness verification checks."""
        # Check 1: Process is responsive (can allocate memory)
        def check_memory_allocation() -> bool:
            try:
                _ = [0] * 1000  # Small allocation
                return True
            except MemoryError:
                return False

        # Check 2: PyTorch is functional
        def check_pytorch_functional() -> bool:
            try:
                x = torch.tensor([1.0, 2.0])
                y = x + x
                return y.sum().item() == 6.0
            except Exception:
                return False

        # Check 3: GC is working
        def check_gc_functional() -> bool:
            try:
                gc.collect()
                return True
            except Exception:
                return False

        self._liveness_checks.extend([
            check_memory_allocation,
            check_pytorch_functional,
            check_gc_functional,
        ])

    def set_thresholds(
        self,
        cpu_warning: float | None = None,
        cpu_critical: float | None = None,
        memory_warning: float | None = None,
        memory_critical: float | None = None,
        disk_warning: float | None = None,
        disk_critical: float | None = None,
        gpu_memory_warning: float | None = None,
        gpu_memory_critical: float | None = None,
        response_time_warning: float | None = None,
        response_time_critical: float | None = None,
    ) -> None:
        """Set resource monitoring thresholds."""
        if cpu_warning is not None:
            self._thresholds.cpu_warning_percent = cpu_warning
        if cpu_critical is not None:
            self._thresholds.cpu_critical_percent = cpu_critical
        if memory_warning is not None:
            self._thresholds.memory_warning_percent = memory_warning
        if memory_critical is not None:
            self._thresholds.memory_critical_percent = memory_critical
        if disk_warning is not None:
            self._thresholds.disk_warning_percent = disk_warning
        if disk_critical is not None:
            self._thresholds.disk_critical_percent = disk_critical
        if gpu_memory_warning is not None:
            self._thresholds.gpu_memory_warning_percent = gpu_memory_warning
        if gpu_memory_critical is not None:
            self._thresholds.gpu_memory_critical_percent = gpu_memory_critical
        if response_time_warning is not None:
            self._thresholds.response_time_warning_ms = response_time_warning
        if response_time_critical is not None:
            self._thresholds.response_time_critical_ms = response_time_critical

    def add_liveness_check(self, check: Callable[[], bool]) -> None:
        """Add a custom liveness check."""
        self._liveness_checks.append(check)

    def is_live(self) -> bool:
        """
        Kubernetes liveness probe with actual verification.

        Unlike the base implementation that always returns True,
        this actually verifies the system is functional.

        Returns:
            True if all liveness checks pass
        """
        current_time = time.time()

        # Use cached result if recent enough
        if (
            self._last_liveness_check is not None
            and current_time - self._last_liveness_check < self._liveness_cache_seconds
        ):
            return self._cached_liveness

        with self._lock:
            try:
                # Run all liveness checks
                for check in self._liveness_checks:
                    if not check():
                        logger.warning("Liveness check failed", check_name=check.__name__)
                        self._cached_liveness = False
                        self._last_liveness_check = current_time
                        return False

                self._cached_liveness = True
                self._last_liveness_check = current_time
                return True

            except Exception as e:
                logger.error(f"Liveness check error: {e}")
                self._cached_liveness = False
                self._last_liveness_check = current_time
                return False

    def _check_cpu(self) -> ComponentHealth:
        """Check CPU health."""
        start = time.time()
        cpu_percent = self._resource_monitor.get_cpu_percent()

        if cpu_percent is None:
            return ComponentHealth(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                message="CPU monitoring unavailable (install psutil)",
                last_check=datetime.now(),
            )

        if cpu_percent >= self._thresholds.cpu_critical_percent:
            status = HealthStatus.UNHEALTHY
            message = f"CPU critically high: {cpu_percent:.1f}%"
        elif cpu_percent >= self._thresholds.cpu_warning_percent:
            status = HealthStatus.DEGRADED
            message = f"CPU usage elevated: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU healthy: {cpu_percent:.1f}%"

        return ComponentHealth(
            name="cpu",
            status=status,
            message=message,
            last_check=datetime.now(),
            latency_ms=(time.time() - start) * 1000,
            details={"cpu_percent": cpu_percent},
        )

    def _check_memory(self) -> ComponentHealth:
        """Check system memory health."""
        start = time.time()
        mem_info = self._resource_monitor.get_memory_info()

        if mem_info is None:
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="Memory monitoring unavailable (install psutil)",
                last_check=datetime.now(),
            )

        percent = mem_info["percent"]

        if percent >= self._thresholds.memory_critical_percent:
            status = HealthStatus.UNHEALTHY
            message = f"Memory critically high: {percent:.1f}%"
        elif percent >= self._thresholds.memory_warning_percent:
            status = HealthStatus.DEGRADED
            message = f"Memory usage elevated: {percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory healthy: {percent:.1f}%"

        return ComponentHealth(
            name="memory",
            status=status,
            message=message,
            last_check=datetime.now(),
            latency_ms=(time.time() - start) * 1000,
            details=mem_info,
        )

    def _check_disk(self) -> ComponentHealth:
        """Check disk health."""
        start = time.time()
        disk_info = self._resource_monitor.get_disk_info()

        if disk_info is None:
            return ComponentHealth(
                name="disk",
                status=HealthStatus.UNKNOWN,
                message="Disk monitoring unavailable (install psutil)",
                last_check=datetime.now(),
            )

        percent = disk_info["percent"]

        if percent >= self._thresholds.disk_critical_percent:
            status = HealthStatus.UNHEALTHY
            message = f"Disk critically full: {percent:.1f}%"
        elif percent >= self._thresholds.disk_warning_percent:
            status = HealthStatus.DEGRADED
            message = f"Disk usage elevated: {percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk healthy: {percent:.1f}%"

        return ComponentHealth(
            name="disk",
            status=status,
            message=message,
            last_check=datetime.now(),
            latency_ms=(time.time() - start) * 1000,
            details=disk_info,
        )

    def _check_process(self) -> ComponentHealth:
        """Check current process health."""
        start = time.time()
        process_info = self._resource_monitor.get_process_info()

        if process_info is None:
            return ComponentHealth(
                name="process",
                status=HealthStatus.UNKNOWN,
                message="Process monitoring unavailable",
                last_check=datetime.now(),
            )

        return ComponentHealth(
            name="process",
            status=HealthStatus.HEALTHY,
            message="Process running",
            last_check=datetime.now(),
            latency_ms=(time.time() - start) * 1000,
            details=process_info,
        )

    def check_health(self) -> HealthCheck:
        """
        Perform comprehensive health check including system resources.

        Returns:
            HealthCheck with overall status and component details
        """
        with correlation_context():
            logger.debug("Starting health check")
            start_time = time.time()

            components = []

            # Base checks from parent
            components.append(self._check_model())
            components.append(self._check_gpu())
            components.append(self._check_inference())

            # Enhanced system resource checks
            components.append(self._check_cpu())
            components.append(self._check_memory())
            components.append(self._check_disk())
            components.append(self._check_process())

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

            health_check = HealthCheck(
                overall_status=overall,
                components=components,
                timestamp=datetime.now(),
                uptime_seconds=time.time() - self._start_time,
            )

            # Record in history
            self._record_health(health_check)

            duration_ms = (time.time() - start_time) * 1000
            logger.debug(
                "Health check completed",
                status=overall.value,
                duration_ms=round(duration_ms, 2),
                component_count=len(components),
            )

            return health_check

    def _record_health(self, health: HealthCheck) -> None:
        """Record health check in history."""
        metrics: dict[str, float] = {}

        for component in health.components:
            if component.details:
                for key, value in component.details.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{component.name}_{key}"] = float(value)

        entry = HealthHistoryEntry(
            timestamp=health.timestamp,
            status=health.overall_status,
            components={c.name: c.status for c in health.components},
            metrics=metrics,
        )

        self._history.append(entry)

    def get_health_history(
        self,
        minutes: int | None = None,
        limit: int | None = None,
    ) -> list[HealthHistoryEntry]:
        """
        Get health check history.

        Args:
            minutes: Only return entries from the last N minutes
            limit: Maximum number of entries to return

        Returns:
            List of health history entries
        """
        entries = list(self._history)

        if minutes is not None:
            cutoff = datetime.now() - timedelta(minutes=minutes)
            entries = [e for e in entries if e.timestamp >= cutoff]

        if limit is not None:
            entries = entries[-limit:]

        return entries

    def get_predictive_health(
        self,
        analysis_window_minutes: int = 30,
    ) -> PredictiveHealthReport:
        """
        Analyze health trends and predict future status.

        Args:
            analysis_window_minutes: Time window for trend analysis

        Returns:
            PredictiveHealthReport with trend analysis
        """
        if not self._enable_predictive:
            return PredictiveHealthReport(
                current_status=HealthStatus.UNKNOWN,
                predicted_status=HealthStatus.UNKNOWN,
                trend=HealthTrend.STABLE,
                time_to_degradation=None,
                recommendations=["Predictive monitoring disabled"],
                confidence=0.0,
                analysis_window_minutes=analysis_window_minutes,
                data_points=0,
            )

        history = self.get_health_history(minutes=analysis_window_minutes)

        if len(history) < 3:
            current = history[-1].status if history else HealthStatus.UNKNOWN
            return PredictiveHealthReport(
                current_status=current,
                predicted_status=current,
                trend=HealthTrend.STABLE,
                time_to_degradation=None,
                recommendations=["Insufficient data for prediction"],
                confidence=0.0,
                analysis_window_minutes=analysis_window_minutes,
                data_points=len(history),
            )

        # Analyze status transitions
        status_values = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 2,
            HealthStatus.UNKNOWN: 1,
        }

        values = [status_values[h.status] for h in history]
        current_status = history[-1].status

        # Calculate trend
        recent = values[-min(10, len(values)):]
        older = values[:-len(recent)] if len(values) > len(recent) else recent

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg + 0.3:
            trend = HealthTrend.DEGRADING
        elif recent_avg < older_avg - 0.3:
            trend = HealthTrend.IMPROVING
        elif recent_avg >= 1.5:
            trend = HealthTrend.CRITICAL
        else:
            trend = HealthTrend.STABLE

        # Predict future status
        if trend == HealthTrend.DEGRADING:
            predicted = HealthStatus.DEGRADED if current_status == HealthStatus.HEALTHY else HealthStatus.UNHEALTHY
        elif trend == HealthTrend.IMPROVING:
            predicted = HealthStatus.HEALTHY if current_status == HealthStatus.DEGRADED else current_status
        else:
            predicted = current_status

        # Estimate time to degradation
        time_to_degradation = None
        if trend == HealthTrend.DEGRADING and current_status == HealthStatus.HEALTHY:
            # Simple linear extrapolation
            degradation_rate = (recent_avg - older_avg) / analysis_window_minutes
            if degradation_rate > 0:
                minutes_to_degrade = (1 - recent_avg) / degradation_rate
                time_to_degradation = timedelta(minutes=max(0, minutes_to_degrade))

        # Generate recommendations
        recommendations = self._generate_recommendations(history, trend)

        # Calculate confidence
        confidence = min(1.0, len(history) / 100)

        return PredictiveHealthReport(
            current_status=current_status,
            predicted_status=predicted,
            trend=trend,
            time_to_degradation=time_to_degradation,
            recommendations=recommendations,
            confidence=confidence,
            analysis_window_minutes=analysis_window_minutes,
            data_points=len(history),
        )

    def _generate_recommendations(
        self,
        history: list[HealthHistoryEntry],
        trend: HealthTrend,
    ) -> list[str]:
        """Generate auto-recovery recommendations."""
        recommendations = []

        if not history:
            return recommendations

        # Analyze recent metrics
        recent = history[-min(10, len(history)):]
        recent[-1]

        # CPU recommendations
        cpu_metrics = [h.metrics.get("cpu_percent", 0) for h in recent]
        if cpu_metrics and max(cpu_metrics) > self._thresholds.cpu_warning_percent:
            recommendations.append(
                f"High CPU usage detected (max {max(cpu_metrics):.1f}%). "
                "Consider scaling horizontally or optimizing compute-intensive operations."
            )

        # Memory recommendations
        memory_metrics = [h.metrics.get("memory_percent", 0) for h in recent]
        if memory_metrics and max(memory_metrics) > self._thresholds.memory_warning_percent:
            recommendations.append(
                f"High memory usage detected (max {max(memory_metrics):.1f}%). "
                "Consider enabling gradient checkpointing or reducing batch size."
            )

        # GPU memory recommendations
        gpu_metrics = [h.metrics.get("gpu_memory_percent", 0) for h in recent if "gpu_memory_percent" in h.metrics]
        if gpu_metrics and max(gpu_metrics) > self._thresholds.gpu_memory_warning_percent:
            recommendations.append(
                f"High GPU memory usage detected (max {max(gpu_metrics):.1f}%). "
                "Consider using memory-efficient attention or model parallelism."
            )

        # Trend-based recommendations
        if trend == HealthTrend.DEGRADING:
            recommendations.append(
                "Health is degrading. Investigate root cause and consider proactive scaling."
            )
        elif trend == HealthTrend.CRITICAL:
            recommendations.append(
                "CRITICAL: System health is poor. Immediate attention required."
            )

        # Disk recommendations
        disk_metrics = [h.metrics.get("disk_percent", 0) for h in recent]
        if disk_metrics and max(disk_metrics) > self._thresholds.disk_warning_percent:
            recommendations.append(
                f"High disk usage detected ({max(disk_metrics):.1f}%). "
                "Clean up checkpoints or logs."
            )

        return recommendations


def create_enhanced_health_monitor(
    model: nn.Module | None = None,
    model_name: str = "model",
    enable_predictive: bool = True,
) -> EnhancedHealthMonitor:
    """
    Create an enhanced health monitor.

    Args:
        model: Model to monitor
        model_name: Name of the model
        enable_predictive: Enable predictive health analysis

    Returns:
        Configured EnhancedHealthMonitor
    """
    return EnhancedHealthMonitor(
        model=model,
        model_name=model_name,
        enable_predictive=enable_predictive,
    )


# Export all public APIs
__all__ = [
    "ResourceType",
    "HealthTrend",
    "ResourceThresholds",
    "HealthHistoryEntry",
    "PredictiveHealthReport",
    "SystemResourceMonitor",
    "EnhancedHealthMonitor",
    "create_enhanced_health_monitor",
]
