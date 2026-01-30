"""
Performance Tracking and Regression Detection Module

Tracks performance metrics over time and detects when optimizations
degrade performance instead of improving it.

Stage 3B: Performance Regression Detection
"""

import json
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class MetricType(Enum):
    """Types of performance metrics."""
    LATENCY = "latency_ms"
    THROUGHPUT = "throughput_samples_per_sec"
    MEMORY = "memory_mb"
    ACCURACY = "accuracy"


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""
    NONE = "none"
    MINOR = "minor"  # < 10% regression
    MODERATE = "moderate"  # 10-25% regression
    SEVERE = "severe"  # > 25% regression


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""

    # Model identification
    model_name: str
    model_hash: str
    timestamp: str

    # Performance metrics
    latency_ms: float
    throughput: float  # samples/second
    memory_mb: float

    # Configuration
    backend: str
    optimization_level: str
    device: str
    batch_size: int

    # Optional metadata
    accuracy: float | None = None
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RegressionResult:
    """Result of regression detection."""

    has_regression: bool
    severity: RegressionSeverity
    metric_type: MetricType
    baseline_value: float
    current_value: float
    change_percent: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            'has_regression': self.has_regression,
            'severity': self.severity.value,
            'metric_type': self.metric_type.value,
            'baseline_value': self.baseline_value,
            'current_value': self.current_value,
            'change_percent': self.change_percent,
            'message': self.message
        }


class PerformanceTracker:
    """
    Track performance metrics and detect regressions.

    Stores performance baselines and compares new measurements
    against them to detect when optimizations degrade performance.
    """

    def __init__(self, storage_path: str | None = None):
        """
        Initialize performance tracker.

        Args:
            storage_path: Path to store performance metrics (JSON file)
        """
        if storage_path is None:
            storage_path = Path.home() / '.kernel_pytorch' / 'performance_metrics.json'

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing metrics
        self.metrics_history: dict[str, list[PerformanceMetrics]] = {}
        self._load_metrics()

        # Regression thresholds (as fractions)
        self.minor_threshold = 0.10  # 10%
        self.moderate_threshold = 0.25  # 25%

    def _load_metrics(self):
        """Load metrics from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)

                for model_hash, metrics_list in data.items():
                    self.metrics_history[model_hash] = [
                        PerformanceMetrics.from_dict(m) for m in metrics_list
                    ]
            except Exception as e:
                warnings.warn(f"Failed to load metrics: {e}", stacklevel=2)
                self.metrics_history = {}

    def _save_metrics(self):
        """Save metrics to storage."""
        try:
            data = {
                model_hash: [m.to_dict() for m in metrics]
                for model_hash, metrics in self.metrics_history.items()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            warnings.warn(f"Failed to save metrics: {e}", stacklevel=2)

    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute unique hash for model architecture."""
        # Use model structure as hash
        structure = str(model)
        return f"{model.__class__.__name__}_{hash(structure) % 10**8}"

    def benchmark_model(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> tuple[float, float, float]:
        """
        Benchmark model performance.

        Args:
            model: PyTorch model to benchmark
            sample_inputs: Sample inputs for the model
            num_iterations: Number of iterations for benchmarking
            warmup_iterations: Number of warmup iterations

        Returns:
            Tuple of (latency_ms, throughput, memory_mb)
        """
        model.eval()
        device = next(model.parameters()).device

        # Move inputs to correct device
        if sample_inputs.device != device:
            sample_inputs = sample_inputs.to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(sample_inputs)

        # Benchmark latency
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(sample_inputs)
        end_time = time.time()

        latency_ms = (end_time - start_time) / num_iterations * 1000

        # Calculate throughput
        batch_size = sample_inputs.shape[0] if len(sample_inputs.shape) > 0 else 1
        throughput = (batch_size * num_iterations) / (end_time - start_time)

        # Estimate memory usage
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                torch.cuda.reset_peak_memory_stats(device)
            else:
                # Rough estimate for CPU
                param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
                memory_mb = param_memory / 1024**2
        except Exception:
            memory_mb = 0.0

        return latency_ms, throughput, memory_mb

    def record_performance(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        model_name: str,
        backend: str = "cpu",
        optimization_level: str = "conservative",
        accuracy: float | None = None,
        additional_metrics: dict[str, Any] | None = None
    ) -> PerformanceMetrics:
        """
        Record performance metrics for a model.

        Args:
            model: PyTorch model
            sample_inputs: Sample inputs for benchmarking
            model_name: Name of the model
            backend: Backend used (nvidia/tpu/cpu)
            optimization_level: Optimization level applied
            accuracy: Optional accuracy metric
            additional_metrics: Optional additional metrics

        Returns:
            PerformanceMetrics object
        """
        # Benchmark the model
        latency_ms, throughput, memory_mb = self.benchmark_model(model, sample_inputs)

        # Get device
        device = str(next(model.parameters()).device)

        # Get batch size
        batch_size = sample_inputs.shape[0] if len(sample_inputs.shape) > 0 else 1

        # Compute model hash
        model_hash = self._compute_model_hash(model)

        # Create metrics
        metrics = PerformanceMetrics(
            model_name=model_name,
            model_hash=model_hash,
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms,
            throughput=throughput,
            memory_mb=memory_mb,
            backend=backend,
            optimization_level=optimization_level,
            device=device,
            batch_size=batch_size,
            accuracy=accuracy,
            additional_metrics=additional_metrics or {}
        )

        # Store metrics
        if model_hash not in self.metrics_history:
            self.metrics_history[model_hash] = []
        self.metrics_history[model_hash].append(metrics)

        # Save to disk
        self._save_metrics()

        return metrics

    def get_baseline(
        self,
        model: nn.Module,
        backend: str | None = None,
        optimization_level: str | None = None
    ) -> PerformanceMetrics | None:
        """
        Get baseline performance metrics for a model.

        Args:
            model: PyTorch model
            backend: Optional backend filter
            optimization_level: Optional optimization level filter

        Returns:
            Baseline PerformanceMetrics or None if no baseline exists
        """
        model_hash = self._compute_model_hash(model)

        if model_hash not in self.metrics_history:
            return None

        metrics_list = self.metrics_history[model_hash]

        # Filter by backend and optimization level if specified
        if backend:
            metrics_list = [m for m in metrics_list if m.backend == backend]
        if optimization_level:
            metrics_list = [m for m in metrics_list if m.optimization_level == optimization_level]

        if not metrics_list:
            return None

        # Return the first recorded metric as baseline
        return metrics_list[0]

    def detect_regression(
        self,
        model: nn.Module,
        current_metrics: PerformanceMetrics,
        baseline: PerformanceMetrics | None = None
    ) -> list[RegressionResult]:
        """
        Detect performance regressions.

        Args:
            model: PyTorch model
            current_metrics: Current performance metrics
            baseline: Optional baseline to compare against (auto-detected if None)

        Returns:
            List of RegressionResult objects
        """
        if baseline is None:
            baseline = self.get_baseline(model)

        if baseline is None:
            # No baseline, can't detect regression
            return []

        regressions = []

        # Check latency regression (higher is worse)
        if current_metrics.latency_ms > baseline.latency_ms:
            change_percent = (current_metrics.latency_ms - baseline.latency_ms) / baseline.latency_ms * 100

            severity = self._get_severity(change_percent)

            if severity != RegressionSeverity.NONE:
                regressions.append(RegressionResult(
                    has_regression=True,
                    severity=severity,
                    metric_type=MetricType.LATENCY,
                    baseline_value=baseline.latency_ms,
                    current_value=current_metrics.latency_ms,
                    change_percent=change_percent,
                    message=f"Latency regression detected: {baseline.latency_ms:.3f}ms → {current_metrics.latency_ms:.3f}ms ({change_percent:+.1f}%)"
                ))

        # Check throughput regression (lower is worse)
        if current_metrics.throughput < baseline.throughput:
            change_percent = (baseline.throughput - current_metrics.throughput) / baseline.throughput * 100

            severity = self._get_severity(change_percent)

            if severity != RegressionSeverity.NONE:
                regressions.append(RegressionResult(
                    has_regression=True,
                    severity=severity,
                    metric_type=MetricType.THROUGHPUT,
                    baseline_value=baseline.throughput,
                    current_value=current_metrics.throughput,
                    change_percent=-change_percent,  # Negative because throughput decreased
                    message=f"Throughput regression detected: {baseline.throughput:.1f} → {current_metrics.throughput:.1f} samples/sec ({-change_percent:+.1f}%)"
                ))

        # Check memory regression (higher is worse)
        if current_metrics.memory_mb > baseline.memory_mb * 1.1:  # Allow 10% variance
            change_percent = (current_metrics.memory_mb - baseline.memory_mb) / baseline.memory_mb * 100

            severity = self._get_severity(change_percent)

            if severity != RegressionSeverity.NONE:
                regressions.append(RegressionResult(
                    has_regression=True,
                    severity=severity,
                    metric_type=MetricType.MEMORY,
                    baseline_value=baseline.memory_mb,
                    current_value=current_metrics.memory_mb,
                    change_percent=change_percent,
                    message=f"Memory regression detected: {baseline.memory_mb:.1f}MB → {current_metrics.memory_mb:.1f}MB ({change_percent:+.1f}%)"
                ))

        return regressions

    def _get_severity(self, change_percent: float) -> RegressionSeverity:
        """Get regression severity based on change percentage."""
        abs_change = abs(change_percent)

        if abs_change >= self.moderate_threshold * 100:
            return RegressionSeverity.SEVERE
        elif abs_change >= self.minor_threshold * 100:
            return RegressionSeverity.MODERATE
        else:
            return RegressionSeverity.MINOR

    def warn_if_regression(
        self,
        model: nn.Module,
        current_metrics: PerformanceMetrics,
        baseline: PerformanceMetrics | None = None
    ):
        """
        Warn if performance regression is detected.

        Args:
            model: PyTorch model
            current_metrics: Current performance metrics
            baseline: Optional baseline to compare against
        """
        regressions = self.detect_regression(model, current_metrics, baseline)

        for regression in regressions:
            if regression.severity == RegressionSeverity.SEVERE:
                warnings.warn(f"  SEVERE REGRESSION: {regression.message}", UserWarning, stacklevel=2)
            elif regression.severity == RegressionSeverity.MODERATE:
                warnings.warn(f"  MODERATE REGRESSION: {regression.message}", UserWarning, stacklevel=2)
            else:
                warnings.warn(f"ℹ  Minor regression: {regression.message}", UserWarning, stacklevel=2)

    def get_performance_history(
        self,
        model: nn.Module,
        limit: int = 10
    ) -> list[PerformanceMetrics]:
        """
        Get performance history for a model.

        Args:
            model: PyTorch model
            limit: Maximum number of historical records to return

        Returns:
            List of PerformanceMetrics (most recent first)
        """
        model_hash = self._compute_model_hash(model)

        if model_hash not in self.metrics_history:
            return []

        history = self.metrics_history[model_hash]
        return list(reversed(history[-limit:]))

    def clear_history(self, model: nn.Module | None = None):
        """
        Clear performance history.

        Args:
            model: Optional specific model to clear (clears all if None)
        """
        if model is None:
            self.metrics_history = {}
        else:
            model_hash = self._compute_model_hash(model)
            if model_hash in self.metrics_history:
                del self.metrics_history[model_hash]

        self._save_metrics()


# Global tracker instance
_global_tracker: PerformanceTracker | None = None


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


def track_performance(
    model: nn.Module,
    sample_inputs: torch.Tensor,
    model_name: str,
    **kwargs
) -> PerformanceMetrics:
    """Convenience function to track performance."""
    return get_performance_tracker().record_performance(
        model, sample_inputs, model_name, **kwargs
    )


def detect_regression(
    model: nn.Module,
    current_metrics: PerformanceMetrics,
    **kwargs
) -> list[RegressionResult]:
    """Convenience function to detect regression."""
    return get_performance_tracker().detect_regression(
        model, current_metrics, **kwargs
    )
