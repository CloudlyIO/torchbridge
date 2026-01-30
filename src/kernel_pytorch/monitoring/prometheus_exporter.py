"""
Prometheus Metrics Exporter for KernelPyTorch

This module provides Prometheus-compatible metrics export for monitoring
KernelPyTorch model inference in production environments.

Features:
- Inference latency histograms
- Throughput counters
- GPU memory usage gauges
- Model-specific metrics
- System resource metrics
- Custom metric registration

Example:
    ```python
    from kernel_pytorch.monitoring import MetricsExporter, start_metrics_server

    # Create exporter
    exporter = MetricsExporter(model_name="transformer")

    # Start HTTP server for Prometheus scraping
    start_metrics_server(exporter, port=9090)

    # Record metrics during inference
    with exporter.track_inference():
        output = model(input)

    # Or manually
    exporter.record_inference(latency_ms=5.2, batch_size=32)
    ```

Version: 0.3.10
"""

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,  # noqa: F401
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,  # noqa: F401
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Stub classes for when prometheus_client is not available
    CollectorRegistry = object


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""

    # Server settings
    port: int = 9090
    host: str = "0.0.0.0"

    # Metric settings
    model_name: str = "model"
    namespace: str = "kernelpytorch"
    subsystem: str = "inference"

    # Collection settings
    enable_gpu_metrics: bool = True
    enable_system_metrics: bool = True
    collection_interval_seconds: float = 15.0

    # Histogram buckets (in milliseconds)
    latency_buckets: list[float] = field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    )


@dataclass
class InferenceMetrics:
    """Inference-related metrics snapshot."""

    total_requests: int = 0
    total_samples: int = 0
    total_errors: int = 0
    average_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_samples_per_second: float = 0.0


@dataclass
class SystemMetrics:
    """System resource metrics snapshot."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0


@dataclass
class ModelMetrics:
    """Model-specific metrics snapshot."""

    model_name: str = ""
    model_version: str = ""
    optimization_level: str = ""
    precision: str = ""
    device: str = ""
    is_compiled: bool = False


class MetricsExporter:
    """
    Prometheus metrics exporter for KernelPyTorch.

    Exports metrics in Prometheus format for scraping and visualization.
    """

    def __init__(self, config: MetricsConfig | None = None):
        """
        Initialize the metrics exporter.

        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self._registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self._lock = threading.Lock()

        # Internal tracking
        self._latencies: list[float] = []
        self._start_time = time.time()
        self._total_requests = 0
        self._total_samples = 0
        self._total_errors = 0

        # Initialize Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        else:
            logger.warning(
                "prometheus_client not available. "
                "Install with: pip install prometheus-client"
            )

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        ns = self.config.namespace
        sub = self.config.subsystem

        # Counters
        self._request_counter = Counter(
            f"{ns}_{sub}_requests_total",
            "Total number of inference requests",
            ["model", "status"],
            registry=self._registry,
        )

        self._sample_counter = Counter(
            f"{ns}_{sub}_samples_total",
            "Total number of samples processed",
            ["model"],
            registry=self._registry,
        )

        # Histograms
        self._latency_histogram = Histogram(
            f"{ns}_{sub}_latency_milliseconds",
            "Inference latency in milliseconds",
            ["model"],
            buckets=self.config.latency_buckets,
            registry=self._registry,
        )

        self._batch_size_histogram = Histogram(
            f"{ns}_{sub}_batch_size",
            "Batch sizes of inference requests",
            ["model"],
            buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256],
            registry=self._registry,
        )

        # Gauges
        self._throughput_gauge = Gauge(
            f"{ns}_{sub}_throughput_samples_per_second",
            "Current throughput in samples per second",
            ["model"],
            registry=self._registry,
        )

        self._gpu_memory_gauge = Gauge(
            f"{ns}_gpu_memory_used_bytes",
            "GPU memory currently in use",
            ["device"],
            registry=self._registry,
        )

        self._gpu_memory_total_gauge = Gauge(
            f"{ns}_gpu_memory_total_bytes",
            "Total GPU memory available",
            ["device"],
            registry=self._registry,
        )

        self._model_info_gauge = Gauge(
            f"{ns}_model_info",
            "Model information",
            ["model", "version", "device", "precision"],
            registry=self._registry,
        )

    def record_inference(
        self,
        latency_ms: float,
        batch_size: int = 1,
        success: bool = True,
    ) -> None:
        """
        Record an inference request.

        Args:
            latency_ms: Inference latency in milliseconds
            batch_size: Number of samples in the batch
            success: Whether the inference was successful
        """
        with self._lock:
            self._latencies.append(latency_ms)
            self._total_requests += 1

            if success:
                self._total_samples += batch_size
            else:
                self._total_errors += 1

        if PROMETHEUS_AVAILABLE:
            status = "success" if success else "error"
            model = self.config.model_name

            self._request_counter.labels(model=model, status=status).inc()
            self._sample_counter.labels(model=model).inc(batch_size)
            self._latency_histogram.labels(model=model).observe(latency_ms)
            self._batch_size_histogram.labels(model=model).observe(batch_size)

            # Update throughput
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                throughput = self._total_samples / elapsed
                self._throughput_gauge.labels(model=model).set(throughput)

    @contextmanager
    def track_inference(self, batch_size: int = 1):
        """
        Context manager to track inference timing.

        Args:
            batch_size: Number of samples in the batch

        Example:
            with exporter.track_inference(batch_size=32):
                output = model(input)
        """
        start_time = time.time()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.record_inference(latency_ms, batch_size, success)

    def update_gpu_metrics(self) -> None:
        """Update GPU memory metrics."""
        if not torch.cuda.is_available():
            return

        if not PROMETHEUS_AVAILABLE:
            return

        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory

            self._gpu_memory_gauge.labels(device=device).set(memory_allocated)
            self._gpu_memory_total_gauge.labels(device=device).set(memory_total)

    def set_model_info(
        self,
        version: str = "1.0",
        device: str = "cpu",
        precision: str = "fp32",
    ) -> None:
        """
        Set model information metric.

        Args:
            version: Model version
            device: Device being used
            precision: Precision mode (fp32, fp16, fp8, etc.)
        """
        if PROMETHEUS_AVAILABLE:
            self._model_info_gauge.labels(
                model=self.config.model_name,
                version=version,
                device=device,
                precision=precision,
            ).set(1)

    def get_inference_metrics(self) -> InferenceMetrics:
        """Get current inference metrics snapshot."""
        with self._lock:
            latencies = sorted(self._latencies) if self._latencies else [0]
            n = len(latencies)

            elapsed = time.time() - self._start_time
            throughput = self._total_samples / elapsed if elapsed > 0 else 0

            return InferenceMetrics(
                total_requests=self._total_requests,
                total_samples=self._total_samples,
                total_errors=self._total_errors,
                average_latency_ms=sum(latencies) / n if n > 0 else 0,
                p50_latency_ms=latencies[int(n * 0.50)] if n > 0 else 0,
                p95_latency_ms=latencies[int(n * 0.95)] if n > 0 else 0,
                p99_latency_ms=latencies[int(n * 0.99)] if n > 0 else 0,
                throughput_samples_per_second=throughput,
            )

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics snapshot."""
        metrics = SystemMetrics()

        # Try to get CPU/memory metrics
        try:
            import psutil
            metrics.cpu_percent = psutil.cpu_percent()
            metrics.memory_percent = psutil.virtual_memory().percent
        except ImportError:
            pass

        # Get GPU metrics
        if torch.cuda.is_available():
            metrics.gpu_memory_used_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            metrics.gpu_memory_total_mb = total_memory / (1024 * 1024)

        return metrics

    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if PROMETHEUS_AVAILABLE:
            self.update_gpu_metrics()
            return generate_latest(self._registry).decode("utf-8")
        else:
            # Return basic metrics as text
            metrics = self.get_inference_metrics()
            lines = [
                "# HELP kernelpytorch_inference_requests_total Total requests",
                "# TYPE kernelpytorch_inference_requests_total counter",
                f"kernelpytorch_inference_requests_total {metrics.total_requests}",
                "# HELP kernelpytorch_inference_latency_ms Average latency",
                "# TYPE kernelpytorch_inference_latency_ms gauge",
                f"kernelpytorch_inference_latency_ms {metrics.average_latency_ms:.2f}",
            ]
            return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._start_time = time.time()
            self._total_requests = 0
            self._total_samples = 0
            self._total_errors = 0


class MetricsHTTPHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    exporter: MetricsExporter | None = None

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_error(404, "Not Found")

    def _serve_metrics(self):
        """Serve Prometheus metrics."""
        if self.exporter is None:
            self.send_error(500, "Exporter not initialized")
            return

        metrics_text = self.exporter.get_metrics_text()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(metrics_text.encode("utf-8"))

    def _serve_health(self):
        """Serve health check."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "healthy"}).encode("utf-8"))

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def start_metrics_server(
    exporter: MetricsExporter,
    port: int | None = None,
    host: str = "0.0.0.0",
    background: bool = True,
) -> HTTPServer | None:
    """
    Start an HTTP server for Prometheus metrics scraping.

    Args:
        exporter: MetricsExporter instance
        port: Port to listen on (default from config)
        host: Host to bind to
        background: Run in background thread

    Returns:
        HTTPServer instance if background=False, None otherwise
    """
    port = port or exporter.config.port

    # Create handler class with exporter
    class Handler(MetricsHTTPHandler):
        pass
    Handler.exporter = exporter

    server = HTTPServer((host, port), Handler)
    logger.info(f"Starting metrics server on {host}:{port}")

    if background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return None
    else:
        return server


def create_metrics_exporter(
    model_name: str = "model",
    port: int = 9090,
    enable_gpu_metrics: bool = True,
) -> MetricsExporter:
    """
    Create a metrics exporter with common configuration.

    Args:
        model_name: Name of the model being served
        port: Port for metrics server
        enable_gpu_metrics: Enable GPU metrics collection

    Returns:
        Configured MetricsExporter instance
    """
    config = MetricsConfig(
        model_name=model_name,
        port=port,
        enable_gpu_metrics=enable_gpu_metrics,
    )
    return MetricsExporter(config)
