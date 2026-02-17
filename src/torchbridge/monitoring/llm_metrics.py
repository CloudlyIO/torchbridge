"""
LLM Serving Metrics

Specialized metrics for LLM inference serving: Time-To-First-Token (TTFT),
Time-Per-Output-Token (TPOT), Inter-Token Latency (ITL), throughput,
cache hit rates, and batch utilization. Optional Prometheus integration.
"""

from __future__ import annotations

import logging
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# -- Data classes -------------------------------------------------------------


@dataclass
class LLMRequestMetrics:
    """Metrics for a single LLM generation request."""
    request_id: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float  # Time to first token
    tpot_ms: float  # Time per output token (average)
    total_latency_ms: float
    cache_hit: bool = False

    @property
    def tokens_per_second(self) -> float:
        """Throughput in tokens/second."""
        if self.total_latency_ms <= 0:
            return 0.0
        return self.generated_tokens / (self.total_latency_ms / 1000.0)


@dataclass
class LLMMetricsSnapshot:
    """Aggregated snapshot of LLM serving metrics."""
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    tpot_p99_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    cache_hit_rate: float = 0.0
    tokens_per_second: float = 0.0
    avg_batch_size: float = 0.0
    total_requests: int = 0


# -- Generation Timer ---------------------------------------------------------


class GenerationTimer:
    """Context manager for timing LLM token generation.

    Records TTFT, per-token decode times, and total latency.

    Example::

        timer = GenerationTimer(prompt_tokens=128)
        with timer:
            first_token = generate_first()
            timer.record_first_token()
            for tok in generate_rest():
                timer.record_token()
        metrics = timer.finalize(generated_tokens=50)
    """

    def __init__(
        self,
        prompt_tokens: int = 0,
        request_id: str | None = None,
        cache_hit: bool = False,
    ) -> None:
        self._prompt_tokens = prompt_tokens
        self._request_id = request_id or str(uuid.uuid4())
        self._cache_hit = cache_hit
        self._start_time: float | None = None
        self._first_token_time: float | None = None
        self._token_times: list[float] = []
        self._end_time: float | None = None

    def __enter__(self) -> GenerationTimer:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._end_time = time.perf_counter()

    def record_first_token(self) -> None:
        """Record the time when the first token is generated."""
        self._first_token_time = time.perf_counter()
        self._token_times.append(self._first_token_time)

    def record_token(self) -> None:
        """Record the time when a subsequent token is generated."""
        self._token_times.append(time.perf_counter())

    @property
    def itl_series(self) -> list[float]:
        """Inter-token latencies in milliseconds."""
        if len(self._token_times) < 2:
            return []
        return [
            (self._token_times[i] - self._token_times[i - 1]) * 1000.0
            for i in range(1, len(self._token_times))
        ]

    def finalize(self, generated_tokens: int) -> LLMRequestMetrics:
        """Produce final request metrics.

        Args:
            generated_tokens: Number of tokens generated.

        Returns:
            LLMRequestMetrics with computed TTFT, TPOT, total latency.
        """
        if self._start_time is None:
            raise RuntimeError("Timer was never started (use as context manager)")

        end = self._end_time or time.perf_counter()
        total_ms = (end - self._start_time) * 1000.0

        # TTFT
        if self._first_token_time is not None:
            ttft_ms = (self._first_token_time - self._start_time) * 1000.0
        else:
            ttft_ms = total_ms

        # TPOT (average decode time per token, excluding prefill)
        if generated_tokens > 0 and self._first_token_time is not None:
            decode_time = end - self._first_token_time
            tpot_ms = (decode_time / generated_tokens) * 1000.0
        elif generated_tokens > 0:
            tpot_ms = total_ms / generated_tokens
        else:
            tpot_ms = 0.0

        return LLMRequestMetrics(
            request_id=self._request_id,
            prompt_tokens=self._prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_ms,
            cache_hit=self._cache_hit,
        )


# -- Metrics Collector --------------------------------------------------------


class LLMMetricsCollector:
    """Collects and aggregates LLM serving metrics with rolling windows.

    Thread-safe. Optionally integrates with Prometheus for scraping.

    Args:
        window_size: Number of recent samples to keep for percentile computation.
        namespace: Prometheus metric namespace.
        enable_prometheus: Whether to create Prometheus metrics.
    """

    def __init__(
        self,
        window_size: int = 1000,
        namespace: str = "torchbridge_llm",
        enable_prometheus: bool = False,
    ) -> None:
        self._window_size = window_size
        self._lock = threading.Lock()

        # Rolling windows
        self._ttft_samples: deque[float] = deque(maxlen=window_size)
        self._tpot_samples: deque[float] = deque(maxlen=window_size)
        self._itl_samples: deque[float] = deque(maxlen=window_size)
        self._tps_samples: deque[float] = deque(maxlen=window_size)
        self._batch_sizes: deque[int] = deque(maxlen=window_size)

        # Counters
        self._total_requests = 0
        self._cache_hits = 0

        # Prometheus
        self._prom_enabled = enable_prometheus and PROMETHEUS_AVAILABLE
        self._prom_ttft: Histogram | None = None
        self._prom_tpot: Histogram | None = None
        self._prom_itl: Histogram | None = None
        self._prom_requests: Counter | None = None
        self._prom_tokens: Counter | None = None
        self._prom_cache_hits: Counter | None = None
        self._prom_tps: Gauge | None = None

        if self._prom_enabled:
            self._init_prometheus(namespace)

    def _init_prometheus(self, namespace: str) -> None:
        """Initialize Prometheus metrics."""
        latency_buckets = (1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

        self._prom_ttft = Histogram(
            f"{namespace}_ttft_milliseconds",
            "Time to first token in milliseconds",
            labelnames=["model"],
            buckets=latency_buckets,
        )
        self._prom_tpot = Histogram(
            f"{namespace}_tpot_milliseconds",
            "Time per output token in milliseconds",
            labelnames=["model"],
            buckets=latency_buckets,
        )
        self._prom_itl = Histogram(
            f"{namespace}_itl_milliseconds",
            "Inter-token latency in milliseconds",
            labelnames=["model"],
            buckets=latency_buckets,
        )
        self._prom_requests = Counter(
            f"{namespace}_requests_total",
            "Total generation requests",
            labelnames=["model", "status"],
        )
        self._prom_tokens = Counter(
            f"{namespace}_tokens_total",
            "Total tokens generated",
            labelnames=["model"],
        )
        self._prom_cache_hits = Counter(
            f"{namespace}_cache_hits_total",
            "Total prefix cache hits",
            labelnames=["model"],
        )
        self._prom_tps = Gauge(
            f"{namespace}_tokens_per_second",
            "Current tokens per second throughput",
            labelnames=["model"],
        )

    def record_request(
        self,
        metrics: LLMRequestMetrics,
        model_name: str = "default",
        batch_size: int = 1,
        itl_series: list[float] | None = None,
    ) -> None:
        """Record metrics from a completed generation request.

        Args:
            metrics: Request-level metrics from GenerationTimer.finalize().
            model_name: Model identifier for Prometheus labels.
            batch_size: Batch size used for this request.
            itl_series: Optional per-token inter-token latencies in ms.
        """
        with self._lock:
            self._total_requests += 1
            self._ttft_samples.append(metrics.ttft_ms)
            self._tpot_samples.append(metrics.tpot_ms)
            self._tps_samples.append(metrics.tokens_per_second)
            self._batch_sizes.append(batch_size)

            if metrics.cache_hit:
                self._cache_hits += 1

            if itl_series:
                self._itl_samples.extend(itl_series)

        # Prometheus
        if self._prom_enabled:
            if self._prom_ttft is not None:
                self._prom_ttft.labels(model=model_name).observe(metrics.ttft_ms)
            if self._prom_tpot is not None:
                self._prom_tpot.labels(model=model_name).observe(metrics.tpot_ms)
            if self._prom_requests is not None:
                self._prom_requests.labels(
                    model=model_name, status="success"
                ).inc()
            if self._prom_tokens is not None:
                self._prom_tokens.labels(model=model_name).inc(
                    metrics.generated_tokens
                )
            if metrics.cache_hit and self._prom_cache_hits is not None:
                self._prom_cache_hits.labels(model=model_name).inc()
            if self._prom_tps is not None:
                self._prom_tps.labels(model=model_name).set(
                    metrics.tokens_per_second
                )
            if itl_series and self._prom_itl is not None:
                for itl in itl_series:
                    self._prom_itl.labels(model=model_name).observe(itl)

    def get_snapshot(self) -> LLMMetricsSnapshot:
        """Return aggregated metrics snapshot with percentiles."""
        with self._lock:
            snapshot = LLMMetricsSnapshot(total_requests=self._total_requests)

            if self._ttft_samples:
                sorted_ttft = sorted(self._ttft_samples)
                snapshot.ttft_p50_ms = _percentile(sorted_ttft, 50)
                snapshot.ttft_p95_ms = _percentile(sorted_ttft, 95)
                snapshot.ttft_p99_ms = _percentile(sorted_ttft, 99)

            if self._tpot_samples:
                sorted_tpot = sorted(self._tpot_samples)
                snapshot.tpot_p50_ms = _percentile(sorted_tpot, 50)
                snapshot.tpot_p95_ms = _percentile(sorted_tpot, 95)
                snapshot.tpot_p99_ms = _percentile(sorted_tpot, 99)

            if self._itl_samples:
                sorted_itl = sorted(self._itl_samples)
                snapshot.itl_p50_ms = _percentile(sorted_itl, 50)
                snapshot.itl_p95_ms = _percentile(sorted_itl, 95)
                snapshot.itl_p99_ms = _percentile(sorted_itl, 99)

            if self._total_requests > 0:
                snapshot.cache_hit_rate = (
                    self._cache_hits / self._total_requests
                )

            if self._tps_samples:
                snapshot.tokens_per_second = statistics.mean(self._tps_samples)

            if self._batch_sizes:
                snapshot.avg_batch_size = statistics.mean(self._batch_sizes)

            return snapshot

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._ttft_samples.clear()
            self._tpot_samples.clear()
            self._itl_samples.clear()
            self._tps_samples.clear()
            self._batch_sizes.clear()
            self._total_requests = 0
            self._cache_hits = 0


# -- Helpers ------------------------------------------------------------------


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute percentile from pre-sorted data."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (pct / 100.0) * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac
