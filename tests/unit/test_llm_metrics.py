"""
Tests for LLM Serving Metrics

Tests LLMRequestMetrics, GenerationTimer, and LLMMetricsCollector.
"""

import time

import pytest

from torchbridge.monitoring.llm_metrics import (
    GenerationTimer,
    LLMMetricsCollector,
    LLMMetricsSnapshot,
    LLMRequestMetrics,
)

# =============================================================================
# LLMRequestMetrics Tests
# =============================================================================


class TestRequestMetrics:
    """Tests for LLMRequestMetrics dataclass."""

    def test_tokens_per_second(self):
        """tokens_per_second should compute correctly."""
        m = LLMRequestMetrics(
            request_id="test",
            prompt_tokens=10,
            generated_tokens=100,
            ttft_ms=50.0,
            tpot_ms=10.0,
            total_latency_ms=1000.0,
        )
        assert m.tokens_per_second == pytest.approx(100.0)

    def test_tokens_per_second_zero_latency(self):
        """tokens_per_second should return 0 for zero latency."""
        m = LLMRequestMetrics(
            request_id="test",
            prompt_tokens=10,
            generated_tokens=50,
            ttft_ms=0.0,
            tpot_ms=0.0,
            total_latency_ms=0.0,
        )
        assert m.tokens_per_second == 0.0

    def test_cache_hit_default_false(self):
        """cache_hit should default to False."""
        m = LLMRequestMetrics(
            request_id="test",
            prompt_tokens=10,
            generated_tokens=50,
            ttft_ms=50.0,
            tpot_ms=10.0,
            total_latency_ms=500.0,
        )
        assert m.cache_hit is False

    def test_cache_hit_set(self):
        """cache_hit should be settable."""
        m = LLMRequestMetrics(
            request_id="test",
            prompt_tokens=10,
            generated_tokens=50,
            ttft_ms=50.0,
            tpot_ms=10.0,
            total_latency_ms=500.0,
            cache_hit=True,
        )
        assert m.cache_hit is True


# =============================================================================
# GenerationTimer Tests
# =============================================================================


class TestGenerationTimer:
    """Tests for the GenerationTimer context manager."""

    def test_basic_timing(self):
        """Timer should capture total latency."""
        timer = GenerationTimer(prompt_tokens=10)
        with timer:
            time.sleep(0.01)
            timer.record_first_token()
        metrics = timer.finalize(generated_tokens=5)
        assert metrics.total_latency_ms > 0
        assert metrics.prompt_tokens == 10
        assert metrics.generated_tokens == 5

    def test_ttft_captured(self):
        """TTFT should be captured when record_first_token is called."""
        timer = GenerationTimer(prompt_tokens=10)
        with timer:
            time.sleep(0.01)
            timer.record_first_token()
            time.sleep(0.01)
        metrics = timer.finalize(generated_tokens=5)
        assert metrics.ttft_ms > 0
        assert metrics.ttft_ms < metrics.total_latency_ms

    def test_tpot_computed(self):
        """TPOT should be average decode time per token."""
        timer = GenerationTimer(prompt_tokens=10)
        with timer:
            timer.record_first_token()
            for _ in range(5):
                timer.record_token()
        metrics = timer.finalize(generated_tokens=5)
        assert metrics.tpot_ms > 0

    def test_itl_series(self):
        """itl_series should contain inter-token latencies."""
        timer = GenerationTimer()
        with timer:
            timer.record_first_token()
            time.sleep(0.005)
            timer.record_token()
            time.sleep(0.005)
            timer.record_token()
        itl = timer.itl_series
        assert len(itl) == 2
        assert all(v > 0 for v in itl)

    def test_itl_series_empty_without_tokens(self):
        """itl_series should be empty with fewer than 2 token times."""
        timer = GenerationTimer()
        with timer:
            timer.record_first_token()
        assert timer.itl_series == []

    def test_request_id_auto_generated(self):
        """request_id should be auto-generated if not provided."""
        timer = GenerationTimer()
        with timer:
            timer.record_first_token()
        metrics = timer.finalize(generated_tokens=1)
        assert len(metrics.request_id) > 0

    def test_custom_request_id(self):
        """Custom request_id should be preserved."""
        timer = GenerationTimer(request_id="custom-123")
        with timer:
            timer.record_first_token()
        metrics = timer.finalize(generated_tokens=1)
        assert metrics.request_id == "custom-123"

    def test_cache_hit_flag(self):
        """cache_hit should be passed through to metrics."""
        timer = GenerationTimer(cache_hit=True)
        with timer:
            timer.record_first_token()
        metrics = timer.finalize(generated_tokens=1)
        assert metrics.cache_hit is True

    def test_finalize_without_context_raises(self):
        """finalize without starting timer should raise RuntimeError."""
        timer = GenerationTimer()
        with pytest.raises(RuntimeError, match="Timer was never started"):
            timer.finalize(generated_tokens=1)

    def test_finalize_without_first_token(self):
        """finalize without record_first_token should use total time as TTFT."""
        timer = GenerationTimer()
        with timer:
            time.sleep(0.01)
        metrics = timer.finalize(generated_tokens=0)
        assert metrics.ttft_ms == pytest.approx(metrics.total_latency_ms)

    def test_zero_generated_tokens(self):
        """Zero generated tokens should produce 0 TPOT."""
        timer = GenerationTimer()
        with timer:
            pass
        metrics = timer.finalize(generated_tokens=0)
        assert metrics.tpot_ms == 0.0


# =============================================================================
# LLMMetricsCollector Tests
# =============================================================================


class TestMetricsCollector:
    """Tests for the LLMMetricsCollector."""

    def _make_metrics(self, **kwargs):
        defaults = {
            "request_id": "test",
            "prompt_tokens": 10,
            "generated_tokens": 50,
            "ttft_ms": 25.0,
            "tpot_ms": 10.0,
            "total_latency_ms": 500.0,
        }
        defaults.update(kwargs)
        return LLMRequestMetrics(**defaults)

    def test_empty_snapshot(self):
        """Snapshot of empty collector should have zero values."""
        collector = LLMMetricsCollector()
        snap = collector.get_snapshot()
        assert snap.total_requests == 0
        assert snap.ttft_p50_ms == 0.0
        assert snap.cache_hit_rate == 0.0

    def test_single_request(self):
        """Single request should produce correct snapshot."""
        collector = LLMMetricsCollector()
        collector.record_request(self._make_metrics())
        snap = collector.get_snapshot()
        assert snap.total_requests == 1
        assert snap.ttft_p50_ms == pytest.approx(25.0)
        assert snap.tpot_p50_ms == pytest.approx(10.0)

    def test_multiple_requests_percentiles(self):
        """Multiple requests should produce reasonable percentiles."""
        collector = LLMMetricsCollector()
        for i in range(100):
            collector.record_request(
                self._make_metrics(ttft_ms=float(i), tpot_ms=float(i) * 0.5)
            )
        snap = collector.get_snapshot()
        assert snap.total_requests == 100
        assert snap.ttft_p50_ms > 0
        assert snap.ttft_p95_ms > snap.ttft_p50_ms
        assert snap.ttft_p99_ms >= snap.ttft_p95_ms

    def test_cache_hit_rate(self):
        """Cache hit rate should reflect recorded hits."""
        collector = LLMMetricsCollector()
        collector.record_request(self._make_metrics(cache_hit=True))
        collector.record_request(self._make_metrics(cache_hit=False))
        collector.record_request(self._make_metrics(cache_hit=True))
        snap = collector.get_snapshot()
        assert snap.cache_hit_rate == pytest.approx(2 / 3)

    def test_tokens_per_second(self):
        """tokens_per_second should be average of recorded TPS."""
        collector = LLMMetricsCollector()
        collector.record_request(self._make_metrics(
            generated_tokens=100, total_latency_ms=1000.0
        ))
        snap = collector.get_snapshot()
        assert snap.tokens_per_second > 0

    def test_itl_recording(self):
        """ITL series should be recorded and included in snapshot."""
        collector = LLMMetricsCollector()
        collector.record_request(
            self._make_metrics(),
            itl_series=[5.0, 6.0, 7.0, 8.0, 9.0],
        )
        snap = collector.get_snapshot()
        assert snap.itl_p50_ms > 0

    def test_avg_batch_size(self):
        """avg_batch_size should reflect recorded batch sizes."""
        collector = LLMMetricsCollector()
        collector.record_request(self._make_metrics(), batch_size=4)
        collector.record_request(self._make_metrics(), batch_size=8)
        snap = collector.get_snapshot()
        assert snap.avg_batch_size == pytest.approx(6.0)

    def test_reset(self):
        """Reset should clear all collected metrics."""
        collector = LLMMetricsCollector()
        collector.record_request(self._make_metrics())
        collector.reset()
        snap = collector.get_snapshot()
        assert snap.total_requests == 0

    def test_window_size_respected(self):
        """Rolling window should respect window_size limit."""
        collector = LLMMetricsCollector(window_size=10)
        for i in range(100):
            collector.record_request(self._make_metrics(ttft_ms=float(i)))
        snap = collector.get_snapshot()
        assert snap.total_requests == 100
        # p50 should reflect only recent samples (90-99 range)
        assert snap.ttft_p50_ms >= 90.0

    def test_snapshot_is_dataclass(self):
        """Snapshot should be an LLMMetricsSnapshot dataclass."""
        collector = LLMMetricsCollector()
        snap = collector.get_snapshot()
        assert isinstance(snap, LLMMetricsSnapshot)

    def test_prometheus_disabled_by_default(self):
        """Prometheus should be disabled by default."""
        collector = LLMMetricsCollector()
        assert collector._prom_enabled is False
