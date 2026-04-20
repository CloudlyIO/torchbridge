"""
Config-path performance regression tests.

Ensures that TorchBridge's own config-generation critical paths stay fast
(sub-millisecond for matrix lookups, sub-10ms for config generation).
All thresholds are 50-100× the expected actual latency to avoid flakiness
while still catching catastrophic regressions (O(n²) loops, accidental I/O,
lock contention, network calls).

Mark: benchmark — can be skipped with `-m "not benchmark"` in fast CI runs.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.benchmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _elapsed_ms(fn, iterations: int) -> float:
    """Return total elapsed milliseconds for `iterations` calls to `fn`."""
    t0 = time.perf_counter()
    for _ in range(iterations):
        fn()
    return (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Matrix lookups — must each complete 1000 iterations in < 1 000 ms (< 1 ms each)
# ---------------------------------------------------------------------------


class TestMatrixLookupPerf:
    """Compatibility matrix lookups must be effectively instantaneous."""

    def test_quantization_matrix_lookup_under_1ms(self):
        from torchbridge.core.config import HardwareBackend
        from torchbridge.precision.compatibility import (
            QuantizationCompatibilityMatrix,
        )

        elapsed = _elapsed_ms(
            lambda: QuantizationCompatibilityMatrix.get_optimal_format(
                HardwareBackend.CUDA
            ),
            iterations=1000,
        )
        assert elapsed < 1000, (
            f"1000 × QuantizationCompatibilityMatrix.get_optimal_format took {elapsed:.1f}ms "
            f"(> 1 000 ms). Potential O(n) regression."
        )

    def test_attention_dispatch_matrix_under_1ms(self):
        from torchbridge.attention.dispatch.compatibility import AttentionDispatchMatrix
        from torchbridge.core.config import HardwareBackend

        elapsed = _elapsed_ms(
            lambda: AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.CPU),
            iterations=1000,
        )
        assert elapsed < 1000, (
            f"1000 × AttentionDispatchMatrix.get_supported_kernels took {elapsed:.1f}ms"
        )

    def test_adapter_matrix_lookup_under_1ms(self):
        from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
        from torchbridge.core.config import HardwareBackend

        elapsed = _elapsed_ms(
            lambda: AdapterCompatibilityMatrix.get_optimal(HardwareBackend.CPU),
            iterations=1000,
        )
        assert elapsed < 1000, (
            f"1000 × AdapterCompatibilityMatrix.get_optimal took {elapsed:.1f}ms"
        )

    def test_adapter_quant_format_lookup_under_1ms(self):
        from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
        from torchbridge.core.config import HardwareBackend

        elapsed = _elapsed_ms(
            lambda: AdapterCompatibilityMatrix.get_base_quant_format(
                HardwareBackend.CPU
            ),
            iterations=1000,
        )
        assert elapsed < 1000, (
            f"1000 × AdapterCompatibilityMatrix.get_base_quant_format took {elapsed:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Config generation — DistributedConfig.auto() must stay under 10 ms each
# ---------------------------------------------------------------------------


class TestConfigGenerationPerf:
    """DistributedConfig.auto() and hardware detection must stay fast."""

    def test_distributed_config_auto_under_10ms(self):
        from torchbridge.distributed.config import DistributedConfig

        elapsed = _elapsed_ms(
            lambda: DistributedConfig.auto(model_params=1_000_000_000, world_size=8),
            iterations=100,
        )
        assert elapsed < 1000, (
            f"100 × DistributedConfig.auto took {elapsed:.1f}ms "
            f"(> 1 000 ms, i.e., > 10 ms/call). Potential blocking operation."
        )

    def test_benchmark_cache_warm_lookup_under_1ms(self):
        """A warm cache lookup (OrderedDict access) must be sub-millisecond."""
        from torchbridge.attention.dispatch.benchmark_cache import (
            BenchmarkEntry,
            KernelBenchmarkCache,
        )

        cache = KernelBenchmarkCache(cache_dir=None)  # no-disk mode
        # Pre-populate with a valid entry using the actual BenchmarkEntry schema
        fake_entry = BenchmarkEntry(
            kernel_type="pytorch_sdpa",
            latency_ms=1.5,
            throughput_tflops=0.5,
            seq_length=512,
            num_heads=8,
            head_dim=64,
        )
        key = "pytorch_sdpa_512_8_64"
        cache._entries[key] = fake_entry

        # Test that direct OrderedDict lookup is sub-millisecond
        elapsed = _elapsed_ms(lambda: cache._entries.get(key), iterations=1000)
        assert elapsed < 1000, (
            f"1000 × warm KernelBenchmarkCache._entries.get took {elapsed:.1f}ms"
        )
