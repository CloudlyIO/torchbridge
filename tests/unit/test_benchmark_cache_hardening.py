"""
KernelBenchmarkCache hardening tests.

Verifies:
- max_entries bound is respected (LRU eviction)
- Oldest entry is evicted first
- _save() survives json.dump failure without corrupting existing file
- Concurrent run_benchmark calls do not crash
- Large configs remain bounded

All tests FAIL until max_entries, LRU eviction, atomic writes, and
threading.Lock are added to benchmark_cache.py.
"""

import json
import threading
from unittest.mock import patch

import pytest


def _make_cache(tmpdir: str, max_entries: int = 4):
    """Helper: create a KernelBenchmarkCache with given max_entries."""
    from torchbridge.attention.dispatch.benchmark_cache import KernelBenchmarkCache

    return KernelBenchmarkCache(cache_dir=tmpdir, max_entries=max_entries)


class TestBenchmarkCacheHardening:
    """KernelBenchmarkCache must be bounded, atomic, and thread-safe."""

    def test_max_entries_respected(self, tmp_path):
        """Adding max_entries+1 entries must keep len(_entries) == max_entries."""
        from torchbridge.attention.dispatch.benchmark_cache import BenchmarkEntry

        cache = _make_cache(str(tmp_path), max_entries=4)

        # Directly inject entries to avoid actual GPU benchmarking
        for i in range(5):
            key = f"pytorch_sdpa_{64 + i}_4_32"
            cache._entries[key] = BenchmarkEntry(
                kernel_type="pytorch_sdpa",
                latency_ms=float(i),
                throughput_tflops=0.0,
                seq_length=64 + i,
                num_heads=4,
                head_dim=32,
            )
            # Simulate the eviction that run_benchmark should do
            cache._evict_if_needed()

        assert len(cache._entries) <= 4, (
            f"Cache has {len(cache._entries)} entries but max_entries=4"
        )

    def test_lru_eviction_removes_oldest(self, tmp_path):
        """Fill to max, add one more — the first key inserted must be gone."""
        from torchbridge.attention.dispatch.benchmark_cache import BenchmarkEntry

        max_entries = 3
        cache = _make_cache(str(tmp_path), max_entries=max_entries)

        first_key = "pytorch_sdpa_64_4_32"
        for i in range(max_entries):
            key = f"pytorch_sdpa_{64 + i}_4_32"
            cache._entries[key] = BenchmarkEntry(
                kernel_type="pytorch_sdpa",
                latency_ms=float(i),
                throughput_tflops=0.0,
                seq_length=64 + i,
                num_heads=4,
                head_dim=32,
            )

        # first_key is the oldest; now add one more
        cache._entries["pytorch_sdpa_200_4_32"] = BenchmarkEntry(
            kernel_type="pytorch_sdpa",
            latency_ms=99.0,
            throughput_tflops=0.0,
            seq_length=200,
            num_heads=4,
            head_dim=32,
        )
        cache._evict_if_needed()

        assert first_key not in cache._entries, (
            f"Oldest key '{first_key}' should have been evicted but is still present"
        )
        assert len(cache._entries) <= max_entries

    def test_save_is_atomic(self, tmp_path):
        """If json.dump raises mid-write, the existing cache file must survive."""
        from torchbridge.attention.dispatch.benchmark_cache import (
            BenchmarkEntry,
            KernelBenchmarkCache,
        )

        cache = KernelBenchmarkCache(cache_dir=str(tmp_path))
        # Write a valid initial state
        cache._entries["existing_key"] = BenchmarkEntry(
            kernel_type="pytorch_sdpa",
            latency_ms=1.0,
            throughput_tflops=0.0,
            seq_length=64,
            num_heads=4,
            head_dim=32,
        )
        cache._save()  # should succeed

        original_content = json.loads(open(cache._cache_path).read())
        assert "entries" in original_content

        # Now simulate a mid-write failure
        with patch("json.dump", side_effect=OSError("disk full")):
            try:
                cache._save()
            except Exception:
                pass

        # The file must still be readable and contain the original valid content
        try:
            recovered = json.loads(open(cache._cache_path).read())
        except json.JSONDecodeError:
            pytest.fail(
                "Cache file was corrupted by a failed _save() — atomic write required"
            )

        assert "entries" in recovered, (
            "Cache file lost its 'entries' key after failed save"
        )

    def test_concurrent_run_benchmark_no_crash(self, tmp_path):
        """5 threads calling run_benchmark simultaneously must not raise."""
        from torchbridge.attention.dispatch.benchmark_cache import KernelBenchmarkCache
        from torchbridge.attention.dispatch.kernel_types import AttentionKernelType

        cache = KernelBenchmarkCache(cache_dir=str(tmp_path))
        errors: list[Exception] = []

        def worker() -> None:
            try:
                cache.run_benchmark(
                    AttentionKernelType.PYTORCH_SDPA,
                    seq_length=32,
                    num_heads=2,
                    head_dim=8,
                    warmup=1,
                    iterations=2,
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Exceptions during concurrent run_benchmark: {errors}"

    def test_large_cache_still_bounded(self, tmp_path):
        """Warming 600 different configs must not exceed max_entries."""
        from torchbridge.attention.dispatch.benchmark_cache import BenchmarkEntry

        max_entries = 512
        cache = _make_cache(str(tmp_path), max_entries=max_entries)

        for i in range(600):
            key = f"pytorch_sdpa_{64 + i}_4_32"
            cache._entries[key] = BenchmarkEntry(
                kernel_type="pytorch_sdpa",
                latency_ms=float(i),
                throughput_tflops=0.0,
                seq_length=64 + i,
                num_heads=4,
                head_dim=32,
            )
            cache._evict_if_needed()

        assert len(cache._entries) <= max_entries, (
            f"Cache has {len(cache._entries)} entries, exceeds max_entries={max_entries}"
        )
