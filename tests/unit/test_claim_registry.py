"""
Tests for Claim Registry

Tests that each registered claim benchmark builds and runs without error,
and that the registry functions work correctly.
"""

from torchbridge.benchmarks.claim_benchmarks import (
    BenchmarkSuite,
    ClaimBenchmark,
    ClaimResult,
)
from torchbridge.benchmarks.claim_registry import (
    build_attention_dispatch_benchmark,
    build_claim_suite,
    build_quantization_speedup_benchmark,
    get_all_claim_benchmarks,
)


class TestPublicAPIExports:
    """Tests that key functions are accessible from torchbridge.benchmarks package."""

    def test_build_claim_suite_importable_from_package(self):
        """build_claim_suite should be importable from torchbridge.benchmarks."""
        from torchbridge.benchmarks import build_claim_suite as _bcs

        assert callable(_bcs)

    def test_get_all_claim_benchmarks_importable_from_package(self):
        """get_all_claim_benchmarks should be importable from torchbridge.benchmarks."""
        from torchbridge.benchmarks import get_all_claim_benchmarks as _gacb

        assert callable(_gacb)


class TestRegistryFunctions:
    """Tests for registry-level functions."""

    def test_get_all_returns_list(self):
        """get_all_claim_benchmarks should return a list."""
        benchmarks = get_all_claim_benchmarks()
        assert isinstance(benchmarks, list)

    def test_get_all_returns_two(self):
        """Should have exactly 2 registered claims (attention_dispatch + quantization)."""
        benchmarks = get_all_claim_benchmarks()
        assert len(benchmarks) == 2

    def test_all_are_claim_benchmarks(self):
        """All returned items should be ClaimBenchmark instances."""
        benchmarks = get_all_claim_benchmarks()
        for b in benchmarks:
            assert isinstance(b, ClaimBenchmark)

    def test_unique_names(self):
        """All claim names should be unique."""
        benchmarks = get_all_claim_benchmarks()
        names = [b.name for b in benchmarks]
        assert len(names) == len(set(names))

    def test_build_claim_suite_returns_suite(self):
        """build_claim_suite should return a BenchmarkSuite."""
        suite = build_claim_suite()
        assert isinstance(suite, BenchmarkSuite)
        assert len(suite._benchmarks) == 2

    def test_registered_claim_names(self):
        """Registry must contain exactly attention_dispatch and quantization claims."""
        names = {b.name for b in get_all_claim_benchmarks()}
        assert names == {"attention_dispatch_overhead", "quantization_int8_dynamic"}


class TestAttentionDispatchBenchmark:
    """Tests for attention dispatch overhead claim."""

    def test_builds_without_error(self):
        """Should build successfully."""
        bench = build_attention_dispatch_benchmark()
        assert bench.name == "attention_dispatch_overhead"

    def test_negative_threshold(self):
        """Should use negative threshold (allows overhead)."""
        bench = build_attention_dispatch_benchmark()
        assert bench._threshold_pct < 0

    def test_runs_on_cpu(self):
        """Should run on CPU and produce a result."""
        bench = build_attention_dispatch_benchmark()
        result = bench.run(device="cpu")
        assert isinstance(result, ClaimResult)
        assert result.runs > 0

    def test_tensors_not_created_inside_timed_functions(self):
        """baseline and optimized functions must not create tensors internally."""
        import inspect

        bench = build_attention_dispatch_benchmark()
        assert "torch.randn" not in inspect.getsource(bench._baseline_fn)
        assert "torch.randn" not in inspect.getsource(bench._optimized_fn)


class TestQuantizationSpeedupBenchmark:
    """Tests for INT8 dynamic quantization claim."""

    def test_builds_without_error(self):
        """Should build successfully."""
        bench = build_quantization_speedup_benchmark()
        assert bench.name == "quantization_int8_dynamic"

    def test_runs_or_skips_on_cpu(self):
        """Should run when FBGEMM is available, or skip with a reason when it isn't."""
        bench = build_quantization_speedup_benchmark()
        if bench.skip_reason is not None:
            # FBGEMM not available (e.g. macOS) — skip_reason should be set
            assert "FBGEMM" in bench.skip_reason
            result = bench.run(device="cpu")
            assert result.runs == 0
            assert "SKIPPED" in result.notes[0]
        else:
            # FBGEMM available (Linux x86_64) — should run and produce a result
            result = bench.run(device="cpu")
            assert isinstance(result, ClaimResult)
            assert result.runs > 0
