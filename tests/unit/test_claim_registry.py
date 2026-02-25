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
    build_channels_last_benchmark,
    build_claim_suite,
    build_quantization_speedup_benchmark,
    build_tensor_core_alignment_benchmark,
    build_tunableop_benchmark,
    get_all_claim_benchmarks,
)


class TestRegistryFunctions:
    """Tests for registry-level functions."""

    def test_get_all_returns_list(self):
        """get_all_claim_benchmarks should return a list."""
        benchmarks = get_all_claim_benchmarks()
        assert isinstance(benchmarks, list)

    def test_get_all_returns_five(self):
        """Should have exactly 5 registered claims."""
        benchmarks = get_all_claim_benchmarks()
        assert len(benchmarks) == 5

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
        assert len(suite.benchmarks) == 5


class TestTensorCoreAlignmentBenchmark:
    """Tests for tensor core alignment claim."""

    def test_builds_without_error(self):
        """Should build successfully."""
        bench = build_tensor_core_alignment_benchmark()
        assert bench.name == "tensor_core_alignment"

    def test_runs_on_cpu(self):
        """Should run on CPU and produce a result."""
        bench = build_tensor_core_alignment_benchmark()
        result = bench.run(device="cpu")
        assert isinstance(result, ClaimResult)
        assert result.runs > 0
        assert result.baseline_ms > 0
        assert result.optimized_ms > 0


class TestChannelsLastBenchmark:
    """Tests for channels_last layout claim."""

    def test_builds_without_error(self):
        """Should build successfully."""
        bench = build_channels_last_benchmark()
        assert bench.name == "channels_last_layout"

    def test_runs_on_cpu(self):
        """Should run on CPU and produce a result."""
        bench = build_channels_last_benchmark()
        result = bench.run(device="cpu")
        assert isinstance(result, ClaimResult)
        assert result.runs > 0


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


class TestQuantizationSpeedupBenchmark:
    """Tests for INT8 dynamic quantization claim."""

    def test_builds_without_error(self):
        """Should build successfully."""
        bench = build_quantization_speedup_benchmark()
        assert bench.name == "quantization_int8_dynamic"

    def test_runs_on_cpu(self):
        """Should run on CPU and produce a result."""
        bench = build_quantization_speedup_benchmark()
        result = bench.run(device="cpu")
        assert isinstance(result, ClaimResult)
        assert result.runs > 0


class TestTunableOpBenchmark:
    """Tests for AMD TunableOp claim."""

    def test_builds_without_error(self):
        """Should build successfully."""
        bench = build_tunableop_benchmark()
        assert bench.name == "amd_tunableop"

    def test_requires_rocm(self):
        """Should require ROCm backend."""
        bench = build_tunableop_benchmark()
        assert bench.requires_backend == "rocm"

    def test_skipped_on_cpu(self):
        """Should be skipped when run via suite on CPU."""
        suite = BenchmarkSuite()
        suite.add(build_tunableop_benchmark())
        report = suite.run_all(device="cpu")
        assert report.results[0].runs == 0
        assert "SKIPPED" in report.results[0].notes[0]
