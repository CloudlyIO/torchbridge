"""
Tests for Claim-Level Benchmark Harness

Tests ClaimBenchmark, ClaimResult, BenchmarkSuite, BenchmarkReport,
and the _time_fn timing utility.
"""

import json
import tempfile

from torchbridge.benchmarks.claim_benchmarks import (
    BenchmarkReport,
    BenchmarkSuite,
    ClaimBenchmark,
    ClaimResult,
    _time_fn,
)


class TestClaimResult:
    """Tests for ClaimResult dataclass."""

    def test_basic_fields(self):
        """All fields should be accessible."""
        result = ClaimResult(
            claim_name="test_claim",
            baseline_ms=10.0,
            optimized_ms=8.0,
            speedup_pct=20.0,
            passed=True,
            threshold_pct=3.0,
            runs=50,
            std_baseline_ms=0.5,
            std_optimized_ms=0.3,
            device="cpu",
        )
        assert result.claim_name == "test_claim"
        assert result.passed is True
        assert result.speedup_pct == 20.0

    def test_to_dict_has_all_keys(self):
        """to_dict should include all fields."""
        result = ClaimResult(
            claim_name="test",
            baseline_ms=1.0,
            optimized_ms=0.9,
            speedup_pct=10.0,
            passed=True,
            threshold_pct=3.0,
            runs=10,
            std_baseline_ms=0.1,
            std_optimized_ms=0.1,
            device="cpu",
            notes=["note1"],
        )
        d = result.to_dict()
        assert "claim_name" in d
        assert "speedup_pct" in d
        assert "notes" in d
        assert d["notes"] == ["note1"]

    def test_to_dict_rounds_values(self):
        """to_dict should round float values."""
        result = ClaimResult(
            claim_name="test",
            baseline_ms=1.23456789,
            optimized_ms=0.98765432,
            speedup_pct=19.999,
            passed=True,
            threshold_pct=3.0,
            runs=10,
            std_baseline_ms=0.123456,
            std_optimized_ms=0.098765,
            device="cpu",
        )
        d = result.to_dict()
        assert d["baseline_ms"] == 1.2346
        assert d["speedup_pct"] == 20.0

    def test_default_notes_empty(self):
        """Notes should default to empty list."""
        result = ClaimResult(
            claim_name="t", baseline_ms=1.0, optimized_ms=1.0,
            speedup_pct=0.0, passed=False, threshold_pct=3.0,
            runs=1, std_baseline_ms=0.0, std_optimized_ms=0.0,
            device="cpu",
        )
        assert result.notes == []


class TestTimeFn:
    """Tests for _time_fn utility."""

    def test_returns_positive_mean(self):
        """Mean time should be positive."""
        mean, std = _time_fn(lambda: None, warmup=1, runs=5)
        assert mean > 0.0

    def test_returns_non_negative_std(self):
        """Standard deviation should be non-negative."""
        mean, std = _time_fn(lambda: None, warmup=1, runs=5)
        assert std >= 0.0

    def test_warmup_runs_before_timing(self):
        """Warmup calls should not be included in timing."""
        call_count = 0

        def counter():
            nonlocal call_count
            call_count += 1

        _time_fn(counter, warmup=3, runs=5)
        assert call_count == 8  # 3 warmup + 5 runs


class TestClaimBenchmark:
    """Tests for ClaimBenchmark."""

    def test_run_returns_claim_result(self):
        """run() should return a ClaimResult."""
        bench = ClaimBenchmark(
            name="test",
            baseline_fn=lambda: None,
            optimized_fn=lambda: None,
            warmup=1,
            runs=3,
        )
        result = bench.run()
        assert isinstance(result, ClaimResult)
        assert result.claim_name == "test"

    def test_speedup_positive_when_optimized_faster(self):
        """Speedup should be positive when optimized is faster."""
        import time

        bench = ClaimBenchmark(
            name="fast_opt",
            baseline_fn=lambda: time.sleep(0.002),
            optimized_fn=lambda: None,
            warmup=1,
            runs=3,
        )
        result = bench.run()
        assert result.speedup_pct > 0

    def test_pass_when_above_threshold(self):
        """Should pass when speedup exceeds threshold."""
        import time

        bench = ClaimBenchmark(
            name="pass_test",
            baseline_fn=lambda: time.sleep(0.005),
            optimized_fn=lambda: None,
            warmup=1,
            runs=3,
            threshold_pct=1.0,
        )
        result = bench.run()
        assert result.passed is True

    def test_fail_when_below_threshold(self):
        """Should fail when speedup is below threshold."""
        bench = ClaimBenchmark(
            name="fail_test",
            baseline_fn=lambda: None,
            optimized_fn=lambda: None,
            warmup=1,
            runs=3,
            threshold_pct=50.0,
        )
        result = bench.run()
        assert result.passed is False

    def test_negative_threshold_for_overhead(self):
        """Negative threshold should allow small overhead."""
        bench = ClaimBenchmark(
            name="overhead_test",
            baseline_fn=lambda: None,
            optimized_fn=lambda: None,
            warmup=1,
            runs=3,
            threshold_pct=-5.0,
        )
        result = bench.run()
        # Both are near-zero, speedup ≈ 0%, which is > -5%
        assert result.passed is True

    def test_notes_preserved(self):
        """Notes should be passed through to result."""
        bench = ClaimBenchmark(
            name="notes_test",
            baseline_fn=lambda: None,
            optimized_fn=lambda: None,
            warmup=1,
            runs=3,
            notes=["note A", "note B"],
        )
        result = bench.run()
        assert result.notes == ["note A", "note B"]

    def test_requires_backend_attribute(self):
        """requires_backend should be stored."""
        bench = ClaimBenchmark(
            name="rocm_test",
            baseline_fn=lambda: None,
            optimized_fn=lambda: None,
            requires_backend="rocm",
        )
        assert bench.requires_backend == "rocm"

    def test_device_passed_through(self):
        """device parameter should appear in result."""
        bench = ClaimBenchmark(
            name="dev_test",
            baseline_fn=lambda: None,
            optimized_fn=lambda: None,
            warmup=1,
            runs=3,
        )
        result = bench.run(device="cuda:0")
        assert result.device == "cuda:0"


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite."""

    def test_add_and_list(self):
        """Should store added benchmarks."""
        suite = BenchmarkSuite()
        b1 = ClaimBenchmark(name="a", baseline_fn=lambda: None, optimized_fn=lambda: None)
        b2 = ClaimBenchmark(name="b", baseline_fn=lambda: None, optimized_fn=lambda: None)
        suite.add(b1)
        suite.add(b2)
        assert len(suite.benchmarks) == 2

    def test_run_all_returns_report(self):
        """run_all should return a BenchmarkReport."""
        suite = BenchmarkSuite()
        suite.add(ClaimBenchmark(
            name="fast", baseline_fn=lambda: None, optimized_fn=lambda: None,
            warmup=1, runs=2,
        ))
        report = suite.run_all()
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 1

    def test_skip_requires_backend(self):
        """Benchmarks requiring a different backend should be skipped."""
        suite = BenchmarkSuite()
        suite.add(ClaimBenchmark(
            name="gpu_only", baseline_fn=lambda: None, optimized_fn=lambda: None,
            warmup=1, runs=2, requires_backend="rocm",
        ))
        report = suite.run_all(device="cpu")
        assert report.results[0].runs == 0
        assert "SKIPPED" in report.results[0].notes[0]

    def test_matching_backend_runs(self):
        """Benchmarks matching the device backend should run."""
        suite = BenchmarkSuite()
        suite.add(ClaimBenchmark(
            name="cpu_ok", baseline_fn=lambda: None, optimized_fn=lambda: None,
            warmup=1, runs=2, requires_backend="cpu",
        ))
        report = suite.run_all(device="cpu")
        assert report.results[0].runs == 2


class TestBenchmarkReport:
    """Tests for BenchmarkReport."""

    def _make_report(self):
        """Create a sample report with mixed results."""
        results = [
            ClaimResult(
                claim_name="passing", baseline_ms=10.0, optimized_ms=8.0,
                speedup_pct=20.0, passed=True, threshold_pct=3.0, runs=50,
                std_baseline_ms=0.5, std_optimized_ms=0.3, device="cpu",
            ),
            ClaimResult(
                claim_name="failing", baseline_ms=10.0, optimized_ms=10.5,
                speedup_pct=-5.0, passed=False, threshold_pct=3.0, runs=50,
                std_baseline_ms=0.5, std_optimized_ms=0.3, device="cpu",
            ),
            ClaimResult(
                claim_name="skipped", baseline_ms=0.0, optimized_ms=0.0,
                speedup_pct=0.0, passed=False, threshold_pct=3.0, runs=0,
                std_baseline_ms=0.0, std_optimized_ms=0.0, device="cpu",
                notes=["SKIPPED: requires rocm, running on cpu"],
            ),
        ]
        return BenchmarkReport(results=results, device="cpu")

    def test_summary_format(self):
        """Summary should show pass/fail/skip counts."""
        report = self._make_report()
        summary = report.summary()
        assert "1 passed" in summary
        assert "1 failed" in summary
        assert "1 skipped" in summary

    def test_claims_to_delete(self):
        """claims_to_delete should return failed (non-skipped) claims."""
        report = self._make_report()
        to_delete = report.claims_to_delete()
        assert to_delete == ["failing"]

    def test_to_dict_structure(self):
        """to_dict should have expected keys."""
        report = self._make_report()
        d = report.to_dict()
        assert "device" in d
        assert "timestamp" in d
        assert "summary" in d
        assert "results" in d
        assert "claims_to_delete" in d
        assert len(d["results"]) == 3

    def test_to_json_valid(self):
        """to_json should produce valid JSON."""
        report = self._make_report()
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["device"] == "cpu"

    def test_save_to_file(self):
        """save should write JSON to file."""
        report = self._make_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        report.save(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["device"] == "cpu"
        assert len(loaded["results"]) == 3
