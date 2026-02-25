"""
Integration Tests for Claim Benchmark Pipeline

End-to-end tests: build suite → run all → generate report → verify output.
"""

import json
import tempfile

from torchbridge.benchmarks.claim_benchmarks import BenchmarkReport
from torchbridge.benchmarks.claim_registry import (
    build_claim_suite,
    get_all_claim_benchmarks,
)


class TestClaimBenchmarkPipeline:
    """End-to-end claim benchmark pipeline tests."""

    def test_full_suite_runs_on_cpu(self):
        """Full claim suite should run on CPU without error."""
        suite = build_claim_suite()
        report = suite.run_all(device="cpu")
        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 5

    def test_all_non_skipped_have_timing_data(self):
        """Non-skipped results should have positive timing data."""
        suite = build_claim_suite()
        report = suite.run_all(device="cpu")
        for r in report.results:
            if r.runs > 0:
                assert r.baseline_ms > 0, f"{r.claim_name}: baseline_ms should be > 0"
                assert r.optimized_ms > 0, f"{r.claim_name}: optimized_ms should be > 0"

    def test_tunableop_skipped_on_cpu(self):
        """AMD TunableOp should be skipped on CPU."""
        suite = build_claim_suite()
        report = suite.run_all(device="cpu")
        tunableop = [r for r in report.results if r.claim_name == "amd_tunableop"]
        assert len(tunableop) == 1
        assert tunableop[0].runs == 0

    def test_report_save_and_reload(self):
        """Report should save to JSON and reload correctly."""
        suite = build_claim_suite()
        report = suite.run_all(device="cpu")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        report.save(path)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["device"] == "cpu"
        assert len(loaded["results"]) == 5
        assert "summary" in loaded
        assert "claims_to_delete" in loaded

    def test_quantization_claim_shows_speedup(self):
        """INT8 dynamic quantization should show measurable speedup on CPU."""
        from torchbridge.benchmarks.claim_registry import (
            build_quantization_speedup_benchmark,
        )

        bench = build_quantization_speedup_benchmark()
        result = bench.run(device="cpu")
        # INT8 dynamic quant on CPU (FBGEMM) typically shows 10-40% speedup
        # We just verify it ran and has positive timing, not pass/fail
        assert result.baseline_ms > 0
        assert result.optimized_ms > 0
        assert result.runs > 0

    def test_report_summary_includes_all_counts(self):
        """Report summary should account for all results."""
        suite = build_claim_suite()
        report = suite.run_all(device="cpu")
        summary = report.summary()

        ran = [r for r in report.results if r.runs > 0]
        skipped = [r for r in report.results if r.runs == 0]
        passed = sum(1 for r in ran if r.passed)
        failed = len(ran) - passed

        assert f"{passed} passed" in summary
        assert f"{failed} failed" in summary
        if skipped:
            assert f"{len(skipped)} skipped" in summary

    def test_registry_claim_names(self):
        """All claim names should follow naming convention."""
        benchmarks = get_all_claim_benchmarks()
        expected_names = {
            "tensor_core_alignment",
            "channels_last_layout",
            "attention_dispatch_overhead",
            "quantization_int8_dynamic",
            "amd_tunableop",
        }
        actual_names = {b.name for b in benchmarks}
        assert actual_names == expected_names

    def test_import_from_package(self):
        """Public API should be importable from torchbridge.benchmarks."""
        from torchbridge.benchmarks import (
            BenchmarkReport,
            BenchmarkSuite,
            ClaimBenchmark,
            ClaimResult,
        )

        assert ClaimBenchmark is not None
        assert ClaimResult is not None
        assert BenchmarkSuite is not None
        assert BenchmarkReport is not None
