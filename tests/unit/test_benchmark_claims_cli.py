"""
Tests for Benchmark Claims CLI Integration

Tests the --type claims argument to the benchmark CLI command.
"""

import argparse
import json

from torchbridge.cli.benchmark import BenchmarkCommand


class TestBenchmarkClaimsCLI:
    """Tests for --type claims CLI integration."""

    def _make_args(self, **overrides):
        """Create args namespace with claim defaults."""
        defaults = {
            "type": "claims",
            "claim": None,
            "claims_threshold": 3.0,
            "ci": False,
            "model": None,
            "predefined": None,
            "quick": False,
            "warmup": 10,
            "runs": 100,
            "output": None,
            "verbose": False,
            "format": "json",
            "compare_baseline": None,
            "regression_threshold": 0.15,
            "levels": "basic,compile",
            "batch_sizes": "1,8,16",
            "input_shape": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_claims_type_in_choices(self):
        """'claims' should be a valid --type choice."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        BenchmarkCommand.register(subparsers)

        # Parse with --type claims — should not raise
        bench_parser = subparsers.choices["benchmark"]
        args = bench_parser.parse_args(["--type", "claims"])
        assert args.type == "claims"

    def test_claims_runs_and_returns_zero(self):
        """--type claims should run and return 0 (some may fail on CPU)."""
        args = self._make_args(quick=True, runs=3, warmup=1)
        # This returns 0 or 1 depending on claim results — just verify it doesn't crash
        result = BenchmarkCommand.execute(args)
        assert result in (0, 1)

    def test_single_claim_by_name(self):
        """--claim <name> should run only that claim."""
        args = self._make_args(claim="quantization_int8_dynamic", quick=True, runs=3, warmup=1)
        result = BenchmarkCommand.execute(args)
        assert result in (0, 1)

    def test_unknown_claim_returns_error(self):
        """Unknown claim name should return error code."""
        args = self._make_args(claim="nonexistent_claim")
        result = BenchmarkCommand.execute(args)
        assert result == 1

    def test_ci_mode_produces_json(self, capsys):
        """--ci should output valid JSON."""
        args = self._make_args(ci=True, quick=True, runs=3, warmup=1)
        BenchmarkCommand.execute(args)
        captured = capsys.readouterr()
        # The execute() prints a header before dispatching to claims;
        # extract the JSON portion (starts with '{')
        output = captured.out
        json_start = output.index("{")
        json_str = output[json_start:]
        parsed = json.loads(json_str)
        assert "results" in parsed
        assert "device" in parsed
