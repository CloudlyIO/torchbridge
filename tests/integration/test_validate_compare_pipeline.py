"""
Integration tests for tb-validate --compare end-to-end pipeline.

Tests the full execute() → _run_compare() path, forced-fail scenarios,
and JSON report round-trip. All tests use CPU only (no GPU required).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from torchbridge.cli.validate import ValidateCommand


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        "compare": ["cpu", "cpu"],
        "model": None,
        "input_shape": "1,32",
        "per_layer": False,
        "dtype": "float32",
        "output": None,
        "ci": False,
        "verbose": False,
        "level": "standard",
        "quantized": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestFullCpuPipeline:
    """End-to-end: execute() routes to _run_compare() and produces correct result."""

    def test_cpu_cpu_execute_returns_zero(self):
        """execute() with --compare cpu cpu must return 0."""
        args = _make_args(compare=["cpu", "cpu"])
        assert ValidateCommand.execute(args) == 0

    def test_cpu_cpu_ci_json_via_execute(self, capsys):
        """execute() with --compare + --ci must produce valid JSON with passed=True."""
        args = _make_args(compare=["cpu", "cpu"], ci=True)
        rc = ValidateCommand.execute(args)
        assert rc == 0
        out = json.loads(capsys.readouterr().out)
        assert out["passed"] is True
        assert out["max_diff"] == 0.0

    def test_execute_without_compare_uses_standard_pipeline(self, capsys):
        """execute() without --compare must NOT produce compare JSON keys."""
        args = _make_args(compare=None, level="quick", ci=True)
        ValidateCommand.execute(args)
        out_text = capsys.readouterr().out
        if out_text.strip():
            data = json.loads(out_text)
            # Standard pipeline JSON has 'results' key, not 'max_diff'
            assert "max_diff" not in data


class TestForcedFailScenario:
    """Verify that divergent outputs produce exit code 1."""

    def test_forced_output_difference_returns_1(self):
        """If both backends return different tensors, comparison must FAIL."""
        call_count = [0]

        class FakeModel(nn.Module):
            def forward(self, x):
                call_count[0] += 1
                # First call returns zeros, second returns ones → big diff
                if call_count[0] == 1:
                    return torch.zeros_like(x)
                return torch.ones_like(x)

        fake_model = FakeModel()

        def fake_load(*a, **kw):
            return fake_model

        with patch("torch.load", fake_load):
            # Use a local path that "exists" on disk — patch Path.exists
            with patch.object(Path, "exists", return_value=True):
                args = _make_args(
                    compare=["cpu", "cpu"],
                    model="/fake/model.pt",
                    input_shape="1,8",
                )
                call_count[0] = 0
                result = ValidateCommand._run_compare(args)

        assert result == 1

    def test_forced_fail_ci_json_has_passed_false(self, capsys):
        """Forced-fail path must output passed=false in CI JSON."""
        call_count = [0]

        class FakeModel(nn.Module):
            def forward(self, x):
                call_count[0] += 1
                return (
                    torch.zeros_like(x)
                    if call_count[0] == 1
                    else torch.ones_like(x) * 1e6
                )

        fake_model = FakeModel()

        with patch("torch.load", lambda *a, **kw: fake_model):
            with patch.object(Path, "exists", return_value=True):
                args = _make_args(
                    compare=["cpu", "cpu"],
                    model="/fake/model.pt",
                    input_shape="1,8",
                    ci=True,
                )
                call_count[0] = 0
                ValidateCommand._run_compare(args)

        out = json.loads(capsys.readouterr().out)
        assert out["passed"] is False


class TestCudaUnavailable:
    """When CUDA is unavailable, --compare cuda cpu must exit 1 with clear message."""

    def test_cuda_unavailable_returns_1_via_execute(self):
        with patch("torch.cuda.is_available", return_value=False):
            args = _make_args(compare=["cuda", "cpu"])
            result = ValidateCommand.execute(args)
        assert result == 1

    def test_cuda_unavailable_ci_error_key(self, capsys):
        with patch("torch.cuda.is_available", return_value=False):
            args = _make_args(compare=["cuda", "cpu"], ci=True)
            ValidateCommand.execute(args)
        out = json.loads(capsys.readouterr().out)
        assert "error" in out


class TestReportRoundTrip:
    """--output saves a JSON file that can be reloaded and verified."""

    def test_report_save_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "report.json")
            args = _make_args(
                compare=["cpu", "cpu"], output=out_path, input_shape="1,16"
            )
            rc = ValidateCommand.execute(args)
            assert rc == 0
            assert Path(out_path).exists()
            with open(out_path) as f:
                data = json.load(f)
            assert data["passed"] is True
            assert data["backend1"] == "cpu"
            assert data["backend2"] == "cpu"
            assert data["input_shape"] == [1, 16]
            assert data["max_diff"] == 0.0

    def test_report_duration_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "report.json")
            args = _make_args(compare=["cpu", "cpu"], output=out_path)
            ValidateCommand.execute(args)
            with open(out_path) as f:
                data = json.load(f)
            assert data["duration_ms"] > 0


class TestPerLayerFlag:
    """--per-layer populates per_layer list (may be empty without HF model)."""

    def test_per_layer_key_present_in_output(self, capsys):
        """per_layer key must always be in CI JSON, even if empty."""
        args = _make_args(compare=["cpu", "cpu"], per_layer=True, ci=True)
        ValidateCommand._run_compare(args)
        out = json.loads(capsys.readouterr().out)
        assert "per_layer" in out
        assert isinstance(out["per_layer"], list)
