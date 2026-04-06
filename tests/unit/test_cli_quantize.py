"""
Unit tests for tb-quantize CLI fixes (v0.5.75).

Covers:
- Removal of dead --strategy option
- Removal of dead --calibration-samples option
- _validate_quality() now reports latency metrics in CI mode
"""

import argparse
import json

import pytest
import torch
import torch.nn as nn


def _build_parser() -> argparse.ArgumentParser:
    """Build a fresh quantize argument parser via QuantizeCommand.register."""
    from torchbridge.cli.quantize import QuantizeCommand

    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd")
    QuantizeCommand.register(sub)
    return root


class TestDeadOptionsRemoved:
    def test_strategy_option_removed(self):
        """--strategy must no longer be accepted by the quantize parser."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["quantize", "--model", "m.pt", "--strategy", "auto"])

    def test_calibration_samples_option_removed(self):
        """--calibration-samples must no longer be accepted by the quantize parser."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["quantize", "--model", "m.pt", "--calibration-samples", "128"]
            )

    def test_valid_options_still_accepted(self):
        """Core options --model, --format, --backend, --validate, --ci must still work."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "quantize",
                "--model",
                "m.pt",
                "--format",
                "int8_dynamic",
                "--backend",
                "cpu",
                "--validate",
                "--ci",
            ]
        )
        assert args.model == "m.pt"
        assert args.format == "int8_dynamic"
        assert args.validate is True
        assert args.ci is True


class TestValidateQualityLatency:
    """_validate_quality() must include latency metrics in its CI JSON output."""

    def _make_models(self) -> tuple[nn.Module, nn.Module]:
        model = nn.Linear(16, 8)
        model.eval()
        quantized = nn.Linear(16, 8)
        quantized.eval()
        # Copy weights so outputs match (cosine_sim will be high)
        with torch.no_grad():
            quantized.weight.copy_(model.weight)
            quantized.bias.copy_(model.bias)
        return model, quantized

    def test_validate_quality_emits_latency_keys_ci(self, capsys):
        """In CI mode, _validate_quality must emit original_latency_ms,
        quantized_latency_ms, and speedup_ratio in the JSON output."""
        from torchbridge.cli.quantize import QuantizeCommand

        original, quantized = self._make_models()
        QuantizeCommand._validate_quality(original, quantized, ci_mode=True)

        captured = capsys.readouterr().out
        # May emit multiple JSON blobs; find the validation one
        for line in captured.strip().splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "validation" in data:
                v = data["validation"]
                assert "original_latency_ms" in v, "missing original_latency_ms"
                assert "quantized_latency_ms" in v, "missing quantized_latency_ms"
                assert "speedup_ratio" in v, "missing speedup_ratio"
                assert isinstance(v["original_latency_ms"], float)
                assert isinstance(v["quantized_latency_ms"], float)
                assert isinstance(v["speedup_ratio"], float)
                return
        pytest.fail("No JSON blob with 'validation' key found in output")

    def test_validate_quality_emits_latency_keys_human(self, capsys):
        """In human-readable mode, _validate_quality must print latency lines."""
        from torchbridge.cli.quantize import QuantizeCommand

        original, quantized = self._make_models()
        QuantizeCommand._validate_quality(original, quantized, ci_mode=False)

        out = capsys.readouterr().out
        assert "latency" in out.lower() or "speedup" in out.lower(), (
            "Human output must mention latency or speedup"
        )

    def test_speedup_ratio_is_positive(self, capsys):
        """speedup_ratio must be a positive finite float."""
        from torchbridge.cli.quantize import QuantizeCommand

        original, quantized = self._make_models()
        QuantizeCommand._validate_quality(original, quantized, ci_mode=True)

        captured = capsys.readouterr().out
        for line in captured.strip().splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "validation" in data:
                ratio = data["validation"]["speedup_ratio"]
                assert ratio > 0, f"speedup_ratio must be positive, got {ratio}"
                return
        pytest.fail("No validation JSON found")
