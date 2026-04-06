"""
Unit tests for tb-validate CLI fixes (v0.5.75).

Covers:
- --format yaml gracefully handles missing pyyaml (instead of silent JSON fallback)
"""

import sys
from unittest.mock import patch

import pytest


def _make_dummy_report():
    """Build a minimal ValidationReport for testing _save_report."""
    from torchbridge.cli.validate import ValidationReport, ValidationResult

    result = ValidationResult(
        name="test_check",
        status="pass",
        message="ok",
        details=None,
        duration_ms=1.0,
    )
    return ValidationReport(
        level="quick",
        results=[result],
        timestamp=0.0,
        duration_ms=1.0,
    )


class TestYamlFormatMissingPyyaml:
    """_save_report must not silently fall back to JSON when pyyaml is absent."""

    def test_missing_pyyaml_prints_install_hint(self, capsys, tmp_path):
        """When yaml is unavailable, must print a message mentioning 'pip install pyyaml'."""
        from torchbridge.cli.validate import ValidateCommand

        report = _make_dummy_report()
        output_file = str(tmp_path / "report.yaml")

        with patch.dict(sys.modules, {"yaml": None}):
            ValidateCommand._save_report(report, output_file, "yaml", verbose=False)

        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        assert "pip install pyyaml" in combined, (
            f"Must print 'pip install pyyaml' when pyyaml is absent.\n"
            f"stdout: {captured.out!r}\nstderr: {captured.err!r}"
        )

    def test_missing_pyyaml_does_not_silently_write_json(self, tmp_path):
        """When yaml is unavailable, must NOT silently write JSON to the output file."""
        import json

        from torchbridge.cli.validate import ValidateCommand

        report = _make_dummy_report()
        output_file = tmp_path / "report.yaml"

        with patch.dict(sys.modules, {"yaml": None}):
            ValidateCommand._save_report(
                report, str(output_file), "yaml", verbose=False
            )

        if output_file.exists():
            content = output_file.read_text()
            # Must NOT be valid JSON (no silent fallback)
            try:
                json.loads(content)
                pytest.fail(
                    "Output file contains JSON — _save_report silently fell back "
                    "to JSON instead of reporting the missing pyyaml dependency."
                )
            except json.JSONDecodeError:
                pass  # Non-JSON content is acceptable (e.g. error message written to file)
        # File not existing is also acceptable (early return on ImportError)

    def test_json_format_still_works(self, tmp_path):
        """--format json must still write valid JSON when yaml is absent."""
        import json

        from torchbridge.cli.validate import ValidateCommand

        report = _make_dummy_report()
        output_file = tmp_path / "report.json"

        ValidateCommand._save_report(report, str(output_file), "json", verbose=False)

        assert output_file.exists(), "JSON output file must be written"
        data = json.loads(output_file.read_text())
        assert "results" in data
