"""
Tests for the validate CLI command.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from torchbridge.cli.validate import ValidateCommand, ValidationReport, ValidationResult


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test creating a ValidationResult."""
        result = ValidationResult(
            name="Test Check",
            status="pass",
            message="OK",
            details="Some details",
            duration_ms=10.5,
        )
        assert result.name == "Test Check"
        assert result.status == "pass"
        assert result.message == "OK"
        assert result.details == "Some details"
        assert result.duration_ms == 10.5

    def test_validation_result_defaults(self):
        """Test ValidationResult default values."""
        result = ValidationResult(name="Test", status="pass", message="OK")
        assert result.details is None
        assert result.duration_ms == 0.0


class TestValidationReport:
    """Test ValidationReport dataclass."""

    def test_report_properties(self):
        """Test report computed properties."""
        report = ValidationReport(
            level="standard",
            results=[
                ValidationResult("A", "pass", "ok"),
                ValidationResult("B", "warning", "warn"),
                ValidationResult("C", "fail", "bad"),
                ValidationResult("D", "pass", "ok"),
            ],
        )
        assert report.passed == 2
        assert report.warnings == 1
        assert report.failures == 1
        assert report.has_failures is True
        assert report.has_warnings is True

    def test_report_all_pass(self):
        """Test report with all passing."""
        report = ValidationReport(
            level="quick",
            results=[
                ValidationResult("A", "pass", "ok"),
                ValidationResult("B", "pass", "ok"),
            ],
        )
        assert report.passed == 2
        assert report.has_failures is False
        assert report.has_warnings is False


class TestValidateCommand:
    """Test the validate CLI command."""

    def test_run_quick_checks(self):
        """Test quick checks return results."""
        results = ValidateCommand._run_quick_checks(verbose=False)
        assert len(results) > 0
        # Should include TorchBridge import check
        assert any("TorchBridge" in r.name for r in results)

    def test_run_standard_checks_no_model(self):
        """Test standard checks without a model."""
        results = ValidateCommand._run_standard_checks(None, verbose=False)
        assert len(results) > 0
        # Should include UnifiedValidator check
        assert any("UnifiedValidator" in r.name for r in results)
        # Should include export format checks
        assert any("Export" in r.name for r in results)

    def test_run_standard_checks_missing_model(self):
        """Test standard checks with a missing model path."""
        results = ValidateCommand._run_standard_checks(
            "/nonexistent/model.pt", verbose=False
        )
        model_result = next((r for r in results if r.name == "Model Load"), None)
        assert model_result is not None
        assert model_result.status == "fail"

    def test_run_full_checks(self):
        """Test full checks return results."""
        results = ValidateCommand._run_full_checks(verbose=False)
        assert len(results) > 0
        assert any("Benchmark" in r.name for r in results)
        assert any("Consistency" in r.name for r in results)

    def test_run_cloud_checks_no_script(self):
        """Test cloud checks when script is missing."""
        with patch('pathlib.Path.exists', return_value=False):
            results = ValidateCommand._run_cloud_checks(verbose=False)
        assert len(results) >= 1
        assert results[0].status == "warning"
        assert "not found" in results[0].message

    def test_execute_quick_level(self):
        """Test executing quick validation."""
        args = MagicMock()
        args.level = 'quick'
        args.model = None
        args.output = None
        args.format = 'text'
        args.ci = False
        args.verbose = False

        result = ValidateCommand.execute(args)
        assert result in [0, 1, 2]

    def test_execute_standard_level(self):
        """Test executing standard validation."""
        args = MagicMock()
        args.level = 'standard'
        args.model = None
        args.output = None
        args.format = 'text'
        args.ci = False
        args.verbose = False

        result = ValidateCommand.execute(args)
        assert result in [0, 1, 2]

    def test_execute_full_level(self):
        """Test executing full validation."""
        args = MagicMock()
        args.level = 'full'
        args.model = None
        args.output = None
        args.format = 'text'
        args.ci = False
        args.verbose = False

        result = ValidateCommand.execute(args)
        assert result in [0, 1, 2]

    def test_execute_ci_mode(self, capsys):
        """Test execute in CI mode outputs JSON."""
        args = MagicMock()
        args.level = 'quick'
        args.model = None
        args.output = None
        args.format = 'json'
        args.ci = True
        args.verbose = False

        result = ValidateCommand.execute(args)
        assert result in [0, 1, 2]

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert 'results' in data
        assert 'summary' in data
        assert data['level'] == 'quick'

    def test_ci_mode_suppresses_print(self, capsys):
        """Test CI mode suppresses normal output."""
        args = MagicMock()
        args.level = 'quick'
        args.model = None
        args.output = None
        args.format = 'json'
        args.ci = True
        args.verbose = False

        ValidateCommand.execute(args)

        captured = capsys.readouterr()
        assert "TorchBridge Validation Pipeline" not in captured.out

    def test_output_ci_json_all_pass(self, capsys):
        """Test CI JSON output for all pass."""
        report = ValidationReport(
            level="quick",
            results=[
                ValidationResult("A", "pass", "ok"),
                ValidationResult("B", "pass", "ok"),
            ],
            timestamp=1000.0,
            duration_ms=50.0,
        )
        exit_code = ValidateCommand._output_ci_json(report)
        assert exit_code == 0

    def test_output_ci_json_failures(self, capsys):
        """Test CI JSON output returns 1 on failures."""
        report = ValidationReport(
            level="standard",
            results=[ValidationResult("A", "fail", "bad")],
        )
        exit_code = ValidateCommand._output_ci_json(report)
        assert exit_code == 1

    def test_output_ci_json_warnings(self, capsys):
        """Test CI JSON output returns 2 on warnings only."""
        report = ValidationReport(
            level="standard",
            results=[ValidationResult("A", "warning", "warn")],
        )
        exit_code = ValidateCommand._output_ci_json(report)
        assert exit_code == 2

    def test_save_report_json(self):
        """Test saving report in JSON format."""
        report = ValidationReport(
            level="quick",
            results=[ValidationResult("A", "pass", "ok")],
            timestamp=1000.0,
            duration_ms=50.0,
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            try:
                ValidateCommand._save_report(report, f.name, 'json', verbose=False)
                assert os.path.exists(f.name)

                with open(f.name) as jf:
                    data = json.load(jf)
                assert data['level'] == 'quick'
                assert len(data['results']) == 1
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_save_report_text(self):
        """Test saving report in text format."""
        report = ValidationReport(
            level="standard",
            results=[
                ValidationResult("A", "pass", "ok"),
                ValidationResult("B", "fail", "bad", details="detail"),
            ],
            timestamp=1000.0,
            duration_ms=100.0,
        )

        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            try:
                ValidateCommand._save_report(report, f.name, 'text', verbose=False)
                assert os.path.exists(f.name)

                with open(f.name) as tf:
                    content = tf.read()
                assert 'PASS' in content
                assert 'FAIL' in content
                assert 'detail' in content
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_execute_with_output(self):
        """Test execute with output file."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            try:
                args = MagicMock()
                args.level = 'quick'
                args.model = None
                args.output = f.name
                args.format = 'json'
                args.ci = False
                args.verbose = False

                result = ValidateCommand.execute(args)
                assert result in [0, 1, 2]
                assert os.path.exists(f.name)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_display_report(self, capsys):
        """Test display report output."""
        report = ValidationReport(
            level="quick",
            results=[
                ValidationResult("A", "pass", "ok"),
                ValidationResult("B", "warning", "warn"),
            ],
            duration_ms=100.0,
        )

        ValidateCommand._display_report(report, verbose=False)

        captured = capsys.readouterr()
        assert "Validation Results" in captured.out
        assert "Summary" in captured.out

    def test_execute_error_handling(self, capsys):
        """Test error handling in execute."""
        args = MagicMock()
        args.level = 'quick'
        args.model = None
        args.output = None
        args.format = 'text'
        args.ci = False
        args.verbose = False

        with patch.object(
            ValidateCommand, '_run_quick_checks', side_effect=Exception("Test error")
        ):
            result = ValidateCommand.execute(args)
            assert result == 1

    def test_execute_ci_error_handling(self, capsys):
        """Test CI mode error handling returns JSON."""
        args = MagicMock()
        args.level = 'quick'
        args.model = None
        args.output = None
        args.format = 'json'
        args.ci = True
        args.verbose = False

        with patch.object(
            ValidateCommand, '_run_quick_checks', side_effect=Exception("Test error")
        ):
            result = ValidateCommand.execute(args)
            assert result == 1

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert 'error' in data
