"""
Unit tests for tb-validate --compare

Tests argument parsing, device resolution, output shape, and the
CPU-vs-CPU smoke path (always available, no GPU required).
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from torchbridge.cli.validate import ValidateCommand


def _make_args(**kwargs) -> argparse.Namespace:
    """Build a minimal Namespace for _run_compare."""
    defaults = {
        'compare': ['cpu', 'cpu'],
        'model': None,
        'input_shape': '1,64',
        'per_layer': False,
        'dtype': 'float32',
        'output': None,
        'ci': False,
        'verbose': False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestCompareArgParsing:
    """Verify --compare args are accepted and routed correctly."""

    def test_compare_args_present_in_register(self):
        """register() must add --compare, --input-shape, --per-layer, --dtype."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ValidateCommand.register(subparsers)
        subparser = subparsers.choices['validate']
        option_strings = {a.option_strings[0] for a in subparser._actions if a.option_strings}
        assert '--compare' in option_strings
        assert '--input-shape' in option_strings
        assert '--per-layer' in option_strings
        assert '--dtype' in option_strings

    def test_compare_routes_away_from_standard_pipeline(self):
        """execute() with --compare must call _run_compare, not the level pipeline."""
        args = _make_args(compare=['cpu', 'cpu'])
        called = []

        def fake_compare(a):
            called.append(a)
            return 0

        with patch.object(ValidateCommand, '_run_compare', staticmethod(fake_compare)):
            result = ValidateCommand.execute(args)

        assert result == 0
        assert len(called) == 1

    def test_no_compare_does_not_call_run_compare(self):
        """execute() without --compare must not call _run_compare."""
        args = _make_args(compare=None, level='quick', ci=True)
        called = []

        def fake_compare(a):
            called.append(a)
            return 0

        with patch.object(ValidateCommand, '_run_compare', staticmethod(fake_compare)):
            ValidateCommand.execute(args)

        assert len(called) == 0


class TestCpuCpuComparison:
    """CPU vs CPU always passes (deterministic, same device)."""

    def test_cpu_cpu_returns_zero(self):
        """CPU vs CPU comparison must return 0 (PASSED)."""
        args = _make_args(compare=['cpu', 'cpu'])
        result = ValidateCommand._run_compare(args)
        assert result == 0

    def test_cpu_cpu_max_diff_is_zero(self, capsys):
        """CPU vs CPU must report max_diff=0.0 in CI JSON output."""
        args = _make_args(compare=['cpu', 'cpu'], ci=True)
        ValidateCommand._run_compare(args)
        out = json.loads(capsys.readouterr().out)
        assert out['max_diff'] == 0.0
        assert out['passed'] is True
        assert out['backend1'] == 'cpu'
        assert out['backend2'] == 'cpu'

    def test_output_json_contains_required_keys(self, capsys):
        """CI JSON output must contain all required keys."""
        args = _make_args(compare=['cpu', 'cpu'], ci=True)
        ValidateCommand._run_compare(args)
        out = json.loads(capsys.readouterr().out)
        for key in ('backend1', 'backend2', 'model', 'dtype', 'input_shape',
                    'max_diff', 'cosine_sim', 'tolerance_atol', 'tolerance_rtol',
                    'passed', 'duration_ms', 'per_layer'):
            assert key in out, f"Missing key: {key}"

    def test_custom_input_shape(self, capsys):
        """--input-shape is reflected in CI JSON output."""
        args = _make_args(compare=['cpu', 'cpu'], input_shape='2,32', ci=True)
        ValidateCommand._run_compare(args)
        out = json.loads(capsys.readouterr().out)
        assert out['input_shape'] == [2, 32]

    def test_dtype_float32_default(self, capsys):
        """Default dtype is float32."""
        args = _make_args(compare=['cpu', 'cpu'], ci=True)
        ValidateCommand._run_compare(args)
        out = json.loads(capsys.readouterr().out)
        assert out['dtype'] == 'float32'

    def test_smoke_model_label(self, capsys):
        """Without --model, output must note a smoke model was used."""
        args = _make_args(compare=['cpu', 'cpu'], ci=True)
        ValidateCommand._run_compare(args)
        out = json.loads(capsys.readouterr().out)
        assert 'smoke_model' in out['model']


class TestUnknownBackend:
    """Unknown or unavailable backends must return exit code 1."""

    def test_unknown_backend1_returns_1(self, capsys):
        args = _make_args(compare=['xpu_fake', 'cpu'])
        result = ValidateCommand._run_compare(args)
        assert result == 1

    def test_unknown_backend2_returns_1(self, capsys):
        args = _make_args(compare=['cpu', 'xpu_fake'])
        result = ValidateCommand._run_compare(args)
        assert result == 1

    def test_cuda_unavailable_returns_1(self):
        """If CUDA is not available, --compare cuda cpu must return 1."""
        with patch('torch.cuda.is_available', return_value=False):
            args = _make_args(compare=['cuda', 'cpu'])
            result = ValidateCommand._run_compare(args)
        assert result == 1

    def test_unknown_backend_ci_json_has_error_key(self, capsys):
        """CI JSON for unknown backend must contain 'error' key."""
        args = _make_args(compare=['xpu_fake', 'cpu'], ci=True)
        ValidateCommand._run_compare(args)
        out = json.loads(capsys.readouterr().out)
        assert 'error' in out


class TestOutputFile:
    """--output saves JSON to disk."""

    def test_output_file_created(self):
        """--output must create a valid JSON file."""
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / 'compare.json')
            args = _make_args(compare=['cpu', 'cpu'], output=out_path)
            ValidateCommand._run_compare(args)
            assert Path(out_path).exists()
            with open(out_path) as f:
                data = json.load(f)
            assert data['passed'] is True

    def test_output_file_has_all_keys(self):
        """Saved JSON file must have all required keys."""
        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / 'compare.json')
            args = _make_args(compare=['cpu', 'cpu'], output=out_path)
            ValidateCommand._run_compare(args)
            with open(out_path) as f:
                data = json.load(f)
            for key in ('backend1', 'backend2', 'max_diff', 'cosine_sim', 'passed'):
                assert key in data


class TestHumanReadableOutput:
    """Human-readable (non-CI) output format."""

    def test_human_output_contains_status(self, capsys):
        """Human output must contain 'PASSED' or 'FAILED'."""
        args = _make_args(compare=['cpu', 'cpu'], ci=False)
        ValidateCommand._run_compare(args)
        out = capsys.readouterr().out
        assert 'PASSED' in out or 'FAILED' in out

    def test_human_output_contains_backends(self, capsys):
        """Human output must show both backends."""
        args = _make_args(compare=['cpu', 'cpu'], ci=False)
        ValidateCommand._run_compare(args)
        out = capsys.readouterr().out
        assert 'cpu' in out


# ── v0.5.69: fallback tolerance annotation ───────────────────────────────────

class TestFallbackToleranceAnnotation:
    def test_fallback_source_annotated_in_output(self, capsys):
        """When tolerance source is 'fallback', Tolerance line must include annotation."""
        from unittest.mock import patch

        from torchbridge.testing.tolerance_db import ToleranceEntry

        fallback_entry = ToleranceEntry(atol=1e-3, rtol=1e-3, source="fallback",
                                        notes="unknown backend")
        args = _make_args(compare=['cpu', 'cpu'], ci=False)
        with patch("torchbridge.testing.tolerance_db.ToleranceDB.get",
                   return_value=fallback_entry):
            ValidateCommand._run_compare(args)
        out = capsys.readouterr().out
        assert 'fallback' in out.lower()

    def test_known_backend_no_fallback_annotation(self, capsys):
        """Known backend (cpu) must not show fallback annotation."""
        args = _make_args(compare=['cpu', 'cpu'], ci=False)
        ValidateCommand._run_compare(args)
        out = capsys.readouterr().out
        assert 'fallback — backend not in tolerance DB' not in out
