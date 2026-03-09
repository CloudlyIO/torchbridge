"""
Integration tests for the --trace CLI pipeline.

Tests invoke ValidateCommand.execute() directly (not subprocess) so they run
fast and deterministically on CPU without real hardware.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from torchbridge.cli.validate import ValidateCommand

# ── Helpers ───────────────────────────────────────────────────────────────────

def _args(**kwargs) -> SimpleNamespace:
    """Build a minimal args namespace with sensible defaults for trace tests."""
    defaults = {
        'compare': ['cpu', 'cpu'],
        'trace': True,
        'steps': 3,
        'autoregressive': False,
        'trace_output': None,
        'model': None,
        'input_shape': '1,8',
        'dtype': 'float32',
        'output': None,
        'ci': False,
        'verbose': False,
        'level': 'standard',
        'per_layer': False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ── Argument registration ─────────────────────────────────────────────────────

class TestCliArgumentRegistration:
    def test_trace_flag_registered(self):
        """--trace must be a recognised argument in the subparser."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ValidateCommand.register(subparsers)
        args = parser.parse_args(['validate', '--compare', 'cpu', 'cpu', '--trace'])
        assert args.trace is True

    def test_steps_registered(self):
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ValidateCommand.register(subparsers)
        args = parser.parse_args(['validate', '--compare', 'cpu', 'cpu',
                                  '--trace', '--steps', '7'])
        assert args.steps == 7

    def test_autoregressive_flag_registered(self):
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ValidateCommand.register(subparsers)
        args = parser.parse_args(['validate', '--compare', 'cpu', 'cpu',
                                  '--trace', '--autoregressive'])
        assert args.autoregressive is True

    def test_trace_output_registered(self):
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ValidateCommand.register(subparsers)
        args = parser.parse_args(['validate', '--compare', 'cpu', 'cpu',
                                  '--trace', '--trace-output', '/tmp/trace.json'])
        assert args.trace_output == '/tmp/trace.json'

    def test_steps_default_is_ten(self):
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        ValidateCommand.register(subparsers)
        args = parser.parse_args(['validate', '--compare', 'cpu', 'cpu', '--trace'])
        assert args.steps == 10


# ── Guard: --trace without --compare ─────────────────────────────────────────

class TestTraceRequiresCompare:
    def test_trace_without_compare_returns_error(self, capsys):
        args = SimpleNamespace(
            compare=None,
            trace=True,
            steps=3,
            autoregressive=False,
            trace_output=None,
            model=None,
            input_shape='1,8',
            dtype='float32',
            output=None,
            ci=False,
            verbose=False,
            level='standard',
            per_layer=False,
            quantized=False,
        )
        rc = ValidateCommand.execute(args)
        assert rc == 1
        captured = capsys.readouterr()
        assert '--trace requires --compare' in captured.out


# ── CPU vs CPU smoke runs ─────────────────────────────────────────────────────

class TestCpuCpuTrace:
    def test_cpu_cpu_trace_exits_zero(self):
        rc = ValidateCommand.execute(_args())
        assert rc == 0

    def test_steps_respected(self):
        """--steps N must produce exactly N step_results in JSON output."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ValidateCommand.execute(_args(steps=3, ci=True))
        assert rc == 0
        data = json.loads(buf.getvalue())
        assert len(data['step_results']) == 3

    def test_steps_default_ten(self):
        """Without explicit --steps, output has 10 step_results."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = ValidateCommand.execute(_args(steps=10, ci=True))
        assert rc == 0
        data = json.loads(buf.getvalue())
        assert len(data['step_results']) == 10

    def test_autoregressive_flag_accepted(self):
        """--autoregressive on a non-LM model must not crash."""
        rc = ValidateCommand.execute(_args(autoregressive=True))
        assert rc == 0


# ── CI / JSON output schema ───────────────────────────────────────────────────

class TestCiJsonSchema:
    def _capture_json(self, **kwargs) -> dict:
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            ValidateCommand.execute(_args(ci=True, **kwargs))
        return json.loads(buf.getvalue())

    def test_top_level_keys_present(self):
        data = self._capture_json(steps=2)
        for key in ('backend_a', 'backend_b', 'steps', 'dtype', 'autoregressive',
                    'first_divergence_step', 'max_amplification', 'final_passed',
                    'step_results', 'model'):
            assert key in data, f"Missing key: {key}"

    def test_step_results_schema(self):
        data = self._capture_json(steps=2)
        assert len(data['step_results']) == 2
        for row in data['step_results']:
            for key in ('step', 'max_diff', 'cosine_sim', 'within_tolerance',
                        'cumulative_amplification'):
                assert key in row, f"step_results row missing key: {key}"

    def test_final_passed_is_bool(self):
        data = self._capture_json(steps=2)
        assert isinstance(data['final_passed'], bool)

    def test_steps_field_matches_request(self):
        data = self._capture_json(steps=4)
        assert data['steps'] == 4


# ── --trace-output file ───────────────────────────────────────────────────────

class TestTraceOutputFile:
    def test_trace_output_file_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = str(Path(tmpdir) / 'trace.json')
            rc = ValidateCommand.execute(_args(steps=2, trace_output=out_file))
            assert rc == 0
            assert Path(out_file).exists()
            with open(out_file) as f:
                data = json.load(f)
            assert 'step_results' in data
            assert len(data['step_results']) == 2

    def test_trace_output_contains_correct_step_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = str(Path(tmpdir) / 'trace.json')
            ValidateCommand.execute(_args(steps=5, trace_output=out_file))
            with open(out_file) as f:
                data = json.load(f)
            assert len(data['step_results']) == 5


# ── Steps validation ──────────────────────────────────────────────────────────

class TestStepsValidation:
    def test_steps_out_of_range_returns_error(self):
        rc = ValidateCommand.execute(_args(steps=0))
        assert rc == 1

    def test_steps_1001_returns_error(self):
        rc = ValidateCommand.execute(_args(steps=1001))
        assert rc == 1

    def test_steps_1_is_valid(self):
        rc = ValidateCommand.execute(_args(steps=1))
        assert rc == 0

    def test_steps_1000_is_valid(self):
        rc = ValidateCommand.execute(_args(steps=1000))
        assert rc == 0
