"""
Unit tests for tb-adapter CLI fixes (v0.5.75).

Covers:
- "nvidia" replaces "cuda" in backend choices for recommend subcommand
- --hidden-dim and --num-modules flags added to recommend subcommand
"""

import argparse

import pytest


def _build_parser() -> argparse.ArgumentParser:
    """Build a fresh adapter argument parser via AdapterCommand.register."""
    from torchbridge.cli.adapter import AdapterCommand

    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="cmd")
    AdapterCommand.register(sub)
    return root


class TestBackendChoiceNvidiaVsCuda:
    def test_backend_choice_nvidia_accepted(self):
        """recommend --backend nvidia must be accepted (not cuda)."""
        parser = _build_parser()
        args = parser.parse_args(["adapter", "recommend", "--backend", "nvidia"])
        assert args.backend == "nvidia"

    def test_backend_choice_cuda_rejected(self):
        """recommend --backend cuda must be REJECTED after renaming to nvidia."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["adapter", "recommend", "--backend", "cuda"])

    def test_other_backends_still_accepted(self):
        """amd, trainium, tpu, cpu must still be valid backend choices."""
        parser = _build_parser()
        for backend in ["amd", "trainium", "tpu", "cpu"]:
            args = parser.parse_args(["adapter", "recommend", "--backend", backend])
            assert args.backend == backend


class TestHiddenDimAndNumModulesFlags:
    def test_hidden_dim_flag_accepted(self):
        """--hidden-dim must be accepted by the recommend subcommand."""
        parser = _build_parser()
        args = parser.parse_args(
            ["adapter", "recommend", "--backend", "cpu", "--hidden-dim", "8192"]
        )
        assert args.hidden_dim == 8192

    def test_num_modules_flag_accepted(self):
        """--num-modules must be accepted by the recommend subcommand."""
        parser = _build_parser()
        args = parser.parse_args(
            ["adapter", "recommend", "--backend", "cpu", "--num-modules", "8"]
        )
        assert args.num_modules == 8

    def test_hidden_dim_default_is_4096(self):
        """--hidden-dim must default to 4096."""
        parser = _build_parser()
        args = parser.parse_args(["adapter", "recommend", "--backend", "cpu"])
        assert args.hidden_dim == 4096

    def test_num_modules_default_is_4(self):
        """--num-modules must default to 4."""
        parser = _build_parser()
        args = parser.parse_args(["adapter", "recommend", "--backend", "cpu"])
        assert args.num_modules == 4

    def test_custom_dims_reflected_in_output(self, capsys):
        """With --hidden-dim 256 --num-modules 2 --rank 8, param count must match 2 * 2 * 256 * 8."""
        from torchbridge.cli.adapter import _show_recommend

        class FakeArgs:
            backend = "cpu"
            rank = 8
            hidden_dim = 256
            num_modules = 2
            ci = False

        _show_recommend(FakeArgs())
        out = capsys.readouterr().out
        # 2 modules * 2 matrices * 256 * 8 = 8192 params
        expected = 2 * 2 * 256 * 8
        # Output uses locale thousands separator (e.g. "8,192"); check either form
        assert str(expected) in out or f"{expected:,}" in out, (
            f"Expected param count {expected} not found in output:\n{out}"
        )
