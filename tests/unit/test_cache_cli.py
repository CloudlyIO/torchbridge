"""
Tests for the Cache CLI Command

Tests CacheCommand argument registration, execution, and output modes.
"""

import argparse
import json

from torchbridge.cli.cache import CacheCommand


class TestCacheCommand:
    """Tests for the CacheCommand CLI."""

    def test_register_adds_cache_subcommand(self):
        """register() should add a 'cache' subparser."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        CacheCommand.register(subparsers)
        args = parser.parse_args(["cache"])
        assert args.command == "cache"

    def test_execute_auto_backend(self):
        """Execute with auto backend (CPU on test machines) should succeed."""
        args = argparse.Namespace(backend="cpu", show_matrix=False, ci=False)
        exit_code = CacheCommand.execute(args)
        assert exit_code == 0

    def test_execute_ci_json_output(self, capsys):
        """CI mode should produce valid JSON."""
        args = argparse.Namespace(backend="cpu", show_matrix=False, ci=True)
        exit_code = CacheCommand.execute(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "backend" in data
        assert "optimal_dtype" in data
        assert "supported_dtypes" in data

    def test_execute_show_matrix(self):
        """--show-matrix should succeed."""
        args = argparse.Namespace(backend="cpu", show_matrix=True, ci=False)
        exit_code = CacheCommand.execute(args)
        assert exit_code == 0

    def test_execute_show_matrix_ci(self, capsys):
        """--show-matrix --ci should produce JSON array."""
        args = argparse.Namespace(backend="cpu", show_matrix=True, ci=True)
        exit_code = CacheCommand.execute(args)
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "backend" in data[0]
