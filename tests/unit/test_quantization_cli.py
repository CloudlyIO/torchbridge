"""
Tests for the Quantize CLI Command

Tests subparser registration, argument parsing, execute return codes,
and CI mode JSON output.
"""

import argparse
import json

import pytest

from torchbridge.cli.quantize import QuantizeCommand


class TestQuantizeCommandRegistration:
    """Tests for QuantizeCommand.register."""

    @pytest.fixture
    def parser(self):
        """Create a parser with quantize subcommand registered."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        QuantizeCommand.register(subparsers)
        return parser

    def test_subparser_registered(self, parser):
        """Quantize subparser should be registered."""
        args = parser.parse_args(["quantize", "--model", "test.pt"])
        assert args.command == "quantize"

    def test_default_format_is_auto(self, parser):
        """Default format should be 'auto'."""
        args = parser.parse_args(["quantize", "--model", "test.pt"])
        assert args.format == "auto"

    def test_default_backend_is_auto(self, parser):
        """Default backend should be 'auto'."""
        args = parser.parse_args(["quantize", "--model", "test.pt"])
        assert args.backend == "auto"

    def test_format_override(self, parser):
        """--format should be parsed correctly."""
        args = parser.parse_args(
            ["quantize", "--model", "test.pt", "--format", "int8_dynamic"]
        )
        assert args.format == "int8_dynamic"

    def test_backend_choices(self, parser):
        """Valid backend choices should parse."""
        for backend in ("auto", "nvidia", "amd", "trainium", "tpu", "cpu"):
            args = parser.parse_args(
                ["quantize", "--model", "test.pt", "--backend", backend]
            )
            assert args.backend == backend

    def test_validate_flag(self, parser):
        """--validate should set the flag."""
        args = parser.parse_args(["quantize", "--model", "test.pt", "--validate"])
        assert args.validate is True

    def test_ci_flag(self, parser):
        """--ci should set CI mode."""
        args = parser.parse_args(["quantize", "--model", "test.pt", "--ci"])
        assert args.ci is True

    def test_output_path(self, parser):
        """--output should be parsed."""
        args = parser.parse_args(["quantize", "--model", "test.pt", "-o", "output.pt"])
        assert args.output == "output.pt"


class TestQuantizeCommandExecution:
    """Tests for QuantizeCommand.execute."""

    def test_nonexistent_model_returns_1(self):
        """Non-existent model should return exit code 1."""
        args = argparse.Namespace(
            model="/nonexistent/model.pt",
            format="auto",
            backend="auto",
            output=None,
            validate=False,
            verbose=False,
            ci=False,
        )
        result = QuantizeCommand.execute(args)
        assert result == 1

    def test_nonexistent_model_ci_mode(self, capsys):
        """CI mode should output JSON for errors."""
        args = argparse.Namespace(
            model="/nonexistent/model.pt",
            format="auto",
            backend="auto",
            output=None,
            validate=False,
            verbose=False,
            ci=True,
        )
        result = QuantizeCommand.execute(args)
        assert result == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "error" in data
