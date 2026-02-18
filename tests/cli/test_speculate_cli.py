"""
Tests for the speculate CLI command.

Tests argument parsing, execution, matrix display, and CI output.
"""

import json

from torchbridge.cli import main


class TestSpeculateCLI:
    """Tests for 'torchbridge speculate' command."""

    def test_speculate_default(self, capsys):
        """Default invocation produces human-readable output."""
        result = main(["speculate", "--backend", "cpu"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Speculative Decoding" in captured.out
        assert "prompt_lookup" in captured.out

    def test_speculate_ci_json(self, capsys):
        """--ci flag produces valid JSON."""
        result = main(["speculate", "--backend", "cpu", "--ci"])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["backend"] == "cpu"
        assert "prompt_lookup" in data["supported_methods"]

    def test_speculate_show_matrix(self, capsys):
        """--show-matrix displays the full table."""
        result = main(["speculate", "--show-matrix"])
        assert result == 0
        captured = capsys.readouterr()
        assert "Compatibility Matrix" in captured.out
        assert "blackwell_dc" in captured.out

    def test_speculate_show_matrix_ci(self, capsys):
        """--show-matrix --ci produces JSON array."""
        result = main(["speculate", "--show-matrix", "--ci"])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "backend" in data[0]

    def test_speculate_check_method(self, capsys):
        """--method checks specific method support."""
        result = main(["speculate", "--backend", "cpu", "--method", "eagle"])
        assert result == 0
        captured = capsys.readouterr()
        assert "NOT SUPPORTED" in captured.out
