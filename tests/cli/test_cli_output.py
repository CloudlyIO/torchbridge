"""CLI output validation tests — check meaningful output, not just exit codes."""

import subprocess
import sys


class TestCLIOutput:
    def _run(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "torchbridge.cli", *args],
            capture_output=True,
            text=True,
        )

    def test_doctor_output_contains_backend(self):
        result = self._run("doctor")
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert any(
            word in combined.lower()
            for word in ("backend", "cuda", "cpu", "amd", "rocm", "mps", "neuron")
        )

    def test_version_output(self):
        result = self._run("--version")
        assert result.returncode == 0
        assert "0.5" in result.stdout + result.stderr

    def test_invalid_command_shows_help(self):
        result = self._run("nonexistent-subcommand")
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert len(combined.strip()) > 0  # must print something useful
