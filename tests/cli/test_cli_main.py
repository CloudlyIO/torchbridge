"""
Tests for the main CLI interface.
"""

import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock

from kernel_pytorch.cli import main
from kernel_pytorch.cli.optimize import OptimizeCommand
from kernel_pytorch.cli.benchmark import BenchmarkCommand
from kernel_pytorch.cli.doctor import DoctorCommand


class TestCLIMain:
    """Test the main CLI interface."""

    def test_cli_version(self):
        """Test --version flag."""
        # Test with argument list
        with pytest.raises(SystemExit) as exc_info:
            main(['--version'])
        assert exc_info.value.code == 0

    def test_cli_help(self):
        """Test help output."""
        with pytest.raises(SystemExit) as exc_info:
            main(['--help'])
        assert exc_info.value.code == 0

    def test_cli_no_command(self):
        """Test CLI with no command shows help."""
        result = main([])
        assert result == 1

    def test_cli_invalid_command(self):
        """Test invalid command."""
        result = main(['invalid-command'])
        assert result == 1

    @patch('kernel_pytorch.cli.OptimizeCommand.execute')
    def test_cli_optimize_command(self, mock_execute):
        """Test optimize command routing."""
        mock_execute.return_value = 0
        result = main(['optimize', '--model', 'test.pt', '--level', 'basic'])
        assert result == 0
        mock_execute.assert_called_once()

    @patch('kernel_pytorch.cli.BenchmarkCommand.execute')
    def test_cli_benchmark_command(self, mock_execute):
        """Test benchmark command routing."""
        mock_execute.return_value = 0
        result = main(['benchmark', '--model', 'test.pt'])
        assert result == 0
        mock_execute.assert_called_once()

    @patch('kernel_pytorch.cli.DoctorCommand.execute')
    def test_cli_doctor_command(self, mock_execute):
        """Test doctor command routing."""
        mock_execute.return_value = 0
        result = main(['doctor'])
        assert result == 0
        mock_execute.assert_called_once()

    def test_cli_keyboard_interrupt(self):
        """Test keyboard interrupt handling."""
        with patch('kernel_pytorch.cli.OptimizeCommand.execute') as mock_execute:
            mock_execute.side_effect = KeyboardInterrupt()
            result = main(['optimize', '--model', 'test.pt'])
            assert result == 130

    def test_cli_exception_handling(self):
        """Test general exception handling."""
        with patch('kernel_pytorch.cli.OptimizeCommand.execute') as mock_execute:
            mock_execute.side_effect = ValueError("Test error")
            result = main(['optimize', '--model', 'test.pt'])
            assert result == 1


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_executable(self):
        """Test that CLI is properly installed and executable."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'kernel_pytorch.cli', '--version'],
                capture_output=True, text=True, timeout=30
            )
            assert result.returncode == 0
            assert 'kernelpytorch' in result.stdout or 'kernelpytorch' in result.stderr
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI not available in test environment")

    def test_doctor_command_integration(self):
        """Test doctor command integration."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'kernel_pytorch.cli.doctor'],
                capture_output=True, text=True, timeout=60
            )
            # Should complete without error (warnings are okay)
            assert result.returncode in [0, 1]  # 0 = success, 1 = warnings
            assert 'Diagnostics' in result.stdout
        except subprocess.TimeoutExpired:
            pytest.skip("Doctor command timed out")

    @pytest.mark.slow
    def test_benchmark_quick_integration(self):
        """Test quick benchmark integration."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'kernel_pytorch.cli.benchmark',
                 '--predefined', 'optimization', '--quick'],
                capture_output=True, text=True, timeout=120
            )
            # Should complete (may have warnings but shouldn't fail completely)
            assert result.returncode in [0, 1]
        except subprocess.TimeoutExpired:
            pytest.skip("Benchmark command timed out")


class TestCLIScriptEntryPoints:
    """Test script entry points defined in pyproject.toml."""

    def test_optimize_entry_point(self):
        """Test kpt-optimize entry point."""
        with patch('kernel_pytorch.cli.optimize.OptimizeCommand.execute') as mock_execute:
            mock_execute.return_value = 0

            # Import and test the main function
            from kernel_pytorch.cli.optimize import main as optimize_main
            with patch('sys.argv', ['kpt-optimize', '--model', 'test.pt']):
                result = optimize_main()
                assert result == 0

    def test_benchmark_entry_point(self):
        """Test kpt-benchmark entry point."""
        with patch('kernel_pytorch.cli.benchmark.BenchmarkCommand.execute') as mock_execute:
            mock_execute.return_value = 0

            from kernel_pytorch.cli.benchmark import main as benchmark_main
            with patch('sys.argv', ['kpt-benchmark', '--model', 'test.pt']):
                result = benchmark_main()
                assert result == 0

    def test_doctor_entry_point(self):
        """Test kpt-doctor entry point."""
        with patch('kernel_pytorch.cli.doctor.DoctorCommand.execute') as mock_execute:
            mock_execute.return_value = 0

            from kernel_pytorch.cli.doctor import main as doctor_main
            with patch('sys.argv', ['kpt-doctor']):
                result = doctor_main()
                assert result == 0