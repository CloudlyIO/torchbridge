"""
Tests for CLI benchmarking functionality.
"""

import pytest
import time
import tempfile
import os
from unittest.mock import patch, MagicMock

from benchmarks.cli_performance_benchmark import (
    CLIPerformanceBenchmark,
    CLIBenchmarkResult,
    PackagingBenchmark
)


class TestCLIBenchmarkResult:
    """Test CLIBenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a CLIBenchmarkResult."""
        result = CLIBenchmarkResult(
            command="test command",
            execution_time_ms=100.5,
            memory_usage_mb=50.0,
            exit_code=0,
            stdout_lines=10,
            stderr_lines=2,
            success=True
        )

        assert result.command == "test command"
        assert result.execution_time_ms == 100.5
        assert result.memory_usage_mb == 50.0
        assert result.exit_code == 0
        assert result.stdout_lines == 10
        assert result.stderr_lines == 2
        assert result.success is True


class TestCLIPerformanceBenchmark:
    """Test CLI performance benchmarking."""

    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = CLIPerformanceBenchmark()
        assert benchmark.results == []
        assert benchmark.python_executable is not None

    @patch('subprocess.run')
    def test_benchmark_cli_command_success(self, mock_run):
        """Test successful CLI command benchmarking."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output\nLine 2"
        mock_result.stderr = "Warning"
        mock_run.return_value = mock_result

        benchmark = CLIPerformanceBenchmark()
        result = benchmark.benchmark_cli_command(['-m', 'kernel_pytorch.cli', '--version'])

        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout_lines == 2
        assert result.stderr_lines == 1
        assert result.execution_time_ms > 0

    @patch('subprocess.run')
    def test_benchmark_cli_command_timeout(self, mock_run):
        """Test CLI command timeout handling."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired('test', 60)

        benchmark = CLIPerformanceBenchmark()
        result = benchmark.benchmark_cli_command(['-m', 'test'], timeout=60)

        assert result.success is False
        assert result.exit_code == -1
        assert result.execution_time_ms == 60000  # timeout * 1000

    @patch('subprocess.run')
    def test_benchmark_cli_command_exception(self, mock_run):
        """Test CLI command exception handling."""
        mock_run.side_effect = Exception("Test error")

        benchmark = CLIPerformanceBenchmark()
        result = benchmark.benchmark_cli_command(['-m', 'test'])

        assert result.success is False
        assert result.exit_code == -2

    def test_benchmark_import_performance(self):
        """Test import performance benchmarking."""
        benchmark = CLIPerformanceBenchmark()
        results = benchmark.benchmark_import_performance()

        # Should have results for main modules
        assert 'kernel_pytorch' in results
        assert 'kernel_pytorch.cli' in results

        # Times should be reasonable (positive numbers)
        for module, time_ms in results.items():
            if time_ms >= 0:  # Skip failed imports
                assert time_ms < 10000  # Should import in < 10 seconds

    @patch('subprocess.run')
    def test_benchmark_cli_help_commands(self, mock_run):
        """Test CLI help commands benchmarking."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Help output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        benchmark = CLIPerformanceBenchmark()
        results = benchmark.benchmark_cli_help_commands()

        assert len(results) > 0
        # All help commands should succeed
        for result in results:
            assert result.success is True

    @patch('subprocess.run')
    def test_benchmark_doctor_command(self, mock_run):
        """Test doctor command benchmarking."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Doctor output"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        benchmark = CLIPerformanceBenchmark()
        result = benchmark.benchmark_doctor_command()

        assert result.success is True
        assert 'doctor' in result.command

    @patch('torch.save')
    @patch('subprocess.run')
    def test_benchmark_quick_optimization(self, mock_run, mock_save):
        """Test quick optimization benchmarking."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Optimization complete"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        benchmark = CLIPerformanceBenchmark()
        result = benchmark.benchmark_quick_optimization()

        assert result.success is True
        assert 'optimize' in result.command

    @patch('subprocess.run')
    def test_benchmark_quick_benchmark_command(self, mock_run):
        """Test benchmark command benchmarking."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Benchmark complete"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        benchmark = CLIPerformanceBenchmark()
        result = benchmark.benchmark_quick_benchmark_command()

        assert result.success is True
        assert 'benchmark' in result.command

    @patch('benchmarks.cli_performance_benchmark.CLIPerformanceBenchmark.benchmark_import_performance')
    @patch('benchmarks.cli_performance_benchmark.CLIPerformanceBenchmark.benchmark_cli_help_commands')
    @patch('benchmarks.cli_performance_benchmark.CLIPerformanceBenchmark.benchmark_doctor_command')
    def test_run_all_benchmarks(self, mock_doctor, mock_help, mock_import):
        """Test running all benchmarks."""
        # Mock return values
        mock_import.return_value = {'kernel_pytorch': 100.0}
        mock_help.return_value = [CLIBenchmarkResult(
            command='test', execution_time_ms=50.0, memory_usage_mb=10.0,
            exit_code=0, stdout_lines=5, stderr_lines=0, success=True
        )]
        mock_doctor.return_value = CLIBenchmarkResult(
            command='doctor', execution_time_ms=75.0, memory_usage_mb=15.0,
            exit_code=0, stdout_lines=10, stderr_lines=0, success=True
        )

        benchmark = CLIPerformanceBenchmark()
        results = benchmark.run_all_benchmarks()

        assert 'import_performance' in results
        assert 'help_commands' in results
        assert 'doctor_command' in results

    def test_display_results(self):
        """Test results display."""
        benchmark = CLIPerformanceBenchmark()
        results = {
            'import_performance': {'kernel_pytorch': 100.0},
            'help_commands': [{
                'command': 'test --help',
                'execution_time_ms': 50.0,
                'success': True
            }],
            'doctor_command': {
                'execution_time_ms': 75.0,
                'success': True
            }
        }

        # Should not raise any exceptions
        benchmark.display_results(results)

    def test_save_results(self):
        """Test saving results to file."""
        benchmark = CLIPerformanceBenchmark()
        results = {'test': 'data'}

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            try:
                benchmark.save_results(results, f.name)
                assert os.path.exists(f.name)

                # Verify file content
                import json
                with open(f.name, 'r') as json_file:
                    saved_data = json.load(json_file)

                assert 'timestamp' in saved_data
                assert 'results' in saved_data
                assert saved_data['results'] == results

            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)


class TestPackagingBenchmark:
    """Test packaging benchmarking."""

    def test_packaging_benchmark_initialization(self):
        """Test packaging benchmark initialization."""
        benchmark = PackagingBenchmark()
        assert benchmark.package_root is not None

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.rglob')
    @patch('pathlib.Path.stat')
    def test_benchmark_package_size(self, mock_stat, mock_rglob, mock_exists):
        """Test package size benchmarking."""
        # Mock file system
        mock_exists.return_value = True

        # Mock file objects
        mock_file = MagicMock()
        mock_file.stat.return_value.st_size = 1024  # 1KB per file
        mock_rglob.return_value = [mock_file, mock_file, mock_file]  # 3 files

        benchmark = PackagingBenchmark()
        results = benchmark.benchmark_package_size()

        # Should have metrics for existing directories
        assert len(results) > 0
        for category, metrics in results.items():
            assert 'total_bytes' in metrics
            assert 'total_mb' in metrics
            assert 'file_count' in metrics

    @patch('subprocess.run')
    @patch('pathlib.Path.glob')
    def test_benchmark_wheel_build_time_success(self, mock_glob, mock_run):
        """Test successful wheel build benchmarking."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Build successful"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Mock wheel file creation
        mock_wheel = MagicMock()
        mock_glob.return_value = [mock_wheel]

        benchmark = PackagingBenchmark()
        result = benchmark.benchmark_wheel_build_time()

        assert result['success'] is True
        assert result['wheel_created'] is True
        assert result['build_time_seconds'] > 0

    @patch('subprocess.run')
    def test_benchmark_wheel_build_time_timeout(self, mock_run):
        """Test wheel build timeout handling."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired('build', 300)

        benchmark = PackagingBenchmark()
        result = benchmark.benchmark_wheel_build_time()

        assert result['success'] is False
        assert result['error'] == 'timeout'

    @patch('subprocess.run')
    def test_benchmark_wheel_build_time_exception(self, mock_run):
        """Test wheel build exception handling."""
        mock_run.side_effect = Exception("Build error")

        benchmark = PackagingBenchmark()
        result = benchmark.benchmark_wheel_build_time()

        assert result['success'] is False
        assert 'error' in result


class TestBenchmarkIntegration:
    """Integration tests for benchmark functionality."""

    @pytest.mark.slow
    def test_benchmark_import_performance_integration(self):
        """Integration test for import performance benchmarking."""
        benchmark = CLIPerformanceBenchmark()
        results = benchmark.benchmark_import_performance()

        # Should successfully benchmark core imports
        assert 'kernel_pytorch' in results
        assert results['kernel_pytorch'] > 0  # Should take some time

        # CLI imports should also work
        assert 'kernel_pytorch.cli' in results

    @pytest.mark.slow
    def test_cli_command_availability(self):
        """Test that CLI commands are actually available."""
        benchmark = CLIPerformanceBenchmark()

        # Test that we can at least check version
        result = benchmark.benchmark_cli_command(['-m', 'kernel_pytorch.cli', '--version'], timeout=30)

        # Command should exist and run (success or help exit code)
        assert result.exit_code in [0, 2, -1]  # 0=success, 2=help, -1=timeout

    def test_package_size_realistic(self):
        """Test that package size metrics are realistic."""
        benchmark = PackagingBenchmark()
        results = benchmark.benchmark_package_size()

        # Should have some package metrics
        if 'source_code' in results:
            source_metrics = results['source_code']
            # Source code should be reasonable size (> 100KB, < 100MB)
            assert 0.1 < source_metrics['total_mb'] < 100
            assert source_metrics['file_count'] > 10  # Should have multiple files