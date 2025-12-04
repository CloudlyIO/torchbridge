"""
CLI Performance Benchmarking

Benchmarks for CLI tool performance, import times, and packaging efficiency.
"""

import time
import subprocess
import sys
import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

import torch
import pytest


@dataclass
class CLIBenchmarkResult:
    """Results from CLI benchmarking."""
    command: str
    execution_time_ms: float
    memory_usage_mb: float
    exit_code: int
    stdout_lines: int
    stderr_lines: int
    success: bool


class CLIPerformanceBenchmark:
    """Benchmark CLI tools performance."""

    def __init__(self):
        self.results: List[CLIBenchmarkResult] = []
        self.python_executable = sys.executable

    def benchmark_cli_command(self, command: List[str], timeout: int = 60) -> CLIBenchmarkResult:
        """Benchmark a single CLI command."""
        start_time = time.time()

        try:
            # Run the command
            result = subprocess.run(
                [self.python_executable] + command,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # Estimate memory usage (rough approximation)
            memory_usage = len(result.stdout) + len(result.stderr)  # Bytes
            memory_usage_mb = memory_usage / (1024 * 1024)  # Convert to MB

            benchmark_result = CLIBenchmarkResult(
                command=' '.join(command),
                execution_time_ms=execution_time,
                memory_usage_mb=memory_usage_mb,
                exit_code=result.returncode,
                stdout_lines=len(result.stdout.splitlines()),
                stderr_lines=len(result.stderr.splitlines()),
                success=result.returncode in [0, 2]  # 2 is help exit code
            )

        except subprocess.TimeoutExpired:
            benchmark_result = CLIBenchmarkResult(
                command=' '.join(command),
                execution_time_ms=timeout * 1000,
                memory_usage_mb=0.0,
                exit_code=-1,
                stdout_lines=0,
                stderr_lines=0,
                success=False
            )

        except Exception as e:
            benchmark_result = CLIBenchmarkResult(
                command=' '.join(command),
                execution_time_ms=0.0,
                memory_usage_mb=0.0,
                exit_code=-2,
                stdout_lines=0,
                stderr_lines=0,
                success=False
            )

        self.results.append(benchmark_result)
        return benchmark_result

    def benchmark_import_performance(self) -> Dict[str, float]:
        """Benchmark import performance for CLI modules."""
        import_benchmarks = {}

        # Test main package import
        start_time = time.time()
        exec("import kernel_pytorch")
        import_benchmarks['kernel_pytorch'] = (time.time() - start_time) * 1000

        # Test CLI module imports
        cli_modules = [
            'kernel_pytorch.cli',
            'kernel_pytorch.cli.optimize',
            'kernel_pytorch.cli.benchmark',
            'kernel_pytorch.cli.doctor'
        ]

        for module in cli_modules:
            start_time = time.time()
            try:
                exec(f"import {module}")
                import_benchmarks[module] = (time.time() - start_time) * 1000
            except ImportError as e:
                import_benchmarks[module] = -1  # Import failed

        return import_benchmarks

    def benchmark_cli_help_commands(self) -> List[CLIBenchmarkResult]:
        """Benchmark CLI help commands."""
        help_commands = [
            ['-m', 'kernel_pytorch.cli', '--help'],
            ['-m', 'kernel_pytorch.cli', '--version'],
            ['-m', 'kernel_pytorch.cli.optimize', '--help'],
            ['-m', 'kernel_pytorch.cli.benchmark', '--help'],
            ['-m', 'kernel_pytorch.cli.doctor', '--help'],
        ]

        results = []
        for cmd in help_commands:
            result = self.benchmark_cli_command(cmd, timeout=30)
            results.append(result)

        return results

    def benchmark_doctor_command(self) -> CLIBenchmarkResult:
        """Benchmark doctor command performance."""
        return self.benchmark_cli_command([
            '-m', 'kernel_pytorch.cli.doctor',
            '--category', 'basic'
        ], timeout=60)

    def benchmark_quick_optimization(self) -> CLIBenchmarkResult:
        """Benchmark quick model optimization."""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            try:
                # Save a simple model
                model = torch.nn.Linear(32, 16)
                torch.save(model, f.name)

                # Benchmark optimization
                result = self.benchmark_cli_command([
                    '-m', 'kernel_pytorch.cli.optimize',
                    '--model', f.name,
                    '--level', 'basic',
                    '--hardware', 'cpu'
                ], timeout=120)

                return result

            finally:
                # Clean up
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def benchmark_quick_benchmark_command(self) -> CLIBenchmarkResult:
        """Benchmark the benchmark command itself."""
        return self.benchmark_cli_command([
            '-m', 'kernel_pytorch.cli.benchmark',
            '--predefined', 'optimization',
            '--quick'
        ], timeout=180)

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all CLI benchmarks."""
        print("🚀 Running CLI Performance Benchmarks")
        print("=" * 50)

        all_results = {}

        # Import performance
        print("📦 Benchmarking import performance...")
        all_results['import_performance'] = self.benchmark_import_performance()

        # Help commands
        print("📖 Benchmarking help commands...")
        help_results = self.benchmark_cli_help_commands()
        all_results['help_commands'] = [asdict(r) for r in help_results]

        # Doctor command
        print("🩺 Benchmarking doctor command...")
        doctor_result = self.benchmark_doctor_command()
        all_results['doctor_command'] = asdict(doctor_result)

        # Quick optimization (if possible)
        print("⚙️ Benchmarking quick optimization...")
        try:
            opt_result = self.benchmark_quick_optimization()
            all_results['optimization_command'] = asdict(opt_result)
        except Exception as e:
            print(f"   Optimization benchmark skipped: {e}")
            all_results['optimization_command'] = None

        # Benchmark command
        print("📊 Benchmarking benchmark command...")
        try:
            bench_result = self.benchmark_quick_benchmark_command()
            all_results['benchmark_command'] = asdict(bench_result)
        except Exception as e:
            print(f"   Benchmark command benchmark skipped: {e}")
            all_results['benchmark_command'] = None

        return all_results

    def display_results(self, results: Dict[str, Any]) -> None:
        """Display benchmark results."""
        print("\n📊 CLI Performance Benchmark Results")
        print("=" * 60)

        # Import performance
        print("\n📦 Import Performance:")
        import_perf = results.get('import_performance', {})
        for module, time_ms in import_perf.items():
            if time_ms >= 0:
                print(f"  {module:<40} {time_ms:>8.2f} ms")
            else:
                print(f"  {module:<40} {'FAILED':>8}")

        # Help commands
        print("\n📖 Help Commands Performance:")
        help_commands = results.get('help_commands', [])
        for cmd_result in help_commands:
            cmd = cmd_result['command'].split()[-1]  # Get last part of command
            time_ms = cmd_result['execution_time_ms']
            success = "✓" if cmd_result['success'] else "✗"
            print(f"  {cmd:<30} {success} {time_ms:>8.2f} ms")

        # Individual commands
        print("\n🛠️  Command Performance:")
        commands_to_check = ['doctor_command', 'optimization_command', 'benchmark_command']

        for cmd_name in commands_to_check:
            cmd_result = results.get(cmd_name)
            if cmd_result:
                time_ms = cmd_result['execution_time_ms']
                success = "✓" if cmd_result['success'] else "✗"
                cmd_display = cmd_name.replace('_command', '').title()
                print(f"  {cmd_display:<30} {success} {time_ms:>8.2f} ms")

        # Performance summary
        print("\n📈 Performance Summary:")
        total_commands = len([r for r in results.values() if isinstance(r, dict) and 'success' in r])
        successful_commands = len([r for r in results.values() if isinstance(r, dict) and r.get('success')])

        if total_commands > 0:
            success_rate = (successful_commands / total_commands) * 100
            print(f"  Commands tested: {total_commands}")
            print(f"  Success rate: {success_rate:.1f}%")

        # Import performance summary
        import_times = [t for t in import_perf.values() if t >= 0]
        if import_times:
            avg_import_time = sum(import_times) / len(import_times)
            max_import_time = max(import_times)
            print(f"  Average import time: {avg_import_time:.2f} ms")
            print(f"  Slowest import: {max_import_time:.2f} ms")

    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save benchmark results to file."""
        results_with_metadata = {
            'timestamp': time.time(),
            'python_version': sys.version,
            'platform': sys.platform,
            'results': results
        }

        with open(output_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")


class PackagingBenchmark:
    """Benchmark packaging and installation performance."""

    def __init__(self):
        self.package_root = Path(__file__).parent.parent

    def benchmark_package_size(self) -> Dict[str, Any]:
        """Benchmark package size metrics."""
        size_metrics = {}

        # Source code size
        src_dir = self.package_root / 'src'
        if src_dir.exists():
            total_size = sum(f.stat().st_size for f in src_dir.rglob('*.py'))
            file_count = len(list(src_dir.rglob('*.py')))
            size_metrics['source_code'] = {
                'total_bytes': total_size,
                'total_mb': total_size / (1024 * 1024),
                'file_count': file_count
            }

        # Test code size
        tests_dir = self.package_root / 'tests'
        if tests_dir.exists():
            total_size = sum(f.stat().st_size for f in tests_dir.rglob('*.py'))
            file_count = len(list(tests_dir.rglob('*.py')))
            size_metrics['tests'] = {
                'total_bytes': total_size,
                'total_mb': total_size / (1024 * 1024),
                'file_count': file_count
            }

        # Documentation size
        docs_dir = self.package_root / 'docs'
        if docs_dir.exists():
            total_size = sum(f.stat().st_size for f in docs_dir.rglob('*.md'))
            file_count = len(list(docs_dir.rglob('*.md')))
            size_metrics['documentation'] = {
                'total_bytes': total_size,
                'total_mb': total_size / (1024 * 1024),
                'file_count': file_count
            }

        return size_metrics

    def benchmark_wheel_build_time(self) -> Dict[str, Any]:
        """Benchmark wheel building performance."""
        try:
            start_time = time.time()

            result = subprocess.run(
                [sys.executable, '-m', 'build', '--wheel', '--no-isolation'],
                cwd=self.package_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )

            build_time = time.time() - start_time

            # Check if wheel was created
            dist_dir = self.package_root / 'dist'
            wheel_files = list(dist_dir.glob('*.whl'))

            return {
                'success': result.returncode == 0,
                'build_time_seconds': build_time,
                'wheel_created': len(wheel_files) > 0,
                'wheel_count': len(wheel_files),
                'exit_code': result.returncode,
                'stdout_length': len(result.stdout),
                'stderr_length': len(result.stderr)
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'build_time_seconds': 300,
                'wheel_created': False,
                'error': 'timeout'
            }
        except Exception as e:
            return {
                'success': False,
                'build_time_seconds': 0,
                'wheel_created': False,
                'error': str(e)
            }


def run_cli_performance_benchmark():
    """Main function to run CLI performance benchmarks."""
    benchmark = CLIPerformanceBenchmark()

    # Run all benchmarks
    results = benchmark.run_all_benchmarks()

    # Display results
    benchmark.display_results(results)

    # Save results
    output_file = 'cli_benchmark_results.json'
    benchmark.save_results(results, output_file)

    return results


def run_packaging_benchmark():
    """Main function to run packaging benchmarks."""
    print("\n🏗️  Running Packaging Benchmarks")
    print("=" * 50)

    packaging_benchmark = PackagingBenchmark()

    # Package size metrics
    print("📏 Analyzing package size...")
    size_metrics = packaging_benchmark.benchmark_package_size()

    for category, metrics in size_metrics.items():
        print(f"  {category.title()}: {metrics['total_mb']:.2f} MB ({metrics['file_count']} files)")

    # Wheel build performance (optional, may not work in all environments)
    print("\n🏗️ Testing wheel build performance...")
    try:
        build_metrics = packaging_benchmark.benchmark_wheel_build_time()
        if build_metrics['success']:
            print(f"  Build time: {build_metrics['build_time_seconds']:.2f} seconds")
            print(f"  Wheel created: {'Yes' if build_metrics['wheel_created'] else 'No'}")
        else:
            print(f"  Build failed: {build_metrics.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"  Build test skipped: {e}")

    return {'size_metrics': size_metrics}


if __name__ == '__main__':
    # Run CLI benchmarks
    cli_results = run_cli_performance_benchmark()

    # Run packaging benchmarks
    packaging_results = run_packaging_benchmark()

    print("\n✅ All benchmarks completed!")