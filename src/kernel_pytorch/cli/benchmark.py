"""
Benchmarking commands for KernelPyTorch CLI.

Provides comprehensive performance benchmarking and validation tools.
"""

import argparse
import sys
import torch
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import kernel_pytorch as kpt
from kernel_pytorch.testing_framework.performance_benchmarks import PerformanceBenchmarkSuite


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    mean_time_ms: float
    std_time_ms: float
    throughput_ops_per_sec: float
    memory_usage_mb: float
    gpu_utilization_percent: float = 0.0


class BenchmarkCommand:
    """Benchmark command implementation."""

    @staticmethod
    def register(subparsers) -> None:
        """Register the benchmark command with argument parser."""
        parser = subparsers.add_parser(
            'benchmark',
            help='Benchmark PyTorch models and optimizations',
            description='Run comprehensive performance benchmarks',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Benchmark Types:
  model      - Single model performance benchmark
  compare    - Compare multiple optimization levels
  regression - Performance regression testing
  stress     - Stress test with various batch sizes

Examples:
  kpt-benchmark --model resnet50 --quick
  kpt-benchmark --type compare --levels basic,compile,triton
  kpt-benchmark --type stress --batch-sizes 1,8,16,32
  kpt-benchmark --predefined transformers --output results.json
            """
        )

        parser.add_argument(
            '--model',
            type=str,
            help='Model to benchmark (file path or predefined name)'
        )

        parser.add_argument(
            '--type',
            choices=['model', 'compare', 'regression', 'stress'],
            default='model',
            help='Benchmark type (default: model)'
        )

        parser.add_argument(
            '--levels',
            type=str,
            default='basic,compile',
            help='Optimization levels to compare (comma-separated)'
        )

        parser.add_argument(
            '--batch-sizes',
            type=str,
            default='1,8,16',
            help='Batch sizes for stress testing (comma-separated)'
        )

        parser.add_argument(
            '--input-shape',
            type=str,
            help='Input tensor shape (e.g., "1,3,224,224")'
        )

        parser.add_argument(
            '--predefined',
            choices=['transformers', 'vision', 'optimization'],
            help='Run predefined benchmark suite'
        )

        parser.add_argument(
            '--quick',
            action='store_true',
            help='Quick benchmark (fewer runs for faster results)'
        )

        parser.add_argument(
            '--warmup',
            type=int,
            default=10,
            help='Number of warmup runs (default: 10)'
        )

        parser.add_argument(
            '--runs',
            type=int,
            default=100,
            help='Number of benchmark runs (default: 100, 20 if --quick)'
        )

        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output file for results (JSON format)'
        )

        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )

    @staticmethod
    def execute(args) -> int:
        """Execute the benchmark command."""
        print("📊 KernelPyTorch Performance Benchmarking")
        print("=" * 50)

        try:
            # Adjust runs for quick mode
            if args.quick and args.runs == 100:  # Only adjust if default
                args.runs = 20

            # Detect hardware
            device = BenchmarkCommand._detect_hardware(args.verbose)

            # Execute benchmark based on type
            if args.predefined:
                results = BenchmarkCommand._run_predefined_benchmarks(args, device)
            elif args.type == 'model':
                results = BenchmarkCommand._benchmark_single_model(args, device)
            elif args.type == 'compare':
                results = BenchmarkCommand._compare_optimization_levels(args, device)
            elif args.type == 'regression':
                results = BenchmarkCommand._regression_benchmark(args, device)
            elif args.type == 'stress':
                results = BenchmarkCommand._stress_test(args, device)
            else:
                raise ValueError(f"Unknown benchmark type: {args.type}")

            # Display results
            BenchmarkCommand._display_results(results, args.verbose)

            # Save results if requested
            if args.output:
                BenchmarkCommand._save_results(results, args.output, args.verbose)

            print("\n✅ Benchmarking completed successfully!")
            return 0

        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    @staticmethod
    def _detect_hardware(verbose: bool) -> torch.device:
        """Detect hardware capabilities."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if verbose:
                print(f"🖥️  GPU: {torch.cuda.get_device_name()}")
                print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            if verbose:
                print("🖥️  Apple Silicon GPU")
        else:
            device = torch.device('cpu')
            if verbose:
                print("🖥️  CPU")

        return device

    @staticmethod
    def _run_predefined_benchmarks(args, device: torch.device) -> List[BenchmarkResult]:
        """Run predefined benchmark suites."""
        if args.verbose:
            print(f"🎯 Running predefined benchmark: {args.predefined}")

        results = []

        if args.predefined == 'optimization':
            # Use existing performance benchmarking framework
            benchmark = PerformanceBenchmarkSuite()
            predefined_results = benchmark.run_predefined_benchmarks()

            # Handle the actual nested structure from PerformanceBenchmarkSuite
            for category_name, category_data in predefined_results.items():
                if isinstance(category_data, dict):
                    for sub_name, sub_data_list in category_data.items():
                        if isinstance(sub_data_list, list) and sub_data_list:
                            # Take the first result as representative
                            result_data = sub_data_list[0]
                            if 'latency_ms' in result_data:
                                latency_ms = float(result_data['latency_ms'])
                                results.append(BenchmarkResult(
                                    name=f"{category_name}_{sub_name}",
                                    mean_time_ms=latency_ms,
                                    std_time_ms=0,  # Not available in this format
                                    throughput_ops_per_sec=float(result_data.get('throughput_ops', 0)),
                                    memory_usage_mb=float(result_data.get('memory_mb', 0)),
                                    gpu_utilization_percent=0  # Not available
                                ))

        elif args.predefined == 'transformers':
            # Transformer-specific benchmarks
            shapes = [(1, 512, 768), (8, 512, 768), (16, 512, 768)]
            for batch_size, seq_len, hidden_size in shapes:
                model = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_size * 4, hidden_size)
                ).to(device)

                result = BenchmarkCommand._benchmark_model(
                    model, f"Transformer_B{batch_size}_S{seq_len}",
                    (batch_size, seq_len, hidden_size), device, args
                )
                results.append(result)

        elif args.predefined == 'vision':
            # Vision model benchmarks
            try:
                import torchvision.models as models
                shapes = [(1, 3, 224, 224), (8, 3, 224, 224), (16, 3, 224, 224)]

                for batch_size, channels, height, width in shapes:
                    model = models.resnet18(pretrained=False).to(device).eval()
                    result = BenchmarkCommand._benchmark_model(
                        model, f"ResNet18_B{batch_size}",
                        (batch_size, channels, height, width), device, args
                    )
                    results.append(result)
            except ImportError:
                print("⚠️  torchvision not available, skipping vision benchmarks")

        return results

    @staticmethod
    def _benchmark_single_model(args, device: torch.device) -> List[BenchmarkResult]:
        """Benchmark a single model."""
        if not args.model:
            raise ValueError("Model path required for single model benchmark")

        if args.verbose:
            print(f"🎯 Benchmarking model: {args.model}")

        # Load or create model
        model = BenchmarkCommand._load_model(args.model, device)
        input_shape = BenchmarkCommand._parse_input_shape(args.input_shape, args.model)

        # Benchmark model
        result = BenchmarkCommand._benchmark_model(model, args.model, input_shape, device, args)
        return [result]

    @staticmethod
    def _compare_optimization_levels(args, device: torch.device) -> List[BenchmarkResult]:
        """Compare different optimization levels."""
        if not args.model:
            raise ValueError("Model path required for optimization comparison")

        if args.verbose:
            print(f"🎯 Comparing optimization levels: {args.levels}")

        levels = args.levels.split(',')
        results = []

        # Load base model
        base_model = BenchmarkCommand._load_model(args.model, device)
        input_shape = BenchmarkCommand._parse_input_shape(args.input_shape, args.model)

        for level in levels:
            level = level.strip()
            optimized_model = BenchmarkCommand._apply_optimization(base_model, level, input_shape, device)

            result = BenchmarkCommand._benchmark_model(
                optimized_model, f"{args.model}_{level}", input_shape, device, args
            )
            results.append(result)

        return results

    @staticmethod
    def _regression_benchmark(args, device: torch.device) -> List[BenchmarkResult]:
        """Run regression benchmarks to detect performance changes."""
        if args.verbose:
            print("🎯 Running regression benchmarks")

        # Use a standard set of models and optimizations
        standard_benchmarks = [
            ("linear_512_512", torch.nn.Linear(512, 512), (16, 512)),
            ("attention_mock", torch.nn.MultiheadAttention(768, 12, batch_first=True), (8, 512, 768)),
        ]

        results = []
        for name, model, input_shape in standard_benchmarks:
            model = model.to(device).eval()
            result = BenchmarkCommand._benchmark_model(model, name, input_shape, device, args)
            results.append(result)

        return results

    @staticmethod
    def _stress_test(args, device: torch.device) -> List[BenchmarkResult]:
        """Run stress tests with varying batch sizes."""
        if not args.model:
            # Use default model for stress test
            args.model = "linear_stress_test"

        if args.verbose:
            print(f"🎯 Stress testing with batch sizes: {args.batch_sizes}")

        batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(',')]
        results = []

        base_model = BenchmarkCommand._load_model(args.model, device)
        base_shape = BenchmarkCommand._parse_input_shape(args.input_shape, args.model)

        for batch_size in batch_sizes:
            # Modify shape for different batch sizes
            stress_shape = (batch_size,) + base_shape[1:]

            result = BenchmarkCommand._benchmark_model(
                base_model, f"{args.model}_batch_{batch_size}", stress_shape, device, args
            )
            results.append(result)

        return results

    @staticmethod
    def _load_model(model_name: str, device: torch.device) -> torch.nn.Module:
        """Load or create a model for benchmarking."""
        if model_name == "linear_stress_test":
            return torch.nn.Linear(1024, 1024).to(device).eval()
        elif model_name == "resnet50":
            try:
                import torchvision.models as models
                return models.resnet50(pretrained=False).to(device).eval()
            except ImportError:
                # Fallback simple model
                return torch.nn.Sequential(
                    torch.nn.Conv2d(3, 64, 7, 2, 3),
                    torch.nn.ReLU(),
                    torch.nn.AdaptiveAvgPool2d((1, 1)),
                    torch.nn.Flatten(),
                    torch.nn.Linear(64, 1000)
                ).to(device).eval()
        else:
            # Try to load from file
            if Path(model_name).exists():
                return torch.load(model_name, map_location=device).eval()
            else:
                # Default fallback
                return torch.nn.Linear(512, 512).to(device).eval()

    @staticmethod
    def _parse_input_shape(input_shape_str: Optional[str], model_name: str) -> tuple:
        """Parse input shape string or infer from model name."""
        if input_shape_str:
            return tuple(map(int, input_shape_str.split(',')))

        # Infer shape from model name
        if 'resnet' in model_name.lower() or 'vision' in model_name.lower():
            return (1, 3, 224, 224)
        elif 'transformer' in model_name.lower() or 'bert' in model_name.lower():
            return (1, 512, 768)
        elif model_name == 'linear_stress_test':
            return (16, 1024)  # Match the input size of Linear(1024, 1024)
        else:
            return (16, 512)  # Default for linear models

    @staticmethod
    def _apply_optimization(model: torch.nn.Module, level: str, input_shape: tuple, device: torch.device) -> torch.nn.Module:
        """Apply optimization level to model."""
        model_copy = model

        if level == 'basic':
            return model_copy.eval()
        elif level == 'jit':
            sample_input = torch.randn(input_shape, device=device)
            return torch.jit.trace(model_copy, sample_input)
        elif level == 'compile':
            return torch.compile(model_copy, mode='default')
        elif level == 'triton':
            return torch.compile(model_copy, mode='max-autotune')
        else:
            return model_copy

    @staticmethod
    def _benchmark_model(model: torch.nn.Module, name: str, input_shape: tuple,
                        device: torch.device, args) -> BenchmarkResult:
        """Benchmark a single model."""
        model.eval()
        sample_input = torch.randn(input_shape, device=device)

        # Memory before
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            memory_before = torch.cuda.memory_allocated()

        times = []

        with torch.no_grad():
            # Warmup
            for _ in range(args.warmup):
                _ = model(sample_input)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            # Benchmark
            for _ in range(args.runs):
                start_time = time.time()
                _ = model(sample_input)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms

        # Calculate statistics
        mean_time = sum(times) / len(times)
        std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        throughput = 1000 / mean_time if mean_time > 0 else 0

        # Memory usage
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated() - memory_before
            memory_usage_mb = peak_memory / 1e6
        else:
            memory_usage_mb = 0

        return BenchmarkResult(
            name=name,
            mean_time_ms=mean_time,
            std_time_ms=std_time,
            throughput_ops_per_sec=throughput,
            memory_usage_mb=memory_usage_mb
        )

    @staticmethod
    def _display_results(results: List[BenchmarkResult], verbose: bool) -> None:
        """Display benchmark results in a formatted table."""
        print("\n📊 Benchmark Results:")
        print("-" * 80)

        if verbose:
            print(f"{'Name':<30} {'Time (ms)':<12} {'Std (ms)':<10} {'Throughput':<12} {'Memory (MB)':<12}")
            print("-" * 80)
            for result in results:
                print(f"{result.name:<30} {result.mean_time_ms:<12.2f} {result.std_time_ms:<10.2f} "
                      f"{result.throughput_ops_per_sec:<12.1f} {result.memory_usage_mb:<12.1f}")
        else:
            print(f"{'Name':<30} {'Time (ms)':<12} {'Throughput (ops/s)':<20}")
            print("-" * 65)
            for result in results:
                print(f"{result.name:<30} {result.mean_time_ms:<12.2f} {result.throughput_ops_per_sec:<20.1f}")

    @staticmethod
    def _save_results(results: List[BenchmarkResult], output_path: str, verbose: bool) -> None:
        """Save results to JSON file."""
        if verbose:
            print(f"💾 Saving results to: {output_path}")

        # Convert results to serializable format
        results_dict = {
            'benchmark_results': [
                {
                    'name': r.name,
                    'mean_time_ms': r.mean_time_ms,
                    'std_time_ms': r.std_time_ms,
                    'throughput_ops_per_sec': r.throughput_ops_per_sec,
                    'memory_usage_mb': r.memory_usage_mb,
                    'gpu_utilization_percent': r.gpu_utilization_percent
                }
                for r in results
            ],
            'timestamp': time.time(),
            'device': str(torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')
        }

        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        if verbose:
            print(f"   Results saved ({len(results)} benchmarks)")


def main():
    """Standalone entry point for kpt-benchmark."""
    parser = argparse.ArgumentParser(
        prog='kpt-benchmark',
        description='Run comprehensive performance benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add benchmark command arguments directly
    parser.add_argument(
        '--model',
        type=str,
        help='Model to benchmark (file path or predefined name)'
    )
    parser.add_argument(
        '--type',
        choices=['model', 'compare', 'regression', 'stress'],
        default='model',
        help='Benchmark type (default: model)'
    )
    parser.add_argument(
        '--levels',
        type=str,
        default='basic,compile',
        help='Optimization levels to compare (comma-separated)'
    )
    parser.add_argument(
        '--batch-sizes',
        type=str,
        default='1,8,16',
        help='Batch sizes for stress testing (comma-separated)'
    )
    parser.add_argument(
        '--input-shape',
        type=str,
        help='Input tensor shape (e.g., "1,3,224,224")'
    )
    parser.add_argument(
        '--predefined',
        choices=['transformers', 'vision', 'optimization'],
        help='Run predefined benchmark suite'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick benchmark (fewer runs for faster results)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=10,
        help='Number of warmup runs (default: 10)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=100,
        help='Number of benchmark runs (default: 100, 20 if --quick)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for results (JSON format)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()
    return BenchmarkCommand.execute(args)


if __name__ == '__main__':
    sys.exit(main())