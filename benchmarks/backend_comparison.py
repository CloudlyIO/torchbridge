#!/usr/bin/env python3
"""
Backend Comparison Benchmarks (v0.4.8)

Benchmarks for comparing performance across unified backends (NVIDIA, AMD, TPU,
Intel, CPU) using the standardized BaseBackend interface.

This benchmark measures:
- Backend initialization time
- Model preparation time
- Inference latency
- Memory usage
- Optimization overhead

Usage:
    PYTHONPATH=src python3 benchmarks/backend_comparison.py
    PYTHONPATH=src python3 benchmarks/backend_comparison.py --quick
    PYTHONPATH=src python3 benchmarks/backend_comparison.py --warmup 5 --iterations 20
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json

import torch
import torch.nn as nn


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    backend: str
    metric: str
    value: float
    unit: str
    iterations: int = 1
    std_dev: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'backend': self.backend,
            'metric': self.metric,
            'value': self.value,
            'unit': self.unit,
            'iterations': self.iterations,
            'std_dev': self.std_dev,
            'metadata': self.metadata,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    results: List[BenchmarkResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: BenchmarkResult):
        self.results.append(result)

    def get_by_backend(self, backend: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.backend == backend]

    def get_by_metric(self, metric: str) -> List[BenchmarkResult]:
        return [r for r in self.results if r.metric == metric]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'metadata': self.metadata,
            'results': [r.to_dict() for r in self.results],
        }

    def print_summary(self):
        """Print formatted summary of results."""
        print("\n" + "=" * 70)
        print("  Backend Comparison Benchmark Results")
        print("=" * 70)

        # Group by metric
        metrics = set(r.metric for r in self.results)

        for metric in sorted(metrics):
            print(f"\n  {metric}:")
            print("  " + "-" * 50)

            metric_results = self.get_by_metric(metric)
            metric_results.sort(key=lambda x: x.value)

            for r in metric_results:
                std_str = f" (±{r.std_dev:.2f})" if r.std_dev > 0 else ""
                print(f"    {r.backend:15} {r.value:10.3f} {r.unit}{std_str}")

        print("\n" + "=" * 70)


def create_test_model(size: str = "medium") -> nn.Module:
    """Create a test model for benchmarking."""
    sizes = {
        "small": (128, 256, 2),
        "medium": (256, 512, 4),
        "large": (512, 1024, 6),
    }

    hidden, intermediate, layers = sizes.get(size, sizes["medium"])

    class BenchmarkModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList()
            for i in range(layers):
                in_dim = hidden if i > 0 else hidden
                self.layers.append(nn.Sequential(
                    nn.Linear(in_dim, intermediate),
                    nn.GELU(),
                    nn.Linear(intermediate, hidden),
                    nn.LayerNorm(hidden),
                ))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x) + x  # Residual
            return x

    return BenchmarkModel()


def benchmark_initialization(suite: BenchmarkSuite, backends: List[str], iterations: int = 5):
    """Benchmark backend initialization time."""
    from torchbridge.backends import BackendFactory, BackendType

    print("\n  Benchmarking initialization time...")

    for backend_name in backends:
        times = []

        for _ in range(iterations):
            gc.collect()

            start = time.perf_counter()
            try:
                backend_type = BackendType.from_string(backend_name)
                backend = BackendFactory.create(backend_type)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
                del backend
            except Exception as e:
                print(f"    {backend_name}: Failed - {e}")
                break

        if times:
            avg_time = sum(times) / len(times)
            std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

            suite.add(BenchmarkResult(
                name="initialization",
                backend=backend_name,
                metric="Initialization Time",
                value=avg_time,
                unit="ms",
                iterations=iterations,
                std_dev=std_dev,
            ))
            print(f"    {backend_name}: {avg_time:.2f}ms (±{std_dev:.2f})")


def benchmark_model_preparation(
    suite: BenchmarkSuite,
    backends: List[str],
    model_size: str = "medium",
    iterations: int = 5
):
    """Benchmark model preparation time."""
    from torchbridge.backends import BackendFactory, BackendType, OptimizationLevel

    print("\n  Benchmarking model preparation...")

    for backend_name in backends:
        times = []

        try:
            backend_type = BackendType.from_string(backend_name)
            backend = BackendFactory.create(backend_type)

            for _ in range(iterations):
                model = create_test_model(model_size)
                gc.collect()

                start = time.perf_counter()
                prepared = backend.prepare_model(model, OptimizationLevel.O2)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

                del prepared, model

            avg_time = sum(times) / len(times)
            std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

            suite.add(BenchmarkResult(
                name="model_preparation",
                backend=backend_name,
                metric="Model Preparation Time",
                value=avg_time,
                unit="ms",
                iterations=iterations,
                std_dev=std_dev,
                metadata={'model_size': model_size},
            ))
            print(f"    {backend_name}: {avg_time:.2f}ms (±{std_dev:.2f})")

        except Exception as e:
            print(f"    {backend_name}: Failed - {e}")


def benchmark_inference_latency(
    suite: BenchmarkSuite,
    backends: List[str],
    model_size: str = "medium",
    batch_size: int = 8,
    warmup: int = 3,
    iterations: int = 10
):
    """Benchmark inference latency."""
    from torchbridge.backends import BackendFactory, BackendType

    print("\n  Benchmarking inference latency...")

    hidden_sizes = {"small": 128, "medium": 256, "large": 512}
    hidden = hidden_sizes.get(model_size, 256)

    for backend_name in backends:
        try:
            backend_type = BackendType.from_string(backend_name)
            backend = BackendFactory.create(backend_type)

            model = create_test_model(model_size)
            model = backend.optimize_for_inference(model)

            x = backend.to_device(torch.randn(batch_size, 64, hidden))

            # Warmup
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(x)
                    backend.synchronize()

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(iterations):
                    backend.synchronize()
                    start = time.perf_counter()
                    _ = model(x)
                    backend.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)

            avg_time = sum(times) / len(times)
            std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

            suite.add(BenchmarkResult(
                name="inference_latency",
                backend=backend_name,
                metric="Inference Latency",
                value=avg_time,
                unit="ms",
                iterations=iterations,
                std_dev=std_dev,
                metadata={
                    'model_size': model_size,
                    'batch_size': batch_size,
                    'warmup': warmup,
                },
            ))
            print(f"    {backend_name}: {avg_time:.2f}ms (±{std_dev:.2f})")

            del model, x

        except Exception as e:
            print(f"    {backend_name}: Failed - {e}")


def benchmark_optimization_overhead(
    suite: BenchmarkSuite,
    backends: List[str],
    model_size: str = "medium",
    iterations: int = 5
):
    """Benchmark optimization overhead for different levels."""
    from torchbridge.backends import BackendFactory, BackendType, OptimizationLevel

    print("\n  Benchmarking optimization overhead...")

    levels = [OptimizationLevel.O0, OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3]

    for backend_name in backends:
        try:
            backend_type = BackendType.from_string(backend_name)
            backend = BackendFactory.create(backend_type)

            level_times = {}

            for level in levels:
                times = []

                for _ in range(iterations):
                    model = create_test_model(model_size)
                    gc.collect()

                    start = time.perf_counter()
                    _ = backend.prepare_model(model, level)
                    elapsed = (time.perf_counter() - start) * 1000
                    times.append(elapsed)

                avg_time = sum(times) / len(times)
                level_times[level.value] = avg_time

            # Report O2 as main metric (most common)
            suite.add(BenchmarkResult(
                name="optimization_overhead",
                backend=backend_name,
                metric="Optimization Overhead (O2)",
                value=level_times.get("O2", 0),
                unit="ms",
                iterations=iterations,
                metadata={'all_levels': level_times},
            ))

            levels_str = ", ".join(f"{k}:{v:.1f}ms" for k, v in level_times.items())
            print(f"    {backend_name}: {levels_str}")

        except Exception as e:
            print(f"    {backend_name}: Failed - {e}")


def benchmark_device_info_overhead(suite: BenchmarkSuite, backends: List[str], iterations: int = 100):
    """Benchmark device info retrieval overhead."""
    from torchbridge.backends import BackendFactory, BackendType

    print("\n  Benchmarking device info retrieval...")

    for backend_name in backends:
        try:
            backend_type = BackendType.from_string(backend_name)
            backend = BackendFactory.create(backend_type)

            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = backend.get_device_info()
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg_time = sum(times) / len(times)
            std_dev = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

            suite.add(BenchmarkResult(
                name="device_info",
                backend=backend_name,
                metric="Device Info Retrieval",
                value=avg_time,
                unit="ms",
                iterations=iterations,
                std_dev=std_dev,
            ))
            print(f"    {backend_name}: {avg_time:.4f}ms (±{std_dev:.4f})")

        except Exception as e:
            print(f"    {backend_name}: Failed - {e}")


def benchmark_throughput(
    suite: BenchmarkSuite,
    backends: List[str],
    model_size: str = "medium",
    batch_sizes: List[int] = None,
    duration_seconds: float = 2.0
):
    """Benchmark throughput (samples/second)."""
    from torchbridge.backends import BackendFactory, BackendType

    print("\n  Benchmarking throughput...")

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16]

    hidden_sizes = {"small": 128, "medium": 256, "large": 512}
    hidden = hidden_sizes.get(model_size, 256)

    for backend_name in backends:
        try:
            backend_type = BackendType.from_string(backend_name)
            backend = BackendFactory.create(backend_type)

            model = create_test_model(model_size)
            model = backend.optimize_for_inference(model)

            throughputs = {}

            for batch_size in batch_sizes:
                x = backend.to_device(torch.randn(batch_size, 64, hidden))

                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(x)
                        backend.synchronize()

                # Measure
                samples = 0
                backend.synchronize()
                start = time.perf_counter()

                with torch.no_grad():
                    while (time.perf_counter() - start) < duration_seconds:
                        _ = model(x)
                        samples += batch_size

                backend.synchronize()
                elapsed = time.perf_counter() - start
                throughput = samples / elapsed
                throughputs[batch_size] = throughput

            # Report best throughput
            best_batch, best_throughput = max(throughputs.items(), key=lambda x: x[1])

            suite.add(BenchmarkResult(
                name="throughput",
                backend=backend_name,
                metric="Throughput",
                value=best_throughput,
                unit="samples/s",
                metadata={
                    'best_batch_size': best_batch,
                    'all_throughputs': throughputs,
                    'model_size': model_size,
                },
            ))
            print(f"    {backend_name}: {best_throughput:.1f} samples/s (batch={best_batch})")

        except Exception as e:
            print(f"    {backend_name}: Failed - {e}")


def get_available_backends() -> List[str]:
    """Get list of available backends for benchmarking."""
    from torchbridge.backends import BackendFactory, BackendType

    available = ["cpu"]  # CPU is always available

    backend_checks = [
        ("nvidia", BackendType.NVIDIA),
        ("amd", BackendType.AMD),
        ("tpu", BackendType.TPU),
        ("intel", BackendType.INTEL),
    ]

    for name, backend_type in backend_checks:
        try:
            backend = BackendFactory.create(backend_type)
            if backend.is_available:
                available.append(name)
        except Exception:
            pass

    return available


def main():
    """Run backend comparison benchmarks."""
    parser = argparse.ArgumentParser(description="Backend Comparison Benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmarks")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--backends", nargs="+", help="Specific backends to benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()

    # Adjust for quick mode
    if args.quick:
        args.warmup = 2
        args.iterations = 5
        args.model_size = "small"

    print("\n" + "=" * 60)
    print("  Backend Comparison Benchmarks (v0.4.8)")
    print("  TorchBridge - Unified Backend Performance")
    print("=" * 60)

    # Detect available backends
    available = get_available_backends()
    backends = args.backends if args.backends else available

    print(f"\n  Available backends: {', '.join(available)}")
    print(f"  Benchmarking: {', '.join(backends)}")
    print(f"  Model size: {args.model_size}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")

    # Create benchmark suite
    suite = BenchmarkSuite(metadata={
        'version': '0.4.8',
        'model_size': args.model_size,
        'warmup': args.warmup,
        'iterations': args.iterations,
        'backends': backends,
        'torch_version': torch.__version__,
    })

    # Run benchmarks
    benchmark_initialization(suite, backends, iterations=args.iterations)
    benchmark_model_preparation(suite, backends, args.model_size, iterations=args.iterations)
    benchmark_inference_latency(
        suite, backends, args.model_size,
        warmup=args.warmup, iterations=args.iterations
    )
    benchmark_optimization_overhead(suite, backends, args.model_size, iterations=args.iterations)
    benchmark_device_info_overhead(suite, backends, iterations=args.iterations * 10)

    if not args.quick:
        benchmark_throughput(suite, backends, args.model_size)

    # Print summary
    suite.print_summary()

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\n  Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
