"""
Performance Benchmarking Suite for GPU Optimizations

Comprehensive benchmarking framework for testing and validating performance
improvements from compiler optimizations, kernel fusion, and memory optimizations.
"""

import time
import logging
import statistics
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import psutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks"""
    KERNEL_PERFORMANCE = "kernel_performance"
    COMPILER_OPTIMIZATION = "compiler_optimization"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    TENSOR_OPERATIONS = "tensor_operations"
    END_TO_END_MODEL = "end_to_end_model"


class MetricType(Enum):
    """Performance metrics to collect"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"
    POWER_CONSUMPTION = "power_consumption"
    ACCURACY = "accuracy"


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    warmup_iterations: int = 10
    measurement_iterations: int = 100
    timeout_seconds: int = 300
    enable_profiling: bool = True
    profile_memory: bool = True
    profile_compute: bool = True
    collect_traces: bool = False
    statistical_significance: float = 0.95


@dataclass
class BenchmarkResult:
    """Results from benchmark execution"""
    benchmark_name: str
    benchmark_type: BenchmarkType
    metrics: Dict[MetricType, List[float]]
    statistics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class ComparisonResult:
    """Results from comparing two benchmarks"""
    baseline_result: BenchmarkResult
    optimized_result: BenchmarkResult
    improvements: Dict[MetricType, float]  # Percentage improvements
    statistical_significance: Dict[MetricType, bool]
    regression_detected: bool = False


class SystemProfiler:
    """System resource profiler for benchmarks"""

    def __init__(self):
        self.monitoring_active = False
        self.cpu_usage_history = []
        self.memory_usage_history = []
        self.gpu_usage_history = []
        self.power_usage_history = []

    def start_profiling(self):
        """Start system profiling"""
        self.monitoring_active = True
        self.cpu_usage_history.clear()
        self.memory_usage_history.clear()
        self.gpu_usage_history.clear()
        self.power_usage_history.clear()

        self.profile_thread = threading.Thread(target=self._profile_loop, daemon=True)
        self.profile_thread.start()

    def stop_profiling(self):
        """Stop system profiling"""
        self.monitoring_active = False
        if hasattr(self, 'profile_thread'):
            self.profile_thread.join(timeout=1.0)

    def _profile_loop(self):
        """Main profiling loop"""
        while self.monitoring_active:
            try:
                # CPU and memory usage
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()

                self.cpu_usage_history.append(cpu_percent)
                self.memory_usage_history.append(memory_info.percent)

                # GPU usage (if available)
                gpu_usage = self._get_gpu_usage()
                if gpu_usage is not None:
                    self.gpu_usage_history.append(gpu_usage)

                time.sleep(0.1)  # Sample every 100ms

            except Exception as e:
                logger.warning(f"Profiling error: {e}")
                time.sleep(0.5)

    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU utilization using nvidia-smi"""
        try:
            if torch.cuda.is_available():
                # Try using pytorch
                return torch.cuda.utilization()
            else:
                # Fallback to nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    return float(result.stdout.strip().split('\n')[0])
        except Exception:
            pass
        return None

    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profiling summary"""
        summary = {}

        if self.cpu_usage_history:
            summary['cpu'] = {
                'avg_usage_percent': np.mean(self.cpu_usage_history),
                'max_usage_percent': np.max(self.cpu_usage_history),
                'min_usage_percent': np.min(self.cpu_usage_history)
            }

        if self.memory_usage_history:
            summary['memory'] = {
                'avg_usage_percent': np.mean(self.memory_usage_history),
                'max_usage_percent': np.max(self.memory_usage_history),
                'min_usage_percent': np.min(self.memory_usage_history)
            }

        if self.gpu_usage_history:
            summary['gpu'] = {
                'avg_usage_percent': np.mean(self.gpu_usage_history),
                'max_usage_percent': np.max(self.gpu_usage_history),
                'min_usage_percent': np.min(self.gpu_usage_history)
            }

        return summary


class KernelBenchmark:
    """
    Benchmark for individual kernel performance

    Tests kernel execution time, memory bandwidth utilization,
    and compute throughput for various kernel implementations.
    """

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.profiler = SystemProfiler()

    def benchmark_kernel(
        self,
        kernel_fn: Callable,
        input_tensors: List[torch.Tensor],
        kernel_name: str,
        expected_output: Optional[torch.Tensor] = None
    ) -> BenchmarkResult:
        """
        Benchmark a single kernel function

        Args:
            kernel_fn: Kernel function to benchmark
            input_tensors: Input tensors for the kernel
            kernel_name: Name for the benchmark
            expected_output: Expected output for correctness verification

        Returns:
            BenchmarkResult with performance metrics
        """
        logger.info(f"Benchmarking kernel: {kernel_name}")

        result = BenchmarkResult(
            benchmark_name=kernel_name,
            benchmark_type=BenchmarkType.KERNEL_PERFORMANCE,
            metrics={metric: [] for metric in MetricType}
        )

        try:
            # Start profiling
            if self.config.enable_profiling:
                self.profiler.start_profiling()

            # Warmup phase
            logger.debug(f"Warmup phase: {self.config.warmup_iterations} iterations")
            for _ in range(self.config.warmup_iterations):
                with torch.no_grad():
                    _ = kernel_fn(*input_tensors)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            # Measurement phase
            logger.debug(f"Measurement phase: {self.config.measurement_iterations} iterations")
            latencies = []
            memory_usages = []

            for i in range(self.config.measurement_iterations):
                # Memory usage before
                if torch.cuda.is_available():
                    memory_before = torch.cuda.memory_allocated()
                else:
                    memory_before = 0

                # Time kernel execution
                start_time = time.perf_counter()

                with torch.no_grad():
                    output = kernel_fn(*input_tensors)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to ms

                # Memory usage after
                if torch.cuda.is_available():
                    memory_after = torch.cuda.memory_allocated()
                    memory_used = (memory_after - memory_before) / (1024 ** 2)  # MB
                else:
                    memory_used = 0

                latencies.append(execution_time)
                memory_usages.append(memory_used)

                # Verify correctness (first iteration only)
                if i == 0 and expected_output is not None:
                    if not self._verify_output(output, expected_output):
                        result.success = False
                        result.error_message = "Output verification failed"
                        return result

            # Stop profiling
            if self.config.enable_profiling:
                self.profiler.stop_profiling()

            # Store metrics
            result.metrics[MetricType.LATENCY] = latencies
            result.metrics[MetricType.MEMORY_USAGE] = memory_usages

            # Calculate throughput (operations per second)
            avg_latency_s = np.mean(latencies) / 1000.0
            throughput = 1.0 / avg_latency_s if avg_latency_s > 0 else 0.0
            result.metrics[MetricType.THROUGHPUT] = [throughput] * len(latencies)

            # Calculate statistics
            result.statistics = self._calculate_statistics(result.metrics)

            # Add metadata
            result.metadata = {
                'input_shapes': [list(tensor.shape) for tensor in input_tensors],
                'input_dtypes': [str(tensor.dtype) for tensor in input_tensors],
                'device': str(input_tensors[0].device) if input_tensors else 'cpu',
                'system_profile': self.profiler.get_profile_summary()
            }

            result.success = True
            logger.info(f"Kernel benchmark completed: {kernel_name}")

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Kernel benchmark failed: {kernel_name}, Error: {e}")

        return result

    def _verify_output(
        self,
        actual_output: torch.Tensor,
        expected_output: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """Verify kernel output correctness"""
        try:
            return torch.allclose(actual_output, expected_output, rtol=rtol, atol=atol)
        except Exception:
            return False

    def _calculate_statistics(self, metrics: Dict[MetricType, List[float]]) -> Dict[str, float]:
        """Calculate statistical summary of metrics"""
        stats = {}

        for metric_type, values in metrics.items():
            if not values:
                continue

            prefix = metric_type.value
            stats[f'{prefix}_mean'] = np.mean(values)
            stats[f'{prefix}_median'] = np.median(values)
            stats[f'{prefix}_std'] = np.std(values)
            stats[f'{prefix}_min'] = np.min(values)
            stats[f'{prefix}_max'] = np.max(values)
            stats[f'{prefix}_p95'] = np.percentile(values, 95)
            stats[f'{prefix}_p99'] = np.percentile(values, 99)

        return stats


class CompilerBenchmark:
    """
    Benchmark for compiler optimization effectiveness

    Compares performance before and after compiler optimizations
    including kernel fusion, memory optimization, and code generation.
    """

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.kernel_benchmark = KernelBenchmark(config)

    def compare_optimizations(
        self,
        baseline_fn: Callable,
        optimized_fn: Callable,
        input_tensors: List[torch.Tensor],
        benchmark_name: str
    ) -> ComparisonResult:
        """
        Compare baseline vs optimized implementation

        Args:
            baseline_fn: Baseline implementation
            optimized_fn: Optimized implementation
            input_tensors: Input tensors for both implementations
            benchmark_name: Name for the benchmark

        Returns:
            ComparisonResult with performance comparison
        """
        logger.info(f"Comparing optimizations: {benchmark_name}")

        # Benchmark baseline
        baseline_result = self.kernel_benchmark.benchmark_kernel(
            baseline_fn, input_tensors, f"{benchmark_name}_baseline"
        )

        # Benchmark optimized
        optimized_result = self.kernel_benchmark.benchmark_kernel(
            optimized_fn, input_tensors, f"{benchmark_name}_optimized"
        )

        # Calculate improvements
        improvements = {}
        statistical_significance = {}
        regression_detected = False

        for metric_type in MetricType:
            if (metric_type in baseline_result.metrics and
                metric_type in optimized_result.metrics and
                baseline_result.metrics[metric_type] and
                optimized_result.metrics[metric_type]):

                baseline_values = baseline_result.metrics[metric_type]
                optimized_values = optimized_result.metrics[metric_type]

                baseline_mean = np.mean(baseline_values)
                optimized_mean = np.mean(optimized_values)

                # Calculate improvement (positive = better)
                if metric_type == MetricType.LATENCY:
                    # For latency, lower is better
                    improvement = ((baseline_mean - optimized_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0.0
                elif metric_type == MetricType.MEMORY_USAGE:
                    # For memory, lower is better
                    improvement = ((baseline_mean - optimized_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0.0
                else:
                    # For throughput, utilization, etc., higher is better
                    improvement = ((optimized_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0.0

                improvements[metric_type] = improvement

                # Statistical significance test
                is_significant = self._test_statistical_significance(
                    baseline_values, optimized_values
                )
                statistical_significance[metric_type] = is_significant

                # Check for regression
                if improvement < -5.0 and is_significant:  # 5% regression threshold
                    regression_detected = True

        comparison_result = ComparisonResult(
            baseline_result=baseline_result,
            optimized_result=optimized_result,
            improvements=improvements,
            statistical_significance=statistical_significance,
            regression_detected=regression_detected
        )

        logger.info(f"Optimization comparison completed: {benchmark_name}")
        return comparison_result

    def _test_statistical_significance(
        self,
        baseline_values: List[float],
        optimized_values: List[float]
    ) -> bool:
        """Test statistical significance using t-test"""
        try:
            from scipy import stats
            _, p_value = stats.ttest_ind(baseline_values, optimized_values)
            return p_value < (1.0 - self.config.statistical_significance)
        except ImportError:
            # Fallback: simple variance comparison
            baseline_mean = np.mean(baseline_values)
            optimized_mean = np.mean(optimized_values)
            baseline_cv = np.std(baseline_values) / baseline_mean if baseline_mean != 0 else 0.0
            optimized_cv = np.std(optimized_values) / optimized_mean if optimized_mean != 0 else 0.0
            return baseline_cv < 0.1 and optimized_cv < 0.1  # Low variance indicates reliable results


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite

    Orchestrates multiple types of benchmarks and provides
    unified reporting and analysis of performance improvements.
    """

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.kernel_benchmark = KernelBenchmark(config)
        self.compiler_benchmark = CompilerBenchmark(config)

        # Benchmark registry
        self.benchmark_results: List[BenchmarkResult] = []
        self.comparison_results: List[ComparisonResult] = []

    def add_kernel_benchmark(
        self,
        kernel_fn: Callable,
        input_generator: Callable[[], List[torch.Tensor]],
        benchmark_name: str,
        expected_output_fn: Optional[Callable] = None
    ):
        """Add a kernel benchmark to the suite"""
        input_tensors = input_generator()
        expected_output = expected_output_fn(*input_tensors) if expected_output_fn else None

        result = self.kernel_benchmark.benchmark_kernel(
            kernel_fn, input_tensors, benchmark_name, expected_output
        )

        self.benchmark_results.append(result)
        return result

    def add_optimization_comparison(
        self,
        baseline_fn: Callable,
        optimized_fn: Callable,
        input_generator: Callable[[], List[torch.Tensor]],
        benchmark_name: str
    ):
        """Add an optimization comparison to the suite"""
        input_tensors = input_generator()

        comparison = self.compiler_benchmark.compare_optimizations(
            baseline_fn, optimized_fn, input_tensors, benchmark_name
        )

        self.comparison_results.append(comparison)
        return comparison

    def run_predefined_benchmarks(self) -> Dict[str, Any]:
        """Run a set of predefined benchmarks"""
        results = {}

        # Matrix multiplication benchmarks
        results['matmul'] = self._benchmark_matrix_operations()

        # Attention mechanism benchmarks
        results['attention'] = self._benchmark_attention_operations()

        # Memory bandwidth benchmarks
        results['memory'] = self._benchmark_memory_operations()

        # Compiler optimization benchmarks
        results['optimizations'] = self._benchmark_compiler_optimizations()

        return results

    def _benchmark_matrix_operations(self) -> Dict[str, Any]:
        """Benchmark matrix multiplication operations"""
        results = []

        matrix_sizes = [(64, 64), (128, 128), (256, 256)]  # Reduced sizes for testing

        for m, n in matrix_sizes:
            k = n  # Square matrices

            def input_generator():
                a = torch.randn(m, k, device='cuda' if torch.cuda.is_available() else 'cpu')
                b = torch.randn(k, n, device=a.device)
                return [a, b]

            def matmul_kernel(a, b):
                return torch.mm(a, b)

            result = self.add_kernel_benchmark(
                matmul_kernel,
                input_generator,
                f"matmul_{m}x{k}x{n}"
            )

            results.append({
                'size': f"{m}x{k}x{n}",
                'latency_ms': result.statistics.get('latency_mean', 0),
                'throughput_ops': result.statistics.get('throughput_mean', 0),
                'memory_mb': result.statistics.get('memory_usage_mean', 0)
            })

        return {'matrix_multiplication': results}

    def _benchmark_attention_operations(self) -> Dict[str, Any]:
        """Benchmark attention mechanism operations"""
        results = []

        configs = [
            (64, 4, 16),   # seq_len, num_heads, head_dim - much smaller for testing
            (128, 4, 16),
            (256, 8, 32),
        ]

        for seq_len, num_heads, head_dim in configs:
            batch_size = 2  # Reduced batch size for testing

            def input_generator():
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
                k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
                v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device)
                return [q, k, v]

            def attention_kernel(q, k, v):
                # Simplified attention computation
                scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / (head_dim ** 0.5)
                attn = torch.softmax(scores, dim=-1)
                output = torch.einsum('bhqk,bvhd->bqhd', attn, v)
                return output

            result = self.add_kernel_benchmark(
                attention_kernel,
                input_generator,
                f"attention_{seq_len}x{num_heads}x{head_dim}"
            )

            results.append({
                'config': f"{seq_len}x{num_heads}x{head_dim}",
                'latency_ms': result.statistics.get('latency_mean', 0),
                'throughput_ops': result.statistics.get('throughput_mean', 0),
                'memory_mb': result.statistics.get('memory_usage_mean', 0)
            })

        return {'attention_mechanisms': results}

    def _benchmark_memory_operations(self) -> Dict[str, Any]:
        """Benchmark memory bandwidth operations"""
        results = []

        data_sizes = [1024, 4*1024, 16*1024]  # Much smaller sizes for testing

        for size_bytes in data_sizes:
            num_elements = size_bytes // 4  # Assuming float32

            def input_generator():
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                data = torch.randn(num_elements, device=device)
                return [data]

            def copy_kernel(data):
                return data.clone()

            result = self.add_kernel_benchmark(
                copy_kernel,
                input_generator,
                f"memory_copy_{size_bytes//1024}KB"
            )

            # Calculate bandwidth
            latency_s = (result.statistics.get('latency_mean', 1) / 1000.0)
            bandwidth_gb_s = (size_bytes / (1024**3)) / latency_s if latency_s > 0 else 0

            results.append({
                'size_kb': size_bytes // 1024,
                'latency_ms': result.statistics.get('latency_mean', 0),
                'bandwidth_gb_s': bandwidth_gb_s,
                'memory_mb': result.statistics.get('memory_usage_mean', 0)
            })

        return {'memory_bandwidth': results}

    def _benchmark_compiler_optimizations(self) -> Dict[str, Any]:
        """Benchmark compiler optimization effectiveness"""
        results = []

        # Example: Compare fused vs unfused operations
        def input_generator():
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = torch.randn(128, 128, device=device)  # Reduced size for testing
            return [x]

        def unfused_gelu(x):
            # Unfused GELU implementation
            return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

        def fused_gelu(x):
            # Use PyTorch's optimized GELU
            return torch.nn.functional.gelu(x)

        comparison = self.add_optimization_comparison(
            unfused_gelu,
            fused_gelu,
            input_generator,
            "gelu_fusion"
        )

        results.append({
            'optimization': 'gelu_fusion',
            'latency_improvement': comparison.improvements.get(MetricType.LATENCY, 0),
            'throughput_improvement': comparison.improvements.get(MetricType.THROUGHPUT, 0),
            'memory_improvement': comparison.improvements.get(MetricType.MEMORY_USAGE, 0),
            'statistically_significant': any(comparison.statistical_significance.values())
        })

        return {'compiler_optimizations': results}

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        report = {
            'summary': {
                'total_benchmarks': len(self.benchmark_results),
                'total_comparisons': len(self.comparison_results),
                'successful_benchmarks': len([r for r in self.benchmark_results if r.success]),
                'regressions_detected': len([c for c in self.comparison_results if c.regression_detected])
            },
            'benchmark_results': [],
            'optimization_results': [],
            'performance_insights': self._analyze_performance_insights()
        }

        # Add individual benchmark results
        for result in self.benchmark_results:
            report['benchmark_results'].append({
                'name': result.benchmark_name,
                'type': result.benchmark_type.value,
                'success': result.success,
                'statistics': result.statistics,
                'metadata': result.metadata
            })

        # Add optimization comparison results
        for comparison in self.comparison_results:
            report['optimization_results'].append({
                'name': comparison.baseline_result.benchmark_name.replace('_baseline', ''),
                'improvements': {k.value: v for k, v in comparison.improvements.items()},
                'statistical_significance': {k.value: v for k, v in comparison.statistical_significance.items()},
                'regression_detected': comparison.regression_detected
            })

        return report

    def _analyze_performance_insights(self) -> Dict[str, Any]:
        """Analyze performance insights across all benchmarks"""
        insights = {
            'best_performing_benchmarks': [],
            'optimization_effectiveness': {},
            'resource_utilization': {},
            'recommendations': []
        }

        # Find best performing benchmarks
        successful_results = [r for r in self.benchmark_results if r.success]
        if successful_results:
            # Sort by throughput
            sorted_by_throughput = sorted(
                successful_results,
                key=lambda r: r.statistics.get('throughput_mean', 0),
                reverse=True
            )
            insights['best_performing_benchmarks'] = [
                {
                    'name': r.benchmark_name,
                    'throughput': r.statistics.get('throughput_mean', 0),
                    'latency': r.statistics.get('latency_mean', 0)
                }
                for r in sorted_by_throughput[:5]
            ]

        # Analyze optimization effectiveness
        if self.comparison_results:
            improvements = []
            for comparison in self.comparison_results:
                for metric_type, improvement in comparison.improvements.items():
                    improvements.append(improvement)

            insights['optimization_effectiveness'] = {
                'average_improvement_percent': np.mean(improvements),
                'best_improvement_percent': np.max(improvements),
                'worst_improvement_percent': np.min(improvements),
                'optimizations_with_regression': len([c for c in self.comparison_results if c.regression_detected])
            }

        return insights

    def export_results(self, filepath: str):
        """Export benchmark results to file"""
        report = self.generate_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Benchmark results exported to {filepath}")


def create_benchmark_suite(
    warmup_iterations: int = 10,
    measurement_iterations: int = 100,
    enable_profiling: bool = True
) -> PerformanceBenchmarkSuite:
    """
    Factory function to create benchmark suite

    Args:
        warmup_iterations: Number of warmup iterations
        measurement_iterations: Number of measurement iterations
        enable_profiling: Enable system resource profiling

    Returns:
        Configured PerformanceBenchmarkSuite
    """
    config = BenchmarkConfig(
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        enable_profiling=enable_profiling,
        profile_memory=True,
        profile_compute=True
    )

    return PerformanceBenchmarkSuite(config)