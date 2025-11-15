"""
Profiling and Benchmarking Utilities

This module provides comprehensive tools for profiling kernel performance,
memory usage, and computational efficiency of the optimized components.
"""

import torch
import time
import psutil
import gc
from typing import Dict, List, Optional, Callable, Any, Tuple
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import os


class KernelProfiler:
    """
    Comprehensive profiler for analyzing kernel performance and optimization effectiveness.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = defaultdict(list)
        self.memory_stats = []
        self.timing_data = {}

    @contextmanager
    def profile(self, operation_name: str):
        """
        Context manager for profiling operations with automatic cleanup and synchronization.
        """
        # Cleanup before profiling
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Memory before
        if self.device == "cuda" and torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = psutil.virtual_memory().used

        # Start timing
        start_time = time.perf_counter()

        try:
            yield
        finally:
            # End timing
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Memory after
            if self.device == "cuda" and torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
            else:
                memory_after = psutil.virtual_memory().used

            # Record results
            self.results[operation_name].append({
                "time": end_time - start_time,
                "memory_used": memory_after - memory_before,
                "peak_memory": torch.cuda.max_memory_allocated() if (self.device == "cuda" and torch.cuda.is_available()) else memory_after
            })

    def benchmark_function(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict = {},
        num_iterations: int = 100,
        warmup_iterations: int = 10,
        name: str = "operation"
    ) -> Dict[str, float]:
        """
        Benchmark a function with multiple iterations and statistical analysis.
        """
        times = []
        memory_usage = []

        # Warmup
        for _ in range(warmup_iterations):
            with self.profile(f"{name}_warmup"):
                result = func(*args, **kwargs)

        # Actual benchmarking
        for i in range(num_iterations):
            with self.profile(f"{name}_iteration_{i}"):
                result = func(*args, **kwargs)

        # Extract timing data for this operation
        operation_times = [r["time"] for r in self.results[f"{name}_iteration_0"]]
        operation_memory = [r["memory_used"] for r in self.results[f"{name}_iteration_0"]]

        # Calculate statistics
        stats = {
            "mean_time": np.mean(operation_times),
            "std_time": np.std(operation_times),
            "min_time": np.min(operation_times),
            "max_time": np.max(operation_times),
            "median_time": np.median(operation_times),
            "mean_memory": np.mean(operation_memory),
            "peak_memory": np.max(operation_memory)
        }

        return stats

    def compare_implementations(
        self,
        implementations: Dict[str, Callable],
        args: Tuple = (),
        kwargs: Dict = {},
        num_iterations: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple implementations of the same operation.
        """
        comparison_results = {}

        for name, impl in implementations.items():
            print(f"Benchmarking {name}...")
            stats = self.benchmark_function(
                impl, args, kwargs, num_iterations, name=name
            )
            comparison_results[name] = stats

        return comparison_results

    def plot_comparison(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        metric: str = "mean_time",
        title: str = "Performance Comparison"
    ):
        """
        Create visualization of performance comparison.
        """
        names = list(comparison_results.keys())
        values = [comparison_results[name][metric] for name in names]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, values)
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(title)
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def profile_memory_usage(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        num_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Profile detailed memory usage patterns during model execution.
        """
        if self.device != "cuda":
            print("Detailed memory profiling only available for CUDA")
            return {}

        memory_timeline = []

        def memory_hook(module, input, output):
            memory_timeline.append({
                "module": module.__class__.__name__,
                "memory_allocated": torch.cuda.memory_allocated(),
                "memory_reserved": torch.cuda.memory_reserved()
            })

        # Register hooks on all modules
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(memory_hook)
                hooks.append(hook)

        try:
            # Clear cache and reset stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run forward pass
            with torch.no_grad():
                _ = model(input_data)

            # Collect final stats
            peak_memory = torch.cuda.max_memory_allocated()
            memory_efficiency = len(memory_timeline) / peak_memory if peak_memory > 0 else 0

        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()

        return {
            "memory_timeline": memory_timeline,
            "peak_memory_mb": peak_memory / (1024 * 1024),
            "memory_efficiency": memory_efficiency,
            "num_allocations": len(memory_timeline)
        }

    def analyze_kernel_efficiency(
        self,
        operation: Callable,
        input_sizes: List[Tuple],
        operation_name: str = "kernel_analysis"
    ) -> Dict[str, Any]:
        """
        Analyze how kernel performance scales with input size.
        """
        scaling_results = []

        for size in input_sizes:
            # Create test input of specified size
            device = self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu"
            if len(size) == 1:
                test_input = torch.randn(size[0], device=device)
            elif len(size) == 2:
                test_input = torch.randn(size[0], size[1], device=device)
            elif len(size) == 3:
                test_input = torch.randn(size[0], size[1], size[2], device=device)
            else:
                test_input = torch.randn(*size, device=device)

            # Profile the operation
            stats = self.benchmark_function(
                operation,
                args=(test_input,),
                num_iterations=20,
                name=f"{operation_name}_{size}"
            )

            # Calculate throughput metrics
            num_elements = test_input.numel()
            throughput = num_elements / stats["mean_time"] / 1e9  # Billion elements per second

            scaling_results.append({
                "input_size": size,
                "num_elements": num_elements,
                "mean_time": stats["mean_time"],
                "throughput_gelements_per_sec": throughput,
                "memory_mb": stats["mean_memory"] / (1024 * 1024),
                "compute_intensity": num_elements / stats["mean_memory"] if stats["mean_memory"] > 0 else 0
            })

        return {
            "scaling_data": scaling_results,
            "analysis": self._analyze_scaling_pattern(scaling_results)
        }

    def _analyze_scaling_pattern(self, scaling_results: List[Dict]) -> Dict[str, str]:
        """
        Analyze scaling patterns to identify performance characteristics.
        """
        if len(scaling_results) < 3:
            return {"pattern": "insufficient_data"}

        # Extract data for analysis
        sizes = [r["num_elements"] for r in scaling_results]
        times = [r["mean_time"] for r in scaling_results]
        throughputs = [r["throughput_gelements_per_sec"] for r in scaling_results]

        # Simple linear regression to identify scaling pattern
        log_sizes = np.log(sizes)
        log_times = np.log(times)

        # Fit y = mx + b to log-log data
        coeffs = np.polyfit(log_sizes, log_times, 1)
        slope = coeffs[0]

        # Analyze scaling behavior
        if slope < 0.5:
            pattern = "sublinear_scaling"
            efficiency = "excellent"
        elif slope < 0.9:
            pattern = "near_linear_scaling"
            efficiency = "good"
        elif slope < 1.1:
            pattern = "linear_scaling"
            efficiency = "expected"
        elif slope < 1.5:
            pattern = "superlinear_scaling"
            efficiency = "concerning"
        else:
            pattern = "quadratic_or_worse"
            efficiency = "poor"

        # Check memory efficiency
        memory_sizes = [r["memory_mb"] for r in scaling_results]
        memory_growth = memory_sizes[-1] / memory_sizes[0] if memory_sizes[0] > 0 else float('inf')

        return {
            "pattern": pattern,
            "efficiency": efficiency,
            "time_scaling_slope": slope,
            "memory_growth_factor": memory_growth,
            "throughput_trend": "increasing" if throughputs[-1] > throughputs[0] else "decreasing"
        }

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive performance report.
        """
        report = []
        report.append("=== KERNEL OPTIMIZATION PERFORMANCE REPORT ===\n")

        # Summary statistics
        if self.results:
            report.append("OPERATION SUMMARY:")
            for operation, measurements in self.results.items():
                if measurements:
                    times = [m["time"] for m in measurements]
                    memories = [m["memory_used"] for m in measurements]

                    report.append(f"\n{operation}:")
                    report.append(f"  Average Time: {np.mean(times):.6f}s")
                    report.append(f"  Time Std Dev: {np.std(times):.6f}s")
                    report.append(f"  Average Memory: {np.mean(memories) / (1024*1024):.2f} MB")

        # Device information
        report.append(f"\nDEVICE INFORMATION:")
        report.append(f"  Device: {self.device}")
        if self.device == "cuda" and torch.cuda.is_available():
            report.append(f"  GPU: {torch.cuda.get_device_name()}")
            report.append(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        elif self.device == "cuda":
            report.append(f"  GPU: Not available (CUDA not compiled)")
        else:
            report.append(f"  CPU: {self.device}")

        # Recommendations
        report.append(f"\nOPTIMIZATION RECOMMENDATIONS:")
        report.extend(self._generate_recommendations())

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")

        return report_text

    def _generate_recommendations(self) -> List[str]:
        """
        Generate optimization recommendations based on profiling results.
        """
        recommendations = []

        # Analyze timing patterns
        if self.results:
            all_times = []
            for measurements in self.results.values():
                all_times.extend([m["time"] for m in measurements])

            if all_times:
                variance = np.var(all_times)
                mean_time = np.mean(all_times)

                if variance / mean_time > 0.1:
                    recommendations.append("  - High timing variance detected. Consider adding warmup iterations.")

                if mean_time > 0.1:
                    recommendations.append("  - Long execution times detected. Consider kernel fusion optimizations.")

        # Memory recommendations
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                current_usage = torch.cuda.memory_allocated()
                utilization = current_usage / total_memory

                if utilization > 0.8:
                    recommendations.append("  - High memory utilization. Consider gradient checkpointing or model sharding.")
                elif utilization < 0.3:
                    recommendations.append("  - Low memory utilization. Consider increasing batch size for better efficiency.")
            except:
                pass

        if not recommendations:
            recommendations.append("  - Performance appears optimal based on current analysis.")

        return recommendations


class ComparisonSuite:
    """
    Suite for comparing different optimization levels systematically.
    """

    def __init__(self):
        self.profiler = KernelProfiler()
        self.comparison_data = {}

    def compare_optimization_levels(
        self,
        model_factory: Callable,
        input_data: torch.Tensor,
        optimization_levels: List[str] = ["basic", "jit", "triton", "cuda"]
    ) -> Dict[str, Any]:
        """
        Compare different optimization levels systematically.
        """
        results = {}

        for level in optimization_levels:
            try:
                print(f"Testing {level} optimization...")

                # Create model for this optimization level
                model = model_factory(optimization_level=level)
                model.eval()

                # Profile the model
                with torch.no_grad():
                    stats = self.profiler.benchmark_function(
                        model,
                        args=(input_data,),
                        num_iterations=50,
                        name=f"{level}_model"
                    )

                results[level] = stats

            except Exception as e:
                print(f"Failed to test {level}: {e}")
                results[level] = {"error": str(e)}

        return results

    def memory_scaling_analysis(
        self,
        model_factory: Callable,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        sequence_lengths: List[int] = [128, 256, 512, 1024]
    ) -> Dict[str, Any]:
        """
        Analyze memory scaling characteristics across different input sizes.
        """
        scaling_data = []

        for batch_size in batch_sizes:
            for seq_len in sequence_lengths:
                try:
                    # Create test input
                    input_data = torch.randint(0, 1000, (batch_size, seq_len))

                    # Test memory usage
                    model = model_factory()
                    memory_stats = self.profiler.profile_memory_usage(model, input_data)

                    scaling_data.append({
                        "batch_size": batch_size,
                        "sequence_length": seq_len,
                        "total_elements": batch_size * seq_len,
                        **memory_stats
                    })

                except Exception as e:
                    print(f"Failed memory test for batch={batch_size}, seq={seq_len}: {e}")

        return {"scaling_data": scaling_data}


# Utility functions for common profiling tasks
def quick_benchmark(func: Callable, *args, **kwargs) -> Dict[str, float]:
    """
    Quick benchmark of a single function.
    """
    profiler = KernelProfiler()
    return profiler.benchmark_function(func, args, kwargs, num_iterations=20)


def compare_functions(functions: Dict[str, Callable], *args, **kwargs):
    """
    Quick comparison of multiple function implementations.
    """
    profiler = KernelProfiler()
    return profiler.compare_implementations(functions, args, kwargs)


def profile_model_inference(model: torch.nn.Module, input_data: torch.Tensor) -> str:
    """
    Quick profiling of model inference.
    """
    profiler = KernelProfiler()
    memory_stats = profiler.profile_memory_usage(model, input_data)
    timing_stats = profiler.benchmark_function(
        model, args=(input_data,), num_iterations=10
    )

    return profiler.generate_report()


# Example usage
if __name__ == "__main__":
    # Example: Profile a simple operation
    def test_operation(x):
        return torch.matmul(x, x.t())

    # Quick benchmark
    x = torch.randn(1000, 1000, device="cuda" if torch.cuda.is_available() else "cpu")
    result = quick_benchmark(test_operation, x)
    print(f"Matrix multiplication took {result['mean_time']:.6f}s on average")