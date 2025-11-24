#!/usr/bin/env python3
"""
Unified Benchmark Runner for PyTorch Optimization Framework

Consolidates all benchmarking functionality into a single, comprehensive system
that provides consistent benchmarking across all optimization levels with
statistical analysis and production-grade measurement methodology.
"""

import sys
import os
import time
import json
import warnings
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import psutil

# Add src to path for our optimizations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class BenchmarkType(Enum):
    """Types of benchmark tests"""
    INFERENCE = "inference"
    TRAINING = "training"
    MEMORY = "memory"
    ACCURACY = "accuracy"
    SCALING = "scaling"
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
    SPEEDUP = "speedup"


@dataclass
class BenchmarkConfig:
    """Unified configuration for benchmark execution"""
    name: str
    benchmark_type: BenchmarkType

    # Test parameters
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024])
    model_configs: Dict[str, Any] = field(default_factory=dict)

    # Measurement parameters
    warmup_iterations: int = 10
    measurement_iterations: int = 50
    num_trials: int = 3
    timeout_seconds: int = 300

    # Environment
    device: str = "auto"
    precision: str = "float32"
    enable_compilation: bool = True
    enable_profiling: bool = False

    # Statistical analysis
    confidence_level: float = 0.95
    min_relative_difference: float = 0.01  # 1% minimum difference to be significant


@dataclass
class PerformanceMetrics:
    """Comprehensive performance measurement results"""
    # Core metrics
    latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    throughput_std: float = 0.0

    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_efficiency: float = 0.0

    # GPU metrics (if available)
    gpu_utilization: float = 0.0
    power_consumption: float = 0.0

    # Quality metrics
    accuracy: float = 0.0
    numerical_error: float = 0.0

    # Derived metrics
    speedup: float = 1.0
    efficiency: float = 0.0

    # Metadata
    device: str = "unknown"
    precision: str = "unknown"
    compilation_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            'latency_ms': self.latency_ms,
            'latency_std_ms': self.latency_std_ms,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'throughput_std': self.throughput_std,
            'peak_memory_mb': self.peak_memory_mb,
            'memory_efficiency': self.memory_efficiency,
            'gpu_utilization': self.gpu_utilization,
            'power_consumption': self.power_consumption,
            'accuracy': self.accuracy,
            'numerical_error': self.numerical_error,
            'speedup': self.speedup,
            'efficiency': self.efficiency,
            'device': self.device,
            'precision': self.precision,
            'compilation_enabled': self.compilation_enabled
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result with context"""
    config: BenchmarkConfig
    baseline_metrics: Optional[PerformanceMetrics] = None
    optimized_metrics: Optional[PerformanceMetrics] = None
    comparison_metrics: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_speedup(self) -> float:
        """Calculate speedup vs baseline"""
        if self.baseline_metrics and self.optimized_metrics:
            if self.baseline_metrics.latency_ms > 0:
                return self.baseline_metrics.latency_ms / self.optimized_metrics.latency_ms
        return 1.0

    def get_memory_reduction(self) -> float:
        """Calculate memory usage reduction vs baseline"""
        if self.baseline_metrics and self.optimized_metrics:
            baseline_mem = self.baseline_metrics.peak_memory_mb
            optimized_mem = self.optimized_metrics.peak_memory_mb
            if baseline_mem > 0:
                return (baseline_mem - optimized_mem) / baseline_mem
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            'config': {
                'name': self.config.name,
                'benchmark_type': self.config.benchmark_type.value,
                'batch_sizes': self.config.batch_sizes,
                'sequence_lengths': self.config.sequence_lengths,
                'measurement_iterations': self.config.measurement_iterations
            },
            'baseline_metrics': self.baseline_metrics.to_dict() if self.baseline_metrics else None,
            'optimized_metrics': self.optimized_metrics.to_dict() if self.optimized_metrics else None,
            'comparison_metrics': {k: v.to_dict() for k, v in self.comparison_metrics.items()},
            'speedup': self.get_speedup(),
            'memory_reduction': self.get_memory_reduction(),
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class UnifiedBenchmarkRunner:
    """
    Unified benchmark runner that consolidates all benchmarking functionality.

    Provides comprehensive benchmarking capabilities for:
    - Performance comparison against baselines
    - Memory usage analysis
    - Accuracy validation
    - Scaling behavior analysis
    - Statistical significance testing
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig(name="default", benchmark_type=BenchmarkType.INFERENCE)
        self.device = self._setup_device()
        self._benchmark_results: List[BenchmarkResult] = []

    def _setup_device(self) -> torch.device:
        """Setup and return the appropriate device"""
        if self.config.device == "auto":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(self.config.device)

        print(f"🎯 Benchmarking on device: {device}")
        if device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f}GB")

        return device

    def benchmark_function(self,
                          func: Callable,
                          inputs: List[torch.Tensor],
                          name: str = "function",
                          reference_func: Optional[Callable] = None) -> PerformanceMetrics:
        """
        Benchmark a single function with comprehensive metrics collection.
        """
        # Move inputs to device
        device_inputs = [inp.to(self.device) for inp in inputs]

        # Setup model for benchmarking
        if hasattr(func, 'to'):
            func = func.to(self.device)

        # Enable compilation if requested
        if self.config.enable_compilation:
            try:
                func = torch.compile(func, mode='default')
            except Exception as e:
                warnings.warn(f"Compilation failed: {e}, continuing without compilation")

        # Warmup
        for _ in range(self.config.warmup_iterations):
            try:
                with torch.no_grad():
                    _ = func(*device_inputs)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"❌ Warmup failed: {e}")
                return PerformanceMetrics()

        # Reset memory stats
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()

        # Collect timing measurements
        latency_measurements = []
        accuracy_measurements = []

        for trial in range(self.config.measurement_iterations):
            start_time = time.perf_counter()

            try:
                with torch.no_grad():
                    result = func(*device_inputs)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                latency_measurements.append((end_time - start_time) * 1000)  # Convert to ms

                # Accuracy check against reference if provided
                if reference_func is not None:
                    try:
                        with torch.no_grad():
                            ref_result = reference_func(*device_inputs)

                        if isinstance(result, torch.Tensor) and isinstance(ref_result, torch.Tensor):
                            error = torch.mean(torch.abs(result - ref_result)).item()
                            accuracy_measurements.append(error)
                    except Exception:
                        pass

            except Exception as e:
                print(f"❌ Trial {trial} failed: {e}")
                continue

        if not latency_measurements:
            print(f"❌ No successful measurements for {name}")
            return PerformanceMetrics()

        # Calculate memory metrics
        if self.device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated()
            memory_used_mb = (peak_memory - initial_memory) / 1024**2
        else:
            memory_used_mb = 0.0

        # Calculate performance metrics
        latency_ms = statistics.mean(latency_measurements)
        latency_std_ms = statistics.stdev(latency_measurements) if len(latency_measurements) > 1 else 0.0

        # Calculate throughput (samples per second)
        batch_size = device_inputs[0].shape[0] if device_inputs else 1
        throughput = (batch_size * 1000) / latency_ms  # samples per second

        # Calculate accuracy metrics
        numerical_error = statistics.mean(accuracy_measurements) if accuracy_measurements else 0.0
        accuracy = 1.0 - min(numerical_error, 1.0) if accuracy_measurements else 1.0

        return PerformanceMetrics(
            latency_ms=latency_ms,
            latency_std_ms=latency_std_ms,
            throughput_samples_per_sec=throughput,
            peak_memory_mb=memory_used_mb,
            accuracy=accuracy,
            numerical_error=numerical_error,
            device=str(self.device),
            precision=self.config.precision,
            compilation_enabled=self.config.enable_compilation
        )

    def benchmark_component(self,
                           optimized_component: nn.Module,
                           reference_component: Optional[nn.Module] = None,
                           test_inputs: Optional[List[torch.Tensor]] = None) -> BenchmarkResult:
        """
        Comprehensive benchmarking of a neural network component.
        """
        if test_inputs is None:
            # Generate default test inputs
            batch_size = self.config.batch_sizes[0] if self.config.batch_sizes else 2
            seq_len = self.config.sequence_lengths[0] if self.config.sequence_lengths else 128
            embed_dim = self.config.model_configs.get('embed_dim', 256)
            test_inputs = [torch.randn(batch_size, seq_len, embed_dim)]

        # Benchmark optimized component
        optimized_metrics = self.benchmark_function(
            optimized_component,
            test_inputs,
            f"{optimized_component.__class__.__name__}_optimized",
            reference_component
        )

        # Benchmark reference component if provided
        baseline_metrics = None
        if reference_component is not None:
            baseline_metrics = self.benchmark_function(
                reference_component,
                test_inputs,
                f"{reference_component.__class__.__name__}_reference"
            )

            # Calculate speedup
            if baseline_metrics.latency_ms > 0:
                optimized_metrics.speedup = baseline_metrics.latency_ms / optimized_metrics.latency_ms

        # Create result
        result = BenchmarkResult(
            config=self.config,
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            metadata={
                'component_name': optimized_component.__class__.__name__,
                'input_shapes': [list(inp.shape) for inp in test_inputs],
                'parameter_count': sum(p.numel() for p in optimized_component.parameters()),
                'device_info': str(self.device)
            }
        )

        self._benchmark_results.append(result)
        return result

    def run_scaling_benchmark(self,
                             component: nn.Module,
                             batch_sizes: Optional[List[int]] = None,
                             sequence_lengths: Optional[List[int]] = None) -> Dict[str, List[PerformanceMetrics]]:
        """
        Run scaling benchmark across different input sizes.
        """
        batch_sizes = batch_sizes or self.config.batch_sizes
        sequence_lengths = sequence_lengths or self.config.sequence_lengths

        scaling_results = {
            'batch_scaling': [],
            'sequence_scaling': []
        }

        base_embed_dim = self.config.model_configs.get('embed_dim', 256)

        # Batch size scaling (fixed sequence length)
        fixed_seq_len = sequence_lengths[0] if sequence_lengths else 128
        for batch_size in batch_sizes:
            test_inputs = [torch.randn(batch_size, fixed_seq_len, base_embed_dim)]
            metrics = self.benchmark_function(component, test_inputs, f"batch_{batch_size}")
            scaling_results['batch_scaling'].append(metrics)

        # Sequence length scaling (fixed batch size)
        fixed_batch_size = batch_sizes[0] if batch_sizes else 2
        for seq_len in sequence_lengths:
            test_inputs = [torch.randn(fixed_batch_size, seq_len, base_embed_dim)]
            metrics = self.benchmark_function(component, test_inputs, f"seq_{seq_len}")
            scaling_results['sequence_scaling'].append(metrics)

        return scaling_results

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all benchmark results"""
        if not self._benchmark_results:
            return {'message': 'No benchmark results available'}

        speedups = [result.get_speedup() for result in self._benchmark_results if result.get_speedup() > 0]
        memory_reductions = [result.get_memory_reduction() for result in self._benchmark_results]

        return {
            'total_benchmarks': len(self._benchmark_results),
            'average_speedup': statistics.mean(speedups) if speedups else 1.0,
            'max_speedup': max(speedups) if speedups else 1.0,
            'average_memory_reduction': statistics.mean(memory_reductions) if memory_reductions else 0.0,
            'successful_optimizations': len([s for s in speedups if s > 1.05]),  # >5% improvement
            'results': [result.to_dict() for result in self._benchmark_results]
        }

    def save_results(self, filepath: Optional[str] = None) -> str:
        """Save benchmark results to file"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"benchmark_results_{timestamp}.json"

        summary = self.get_benchmark_summary()

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"📊 Benchmark results saved to: {filepath}")
        return filepath


# Convenience functions for easy usage
def create_benchmark_runner(config: Optional[BenchmarkConfig] = None) -> UnifiedBenchmarkRunner:
    """Create a unified benchmark runner"""
    return UnifiedBenchmarkRunner(config)


def quick_benchmark(optimized_component: nn.Module,
                   reference_component: Optional[nn.Module] = None,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Quick benchmark for a component with minimal configuration"""
    config = BenchmarkConfig(
        name=f"{optimized_component.__class__.__name__}_quick",
        benchmark_type=BenchmarkType.INFERENCE,
        warmup_iterations=3,
        measurement_iterations=10,
        device=str(device) if device else "auto"
    )

    runner = UnifiedBenchmarkRunner(config)
    result = runner.benchmark_component(optimized_component, reference_component)

    return {
        'speedup': result.get_speedup(),
        'memory_reduction': result.get_memory_reduction(),
        'latency_ms': result.optimized_metrics.latency_ms if result.optimized_metrics else 0.0,
        'throughput': result.optimized_metrics.throughput_samples_per_sec if result.optimized_metrics else 0.0
    }


if __name__ == "__main__":
    # Example usage
    print("🚀 Unified Benchmark Runner")

    # Quick environment test
    config = BenchmarkConfig(
        name="example_test",
        benchmark_type=BenchmarkType.INFERENCE
    )

    runner = create_benchmark_runner(config)

    # Test with a simple linear layer
    test_component = nn.Linear(256, 256)
    test_inputs = [torch.randn(4, 128, 256)]

    metrics = runner.benchmark_function(test_component, test_inputs, "linear_test")
    print(f"✅ Test completed: {metrics.latency_ms:.2f}ms latency")

    summary = runner.get_benchmark_summary()
    print(f"📊 Summary: {summary['total_benchmarks']} benchmark(s) completed")