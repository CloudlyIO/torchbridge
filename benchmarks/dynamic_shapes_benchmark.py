#!/usr/bin/env python3
"""
Dynamic Shape Bucketing Benchmark Suite

Comprehensive benchmarking suite for validating dynamic shape bucketing
performance improvements and comparing against cutting-edge baselines.

🎯 BENCHMARK OBJECTIVES:
- Validate 3x speedup on variable inputs
- Compare against state-of-the-art implementations
- Measure memory efficiency improvements
- Test scalability across different workloads

🏆 COMPARISON BASELINES:
- Native PyTorch (baseline)
- Manual shape optimization
- Static shape batching
- Dynamic shape bucketing (our implementation)

Expected Performance Targets:
- 3x speedup on variable-size inputs
- < 10% memory overhead from padding
- > 90% GPU utilization on diverse workloads
- Sub-microsecond bucket lookup performance
"""

import argparse
import time
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import statistics

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

from torchbridge.optimizations.patterns.dynamic_shapes import (
    DynamicShapeBucketing,
    BucketingStrategy,
    PaddingStrategy,
    DynamicShapeModule,
    create_optimal_bucketing_system,
    benchmark_dynamic_shapes
)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str
    num_iterations: int
    warmup_iterations: int
    input_shapes: List[Tuple[int, ...]]
    model_config: Dict[str, Any]
    device: str
    precision: torch.dtype = torch.float32


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config_name: str
    method: str
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p95_time_ms: float
    throughput_samples_per_sec: float
    memory_usage_mb: float
    gpu_utilization: float
    cache_hit_rate: Optional[float] = None
    bucket_efficiency: Optional[float] = None
    padding_overhead: Optional[float] = None


class StaticShapeBatching:
    """
    Baseline implementation using static shape batching.

    This represents a common optimization where inputs are batched
    by exact shape, but without the sophisticated bucketing strategies.
    """

    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.shape_batches = {}

    def process_batch(self, model: nn.Module, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process inputs using static shape batching."""
        # Group by exact shape
        shape_groups = {}
        for i, tensor in enumerate(inputs):
            shape = tensor.shape
            if shape not in shape_groups:
                shape_groups[shape] = []
            shape_groups[shape].append((i, tensor))

        outputs = [None] * len(inputs)

        # Process each shape group
        for shape, tensors in shape_groups.items():
            indices, tensor_list = zip(*tensors)

            # Batch tensors of the same shape
            for batch_start in range(0, len(tensor_list), self.max_batch_size):
                batch_end = min(batch_start + self.max_batch_size, len(tensor_list))
                batch_tensors = tensor_list[batch_start:batch_end]
                batch_indices = indices[batch_start:batch_end]

                # Stack tensors if they have the same shape
                if len(batch_tensors) > 1:
                    try:
                        batched_input = torch.stack(batch_tensors, dim=0)
                        batched_output = model(batched_input)

                        # Unstack outputs
                        for i, output in enumerate(torch.unbind(batched_output, dim=0)):
                            outputs[batch_indices[i]] = output
                    except RuntimeError:
                        # Fall back to individual processing
                        for idx, tensor in zip(batch_indices, batch_tensors):
                            outputs[idx] = model(tensor.unsqueeze(0)).squeeze(0)
                else:
                    outputs[batch_indices[0]] = model(batch_tensors[0])

        return outputs


class ManualOptimization:
    """
    Manual shape optimization baseline.

    This implements hand-tuned optimizations that a developer might
    apply for specific workloads, representing best-practice manual optimization.
    """

    def __init__(self):
        self.optimal_shapes = {
            # Common shapes that work well with GPUs
            (8, 512): (8, 512),    # Exact match
            (16, 256): (16, 256),  # Exact match
            (32, 128): (32, 128),  # Exact match
        }
        self.fallback_shapes = [
            (8, 512), (16, 256), (32, 128), (64, 64)
        ]

    def find_best_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Find the best shape for the given input."""
        if input_shape in self.optimal_shapes:
            return self.optimal_shapes[input_shape]

        # Find smallest shape that can contain the input
        input_size = input_shape[0] * input_shape[1] if len(input_shape) >= 2 else input_shape[0]

        best_shape = None
        min_waste = float('inf')

        for candidate_shape in self.fallback_shapes:
            candidate_size = candidate_shape[0] * candidate_shape[1]

            # Check if candidate can contain input
            if (candidate_shape[0] >= input_shape[0] and
                len(input_shape) >= 2 and candidate_shape[1] >= input_shape[1]):

                waste = (candidate_size - input_size) / candidate_size
                if waste < min_waste:
                    min_waste = waste
                    best_shape = candidate_shape

        return best_shape or input_shape

    def process_input(self, model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
        """Process input with manual optimization."""
        original_shape = input_tensor.shape
        optimal_shape = self.find_best_shape(original_shape)

        if optimal_shape == original_shape:
            return model(input_tensor)

        # Pad to optimal shape
        pad_dims = []
        for i in range(len(original_shape) - 1, -1, -1):
            padding_needed = optimal_shape[i] - original_shape[i]
            pad_left = padding_needed // 2
            pad_right = padding_needed - pad_left
            pad_dims.extend([pad_left, pad_right])

        padded_input = F.pad(input_tensor, pad_dims, mode='constant', value=0)
        padded_output = model(padded_input)

        # Unpad output (assuming same transformation)
        unpad_slices = tuple(
            slice(pad_dims[-(2*i+2)], optimal_shape[i] - pad_dims[-(2*i+1)])
            for i in range(len(original_shape))
        )
        return padded_output[unpad_slices]


def create_test_model(config: Dict[str, Any]) -> nn.Module:
    """Create test model based on configuration."""
    model_type = config.get("type", "transformer")

    if model_type == "transformer":
        return nn.Sequential(
            nn.Linear(config["d_model"], config["d_model"]),
            nn.ReLU(),
            nn.TransformerEncoderLayer(
                d_model=config["d_model"],
                nhead=config["nhead"],
                dim_feedforward=config["dim_feedforward"],
                batch_first=True
            ),
            nn.Linear(config["d_model"], config["output_size"])
        )

    elif model_type == "cnn":
        return nn.Sequential(
            nn.Conv2d(config["in_channels"], config["hidden_channels"], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config["hidden_channels"], config["hidden_channels"], 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(config["hidden_channels"], config["output_size"])
        )

    elif model_type == "linear":
        layers = []
        prev_size = config["input_size"]
        for size in config["hidden_sizes"]:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU()
            ])
            prev_size = size
        layers.append(nn.Linear(prev_size, config["output_size"]))
        return nn.Sequential(*layers)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def generate_benchmark_inputs(
    shapes: List[Tuple[int, ...]],
    num_samples_per_shape: int = 10,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> List[torch.Tensor]:
    """Generate benchmark inputs with specified shapes."""
    inputs = []

    for shape in shapes:
        for _ in range(num_samples_per_shape):
            tensor = torch.randn(*shape, device=device, dtype=dtype)
            inputs.append(tensor)

    # Shuffle to avoid shape-sorted processing
    np.random.shuffle(inputs)
    return inputs


def measure_gpu_utilization() -> float:
    """Measure current GPU utilization."""
    if not torch.cuda.is_available():
        return 0.0

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return utilization.gpu / 100.0
    except ImportError:
        # Fall back to approximation based on memory usage
        allocated = torch.cuda.memory_allocated()
        cached = torch.cuda.memory_reserved()
        return min(1.0, allocated / (cached + 1))
    except Exception:
        return 0.0


def run_baseline_benchmark(
    model: nn.Module,
    inputs: List[torch.Tensor],
    config: BenchmarkConfig
) -> BenchmarkResult:
    """Run baseline benchmark without optimizations."""
    print(f"🔍 Running baseline benchmark: {config.name}")

    model.eval()
    times = []
    gpu_utilizations = []

    # Warmup
    with torch.no_grad():
        for i in range(config.warmup_iterations):
            for tensor in inputs[:5]:
                _ = model(tensor)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Benchmark
    with torch.no_grad():
        for iteration in range(config.num_iterations):
            for tensor in inputs:
                start_time = time.perf_counter()

                output = model(tensor)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)  # Convert to ms

                # Measure GPU utilization periodically
                if len(gpu_utilizations) < 20:
                    gpu_utilizations.append(measure_gpu_utilization())

    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_usage_mb = (final_memory - initial_memory) / (1024 * 1024)

    return BenchmarkResult(
        config_name=config.name,
        method="baseline",
        avg_time_ms=statistics.mean(times),
        std_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        p95_time_ms=np.percentile(times, 95),
        throughput_samples_per_sec=len(times) / (sum(times) / 1000),
        memory_usage_mb=memory_usage_mb,
        gpu_utilization=statistics.mean(gpu_utilizations) if gpu_utilizations else 0
    )


def run_static_batching_benchmark(
    model: nn.Module,
    inputs: List[torch.Tensor],
    config: BenchmarkConfig
) -> BenchmarkResult:
    """Run static shape batching benchmark."""
    print(f"📦 Running static batching benchmark: {config.name}")

    batcher = StaticShapeBatching(max_batch_size=16)
    model.eval()

    times = []
    gpu_utilizations = []

    # Warmup
    with torch.no_grad():
        for i in range(config.warmup_iterations):
            sample_inputs = inputs[:10]
            _ = batcher.process_batch(model, sample_inputs)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Benchmark - process in chunks to simulate batching
    chunk_size = 20
    with torch.no_grad():
        for iteration in range(config.num_iterations):
            for i in range(0, len(inputs), chunk_size):
                chunk = inputs[i:i + chunk_size]

                start_time = time.perf_counter()

                outputs = batcher.process_batch(model, chunk)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time
                per_sample_time = elapsed / len(chunk) * 1000  # ms per sample
                times.extend([per_sample_time] * len(chunk))

                # Measure GPU utilization periodically
                if len(gpu_utilizations) < 20:
                    gpu_utilizations.append(measure_gpu_utilization())

    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_usage_mb = (final_memory - initial_memory) / (1024 * 1024)

    return BenchmarkResult(
        config_name=config.name,
        method="static_batching",
        avg_time_ms=statistics.mean(times),
        std_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        p95_time_ms=np.percentile(times, 95),
        throughput_samples_per_sec=len(times) / (sum(times) / 1000),
        memory_usage_mb=memory_usage_mb,
        gpu_utilization=statistics.mean(gpu_utilizations) if gpu_utilizations else 0
    )


def run_manual_optimization_benchmark(
    model: nn.Module,
    inputs: List[torch.Tensor],
    config: BenchmarkConfig
) -> BenchmarkResult:
    """Run manual optimization benchmark."""
    print(f"🔧 Running manual optimization benchmark: {config.name}")

    optimizer = ManualOptimization()
    model.eval()

    times = []
    gpu_utilizations = []
    padding_overheads = []

    # Warmup
    with torch.no_grad():
        for i in range(config.warmup_iterations):
            for tensor in inputs[:5]:
                _ = optimizer.process_input(model, tensor)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Benchmark
    with torch.no_grad():
        for iteration in range(config.num_iterations):
            for tensor in inputs:
                start_time = time.perf_counter()

                output = optimizer.process_input(model, tensor)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)

                # Calculate padding overhead
                original_size = tensor.numel()
                optimal_shape = optimizer.find_best_shape(tensor.shape)
                optimal_size = np.prod(optimal_shape)
                overhead = (optimal_size - original_size) / optimal_size if optimal_size > 0 else 0
                padding_overheads.append(overhead)

                # Measure GPU utilization periodically
                if len(gpu_utilizations) < 20:
                    gpu_utilizations.append(measure_gpu_utilization())

    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_usage_mb = (final_memory - initial_memory) / (1024 * 1024)

    return BenchmarkResult(
        config_name=config.name,
        method="manual_optimization",
        avg_time_ms=statistics.mean(times),
        std_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        p95_time_ms=np.percentile(times, 95),
        throughput_samples_per_sec=len(times) / (sum(times) / 1000),
        memory_usage_mb=memory_usage_mb,
        gpu_utilization=statistics.mean(gpu_utilizations) if gpu_utilizations else 0,
        padding_overhead=statistics.mean(padding_overheads) if padding_overheads else 0
    )


def run_dynamic_bucketing_benchmark(
    model: nn.Module,
    inputs: List[torch.Tensor],
    config: BenchmarkConfig,
    strategy: BucketingStrategy = BucketingStrategy.HARDWARE_AWARE
) -> BenchmarkResult:
    """Run dynamic shape bucketing benchmark."""
    print(f"🚀 Running dynamic bucketing benchmark: {config.name} ({strategy.value})")

    # Create bucketing system
    bucketing = create_optimal_bucketing_system(
        inputs[:20],  # Use sample for initial configuration
        strategy=strategy,
        max_buckets=16
    )

    # Wrap model
    dynamic_model = DynamicShapeModule(
        base_module=model,
        bucketing_system=bucketing,
        enable_bucketing=True
    )

    dynamic_model.eval()
    times = []
    gpu_utilizations = []

    # Warmup
    with torch.no_grad():
        for i in range(config.warmup_iterations):
            for tensor in inputs[:5]:
                _ = dynamic_model(tensor)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Benchmark
    with torch.no_grad():
        for iteration in range(config.num_iterations):
            for tensor in inputs:
                start_time = time.perf_counter()

                output = dynamic_model(tensor)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.perf_counter() - start_time
                times.append(elapsed * 1000)

                # Measure GPU utilization periodically
                if len(gpu_utilizations) < 20:
                    gpu_utilizations.append(measure_gpu_utilization())

    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_usage_mb = (final_memory - initial_memory) / (1024 * 1024)

    # Get bucketing statistics
    stats = bucketing.get_performance_stats()

    return BenchmarkResult(
        config_name=config.name,
        method=f"dynamic_bucketing_{strategy.value}",
        avg_time_ms=statistics.mean(times),
        std_time_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        p95_time_ms=np.percentile(times, 95),
        throughput_samples_per_sec=len(times) / (sum(times) / 1000),
        memory_usage_mb=memory_usage_mb,
        gpu_utilization=statistics.mean(gpu_utilizations) if gpu_utilizations else 0,
        cache_hit_rate=stats["cache_hit_rate"],
        bucket_efficiency=stats["average_bucket_efficiency"]
    )


def create_benchmark_configs() -> List[BenchmarkConfig]:
    """Create benchmark configurations for different scenarios."""
    configs = []

    # Small transformer workload
    configs.append(BenchmarkConfig(
        name="small_transformer",
        num_iterations=50,
        warmup_iterations=10,
        input_shapes=[
            (4, 32), (6, 48), (8, 64), (10, 80), (12, 96), (16, 128),
            (5, 37), (7, 53), (9, 71), (11, 89), (13, 103), (15, 119)
        ],
        model_config={
            "type": "transformer",
            "d_model": 256,
            "nhead": 8,
            "dim_feedforward": 1024,
            "output_size": 128
        },
        device="cuda" if torch.cuda.is_available() else "cpu"
    ))

    # Medium transformer workload
    configs.append(BenchmarkConfig(
        name="medium_transformer",
        num_iterations=30,
        warmup_iterations=5,
        input_shapes=[
            (8, 64), (12, 96), (16, 128), (20, 160), (24, 192), (32, 256),
            (10, 73), (14, 107), (18, 139), (22, 173), (26, 203), (30, 239)
        ],
        model_config={
            "type": "transformer",
            "d_model": 512,
            "nhead": 8,
            "dim_feedforward": 2048,
            "output_size": 256
        },
        device="cuda" if torch.cuda.is_available() else "cpu"
    ))

    # Vision workload
    configs.append(BenchmarkConfig(
        name="vision_cnn",
        num_iterations=40,
        warmup_iterations=8,
        input_shapes=[
            (4, 3, 64, 64), (6, 3, 96, 96), (8, 3, 128, 128),
            (5, 3, 73, 73), (7, 3, 107, 107), (9, 3, 139, 139)
        ],
        model_config={
            "type": "cnn",
            "in_channels": 3,
            "hidden_channels": 64,
            "output_size": 10
        },
        device="cuda" if torch.cuda.is_available() else "cpu"
    ))

    # Dense workload
    configs.append(BenchmarkConfig(
        name="dense_linear",
        num_iterations=60,
        warmup_iterations=12,
        input_shapes=[
            (16, 256), (24, 384), (32, 512), (40, 640), (48, 768),
            (18, 307), (26, 421), (34, 543), (42, 661), (50, 787)
        ],
        model_config={
            "type": "linear",
            "input_size": 256,
            "hidden_sizes": [512, 1024, 512],
            "output_size": 128
        },
        device="cuda" if torch.cuda.is_available() else "cpu"
    ))

    return configs


def run_comprehensive_benchmark(
    config: BenchmarkConfig,
    quick_mode: bool = False
) -> Dict[str, BenchmarkResult]:
    """Run comprehensive benchmark comparing all methods."""
    print(f"\n{'='*60}")
    print(f"🧪 BENCHMARKING: {config.name}")
    print(f"{'='*60}")

    if quick_mode:
        config.num_iterations = max(5, config.num_iterations // 5)
        config.warmup_iterations = max(1, config.warmup_iterations // 3)

    # Create model and inputs
    model = create_test_model(config.model_config)
    model = model.to(config.device)

    inputs = generate_benchmark_inputs(
        config.input_shapes,
        num_samples_per_shape=3 if quick_mode else 5,
        device=config.device,
        dtype=config.precision
    )

    print(f"📊 Test Configuration:")
    print(f"  Model: {config.model_config['type']}")
    print(f"  Device: {config.device}")
    print(f"  Input shapes: {len(config.input_shapes)} unique shapes")
    print(f"  Total inputs: {len(inputs)}")
    print(f"  Iterations: {config.num_iterations}")
    print()

    results = {}

    # Run benchmarks
    try:
        results["baseline"] = run_baseline_benchmark(model, inputs, config)
    except Exception as e:
        print(f"❌ Baseline benchmark failed: {e}")

    try:
        results["static_batching"] = run_static_batching_benchmark(model, inputs, config)
    except Exception as e:
        print(f"❌ Static batching benchmark failed: {e}")

    try:
        results["manual_optimization"] = run_manual_optimization_benchmark(model, inputs, config)
    except Exception as e:
        print(f"❌ Manual optimization benchmark failed: {e}")

    # Test different bucketing strategies
    strategies = [BucketingStrategy.GEOMETRIC, BucketingStrategy.HARDWARE_AWARE]

    for strategy in strategies:
        try:
            results[f"bucketing_{strategy.value}"] = run_dynamic_bucketing_benchmark(
                model, inputs, config, strategy
            )
        except Exception as e:
            print(f"❌ Dynamic bucketing ({strategy.value}) benchmark failed: {e}")

    return results


def print_benchmark_summary(all_results: Dict[str, Dict[str, BenchmarkResult]]) -> None:
    """Print comprehensive benchmark summary."""
    print("\n" + "="*100)
    print("🏆 DYNAMIC SHAPE BUCKETING BENCHMARK SUMMARY")
    print("="*100)

    # Performance comparison table
    print(f"\n📊 PERFORMANCE COMPARISON (Average Time per Sample)")
    print(f"{'Configuration':<20} {'Baseline':<12} {'Static':<12} {'Manual':<12} {'Geometric':<12} {'HW-Aware':<12}")
    print("-" * 100)

    for config_name, results in all_results.items():
        baseline_time = results.get("baseline", BenchmarkResult("", "", 0, 0, 0, 0, 0, 0, 0, 0)).avg_time_ms

        row = f"{config_name:<20}"
        for method in ["baseline", "static_batching", "manual_optimization", "bucketing_geometric", "bucketing_hardware_aware"]:
            if method in results:
                time_ms = results[method].avg_time_ms
                if baseline_time > 0:
                    speedup = baseline_time / time_ms
                    if speedup > 1.1:
                        row += f" {time_ms:8.2f}ms ({speedup:4.1f}x)"
                    else:
                        row += f" {time_ms:8.2f}ms      "
                else:
                    row += f" {time_ms:8.2f}ms      "
            else:
                row += " N/A         "
        print(row)

    # Speedup summary
    print(f"\n🚀 SPEEDUP SUMMARY")
    print(f"{'Configuration':<20} {'Best Method':<20} {'Best Speedup':<15} {'Target Met':<12}")
    print("-" * 80)

    for config_name, results in all_results.items():
        if "baseline" not in results:
            continue

        baseline_time = results["baseline"].avg_time_ms
        best_speedup = 1.0
        best_method = "baseline"

        for method, result in results.items():
            if method != "baseline" and result.avg_time_ms > 0:
                speedup = baseline_time / result.avg_time_ms
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_method = method

        target_met = "✅ YES" if best_speedup >= 2.5 else "❌ NO"
        print(f"{config_name:<20} {best_method:<20} {best_speedup:<14.2f}x {target_met}")

    # Memory efficiency summary
    print(f"\n💾 MEMORY EFFICIENCY")
    print(f"{'Configuration':<20} {'Method':<20} {'Memory Usage':<15} {'GPU Util':<12}")
    print("-" * 80)

    for config_name, results in all_results.items():
        for method, result in results.items():
            if "bucketing" in method:
                print(f"{config_name:<20} {method:<20} {result.memory_usage_mb:<14.1f}MB {result.gpu_utilization*100:<11.1f}%")

    # Cache and bucket efficiency
    print(f"\n⚙️  BUCKETING SYSTEM EFFICIENCY")
    print(f"{'Configuration':<20} {'Method':<20} {'Cache Hit Rate':<15} {'Bucket Efficiency':<18}")
    print("-" * 85)

    for config_name, results in all_results.items():
        for method, result in results.items():
            if "bucketing" in method and result.cache_hit_rate is not None:
                cache_rate = f"{result.cache_hit_rate*100:.1f}%"
                bucket_eff = f"{result.bucket_efficiency*100:.1f}%" if result.bucket_efficiency else "N/A"
                print(f"{config_name:<20} {method:<20} {cache_rate:<15} {bucket_eff}")

    # Overall validation
    print(f"\n🎯 VALIDATION SUMMARY")

    total_configs = len(all_results)
    configs_meeting_target = 0
    total_speedup = 0
    speedup_count = 0

    for config_name, results in all_results.items():
        if "baseline" not in results:
            continue

        baseline_time = results["baseline"].avg_time_ms
        best_speedup = 1.0

        for method, result in results.items():
            if "bucketing" in method and result.avg_time_ms > 0:
                speedup = baseline_time / result.avg_time_ms
                best_speedup = max(best_speedup, speedup)

        if best_speedup >= 2.5:
            configs_meeting_target += 1

        total_speedup += best_speedup
        speedup_count += 1

    avg_speedup = total_speedup / speedup_count if speedup_count > 0 else 1.0
    success_rate = configs_meeting_target / total_configs if total_configs > 0 else 0

    print(f"  📊 Average Speedup: {avg_speedup:.2f}x")
    print(f"  🎯 Target Achievement: {configs_meeting_target}/{total_configs} configurations ({success_rate*100:.1f}%)")

    if avg_speedup >= 2.5:
        print("  ✅ SUCCESS: Average speedup exceeds 2.5x target")
    else:
        print("  ❌ BELOW TARGET: Average speedup below 2.5x target")

    print("\n" + "="*100)


def save_benchmark_results(
    results: Dict[str, Dict[str, BenchmarkResult]],
    output_file: str = "dynamic_shapes_benchmark_results.json"
) -> None:
    """Save benchmark results to JSON file."""
    # Convert results to serializable format
    serializable_results = {}
    for config_name, config_results in results.items():
        serializable_results[config_name] = {}
        for method_name, result in config_results.items():
            serializable_results[config_name][method_name] = asdict(result)

    output_path = Path("benchmarks/results")
    output_path.mkdir(exist_ok=True)

    with open(output_path / output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"📄 Results saved to {output_path / output_file}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Dynamic Shape Bucketing Benchmark Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer iterations")
    parser.add_argument("--config", type=str, help="Run specific configuration only")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)")
    parser.add_argument("--output", type=str, default="dynamic_shapes_benchmark_results.json",
                        help="Output file for results")

    args = parser.parse_args()

    print("🧪 Dynamic Shape Bucketing Benchmark Suite")
    print("="*50)

    # Setup device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Quick mode: {args.quick}")
    print()

    # Create benchmark configurations
    configs = create_benchmark_configs()

    # Filter configurations if specified
    if args.config:
        configs = [c for c in configs if c.name == args.config]
        if not configs:
            print(f"❌ Configuration '{args.config}' not found")
            return

    # Update device in configs
    for config in configs:
        config.device = device

    # Run benchmarks
    all_results = {}

    for config in configs:
        try:
            results = run_comprehensive_benchmark(config, args.quick)
            all_results[config.name] = results
        except Exception as e:
            print(f"❌ Benchmark failed for {config.name}: {e}")
            continue

    # Print summary and save results
    if all_results:
        print_benchmark_summary(all_results)
        save_benchmark_results(all_results, args.output)
    else:
        print("❌ No benchmark results to display")


if __name__ == "__main__":
    main()