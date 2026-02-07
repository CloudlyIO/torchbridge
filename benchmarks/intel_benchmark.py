"""
Intel XPU Backend Benchmarks (v0.5.3)

Comprehensive benchmark suite for Intel XPU backend optimization.
Measures performance across different optimization levels, precisions,
and workloads on Intel GPUs (PVC, Arc, Flex).

Usage:
    python benchmarks/intel_benchmark.py

Output includes:
- Optimization level comparison (O0, O1, O2, O3)
- Precision comparison (FP32, BF16, FP16)
- Memory management benchmarks
- IPEX optimization impact
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torchbridge.backends.intel import (
    IntelBackend,
    IntelMemoryManager,
    IntelOptimizer,
    get_ipex_version,
    is_ipex_available,
    is_xpu_available,
)
from torchbridge.core.config import TorchBridgeConfig


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    optimization_level: str
    dtype: str
    duration_ms: float
    throughput: float
    memory_mb: float
    success: bool
    error: str | None = None


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def get_device():
    """Get the appropriate device."""
    if is_xpu_available():
        return torch.device('xpu:0')
    return torch.device('cpu')


def create_transformer_block(hidden_size: int = 256) -> nn.Module:
    """Create a transformer-like block for benchmarking."""
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size * 4),
        nn.GELU(),
        nn.Linear(hidden_size * 4, hidden_size),
        nn.LayerNorm(hidden_size),
    )


def create_conv_block() -> nn.Module:
    """Create a convolutional block for benchmarking."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )


def benchmark_optimization_level(
    model: nn.Module,
    input_tensor: torch.Tensor,
    level: str,
    num_iterations: int = 50,
    warmup: int = 10
) -> BenchmarkResult:
    """Benchmark a single optimization level."""
    try:
        config = TorchBridgeConfig()
        backend = IntelBackend(config)
        optimizer = IntelOptimizer(config)

        # Prepare and optimize model
        model_copy = type(model)()
        prepared = backend.prepare_model(model_copy)
        optimized = optimizer.optimize(prepared, level=level)

        device = backend.device
        input_on_device = input_tensor.to(device)

        # Warmup
        optimized.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = optimized(input_on_device)

        if device.type == 'xpu':
            torch.xpu.synchronize()

        # Benchmark
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = optimized(input_on_device)

        if device.type == 'xpu':
            torch.xpu.synchronize()

        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000
        throughput = num_iterations / (duration_ms / 1000)

        # Get memory usage
        memory_mb = 0
        if device.type == 'xpu':
            memory_mb = torch.xpu.memory_allocated() / (1024 ** 2)

        return BenchmarkResult(
            name=f"opt_{level}",
            optimization_level=level,
            dtype="float32",
            duration_ms=duration_ms,
            throughput=throughput,
            memory_mb=memory_mb,
            success=True
        )

    except Exception as e:
        return BenchmarkResult(
            name=f"opt_{level}",
            optimization_level=level,
            dtype="float32",
            duration_ms=0,
            throughput=0,
            memory_mb=0,
            success=False,
            error=str(e)
        )


def benchmark_precision(
    model: nn.Module,
    input_tensor: torch.Tensor,
    dtype: torch.dtype,
    num_iterations: int = 50,
    warmup: int = 10
) -> BenchmarkResult:
    """Benchmark different precision levels."""
    try:
        backend = IntelBackend()
        device = backend.device

        # Prepare model
        model_copy = type(model)()
        model_copy = model_copy.to(device)

        if dtype != torch.float32:
            model_copy = model_copy.to(dtype)

        input_on_device = input_tensor.to(device)
        if dtype != torch.float32:
            input_on_device = input_on_device.to(dtype)

        # Warmup
        model_copy.eval()
        with torch.no_grad():
            for _ in range(warmup):
                _ = model_copy(input_on_device)

        if device.type == 'xpu':
            torch.xpu.synchronize()

        # Benchmark
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model_copy(input_on_device)

        if device.type == 'xpu':
            torch.xpu.synchronize()

        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000
        throughput = num_iterations / (duration_ms / 1000)

        memory_mb = 0
        if device.type == 'xpu':
            memory_mb = torch.xpu.memory_allocated() / (1024 ** 2)

        dtype_str = str(dtype).replace('torch.', '')

        return BenchmarkResult(
            name=f"precision_{dtype_str}",
            optimization_level="O2",
            dtype=dtype_str,
            duration_ms=duration_ms,
            throughput=throughput,
            memory_mb=memory_mb,
            success=True
        )

    except Exception as e:
        dtype_str = str(dtype).replace('torch.', '')
        return BenchmarkResult(
            name=f"precision_{dtype_str}",
            optimization_level="O2",
            dtype=dtype_str,
            duration_ms=0,
            throughput=0,
            memory_mb=0,
            success=False,
            error=str(e)
        )


def benchmark_memory_manager() -> dict[str, Any]:
    """Benchmark memory management operations."""
    try:
        config = TorchBridgeConfig()
        manager = IntelMemoryManager(config=None, device_id=0)

        allocation_times = []

        # Benchmark allocations of various sizes
        for size in [64, 256, 1024, 2048]:
            start = time.perf_counter()
            tensor = manager.allocate_tensor(
                shape=(size, size),
                dtype=torch.float32,
                purpose="benchmark"
            )
            alloc_time = (time.perf_counter() - start) * 1000
            allocation_times.append({
                "size": size * size * 4 / 1024 / 1024,  # MB
                "time_ms": alloc_time
            })
            manager.free_tensor(tensor)

        stats = manager.get_memory_stats()

        return {
            "allocation_times": allocation_times,
            "memory_stats": {
                "total_mb": getattr(stats, 'total_mb', 0),
                "allocated_mb": getattr(stats, 'allocated_mb', 0),
                "utilization": getattr(stats, 'utilization', 0),
            },
            "success": True
        }

    except Exception as e:
        return {
            "allocation_times": [],
            "memory_stats": {},
            "success": False,
            "error": str(e)
        }


def benchmark_ipex_optimization() -> dict[str, Any]:
    """Benchmark IPEX optimization impact."""
    results = {
        "ipex_available": is_ipex_available(),
        "ipex_version": get_ipex_version(),
        "optimizations": []
    }

    if not is_ipex_available():
        results["note"] = "IPEX not available, skipping IPEX-specific benchmarks"
        return results

    try:
        import intel_extension_for_pytorch as ipex

        backend = IntelBackend()
        device = backend.device

        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        ).to(device)

        sample_input = torch.randn(32, 512).to(device)

        # Benchmark without IPEX optimization
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        if device.type == 'xpu':
            torch.xpu.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = model(sample_input)
        if device.type == 'xpu':
            torch.xpu.synchronize()
        baseline_time = (time.perf_counter() - start) * 1000

        # Benchmark with IPEX optimization
        optimized_model = ipex.optimize(model, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(10):
                _ = optimized_model(sample_input)
        if device.type == 'xpu':
            torch.xpu.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(50):
                _ = optimized_model(sample_input)
        if device.type == 'xpu':
            torch.xpu.synchronize()
        optimized_time = (time.perf_counter() - start) * 1000

        speedup = baseline_time / optimized_time if optimized_time > 0 else 1.0

        results["optimizations"].append({
            "name": "ipex.optimize",
            "baseline_ms": baseline_time,
            "optimized_ms": optimized_time,
            "speedup": speedup
        })

    except Exception as e:
        results["error"] = str(e)

    return results


def run_all_benchmarks() -> dict[str, Any]:
    """Run all Intel XPU benchmarks."""
    print_section("Intel XPU Backend Benchmarks (v0.5.3)")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "xpu_available": is_xpu_available(),
        "ipex_available": is_ipex_available(),
        "ipex_version": get_ipex_version(),
        "device": str(get_device()),
        "benchmarks": {}
    }

    # Print environment info
    print(f"XPU Available: {results['xpu_available']}")
    print(f"IPEX Available: {results['ipex_available']}")
    print(f"IPEX Version: {results['ipex_version'] or 'N/A'}")
    print(f"Device: {results['device']}")

    # 1. Optimization Level Comparison
    print_section("1. Optimization Level Comparison")

    batch_size = 32
    hidden_size = 256

    transformer_model = create_transformer_block(hidden_size)
    transformer_input = torch.randn(batch_size, hidden_size)

    opt_level_results = []
    for level in ["O0", "O1", "O2", "O3"]:
        result = benchmark_optimization_level(
            transformer_model,
            transformer_input,
            level,
            num_iterations=50
        )
        opt_level_results.append(asdict(result))

        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  {level:4} | {result.throughput:8.1f} iter/s | "
              f"{result.duration_ms:8.2f} ms | {status}")

    results["benchmarks"]["optimization_levels"] = opt_level_results

    # 2. Precision Comparison
    print_section("2. Precision Comparison")

    precision_results = []
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        result = benchmark_precision(
            transformer_model,
            transformer_input,
            dtype,
            num_iterations=50
        )
        precision_results.append(asdict(result))

        dtype_str = str(dtype).replace('torch.', '')
        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  {dtype_str:10} | {result.throughput:8.1f} iter/s | "
              f"{result.duration_ms:8.2f} ms | {status}")

    results["benchmarks"]["precision"] = precision_results

    # 3. Memory Management
    print_section("3. Memory Management")

    memory_results = benchmark_memory_manager()
    results["benchmarks"]["memory_management"] = memory_results

    if memory_results["success"]:
        print("  Allocation times:")
        for alloc in memory_results["allocation_times"]:
            print(f"    {alloc['size']:6.1f} MB: {alloc['time_ms']:.4f} ms")
    else:
        print(f"  Memory benchmark failed: {memory_results.get('error', 'Unknown')}")

    # 4. IPEX Optimization Impact
    print_section("4. IPEX Optimization Impact")

    ipex_results = benchmark_ipex_optimization()
    results["benchmarks"]["ipex_optimization"] = ipex_results

    if ipex_results.get("optimizations"):
        for opt in ipex_results["optimizations"]:
            print(f"  {opt['name']}:")
            print(f"    Baseline: {opt['baseline_ms']:.2f} ms")
            print(f"    Optimized: {opt['optimized_ms']:.2f} ms")
            print(f"    Speedup: {opt['speedup']:.2f}x")
    elif ipex_results.get("note"):
        print(f"  {ipex_results['note']}")

    # 5. Convolutional Block (CNN workload)
    print_section("5. CNN Workload (Conv Block)")

    conv_model = create_conv_block()
    conv_model.eval()
    conv_input = torch.randn(batch_size, 3, 64, 64)

    conv_results = []
    for level in ["O1", "O2", "O3"]:
        result = benchmark_optimization_level(
            conv_model,
            conv_input,
            level,
            num_iterations=30
        )
        conv_results.append(asdict(result))

        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  {level:4} | {result.throughput:8.1f} iter/s | {status}")

    results["benchmarks"]["cnn_workload"] = conv_results

    # Summary
    print_section("Benchmark Summary")

    successful = sum(1 for r in opt_level_results if r["success"])
    print(f"  Optimization level benchmarks: {successful}/{len(opt_level_results)} successful")

    successful = sum(1 for r in precision_results if r["success"])
    print(f"  Precision benchmarks: {successful}/{len(precision_results)} successful")

    print(f"  Memory management: {'OK' if memory_results['success'] else 'FAILED'}")
    print(f"  IPEX optimization: {'OK' if ipex_results.get('optimizations') else 'N/A'}")

    # Save results
    output_path = Path(__file__).parent / "results" / "intel_benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
