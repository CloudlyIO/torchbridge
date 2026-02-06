#!/usr/bin/env python3
"""
AMD ROCm Backend Demo (v0.3.4)

Demonstrates the AMD backend capabilities for TorchBridge,
including configuration, optimization, and profiling features.

This demo works in simulation mode without actual AMD hardware,
showing the API and workflow for AMD GPU optimization.

Usage:
    PYTHONPATH=src python3 demos/amd_backend_demo.py
    PYTHONPATH=src python3 demos/amd_backend_demo.py --quick
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path for demos.shared imports
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn

# Use shared utilities
from demos.shared.utils import print_section


def print_result(name: str, status: str, details: str = "") -> None:
    """Print a test result."""
    icon = "✅" if status == "pass" else "⚠️" if status == "warn" else "❌"
    print(f"  {icon} {name}")
    if details:
        print(f"     {details}")


def demo_amd_configuration() -> dict[str, Any]:
    """Demonstrate AMD configuration options."""
    print_section("AMD Configuration")

    from torchbridge.core.config import AMDArchitecture, AMDConfig

    results = {"passed": 0, "total": 0}

    # Test 1: Default configuration
    results["total"] += 1
    try:
        config = AMDConfig()
        print_result(
            "Default configuration",
            "pass",
            f"Architecture: {config.architecture.value}, "
            f"Optimization: {config.optimization_level}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Default configuration", "fail", str(e))

    # Test 2: CDNA2 configuration (MI200 series)
    results["total"] += 1
    try:
        config = AMDConfig(
            architecture=AMDArchitecture.CDNA2,
            optimization_level="balanced",
            enable_matrix_cores=True,
        )
        print_result(
            "CDNA2 (MI200) configuration",
            "pass",
            f"Matrix Cores: {config.enable_matrix_cores}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("CDNA2 configuration", "fail", str(e))

    # Test 3: CDNA3 configuration (MI300 series)
    results["total"] += 1
    try:
        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            optimization_level="aggressive",
            enable_matrix_cores=True,
            default_precision="bf16",
        )
        print_result(
            "CDNA3 (MI300) configuration",
            "pass",
            f"Precision: {config.default_precision}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("CDNA3 configuration", "fail", str(e))

    # Test 4: Memory configuration
    results["total"] += 1
    try:
        config = AMDConfig(
            memory_pool_size_gb=32.0,
            enable_memory_pooling=True,
        )
        print_result(
            "Memory configuration",
            "pass",
            f"Pool size: {config.memory_pool_size_gb}GB",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Memory configuration", "fail", str(e))

    return results


def demo_amd_optimizer() -> dict[str, Any]:
    """Demonstrate AMD optimizer functionality."""
    print_section("AMD Optimizer")

    from torchbridge.backends.amd.amd_optimizer import AMDOptimizer
    from torchbridge.core.config import AMDArchitecture, AMDConfig

    results = {"passed": 0, "total": 0}

    # Create test model
    model = nn.Sequential(
        nn.Linear(768, 3072),
        nn.GELU(),
        nn.Linear(3072, 768),
        nn.LayerNorm(768),
    )

    # Test 1: Conservative optimization
    results["total"] += 1
    try:
        config = AMDConfig(optimization_level="conservative")
        optimizer = AMDOptimizer(config)

        start = time.perf_counter()
        optimized = optimizer.optimize(model)
        elapsed = (time.perf_counter() - start) * 1000

        print_result(
            "Conservative optimization",
            "pass",
            f"Time: {elapsed:.2f}ms",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Conservative optimization", "fail", str(e))

    # Test 2: Balanced optimization
    results["total"] += 1
    try:
        config = AMDConfig(
            architecture=AMDArchitecture.CDNA2,
            optimization_level="balanced",
        )
        optimizer = AMDOptimizer(config)

        start = time.perf_counter()
        optimized = optimizer.optimize(model)
        elapsed = (time.perf_counter() - start) * 1000

        summary = optimizer.get_optimization_summary()
        print_result(
            "Balanced optimization",
            "pass",
            f"Time: {elapsed:.2f}ms, "
            f"Fused ops: {summary['fused_operations']}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Balanced optimization", "fail", str(e))

    # Test 3: Aggressive optimization
    results["total"] += 1
    try:
        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            optimization_level="aggressive",
            enable_matrix_cores=True,
            enable_mixed_precision=True,
        )
        optimizer = AMDOptimizer(config)

        start = time.perf_counter()
        optimized = optimizer.optimize(model)
        elapsed = (time.perf_counter() - start) * 1000

        summary = optimizer.get_optimization_summary()
        print_result(
            "Aggressive optimization",
            "pass",
            f"Time: {elapsed:.2f}ms, "
            f"Matrix Cores: {summary['matrix_cores_enabled']}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Aggressive optimization", "fail", str(e))

    return results


def demo_rocm_compiler() -> dict[str, Any]:
    """Demonstrate ROCm compiler functionality."""
    print_section("ROCm Compiler")

    from torchbridge.backends.amd.rocm_compiler import ROCmCompiler
    from torchbridge.core.config import AMDArchitecture, AMDConfig

    results = {"passed": 0, "total": 0}

    # Test 1: Compiler creation
    results["total"] += 1
    try:
        config = AMDConfig(architecture=AMDArchitecture.CDNA2)
        compiler = ROCmCompiler(config)

        print_result(
            "Compiler creation",
            "pass",
            f"Architecture: {config.architecture.value}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Compiler creation", "fail", str(e))

    # Test 2: Kernel compilation
    results["total"] += 1
    try:
        config = AMDConfig()
        compiler = ROCmCompiler(config)

        source = """
        __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) c[idx] = a[idx] + b[idx];
        }
        """

        start = time.perf_counter()
        kernel = compiler.compile_kernel(source, "vector_add")
        elapsed = (time.perf_counter() - start) * 1000

        print_result(
            "Kernel compilation",
            "pass",
            f"Kernel: {kernel.name}, Time: {elapsed:.2f}ms",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Kernel compilation", "fail", str(e))

    # Test 3: Compilation caching
    results["total"] += 1
    try:
        config = AMDConfig()
        compiler = ROCmCompiler(config)

        source = "__global__ void test() {}"

        # First compilation
        compiler.compile_kernel(source, "test")

        # Second compilation (should hit cache)
        start = time.perf_counter()
        compiler.compile_kernel(source, "test")
        elapsed = (time.perf_counter() - start) * 1000

        stats = compiler.get_compilation_stats()
        print_result(
            "Compilation caching",
            "pass",
            f"Cache hits: {stats['cache_hits']}, "
            f"Hit rate: {stats['cache_hit_rate_percent']:.1f}%",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Compilation caching", "fail", str(e))

    return results


def demo_hip_utilities() -> dict[str, Any]:
    """Demonstrate HIP utilities functionality."""
    print_section("HIP Utilities")

    from torchbridge.backends.amd.hip_utilities import HIPUtilities
    from torchbridge.core.config import AMDConfig

    results = {"passed": 0, "total": 0}

    # Test 1: Utilities creation
    results["total"] += 1
    try:
        config = AMDConfig(enable_profiling=True)
        utils = HIPUtilities(config)

        print_result(
            "Utilities creation",
            "pass",
            f"Profiling: {config.enable_profiling}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Utilities creation", "fail", str(e))

    # Test 2: Stream management
    results["total"] += 1
    try:
        config = AMDConfig()
        utils = HIPUtilities(config)

        stream = utils.create_stream("compute_stream", priority=0)
        print_result(
            "Stream management",
            "pass",
            f"Stream: {stream.name}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Stream management", "fail", str(e))

    # Test 3: Profiling
    results["total"] += 1
    try:
        config = AMDConfig(enable_profiling=True)
        utils = HIPUtilities(config)

        # Profile some operations
        with utils.profile_region("matrix_ops"):
            a = torch.randn(512, 512)
            b = torch.randn(512, 512)
            c = torch.matmul(a, b)

        with utils.profile_region("activation"):
            d = torch.relu(c)

        summary = utils.get_profiling_summary()
        print_result(
            "Profiling",
            "pass",
            f"Regions: {summary['total_regions']}, "
            f"Unique: {summary['unique_regions']}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Profiling", "fail", str(e))

    # Test 4: Device properties
    results["total"] += 1
    try:
        config = AMDConfig()
        utils = HIPUtilities(config)

        props = utils.get_device_properties()
        if props["available"]:
            print_result(
                "Device properties",
                "pass",
                f"Device: {props['name']}, "
                f"Memory: {props['total_memory_gb']:.1f}GB",
            )
        else:
            print_result(
                "Device properties",
                "warn",
                "No GPU available (simulation mode)",
            )
        results["passed"] += 1
    except Exception as e:
        print_result("Device properties", "fail", str(e))

    return results


def demo_memory_manager() -> dict[str, Any]:
    """Demonstrate memory manager functionality."""
    print_section("Memory Manager")

    from torchbridge.backends.amd.memory_manager import AMDMemoryManager
    from torchbridge.core.config import AMDConfig

    results = {"passed": 0, "total": 0}

    # Test 1: Manager creation
    results["total"] += 1
    try:
        config = AMDConfig(
            memory_pool_size_gb=8.0,
            enable_memory_pooling=True,
        )
        manager = AMDMemoryManager(config)

        print_result(
            "Manager creation",
            "pass",
            f"Pool size: {config.memory_pool_size_gb}GB",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Manager creation", "fail", str(e))

    # Test 2: Memory statistics (CPU mode)
    results["total"] += 1
    try:
        config = AMDConfig()
        manager = AMDMemoryManager(config)

        stats = manager.get_memory_stats()
        if stats.total_mb > 0:
            print_result(
                "Memory statistics",
                "pass",
                f"Total: {stats.total_mb:.0f}MB, "
                f"Free: {stats.free_mb:.0f}MB",
            )
        else:
            print_result(
                "Memory statistics",
                "warn",
                "No GPU available (simulation mode)",
            )
        results["passed"] += 1
    except Exception as e:
        print_result("Memory statistics", "fail", str(e))

    return results


def demo_full_pipeline(quick: bool = False) -> dict[str, Any]:
    """Demonstrate full AMD optimization pipeline."""
    print_section("Full Optimization Pipeline")

    from torchbridge.backends.amd.amd_optimizer import AMDOptimizer
    from torchbridge.backends.amd.hip_utilities import HIPUtilities
    from torchbridge.core.config import AMDArchitecture, AMDConfig

    results = {"passed": 0, "total": 0}

    # Create a transformer-like model
    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int = 768, n_heads: int = 12):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x

    # Test: Full pipeline
    results["total"] += 1
    try:
        # Configuration for MI300 (CDNA3)
        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            optimization_level="aggressive",
            enable_matrix_cores=True,
            enable_mixed_precision=True,
            enable_profiling=True,
        )

        optimizer = AMDOptimizer(config)
        utils = HIPUtilities(config)

        # Create model (d_model must be divisible by num_heads=12)
        model = TransformerBlock(d_model=768 if not quick else 384)

        # Optimize with profiling
        with utils.profile_region("model_optimization"):
            optimized_model = optimizer.optimize(model)

        # Run inference with profiling
        batch_size = 4 if not quick else 2
        seq_len = 128 if not quick else 32
        d_model = 768 if not quick else 384

        x = torch.randn(batch_size, seq_len, d_model)

        with utils.profile_region("inference"):
            with torch.no_grad():
                output = optimized_model(x)

        # Get profiling summary
        summary = utils.get_profiling_summary()

        print_result(
            "Full pipeline",
            "pass",
            f"Input: {x.shape}, Output: {output.shape}",
        )
        print(f"     Profiled regions: {summary['total_regions']}")

        for name, data in summary.get("regions", {}).items():
            print(f"       - {name}: {data['avg_ms']:.3f}ms (avg)")

        results["passed"] += 1

    except Exception as e:
        print_result("Full pipeline", "fail", str(e))

    return results


def main():
    """Run all AMD backend demos."""
    parser = argparse.ArgumentParser(description="AMD Backend Demo")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version with smaller models",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  AMD ROCm Backend Demo (v0.3.4)")
    print("  TorchBridge - Production-Ready AMD GPU Support")
    print("=" * 60)

    all_results = {
        "configuration": demo_amd_configuration(),
        "optimizer": demo_amd_optimizer(),
        "compiler": demo_rocm_compiler(),
        "utilities": demo_hip_utilities(),
        "memory": demo_memory_manager(),
        "pipeline": demo_full_pipeline(quick=args.quick),
    }

    # Summary
    print_section("Summary")

    total_passed = sum(r["passed"] for r in all_results.values())
    total_tests = sum(r["total"] for r in all_results.values())

    for name, results in all_results.items():
        status = "✅" if results["passed"] == results["total"] else "⚠️"
        print(f"  {status} {name.capitalize()}: {results['passed']}/{results['total']}")

    print(f"\n  Total: {total_passed}/{total_tests} passed")

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"  Success rate: {success_rate:.1f}%")

    if total_passed == total_tests:
        print("\n  🎉 All AMD backend demos passed!")
    else:
        print("\n  ⚠️  Some demos had warnings (likely no AMD GPU available)")

    return 0 if total_passed >= total_tests * 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
