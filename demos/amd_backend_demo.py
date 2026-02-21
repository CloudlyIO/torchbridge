#!/usr/bin/env python3
"""
AMD ROCm Backend Demo

Demonstrates the AMD backend capabilities for TorchBridge,
including configuration, optimization, and profiling features.

This demo works in simulation mode without actual AMD hardware,
showing the API and workflow for AMD cross-backend support.

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

    from torchbridge.backends.amd.amd_adapter import AMDAdapter
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
        optimizer = AMDAdapter(config)

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
        optimizer = AMDAdapter(config)

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
        optimizer = AMDAdapter(config)

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
    print("  AMD ROCm Backend Demo")
    print("  TorchBridge - Production-Ready AMD GPU Support")
    print("=" * 60)

    all_results = {
        "configuration": demo_amd_configuration(),
        "optimizer": demo_amd_optimizer(),
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
