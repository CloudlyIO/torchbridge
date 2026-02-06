#!/usr/bin/env python3
"""
Unified Backend Demo (v0.4.8)

Demonstrates the unified backend interface and BackendFactory for automatic
hardware detection across NVIDIA, AMD, TPU, and Intel platforms.

This demo showcases:
- BackendFactory for automatic backend selection
- Unified BaseBackend interface
- OptimizationLevel enum (O0-O3)
- DeviceInfo standardized format
- Cross-platform model preparation

Works in simulation mode without specific hardware, showing the API and
workflow for the unified backend system.

Usage:
    PYTHONPATH=src python3 demos/unified_backend_demo.py
    PYTHONPATH=src python3 demos/unified_backend_demo.py --quick
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


def demo_backend_factory() -> dict[str, Any]:
    """Demonstrate BackendFactory automatic backend selection."""
    print_section("BackendFactory Auto-Detection")

    from torchbridge.backends import (
        BackendFactory,
        BackendType,
        detect_best_backend,
        get_backend,
        list_available_backends,
    )

    results = {"passed": 0, "total": 0}

    # Test 1: Detect best backend
    results["total"] += 1
    try:
        best_backend = detect_best_backend()
        print_result(
            "Auto-detect best backend",
            "pass",
            f"Detected: {best_backend}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Auto-detect best backend", "fail", str(e))

    # Test 2: List available backends
    results["total"] += 1
    try:
        available = list_available_backends()
        print_result(
            "List available backends",
            "pass",
            f"Available: {', '.join(available)}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("List available backends", "fail", str(e))

    # Test 3: Factory with AUTO selection
    results["total"] += 1
    try:
        backend = BackendFactory.create(BackendType.AUTO)
        print_result(
            "Factory AUTO selection",
            "pass",
            f"Created: {backend.BACKEND_NAME}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Factory AUTO selection", "fail", str(e))

    # Test 4: Factory with string parameter
    results["total"] += 1
    try:
        backend = BackendFactory.create("cpu")
        print_result(
            "Factory with string 'cpu'",
            "pass",
            f"Backend: {backend.BACKEND_NAME}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Factory with string", "fail", str(e))

    # Test 5: Get backend info
    results["total"] += 1
    try:
        info = BackendFactory.get_all_backend_info()
        backend_count = len([b for b in info.values() if b.get('available', False)])
        print_result(
            "Get all backend info",
            "pass",
            f"Total backends: {len(info)}, Available: {backend_count}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Get all backend info", "fail", str(e))

    # Test 6: Convenience function
    results["total"] += 1
    try:
        backend = get_backend("cpu")
        print_result(
            "get_backend() convenience function",
            "pass",
            f"Device: {backend.device}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("get_backend() convenience", "fail", str(e))

    return results


def demo_optimization_levels() -> dict[str, Any]:
    """Demonstrate OptimizationLevel enum and aliases."""
    print_section("Optimization Levels")

    from torchbridge.backends import OptimizationLevel

    results = {"passed": 0, "total": 0}

    # Test 1: Enum values
    results["total"] += 1
    try:
        levels = [OptimizationLevel.O0, OptimizationLevel.O1,
                  OptimizationLevel.O2, OptimizationLevel.O3]
        print_result(
            "OptimizationLevel enum",
            "pass",
            f"Levels: {', '.join(l.value for l in levels)}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("OptimizationLevel enum", "fail", str(e))

    # Test 2: String parsing
    results["total"] += 1
    try:
        level = OptimizationLevel.from_string("O2")
        print_result(
            "From string 'O2'",
            "pass",
            f"Parsed: {level.value}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("From string 'O2'", "fail", str(e))

    # Test 3: Alias parsing
    results["total"] += 1
    try:
        conservative = OptimizationLevel.from_string("conservative")
        balanced = OptimizationLevel.from_string("balanced")
        aggressive = OptimizationLevel.from_string("aggressive")
        print_result(
            "Alias parsing",
            "pass",
            f"conservative={conservative.value}, balanced={balanced.value}, aggressive={aggressive.value}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Alias parsing", "fail", str(e))

    # Test 4: Case insensitive
    results["total"] += 1
    try:
        level1 = OptimizationLevel.from_string("BALANCED")
        level2 = OptimizationLevel.from_string("Aggressive")
        print_result(
            "Case insensitive parsing",
            "pass",
            f"BALANCED={level1.value}, Aggressive={level2.value}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Case insensitive", "fail", str(e))

    return results


def demo_device_info() -> dict[str, Any]:
    """Demonstrate standardized DeviceInfo format."""
    print_section("Standardized DeviceInfo")

    from torchbridge.backends import DeviceInfo, get_backend

    results = {"passed": 0, "total": 0}

    # Test 1: CPU backend DeviceInfo
    results["total"] += 1
    try:
        backend = get_backend("cpu")
        info = backend.get_device_info()
        print_result(
            "CPU DeviceInfo",
            "pass",
            f"Backend: {info.backend}, Type: {info.device_type}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("CPU DeviceInfo", "fail", str(e))

    # Test 2: DeviceInfo to_dict
    results["total"] += 1
    try:
        backend = get_backend("cpu")
        info = backend.get_device_info()
        info_dict = info.to_dict()
        print_result(
            "DeviceInfo.to_dict()",
            "pass",
            f"Keys: {', '.join(info_dict.keys())}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("DeviceInfo.to_dict()", "fail", str(e))

    # Test 3: DeviceInfo memory properties
    results["total"] += 1
    try:
        info = DeviceInfo(
            backend="test",
            device_type="test:0",
            device_id=0,
            device_name="Test Device",
            total_memory_bytes=16 * 1024**3,  # 16GB
            is_available=True,
        )
        print_result(
            "Memory properties",
            "pass",
            f"Total: {info.total_memory_gb:.1f}GB, {info.total_memory_mb:.0f}MB",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Memory properties", "fail", str(e))

    return results


def demo_unified_interface() -> dict[str, Any]:
    """Demonstrate unified interface across all backends."""
    print_section("Unified Backend Interface")

    from torchbridge.backends import DeviceInfo, get_backend

    results = {"passed": 0, "total": 0}

    # Test model
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )

    backends_to_test = ["cpu"]

    # Check which backends are available
    try:
        from torchbridge.backends.nvidia import NVIDIABackend
        backend = NVIDIABackend()
        if backend.is_available:
            backends_to_test.append("nvidia")
    except Exception:
        pass

    for backend_name in backends_to_test:
        # Test: Create backend
        results["total"] += 1
        try:
            backend = get_backend(backend_name)
            print_result(
                f"{backend_name.upper()} backend creation",
                "pass",
                f"Device: {backend.device}",
            )
            results["passed"] += 1
        except Exception as e:
            print_result(f"{backend_name.upper()} backend creation", "fail", str(e))
            continue

        # Test: get_device_info returns DeviceInfo
        results["total"] += 1
        try:
            info = backend.get_device_info()
            assert isinstance(info, DeviceInfo)
            print_result(
                f"{backend_name.upper()} get_device_info()",
                "pass",
                f"Name: {info.device_name}",
            )
            results["passed"] += 1
        except Exception as e:
            print_result(f"{backend_name.upper()} get_device_info()", "fail", str(e))

        # Test: prepare_model
        results["total"] += 1
        try:
            test_model = nn.Linear(64, 64)
            prepared = backend.prepare_model(test_model)
            print_result(
                f"{backend_name.upper()} prepare_model()",
                "pass",
                f"Model on: {next(prepared.parameters()).device}",
            )
            results["passed"] += 1
        except Exception as e:
            print_result(f"{backend_name.upper()} prepare_model()", "fail", str(e))

        # Test: optimize_for_inference
        results["total"] += 1
        try:
            test_model = nn.Linear(64, 64)
            optimized = backend.optimize_for_inference(test_model)
            print_result(
                f"{backend_name.upper()} optimize_for_inference()",
                "pass",
                f"Training mode: {optimized.training}",
            )
            results["passed"] += 1
        except Exception as e:
            print_result(f"{backend_name.upper()} optimize_for_inference()", "fail", str(e))

        # Test: synchronize and empty_cache
        results["total"] += 1
        try:
            backend.synchronize()
            backend.empty_cache()
            print_result(
                f"{backend_name.upper()} synchronize/empty_cache",
                "pass",
            )
            results["passed"] += 1
        except Exception as e:
            print_result(f"{backend_name.upper()} synchronize/empty_cache", "fail", str(e))

    return results


def demo_backend_inheritance() -> dict[str, Any]:
    """Demonstrate that all backends inherit from BaseBackend."""
    print_section("Backend Inheritance")

    from torchbridge.backends import BaseBackend

    results = {"passed": 0, "total": 0}

    # Test each backend
    backend_classes = [
        ("NVIDIA", "torchbridge.backends.nvidia.NVIDIABackend"),
        ("AMD", "torchbridge.backends.amd.AMDBackend"),
        ("TPU", "torchbridge.backends.tpu.TPUBackend"),
        ("Intel", "torchbridge.backends.intel.IntelBackend"),
    ]

    for name, class_path in backend_classes:
        results["total"] += 1
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            backend_class = getattr(module, class_name)

            is_subclass = issubclass(backend_class, BaseBackend)
            if is_subclass:
                print_result(
                    f"{name}Backend inherits BaseBackend",
                    "pass",
                    f"BACKEND_NAME: {backend_class.BACKEND_NAME}",
                )
                results["passed"] += 1
            else:
                print_result(
                    f"{name}Backend inherits BaseBackend",
                    "fail",
                    "Not a subclass of BaseBackend",
                )
        except Exception as e:
            print_result(f"{name}Backend inherits BaseBackend", "fail", str(e))

    return results


def demo_context_manager() -> dict[str, Any]:
    """Demonstrate backend context manager usage."""
    print_section("Context Manager Usage")

    from torchbridge.backends import get_backend

    results = {"passed": 0, "total": 0}

    # Test: Context manager
    results["total"] += 1
    try:
        with get_backend("cpu") as backend:
            model = nn.Linear(64, 64)
            prepared = backend.prepare_model(model)
            output = prepared(torch.randn(1, 64))

        print_result(
            "Backend context manager",
            "pass",
            f"Output shape: {output.shape}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Backend context manager", "fail", str(e))

    # Test: to_device helper
    results["total"] += 1
    try:
        backend = get_backend("cpu")
        tensor = torch.randn(10, 10)
        tensor_on_device = backend.to_device(tensor)
        print_result(
            "to_device() helper",
            "pass",
            f"Tensor device: {tensor_on_device.device}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("to_device() helper", "fail", str(e))

    return results


def demo_optimizer_integration() -> dict[str, Any]:
    """Demonstrate unified optimizer interface."""
    print_section("Optimizer Integration")

    from torchbridge.backends import OptimizationLevel, get_optimizer

    results = {"passed": 0, "total": 0}

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    # Test 1: Get CPU optimizer
    results["total"] += 1
    try:
        optimizer = get_optimizer("cpu")
        print_result(
            "Get CPU optimizer",
            "pass",
            f"Optimizer: {optimizer.OPTIMIZER_NAME}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Get CPU optimizer", "fail", str(e))

    # Test 2: Optimize with level
    results["total"] += 1
    try:
        optimizer = get_optimizer("cpu")
        optimized, result = optimizer.optimize(model, level=OptimizationLevel.O2)
        print_result(
            "Optimize with O2 level",
            "pass",
            f"Applied: {', '.join(result.optimizations_applied)}",
        )
        results["passed"] += 1
    except Exception as e:
        print_result("Optimize with O2 level", "fail", str(e))

    # Test 3: Get available strategies
    results["total"] += 1
    try:
        optimizer = get_optimizer("cpu")
        strategies = optimizer.get_available_strategies()
        print_result(
            "Get optimization strategies",
            "pass",
            f"Strategies: {len(strategies)}",
        )
        for s in strategies[:3]:
            print(f"       - {s.name}: {s.description}")
        results["passed"] += 1
    except Exception as e:
        print_result("Get optimization strategies", "fail", str(e))

    return results


def demo_full_workflow(quick: bool = False) -> dict[str, Any]:
    """Demonstrate complete unified backend workflow."""
    print_section("Complete Unified Workflow")

    from torchbridge.backends import (
        BackendFactory,
        BackendType,
        OptimizationLevel,
    )

    results = {"passed": 0, "total": 0}

    # Create a test model
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model: int = 256, n_heads: int = 4):
            super().__init__()
            self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )

        def forward(self, x):
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            return self.ffn(x) + x

    results["total"] += 1
    try:
        # Step 1: Auto-detect and create backend
        backend = BackendFactory.create(BackendType.AUTO)
        print(f"  Step 1: Backend auto-selected: {backend.BACKEND_NAME}")

        # Step 2: Get device info
        info = backend.get_device_info()
        print(f"  Step 2: Device: {info.device_name} ({info.device_type})")

        # Step 3: Create and prepare model
        d_model = 256 if not quick else 128
        model = SimpleTransformer(d_model=d_model)
        prepared_model = backend.prepare_model(model, OptimizationLevel.O2)
        print("  Step 3: Model prepared with O2 optimization")

        # Step 4: Optimize for inference
        optimized_model = backend.optimize_for_inference(prepared_model)
        print("  Step 4: Model optimized for inference")

        # Step 5: Run inference
        batch_size = 4 if not quick else 2
        seq_len = 64 if not quick else 32
        x = backend.to_device(torch.randn(batch_size, seq_len, d_model))

        backend.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = optimized_model(x)

        backend.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        print("  Step 5: Inference complete")
        print(f"     Input: {x.shape}")
        print(f"     Output: {output.shape}")
        print(f"     Time: {elapsed:.2f}ms")

        # Step 6: Cleanup
        backend.empty_cache()
        print("  Step 6: Cache cleared")

        print_result(
            "Complete workflow",
            "pass",
            f"Backend: {backend.BACKEND_NAME}, Time: {elapsed:.2f}ms",
        )
        results["passed"] += 1

    except Exception as e:
        print_result("Complete workflow", "fail", str(e))

    return results


def main():
    """Run all unified backend demos."""
    parser = argparse.ArgumentParser(description="Unified Backend Demo")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick version with smaller models",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Unified Backend Demo (v0.5.0)")
    print("  TorchBridge - Cross-Platform Backend Unification")
    print("=" * 60)

    all_results = {
        "factory": demo_backend_factory(),
        "levels": demo_optimization_levels(),
        "device_info": demo_device_info(),
        "interface": demo_unified_interface(),
        "inheritance": demo_backend_inheritance(),
        "context": demo_context_manager(),
        "optimizer": demo_optimizer_integration(),
        "workflow": demo_full_workflow(quick=args.quick),
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
        print("\n  🎉 All unified backend demos passed!")
    else:
        print("\n  ⚠️  Some demos had issues (see details above)")

    return 0 if total_passed >= total_tests * 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
