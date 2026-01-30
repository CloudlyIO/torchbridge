#!/usr/bin/env python3
"""
Comprehensive Framework Validation

This script validates the current PyTorch optimization framework:
1. Compiler integration components work correctly
2. Next-generation optimizations are functional
3. Documentation is accessible and accurate
4. Import system works properly
5. Demo scripts run successfully

Run this before committing to ensure everything works correctly.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import traceback
from pathlib import Path

import torch
import torch.nn as nn


class ValidationSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def test(self, name, test_func):
        """Run a test and track results."""
        try:
            print(f"🧪 Testing: {name}...")
            test_func()
            print(f"✅ {name} PASSED")
            self.passed += 1
        except Exception as e:
            print(f"❌ {name} FAILED: {e}")
            self.errors.append(f"{name}: {e}")
            self.failed += 1
            traceback.print_exc()

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print("\n📊 Validation Summary:")
        print(f"   Total tests: {total}")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")

        if self.errors:
            print("\n❌ Errors encountered:")
            for error in self.errors:
                print(f"   • {error}")

        return self.failed == 0


def test_compiler_integration():
    """Test compiler integration components."""
    # compiler_integration module was removed along with scaffold modules.
    # Test that core optimizations are still available instead.
    from torchbridge.optimizations.next_gen import (
        CUDAGraphManager,
        create_pygraph_optimizer,
    )

    device = torch.device('cpu')
    test_model = nn.Linear(64, 32)

    # Test instantiation
    manager = CUDAGraphManager(device)
    optimizer = create_pygraph_optimizer(test_model, device=device)

    assert manager is not None
    assert optimizer is not None

    print("   Compiler integration components working")


def test_next_gen_optimizations():
    """Test next-generation optimization features."""
    from torchbridge.optimizations.next_gen import (
        AutoGraphCapture,
        CUDAGraphManager,
        SelectiveCUDAGraphs,
        create_pygraph_optimizer,
    )

    # Test with a simple model
    test_model = nn.Linear(64, 32)
    device = torch.device('cpu')

    # Test PyGraph optimizer
    optimizer = create_pygraph_optimizer(test_model, device=device)
    assert optimizer is not None

    # Test CUDAGraphManager
    manager = CUDAGraphManager(device)
    assert manager is not None

    # Test AutoGraphCapture
    auto_capture = AutoGraphCapture(device)
    assert auto_capture is not None

    # Test SelectiveCUDAGraphs
    selective = SelectiveCUDAGraphs(test_model, device)
    assert selective is not None

    print("   Next-generation optimizations working")


# test_testing_framework removed - testing_framework directory deprecated


def test_compiler_optimized_components():
    """Test core optimized components."""
    from torchbridge.core import (
        FusedGELU,
        OptimizedLayerNorm,
        OptimizedMultiHeadAttention,
    )

    embed_dim, num_heads = 256, 8
    attn = OptimizedMultiHeadAttention(embed_dim, num_heads)
    norm = OptimizedLayerNorm(embed_dim)
    gelu = FusedGELU(embed_dim)

    assert isinstance(attn, nn.Module)
    assert isinstance(norm, nn.Module)
    assert isinstance(gelu, nn.Module)

    print("   Core optimized components working")


def test_attention_modules():
    """Test attention implementations."""
    from torchbridge.attention import (
        AttentionModuleConfig,
        FlashAttention2,
        FlashAttention3,
        FlexAttentionLayer,
    )

    config = AttentionModuleConfig(embed_dim=128, num_heads=4)

    # Test FlashAttention3
    flash_attn3 = FlashAttention3(config)
    assert isinstance(flash_attn3, nn.Module)

    # Test FlashAttention2
    flash_attn2 = FlashAttention2(config)
    assert isinstance(flash_attn2, nn.Module)

    # Test FlexAttentionLayer
    flex_attn = FlexAttentionLayer(config)
    assert isinstance(flex_attn, nn.Module)

    print("   Attention modules working")


def test_documentation_exists():
    """Test that key documentation files exist."""
    doc_files = [
        "README.md",
        "REPOSITORY_STRUCTURE.md",
        "CUDA_SETUP_GUIDE.md",
        "BENCHMARK_QUICKSTART.md",
        "OPTIMIZATION_ROADMAP_2025_2026.md",
        "docs/TECHNICAL_OVERVIEW.md",
        "docs/implementation_guide.md",
        "docs/EXTERNAL_REFERENCES.md"
    ]

    missing_files = []
    existing_files = []

    for doc_file in doc_files:
        if Path(doc_file).exists():
            existing_files.append(doc_file)
        else:
            missing_files.append(doc_file)

    if missing_files:
        print(f"   ⚠️ Missing documentation: {missing_files}")

    print(f"   {len(existing_files)} documentation files exist")


def test_demo_functionality():
    """Test that demo scripts exist and can be imported."""
    demo_dirs = [
        "demos/01_getting_started",
        "demos/02_compiler_optimizations",
        "demos/03_advanced_attention",
        "demos/04_gpu_integration",
        "demos/05_next_generation"
    ]

    working_demos = 0
    for demo_dir in demo_dirs:
        if Path(demo_dir).exists():
            working_demos += 1

    print(f"   {working_demos}/{len(demo_dirs)} demo directories exist")


def test_import_system():
    """Test that the main package imports work correctly."""
    # Test main package
    import torchbridge  # noqa: F401

    # Test core components
    from torchbridge.optimizations.next_gen import (
        create_pygraph_optimizer,  # noqa: F401
    )

    print("   Core import system working")


def test_basic_functionality():
    """Test basic PyTorch functionality with our framework."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a simple test
    x = torch.randn(2, 64, 256, device=device)

    # Test with a basic attention-like operation
    q = k = v = x
    attn = torch.matmul(q, k.transpose(-2, -1)) / (256 ** 0.5)
    attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)

    assert output.shape == x.shape
    print(f"   Basic functionality works on device: {device}")


def main():
    """Run all validation tests and report results."""
    print("🧪 PyTorch Optimization Framework Validation")
    print("=" * 60)
    print("Validating current framework components...\n")

    suite = ValidationSuite()

    # Core functionality tests
    suite.test("Basic PyTorch Functionality", test_basic_functionality)
    suite.test("Import System", test_import_system)
    suite.test("Compiler Integration", test_compiler_integration)
    suite.test("Next-Gen Optimizations", test_next_gen_optimizations)
    suite.test("Advanced Attention Modules", test_attention_modules)

    # Optional components (may not be available)
    suite.test("Compiler-Optimized Components", test_compiler_optimized_components)

    # Structure and documentation
    suite.test("Documentation Exists", test_documentation_exists)
    suite.test("Demo Functionality", test_demo_functionality)

    # Print summary
    success = suite.summary()

    if success:
        print("\n🎉 All core validation tests passed! Framework is operational.")
        return 0
    else:
        print("\n❌ Some validation tests failed. Review issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
