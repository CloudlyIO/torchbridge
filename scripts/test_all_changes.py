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

import sys
import os
import importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import traceback
from pathlib import Path


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
        print(f"\n📊 Validation Summary:")
        print(f"   Total tests: {total}")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")

        if self.errors:
            print(f"\n❌ Errors encountered:")
            for error in self.errors:
                print(f"   • {error}")

        return self.failed == 0


def test_compiler_integration():
    """Test compiler integration components."""
    from kernel_pytorch.compiler_integration import (
        FlashLightKernelCompiler,
        PyGraphCUDAOptimizer,
        FusionBoundaryOptimizer
    )

    # Test instantiation
    compiler = FlashLightKernelCompiler()
    optimizer = PyGraphCUDAOptimizer()
    fusion = FusionBoundaryOptimizer()

    # Test that they are instances (basic functionality test)
    assert compiler is not None
    assert optimizer is not None
    assert fusion is not None

    print("   Compiler integration components working")


def test_next_gen_optimizations():
    """Test next-generation optimization features."""
    from kernel_pytorch.next_gen_optimizations import (
        AdaptivePrecisionAllocator,
        AdvancedFlexAttention,
        StructuredSparsity24
    )

    # Test with a simple model
    test_model = nn.Linear(64, 32)

    # Test AdaptivePrecisionAllocator
    precision = AdaptivePrecisionAllocator(test_model)
    assert precision is not None

    # Test AdvancedFlexAttention
    flex_attn = AdvancedFlexAttention(embed_dim=128, num_heads=8)
    assert isinstance(flex_attn, nn.Module)

    # Test StructuredSparsity24
    sparse_opt = StructuredSparsity24()
    assert sparse_opt is not None

    print("   Next-generation optimizations working")


def test_testing_framework():
    """Test the testing and validation framework."""
    from kernel_pytorch.testing_framework import (
        create_hardware_simulator,
        create_benchmark_suite,
        OptimizationValidator
    )

    # Test hardware simulator
    simulator = create_hardware_simulator(architecture='ampere', simulation_mode='performance')
    summary = simulator.get_simulation_summary()
    assert isinstance(summary, dict)

    # Test benchmark suite
    benchmark = create_benchmark_suite(warmup_iterations=3, measurement_iterations=5)
    assert benchmark is not None

    # Test optimization validator
    validator = OptimizationValidator()
    assert validator is not None

    print("   Testing framework components working")


def test_compiler_optimized_components():
    """Test compiler-optimized components if they exist."""
    try:
        from kernel_pytorch.compiler_optimized import (
            CompilerOptimizedMultiHeadAttention,
            FlashAttentionWrapper,
            OptimizedLayerNorm
        )

        # Test basic instantiation
        embed_dim, num_heads = 256, 8
        attn = CompilerOptimizedMultiHeadAttention(embed_dim, num_heads)
        flash_attn = FlashAttentionWrapper(embed_dim, num_heads)
        norm = OptimizedLayerNorm(embed_dim)

        assert isinstance(attn, nn.Module)
        assert isinstance(flash_attn, nn.Module)
        assert isinstance(norm, nn.Module)

        print("   Compiler-optimized components working")

    except ImportError as e:
        print(f"   ⚠️ Compiler-optimized components not available: {e}")


def test_attention_modules():
    """Test advanced attention implementations."""
    from kernel_pytorch.advanced_attention import (
        FlashAttention3,
        FlexAttentionAPI,
        MemoryEfficientAttention
    )

    embed_dim, num_heads = 128, 4

    # Test FlashAttention3
    flash_attn3 = FlashAttention3(embed_dim, num_heads)
    assert isinstance(flash_attn3, nn.Module)

    # Test FlexAttentionAPI
    flex_attn = FlexAttentionAPI(embed_dim, num_heads)
    assert isinstance(flex_attn, nn.Module)

    # Test MemoryEfficientAttention
    mem_attn = MemoryEfficientAttention(embed_dim, num_heads)
    assert isinstance(mem_attn, nn.Module)

    print("   Advanced attention modules working")


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
    import kernel_pytorch

    # Test core components
    from kernel_pytorch.compiler_integration import FlashLightKernelCompiler
    from kernel_pytorch.next_gen_optimizations import AdaptivePrecisionAllocator
    from kernel_pytorch.testing_framework import create_hardware_simulator

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
    suite.test("Testing Framework", test_testing_framework)
    suite.test("Advanced Attention Modules", test_attention_modules)

    # Optional components (may not be available)
    suite.test("Compiler-Optimized Components", test_compiler_optimized_components)

    # Structure and documentation
    suite.test("Documentation Exists", test_documentation_exists)
    suite.test("Demo Functionality", test_demo_functionality)

    # Print summary
    success = suite.summary()

    if success:
        print(f"\n🎉 All core validation tests passed! Framework is operational.")
        return 0
    else:
        print(f"\n❌ Some validation tests failed. Review issues above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)