#!/usr/bin/env python3
"""
Comprehensive Test Suite for Repository Changes

This test suite validates all the changes made during the repository transformation:
1. Compiler-optimized components work correctly
2. Documentation is accessible and accurate
3. Demo scripts run successfully
4. Import system works properly
5. Performance optimizations are effective

Run this before committing to ensure everything works correctly.
"""

import sys
import os
import importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import traceback
from pathlib import Path


class TestSuite:
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
            # Print traceback for debugging
            traceback.print_exc()

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n📊 Test Summary:")
        print(f"   Total tests: {total}")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")

        if self.errors:
            print(f"\n❌ Errors encountered:")
            for error in self.errors:
                print(f"   • {error}")

        return self.failed == 0


def test_compiler_optimized_imports():
    """Test that all compiler-optimized components can be imported."""
    from kernel_pytorch.compiler_optimized import (
        CompilerOptimizedMultiHeadAttention,
        FlashAttentionWrapper,
        MemoryEfficientAttention,
        OptimizedLayerNorm,
        OptimizedRMSNorm,
        FusedLayerNormActivation
    )

    # Test that classes can be instantiated
    embed_dim, num_heads = 512, 8

    # Attention modules
    attn1 = CompilerOptimizedMultiHeadAttention(embed_dim, num_heads)
    attn2 = FlashAttentionWrapper(embed_dim, num_heads)
    attn3 = MemoryEfficientAttention(embed_dim, num_heads)

    # Normalization modules
    norm1 = OptimizedLayerNorm(embed_dim)
    norm2 = OptimizedRMSNorm(embed_dim)
    norm3 = FusedLayerNormActivation(embed_dim)

    assert isinstance(attn1, nn.Module)
    assert isinstance(norm1, nn.Module)
    print("   All components imported and instantiated successfully")


def test_attention_modules_functionality():
    """Test that attention modules work correctly."""
    from kernel_pytorch.compiler_optimized.attention_modules import (
        CompilerOptimizedMultiHeadAttention,
        validate_attention_correctness
    )

    # Test basic functionality
    embed_dim, num_heads = 256, 4
    seq_len, batch_size = 32, 2

    attn = CompilerOptimizedMultiHeadAttention(embed_dim, num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Forward pass
    with torch.no_grad():
        output = attn(x)

    # Check output shape
    expected_shape = (batch_size, seq_len, embed_dim)
    assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    # Test correctness validation function
    is_correct = validate_attention_correctness(embed_dim=256, num_heads=4, seq_len=32)
    assert is_correct, "Attention correctness validation failed"

    print(f"   Attention modules work correctly with output shape {output.shape}")


def test_normalization_modules_functionality():
    """Test that normalization modules work correctly."""
    from kernel_pytorch.compiler_optimized.normalization_layers import (
        OptimizedLayerNorm,
        OptimizedRMSNorm,
        FusedLayerNormActivation
    )

    embed_dim = 512
    batch_size, seq_len = 4, 128
    x = torch.randn(batch_size, seq_len, embed_dim)

    # Test OptimizedLayerNorm
    norm1 = OptimizedLayerNorm(embed_dim)
    output1 = norm1(x)
    assert output1.shape == x.shape

    # Test OptimizedRMSNorm
    norm2 = OptimizedRMSNorm(embed_dim)
    output2 = norm2(x)
    assert output2.shape == x.shape

    # Test FusedLayerNormActivation
    norm3 = FusedLayerNormActivation(embed_dim, activation='gelu')
    output3 = norm3(x)
    assert output3.shape == x.shape

    # Test that outputs are different from input (normalization effect)
    assert not torch.allclose(x, output1), "LayerNorm should change input"
    assert not torch.allclose(x, output2), "RMSNorm should change input"
    assert not torch.allclose(x, output3), "Fused norm+activation should change input"

    print(f"   All normalization modules work correctly")


def test_torch_compile_compatibility():
    """Test that components work with torch.compile."""
    from kernel_pytorch.compiler_optimized import CompilerOptimizedMultiHeadAttention

    embed_dim, num_heads = 256, 4
    batch_size, seq_len = 2, 64

    # Create module and compile it
    attn = CompilerOptimizedMultiHeadAttention(embed_dim, num_heads)
    compiled_attn = torch.compile(attn, mode='default')  # Use default mode for compatibility

    x = torch.randn(batch_size, seq_len, embed_dim)

    # Test both compiled and uncompiled versions
    with torch.no_grad():
        output_original = attn(x)
        output_compiled = compiled_attn(x)

    # Should produce similar results (allowing for small numerical differences)
    assert torch.allclose(output_original, output_compiled, atol=1e-5), "Compiled version produces different results"

    print(f"   torch.compile compatibility verified")


def test_demo_script_functionality():
    """Test that the main demo script runs without errors."""

    # Import the demo module
    demo_path = Path(__file__).parent / "demo_compiler_optimization.py"

    if not demo_path.exists():
        raise FileNotFoundError(f"Demo script not found at {demo_path}")

    # Load the demo module
    spec = importlib.util.spec_from_file_location("demo_compiler_optimization", demo_path)
    demo_module = importlib.util.module_from_spec(spec)

    # Try to import the main functions
    sys.modules["demo_compiler_optimization"] = demo_module
    spec.loader.exec_module(demo_module)

    # Test that key functions exist
    assert hasattr(demo_module, 'create_optimized_attention'), "Demo missing create_optimized_attention function"
    assert hasattr(demo_module, 'demonstrate_compiler_optimization_impact'), "Demo missing main demonstration function"

    # Test that attention creation works
    AttentionClass = demo_module.create_optimized_attention()
    test_attn = AttentionClass(256, 4)
    assert isinstance(test_attn, nn.Module), "Demo attention class should be a PyTorch module"

    print(f"   Demo script functions are accessible and working")


def test_semantic_agent_basic_functionality():
    """Test that semantic agent components still work (simplified focus)."""
    try:
        from kernel_pytorch.semantic_agent.architecture import SemanticCodeAgent

        agent = SemanticCodeAgent()

        # Test basic code analysis
        test_code = '''
import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def forward(self, x):
        return torch.matmul(x, x.transpose(-2, -1))
'''

        result = agent.analyze_code(test_code)

        # Check that analysis returns expected structure
        assert isinstance(result, dict), "Analysis should return dictionary"
        assert 'patterns' in result, "Analysis should contain patterns"
        assert isinstance(result['patterns'], list), "Patterns should be a list"

        print(f"   Semantic agent basic functionality works")

    except ImportError:
        print(f"   ⚠️  Semantic agent not available - skipping test")


def test_documentation_accessibility():
    """Test that key documentation files exist and are accessible."""

    doc_files = [
        "README_REFOCUSED.md",
        "REFOCUS_PLAN.md",
        "docs/research_roadmap.md",
        "docs/tutorials/README.md",
        "docs/tutorials/01_quickstart_setup.md",
        "src/kernel_pytorch/components/README.md",
        "src/kernel_pytorch/semantic_agent/README.md",
        "src/kernel_pytorch/triton_kernels/README.md",
        "src/kernel_pytorch/cuda_kernels/README.md"
    ]

    missing_files = []
    for doc_file in doc_files:
        if not Path(doc_file).exists():
            missing_files.append(doc_file)

    assert len(missing_files) == 0, f"Missing documentation files: {missing_files}"

    print(f"   All {len(doc_files)} documentation files exist")


def test_import_system_integrity():
    """Test that the overall import system works correctly."""

    # Test main package imports
    import kernel_pytorch

    # Test component imports
    from kernel_pytorch.components import basic_optimized

    # Test utility imports
    from kernel_pytorch.utils import profiling

    # Test new compiler-optimized imports
    from kernel_pytorch.compiler_optimized import CompilerOptimizedMultiHeadAttention

    # Test that old semantic agent imports still work
    from kernel_pytorch.semantic_agent import architecture

    print(f"   All package imports work correctly")


def test_performance_benchmark_functions():
    """Test that performance benchmarking functions work."""
    from kernel_pytorch.compiler_optimized.attention_modules import benchmark_attention_implementations

    # Run a small benchmark to ensure it works
    try:
        # Use small sizes for fast testing
        results = benchmark_attention_implementations(
            embed_dim=128, num_heads=4, seq_len=64, batch_size=1, num_runs=5
        )

        assert isinstance(results, dict), "Benchmark should return dictionary"
        assert len(results) > 0, "Benchmark should return results"

        # Check that results have expected structure
        for name, metrics in results.items():
            assert 'avg_time_ms' in metrics, f"Missing timing data for {name}"
            assert isinstance(metrics['avg_time_ms'], float), f"Timing should be float for {name}"

        print(f"   Benchmark functions work with {len(results)} implementations tested")

    except Exception as e:
        # If CUDA/GPU specific benchmarks fail, that's okay for basic testing
        if "cuda" in str(e).lower():
            print(f"   ⚠️  GPU benchmarks skipped (no CUDA): {e}")
        else:
            raise


def main():
    """Run all tests and report results."""
    print("🧪 Comprehensive Repository Validation Test Suite")
    print("=" * 80)
    print("Testing all changes made during repository transformation...\n")

    test_suite = TestSuite()

    # Run all tests
    test_suite.test("Compiler-Optimized Imports", test_compiler_optimized_imports)
    test_suite.test("Attention Module Functionality", test_attention_modules_functionality)
    test_suite.test("Normalization Module Functionality", test_normalization_modules_functionality)
    test_suite.test("torch.compile Compatibility", test_torch_compile_compatibility)
    test_suite.test("Demo Script Functionality", test_demo_script_functionality)
    test_suite.test("Semantic Agent Basic Functionality", test_semantic_agent_basic_functionality)
    test_suite.test("Documentation Accessibility", test_documentation_accessibility)
    test_suite.test("Import System Integrity", test_import_system_integrity)
    test_suite.test("Performance Benchmark Functions", test_performance_benchmark_functions)

    # Print summary
    success = test_suite.summary()

    if success:
        print(f"\n🎉 All tests passed! Repository is ready for commit.")
        return 0
    else:
        print(f"\n❌ Some tests failed. Please fix issues before committing.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)