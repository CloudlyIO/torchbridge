#!/usr/bin/env python3
"""
Core functionality tests for next-generation optimizations
Tests the essential functionality without complex dependencies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from kernel_pytorch.next_gen_optimizations import (
    FlashLightCompiler,
    AdvancedFlexAttention,
    StructuredSparsity24,
    FP4Quantizer,
    create_advanced_flex_attention,
    create_structured_sparsity_optimizer
)

def test_flashlight_compiler():
    """Test FlashLight compiler creation and compilation"""
    print("Testing FlashLight Compiler...")

    compiler = FlashLightCompiler()

    # Test kernel compilation
    kernel = compiler.compile_attention_kernel("differential", 128, 64)
    assert callable(kernel), "Compiled kernel should be callable"

    # Test execution
    q = torch.randn(2, 8, 128, 64)
    k = torch.randn(2, 8, 128, 64)
    v = torch.randn(2, 8, 128, 64)

    output = kernel(q, k, v)
    assert output.shape == (2, 8, 128, 64), f"Expected (2, 8, 128, 64), got {output.shape}"

    print("✅ FlashLight Compiler working")

def test_advanced_flex_attention():
    """Test advanced FlexAttention functionality"""
    print("Testing Advanced FlexAttention...")

    attention = AdvancedFlexAttention(
        embed_dim=256,
        num_heads=4,
        pattern="standard"
    )

    x = torch.randn(2, 64, 256)
    output = attention(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN values"

    # Test performance tracking
    output, stats = attention(x, return_performance_stats=True)
    assert 'avg_forward_time' in stats, "Performance stats missing"

    print("✅ Advanced FlexAttention working")

def test_structured_sparsity():
    """Test structured sparsity functionality"""
    print("Testing Structured Sparsity...")

    sparsity = StructuredSparsity24()

    # Test 2:4 pattern creation
    tensor = torch.randn(32, 128)
    sparse_tensor, mask = sparsity.create_24_pattern(tensor)

    assert sparse_tensor.shape == tensor.shape, "Shape mismatch after sparsity"
    assert mask.shape == tensor.shape, "Mask shape mismatch"

    # Test sparsity ratio (should be around 0.5 for 2:4 pattern)
    actual_sparsity = 1.0 - (sparse_tensor != 0).float().mean().item()
    assert 0.4 <= actual_sparsity <= 0.6, f"Sparsity {actual_sparsity} not in expected range"

    print("✅ Structured Sparsity working")

def test_fp4_quantizer():
    """Test FP4 quantization functionality"""
    print("Testing FP4 Quantizer...")

    quantizer = FP4Quantizer(use_double_quantization=False)

    # Test forward pass in eval mode
    quantizer.eval()
    x = torch.randn(16, 64)
    output = quantizer(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"

    # Test training mode
    quantizer.train()
    output_train = quantizer(x)
    assert output_train.shape == x.shape, "Training mode shape mismatch"

    print("✅ FP4 Quantizer working")

def test_factory_functions():
    """Test factory functions"""
    print("Testing Factory Functions...")

    # Test advanced attention factory
    attention = create_advanced_flex_attention(
        embed_dim=128,
        num_heads=4,
        pattern="standard"
    )

    x = torch.randn(2, 32, 128)
    output = attention(x)
    assert output.shape == x.shape, "Factory-created attention failed"

    # Test sparsity optimizer factory
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 32)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    sparsity_opt = create_structured_sparsity_optimizer(
        model,
        sparsity_config={'target_sparsity': 0.3}
    )

    assert sparsity_opt.target_sparsity == 0.3, "Sparsity configuration failed"

    print("✅ Factory Functions working")

def test_integration():
    """Test integration of multiple optimizations"""
    print("Testing Integration...")

    # Create model with multiple optimizations
    class OptimizedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = create_advanced_flex_attention(
                embed_dim=128,
                num_heads=4
            )
            self.linear = nn.Linear(128, 128)

        def forward(self, x):
            x = self.attention(x)
            x = self.linear(x)
            return x

    model = OptimizedModel()
    x = torch.randn(2, 32, 128)

    # Test forward pass
    output = model(x)
    assert output.shape == x.shape, "Integrated model failed"
    assert not torch.isnan(output).any(), "Integrated model output contains NaN"

    print("✅ Integration working")

def main():
    """Run all core functionality tests"""
    print("🧪 Running Core Functionality Tests for Next-Gen Optimizations\n")

    tests = [
        test_flashlight_compiler,
        test_advanced_flex_attention,
        test_structured_sparsity,
        test_fp4_quantizer,
        test_factory_functions,
        test_integration
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"🎯 Success rate: {passed/(passed+failed)*100:.1f}%")

    if failed == 0:
        print("\n🎉 All core functionality tests passed!")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)