"""
Simplified Test Suite for Advanced Optimizations

Basic validation tests for all new optimization implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that all modules can be imported"""
    print("🔍 Testing imports...")

    try:
        from kernel_pytorch.advanced_attention import FlashAttention3, FP8AttentionConfig
        print("  ✅ FlashAttention3 import successful")
    except Exception as e:
        print(f"  ❌ FlashAttention3 import failed: {e}")
        return False

    try:
        from kernel_pytorch.advanced_attention import FlexAttentionAPI, AttentionPatterns
        print("  ✅ FlexAttention import successful")
    except Exception as e:
        print(f"  ❌ FlexAttention import failed: {e}")
        return False

    try:
        from kernel_pytorch.mixture_of_experts import create_moe_layer, MoEConfig
        print("  ✅ MoE import successful")
    except Exception as e:
        print(f"  ❌ MoE import failed: {e}")
        return False

    try:
        from kernel_pytorch.advanced_memory import InterleaveOffloadingOptimizer
        print("  ✅ Advanced memory import successful")
    except Exception as e:
        print(f"  ❌ Advanced memory import failed: {e}")
        return False

    return True

def test_flashattention3():
    """Test FlashAttention-3 basic functionality"""
    print("\n🔥 Testing FlashAttention-3...")

    try:
        from kernel_pytorch.advanced_attention import FlashAttention3, FP8AttentionConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = FP8AttentionConfig(use_fp8=False)  # Disable FP8 for compatibility

        attention = FlashAttention3(
            embed_dim=256,
            num_heads=4,
            config=config
        ).to(device)

        # Test forward pass
        x = torch.randn(2, 32, 256, device=device)
        with torch.no_grad():
            output = attention(x)

        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        print("  ✅ Basic forward pass successful")

        # Test optimization info
        info = attention.get_optimization_info()
        assert isinstance(info, dict), "Optimization info should be a dict"
        print("  ✅ Optimization info retrieval successful")

        return True

    except Exception as e:
        print(f"  ❌ FlashAttention-3 test failed: {e}")
        return False

def test_flex_attention():
    """Test FlexAttention functionality"""
    print("\n🔄 Testing FlexAttention...")

    try:
        from kernel_pytorch.advanced_attention import FlexAttentionAPI, AttentionPatterns

        device = "cuda" if torch.cuda.is_available() else "cpu"

        attention = FlexAttentionAPI(
            embed_dim=128,
            num_heads=2,
            pattern=AttentionPatterns.CAUSAL
        ).to(device)

        # Test forward pass
        x = torch.randn(1, 16, 128, device=device)
        with torch.no_grad():
            output = attention(x)

        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        print("  ✅ Basic forward pass successful")

        # Test pattern switching
        attention.set_pattern(AttentionPatterns.SLIDING_WINDOW, {'window_size': 8})
        with torch.no_grad():
            output2 = attention(x)

        assert output2.shape == x.shape, "Shape should remain consistent after pattern change"
        print("  ✅ Pattern switching successful")

        return True

    except Exception as e:
        print(f"  ❌ FlexAttention test failed: {e}")
        return False

def test_moe():
    """Test Mixture of Experts functionality"""
    print("\n🎯 Testing MoE...")

    try:
        from kernel_pytorch.mixture_of_experts import create_moe_layer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        moe = create_moe_layer(
            moe_type="standard",
            hidden_size=128,
            num_experts=4,
            top_k=2
        ).to(device)

        # Test forward pass
        x = torch.randn(2, 8, 128, device=device)
        with torch.no_grad():
            output = moe(x)

        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        print("  ✅ Basic forward pass successful")

        # Test with auxiliary losses
        output, aux_losses = moe(x, return_router_logits=True)
        assert isinstance(aux_losses, dict), "Auxiliary losses should be a dict"
        assert len(aux_losses) > 0, "Should have auxiliary losses"
        print("  ✅ Auxiliary losses successful")

        # Test expert utilization
        stats = moe.get_expert_utilization_stats()
        assert isinstance(stats, dict), "Stats should be a dict"
        print("  ✅ Expert utilization tracking successful")

        return True

    except Exception as e:
        print(f"  ❌ MoE test failed: {e}")
        return False

def test_advanced_memory():
    """Test advanced memory optimization"""
    print("\n🧠 Testing Advanced Memory...")

    try:
        from kernel_pytorch.advanced_memory import InterleaveOffloadingOptimizer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create simple model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)

        # Create optimizers
        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        advanced_optimizer = InterleaveOffloadingOptimizer(
            optimizer=base_optimizer,
            model=model,
            memory_limit_gb=1.0,
            auto_tune=False  # Disable auto-tune for testing
        )

        # Test optimization step
        x = torch.randn(4, 64, device=device)
        target = torch.randn(4, 64, device=device)

        advanced_optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        metrics = advanced_optimizer.step()
        assert isinstance(metrics, dict), "Step should return metrics dict"
        print("  ✅ Basic optimization step successful")

        # Test state dict operations
        state_dict = advanced_optimizer.state_dict()
        assert isinstance(state_dict, dict), "State dict should be a dict"
        print("  ✅ State dict operations successful")

        return True

    except Exception as e:
        print(f"  ❌ Advanced memory test failed: {e}")
        return False

def test_integration():
    """Test integration of multiple optimizations"""
    print("\n🔗 Testing Integration...")

    try:
        from kernel_pytorch.advanced_attention import FlashAttention3, FP8AttentionConfig
        from kernel_pytorch.mixture_of_experts import create_moe_layer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create components
        attention = FlashAttention3(
            embed_dim=128,
            num_heads=4,
            config=FP8AttentionConfig(use_fp8=False)
        ).to(device)

        moe = create_moe_layer(
            moe_type="standard",
            hidden_size=128,
            num_experts=4,
            top_k=2
        ).to(device)

        # Test combined forward pass
        x = torch.randn(2, 16, 128, device=device)

        with torch.no_grad():
            # Attention first
            attn_output = attention(x)

            # Then MoE
            final_output = moe(attn_output)

        assert final_output.shape == x.shape, "Final output shape should match input"
        print("  ✅ Combined attention + MoE successful")

        return True

    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def benchmark_performance():
    """Basic performance benchmark"""
    print("\n⚡ Running Performance Benchmark...")

    if not torch.cuda.is_available():
        print("  ⏭️  Skipping performance test (CUDA not available)")
        return True

    try:
        from kernel_pytorch.advanced_attention import FlashAttention3, FP8AttentionConfig

        device = "cuda"

        # Create attention layers
        configs = {
            'standard': FP8AttentionConfig(use_fp8=False, async_compute=False),
            'async': FP8AttentionConfig(use_fp8=False, async_compute=True),
        }

        results = {}

        for config_name, config in configs.items():
            attention = FlashAttention3(
                embed_dim=512,
                num_heads=8,
                config=config
            ).to(device)

            x = torch.randn(4, 256, 512, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = attention(x)

            torch.cuda.synchronize()

            # Benchmark
            start_time = time.perf_counter()

            with torch.no_grad():
                for _ in range(20):
                    _ = attention(x)

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            avg_time = (end_time - start_time) / 20 * 1000  # ms
            tokens_per_sec = (4 * 256 * 20) / (end_time - start_time)

            results[config_name] = {
                'avg_time_ms': avg_time,
                'tokens_per_sec': tokens_per_sec
            }

            print(f"  🚀 {config_name}: {avg_time:.2f}ms, {tokens_per_sec:.0f} tokens/s")

        return True

    except Exception as e:
        print(f"  ❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Advanced Optimizations Test Suite")
    print("=" * 60)

    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Running on: {device_info}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    tests = [
        ("Import Tests", test_basic_imports),
        ("FlashAttention-3", test_flashattention3),
        ("FlexAttention", test_flex_attention),
        ("MoE Systems", test_moe),
        ("Advanced Memory", test_advanced_memory),
        ("Integration", test_integration),
        ("Performance", benchmark_performance)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")

    print(f"\n📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")

    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("💥 SOME TESTS FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)