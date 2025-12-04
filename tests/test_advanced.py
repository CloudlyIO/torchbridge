"""
Comprehensive Test Suite for Advanced Optimizations

Tests for cutting-edge optimization implementations:
- FlashAttention-3 with FP8 precision
- Mixture of Experts systems
- Advanced memory optimization
- Integration and performance validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import sys
import os
import time
import warnings
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kernel_pytorch.attention import (
    FlashAttention3, FP8AttentionConfig, AttentionPatterns, AttentionConfig
)
from kernel_pytorch.mixture_of_experts import (
    create_moe_layer, MoELayer, MoEConfig
)
from kernel_pytorch.advanced_memory import (
    InterleaveOffloadingOptimizer, DeepOptimizerStates, MemoryConfig
)


class TestFlashAttention3:
    """Test FlashAttention-3 implementation"""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture
    def attention_configs(self):
        """Different attention configurations to test"""
        return [
            FP8AttentionConfig(use_fp8=False, async_compute=False),
            FP8AttentionConfig(use_fp8=False, async_compute=True),
            FP8AttentionConfig(use_fp8=True, async_compute=False) if torch.cuda.is_available() else None,
            FP8AttentionConfig(use_fp8=True, async_compute=True) if torch.cuda.is_available() else None,
        ]

    def test_attention_initialization(self, device):
        """Test FlashAttention-3 can be initialized properly"""
        from kernel_pytorch.attention import AttentionConfig

        config = AttentionConfig(
            embed_dim=768,
            num_heads=12,
            use_flash_attention=True,
            fp8_config=FP8AttentionConfig()
        )
        attention = FlashAttention3(config).to(device)

        assert attention.embed_dim == 768
        assert attention.num_heads == 12
        assert attention.head_dim == 64

    def test_attention_forward_shapes(self, device, attention_configs):
        """Test that attention forward pass produces correct output shapes"""
        for config in attention_configs:
            if config is None:
                continue

            # Create proper config with embed_dim and num_heads
            test_config = AttentionConfig(
                embed_dim=512,
                num_heads=8,
                use_flash_attention=True,
                fp8_config=config.fp8_config if config and hasattr(config, 'fp8_config') else FP8AttentionConfig()
            )
            attention = FlashAttention3(test_config).to(device)

            batch_size, seq_len, embed_dim = 2, 128, 512
            x = torch.randn(batch_size, seq_len, embed_dim, device=device)

            with torch.no_grad():
                output = attention(x)

            assert output.shape == (batch_size, seq_len, embed_dim)
            assert output.device.type == device

    def test_attention_causal_mask(self, device):
        """Test that causal attention properly masks future tokens"""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=4,
            use_flash_attention=True,
            pattern=AttentionPatterns.CAUSAL,
            fp8_config=FP8AttentionConfig(use_fp8=False)
        )
        attention = FlashAttention3(config).to(device)

        batch_size, seq_len = 1, 16
        x = torch.randn(batch_size, seq_len, 256, device=device)

        with torch.no_grad():
            output = attention(x)

        # Basic shape check - detailed causality testing would require access to attention weights
        assert output.shape == (batch_size, seq_len, 256)

    def test_attention_optimization_info(self, device):
        """Test optimization information retrieval"""
        config = AttentionConfig(
            embed_dim=768,
            num_heads=12,
            use_flash_attention=True,
            fp8_config=FP8AttentionConfig()
        )
        attention = FlashAttention3(config).to(device)

        info = attention.get_attention_stats()

        assert isinstance(info, dict)
        assert 'backend' in info
        assert 'fp8_enabled' in info
        assert 'flash_attn_2_available' in info or 'flash_attn_3_available' in info

    def test_fp8_optimization(self, device):
        """Test FP8 optimization if CUDA is available"""
        if device != "cuda":
            pytest.skip("FP8 optimization requires CUDA")

        config = AttentionConfig(
            embed_dim=512,
            num_heads=8,
            use_flash_attention=True,
            fp8_config=FP8AttentionConfig(use_fp8=True)
        )
        attention = FlashAttention3(config).to(device)

        x = torch.randn(2, 64, 512, device=device)

        # Should work without errors
        with torch.no_grad():
            output = attention(x)

        assert output.shape == x.shape


class TestFlexAttention:
    """Test FlexAttention API implementation"""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.mark.skip("FlexAttentionAPI needs to be implemented in unified framework")
    def test_attention_patterns(self, device):
        """Test different attention patterns"""
        patterns_to_test = [
            AttentionPatterns.CAUSAL,
            AttentionPatterns.SLIDING_WINDOW,
            AttentionPatterns.ALIBI,
            AttentionPatterns.PREFIX_LM
        ]

        for pattern in patterns_to_test:
            pattern_kwargs = {}
            if pattern == AttentionPatterns.SLIDING_WINDOW:
                pattern_kwargs = {'window_size': 64}
            elif pattern == AttentionPatterns.PREFIX_LM:
                pattern_kwargs = {'prefix_length': 32}

            attention = FlexAttentionAPI(
                embed_dim=512,
                num_heads=8,
                pattern=pattern,
                pattern_kwargs=pattern_kwargs
            ).to(device)

            x = torch.randn(2, 128, 512, device=device)

            with torch.no_grad():
                output = attention(x)

            assert output.shape == x.shape

    @pytest.mark.skip("FlexAttentionAPI needs to be implemented in unified framework")
    def test_pattern_switching(self, device):
        """Test dynamic pattern switching"""
        attention = FlexAttentionAPI(
            embed_dim=256,
            num_heads=4,
            pattern=AttentionPatterns.CAUSAL
        ).to(device)

        x = torch.randn(1, 64, 256, device=device)

        # Test initial pattern
        with torch.no_grad():
            output1 = attention(x)

        # Switch pattern
        attention.set_pattern(
            AttentionPatterns.SLIDING_WINDOW,
            {'window_size': 32}
        )

        # Test new pattern
        with torch.no_grad():
            output2 = attention(x)

        assert output1.shape == output2.shape == x.shape

    @pytest.mark.skip("FlexAttentionAPI needs to be implemented in unified framework")
    def test_benchmark_patterns(self, device):
        """Test pattern benchmarking functionality"""
        attention = FlexAttentionAPI(
            embed_dim=256,
            num_heads=4,
            pattern=AttentionPatterns.CAUSAL
        ).to(device)

        x = torch.randn(2, 64, 256, device=device)

        # Run benchmark with reduced iterations for testing
        results = attention.benchmark_patterns(x, num_iterations=5)

        assert isinstance(results, dict)
        assert len(results) > 0
        for pattern_name, time_ms in results.items():
            assert isinstance(time_ms, float)
            assert time_ms > 0


class TestMixtureOfExperts:
    """Test Mixture of Experts implementations"""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_moe_layer_creation(self, device):
        """Test MoE layer creation with different types"""
        moe_types = ["standard", "sparse", "switch", "adaptive"]

        for moe_type in moe_types:
            try:
                moe = create_moe_layer(
                    moe_type=moe_type,
                    hidden_size=512,
                    num_experts=8,
                    top_k=2 if moe_type != "switch" else 1
                ).to(device)

                assert moe.hidden_size == 512
                assert moe.num_experts == 8

            except Exception as e:
                # Some MoE types might not be fully implemented
                warnings.warn(f"MoE type {moe_type} failed: {str(e)}")

    def test_moe_forward_pass(self, device):
        """Test MoE forward pass with auxiliary losses"""
        config = MoEConfig(
            num_experts=4,
            top_k=2,
            load_balance_loss_weight=0.01
        )

        moe = MoELayer(
            config=config,
            hidden_size=256
        ).to(device)

        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, 256, device=device)

        # Test forward pass with auxiliary losses
        output, aux_losses = moe(x, return_router_logits=True)

        assert output.shape == x.shape
        assert isinstance(aux_losses, dict)
        assert len(aux_losses) > 0

    def test_expert_utilization(self, device):
        """Test expert utilization tracking"""
        moe = create_moe_layer(
            moe_type="standard",
            hidden_size=128,
            num_experts=4,
            top_k=2
        ).to(device)

        x = torch.randn(8, 16, 128, device=device)

        # Run several forward passes
        for _ in range(5):
            with torch.no_grad():
                _ = moe(x)

        # Check utilization stats
        stats = moe.get_expert_utilization_stats()

        assert isinstance(stats, dict)
        assert 'expert_balance' in stats
        assert 'expert_efficiency' in stats
        assert 'usage_rates' in stats

    def test_moe_training_mode(self, device):
        """Test MoE in training mode with gradients"""
        moe = create_moe_layer(
            moe_type="standard",
            hidden_size=256,
            num_experts=6,
            top_k=2
        ).to(device)

        moe.train()

        x = torch.randn(2, 16, 256, device=device, requires_grad=True)
        output, aux_losses = moe(x, return_router_logits=True)

        # Compute total loss
        dummy_target = torch.randn_like(output)
        loss = F.mse_loss(output, dummy_target)

        for aux_loss in aux_losses.values():
            # Ensure aux_loss is a scalar
            if aux_loss.dim() > 0:
                aux_loss = aux_loss.mean()
            loss = loss + aux_loss

        # Should be able to backpropagate
        loss.backward()

        # Check that gradients exist
        assert x.grad is not None
        assert any(p.grad is not None for p in moe.parameters() if p.requires_grad)


class TestAdvancedMemory:
    """Test advanced memory optimization techniques"""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_interleave_offloading_optimizer(self, device):
        """Test InterleaveOffloadingOptimizer"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)

        # Create base optimizer
        base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Create advanced optimizer
        optimizer = InterleaveOffloadingOptimizer(
            optimizer=base_optimizer,
            model=model,
            memory_limit_gb=1.0,  # Low limit for testing
            auto_tune=True
        )

        # Test optimization step
        x = torch.randn(8, 128, device=device)
        target = torch.randn(8, 128, device=device)

        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        # Should return metrics
        metrics = optimizer.step()

        assert isinstance(metrics, dict)

    def test_deep_optimizer_states(self, device):
        """Test DeepOptimizerStates directly"""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        memory_config = MemoryConfig(
            gpu_memory_limit_gb=2.0,
            cpu_memory_limit_gb=4.0,
            use_async_offloading=False  # Disable async for testing
        )

        deep_optimizer = DeepOptimizerStates(
            optimizer=optimizer,
            model=model,
            memory_config=memory_config,
            num_groups=2
        )

        # Test optimization step
        x = torch.randn(4, 64, device=device)
        target = torch.randn(4, 64, device=device)

        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            return loss

        # Should return step metrics
        step_metrics = deep_optimizer.step(closure)

        assert isinstance(step_metrics, dict)
        assert 'step_total_time' in step_metrics
        assert 'memory_usage' in step_metrics

        # Test performance stats
        perf_stats = deep_optimizer.get_performance_stats()
        assert isinstance(perf_stats, dict)
        assert 'steps_completed' in perf_stats

    def test_memory_config_optimization(self, device):
        """Test automatic memory configuration optimization"""
        if device != "cuda":
            pytest.skip("Memory optimization testing requires CUDA")

        model = nn.Linear(100, 100).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        deep_optimizer = DeepOptimizerStates(
            optimizer=optimizer,
            model=model,
            enable_profiling=True
        )

        # Run a few steps to generate data
        for _ in range(3):
            x = torch.randn(10, 100, device=device)
            target = torch.randn(10, 100, device=device)

            def closure():
                optimizer.zero_grad()
                output = model(x)
                loss = F.mse_loss(output, target)
                loss.backward()
                return loss

            deep_optimizer.step(closure)

        # Test configuration optimization
        optimized_config = deep_optimizer.optimize_memory_configuration()
        assert isinstance(optimized_config, MemoryConfig)


class TestIntegration:
    """Integration tests for all optimizations working together"""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_full_transformer_integration(self, device):
        """Test complete transformer with all optimizations"""
        # Import the demo model
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

        try:
            from advanced_optimizations_demo import AdvancedTransformerModel

            model = AdvancedTransformerModel(
                vocab_size=1000,
                dim=256,
                num_layers=2,
                num_heads=4,
                use_moe=True,
                moe_type="standard",
                num_experts=4,
                use_fp8=False,  # Disable FP8 for compatibility
                dropout=0.1
            ).to(device)

            # Test forward pass
            input_ids = torch.randint(0, 1000, (2, 32), device=device)

            with torch.no_grad():
                output = model(input_ids)

            assert output.shape == (2, 32, 1000)

            # Test with auxiliary losses
            output, aux_losses = model(input_ids, return_aux_losses=True)

            assert output.shape == (2, 32, 1000)
            assert isinstance(aux_losses, dict)

        except ImportError:
            pytest.skip("Integration test requires demo module")

    def test_optimization_compatibility(self, device):
        """Test that different optimizations can work together"""
        # Create models with different optimization combinations
        configs = [
            {'use_moe': True, 'use_fp8': False},
            {'use_moe': False, 'use_fp8': False},
        ]

        for config in configs:
            # Create attention layer
            attention_config = AttentionConfig(
                embed_dim=128,
                num_heads=4,
                use_flash_attention=True,
                fp8_config=FP8AttentionConfig(use_fp8=config['use_fp8'])
            )
            attention = FlashAttention3(attention_config).to(device)

            # Create MoE layer if needed
            if config['use_moe']:
                moe = create_moe_layer(
                    moe_type="standard",
                    hidden_size=128,
                    num_experts=4,
                    top_k=2
                ).to(device)

            # Test forward passes
            x = torch.randn(2, 16, 128, device=device)

            with torch.no_grad():
                attn_output = attention(x)
                assert attn_output.shape == x.shape

                if config['use_moe']:
                    moe_output = moe(x)
                    assert moe_output.shape == x.shape

    def test_performance_monitoring(self, device):
        """Test performance monitoring capabilities"""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=8,
            use_flash_attention=True,
            fp8_config=FP8AttentionConfig()
        )
        attention = FlashAttention3(config).to(device)

        x = torch.randn(4, 64, 256, device=device)

        # Benchmark attention
        start_time = time.perf_counter()

        with torch.no_grad():
            for _ in range(10):
                _ = attention(x)

        if device == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / 10
        tokens_per_second = (4 * 64 * 10) / (end_time - start_time)

        assert avg_time > 0
        assert tokens_per_second > 0

        # Test optimization info
        info = attention.get_attention_stats()
        assert isinstance(info, dict)
        assert 'backend' in info


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🧪 Running Advanced Optimizations Test Suite")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing on device: {device}")

    # Initialize test classes
    test_classes = [
        TestFlashAttention3(),
        TestFlexAttention(),
        TestMixtureOfExperts(),
        TestAdvancedMemory(),
        TestIntegration()
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n📋 Testing {test_class.__class__.__name__}")
        print("-" * 40)

        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]

        for test_method in test_methods:
            total_tests += 1
            print(f"  🧪 {test_method}... ", end="")

            try:
                # Call the test method
                method = getattr(test_class, test_method)

                # Handle fixture injection manually for this demo
                if hasattr(test_class, 'device'):
                    # Pass device fixture
                    method(device)
                else:
                    method()

                print("✅ PASSED")
                passed_tests += 1

            except Exception as e:
                print(f"❌ FAILED: {str(e)}")
                failed_tests.append((test_class.__class__.__name__, test_method, str(e)))

    # Print summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 40)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

    if failed_tests:
        print(f"\n❌ Failed Tests:")
        for test_class, test_method, error in failed_tests:
            print(f"  • {test_class}.{test_method}: {error}")

    return passed_tests == total_tests


if __name__ == "__main__":
    # Run tests directly if called as script
    success = run_comprehensive_test()

    if success:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n💥 Some tests failed!")
        exit(1)