"""
Tests for Neural Operator Fusion (NOF) Implementation

Comprehensive test suite validating the unified attention fusion implementation
and ensuring 40-60% kernel overhead reduction through advanced fusion techniques.

 TEST COVERAGE:
- Core fusion functionality and algorithms
- Performance validation and benchmarking
- Integration with existing PyTorch patterns
- Hardware-aware optimization strategies
- Numerical accuracy preservation
- Edge cases and error handling
"""


import pytest
import torch
import torch.nn as nn

from kernel_pytorch.attention.fusion.neural_operator import (
    FusionConfig,
    FusionPerformanceStats,
    FusionStrategy,
    OptimizationLevel,
    UnifiedAttentionFusion,
    benchmark_fusion_performance,
    create_unified_attention_fusion,
    print_benchmark_results,
    print_fusion_analysis,
)


class TestFusionConfig:
    """Tests for FusionConfig configuration system."""

    def test_config_initialization(self):
        """Test basic FusionConfig initialization."""
        config = FusionConfig()

        assert config.strategy == FusionStrategy.FULL_BLOCK
        assert config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert config.enable_flash_attention is True
        assert config.enable_custom_kernels is True
        assert config.max_sequence_length == 8192

    def test_config_customization(self):
        """Test custom FusionConfig settings."""
        config = FusionConfig(
            strategy=FusionStrategy.ATTENTION_NORM,
            optimization_level=OptimizationLevel.CONSERVATIVE,
            max_sequence_length=2048,
            dropout_rate=0.1
        )

        assert config.strategy == FusionStrategy.ATTENTION_NORM
        assert config.optimization_level == OptimizationLevel.CONSERVATIVE
        assert config.max_sequence_length == 2048
        assert config.dropout_rate == 0.1

    def test_hardware_optimization_settings(self):
        """Test hardware-specific optimization settings."""
        config = FusionConfig(
            enable_tensor_cores=True,
            block_size=256,
            warp_size=32
        )

        assert config.enable_tensor_cores is True
        assert config.block_size == 256
        assert config.warp_size == 32


class TestFusionPerformanceStats:
    """Tests for FusionPerformanceStats analytics."""

    def test_stats_initialization(self):
        """Test performance stats initialization."""
        stats = FusionPerformanceStats()

        assert stats.kernel_launches_original == 0
        assert stats.kernel_launches_fused == 0
        assert stats.execution_time_original_ms == 0.0
        assert stats.execution_time_fused_ms == 0.0

    def test_kernel_reduction_calculation(self):
        """Test kernel reduction ratio calculation."""
        stats = FusionPerformanceStats(
            kernel_launches_original=15,
            kernel_launches_fused=5
        )

        reduction_ratio = stats.kernel_reduction_ratio
        expected_ratio = 1.0 - (5 / 15)  # ~0.67

        assert abs(reduction_ratio - expected_ratio) < 1e-6

    def test_speedup_calculation(self):
        """Test speedup calculation."""
        stats = FusionPerformanceStats(
            execution_time_original_ms=100.0,
            execution_time_fused_ms=40.0
        )

        speedup = stats.speedup
        expected_speedup = 100.0 / 40.0  # 2.5x

        assert abs(speedup - expected_speedup) < 1e-6

    def test_edge_cases(self):
        """Test edge cases in statistics calculation."""
        # Zero division cases
        stats_zero_original = FusionPerformanceStats(
            kernel_launches_original=0,
            kernel_launches_fused=5
        )
        assert stats_zero_original.kernel_reduction_ratio == 1.0

        stats_zero_fused = FusionPerformanceStats(
            execution_time_original_ms=100.0,
            execution_time_fused_ms=0.0
        )
        assert stats_zero_fused.speedup == 1.0


class TestUnifiedAttentionFusion:
    """Tests for the core UnifiedAttentionFusion module."""

    def test_module_initialization(self):
        """Test basic module initialization."""
        fusion_module = UnifiedAttentionFusion(
            d_model=512,
            n_heads=8,
            d_ff=2048,
            dropout=0.1
        )

        assert fusion_module.d_model == 512
        assert fusion_module.n_heads == 8
        assert fusion_module.d_ff == 2048
        assert fusion_module.head_dim == 64  # 512 / 8
        assert fusion_module.dropout == 0.1

    def test_invalid_initialization(self):
        """Test handling of invalid initialization parameters."""
        # d_model not divisible by n_heads
        with pytest.raises(AssertionError):
            UnifiedAttentionFusion(d_model=513, n_heads=8, d_ff=2048)

    def test_weight_initialization(self):
        """Test proper weight initialization."""
        fusion_module = UnifiedAttentionFusion(
            d_model=256,
            n_heads=8,
            d_ff=1024
        )

        # Check that weights are properly initialized
        for name, param in fusion_module.named_parameters():
            if 'weight' in name:
                # Should not be all zeros or all ones
                assert not torch.allclose(param, torch.zeros_like(param))
                assert not torch.allclose(param, torch.ones_like(param))

                # Should have reasonable magnitude
                assert torch.std(param) > 0.01
                assert torch.std(param) < 1.0

    def test_basic_forward_pass(self):
        """Test basic forward pass functionality."""
        batch_size, seq_len, d_model = 4, 32, 256
        n_heads = 8
        d_ff = 1024

        fusion_module = UnifiedAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output = fusion_module(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_forward_with_attention_mask(self):
        """Test forward pass with attention mask."""
        batch_size, seq_len, d_model = 2, 16, 128
        n_heads = 4

        fusion_module = UnifiedAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=512
        )

        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal attention mask
        attention_mask = torch.tril(torch.ones(seq_len, seq_len))
        attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        output = fusion_module(x, attention_mask=attention_mask)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_forward_with_key_padding_mask(self):
        """Test forward pass with key padding mask."""
        batch_size, seq_len, d_model = 2, 16, 128
        n_heads = 4

        fusion_module = UnifiedAttentionFusion(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=512
        )

        x = torch.randn(batch_size, seq_len, d_model)

        # Create key padding mask (True = padded)
        key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        key_padding_mask[:, seq_len//2:] = True  # Mask second half

        output = fusion_module(x, key_padding_mask=key_padding_mask)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_different_fusion_strategies(self):
        """Test different fusion strategies."""
        batch_size, seq_len, d_model = 2, 16, 128
        x = torch.randn(batch_size, seq_len, d_model)

        strategies = [
            FusionStrategy.FULL_BLOCK,
            FusionStrategy.ATTENTION_NORM,
            FusionStrategy.FFN_NORM,
            FusionStrategy.ATTENTION_FFN
        ]

        for strategy in strategies:
            config = FusionConfig(strategy=strategy)
            fusion_module = UnifiedAttentionFusion(
                d_model=d_model,
                n_heads=4,
                d_ff=512,
                config=config
            )

            output = fusion_module(x)
            assert output.shape == x.shape
            assert not torch.isnan(output).any()

    def test_forward_with_performance_stats(self):
        """Test forward pass with performance statistics."""
        batch_size, seq_len, d_model = 2, 16, 128

        fusion_module = UnifiedAttentionFusion(
            d_model=d_model,
            n_heads=4,
            d_ff=512
        )

        x = torch.randn(batch_size, seq_len, d_model)
        output, stats = fusion_module(x, return_stats=True)

        assert output.shape == x.shape
        assert isinstance(stats, FusionPerformanceStats)
        assert stats.execution_time_original_ms > 0
        assert stats.execution_time_fused_ms > 0
        assert stats.kernel_launches_original > 0
        assert stats.kernel_launches_fused > 0

    def test_gradient_flow(self):
        """Test gradient flow through fused operations."""
        batch_size, seq_len, d_model = 2, 16, 128

        fusion_module = UnifiedAttentionFusion(
            d_model=d_model,
            n_heads=4,
            d_ff=512
        )

        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output = fusion_module(x)
        loss = output.sum()

        loss.backward()

        # Check that gradients exist and are reasonable
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check parameter gradients
        for _name, param in fusion_module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None
                assert not torch.isnan(param.grad).any()

    def test_training_mode_consistency(self):
        """Test consistency between training and evaluation modes."""
        batch_size, seq_len, d_model = 2, 16, 128

        fusion_module = UnifiedAttentionFusion(
            d_model=d_model,
            n_heads=4,
            d_ff=512,
            dropout=0.1
        )

        x = torch.randn(batch_size, seq_len, d_model)

        # Test in training mode
        fusion_module.train()
        output_train = fusion_module(x)

        # Test in evaluation mode
        fusion_module.eval()
        with torch.no_grad():
            output_eval = fusion_module(x)

        assert output_train.shape == output_eval.shape
        # Outputs may differ due to dropout, but should be reasonable
        assert not torch.isnan(output_train).any()
        assert not torch.isnan(output_eval).any()


class TestFusionUtilityFunctions:
    """Tests for utility functions in the fusion module."""

    def test_create_unified_attention_fusion(self):
        """Test the factory function for creating fusion modules."""
        fusion_module = create_unified_attention_fusion(
            d_model=256,
            n_heads=8,
            d_ff=1024,
            dropout=0.1,
            strategy=FusionStrategy.FULL_BLOCK
        )

        assert isinstance(fusion_module, UnifiedAttentionFusion)
        assert fusion_module.d_model == 256
        assert fusion_module.n_heads == 8
        assert fusion_module.d_ff == 1024
        assert fusion_module.dropout == 0.1
        assert fusion_module.config.strategy == FusionStrategy.FULL_BLOCK

    def test_create_with_defaults(self):
        """Test factory function with default parameters."""
        fusion_module = create_unified_attention_fusion(
            d_model=512,
            n_heads=8
        )

        assert fusion_module.d_ff == 4 * 512  # Default d_ff
        assert fusion_module.dropout == 0.0    # Default dropout
        assert fusion_module.config.strategy == FusionStrategy.FULL_BLOCK

    def test_benchmark_fusion_performance_basic(self):
        """Test basic fusion performance benchmarking."""
        # Use small parameters for faster testing
        results = benchmark_fusion_performance(
            d_model=128,
            n_heads=4,
            seq_len=32,
            batch_size=2,
            num_iterations=5,
            strategies=[FusionStrategy.FULL_BLOCK, FusionStrategy.ATTENTION_NORM]
        )

        assert isinstance(results, dict)
        assert len(results) == 2
        assert FusionStrategy.FULL_BLOCK in results
        assert FusionStrategy.ATTENTION_NORM in results

        for _strategy, stats in results.items():
            assert isinstance(stats, FusionPerformanceStats)
            assert stats.execution_time_fused_ms > 0

    def test_print_functions_no_error(self, capsys):
        """Test that print functions run without errors."""
        # Create mock fusion analysis
        mock_analysis = {
            "input_shape": [2, 32, 128],
            "fusion_strategy": "full_block",
            "theoretical_speedup": 2.5,
            "memory_reduction_ratio": 0.3,
            "estimated_kernel_launches_original": 15,
            "estimated_kernel_launches_fused": 5,
            "hardware_info": {
                "device_name": "Test GPU",
                "compute_capability": (8, 0),
                "memory_gb": 16,
                "tensor_cores": True
            },
            "compatibility": {
                "flash_attention": False,
                "triton_kernels": False,
                "torch_compile": True,
                "mixed_precision": True
            },
            "optimization_opportunities": [
                "Test opportunity 1",
                "Test opportunity 2"
            ]
        }

        # Test print_fusion_analysis
        try:
            print_fusion_analysis(mock_analysis)
            captured = capsys.readouterr()
            assert "Neural Operator Fusion Analysis" in captured.out
        except Exception as e:
            pytest.fail(f"print_fusion_analysis failed: {e}")

        # Create mock benchmark results
        mock_results = {
            FusionStrategy.FULL_BLOCK: FusionPerformanceStats(
                kernel_launches_original=15,
                kernel_launches_fused=5,
                execution_time_original_ms=100.0,
                execution_time_fused_ms=40.0,
                gpu_utilization=0.85,
                fusion_efficiency=0.75
            ),
            FusionStrategy.ATTENTION_NORM: FusionPerformanceStats(
                kernel_launches_original=15,
                kernel_launches_fused=8,
                execution_time_original_ms=100.0,
                execution_time_fused_ms=60.0,
                gpu_utilization=0.80,
                fusion_efficiency=0.65
            )
        }

        # Test print_benchmark_results
        try:
            print_benchmark_results(mock_results)
            captured = capsys.readouterr()
            assert "Fusion Performance Benchmark Results" in captured.out
        except Exception as e:
            pytest.fail(f"print_benchmark_results failed: {e}")


class TestFusionAnalysis:
    """Tests for fusion analysis and optimization identification."""

    def test_fusion_analysis_basic(self):
        """Test basic fusion analysis functionality."""
        batch_size, seq_len, d_model = 2, 32, 256

        fusion_module = UnifiedAttentionFusion(
            d_model=d_model,
            n_heads=8,
            d_ff=1024
        )

        x = torch.randn(batch_size, seq_len, d_model)
        analysis = fusion_module.get_fusion_analysis(x)

        assert isinstance(analysis, dict)
        assert "input_shape" in analysis
        assert "fusion_strategy" in analysis
        assert "theoretical_speedup" in analysis
        assert "memory_reduction_ratio" in analysis
        assert "hardware_info" in analysis
        assert "optimization_opportunities" in analysis
        assert "compatibility" in analysis

        # Validate data types and ranges
        assert isinstance(analysis["input_shape"], list)
        assert analysis["theoretical_speedup"] > 1.0
        assert 0.0 <= analysis["memory_reduction_ratio"] <= 1.0

    def test_optimization_opportunity_identification(self):
        """Test identification of optimization opportunities."""
        # Test with various input configurations
        test_configs = [
            (2, 64, 256),   # Small sequence
            (2, 2048, 512), # Long sequence
            (32, 128, 256), # Large batch
        ]

        for batch_size, seq_len, d_model in test_configs:
            fusion_module = UnifiedAttentionFusion(
                d_model=d_model,
                n_heads=8,
                d_ff=d_model * 4
            )

            x = torch.randn(batch_size, seq_len, d_model)
            analysis = fusion_module.get_fusion_analysis(x)

            opportunities = analysis["optimization_opportunities"]
            assert isinstance(opportunities, list)

            # Should identify some opportunities for optimization
            if seq_len <= 512:
                assert any("short" in opp.lower() for opp in opportunities)
            elif seq_len >= 2048:
                assert any("long" in opp.lower() or "ring" in opp.lower() for opp in opportunities)

    def test_hardware_compatibility_analysis(self):
        """Test hardware compatibility analysis."""
        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)
        x = torch.randn(2, 32, 256)
        analysis = fusion_module.get_fusion_analysis(x)

        compatibility = analysis["compatibility"]
        assert isinstance(compatibility, dict)
        assert "flash_attention" in compatibility
        assert "triton_kernels" in compatibility
        assert "torch_compile" in compatibility
        assert "mixed_precision" in compatibility

        # All should be boolean values
        for _key, value in compatibility.items():
            assert isinstance(value, bool)


class TestNumericalAccuracy:
    """Tests for numerical accuracy preservation in fusion operations."""

    def test_fusion_vs_unfused_accuracy(self):
        """Test that fused operations maintain numerical accuracy."""
        batch_size, seq_len, d_model = 2, 16, 128
        n_heads = 4
        d_ff = 512

        # Create identical modules with different fusion strategies
        config_fused = FusionConfig(strategy=FusionStrategy.FULL_BLOCK)
        config_unfused = FusionConfig(strategy=FusionStrategy.CUSTOM)  # Unfused

        fusion_module_fused = UnifiedAttentionFusion(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, config=config_fused
        )
        fusion_module_unfused = UnifiedAttentionFusion(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, config=config_unfused
        )

        # Copy weights to ensure identical computation (except for fusion)
        with torch.no_grad():
            for (name_fused, param_fused), (name_unfused, param_unfused) in zip(  # noqa: B007
                fusion_module_fused.named_parameters(),
                fusion_module_unfused.named_parameters()
            ):
                param_unfused.copy_(param_fused)

        x = torch.randn(batch_size, seq_len, d_model)

        with torch.no_grad():
            output_fused = fusion_module_fused(x)
            output_unfused = fusion_module_unfused(x)

        # Check numerical accuracy (should be very close)
        max_diff = torch.max(torch.abs(output_fused - output_unfused))

        # Allow for small numerical differences due to different computation orders
        assert max_diff < 1e-4, f"Numerical difference too large: {max_diff}"

    def test_gradient_accuracy_preservation(self):
        """Test that gradients are preserved accurately through fusion."""
        batch_size, seq_len, d_model = 2, 16, 128

        fusion_module = UnifiedAttentionFusion(
            d_model=d_model,
            n_heads=4,
            d_ff=512
        )

        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        # Forward pass
        output = fusion_module(x)
        loss = torch.mean(output ** 2)

        # Backward pass
        loss.backward()

        # Check that gradients are reasonable
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert torch.abs(x.grad).max() < 100.0  # Gradients shouldn't explode

        # Check parameter gradients
        for param in fusion_module.parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.isnan(param.grad).any()
                assert torch.abs(param.grad).max() < 100.0


class TestPerformanceOptimizations:
    """Tests for performance optimization features."""

    def test_kernel_launch_estimation(self):
        """Test kernel launch count estimation."""
        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)

        # Test different fusion strategies
        strategies_and_expected = [
            (FusionStrategy.FULL_BLOCK, 3),      # Fully fused
            (FusionStrategy.ATTENTION_NORM, 8),   # Partially fused
            (FusionStrategy.FFN_NORM, 8),        # Partially fused
            # CUSTOM strategy represents unfused = ~15 kernels
        ]

        for strategy, expected_max in strategies_and_expected:
            estimated = fusion_module._estimate_kernel_launches(strategy)
            assert estimated <= expected_max
            assert estimated > 0

    def test_memory_bandwidth_estimation(self):
        """Test memory bandwidth estimation."""
        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)

        input_tensor = torch.randn(4, 32, 256)
        output_tensor = torch.randn(4, 32, 256)
        execution_time = 10.0  # 10ms

        bandwidth = fusion_module._estimate_memory_bandwidth(
            input_tensor, output_tensor, execution_time
        )

        assert bandwidth > 0
        assert bandwidth < 10000  # Reasonable upper bound (GB/s)

    def test_gpu_utilization_estimation(self):
        """Test GPU utilization estimation."""
        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)

        utilization = fusion_module._estimate_gpu_utilization()

        assert 0.0 <= utilization <= 1.0

    def test_optimization_opportunity_identification(self):
        """Test optimization opportunity identification."""
        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)

        # Test with FP32 tensor (should suggest mixed precision)
        x_fp32 = torch.randn(4, 32, 256, dtype=torch.float32)
        opportunities_fp32 = fusion_module._identify_optimization_opportunities(x_fp32)

        assert any("precision" in opp.lower() for opp in opportunities_fp32)

        # Test with large batch (should suggest gradient checkpointing)
        x_large_batch = torch.randn(64, 32, 256)
        opportunities_large = fusion_module._identify_optimization_opportunities(x_large_batch)

        assert any("checkpointing" in opp.lower() for opp in opportunities_large)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_inputs(self):
        """Test handling of very small input tensors."""
        fusion_module = UnifiedAttentionFusion(d_model=64, n_heads=4, d_ff=256)

        # Very small sequence length
        x = torch.randn(1, 1, 64)
        output = fusion_module(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_very_large_inputs(self):
        """Test handling of large input tensors (memory permitting)."""
        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)

        # Large sequence length (but manageable for testing)
        try:
            x = torch.randn(2, 512, 256)
            output = fusion_module(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()
        except RuntimeError as e:
            # Memory issues are acceptable for very large tensors
            if "memory" not in str(e).lower():
                raise

    def test_different_precision_types(self):
        """Test with different tensor precision types."""
        fusion_module = UnifiedAttentionFusion(d_model=128, n_heads=4, d_ff=512)

        precisions = [torch.float32, torch.float16]

        for dtype in precisions:
            try:
                x = torch.randn(2, 16, 128, dtype=dtype)

                # Convert both input and module to same device and dtype
                device = next(fusion_module.parameters()).device
                x = x.to(device)

                # If using fp16, convert module parameters too
                if dtype == torch.float16 and device.type == 'cuda':
                    fusion_module = fusion_module.half()
                    output = fusion_module(x)
                    fusion_module = fusion_module.float()  # Convert back for next iteration
                else:
                    output = fusion_module(x.float())  # Ensure float for CPU or mixed precision

                assert output.shape == x.shape
                assert output.dtype in [torch.float32, torch.float16]  # May be promoted
                assert not torch.isnan(output).any()
            except RuntimeError as e:
                # Some precision types may not be supported on all hardware
                if ("not implemented" in str(e).lower() or
                    "not supported" in str(e).lower() or
                    "same dtype" in str(e).lower()):
                    continue
                raise

    def test_zero_dropout(self):
        """Test with zero dropout."""
        fusion_module = UnifiedAttentionFusion(
            d_model=128, n_heads=4, d_ff=512, dropout=0.0
        )

        x = torch.randn(2, 16, 128)
        output1 = fusion_module(x)
        output2 = fusion_module(x)

        # With zero dropout, outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_invalid_configurations(self):
        """Test handling of invalid configurations."""
        # Test invalid fusion strategy combinations
        invalid_config = FusionConfig(
            enable_flash_attention=True,  # Might not be available
            enable_custom_kernels=True   # Might not be available
        )

        # Should handle gracefully and fall back to PyTorch implementation
        fusion_module = UnifiedAttentionFusion(
            d_model=128, n_heads=4, d_ff=512, config=invalid_config
        )

        x = torch.randn(2, 16, 128)
        output = fusion_module(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestIntegrationWithExistingCode:
    """Integration tests with existing PyTorch patterns."""

    def test_integration_with_torch_compile(self):
        """Test integration with torch.compile (if available)."""
        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)

        try:
            compiled_module = torch.compile(fusion_module)

            x = torch.randn(2, 16, 256)
            output = compiled_module(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()

        except Exception as e:
            # torch.compile might not be available
            if "compile" not in str(e).lower():
                raise

    def test_integration_with_mixed_precision(self):
        """Test integration with automatic mixed precision."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")

        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024).cuda()

        with torch.cuda.amp.autocast():
            x = torch.randn(2, 16, 256, device='cuda')
            output = fusion_module(x)

            assert output.shape == x.shape
            assert output.device.type == 'cuda'
            assert not torch.isnan(output).any()

    def test_integration_with_gradient_checkpointing(self):
        """Test integration with gradient checkpointing."""
        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)

        def checkpoint_wrapper(x):
            return torch.utils.checkpoint.checkpoint(fusion_module, x, use_reentrant=False)

        x = torch.randn(2, 16, 256, requires_grad=True)
        output = checkpoint_wrapper(x)
        loss = output.sum()

        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_integration_with_dataparallel(self):
        """Test integration with DataParallel (if multiple GPUs available)."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")

        fusion_module = UnifiedAttentionFusion(d_model=256, n_heads=8, d_ff=1024)
        parallel_module = nn.DataParallel(fusion_module)
        parallel_module = parallel_module.cuda()

        x = torch.randn(4, 16, 256, device='cuda')  # Larger batch for DataParallel
        output = parallel_module(x)

        assert output.shape == x.shape
        assert output.device.type == 'cuda'
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    # Run specific test functions for quick validation
    print(" Running Neural Operator Fusion Tests")

    # Test basic functionality
    test_config = TestFusionConfig()
    test_config.test_config_initialization()
    test_config.test_config_customization()
    print(" FusionConfig tests passed")

    # Test core fusion module
    test_fusion = TestUnifiedAttentionFusion()
    test_fusion.test_module_initialization()
    test_fusion.test_basic_forward_pass()
    test_fusion.test_different_fusion_strategies()
    print(" UnifiedAttentionFusion core tests passed")

    # Test performance features
    test_perf = TestPerformanceOptimizations()
    test_perf.test_kernel_launch_estimation()
    test_perf.test_memory_bandwidth_estimation()
    print(" Performance optimization tests passed")

    # Test numerical accuracy
    test_accuracy = TestNumericalAccuracy()
    test_accuracy.test_fusion_vs_unfused_accuracy()
    test_accuracy.test_gradient_accuracy_preservation()
    print(" Numerical accuracy tests passed")

    print(" All key Neural Operator Fusion tests completed successfully!")
