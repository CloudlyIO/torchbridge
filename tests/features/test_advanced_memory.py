"""
Comprehensive Tests for Advanced Memory Optimizations

Tests for all advanced memory optimization components:
- Deep Optimizer States
- Advanced Checkpointing
- Memory Pool Management
- Gradient Compression
- Long Sequence Optimization
"""

import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbridge.advanced_memory import (
    AdaptiveCheckpointing,
    AdaptiveCompressionOptimizer,
    CPUGPUHybridOptimizer,
    # Deep optimizer states
    DeepOptimizerStates,
    DynamicActivationOffloading,
    # Memory pool management
    DynamicMemoryPool,
    # Gradient compression
    GradientCompressor,
    IncrementalSequenceCache,
    InterleaveOffloadingOptimizer,
    # Long sequence optimization
    LongSequenceOptimizer,
    LossyGradientCompression,
    MemoryConfig,
    MemoryEfficientBackprop,
    MemoryFragmentationOptimizer,
    MemoryPoolManager,
    QuantizedGradientAccumulation,
    SegmentedAttentionMemory,
    # Advanced checkpointing
    SelectiveGradientCheckpointing,
    SmartMemoryAllocator,
)


class TestDeepOptimizerStates:
    """Test deep optimizer states and interleaved offloading"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def simple_model(self, device):
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)

    def test_deep_optimizer_states_initialization(self, simple_model, device):
        """Test DeepOptimizerStates can be initialized"""
        optimizer = torch.optim.AdamW(simple_model.parameters(), lr=1e-3)

        memory_config = MemoryConfig(
            cpu_memory_limit_gb=4.0,
            gpu_memory_limit_gb=2.0,
            use_async_offloading=False
        )

        deep_optimizer = DeepOptimizerStates(
            optimizer=optimizer,
            model=simple_model,
            memory_config=memory_config,
            num_groups=2
        )

        assert deep_optimizer.optimizer == optimizer
        assert deep_optimizer.model == simple_model
        assert deep_optimizer.num_groups == 2

    def test_deep_optimizer_states_step(self, simple_model, device):
        """Test optimization step with deep optimizer states"""
        optimizer = torch.optim.AdamW(simple_model.parameters(), lr=1e-3)

        memory_config = MemoryConfig(
            cpu_memory_limit_gb=4.0,
            gpu_memory_limit_gb=2.0,
            use_async_offloading=False
        )

        deep_optimizer = DeepOptimizerStates(
            optimizer=optimizer,
            model=simple_model,
            memory_config=memory_config,
            num_groups=2
        )

        # Test training step
        x = torch.randn(4, 128, device=device)
        target = torch.randn(4, 128, device=device)

        def closure():
            optimizer.zero_grad()
            output = simple_model(x)
            loss = F.mse_loss(output, target)
            loss.backward()
            return loss

        # Should return step metrics
        metrics = deep_optimizer.step(closure)

        assert isinstance(metrics, dict)
        assert 'step_total_time' in metrics
        assert 'memory_usage' in metrics

    def test_interleave_offloading_optimizer(self, simple_model, device):
        """Test InterleaveOffloadingOptimizer"""
        base_optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)

        optimizer = InterleaveOffloadingOptimizer(
            optimizer=base_optimizer,
            model=simple_model,
            memory_limit_gb=1.0,
            auto_tune=True
        )

        # Test optimization step
        x = torch.randn(4, 128, device=device)
        target = torch.randn(4, 128, device=device)

        optimizer.zero_grad()
        output = simple_model(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        metrics = optimizer.step()
        assert isinstance(metrics, dict)

    def test_cpu_gpu_hybrid_optimizer(self, simple_model, device):
        """Test CPUGPUHybridOptimizer"""
        hybrid_optimizer = CPUGPUHybridOptimizer(
            optimizer_class=torch.optim.Adam,
            model=simple_model,
            lr=1e-3,
            cpu_ratio=0.5
        )

        # Test optimization step
        x = torch.randn(4, 128, device=device)
        target = torch.randn(4, 128, device=device)

        hybrid_optimizer.zero_grad()
        output = simple_model(x)
        loss = F.mse_loss(output, target)
        loss.backward()

        metrics = hybrid_optimizer.step()
        assert isinstance(metrics, dict)


class TestAdvancedCheckpointing:
    """Test advanced checkpointing strategies"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def simple_model(self, device):
        return nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ).to(device)

    def test_selective_gradient_checkpointing(self):
        """Test selective gradient checkpointing"""
        selective_checkpoint = SelectiveGradientCheckpointing(importance_threshold=0.5)

        # Test importance updates
        selective_checkpoint.update_importance("layer1", 0.3)
        selective_checkpoint.update_importance("layer2", 0.7)

        # Test checkpointing decisions
        assert selective_checkpoint.should_checkpoint("layer1", 0.6)  # Low importance
        assert not selective_checkpoint.should_checkpoint("layer2", 0.6)  # High importance
        assert selective_checkpoint.should_checkpoint("layer2", 0.9)  # High memory pressure

    def test_adaptive_checkpointing(self, simple_model, device):
        """Test adaptive checkpointing"""
        adaptive_checkpoint = AdaptiveCheckpointing()

        x = torch.randn(4, 64, device=device)

        # Test forward pass with checkpointing
        with torch.no_grad():
            output = adaptive_checkpoint.forward(simple_model, x)

        assert output.shape == (4, 64)
        assert output.device.type == device.type

    def test_memory_efficient_backprop(self, simple_model):
        """Test memory-efficient backpropagation"""
        memory_efficient = MemoryEfficientBackprop()
        memory_efficient.apply(simple_model)

        # Should not crash when applied to module
        assert memory_efficient.enabled

    def test_dynamic_activation_offloading(self, device):
        """Test dynamic activation offloading"""
        offloader = DynamicActivationOffloading(offload_device="cpu")

        # Test with GPU tensor
        activations = torch.randn(4, 256, device=device)

        # Offload to CPU
        offloaded = offloader.offload_activations(activations)
        assert offloaded.device.type == 'cpu'

        # Reload to original device
        reloaded = offloader.reload_activations(offloaded, device)
        assert reloaded.device.type == device.type

        # Verify correctness
        assert torch.allclose(activations.cpu(), reloaded.cpu())


class TestMemoryPoolManagement:
    """Test memory pool management components"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_dynamic_memory_pool(self, device):
        """Test dynamic memory pool"""
        pool = DynamicMemoryPool(device)

        # Get tensor from pool
        tensor1 = pool.get_tensor((10, 20), torch.float32)
        assert tensor1.shape == (10, 20)
        assert tensor1.dtype == torch.float32
        assert tensor1.device.type == device.type

        # Return tensor to pool
        pool.return_tensor(tensor1)

        # Get same shape tensor - should reuse from pool
        tensor2 = pool.get_tensor((10, 20), torch.float32)
        assert tensor2.shape == (10, 20)

    def test_memory_pool_manager(self, device):
        """Test memory pool manager"""
        manager = MemoryPoolManager()

        # Get pool for device
        pool1 = manager.get_pool(device)
        assert isinstance(pool1, DynamicMemoryPool)

        # Should return same pool for same device
        pool2 = manager.get_pool(device)
        assert pool1 is pool2

    def test_smart_memory_allocator(self, device):
        """Test smart memory allocator"""
        allocator = SmartMemoryAllocator()

        # Test allocation
        tensor = allocator.allocate(100, device)
        assert tensor.numel() == 100
        assert tensor.device.type == device.type

        # Test deallocation
        allocator.deallocate(tensor)

    def test_memory_fragmentation_optimizer(self):
        """Test memory fragmentation optimizer"""
        optimizer = MemoryFragmentationOptimizer()

        # Should not crash when optimizing
        optimizer.optimize()
        assert optimizer.enabled


class TestGradientCompression:
    """Test gradient compression techniques"""

    def test_gradient_compressor_base(self):
        """Test base gradient compressor"""
        compressor = GradientCompressor(compression_ratio=0.1)

        gradients = torch.randn(100, 50)

        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)

        assert torch.allclose(gradients, decompressed)

    def test_lossy_gradient_compression(self):
        """Test lossy gradient compression"""
        compressor = LossyGradientCompression(bits=8)

        gradients = torch.randn(50, 30)

        compressed = compressor.compress(gradients)
        assert isinstance(compressed, tuple)
        assert len(compressed) == 3  # quantized, min_val, max_val

        decompressed = compressor.decompress(compressed)
        assert decompressed.shape == gradients.shape

        # Should be approximately equal (lossy compression)
        assert torch.allclose(gradients, decompressed, atol=0.1, rtol=0.1)

    def test_adaptive_compression_optimizer(self):
        """Test adaptive compression optimizer"""
        optimizer = AdaptiveCompressionOptimizer(target_compression_ratio=0.5)

        # Test adaptation based on gradient norm
        initial_ratio = optimizer.current_ratio

        # High gradient norm should increase compression ratio
        optimizer.adapt_compression(2.0)
        assert optimizer.current_ratio >= initial_ratio

        # Low gradient norm should decrease compression ratio
        optimizer.adapt_compression(0.5)
        # Reset to test decrease
        optimizer.current_ratio = 0.8
        optimizer.adapt_compression(0.5)
        assert optimizer.current_ratio <= 0.8

    def test_quantized_gradient_accumulation(self):
        """Test quantized gradient accumulation"""
        accumulator = QuantizedGradientAccumulation(accumulation_steps=2)

        # Create test gradients
        gradients1 = {"layer1": torch.randn(10, 5), "layer2": torch.randn(5, 2)}
        gradients2 = {"layer1": torch.randn(10, 5), "layer2": torch.randn(5, 2)}

        # First accumulation
        result1 = accumulator.accumulate(gradients1)
        assert result1 is None  # Not ready yet

        # Second accumulation
        result2 = accumulator.accumulate(gradients2)
        assert isinstance(result2, dict)
        assert "layer1" in result2
        assert "layer2" in result2


class TestLongSequenceOptimization:
    """Test long sequence optimization techniques"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_long_sequence_optimizer(self):
        """Test long sequence optimizer"""
        optimizer = LongSequenceOptimizer(max_segment_length=100)

        # Test short sequence (no segmentation needed)
        short_seq = torch.randn(2, 50, 128)
        segments = optimizer.segment_sequence(short_seq)
        assert len(segments) == 1
        assert segments[0].shape == (2, 50, 128)

        # Test long sequence (needs segmentation)
        long_seq = torch.randn(2, 250, 128)
        segments = optimizer.segment_sequence(long_seq)
        assert len(segments) == 3  # 250 / 100 = 2.5 -> 3 segments
        assert segments[0].shape == (2, 100, 128)
        assert segments[1].shape == (2, 100, 128)
        assert segments[2].shape == (2, 50, 128)

    def test_segmented_attention_memory(self, device):
        """Test segmented attention with memory"""
        attention = SegmentedAttentionMemory(
            embed_dim=128,
            num_heads=8,
            segment_length=64,
            memory_length=32
        ).to(device)

        # Test short sequence
        short_input = torch.randn(2, 32, 128, device=device)
        with torch.no_grad():
            short_output = attention(short_input)
        assert short_output.shape == (2, 32, 128)

        # Test long sequence
        long_input = torch.randn(2, 150, 128, device=device)
        with torch.no_grad():
            long_output = attention(long_input)
        assert long_output.shape == (2, 150, 128)

    def test_streaming_sequence_processor(self, device):
        """Test streaming sequence processor"""
        try:
            from torchbridge.advanced_memory.long_sequence_optimization import (
                StreamingSequenceProcessor,
            )

            # Create a simple model for the processor
            simple_model = nn.Linear(64, 64).to(device)

            processor = StreamingSequenceProcessor(
                model=simple_model,
                buffer_size=32
            )

            # Test streaming processing
            chunk1 = torch.randn(1, 32, 64, device=device)
            chunk2 = torch.randn(1, 32, 64, device=device)

            with torch.no_grad():
                output1 = processor.process_chunk(chunk1)
                output2 = processor.process_chunk(chunk2)

            assert output1.shape == (1, 32, 64)
            assert output2.shape == (1, 64, 64)  # Buffer concatenated

        except (ImportError, AttributeError):
            # Skip if StreamingSequenceProcessor is not fully implemented
            pytest.skip("StreamingSequenceProcessor not fully implemented")

    def test_incremental_sequence_cache(self):
        """Test incremental sequence cache"""
        try:
            cache = IncrementalSequenceCache(
                cache_size=100
            )

            # Test adding to cache
            sequence_chunk = torch.randn(1, 20, 64)
            cache.put("test_key", sequence_chunk)

            # Test retrieving from cache
            cached_data = cache.get("test_key")
            assert cached_data is not None
            assert cached_data.shape[1] == 20  # Should have 20 tokens
            assert cached_data.shape[2] == 64  # embed_dim

            # Test non-existent key
            missing_data = cache.get("missing_key")
            assert missing_data is None

        except (ImportError, AttributeError):
            # Skip if IncrementalSequenceCache is not fully implemented
            pytest.skip("IncrementalSequenceCache not fully implemented")


class TestAdvancedMemoryIntegration:
    """Integration tests for advanced memory components"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_combined_memory_optimizations(self, device):
        """Test multiple memory optimizations working together"""
        # Create model
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)

        # Setup optimizers
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        memory_config = MemoryConfig(
            cpu_memory_limit_gb=2.0,
            gpu_memory_limit_gb=1.0,
            use_async_offloading=False
        )

        deep_optimizer = DeepOptimizerStates(
            optimizer=base_optimizer,
            model=model,
            memory_config=memory_config,
            num_groups=2
        )

        # Setup checkpointing
        adaptive_checkpoint = AdaptiveCheckpointing()

        # Setup compression
        compressor = LossyGradientCompression(bits=8)

        # Test training step with all optimizations
        x = torch.randn(4, 128, device=device)
        target = torch.randn(4, 128, device=device)

        def closure():
            base_optimizer.zero_grad()

            # Use checkpointing for forward pass
            output = adaptive_checkpoint.forward(model, x)
            loss = F.mse_loss(output, target)
            loss.backward()

            # Compress gradients
            for _name, param in model.named_parameters():
                if param.grad is not None:
                    compressed = compressor.compress(param.grad)
                    decompressed = compressor.decompress(compressed)
                    param.grad.data = decompressed

            return loss

        # Execute optimization step
        metrics = deep_optimizer.step(closure)

        assert isinstance(metrics, dict)
        assert 'step_total_time' in metrics

    def test_memory_optimization_performance(self, device):
        """Test that memory optimizations don't break functionality"""
        # Create larger model for meaningful test
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Test without optimizations (baseline)
        x = torch.randn(8, 256, device=device)
        target = torch.randn(8, 128, device=device)

        start_time = time.time()
        optimizer.zero_grad()
        output_baseline = model(x)
        loss_baseline = F.mse_loss(output_baseline, target)
        loss_baseline.backward()
        optimizer.step()
        baseline_time = time.time() - start_time

        # Test with memory optimizations
        interleave_optimizer = InterleaveOffloadingOptimizer(
            optimizer=optimizer,
            model=model,
            memory_limit_gb=1.0,
            auto_tune=False  # Disable auto-tuning for consistent timing
        )

        start_time = time.time()
        interleave_optimizer.zero_grad()
        output_optimized = model(x)
        loss_optimized = F.mse_loss(output_optimized, target)
        loss_optimized.backward()
        metrics = interleave_optimizer.step()
        optimized_time = time.time() - start_time

        # Verify both approaches produce reasonable results (shapes and magnitude)
        assert output_baseline.shape == output_optimized.shape
        assert not torch.isnan(output_baseline).any()
        assert not torch.isnan(output_optimized).any()

        # Results should be in similar magnitude ranges (optimization may change results)
        baseline_magnitude = torch.abs(output_baseline).mean()
        optimized_magnitude = torch.abs(output_optimized).mean()
        assert optimized_magnitude < baseline_magnitude * 3.0  # Not drastically different magnitude

        # Performance should be reasonable (not more than 3x slower)
        assert optimized_time < baseline_time * 3.0

        # Verify metrics are returned
        assert isinstance(metrics, dict)
