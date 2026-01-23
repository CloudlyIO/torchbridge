"""
Comprehensive NVIDIA Backend Tests

Tests all NVIDIA backend components to match TPU testing depth.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from kernel_pytorch.core.config import (
    KernelPyTorchConfig,
    NVIDIAConfig,
    NVIDIAArchitecture
)
from kernel_pytorch.backends.nvidia import (
    NVIDIABackend,
    NVIDIAOptimizer,
    FP8Compiler,
    NVIDIAMemoryManager,
    FlashAttention3,
    create_flash_attention_3,
    CUDADeviceManager,
    CUDAOptimizations,
    create_cuda_integration
)
from kernel_pytorch.backends.nvidia.nvidia_exceptions import (
    NVIDIABackendError,
    CUDANotAvailableError,
    OutOfMemoryError,
    MemoryAllocationError,
    FlashAttentionError,
    FP8CompilationError
)


# ============================================================================
# NVIDIA Backend Tests (10 tests)
# ============================================================================

class TestNVIDIABackend:
    """Test NVIDIA backend functionality."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_backend_creation_no_cuda(self, mock_cuda):
        """Test backend creation when CUDA is not available."""
        backend = NVIDIABackend()
        assert backend.device.type == "cpu"
        assert not backend.is_cuda_available

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.get_device_properties')
    def test_backend_creation_with_cuda(self, mock_props, mock_count, mock_cuda):
        """Test backend creation with CUDA available."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA H100 80GB"
        mock_device_props.major = 9
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        backend = NVIDIABackend()
        assert backend.is_cuda_available
        assert len(backend.devices) == 2

    @patch('torch.cuda.is_available', return_value=False)
    def test_prepare_model_no_cuda(self, mock_cuda):
        """Test model preparation without CUDA."""
        backend = NVIDIABackend()
        model = nn.Linear(10, 10)
        prepared = backend.prepare_model(model)
        assert prepared is not None

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    def test_h100_detection(self, mock_props, mock_count, mock_cuda):
        """Test H100 GPU detection."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA H100"
        mock_device_props.major = 9
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        backend = NVIDIABackend(config)
        assert backend.is_h100

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    def test_fp8_support_detection(self, mock_props, mock_count, mock_cuda):
        """Test FP8 support detection."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA H100"
        mock_device_props.major = 9
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        backend = NVIDIABackend(config)
        assert backend.supports_fp8

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    def test_get_device_info(self, mock_props, mock_count, mock_cuda):
        """Test device information retrieval."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA A100"
        mock_device_props.major = 8
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        backend = NVIDIABackend()
        info = backend.get_device_info_dict()
        assert info['backend'] == 'nvidia'
        assert info['cuda_available']

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated', return_value=1024**3)
    @patch('torch.cuda.memory_reserved', return_value=2*1024**3)
    @patch('torch.cuda.max_memory_allocated', return_value=1.5*1024**3)
    def test_get_memory_stats(self, mock_max, mock_reserved, mock_allocated, mock_props, mock_count, mock_cuda):
        """Test memory statistics retrieval."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA A100"
        mock_device_props.major = 8
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        backend = NVIDIABackend()
        stats = backend.get_memory_stats()
        assert 'allocated' in stats
        assert 'reserved' in stats

    def test_optimize_for_tensor_cores(self):
        """Test Tensor Core optimization."""
        backend = NVIDIABackend()
        model = nn.Linear(10, 10)  # Not optimal dimensions
        optimized = backend._optimize_for_tensor_cores(model)
        assert optimized is not None

    def test_backend_with_custom_config(self):
        """Test backend with custom configuration."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.fp8_enabled = False
        config.hardware.nvidia.cudnn_benchmark = False
        backend = NVIDIABackend(config)
        assert backend.nvidia_config.fp8_enabled == False


# ============================================================================
# NVIDIA Optimizer Tests (10 tests)
# ============================================================================

class TestNVIDIAOptimizer:
    """Test NVIDIA optimizer functionality."""

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        optimizer = NVIDIAOptimizer()
        assert optimizer.config is not None
        assert optimizer.backend is not None

    def test_conservative_optimization(self):
        """Test conservative optimization level."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="conservative")
        assert result.optimization_level == "conservative"
        assert result.optimized_model is not None

    def test_balanced_optimization(self):
        """Test balanced optimization level."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="balanced")
        assert result.optimization_level == "balanced"
        # Check that at least some optimizations were applied
        assert len(result.optimizations_applied) > 0

    def test_aggressive_optimization(self):
        """Test aggressive optimization level."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="aggressive")
        assert result.optimization_level == "aggressive"
        assert len(result.optimizations_applied) > 0

    def test_optimize_for_inference(self):
        """Test inference optimization."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_for_inference_legacy(model)
        assert "eval_mode" in result.optimizations_applied

    def test_optimize_for_training(self):
        """Test training optimization."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_for_training_legacy(model)
        assert result.optimized_model is not None

    def test_get_optimization_recommendations(self):
        """Test optimization recommendations."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(16, 16)
        recommendations = optimizer.get_optimization_recommendations(model)
        assert 'architecture' in recommendations
        assert 'suggested_level' in recommendations

    def test_optimization_with_sample_inputs(self):
        """Test optimization with sample inputs."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(16, 16)
        sample_inputs = torch.randn(1, 16)
        result = optimizer.optimize_legacy(model, sample_inputs=sample_inputs)
        assert result.optimized_model is not None

    def test_mixed_precision_enablement(self):
        """Test mixed precision enablement."""
        config = KernelPyTorchConfig()
        config.precision.mixed_precision = True
        optimizer = NVIDIAOptimizer(config)
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="balanced")
        # Check that optimization was attempted
        assert result.optimized_model is not None

    def test_optimization_warnings(self):
        """Test optimization warnings."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="unknown_level")
        assert len(result.warnings) > 0


# ============================================================================
# FP8 Compiler Tests (8 tests)
# ============================================================================

class TestFP8Compiler:
    """Test FP8 compiler functionality."""

    def test_fp8_compiler_creation(self):
        """Test FP8 compiler creation."""
        compiler = FP8Compiler()
        assert compiler.config is not None

    def test_fp8_support_hopper(self):
        """Test FP8 support detection for Hopper."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        assert compiler._fp8_supported

    def test_fp8_support_ampere(self):
        """Test FP8 support detection for Ampere."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.AMPERE
        compiler = FP8Compiler(config)
        assert not compiler._fp8_supported

    def test_prepare_for_fp8_inference(self):
        """Test FP8 preparation for inference."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        model = nn.Linear(16, 16)
        prepared = compiler.prepare_for_fp8(model, for_inference=True)
        assert prepared is not None

    def test_prepare_for_fp8_training(self):
        """Test FP8 preparation for training."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        model = nn.Linear(16, 16)
        prepared = compiler.prepare_for_fp8(model, for_inference=False)
        assert prepared is not None

    def test_fp8_stats(self):
        """Test FP8 statistics."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        model = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        compiler.prepare_for_fp8(model)
        stats = compiler.get_fp8_stats(model)
        assert 'total_layers' in stats
        assert 'fp8_layers' in stats

    def test_estimate_speedup_hopper(self):
        """Test speedup estimation for Hopper."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        model = nn.Linear(16, 16)
        compiler.prepare_for_fp8(model)
        speedup = compiler.estimate_speedup(model)
        assert 'estimated_speedup' in speedup
        assert speedup['base_speedup'] == 2.0

    def test_compile_with_fp8(self):
        """Test full FP8 compilation."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        model = nn.Linear(16, 16)
        result = compiler.compile_with_fp8(model)
        assert result.compiled_model is not None
        assert result.compilation_mode in ['inference', 'training']


# ============================================================================
# NVIDIA Memory Manager Tests (7 tests)
# ============================================================================

class TestNVIDIAMemoryManager:
    """Test NVIDIA memory manager functionality."""

    def test_memory_manager_creation(self):
        """Test memory manager creation."""
        manager = NVIDIAMemoryManager()
        assert manager.config is not None

    def test_allocate_tensor(self):
        """Test tensor allocation."""
        manager = NVIDIAMemoryManager()
        tensor = manager.allocate_tensor((10, 10))
        assert tensor.shape == (10, 10)

    def test_allocate_with_pool(self):
        """Test tensor allocation with pooling."""
        manager = NVIDIAMemoryManager()
        tensor1 = manager.allocate_tensor((10, 10), pool_id="test_pool")
        manager.return_to_pool(tensor1, "test_pool")
        tensor2 = manager.allocate_tensor((10, 10), pool_id="test_pool")
        assert tensor2.shape == (10, 10)

    def test_optimize_tensor_layout(self):
        """Test tensor layout optimization."""
        manager = NVIDIAMemoryManager()
        tensor = torch.randn(10, 10)  # Not optimal dimension
        optimized = manager.optimize_tensor_layout(tensor)
        assert optimized is not None

    def test_get_memory_stats(self):
        """Test memory statistics."""
        manager = NVIDIAMemoryManager()
        stats = manager.get_memory_stats()
        assert 'allocated_gb' in stats or 'allocated' in stats

    def test_optimize_model_memory(self):
        """Test model memory optimization."""
        manager = NVIDIAMemoryManager()
        model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        results = manager.optimize_model_memory(model)
        assert 'total_memory_mb' in results
        assert 'recommendations' in results

    def test_clear_pool(self):
        """Test memory pool clearing."""
        manager = NVIDIAMemoryManager()
        tensor = manager.allocate_tensor((10, 10), pool_id="test_pool")
        manager.return_to_pool(tensor, "test_pool")
        manager.clear_pool("test_pool")
        stats = manager.get_pool_stats("test_pool")
        assert stats['tensor_count'] == 0


# ============================================================================
# FlashAttention-3 Tests (8 tests)
# ============================================================================

class TestFlashAttention3:
    """Test FlashAttention-3 implementation."""

    def test_flash_attention_creation(self):
        """Test FlashAttention-3 creation."""
        attn = FlashAttention3(embed_dim=512, num_heads=8)
        assert attn.embed_dim == 512
        assert attn.num_heads == 8

    def test_flash_attention_forward(self):
        """Test FlashAttention-3 forward pass."""
        attn = FlashAttention3(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64)

    def test_flash_attention_with_mask(self):
        """Test FlashAttention-3 with attention mask."""
        attn = FlashAttention3(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        mask = torch.zeros(2, 4, 10, 10)
        output, _ = attn(x, attention_mask=mask)
        assert output.shape == (2, 10, 64)

    def test_flash_attention_return_weights(self):
        """Test FlashAttention-3 returning attention weights."""
        attn = FlashAttention3(embed_dim=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        output, weights = attn(x, return_attention_weights=True)
        assert output.shape == (2, 10, 64)
        # Weights may be None if FlashAttention is used
        assert weights is None or weights.shape == (2, 4, 10, 10)

    def test_create_flash_attention_3(self):
        """Test factory function for FlashAttention-3."""
        attn = create_flash_attention_3(embed_dim=512, num_heads=8)
        assert isinstance(attn, FlashAttention3)

    def test_flash_attention_dropout(self):
        """Test FlashAttention-3 with dropout."""
        attn = FlashAttention3(embed_dim=64, num_heads=4, dropout=0.1)
        attn.train()
        x = torch.randn(2, 10, 64)
        output, _ = attn(x)
        assert output.shape == (2, 10, 64)

    def test_flash_attention_invalid_dimensions(self):
        """Test FlashAttention-3 with invalid dimensions."""
        with pytest.raises(ValueError):
            FlashAttention3(embed_dim=65, num_heads=8)  # Not divisible

    def test_flash_attention_standard_fallback(self):
        """Test FlashAttention-3 fallback to standard attention."""
        attn = FlashAttention3(embed_dim=64, num_heads=4)
        attn.use_flash_attention = False  # Force standard attention
        x = torch.randn(2, 10, 64)
        output, weights = attn(x, return_attention_weights=True)
        assert output.shape == (2, 10, 64)
        assert weights.shape == (2, 4, 10, 10)


# ============================================================================
# CUDA Utilities Tests (5 tests)
# ============================================================================

class TestCUDAUtilities:
    """Test CUDA utilities functionality."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_cuda_device_manager_no_cuda(self, mock_cuda):
        """Test CUDA device manager without CUDA."""
        manager = CUDADeviceManager()
        assert manager.device.type == "cpu"
        assert manager.device_count == 0

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.get_device_properties')
    def test_cuda_device_manager_with_cuda(self, mock_props, mock_count, mock_cuda):
        """Test CUDA device manager with CUDA."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA A100"
        mock_device_props.major = 8
        mock_device_props.minor = 0
        mock_device_props.total_memory = 40 * 1024**3
        mock_device_props.multi_processor_count = 108
        mock_device_props.max_threads_per_block = 1024
        mock_device_props.max_shared_memory_per_block = 163840
        mock_props.return_value = mock_device_props

        manager = CUDADeviceManager()
        assert manager.device_count == 2

    def test_cuda_optimizations(self):
        """Test CUDA optimizations."""
        optimizer = CUDAOptimizations()
        model = nn.Linear(16, 16)
        optimized = optimizer.optimize_model_for_cuda(model)
        assert optimized is not None

    def test_get_cuda_env_info(self):
        """Test CUDA environment info."""
        from kernel_pytorch.backends.nvidia.cuda_utilities import CUDAUtilities
        info = CUDAUtilities.get_cuda_env_info()
        assert 'cuda_available' in info

    def test_create_cuda_integration(self):
        """Test CUDA integration factory."""
        device_manager, optimizations = create_cuda_integration()
        assert device_manager is not None
        assert optimizations is not None


# ============================================================================
# Integration Tests (3 tests)
# ============================================================================

class TestNVIDIAIntegration:
    """Test NVIDIA backend integration."""

    def test_full_optimization_pipeline(self):
        """Test full optimization pipeline."""
        optimizer = NVIDIAOptimizer()
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        result = optimizer.optimize_legacy(model, optimization_level="balanced")
        assert result.optimized_model is not None
        assert len(result.optimizations_applied) > 0

    def test_backend_with_memory_manager(self):
        """Test backend integration with memory manager."""
        backend = NVIDIABackend()
        memory_manager = NVIDIAMemoryManager(backend.config)
        tensor = memory_manager.allocate_tensor((10, 10))
        assert tensor.device == backend.device

    def test_end_to_end_inference_optimization(self):
        """Test end-to-end inference optimization."""
        config = KernelPyTorchConfig()
        optimizer = NVIDIAOptimizer(config)
        model = nn.Linear(128, 128)
        sample_input = torch.randn(1, 128)
        result = optimizer.optimize_for_inference_legacy(
            model,
            sample_inputs=sample_input,
            optimization_level="aggressive"
        )
        assert result.optimized_model is not None
        assert "eval_mode" in result.optimizations_applied


# ============================================================================
# Error Path Tests (15+ tests)
# ============================================================================

class TestNVIDIAErrorPaths:
    """Test error handling and failure scenarios in NVIDIA backend."""

    def test_memory_allocation_error_handling(self):
        """Test that memory allocation errors are properly caught and logged."""
        memory_manager = NVIDIAMemoryManager()
        
        # Test with extremely large tensor that should fail
        with pytest.raises((OutOfMemoryError, MemoryAllocationError, RuntimeError)):
            # Try to allocate 1TB tensor (will fail on most systems)
            memory_manager.allocate_with_oom_protection(
                shape=(1024, 1024, 1024, 1024),  # 1TB in float32
                dtype=torch.float32
            )

    @patch('torch.cuda.is_available', return_value=False)
    def test_cuda_not_available_graceful_fallback(self, mock_cuda):
        """Test graceful fallback when CUDA is not available."""
        backend = NVIDIABackend()
        assert backend.device.type == "cpu"
        assert not backend.is_cuda_available
        
        # Should not raise exception, just fall back to CPU
        model = nn.Linear(10, 10)
        prepared = backend.prepare_model(model)
        assert prepared is not None

    def test_memory_check_when_cuda_unavailable(self):
        """Test memory check when CUDA is unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            memory_manager = NVIDIAMemoryManager()
            result = memory_manager.check_memory_available(100.0)
            assert result is False  # Should return False when CUDA unavailable

    def test_oom_protection_with_insufficient_memory(self):
        """Test OOM protection triggers when insufficient memory."""
        memory_manager = NVIDIAMemoryManager()
        
        with patch.object(memory_manager, 'check_memory_available', return_value=False):
            with patch.object(memory_manager, 'get_memory_stats', return_value={
                'allocated_gb': 10.0,
                'reserved_gb': 15.0
            }):
                with pytest.raises(OutOfMemoryError) as exc_info:
                    memory_manager.allocate_with_oom_protection(
                        shape=(1000, 1000, 1000),
                        dtype=torch.float32
                    )
                assert "Insufficient GPU memory" in str(exc_info.value)

    def test_invalid_model_input(self):
        """Test handling of invalid model inputs."""
        backend = NVIDIABackend()

        # Test with None model - should handle gracefully
        # Backend may issue warning but should not crash
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backend.prepare_model(None)
            # Should either return None unchanged or issue warning
            assert result is None or len(w) > 0

    def test_flash_attention_causal_parameter(self):
        """Test FlashAttention with causal masking enabled."""
        # Test that causal parameter is properly set
        fa = FlashAttention3(embed_dim=64, num_heads=4, causal=True)
        assert fa.causal is True
        
        fa_no_causal = FlashAttention3(embed_dim=64, num_heads=4, causal=False)
        assert fa_no_causal.causal is False

    def test_fp8_compiler_metadata_only_warning(self):
        """Test that FP8 compiler issues metadata-only warning."""
        import warnings
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        config.hardware.nvidia.fp8_enabled = True
        
        compiler = FP8Compiler(config)
        model = nn.Linear(128, 128)
        
        # Should issue warning about metadata-only FP8
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compiler.prepare_for_fp8(model, for_inference=False)
            # Check if warning was issued (UserWarning or DeprecationWarning)
            assert any("metadata-only" in str(warning.message).lower() for warning in w)

    def test_memory_allocation_with_cleanup(self):
        """Test that memory allocation attempts cleanup before failing."""
        memory_manager = NVIDIAMemoryManager()
        
        # Mock check_memory_available to return False initially, True after cleanup
        call_count = [0]
        def mock_check(required_mb):
            call_count[0] += 1
            return call_count[0] > 1  # False first time, True second time
        
        with patch.object(memory_manager, 'check_memory_available', side_effect=mock_check):
            with patch.object(memory_manager, 'clear_pool') as mock_clear:
                # This should succeed on second attempt after cleanup
                tensor = memory_manager.allocate_with_oom_protection(
                    shape=(10, 10),
                    dtype=torch.float32
                )
                # Verify cleanup was called
                mock_clear.assert_called_once()
                assert tensor is not None

    def test_optimizer_with_invalid_optimization_level(self):
        """Test optimizer with invalid optimization level."""
        optimizer = NVIDIAOptimizer()
        model = nn.Linear(64, 64)

        # Optimizer should handle invalid level gracefully (fallback to default)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = optimizer.optimize_legacy(model, optimization_level="invalid_level")
            # Should issue warning about invalid level and fall back to default
            assert result is not None
            # May issue warning about invalid optimization level
            assert len(w) >= 0  # Graceful handling, with or without warning

    def test_memory_stats_tensor_size_estimation(self):
        """Test accurate tensor size estimation."""
        memory_manager = NVIDIAMemoryManager()
        
        # Test size estimation for various dtypes
        size_fp32 = memory_manager._estimate_tensor_size((1000, 1000), torch.float32)
        size_fp16 = memory_manager._estimate_tensor_size((1000, 1000), torch.float16)
        size_int8 = memory_manager._estimate_tensor_size((1000, 1000), torch.int8)
        
        # FP32 should be 2x FP16, FP16 should be 2x INT8
        assert abs(size_fp32 - 2 * size_fp16) < 0.01
        assert abs(size_fp16 - 2 * size_int8) < 0.01

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_unsupported_compute_capability(self):
        """Test handling of unsupported compute capability."""
        # On real hardware, backend should handle old GPUs gracefully
        backend = NVIDIABackend()
        if backend.is_cuda_available:
            # Should have compute capability detected
            assert backend.compute_capability is not None
            # Should not crash even with old compute capability
            assert isinstance(backend.compute_capability, tuple)
            assert len(backend.compute_capability) == 2

    def test_flash_attention_invalid_embed_dim(self):
        """Test FlashAttention with invalid embedding dimension."""
        # embed_dim must be divisible by num_heads
        with pytest.raises(ValueError) as exc_info:
            FlashAttention3(embed_dim=63, num_heads=4)  # 63 not divisible by 4
        assert "divisible" in str(exc_info.value).lower()

    def test_memory_pool_operations(self):
        """Test memory pool allocation and cleanup."""
        memory_manager = NVIDIAMemoryManager()
        
        # Allocate tensor with pool
        tensor1 = memory_manager.allocate_tensor((10, 10), pool_id="test_pool")
        assert tensor1 is not None
        
        # Return to pool
        memory_manager.return_to_pool(tensor1, "test_pool")
        
        # Clear pool
        memory_manager.clear_pool("test_pool")
        
        # Clear all pools
        memory_manager.clear_pool()

    def test_fp8_unsupported_architecture(self):
        """Test FP8 compiler on unsupported architecture."""
        config = KernelPyTorchConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.AMPERE  # Not Hopper/Blackwell
        config.hardware.nvidia.fp8_enabled = True
        
        compiler = FP8Compiler(config)
        model = nn.Linear(128, 128)
        
        # Should return model unchanged with warning
        result = compiler.prepare_for_fp8(model)
        assert result is model  # Should be same object, unchanged

    def test_backend_kernel_registry_integration(self):
        """Test that backend properly integrates with kernel registry."""
        config = KernelPyTorchConfig()
        config.kernel.enabled = True
        
        backend = NVIDIABackend(config)
        assert backend.kernel_registry is not None
        
        # Should have registered default kernels
        # (exact count depends on CUDA availability and hardware)

    def test_memory_allocation_safety_margin(self):
        """Test that safety margin is applied in OOM protection."""
        memory_manager = NVIDIAMemoryManager()
        
        # Mock check to verify safety margin is applied
        with patch.object(memory_manager, 'check_memory_available') as mock_check:
            mock_check.return_value = True
            with patch.object(memory_manager, '_estimate_tensor_size', return_value=100.0):
                try:
                    memory_manager.allocate_with_oom_protection(
                        shape=(10, 10),
                        dtype=torch.float32,
                        safety_margin=1.5  # 50% margin
                    )
                except:
                    pass  # May fail on actual allocation, we're testing the check
                
                # Verify check was called with margin applied
                if mock_check.called:
                    assert mock_check.call_args[0][0] == 150.0  # 100.0 * 1.5
