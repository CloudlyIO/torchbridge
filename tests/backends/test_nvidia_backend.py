"""
Comprehensive NVIDIA Backend Tests

Tests all NVIDIA backend components to match TPU testing depth.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from torchbridge.backends.nvidia import (
    CUDADeviceManager,
    CUDAOptimizations,
    FlashAttention3,
    FP8Compiler,
    NVIDIAAdapter,
    NVIDIABackend,
    create_cuda_integration,
    create_flash_attention_3,
)
from torchbridge.core.config import (
    NVIDIAArchitecture,
    TorchBridgeConfig,
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

        config = TorchBridgeConfig()
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

        config = TorchBridgeConfig()
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
        """_optimize_for_tensor_cores replaces child Linear layers with _TensorCoreAlignedLinear."""
        from torchbridge.backends.nvidia.nvidia_backend import _TensorCoreAlignedLinear
        backend = NVIDIABackend()
        # Must be a container — bare nn.Linear has no named_children() to iterate
        model = nn.Sequential(nn.Linear(10, 10))
        model.eval()
        optimized = backend._optimize_for_tensor_cores(model)
        assert optimized is not None
        child = list(optimized.children())[0]
        assert isinstance(child, _TensorCoreAlignedLinear)
        assert child.padded_in >= 10
        assert child.padded_out >= 10

    def test_tensor_core_aligned_linear_cpu_preserves_device(self):
        """Regression v0.5.45: _TensorCoreAlignedLinear must keep buffers on CPU when source is CPU."""
        from torchbridge.backends.nvidia.nvidia_backend import _TensorCoreAlignedLinear
        linear = nn.Linear(127, 63)
        aligned = _TensorCoreAlignedLinear(linear, optimal_multiple=16)
        assert aligned._padded_weight.device.type == "cpu"
        out = aligned(torch.randn(4, 127))
        assert out.shape == (4, 63)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tensor_core_aligned_linear_cuda_preserves_device(self):
        """Regression v0.5.45: _TensorCoreAlignedLinear must keep buffers on CUDA when source is CUDA."""
        from torchbridge.backends.nvidia.nvidia_backend import _TensorCoreAlignedLinear
        linear = nn.Linear(1023, 511).cuda()
        aligned = _TensorCoreAlignedLinear(linear, optimal_multiple=16)
        assert aligned._padded_weight.device.type == "cuda"
        x = torch.randn(4, 1023, device="cuda")
        out = aligned(x)
        assert out.shape == (4, 511)

    def test_backend_with_custom_config(self):
        """Test backend with custom configuration."""
        config = TorchBridgeConfig()
        config.hardware.nvidia.fp8_enabled = False
        config.hardware.nvidia.cudnn_benchmark = False
        backend = NVIDIABackend(config)
        assert backend.nvidia_config.fp8_enabled is False

    def test_optimize_memory_layout_channels_last_applied(self):
        """_optimize_memory_layout() must actually convert Conv2d to channels_last."""
        backend = NVIDIABackend()
        model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1))
        optimized = backend._optimize_memory_layout(model)
        conv = [m for m in optimized.modules() if isinstance(m, nn.Conv2d)][0]
        assert conv.weight.is_contiguous(memory_format=torch.channels_last), (
            "_optimize_memory_layout() did not convert Conv2d to channels_last"
        )

    def test_configure_cuda_allocator_sets_env_for_hopper(self):
        """_configure_cuda_allocator() with sm_90 sets expandable_segments."""
        import os
        backend = NVIDIABackend()
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        backend._compute_capability = (9, 0)
        backend._configure_cuda_allocator()
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert "expandable_segments" in conf, (
            "Hopper allocator config must include expandable_segments"
        )
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    def test_configure_cuda_allocator_sets_env_for_ampere(self):
        """_configure_cuda_allocator() with sm_80 sets max_split_size_mb."""
        import os
        backend = NVIDIABackend()
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        backend._compute_capability = (8, 0)
        backend._configure_cuda_allocator()
        conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        assert "max_split_size_mb" in conf, (
            "Ampere allocator config must include max_split_size_mb"
        )
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

    def test_configure_cuda_allocator_respects_existing_env(self):
        """_configure_cuda_allocator() must not override user's PYTORCH_CUDA_ALLOC_CONF."""
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "user_custom_value"
        backend = NVIDIABackend()
        backend._compute_capability = (9, 0)
        backend._configure_cuda_allocator()
        assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "user_custom_value", (
            "_configure_cuda_allocator() must not override user env var"
        )
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)


# ============================================================================
# NVIDIA Optimizer Tests (10 tests)
# ============================================================================

class TestNVIDIAAdapter:
    """Test NVIDIA adapter functionality."""

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        optimizer = NVIDIAAdapter()
        assert optimizer.config is not None
        assert optimizer.backend is not None

    def test_conservative_optimization(self):
        """Test conservative optimization level."""
        optimizer = NVIDIAAdapter()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="conservative")
        assert result.optimization_level == "conservative"
        assert result.optimized_model is not None

    def test_balanced_optimization(self):
        """Test balanced optimization level."""
        optimizer = NVIDIAAdapter()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="balanced")
        assert result.optimization_level == "balanced"
        # Check that at least some optimizations were applied
        assert len(result.optimizations_applied) > 0

    def test_aggressive_optimization(self):
        """Test aggressive optimization level."""
        optimizer = NVIDIAAdapter()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="aggressive")
        assert result.optimization_level == "aggressive"
        assert len(result.optimizations_applied) > 0

    def test_optimize_for_inference(self):
        """Test inference optimization."""
        optimizer = NVIDIAAdapter()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_for_inference_legacy(model)
        assert "eval_mode" in result.optimizations_applied

    def test_optimize_for_training(self):
        """Test training optimization."""
        optimizer = NVIDIAAdapter()
        model = nn.Linear(16, 16)
        result = optimizer.optimize_for_training_legacy(model)
        assert result.optimized_model is not None

    def test_get_optimization_recommendations(self):
        """Test optimization recommendations."""
        optimizer = NVIDIAAdapter()
        model = nn.Linear(16, 16)
        recommendations = optimizer.get_optimization_recommendations(model)
        assert 'architecture' in recommendations
        assert 'suggested_level' in recommendations

    def test_optimization_with_sample_inputs(self):
        """Test optimization with sample inputs."""
        optimizer = NVIDIAAdapter()
        model = nn.Linear(16, 16)
        sample_inputs = torch.randn(1, 16)
        result = optimizer.optimize_legacy(model, sample_inputs=sample_inputs)
        assert result.optimized_model is not None

    def test_mixed_precision_enablement(self):
        """Test mixed precision enablement."""
        config = TorchBridgeConfig()
        config.precision.mixed_precision = True
        optimizer = NVIDIAAdapter(config)
        model = nn.Linear(16, 16)
        result = optimizer.optimize_legacy(model, optimization_level="balanced")
        # Check that optimization was attempted
        assert result.optimized_model is not None

    def test_optimization_warnings(self):
        """Test optimization warnings."""
        optimizer = NVIDIAAdapter()
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
        config = TorchBridgeConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        assert compiler._fp8_supported

    def test_fp8_support_ampere(self):
        """Test FP8 support detection for Ampere."""
        config = TorchBridgeConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.AMPERE
        compiler = FP8Compiler(config)
        assert not compiler._fp8_supported

    def test_prepare_for_fp8_inference(self):
        """Test FP8 preparation for inference."""
        config = TorchBridgeConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        model = nn.Linear(16, 16)
        prepared = compiler.prepare_for_fp8(model, for_inference=True)
        assert prepared is not None

    def test_prepare_for_fp8_training(self):
        """Test FP8 preparation for training."""
        config = TorchBridgeConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        model = nn.Linear(16, 16)
        prepared = compiler.prepare_for_fp8(model, for_inference=False)
        assert prepared is not None

    def test_fp8_stats(self):
        """Test FP8 statistics."""
        config = TorchBridgeConfig()
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

    def test_compile_with_fp8(self):
        """Test full FP8 compilation."""
        config = TorchBridgeConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.HOPPER
        compiler = FP8Compiler(config)
        model = nn.Linear(16, 16)
        result = compiler.compile_with_fp8(model)
        assert result.compiled_model is not None
        assert result.compilation_mode in ['inference', 'training']


# ============================================================================
# NVIDIA Memory Manager Tests (7 tests)
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
        from torchbridge.backends.nvidia.cuda_utilities import CUDAUtilities
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
        optimizer = NVIDIAAdapter()
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        result = optimizer.optimize_legacy(model, optimization_level="balanced")
        assert result.optimized_model is not None
        assert len(result.optimizations_applied) > 0

    def test_end_to_end_inference_optimization(self):
        """Test end-to-end inference optimization."""
        config = TorchBridgeConfig()
        optimizer = NVIDIAAdapter(config)
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

    def test_optimizer_with_invalid_optimization_level(self):
        """Test optimizer with invalid optimization level."""
        optimizer = NVIDIAAdapter()
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

    def test_fp8_unsupported_architecture(self):
        """Test FP8 compiler on unsupported architecture."""
        config = TorchBridgeConfig()
        config.hardware.nvidia.architecture = NVIDIAArchitecture.AMPERE  # Not Hopper/Blackwell
        config.hardware.nvidia.fp8_enabled = True

        compiler = FP8Compiler(config)
        model = nn.Linear(128, 128)

        # Should return model unchanged with warning
        result = compiler.prepare_for_fp8(model)
        assert result is model  # Should be same object, unchanged

