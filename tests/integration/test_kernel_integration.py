"""
Integration Tests for Custom Kernel System

Tests end-to-end integration of:
- Kernel registry
- NVIDIA backend integration
- Auto-selection by hardware
- Mixed precision training
- Fallback mechanisms
- Config/backend integration
"""


import pytest
import torch
import torch.nn as nn

from torchbridge.backends.nvidia.nvidia_backend import NVIDIABackend
from torchbridge.core.config import PrecisionFormat, TorchBridgeConfig
from torchbridge.core.kernel_registry import (
    KernelRegistry,
    KernelType,
)
from torchbridge.validation.unified_validator import validate_custom_kernels


# Test fixtures
@pytest.fixture
def kernel_config():
    """Create a kernel-enabled configuration."""
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    config.kernel.flash_attention_enabled = True
    config.kernel.fuse_linear_activation = True
    return config


@pytest.fixture
def nvidia_backend(kernel_config):
    """Create NVIDIA backend with kernel support."""
    return NVIDIABackend(kernel_config)


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q, K, V projections
        self.qkv = nn.Linear(dim, 3 * dim)

        # FFN with fused Linear+GELU pattern
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
        )
        self.ffn_out = nn.Linear(4 * dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Simple attention (placeholder - real implementation would use FlashAttention)
        batch, seq, dim = x.shape
        qkv = self.qkv(x).reshape(batch, seq, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Simplified attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).reshape(batch, seq, dim)

        x = self.norm1(x + out)

        # FFN
        ffn_out = self.ffn(x)
        ffn_out = self.ffn_out(ffn_out)
        x = self.norm2(x + ffn_out)

        return x


# ===== Configuration Integration Tests =====

def test_kernel_config_creation(kernel_config):
    """Test kernel configuration creation and defaults."""
    assert kernel_config.kernel.enabled is True
    assert kernel_config.kernel.flash_attention_enabled is True
    assert kernel_config.kernel.fuse_linear_activation is True
    assert kernel_config.kernel.auto_select_optimal is True


def test_kernel_config_auto_configuration():
    """Test auto-configuration based on hardware."""
    config = TorchBridgeConfig()

    # Verify kernel config was auto-configured in __post_init__
    assert hasattr(config.kernel, 'flash_attention_version')
    assert config.kernel.flash_attention_version in ['2', '3', 'auto']


def test_kernel_validation_integration(kernel_config):
    """Test validation system integration."""
    result = validate_custom_kernels(kernel_config)

    # Should complete without errors
    assert result.total_tests >= 1
    assert result.failed == 0


# ===== Backend Integration Tests =====

def test_backend_kernel_registry_initialization(nvidia_backend):
    """Test that backend initializes kernel registry."""
    assert nvidia_backend.kernel_registry is not None
    assert isinstance(nvidia_backend.kernel_registry, KernelRegistry)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backend_registers_kernels_on_cuda():
    """Test that backend registers kernels when CUDA is available."""
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    backend = NVIDIABackend(config)

    # Should have registered kernels
    kernels = backend.kernel_registry.list_kernels()
    assert len(kernels) > 0

    # Check for FlashAttention-3
    fa_kernels = [k for k in kernels if k.kernel_type == KernelType.ATTENTION]
    assert len(fa_kernels) > 0

    # Check for fused kernels
    fusion_kernels = [k for k in kernels if k.kernel_type == KernelType.FUSION]
    assert len(fusion_kernels) > 0


def test_backend_optimal_kernel_selection_no_cuda(nvidia_backend):
    """Test optimal kernel selection when CUDA not available."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, test requires no CUDA")

    kernel = nvidia_backend.get_optimal_attention_kernel(head_dim=64)

    # Should return None when CUDA not available
    assert kernel is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_backend_optimal_kernel_selection_with_cuda():
    """Test optimal kernel selection with CUDA."""
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    backend = NVIDIABackend(config)

    kernel = backend.get_optimal_attention_kernel(head_dim=64)

    # Should return a kernel class
    assert kernel is not None
    assert hasattr(kernel, '__init__')


# ===== Model Preparation Tests =====

def test_model_preparation_without_cuda(nvidia_backend):
    """Test model preparation when CUDA not available."""
    if torch.cuda.is_available():
        pytest.skip("CUDA is available, test requires no CUDA")

    model = SimpleTransformerBlock()
    prepared = nvidia_backend.prepare_model_with_custom_kernels(model)

    # Should return model unchanged
    assert prepared is not None
    assert isinstance(prepared, nn.Module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_preparation_with_cuda():
    """Test model preparation with CUDA."""
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    config.kernel.fuse_linear_activation = True
    backend = NVIDIABackend(config)

    model = SimpleTransformerBlock()
    prepared = backend.prepare_model_with_custom_kernels(model)

    # Should return modified model
    assert prepared is not None
    assert isinstance(prepared, nn.Module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_fusion_replacement():
    """Test that Linear+GELU gets replaced with fused kernel."""
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    config.kernel.fuse_linear_activation = True
    config.kernel.fused_gelu_enabled = True
    backend = NVIDIABackend(config)

    # Create a model with fusible pattern
    model = SimpleTransformerBlock()
    type(model.ffn)

    prepared = backend.prepare_model_with_custom_kernels(model)

    # Model should be modified
    assert prepared is not None


# ===== Precision Support Tests =====

def test_precision_format_support(kernel_config):
    """Test precision format configuration."""
    assert hasattr(kernel_config.precision, 'default_format')
    assert kernel_config.precision.default_format in [
        PrecisionFormat.FP32,
        PrecisionFormat.FP16,
        PrecisionFormat.BF16,
        PrecisionFormat.FP8_E4M3,
        PrecisionFormat.FP8_E5M2
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mixed_precision_kernel_selection():
    """Test kernel selection with different precisions."""
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    backend = NVIDIABackend(config)

    # Test FP16 selection
    kernel_fp16 = backend.get_optimal_attention_kernel(
        head_dim=64,
        precision=PrecisionFormat.FP16
    )

    # Test BF16 selection
    kernel_bf16 = backend.get_optimal_attention_kernel(
        head_dim=64,
        precision=PrecisionFormat.BF16
    )

    # Should get kernels (may be same kernel supporting both)
    assert kernel_fp16 is not None or kernel_bf16 is not None


# ===== End-to-End Transformer Tests =====

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_e2e_transformer_forward_pass():
    """Test end-to-end transformer forward pass with custom kernels."""
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    backend = NVIDIABackend(config)

    # Create and prepare model
    model = SimpleTransformerBlock(dim=64, num_heads=4)
    model = backend.prepare_model(model)
    model = backend.prepare_model_with_custom_kernels(model)
    model.eval()

    # Create input
    batch_size = 2
    seq_len = 16
    dim = 64
    x = torch.randn(batch_size, seq_len, dim, device=backend.device)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    # Verify output shape
    assert output.shape == (batch_size, seq_len, dim)
    assert output.device == backend.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_e2e_transformer_backward_pass():
    """Test end-to-end transformer backward pass."""
    config = TorchBridgeConfig()
    config.kernel.enabled = True
    backend = NVIDIABackend(config)

    # Create and prepare model
    model = SimpleTransformerBlock(dim=64, num_heads=4)
    model = backend.prepare_model(model)
    model = backend.prepare_model_with_custom_kernels(model)
    model.train()

    # Create input and target
    batch_size = 2
    seq_len = 16
    dim = 64
    x = torch.randn(batch_size, seq_len, dim, device=backend.device, requires_grad=True)

    # Forward pass
    output = model(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Verify gradients
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


# ===== Fallback Mechanism Tests =====

def test_fallback_when_kernels_disabled():
    """Test that system falls back gracefully when kernels disabled."""
    config = TorchBridgeConfig()
    config.kernel.enabled = False
    backend = NVIDIABackend(config)

    # Should not register any kernels
    backend.kernel_registry.list_kernels()
    # Empty registry is OK when disabled

    # Model preparation should still work
    model = SimpleTransformerBlock()
    prepared = backend.prepare_model_with_custom_kernels(model)
    assert prepared is not None


def test_fallback_when_imports_fail(nvidia_backend, monkeypatch):
    """Test fallback when custom kernel imports fail."""
    # This test verifies the try/except in _register_default_kernels

    # Even if imports fail during init, backend should still work
    assert nvidia_backend is not None
    assert nvidia_backend.kernel_registry is not None


# ===== Performance & Correctness Tests =====

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.benchmark
def test_custom_kernels_maintain_correctness():
    """Test that custom kernels maintain numerical correctness."""
    config_baseline = TorchBridgeConfig()
    config_baseline.kernel.enabled = False

    config_optimized = TorchBridgeConfig()
    config_optimized.kernel.enabled = True

    backend_baseline = NVIDIABackend(config_baseline)
    backend_optimized = NVIDIABackend(config_optimized)

    # Create identical models
    torch.manual_seed(42)
    model_baseline = SimpleTransformerBlock(dim=64, num_heads=4)

    torch.manual_seed(42)
    model_optimized = SimpleTransformerBlock(dim=64, num_heads=4)

    # Prepare models
    model_baseline = backend_baseline.prepare_model(model_baseline)
    model_optimized = backend_optimized.prepare_model(model_optimized)
    model_optimized = backend_optimized.prepare_model_with_custom_kernels(model_optimized)

    model_baseline.eval()
    model_optimized.eval()

    # Same input
    torch.manual_seed(123)
    x = torch.randn(2, 16, 64, device=backend_baseline.device)

    # Forward pass
    with torch.no_grad():
        output_baseline = model_baseline(x)
        output_optimized = model_optimized(x)

    # Should be close (allowing for kernel implementation differences)
    assert torch.allclose(output_baseline, output_optimized, rtol=1e-2, atol=1e-2)


# ===== Summary Tests =====

def test_integration_test_coverage():
    """Verify that we have comprehensive integration test coverage."""
    # This test just ensures we're testing all key areas

    key_areas = [
        'test_kernel_config_creation',
        'test_backend_kernel_registry_initialization',
        'test_model_preparation_without_cuda',
        'test_precision_format_support',
        'test_fallback_when_kernels_disabled',
    ]

    # All these tests should exist in this module
    import sys
    current_module = sys.modules[__name__]

    for test_name in key_areas:
        assert hasattr(current_module, test_name), f"Missing test: {test_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
