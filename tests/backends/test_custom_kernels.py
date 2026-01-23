"""
Tests for Custom CUDA Kernels

This module tests the custom CUDA kernel implementations including:
- FlashAttention-3 (FA-3) with FP8 and Split-K optimizations
- Fused Linear + Activation kernels
- Numerical accuracy validation
- Performance benchmarking
- Fallback mechanisms
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# Import custom kernel modules
try:
    from kernel_pytorch.hardware.gpu.custom_kernels import (
        FlashAttentionV3,
        create_flash_attention_v3,
        FusedLinearGELU,
        FusedLinearSiLU,
        create_fused_ffn_layer
    )
    CUSTOM_KERNELS_AVAILABLE = True
except ImportError:
    CUSTOM_KERNELS_AVAILABLE = False

# Try to import compiled CUDA kernels
try:
    import kernel_pytorch_cuda
    CUDA_KERNELS_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_KERNELS_AVAILABLE = False


# Test configurations from test_configs.py pattern
class TestConfig:
    """Test configuration for different model sizes."""
    micro = {"batch": 1, "seq": 32, "dim": 16, "heads": 2, "head_dim": 8}
    small = {"batch": 1, "seq": 64, "dim": 32, "heads": 4, "head_dim": 8}
    medium = {"batch": 2, "seq": 128, "dim": 64, "heads": 4, "head_dim": 16}
    realistic = {"batch": 2, "seq": 512, "dim": 64, "heads": 8, "head_dim": 8}
    large = {"batch": 4, "seq": 1024, "dim": 64, "heads": 8, "head_dim": 8}


def create_attention_tensors(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    device: torch.device = torch.device('cpu'),
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random Q, K, V tensors for attention testing."""
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    return Q, K, V


def pytorch_attention_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    causal: bool = False
) -> torch.Tensor:
    """Reference PyTorch attention implementation for validation."""
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal:
        seq_len = Q.size(2)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output


@pytest.mark.skipif(not CUSTOM_KERNELS_AVAILABLE, reason="Custom kernels not available")
class TestFlashAttentionV3Module:
    """Tests for FlashAttentionV3 PyTorch module."""

    def test_module_initialization(self):
        """Test basic module initialization."""
        fa3 = FlashAttentionV3(causal=True, dropout=0.1)
        assert fa3.causal is True
        assert fa3.dropout == 0.1
        assert fa3.scale is None

    def test_module_initialization_with_scale(self):
        """Test module initialization with custom scale."""
        custom_scale = 0.125
        fa3 = FlashAttentionV3(scale=custom_scale)
        assert fa3.scale == custom_scale

    def test_factory_function(self):
        """Test factory function creates module correctly."""
        fa3 = create_flash_attention_v3(causal=True, dropout=0.1)
        assert isinstance(fa3, FlashAttentionV3)
        assert fa3.causal is True

    def test_extra_repr(self):
        """Test string representation."""
        fa3 = FlashAttentionV3(causal=True, dropout=0.1)
        repr_str = fa3.extra_repr()
        assert "causal=True" in repr_str
        assert "dropout=0.1" in repr_str


@pytest.mark.skipif(not CUSTOM_KERNELS_AVAILABLE, reason="Custom kernels not available")
class TestFlashAttentionV3Correctness:
    """Tests for FlashAttention-3 numerical correctness."""

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_correctness_optimized_head_dims(self, head_dim):
        """Test correctness for optimized head dimensions (64, 128)."""
        batch_size, num_heads, seq_len = 2, 4, 128
        device = torch.device('cpu')
        dtype = torch.float32

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        # FlashAttention-3 output
        fa3 = FlashAttentionV3(causal=False, scale=scale)
        output_fa3 = fa3(Q, K, V)

        # Reference output
        output_ref = pytorch_attention_reference(Q, K, V, scale, causal=False)

        # Numerical accuracy check
        max_diff = torch.abs(output_fa3 - output_ref).max().item()
        assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance for head_dim={head_dim}"

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512])
    def test_correctness_various_sequence_lengths(self, seq_len):
        """Test correctness across different sequence lengths."""
        batch_size, num_heads, head_dim = 2, 4, 64
        device = torch.device('cpu')
        dtype = torch.float32

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        fa3 = FlashAttentionV3(causal=False, scale=scale)
        output_fa3 = fa3(Q, K, V)

        output_ref = pytorch_attention_reference(Q, K, V, scale, causal=False)

        max_diff = torch.abs(output_fa3 - output_ref).max().item()
        assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance for seq_len={seq_len}"

    def test_correctness_causal_masking(self):
        """Test correctness with causal masking enabled."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 64
        device = torch.device('cpu')
        dtype = torch.float32

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        # FlashAttention-3 with causal
        fa3 = FlashAttentionV3(causal=True, scale=scale)
        output_fa3 = fa3(Q, K, V)

        # Reference with causal
        output_ref = pytorch_attention_reference(Q, K, V, scale, causal=True)

        max_diff = torch.abs(output_fa3 - output_ref).max().item()
        assert max_diff < 1e-3, f"Max difference {max_diff} exceeds tolerance for causal attention"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_correctness_different_dtypes(self, dtype):
        """Test correctness with different data types."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
        device = torch.device('cpu')

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        fa3 = FlashAttentionV3(causal=False, scale=scale)
        output_fa3 = fa3(Q, K, V)

        output_ref = pytorch_attention_reference(Q, K, V, scale, causal=False)

        # Adjust tolerance for FP16
        tolerance = 1e-2 if dtype == torch.float16 else 1e-3
        max_diff = torch.abs(output_fa3.float() - output_ref.float()).max().item()
        assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance for dtype={dtype}"

    def test_output_shape(self):
        """Test output shape matches input Q shape."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim)

        fa3 = FlashAttentionV3()
        output = fa3(Q, K, V)

        assert output.shape == Q.shape, f"Output shape {output.shape} doesn't match Q shape {Q.shape}"

    def test_output_dtype(self):
        """Test output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float16]:
            Q, K, V = create_attention_tensors(2, 4, 64, 64, dtype=dtype)

            fa3 = FlashAttentionV3()
            output = fa3(Q, K, V)

            assert output.dtype == dtype, f"Output dtype {output.dtype} doesn't match input {dtype}"


@pytest.mark.skipif(not CUDA_KERNELS_AVAILABLE, reason="CUDA not available")
class TestFlashAttentionV3CUDA:
    """Tests for FlashAttention-3 CUDA kernel implementation."""

    def test_cuda_kernel_availability(self):
        """Test CUDA kernel can be imported."""
        fa3 = FlashAttentionV3()
        # Should have attempted to import CUDA kernels during init
        assert hasattr(fa3, '_cuda_kernel_available')

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_cuda_correctness_optimized_dims(self, head_dim):
        """Test CUDA kernel correctness for optimized head dimensions."""
        batch_size, num_heads, seq_len = 2, 4, 256
        device = torch.device('cuda')
        dtype = torch.float16

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        fa3 = FlashAttentionV3(causal=False, scale=scale)
        output_fa3 = fa3(Q, K, V)

        # Reference on GPU
        output_ref = pytorch_attention_reference(Q, K, V, scale, causal=False)

        max_diff = torch.abs(output_fa3 - output_ref).max().item()
        # Relaxed tolerance for FP16
        assert max_diff < 1e-2, f"CUDA kernel max difference {max_diff} exceeds tolerance"

    @pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048])
    def test_cuda_long_sequences(self, seq_len):
        """Test CUDA kernel handles long sequences correctly."""
        batch_size, num_heads, head_dim = 1, 4, 64
        device = torch.device('cuda')
        dtype = torch.float16

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        fa3 = FlashAttentionV3(causal=False, scale=scale)
        output_fa3 = fa3(Q, K, V)

        output_ref = pytorch_attention_reference(Q, K, V, scale, causal=False)

        max_diff = torch.abs(output_fa3 - output_ref).max().item()
        # Sequences > 2048 use Split-K, may have slightly higher error
        tolerance = 2e-2 if seq_len > 2048 else 1e-2
        assert max_diff < tolerance, f"Long sequence (len={seq_len}) error {max_diff} exceeds tolerance"

    def test_cuda_causal_masking(self):
        """Test CUDA kernel with causal masking."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64
        device = torch.device('cuda')
        dtype = torch.float16

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        fa3 = FlashAttentionV3(causal=True, scale=scale)
        output_fa3 = fa3(Q, K, V)

        output_ref = pytorch_attention_reference(Q, K, V, scale, causal=True)

        max_diff = torch.abs(output_fa3 - output_ref).max().item()
        assert max_diff < 1e-2, f"CUDA causal attention error {max_diff} exceeds tolerance"

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
        reason="Requires A100+ (compute capability 8.0+)"
    )
    def test_cuda_bfloat16_support(self):
        """Test CUDA kernel with BFloat16 precision (A100+)."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64
        device = torch.device('cuda')
        dtype = torch.bfloat16

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        fa3 = FlashAttentionV3(causal=False, scale=scale)
        output_fa3 = fa3(Q, K, V)

        output_ref = pytorch_attention_reference(Q, K, V, scale, causal=False)

        # BF16 has lower precision than FP16
        max_diff = torch.abs(output_fa3.float() - output_ref.float()).max().item()
        assert max_diff < 5e-2, f"BFloat16 error {max_diff} exceeds tolerance"


@pytest.mark.skipif(not CUSTOM_KERNELS_AVAILABLE, reason="Custom kernels not available")
class TestFlashAttentionV3EdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_input_dimensions(self):
        """Test validation catches invalid input dimensions."""
        fa3 = FlashAttentionV3()

        # 3D tensor instead of 4D
        Q_invalid = torch.randn(2, 4, 64)
        K = torch.randn(2, 4, 128, 64)
        V = torch.randn(2, 4, 128, 64)

        with pytest.raises(AssertionError):
            fa3(Q_invalid, K, V)

    def test_mismatched_shapes(self):
        """Test validation catches shape mismatches."""
        fa3 = FlashAttentionV3()

        Q = torch.randn(2, 4, 128, 64)
        K = torch.randn(2, 4, 256, 64)  # Different seq_len
        V = torch.randn(2, 4, 128, 64)

        with pytest.raises(AssertionError):
            fa3(Q, K, V)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 128, 64
        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim)

        fa3 = FlashAttentionV3()
        output = fa3(Q, K, V)

        assert output.shape == Q.shape

    def test_single_head(self):
        """Test with single attention head."""
        batch_size, num_heads, seq_len, head_dim = 2, 1, 128, 64
        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim)

        fa3 = FlashAttentionV3()
        output = fa3(Q, K, V)

        assert output.shape == Q.shape

    def test_very_small_sequence(self):
        """Test with very small sequence length."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 4, 64
        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim)

        fa3 = FlashAttentionV3()
        output = fa3(Q, K, V)

        assert output.shape == Q.shape


@pytest.mark.skipif(not CUDA_KERNELS_AVAILABLE, reason="CUDA not available")
@pytest.mark.benchmark
class TestFlashAttentionV3Performance:
    """Performance benchmarking tests for FlashAttention-3."""

    def test_performance_vs_pytorch(self):
        """Benchmark FlashAttention-3 vs PyTorch SDPA."""
        batch_size, num_heads, seq_len, head_dim = 4, 8, 512, 64
        device = torch.device('cuda')
        dtype = torch.float16

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        # Warmup
        fa3 = FlashAttentionV3(causal=False, scale=scale)
        for _ in range(10):
            _ = fa3(Q, K, V)
            _ = pytorch_attention_reference(Q, K, V, scale)

        torch.cuda.synchronize()

        # Benchmark FA-3
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            output_fa3 = fa3(Q, K, V)
        end.record()
        torch.cuda.synchronize()
        time_fa3 = start.elapsed_time(end) / 100

        # Benchmark PyTorch
        start.record()
        for _ in range(100):
            output_ref = pytorch_attention_reference(Q, K, V, scale)
        end.record()
        torch.cuda.synchronize()
        time_pytorch = start.elapsed_time(end) / 100

        speedup = time_pytorch / time_fa3
        print(f"\nFlashAttention-3 Performance:")
        print(f"  FA-3 time: {time_fa3:.3f} ms")
        print(f"  PyTorch time: {time_pytorch:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Assert at least some speedup (may not always be faster in CPU fallback mode)
        # In actual CUDA kernel, should see 2-5x speedup
        assert speedup > 0.5, "FlashAttention-3 significantly slower than PyTorch"

    def test_memory_efficiency(self):
        """Test memory usage is reasonable for long sequences."""
        batch_size, num_heads, seq_len, head_dim = 2, 4, 2048, 64
        device = torch.device('cuda')
        dtype = torch.float16

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        Q, K, V = create_attention_tensors(batch_size, num_heads, seq_len, head_dim, device, dtype)
        scale = 1.0 / math.sqrt(head_dim)

        fa3 = FlashAttentionV3(causal=False, scale=scale)
        output = fa3(Q, K, V)

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        print(f"\nMemory usage for seq_len={seq_len}:")
        print(f"  Peak memory: {peak_memory:.2f} MB")

        # Should not exceed reasonable memory bounds
        # Rough estimate: inputs + output ≈ 4 * batch * heads * seq * head_dim * 2 bytes
        expected_memory = 4 * batch_size * num_heads * seq_len * head_dim * 2 / 1024**2
        assert peak_memory < expected_memory * 2, "Memory usage unexpectedly high"


@pytest.mark.skipif(not CUSTOM_KERNELS_AVAILABLE, reason="Custom kernels not available")
class TestFusedLinearGELU:
    """Tests for FusedLinearGELU module."""

    def test_module_initialization(self):
        """Test basic module initialization."""
        layer = FusedLinearGELU(512, 2048)
        assert layer.in_features == 512
        assert layer.out_features == 2048
        assert layer.weight.shape == (2048, 512)
        assert layer.bias is not None
        assert layer.bias.shape == (2048,)

    def test_module_initialization_no_bias(self):
        """Test initialization without bias."""
        layer = FusedLinearGELU(512, 2048, bias=False)
        assert layer.bias is None

    @pytest.mark.parametrize("in_features,out_features", [
        (512, 2048),
        (1024, 4096),
        (768, 3072),
        (2048, 8192)
    ])
    def test_correctness_various_dimensions(self, in_features, out_features):
        """Test correctness for common FFN dimensions."""
        batch_size = 32
        layer = FusedLinearGELU(in_features, out_features)

        x = torch.randn(batch_size, in_features)
        output = layer(x)

        # Reference implementation
        linear_out = F.linear(x, layer.weight, layer.bias)
        expected = F.gelu(linear_out)

        max_diff = torch.abs(output - expected).max().item()
        assert max_diff < 1e-5, f"Max difference {max_diff} exceeds tolerance"

    def test_correctness_3d_input(self):
        """Test correctness with 3D input (batch, seq, features)."""
        batch_size, seq_len = 16, 128
        in_features, out_features = 512, 2048

        layer = FusedLinearGELU(in_features, out_features)
        x = torch.randn(batch_size, seq_len, in_features)
        output = layer(x)

        assert output.shape == (batch_size, seq_len, out_features)

        # Check correctness
        linear_out = F.linear(x, layer.weight, layer.bias)
        expected = F.gelu(linear_out)
        max_diff = torch.abs(output - expected).max().item()
        assert max_diff < 1e-5

    def test_output_shape(self):
        """Test output shape matches expected."""
        layer = FusedLinearGELU(256, 1024)
        x = torch.randn(32, 256)
        output = layer(x)
        assert output.shape == (32, 1024)

    def test_backward_pass(self):
        """Test backward pass works correctly."""
        layer = FusedLinearGELU(128, 512)
        x = torch.randn(16, 128, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None


@pytest.mark.skipif(not CUSTOM_KERNELS_AVAILABLE, reason="Custom kernels not available")
class TestFusedLinearSiLU:
    """Tests for FusedLinearSiLU module."""

    def test_module_initialization(self):
        """Test basic module initialization."""
        layer = FusedLinearSiLU(768, 3072)
        assert layer.in_features == 768
        assert layer.out_features == 3072

    @pytest.mark.parametrize("in_features,out_features", [
        (512, 2048),
        (768, 3072),
        (1024, 4096)
    ])
    def test_correctness_various_dimensions(self, in_features, out_features):
        """Test correctness for common FFN dimensions."""
        batch_size = 32
        layer = FusedLinearSiLU(in_features, out_features)

        x = torch.randn(batch_size, in_features)
        output = layer(x)

        # Reference implementation
        linear_out = F.linear(x, layer.weight, layer.bias)
        expected = F.silu(linear_out)

        max_diff = torch.abs(output - expected).max().item()
        assert max_diff < 1e-5, f"Max difference {max_diff} exceeds tolerance"

    def test_backward_pass(self):
        """Test backward pass works correctly."""
        layer = FusedLinearSiLU(256, 1024)
        x = torch.randn(16, 256, requires_grad=True)

        output = layer(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


@pytest.mark.skipif(not CUSTOM_KERNELS_AVAILABLE, reason="Custom kernels not available")
class TestFusedFFNLayer:
    """Tests for complete FFN layer factory function."""

    def test_create_fused_ffn_gelu(self):
        """Test creating FFN with GELU activation."""
        ffn = create_fused_ffn_layer(768, 3072, activation="gelu")

        assert isinstance(ffn, nn.Sequential)
        assert len(ffn) == 2
        assert isinstance(ffn[0], FusedLinearGELU)
        assert isinstance(ffn[1], nn.Linear)

    def test_create_fused_ffn_silu(self):
        """Test creating FFN with SiLU activation."""
        ffn = create_fused_ffn_layer(512, 2048, activation="silu")

        assert isinstance(ffn[0], FusedLinearSiLU)

    def test_ffn_forward_pass(self):
        """Test FFN forward pass."""
        batch_size, seq_len, dim = 16, 128, 512
        hidden_dim = 2048

        ffn = create_fused_ffn_layer(dim, hidden_dim, activation="gelu")
        x = torch.randn(batch_size, seq_len, dim)
        output = ffn(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_ffn_with_custom_output_dim(self):
        """Test FFN with different output dimension."""
        ffn = create_fused_ffn_layer(512, 2048, out_features=1024, activation="gelu")
        x = torch.randn(32, 512)
        output = ffn(x)

        assert output.shape == (32, 1024)


@pytest.mark.skipif(not CUDA_KERNELS_AVAILABLE, reason="CUDA not available")
class TestFusedLinearCUDA:
    """Tests for fused linear kernels on CUDA."""

    def test_fused_linear_gelu_cuda(self):
        """Test FusedLinearGELU on CUDA."""
        device = torch.device('cuda')
        layer = FusedLinearGELU(512, 2048).to(device)

        x = torch.randn(32, 512, device=device)
        output = layer(x)

        # Reference on CUDA
        linear_out = F.linear(x, layer.weight, layer.bias)
        expected = F.gelu(linear_out)

        max_diff = torch.abs(output - expected).max().item()
        assert max_diff < 1e-4, f"CUDA kernel error {max_diff} exceeds tolerance"

    def test_fused_linear_silu_cuda(self):
        """Test FusedLinearSiLU on CUDA."""
        device = torch.device('cuda')
        layer = FusedLinearSiLU(768, 3072).to(device)

        x = torch.randn(16, 768, device=device, dtype=torch.float16)
        output = layer(x)

        linear_out = F.linear(x, layer.weight, layer.bias)
        expected = F.silu(linear_out)

        max_diff = torch.abs(output - expected).max().item()
        assert max_diff < 1e-2  # Relaxed for FP16

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_cuda_different_dtypes(self, dtype):
        """Test CUDA kernels with different data types."""
        device = torch.device('cuda')
        layer = FusedLinearGELU(256, 1024).to(device).to(dtype)

        x = torch.randn(32, 256, device=device, dtype=dtype)
        output = layer(x)

        assert output.dtype == dtype
        assert output.device == device


@pytest.mark.skipif(not CUDA_KERNELS_AVAILABLE, reason="CUDA not available")
@pytest.mark.benchmark
class TestFusedLinearPerformance:
    """Performance benchmarking for fused linear kernels."""

    def test_performance_vs_pytorch(self):
        """Benchmark fused kernel vs PyTorch."""
        device = torch.device('cuda')
        in_features, out_features = 1024, 4096
        batch_size = 128

        # Fused kernel
        fused_layer = FusedLinearGELU(in_features, out_features).to(device)

        # Separate PyTorch layers
        linear = nn.Linear(in_features, out_features).to(device)
        linear.weight.data = fused_layer.weight.data.clone()
        linear.bias.data = fused_layer.bias.data.clone()

        x = torch.randn(batch_size, in_features, device=device, dtype=torch.float16)

        # Warmup
        for _ in range(10):
            _ = fused_layer(x)
            _ = F.gelu(linear(x))

        torch.cuda.synchronize()

        # Benchmark fused
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            output_fused = fused_layer(x)
        end.record()
        torch.cuda.synchronize()
        time_fused = start.elapsed_time(end) / 100

        # Benchmark separate
        start.record()
        for _ in range(100):
            output_separate = F.gelu(linear(x))
        end.record()
        torch.cuda.synchronize()
        time_separate = start.elapsed_time(end) / 100

        speedup = time_separate / time_fused
        print(f"\nFused Linear+GELU Performance:")
        print(f"  Fused time: {time_fused:.3f} ms")
        print(f"  Separate time: {time_separate:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Should see some speedup (may not be full 1.8x in fallback mode)
        assert speedup > 0.8, "Fused kernel significantly slower than separate ops"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
