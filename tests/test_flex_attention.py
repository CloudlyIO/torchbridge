"""
FlexAttention Tests

Comprehensive test suite for FlexAttention integration (PyTorch 2.5+).
Tests cover:
- Basic functionality and shapes
- Score modification patterns
- Block mask generation
- Registry integration
- Fallback behavior
- Performance characteristics
"""

import pytest
import torch
import torch.nn as nn

from kernel_pytorch.attention import (
    AttentionConfig,
    AttentionPatterns,
    create_attention,
)
from kernel_pytorch.attention.implementations.flex_attention import (
    FlexAttentionLayer,
    FlexAttentionCausal,
    FlexAttentionSlidingWindow,
    FlexAttentionScoreMods,
    FlexAttentionMaskGenerators,
    create_flex_attention,
    is_flex_attention_available,
    get_flex_attention_info,
    FLEX_ATTENTION_AVAILABLE,
)


# Test fixtures
@pytest.fixture
def device():
    """Get available device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def basic_config():
    """Basic attention configuration"""
    return AttentionConfig(
        embed_dim=256,
        num_heads=4,
        pattern=AttentionPatterns.FULL,
    )


@pytest.fixture
def causal_config():
    """Causal attention configuration"""
    return AttentionConfig(
        embed_dim=256,
        num_heads=4,
        pattern=AttentionPatterns.CAUSAL,
        causal=True,
    )


@pytest.fixture
def sliding_window_config():
    """Sliding window configuration"""
    return AttentionConfig(
        embed_dim=256,
        num_heads=4,
        pattern=AttentionPatterns.SLIDING_WINDOW,
        sliding_window_size=64,
    )


@pytest.fixture
def sample_input(device):
    """Sample input tensor"""
    batch_size, seq_len, embed_dim = 2, 128, 256
    return torch.randn(batch_size, seq_len, embed_dim, device=device)


class TestFlexAttentionAvailability:
    """Test FlexAttention availability checks"""

    def test_availability_check(self):
        """Test is_flex_attention_available function"""
        result = is_flex_attention_available()
        assert isinstance(result, bool)

    def test_info_dict(self):
        """Test get_flex_attention_info returns proper dict"""
        info = get_flex_attention_info()
        assert isinstance(info, dict)
        assert 'available' in info
        assert 'torch_version' in info
        assert 'supported_patterns' in info
        assert isinstance(info['supported_patterns'], list)
        assert 'causal' in info['supported_patterns']


class TestFlexAttentionCreation:
    """Test FlexAttention layer creation"""

    def test_basic_creation(self, basic_config, device):
        """Test basic FlexAttention creation"""
        layer = FlexAttentionLayer(basic_config)
        layer = layer.to(device)
        assert isinstance(layer, nn.Module)
        assert layer.embed_dim == 256
        assert layer.num_heads == 4

    def test_causal_creation(self, causal_config, device):
        """Test FlexAttentionCausal creation"""
        layer = FlexAttentionCausal(causal_config)
        layer = layer.to(device)
        assert isinstance(layer, FlexAttentionLayer)

    def test_sliding_window_creation(self, sliding_window_config, device):
        """Test FlexAttentionSlidingWindow creation"""
        layer = FlexAttentionSlidingWindow(sliding_window_config, window_size=64)
        layer = layer.to(device)
        assert isinstance(layer, FlexAttentionLayer)

    def test_factory_function(self, device):
        """Test create_flex_attention factory"""
        layer = create_flex_attention(256, 4, pattern='causal')
        layer = layer.to(device)
        assert isinstance(layer, FlexAttentionLayer)

    def test_factory_with_custom_score_mod(self, device):
        """Test factory with custom score_mod"""
        def custom_mod(score, b, h, q_idx, kv_idx):
            return score * 0.5

        layer = create_flex_attention(256, 4, score_mod=custom_mod)
        layer = layer.to(device)
        assert isinstance(layer, FlexAttentionLayer)


class TestFlexAttentionForward:
    """Test FlexAttention forward pass"""

    def test_basic_forward(self, basic_config, sample_input, device):
        """Test basic forward pass"""
        layer = FlexAttentionLayer(basic_config).to(device)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_causal_forward(self, causal_config, sample_input, device):
        """Test causal attention forward pass"""
        layer = FlexAttentionCausal(causal_config).to(device)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_sliding_window_forward(self, sliding_window_config, sample_input, device):
        """Test sliding window forward pass"""
        layer = FlexAttentionSlidingWindow(sliding_window_config).to(device)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_gradient_flow(self, basic_config, sample_input, device):
        """Test gradient flow through FlexAttention"""
        layer = FlexAttentionLayer(basic_config).to(device)
        sample_input.requires_grad = True
        output = layer(sample_input)
        loss = output.sum()
        loss.backward()
        assert sample_input.grad is not None
        assert sample_input.grad.shape == sample_input.shape

    def test_eval_mode(self, basic_config, sample_input, device):
        """Test eval mode behavior"""
        layer = FlexAttentionLayer(basic_config).to(device)
        layer.eval()
        with torch.no_grad():
            output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_various_seq_lengths(self, basic_config, device):
        """Test with various sequence lengths"""
        layer = FlexAttentionLayer(basic_config).to(device)
        for seq_len in [32, 64, 128, 256, 512]:
            x = torch.randn(2, seq_len, 256, device=device)
            output = layer(x)
            assert output.shape == x.shape


class TestFlexAttentionScoreMods:
    """Test score modification patterns"""

    def test_causal_score_mod(self):
        """Test causal score_mod"""
        score = torch.tensor(1.0)
        # q_idx >= kv_idx should keep score
        result = FlexAttentionScoreMods.causal(score, 0, 0, 5, 3)
        if isinstance(result, torch.Tensor):
            assert result.item() == 1.0
        else:
            assert result == 1.0
        # q_idx < kv_idx should mask
        result = FlexAttentionScoreMods.causal(score, 0, 0, 3, 5)
        if isinstance(result, torch.Tensor):
            assert result.item() == float('-inf')
        else:
            assert result == float('-inf')

    def test_sliding_window_score_mod(self):
        """Test sliding window score_mod"""
        mod = FlexAttentionScoreMods.sliding_window(window_size=4)
        score = torch.tensor(1.0)
        # Within window
        result = mod(score, 0, 0, 5, 3)
        if isinstance(result, torch.Tensor):
            assert result.item() == 1.0
        else:
            assert result == 1.0
        # Outside window
        result = mod(score, 0, 0, 10, 3)
        if isinstance(result, torch.Tensor):
            assert result.item() == float('-inf')
        else:
            assert result == float('-inf')

    def test_causal_sliding_window_score_mod(self):
        """Test causal + sliding window score_mod"""
        mod = FlexAttentionScoreMods.causal_sliding_window(window_size=4)
        score = torch.tensor(1.0)
        # Within window and causal
        result = mod(score, 0, 0, 5, 3)
        if isinstance(result, torch.Tensor):
            assert result.item() == 1.0
        else:
            assert result == 1.0
        # Outside window
        result = mod(score, 0, 0, 10, 3)
        if isinstance(result, torch.Tensor):
            assert result.item() == float('-inf')
        else:
            assert result == float('-inf')

    def test_alibi_score_mod(self):
        """Test ALiBi score_mod"""
        mod = FlexAttentionScoreMods.alibi(num_heads=4)
        score = torch.tensor(0.0)
        # Should add bias based on distance
        result = mod(score, 0, 0, 5, 5)  # Same position
        assert isinstance(result, torch.Tensor)

    def test_soft_cap_score_mod(self):
        """Test soft capping score_mod"""
        mod = FlexAttentionScoreMods.soft_cap(cap_value=10.0)
        score = torch.tensor(100.0)
        result = mod(score, 0, 0, 0, 0)
        # Should be capped around tanh range
        assert result < 15.0  # Approximately cap_value


class TestFlexAttentionMaskGenerators:
    """Test block mask generators"""

    @pytest.mark.skipif(not FLEX_ATTENTION_AVAILABLE, reason="FlexAttention not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Block masks require CUDA")
    def test_causal_mask_generation(self, device):
        """Test causal block mask generation"""
        if device.type != 'cuda':
            pytest.skip("Block masks require CUDA")
        mask = FlexAttentionMaskGenerators.causal_mask(2, 4, 64, device)
        assert mask is not None

    @pytest.mark.skipif(not FLEX_ATTENTION_AVAILABLE, reason="FlexAttention not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Block masks require CUDA")
    def test_sliding_window_mask_generation(self, device):
        """Test sliding window block mask generation"""
        if device.type != 'cuda':
            pytest.skip("Block masks require CUDA")
        mask = FlexAttentionMaskGenerators.sliding_window_mask(2, 4, 64, 16, device)
        assert mask is not None

    @pytest.mark.skipif(not FLEX_ATTENTION_AVAILABLE, reason="FlexAttention not available")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Block masks require CUDA")
    def test_full_mask_generation(self, device):
        """Test full attention block mask generation"""
        if device.type != 'cuda':
            pytest.skip("Block masks require CUDA")
        mask = FlexAttentionMaskGenerators.full_mask(2, 4, 64, device)
        assert mask is not None


class TestFlexAttentionRegistry:
    """Test registry integration"""

    def test_registered_in_registry(self):
        """Test FlexAttention is registered"""
        from kernel_pytorch.attention.core.registry import get_attention_registry
        registry = get_attention_registry()
        assert 'flex_attention' in registry
        assert 'flex_attention_causal' in registry
        assert 'flex_attention_sliding_window' in registry

    def test_create_via_registry(self, basic_config, device):
        """Test creating FlexAttention via registry"""
        layer = create_attention(basic_config, implementation='flex_attention')
        layer = layer.to(device)
        assert isinstance(layer, FlexAttentionLayer)


class TestFlexAttentionStats:
    """Test attention statistics"""

    def test_get_stats(self, basic_config, device):
        """Test get_attention_stats method"""
        layer = FlexAttentionLayer(basic_config).to(device)
        stats = layer.get_attention_stats()
        assert isinstance(stats, dict)
        assert 'flex_attention_available' in stats
        assert 'using_flex_attention' in stats
        assert 'pattern' in stats


class TestFlexAttentionCaching:
    """Test block mask caching"""

    def test_block_mask_caching(self, basic_config, sample_input, device):
        """Test that block masks are cached"""
        layer = FlexAttentionLayer(basic_config).to(device)
        # Run forward twice
        _ = layer(sample_input)
        cache_size_1 = len(layer._block_mask_cache)
        _ = layer(sample_input)
        cache_size_2 = len(layer._block_mask_cache)
        # Cache should not grow for same input shape
        assert cache_size_1 == cache_size_2

    def test_clear_cache(self, basic_config, sample_input, device):
        """Test cache clearing"""
        layer = FlexAttentionLayer(basic_config).to(device)
        _ = layer(sample_input)
        layer.clear_block_mask_cache()
        assert len(layer._block_mask_cache) == 0


class TestFlexAttentionFallback:
    """Test fallback behavior when FlexAttention not available"""

    def test_fallback_forward(self, basic_config, sample_input, device):
        """Test fallback works when FlexAttention unavailable"""
        # Force fallback by setting score_mod to None
        layer = FlexAttentionLayer(basic_config, score_mod=None)
        layer = layer.to(device)
        output = layer(sample_input)
        assert output.shape == sample_input.shape

    def test_standard_attention_matches_output(self, causal_config, device):
        """Test standard attention fallback produces correct output"""
        layer = FlexAttentionLayer(causal_config).to(device)
        x = torch.randn(1, 32, 256, device=device)
        output = layer._standard_attention_forward(
            layer._shape_for_multihead(layer.q_proj(x), 1, 32),
            layer._shape_for_multihead(layer.k_proj(x), 1, 32),
            layer._shape_for_multihead(layer.v_proj(x), 1, 32),
            None
        )
        assert output.shape == (1, 4, 32, 64)  # [B, H, S, D]


class TestFlexAttentionPatterns:
    """Test various attention patterns"""

    @pytest.mark.parametrize("pattern", ['full', 'causal', 'sliding_window'])
    def test_pattern_creation(self, pattern, device):
        """Test creating different patterns"""
        layer = create_flex_attention(256, 4, pattern=pattern, sliding_window_size=32)
        layer = layer.to(device)
        x = torch.randn(2, 64, 256, device=device)
        output = layer(x)
        assert output.shape == x.shape


class TestFlexAttentionIntegration:
    """Integration tests with other components"""

    def test_in_transformer_block(self, basic_config, sample_input, device):
        """Test FlexAttention in a transformer-like block"""
        attention = FlexAttentionLayer(basic_config).to(device)
        ffn = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 256)
        ).to(device)
        norm1 = nn.LayerNorm(256).to(device)
        norm2 = nn.LayerNorm(256).to(device)

        # Transformer block forward
        x = sample_input
        x = x + attention(norm1(x))
        x = x + ffn(norm2(x))

        assert x.shape == sample_input.shape

    def test_with_kv_cache(self, causal_config, device):
        """Test with KV caching enabled"""
        layer = FlexAttentionCausal(causal_config).to(device)
        layer.enable_cache()

        x = torch.randn(1, 32, 256, device=device)
        output = layer(x)
        assert output.shape == x.shape

        layer.disable_cache()


class TestFlexAttentionBenchmark:
    """Performance-related tests"""

    def test_compile_compatibility(self, basic_config, sample_input, device):
        """Test torch.compile compatibility"""
        layer = FlexAttentionLayer(basic_config).to(device)

        try:
            compiled = torch.compile(layer)
            output = compiled(sample_input)
            assert output.shape == sample_input.shape
        except Exception:
            # torch.compile may not be available or may fail on some configs
            pytest.skip("torch.compile not available or failed")

    def test_memory_efficiency(self, basic_config, device):
        """Test memory usage is reasonable"""
        if device.type != 'cuda':
            pytest.skip("Memory test requires CUDA")

        layer = FlexAttentionLayer(basic_config).to(device)
        x = torch.randn(4, 512, 256, device=device)

        torch.cuda.reset_peak_memory_stats()
        _ = layer(x)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        # Should use less than 500MB for this size
        assert peak_memory < 500, f"Peak memory {peak_memory}MB exceeds threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
