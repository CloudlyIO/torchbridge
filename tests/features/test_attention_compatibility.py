"""
Test Attention Framework Compatibility

Quick compatibility tests for the unified attention framework to ensure
core functionality works before comprehensive test updates.
"""

import pytest
import torch

from kernel_pytorch.attention import (
    AttentionConfig,
    AttentionPatterns,
    FlashAttention3,
    FP8AttentionConfig,
    create_attention,
)


class TestUnifiedAttentionFramework:
    """Test the unified attention framework functionality"""

    @pytest.fixture
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_basic_attention_creation(self, device):
        """Test basic attention layer creation"""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=8,
            use_flash_attention=True
        )

        attention = FlashAttention3(config).to(device)

        assert attention.embed_dim == 256
        assert attention.num_heads == 8
        assert attention.head_dim == 32

    def test_attention_forward_pass(self, device):
        """Test attention forward pass works"""
        config = AttentionConfig(
            embed_dim=512,
            num_heads=8,
            use_flash_attention=True
        )

        attention = FlashAttention3(config).to(device)

        # Test input
        batch_size, seq_len = 2, 64
        x = torch.randn(batch_size, seq_len, 512, device=device)

        with torch.no_grad():
            output = attention(x)

        assert output.shape == x.shape
        assert output.device.type == device

    def test_factory_function(self, device):
        """Test the create_attention factory function"""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=4,
            use_flash_attention=True
        )

        attention = create_attention(config).to(device)

        # Test it works
        x = torch.randn(1, 32, 256, device=device)
        with torch.no_grad():
            output = attention(x)

        assert output.shape == x.shape

    def test_fp8_configuration(self, device):
        """Test FP8 configuration support"""
        fp8_config = FP8AttentionConfig(use_fp8=False)  # Safe for CPU/GPU
        config = AttentionConfig(
            embed_dim=128,
            num_heads=4,
            use_flash_attention=True,
            fp8_config=fp8_config
        )

        attention = FlashAttention3(config).to(device)

        # Test functionality
        x = torch.randn(1, 16, 128, device=device)
        with torch.no_grad():
            output = attention(x)

        assert output.shape == x.shape

    def test_attention_patterns(self, device):
        """Test different attention patterns"""
        patterns_to_test = [
            AttentionPatterns.FULL,
            AttentionPatterns.CAUSAL,
        ]

        for pattern in patterns_to_test:
            config = AttentionConfig(
                embed_dim=128,
                num_heads=4,
                pattern=pattern,
                use_flash_attention=True
            )

            attention = FlashAttention3(config).to(device)

            # Test functionality
            x = torch.randn(1, 16, 128, device=device)
            with torch.no_grad():
                output = attention(x)

            assert output.shape == x.shape

    def test_attention_statistics(self, device):
        """Test attention statistics reporting"""
        config = AttentionConfig(
            embed_dim=256,
            num_heads=8,
            use_flash_attention=True
        )

        attention = FlashAttention3(config).to(device)
        stats = attention.get_attention_stats()

        assert 'pattern' in stats
        assert 'embed_dim' in stats
        assert 'backend' in stats
        assert stats['embed_dim'] == 256
        assert stats['num_heads'] == 8
