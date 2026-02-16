"""Tests for GQA/MQA support in AttentionModuleConfig."""

import pytest

from torchbridge.attention.core.config import AttentionModuleConfig


class TestGQAConfig:
    """Validate grouped-query attention configuration."""

    def test_valid_gqa(self):
        """Standard GQA: 32 query heads, 8 KV heads."""
        config = AttentionModuleConfig(embed_dim=256, num_heads=32, num_kv_heads=8)
        assert config.num_kv_heads == 8
        assert config.kv_head_repeat_factor == 4

    def test_valid_mqa(self):
        """Multi-query attention: single KV head."""
        config = AttentionModuleConfig(embed_dim=256, num_heads=32, num_kv_heads=1)
        assert config.num_kv_heads == 1
        assert config.kv_head_repeat_factor == 32

    def test_mha_default(self):
        """num_kv_heads=None means standard MHA."""
        config = AttentionModuleConfig(embed_dim=256, num_heads=8)
        assert config.num_kv_heads is None
        assert config.kv_head_repeat_factor == 1

    def test_invalid_not_divisible(self):
        """num_heads must be divisible by num_kv_heads."""
        with pytest.raises(ValueError, match="divisible"):
            AttentionModuleConfig(embed_dim=256, num_heads=32, num_kv_heads=5)

    def test_equal_heads_is_mha(self):
        """num_kv_heads == num_heads is equivalent to MHA."""
        config = AttentionModuleConfig(embed_dim=256, num_heads=8, num_kv_heads=8)
        assert config.kv_head_repeat_factor == 1

    def test_to_dict_includes_kv_heads(self):
        config = AttentionModuleConfig(embed_dim=256, num_heads=8, num_kv_heads=2)
        d = config.to_dict()
        assert d["num_kv_heads"] == 2

    def test_to_dict_kv_heads_none(self):
        config = AttentionModuleConfig(embed_dim=256, num_heads=8)
        d = config.to_dict()
        assert d["num_kv_heads"] is None

    def test_head_dim_auto_calculated(self):
        config = AttentionModuleConfig(embed_dim=256, num_heads=8, num_kv_heads=2)
        assert config.head_dim == 32

    def test_gqa_with_causal(self):
        """GQA should work with causal attention."""
        config = AttentionModuleConfig(
            embed_dim=256, num_heads=32, num_kv_heads=8, causal=True
        )
        assert config.causal is True
        assert config.kv_head_repeat_factor == 4
