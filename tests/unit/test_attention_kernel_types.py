"""Tests for AttentionKernelType enum."""

import pytest

from torchbridge.attention.dispatch.kernel_types import AttentionKernelType


class TestAttentionKernelType:
    """Validate kernel type enum completeness and values."""

    def test_all_kernel_types_defined(self):
        expected = {
            "FLEX_ATTENTION",
            "FLASH_ATTENTION_3",
            "FLASH_ATTENTION_2",
            "FLASH_ATTENTION_CK",
            "NEURONX_SDPA",
            "PALLAS_ATTENTION",
            "PYTORCH_SDPA",
        }
        actual = {member.name for member in AttentionKernelType}
        assert actual == expected

    def test_kernel_type_count(self):
        assert len(AttentionKernelType) == 7

    def test_string_values(self):
        assert AttentionKernelType.FLEX_ATTENTION.value == "flex_attention"
        assert AttentionKernelType.FLASH_ATTENTION_3.value == "flash_attention_3"
        assert AttentionKernelType.FLASH_ATTENTION_2.value == "flash_attention_2"
        assert AttentionKernelType.FLASH_ATTENTION_CK.value == "flash_attention_ck"
        assert AttentionKernelType.NEURONX_SDPA.value == "neuronx_sdpa"
        assert AttentionKernelType.PALLAS_ATTENTION.value == "pallas_attention"
        assert AttentionKernelType.PYTORCH_SDPA.value == "pytorch_sdpa"

    def test_lookup_by_value(self):
        assert AttentionKernelType("flex_attention") is AttentionKernelType.FLEX_ATTENTION

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            AttentionKernelType("nonexistent_kernel")
