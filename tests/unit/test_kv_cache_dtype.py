"""
Tests for KV-Cache Data Type Definitions

Tests KVCacheDtype enum, KVDtypeSpec metadata, string round-trips,
and the KV_DTYPE_SPECS registry.
"""

import pytest
import torch

from torchbridge.models.kv.cache_dtype import (
    KV_DTYPE_SPECS,
    KVCacheDtype,
    KVDtypeSpec,
    get_kv_dtype_spec,
)


class TestKVCacheDtype:
    """Tests for the KVCacheDtype enum."""

    def test_all_dtypes_defined(self):
        """All expected dtype members should exist."""
        expected = ["FP16", "BF16", "FP8_E4M3", "NVFP4", "PASSTHROUGH"]
        for name in expected:
            assert hasattr(KVCacheDtype, name), f"Missing dtype: {name}"

    def test_dtype_count(self):
        """Should have exactly 5 dtypes."""
        assert len(KVCacheDtype) == 5

    def test_string_round_trip(self):
        """Every dtype's value should parse back to itself."""
        for dtype in KVCacheDtype:
            parsed = KVCacheDtype.from_string(dtype.value)
            assert parsed == dtype, f"Round-trip failed for {dtype.value}"

    def test_from_string_aliases(self):
        """Common aliases should resolve correctly."""
        assert KVCacheDtype.from_string("fp8") == KVCacheDtype.FP8_E4M3
        assert KVCacheDtype.from_string("fp4") == KVCacheDtype.NVFP4
        assert KVCacheDtype.from_string("float16") == KVCacheDtype.FP16
        assert KVCacheDtype.from_string("bfloat16") == KVCacheDtype.BF16
        assert KVCacheDtype.from_string("none") == KVCacheDtype.PASSTHROUGH
        assert KVCacheDtype.from_string("auto") == KVCacheDtype.PASSTHROUGH

    def test_from_string_case_insensitive(self):
        """Parsing should be case-insensitive."""
        assert KVCacheDtype.from_string("FP16") == KVCacheDtype.FP16
        assert KVCacheDtype.from_string("Bf16") == KVCacheDtype.BF16

    def test_from_string_with_hyphens(self):
        """Hyphens should be normalized to underscores."""
        assert KVCacheDtype.from_string("fp8-e4m3") == KVCacheDtype.FP8_E4M3

    def test_from_string_unknown_raises(self):
        """Unknown dtype strings should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown KV-cache dtype"):
            KVCacheDtype.from_string("unknown_dtype_xyz")

    def test_values_are_strings(self):
        """All dtype values should be non-empty strings."""
        for dtype in KVCacheDtype:
            assert isinstance(dtype.value, str)
            assert len(dtype.value) > 0


class TestKVDtypeSpec:
    """Tests for KVDtypeSpec metadata."""

    def test_spec_is_frozen(self):
        """KVDtypeSpec should be immutable."""
        spec = KVDtypeSpec(
            bits=16,
            display_name="Test",
            memory_factor=1.0,
            torch_dtype=torch.float16,
            requires_hardware_support=False,
        )
        with pytest.raises(AttributeError):
            spec.bits = 4  # type: ignore[misc]

    def test_all_dtypes_have_specs(self):
        """Every KVCacheDtype should have a KVDtypeSpec entry."""
        for dtype in KVCacheDtype:
            assert dtype in KV_DTYPE_SPECS, f"Missing spec for {dtype.value}"

    def test_spec_bits_non_negative(self):
        """All bit widths should be non-negative."""
        for dtype, spec in KV_DTYPE_SPECS.items():
            assert spec.bits >= 0, f"{dtype.value} has negative bits"

    def test_spec_display_names_unique(self):
        """Display names should be unique across dtypes."""
        names = [spec.display_name for spec in KV_DTYPE_SPECS.values()]
        assert len(names) == len(set(names)), "Duplicate display names found"

    def test_spec_memory_factor_range(self):
        """Memory factor should be in (0, 1.0]."""
        for dtype, spec in KV_DTYPE_SPECS.items():
            assert (
                0.0 < spec.memory_factor <= 1.0 or dtype == KVCacheDtype.PASSTHROUGH
            ), f"{dtype.value} memory_factor out of range: {spec.memory_factor}"

    def test_fp8_has_half_memory(self):
        """FP8 E4M3 should have 0.5x memory factor."""
        spec = KV_DTYPE_SPECS[KVCacheDtype.FP8_E4M3]
        assert spec.memory_factor == 0.5
        assert spec.bits == 8

    def test_nvfp4_has_quarter_memory(self):
        """NVFP4 should have 0.25x memory factor."""
        spec = KV_DTYPE_SPECS[KVCacheDtype.NVFP4]
        assert spec.memory_factor == 0.25
        assert spec.bits == 4

    def test_passthrough_has_no_torch_dtype(self):
        """PASSTHROUGH should have no torch_dtype."""
        spec = KV_DTYPE_SPECS[KVCacheDtype.PASSTHROUGH]
        assert spec.torch_dtype is None
        assert spec.requires_hardware_support is False

    def test_fp8_requires_hardware(self):
        """FP8 should require hardware support."""
        spec = KV_DTYPE_SPECS[KVCacheDtype.FP8_E4M3]
        assert spec.requires_hardware_support is True

    def test_fp16_torch_dtype(self):
        """FP16 should map to torch.float16."""
        spec = KV_DTYPE_SPECS[KVCacheDtype.FP16]
        assert spec.torch_dtype == torch.float16

    def test_bf16_torch_dtype(self):
        """BF16 should map to torch.bfloat16."""
        spec = KV_DTYPE_SPECS[KVCacheDtype.BF16]
        assert spec.torch_dtype == torch.bfloat16


class TestGetKVDtypeSpec:
    """Tests for the get_kv_dtype_spec helper."""

    def test_returns_correct_spec(self):
        """get_kv_dtype_spec should return the correct KVDtypeSpec."""
        spec = get_kv_dtype_spec(KVCacheDtype.FP8_E4M3)
        assert spec.bits == 8
        assert spec.display_name == "FP8 E4M3"

    def test_all_dtypes_accessible(self):
        """get_kv_dtype_spec should work for every dtype."""
        for dtype in KVCacheDtype:
            spec = get_kv_dtype_spec(dtype)
            assert isinstance(spec, KVDtypeSpec)
