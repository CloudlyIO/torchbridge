"""
Tests for Quantization Format Definitions

Tests QuantizationFormat enum, FormatSpec metadata, string round-trips,
and the FORMAT_SPECS registry.
"""

import pytest

from torchbridge.precision.quantization.formats import (
    FORMAT_SPECS,
    FormatSpec,
    QuantizationFormat,
    get_format_spec,
)


class TestQuantizationFormat:
    """Tests for the QuantizationFormat enum."""

    def test_all_formats_defined(self):
        """All expected format members should exist."""
        expected = [
            "NONE", "INT8_DYNAMIC", "INT8_DYNAMIC_ACTIVATIONS", "INT4_WEIGHT_ONLY",
            "FP8_E4M3", "FP8_E5M2", "NVFP4", "BF16",
        ]
        for name in expected:
            assert hasattr(QuantizationFormat, name), f"Missing format: {name}"

    def test_format_count(self):
        """Should have exactly 10 formats."""
        assert len(QuantizationFormat) == 8

    def test_string_round_trip(self):
        """Every format's value should parse back to itself."""
        for fmt in QuantizationFormat:
            parsed = QuantizationFormat.from_string(fmt.value)
            assert parsed == fmt, f"Round-trip failed for {fmt.value}"

    def test_from_string_aliases(self):
        """Common aliases should resolve correctly."""
        assert QuantizationFormat.from_string("int8") == QuantizationFormat.INT8_DYNAMIC
        assert QuantizationFormat.from_string("int4") == QuantizationFormat.INT4_WEIGHT_ONLY
        assert QuantizationFormat.from_string("fp8") == QuantizationFormat.FP8_E4M3
        assert QuantizationFormat.from_string("fp4") == QuantizationFormat.NVFP4

    def test_from_string_case_insensitive(self):
        """Parsing should be case-insensitive."""
        assert QuantizationFormat.from_string("INT8_DYNAMIC") == QuantizationFormat.INT8_DYNAMIC
        assert QuantizationFormat.from_string("Fp8_E4M3") == QuantizationFormat.FP8_E4M3

    def test_from_string_with_hyphens(self):
        """Hyphens should be normalized to underscores."""
        assert QuantizationFormat.from_string("int8-dynamic") == QuantizationFormat.INT8_DYNAMIC

    def test_from_string_unknown_raises(self):
        """Unknown format strings should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown quantization format"):
            QuantizationFormat.from_string("unknown_format_xyz")

    def test_values_are_strings(self):
        """All format values should be non-empty strings."""
        for fmt in QuantizationFormat:
            assert isinstance(fmt.value, str)
            assert len(fmt.value) > 0


class TestFormatSpec:
    """Tests for FormatSpec metadata."""

    def test_format_spec_is_frozen(self):
        """FormatSpec should be immutable."""
        spec = FormatSpec(
            bits=8, display_name="Test", perplexity_tolerance_pct=1.0,
            memory_reduction_pct=50.0, requires_calibration=False,
            requires_torchao=False,
        )
        with pytest.raises(AttributeError):
            spec.bits = 4  # type: ignore[misc]

    def test_all_formats_have_specs(self):
        """Every QuantizationFormat should have a FormatSpec entry."""
        for fmt in QuantizationFormat:
            assert fmt in FORMAT_SPECS, f"Missing spec for {fmt.value}"

    def test_spec_bits_positive(self):
        """All bit widths should be positive."""
        for fmt, spec in FORMAT_SPECS.items():
            assert spec.bits > 0, f"{fmt.value} has non-positive bits"

    def test_spec_display_names_unique(self):
        """Display names should be unique across formats."""
        names = [spec.display_name for spec in FORMAT_SPECS.values()]
        assert len(names) == len(set(names)), "Duplicate display names found"

    def test_spec_memory_reduction_range(self):
        """Memory reduction should be in [0, 100]."""
        for fmt, spec in FORMAT_SPECS.items():
            assert 0.0 <= spec.memory_reduction_pct <= 100.0, (
                f"{fmt.value} memory_reduction_pct out of range: {spec.memory_reduction_pct}"
            )

    def test_spec_perplexity_tolerance_non_negative(self):
        """Perplexity tolerance should be non-negative."""
        for fmt, spec in FORMAT_SPECS.items():
            assert spec.perplexity_tolerance_pct >= 0.0, (
                f"{fmt.value} has negative perplexity tolerance"
            )

    def test_none_format_has_zero_reduction(self):
        """NONE format should have 0% memory reduction."""
        spec = FORMAT_SPECS[QuantizationFormat.NONE]
        assert spec.memory_reduction_pct == 0.0
        assert spec.bits == 32

    def test_int4_formats_have_75_pct_reduction(self):
        """INT4 formats should claim ~75% memory reduction."""
        for fmt in (QuantizationFormat.INT4_WEIGHT_ONLY,):
            assert FORMAT_SPECS[fmt].memory_reduction_pct == 75.0

    def test_int8_dynamic_activations_spec(self):
        """INT8_DYNAMIC_ACTIVATIONS should not require calibration but does need torchao."""
        spec = FORMAT_SPECS[QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS]
        assert spec.requires_calibration is False
        assert spec.requires_torchao is True


class TestGetFormatSpec:
    """Tests for the get_format_spec helper."""

    def test_returns_correct_spec(self):
        """get_format_spec should return the correct FormatSpec."""
        spec = get_format_spec(QuantizationFormat.FP8_E4M3)
        assert spec.bits == 8
        assert spec.display_name == "FP8 E4M3"

    def test_all_formats_accessible(self):
        """get_format_spec should work for every format."""
        for fmt in QuantizationFormat:
            spec = get_format_spec(fmt)
            assert isinstance(spec, FormatSpec)
