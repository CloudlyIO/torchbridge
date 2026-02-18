"""
Tests for Speculative Decoding Method Definitions

Tests enum values, from_string parsing, aliases, specs, and edge cases.
"""

import pytest

from torchbridge.inference.speculative.methods import (
    SPECULATIVE_METHOD_SPECS,
    SpeculativeMethod,
    SpeculativeMethodSpec,
    get_method_spec,
)


class TestSpeculativeMethod:
    """Tests for SpeculativeMethod enum."""

    def test_all_methods_exist(self):
        """All expected methods are defined."""
        expected = {"none", "draft_model", "eagle", "layer_skip", "medusa", "prompt_lookup"}
        actual = {m.value for m in SpeculativeMethod}
        assert actual == expected

    def test_from_string_exact_values(self):
        """from_string resolves exact enum values."""
        for method in SpeculativeMethod:
            assert SpeculativeMethod.from_string(method.value) == method

    def test_from_string_case_insensitive(self):
        """from_string is case-insensitive."""
        assert SpeculativeMethod.from_string("DRAFT_MODEL") == SpeculativeMethod.DRAFT_MODEL
        assert SpeculativeMethod.from_string("Eagle") == SpeculativeMethod.EAGLE

    def test_from_string_aliases(self):
        """from_string resolves aliases."""
        assert SpeculativeMethod.from_string("draft") == SpeculativeMethod.DRAFT_MODEL
        assert SpeculativeMethod.from_string("skip") == SpeculativeMethod.LAYER_SKIP
        assert SpeculativeMethod.from_string("lookup") == SpeculativeMethod.PROMPT_LOOKUP
        assert SpeculativeMethod.from_string("auto") == SpeculativeMethod.NONE

    def test_from_string_with_hyphens(self):
        """from_string handles hyphens."""
        assert SpeculativeMethod.from_string("draft-model") == SpeculativeMethod.DRAFT_MODEL
        assert SpeculativeMethod.from_string("layer-skip") == SpeculativeMethod.LAYER_SKIP

    def test_from_string_invalid(self):
        """from_string raises ValueError for invalid input."""
        with pytest.raises(ValueError, match="Unknown speculative method"):
            SpeculativeMethod.from_string("nonexistent")

    def test_from_string_whitespace(self):
        """from_string strips whitespace."""
        assert SpeculativeMethod.from_string("  eagle  ") == SpeculativeMethod.EAGLE


class TestSpeculativeMethodSpecs:
    """Tests for SPECULATIVE_METHOD_SPECS."""

    def test_all_methods_have_specs(self):
        """Every SpeculativeMethod has a corresponding spec."""
        for method in SpeculativeMethod:
            assert method in SPECULATIVE_METHOD_SPECS, f"Missing spec for {method}"

    def test_specs_are_frozen(self):
        """Specs are frozen dataclasses."""
        for spec in SPECULATIVE_METHOD_SPECS.values():
            assert isinstance(spec, SpeculativeMethodSpec)
            with pytest.raises(AttributeError):
                spec.display_name = "modified"  # type: ignore[misc]

    def test_draft_model_requires_draft(self):
        """DRAFT_MODEL spec requires a draft model."""
        spec = SPECULATIVE_METHOD_SPECS[SpeculativeMethod.DRAFT_MODEL]
        assert spec.requires_draft_model is True

    def test_prompt_lookup_no_draft(self):
        """PROMPT_LOOKUP does not require a draft model."""
        spec = SPECULATIVE_METHOD_SPECS[SpeculativeMethod.PROMPT_LOOKUP]
        assert spec.requires_draft_model is False
        assert spec.requires_hardware_support is False

    def test_get_method_spec(self):
        """get_method_spec returns correct spec."""
        spec = get_method_spec(SpeculativeMethod.EAGLE)
        assert spec.display_name == "EAGLE"
        assert spec.requires_hardware_support is True
