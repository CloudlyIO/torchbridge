"""
Tests for Structured Output Format enum and specs.
"""

import pytest

from torchbridge.inference.output_format import (
    OUTPUT_FORMAT_SPECS,
    OutputFormat,
)


class TestOutputFormat:
    """Tests for OutputFormat enum."""

    def test_all_formats_exist(self):
        expected = {"text", "json", "json_schema", "regex"}
        actual = {f.value for f in OutputFormat}
        assert actual == expected

    def test_from_string_exact(self):
        for fmt in OutputFormat:
            assert OutputFormat.from_string(fmt.value) == fmt

    def test_from_string_aliases(self):
        assert OutputFormat.from_string("plain") == OutputFormat.TEXT
        assert OutputFormat.from_string("plaintext") == OutputFormat.TEXT
        assert OutputFormat.from_string("schema") == OutputFormat.JSON_SCHEMA
        assert OutputFormat.from_string("regexp") == OutputFormat.REGEX

    def test_from_string_case_insensitive(self):
        assert OutputFormat.from_string("JSON") == OutputFormat.JSON
        assert OutputFormat.from_string("Regex") == OutputFormat.REGEX

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown output format"):
            OutputFormat.from_string("xml")


class TestOutputFormatSpecs:
    """Tests for OUTPUT_FORMAT_SPECS."""

    def test_all_formats_have_specs(self):
        for fmt in OutputFormat:
            assert fmt in OUTPUT_FORMAT_SPECS

    def test_text_no_grammar(self):
        spec = OUTPUT_FORMAT_SPECS[OutputFormat.TEXT]
        assert spec.requires_grammar_engine is False

    def test_json_requires_grammar(self):
        spec = OUTPUT_FORMAT_SPECS[OutputFormat.JSON]
        assert spec.requires_grammar_engine is True

    def test_spec_dict_lookup(self):
        """Direct dict lookup returns correct spec."""
        spec = OUTPUT_FORMAT_SPECS[OutputFormat.REGEX]
        assert spec.display_name == "Regex"
