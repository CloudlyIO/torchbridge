"""
Tests for Structured Output Processor

Tests output format enum, processor configuration, validation, and
graceful xgrammar fallback.
"""

import pytest

from torchbridge.inference.structured.output_format import (
    OUTPUT_FORMAT_SPECS,
    OutputFormat,
    get_format_spec,
)
from torchbridge.inference.structured.processor import (
    StructuredOutputProcessor,
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

    def test_get_format_spec(self):
        spec = get_format_spec(OutputFormat.REGEX)
        assert spec.display_name == "Regex"


class TestStructuredOutputProcessor:
    """Tests for StructuredOutputProcessor."""

    def test_text_format_no_processors(self):
        """TEXT format returns empty logits processors."""
        proc = StructuredOutputProcessor(format=OutputFormat.TEXT)
        assert proc.get_logits_processor() == []

    def test_json_schema_requires_schema(self):
        """JSON_SCHEMA without schema raises ValueError."""
        with pytest.raises(ValueError, match="schema is required"):
            StructuredOutputProcessor(format=OutputFormat.JSON_SCHEMA)

    def test_regex_requires_pattern(self):
        """REGEX without pattern raises ValueError."""
        with pytest.raises(ValueError, match="pattern is required"):
            StructuredOutputProcessor(format=OutputFormat.REGEX)

    def test_validate_text_always_true(self):
        proc = StructuredOutputProcessor(format=OutputFormat.TEXT)
        assert proc.validate_output("anything goes") is True

    def test_validate_json_valid(self):
        proc = StructuredOutputProcessor(format=OutputFormat.JSON)
        assert proc.validate_output('{"key": "value"}') is True

    def test_validate_json_invalid(self):
        proc = StructuredOutputProcessor(format=OutputFormat.JSON)
        assert proc.validate_output("not json") is False

    def test_validate_json_schema_valid(self):
        schema = {"type": "object", "required": ["name"]}
        proc = StructuredOutputProcessor(
            format=OutputFormat.JSON_SCHEMA, schema=schema
        )
        assert proc.validate_output('{"name": "test"}') is True

    def test_validate_json_schema_missing_required(self):
        schema = {"type": "object", "required": ["name"]}
        proc = StructuredOutputProcessor(
            format=OutputFormat.JSON_SCHEMA, schema=schema
        )
        assert proc.validate_output('{"other": "value"}') is False

    def test_validate_json_schema_wrong_type(self):
        schema = {"type": "object"}
        proc = StructuredOutputProcessor(
            format=OutputFormat.JSON_SCHEMA, schema=schema
        )
        assert proc.validate_output('"just a string"') is False

    def test_validate_regex_match(self):
        proc = StructuredOutputProcessor(
            format=OutputFormat.REGEX, pattern=r"\d{3}-\d{4}"
        )
        assert proc.validate_output("123-4567") is True

    def test_validate_regex_no_match(self):
        proc = StructuredOutputProcessor(
            format=OutputFormat.REGEX, pattern=r"\d{3}-\d{4}"
        )
        assert proc.validate_output("abc-defg") is False

    def test_is_available(self):
        """is_available returns bool (may be True or False)."""
        result = StructuredOutputProcessor.is_available()
        assert isinstance(result, bool)

    def test_get_info(self):
        proc = StructuredOutputProcessor(
            format=OutputFormat.JSON_SCHEMA,
            schema={"type": "object"},
        )
        info = proc.get_info()
        assert info["format"] == "json_schema"
        assert info["has_schema"] is True
        assert info["has_pattern"] is False
        assert "xgrammar_available" in info

    def test_format_property(self):
        proc = StructuredOutputProcessor(format=OutputFormat.JSON)
        assert proc.format == OutputFormat.JSON
