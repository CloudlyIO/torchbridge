"""
Structured Output Format Definitions

Enum of output constraint formats with metadata specs for grammar-guided
generation.
"""

from dataclasses import dataclass
from enum import Enum


class OutputFormat(Enum):
    """Structured output formats for constrained generation."""

    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"
    REGEX = "regex"

    @classmethod
    def from_string(cls, value: str) -> "OutputFormat":
        """Convert string to OutputFormat, case-insensitive."""
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "plain": cls.TEXT,
            "plaintext": cls.TEXT,
            "schema": cls.JSON_SCHEMA,
            "jsonschema": cls.JSON_SCHEMA,
            "json_schema": cls.JSON_SCHEMA,
            "regexp": cls.REGEX,
        }
        if normalized in aliases:
            return aliases[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unknown output format: '{value}'. Valid formats: {[m.value for m in cls]}"
        )


@dataclass(frozen=True)
class OutputFormatSpec:
    """Metadata for a structured output format."""

    display_name: str
    requires_grammar_engine: bool
    description: str


OUTPUT_FORMAT_SPECS: dict[OutputFormat, OutputFormatSpec] = {
    OutputFormat.TEXT: OutputFormatSpec(
        display_name="Plain Text",
        requires_grammar_engine=False,
        description="Unconstrained text generation",
    ),
    OutputFormat.JSON: OutputFormatSpec(
        display_name="JSON",
        requires_grammar_engine=True,
        description="Valid JSON output (any structure)",
    ),
    OutputFormat.JSON_SCHEMA: OutputFormatSpec(
        display_name="JSON Schema",
        requires_grammar_engine=True,
        description="JSON conforming to a user-provided schema",
    ),
    OutputFormat.REGEX: OutputFormatSpec(
        display_name="Regex",
        requires_grammar_engine=True,
        description="Output matching a regular expression pattern",
    ),
}
