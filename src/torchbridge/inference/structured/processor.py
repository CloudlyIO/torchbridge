"""
Structured Output Processor

Provides grammar-guided generation using xgrammar (soft dependency).
Falls back gracefully when xgrammar is not installed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .output_format import OUTPUT_FORMAT_SPECS, OutputFormat

logger = logging.getLogger(__name__)

# Soft import
try:
    import xgrammar  # noqa: F401

    XGRAMMAR_AVAILABLE = True
except ImportError:
    XGRAMMAR_AVAILABLE = False


class StructuredOutputProcessor:
    """Constrained generation processor using xgrammar.

    Produces logits processors that enforce structured output constraints
    during model.generate(). Falls back gracefully when xgrammar is not
    installed.

    Args:
        format: Output format constraint.
        schema: JSON schema dict (required for JSON_SCHEMA format).
        pattern: Regex pattern string (required for REGEX format).
    """

    def __init__(
        self,
        format: OutputFormat = OutputFormat.TEXT,
        schema: dict[str, Any] | None = None,
        pattern: str | None = None,
    ):
        self._format = format
        self._schema = schema
        self._pattern = pattern
        self._compiled_pattern: re.Pattern | None = None

        # Validate arguments
        if format == OutputFormat.JSON_SCHEMA and schema is None:
            raise ValueError("schema is required for JSON_SCHEMA format")
        if format == OutputFormat.REGEX and pattern is None:
            raise ValueError("pattern is required for REGEX format")

        # Pre-compile regex to catch syntax errors early
        if format == OutputFormat.REGEX and pattern is not None:
            self._compiled_pattern = re.compile(pattern)

    @property
    def format(self) -> OutputFormat:
        """Return the configured output format."""
        return self._format

    @staticmethod
    def is_available() -> bool:
        """Check whether the xgrammar backend is installed."""
        return XGRAMMAR_AVAILABLE

    def get_logits_processor(self, tokenizer: Any = None) -> list[Any]:
        """Return a list of logits processors for constrained generation.

        Args:
            tokenizer: HuggingFace tokenizer (needed for grammar compilation).

        Returns:
            List of callables compatible with model.generate(logits_processor=...).
            Empty list if xgrammar is unavailable or format is TEXT.
        """
        if self._format == OutputFormat.TEXT:
            return []

        if not XGRAMMAR_AVAILABLE:
            logger.warning(
                "xgrammar not installed; structured output constraints disabled. "
                "Install with: pip install xgrammar"
            )
            return []

        spec = OUTPUT_FORMAT_SPECS.get(self._format)
        if spec and not spec.requires_grammar_engine:
            return []

        # Build grammar-based processor via xgrammar
        try:
            return self._build_xgrammar_processor(tokenizer)
        except Exception:
            logger.warning(
                "Failed to build xgrammar processor; falling back to unconstrained",
                exc_info=True,
            )
            return []

    def _build_xgrammar_processor(self, tokenizer: Any) -> list[Any]:
        """Build xgrammar logits processor."""
        import xgrammar

        tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer)
        compiler = xgrammar.GrammarCompiler(tokenizer_info)

        if self._format == OutputFormat.JSON:
            grammar = compiler.compile_builtin_json_grammar()
        elif self._format == OutputFormat.JSON_SCHEMA:
            schema_str = json.dumps(self._schema)
            grammar = compiler.compile_json_schema(schema_str)
        elif self._format == OutputFormat.REGEX:
            grammar = compiler.compile_regex(self._pattern)
        else:
            return []

        matcher = xgrammar.GrammarMatcher(grammar)
        return [matcher]

    def validate_output(self, text: str) -> bool:
        """Validate whether generated text matches the format constraint.

        Args:
            text: Generated text to validate.

        Returns:
            True if text conforms to the format, False otherwise.
        """
        if self._format == OutputFormat.TEXT:
            return True

        if self._format in (OutputFormat.JSON, OutputFormat.JSON_SCHEMA):
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return False

            if self._format == OutputFormat.JSON_SCHEMA and self._schema:
                return self._validate_json_schema(parsed)
            return True

        if self._format == OutputFormat.REGEX:
            if self._compiled_pattern is None:
                return True
            return bool(self._compiled_pattern.fullmatch(text))

        return True

    def _validate_json_schema(self, data: Any) -> bool:
        """Basic JSON schema validation (type checking only).

        For full validation, users should use jsonschema library.
        """
        if not self._schema:
            return True

        schema_type = self._schema.get("type")
        if schema_type == "object" and not isinstance(data, dict):
            return False
        if schema_type == "array" and not isinstance(data, list):
            return False
        if schema_type == "string" and not isinstance(data, str):
            return False
        if schema_type == "number" and not isinstance(data, (int, float)):
            return False
        if schema_type == "integer" and (not isinstance(data, int) or isinstance(data, bool)):
            return False
        if schema_type == "boolean" and not isinstance(data, bool):
            return False

        # Check required fields for objects
        if schema_type == "object" and isinstance(data, dict):
            required = self._schema.get("required", [])
            for field_name in required:
                if field_name not in data:
                    return False

        return True

    def get_info(self) -> dict[str, Any]:
        """Return diagnostic info about the processor configuration."""
        spec = OUTPUT_FORMAT_SPECS.get(self._format)
        return {
            "format": self._format.value,
            "display_name": spec.display_name if spec else "Unknown",
            "requires_grammar_engine": (
                spec.requires_grammar_engine if spec else False
            ),
            "xgrammar_available": XGRAMMAR_AVAILABLE,
            "has_schema": self._schema is not None,
            "has_pattern": self._pattern is not None,
        }
