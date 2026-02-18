"""
Structured Output Subpackage

Grammar-guided generation with xgrammar integration for JSON, JSON Schema,
and regex-constrained output.
"""

from .output_format import (
    OUTPUT_FORMAT_SPECS,
    OutputFormat,
    OutputFormatSpec,
    get_format_spec,
)
from .processor import StructuredOutputProcessor

__all__ = [
    "OutputFormat",
    "OutputFormatSpec",
    "OUTPUT_FORMAT_SPECS",
    "get_format_spec",
    "StructuredOutputProcessor",
]
