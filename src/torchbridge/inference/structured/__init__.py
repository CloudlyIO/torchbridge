"""
Structured Output Formats

Format enum and metadata for grammar-guided generation.
For constrained generation use xgrammar, outlines, or vLLM guided decoding.
"""

from .output_format import (
    OUTPUT_FORMAT_SPECS,
    OutputFormat,
    OutputFormatSpec,
    get_format_spec,
)

__all__ = [
    "OutputFormat",
    "OutputFormatSpec",
    "OUTPUT_FORMAT_SPECS",
    "get_format_spec",
]
