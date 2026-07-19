# SPDX-License-Identifier: Apache-2.0
"""
Speculative Decoding Compatibility

Backend-aware speculative method selection: which method (DRAFT_MODEL,
PROMPT_LOOKUP, LAYER_SKIP, etc.) is supported on the given hardware backend.

For speculative decoding execution use vLLM, SGLang, or HuggingFace
generate() with assistant_model directly.
"""

from .compatibility import SpeculationCompatibilityMatrix
from .methods import (
    SPECULATIVE_METHOD_SPECS,
    SpeculativeMethod,
    SpeculativeMethodSpec,
)

__all__ = [
    "SpeculativeMethod",
    "SpeculativeMethodSpec",
    "SPECULATIVE_METHOD_SPECS",
    "SpeculationCompatibilityMatrix",
]
