"""
Speculative Decoding Method Definitions

Enum of speculative decoding algorithms with metadata specs, mirroring the
pattern from attention/dispatch/kernel_types.py and precision/quantization/formats.py.
"""

from dataclasses import dataclass
from enum import Enum


class SpeculativeMethod(Enum):
    """Speculative decoding methods available for dispatch."""

    NONE = "none"
    DRAFT_MODEL = "draft_model"
    EAGLE = "eagle"
    LAYER_SKIP = "layer_skip"
    MEDUSA = "medusa"
    PROMPT_LOOKUP = "prompt_lookup"

    @classmethod
    def from_string(cls, value: str) -> "SpeculativeMethod":
        """Convert string to SpeculativeMethod, case-insensitive."""
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "draft": cls.DRAFT_MODEL,
            "skip": cls.LAYER_SKIP,
            "lookup": cls.PROMPT_LOOKUP,
            "prompt": cls.PROMPT_LOOKUP,
            "auto": cls.NONE,
        }
        if normalized in aliases:
            return aliases[normalized]
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unknown speculative method: '{value}'. "
            f"Valid methods: {[m.value for m in cls]}"
        )


@dataclass(frozen=True)
class SpeculativeMethodSpec:
    """Metadata for a speculative decoding method."""

    display_name: str
    requires_draft_model: bool
    requires_hardware_support: bool
    min_batch_size_benefit: int
    description: str
    is_generate_compatible: bool  # True = works via model.generate() kwargs
    requires_custom_arch: bool = False  # True = needs custom model architecture


SPECULATIVE_METHOD_SPECS: dict[SpeculativeMethod, SpeculativeMethodSpec] = {
    SpeculativeMethod.NONE: SpeculativeMethodSpec(
        display_name="None",
        requires_draft_model=False,
        requires_hardware_support=False,
        min_batch_size_benefit=0,
        description="No speculative decoding",
        is_generate_compatible=True,
    ),
    SpeculativeMethod.DRAFT_MODEL: SpeculativeMethodSpec(
        display_name="Draft Model",
        requires_draft_model=True,
        requires_hardware_support=False,
        min_batch_size_benefit=1,
        description="Standard draft-verify with a smaller assistant model",
        is_generate_compatible=True,
    ),
    SpeculativeMethod.EAGLE: SpeculativeMethodSpec(
        display_name="EAGLE",
        requires_draft_model=True,
        requires_hardware_support=True,
        min_batch_size_benefit=1,
        description=(
            "EAGLE speculative decoding (custom trained draft head). "
            "Not supported via model.generate() — requires a separately "
            "trained EAGLE checkpoint and custom inference loop."
        ),
        is_generate_compatible=False,
        requires_custom_arch=True,
    ),
    SpeculativeMethod.LAYER_SKIP: SpeculativeMethodSpec(
        display_name="Layer Skip",
        requires_draft_model=False,
        requires_hardware_support=False,
        min_batch_size_benefit=1,
        description=(
            "Self-speculative decoding by skipping later transformer layers. "
            "Not supported via model.generate() — requires a model with "
            "early-exit support and a custom inference loop."
        ),
        is_generate_compatible=False,
        requires_custom_arch=True,
    ),
    SpeculativeMethod.MEDUSA: SpeculativeMethodSpec(
        display_name="Medusa",
        requires_draft_model=True,
        requires_hardware_support=True,
        min_batch_size_benefit=1,
        description=(
            "Multi-head speculative decoding with tree attention verification. "
            "Not supported via model.generate() — requires a separately "
            "trained Medusa head and custom inference loop."
        ),
        is_generate_compatible=False,
        requires_custom_arch=True,
    ),
    SpeculativeMethod.PROMPT_LOOKUP: SpeculativeMethodSpec(
        display_name="Prompt Lookup",
        requires_draft_model=False,
        requires_hardware_support=False,
        min_batch_size_benefit=1,
        description=(
            "N-gram matching from prompt as speculative candidates. "
            "Works via prompt_lookup_num_tokens in model.generate()."
        ),
        is_generate_compatible=True,
    ),
}

