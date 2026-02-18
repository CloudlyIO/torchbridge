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


SPECULATIVE_METHOD_SPECS: dict[SpeculativeMethod, SpeculativeMethodSpec] = {
    SpeculativeMethod.NONE: SpeculativeMethodSpec(
        display_name="None",
        requires_draft_model=False,
        requires_hardware_support=False,
        min_batch_size_benefit=0,
        description="No speculative decoding",
    ),
    SpeculativeMethod.DRAFT_MODEL: SpeculativeMethodSpec(
        display_name="Draft Model",
        requires_draft_model=True,
        requires_hardware_support=False,
        min_batch_size_benefit=1,
        description="Standard draft-verify with a smaller assistant model",
    ),
    SpeculativeMethod.EAGLE: SpeculativeMethodSpec(
        display_name="EAGLE",
        requires_draft_model=True,
        requires_hardware_support=True,
        min_batch_size_benefit=1,
        description="EAGLE speculative decoding with custom CUDA kernels",
    ),
    SpeculativeMethod.LAYER_SKIP: SpeculativeMethodSpec(
        display_name="Layer Skip",
        requires_draft_model=False,
        requires_hardware_support=False,
        min_batch_size_benefit=1,
        description="Self-speculative decoding by skipping later layers",
    ),
    SpeculativeMethod.MEDUSA: SpeculativeMethodSpec(
        display_name="Medusa",
        requires_draft_model=True,
        requires_hardware_support=True,
        min_batch_size_benefit=1,
        description="Multi-head speculative decoding with tree attention",
    ),
    SpeculativeMethod.PROMPT_LOOKUP: SpeculativeMethodSpec(
        display_name="Prompt Lookup",
        requires_draft_model=False,
        requires_hardware_support=False,
        min_batch_size_benefit=1,
        description="N-gram matching from prompt for speculative candidates",
    ),
}


def get_method_spec(method: SpeculativeMethod) -> SpeculativeMethodSpec:
    """Get the metadata spec for a speculative method."""
    return SPECULATIVE_METHOD_SPECS[method]
