"""
Model Family Detection and Target Module Mapping

Maps model architectures to their correct LoRA target module names,
enabling auto-detection so users don't need to manually specify
target_modules for common model families.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelFamily(Enum):
    """Known model architecture families."""

    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    PHI = "phi"
    GEMMA = "gemma"
    FALCON = "falcon"
    GPT_NEOX = "gpt_neox"
    BLOOM = "bloom"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ModelFamilySpec:
    """Specification for a model family's adapter target modules.

    Attributes:
        family: Model family identifier.
        target_modules: Default LoRA target modules (attention projections).
        all_linear_names: All attention + MLP linear layer names.
        has_fused_qkv: Whether Q/K/V projections are fused into one layer.
        config_type_hints: HuggingFace ``config.model_type`` values for detection.
        module_name_hints: Module name substrings for heuristic detection.
    """

    family: ModelFamily
    target_modules: list[str]
    all_linear_names: list[str]
    has_fused_qkv: bool
    config_type_hints: list[str]
    module_name_hints: list[str]


MODEL_FAMILY_SPECS: dict[ModelFamily, ModelFamilySpec] = {
    ModelFamily.LLAMA: ModelFamilySpec(
        family=ModelFamily.LLAMA,
        target_modules=["q_proj", "v_proj"],
        all_linear_names=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        has_fused_qkv=False,
        config_type_hints=["llama"],
        module_name_hints=["gate_proj", "up_proj"],
    ),
    ModelFamily.QWEN: ModelFamilySpec(
        family=ModelFamily.QWEN,
        target_modules=["q_proj", "k_proj", "v_proj"],
        all_linear_names=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        has_fused_qkv=False,
        config_type_hints=["qwen2", "qwen3", "qwen2_moe"],
        module_name_hints=["q_proj", "k_proj", "v_proj", "gate_proj"],
    ),
    ModelFamily.MISTRAL: ModelFamilySpec(
        family=ModelFamily.MISTRAL,
        target_modules=["q_proj", "v_proj"],
        all_linear_names=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        has_fused_qkv=False,
        config_type_hints=["mistral", "mixtral"],
        module_name_hints=["gate_proj", "up_proj"],
    ),
    ModelFamily.PHI: ModelFamilySpec(
        family=ModelFamily.PHI,
        target_modules=["q_proj", "v_proj"],
        all_linear_names=[
            "q_proj", "k_proj", "v_proj", "dense",
            "fc1", "fc2",
        ],
        has_fused_qkv=False,
        config_type_hints=["phi", "phi3", "phi4"],
        module_name_hints=["fc1", "fc2"],
    ),
    ModelFamily.GEMMA: ModelFamilySpec(
        family=ModelFamily.GEMMA,
        target_modules=["q_proj", "v_proj"],
        all_linear_names=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        has_fused_qkv=False,
        config_type_hints=["gemma", "gemma2", "gemma3"],
        module_name_hints=["gate_proj", "up_proj"],
    ),
    ModelFamily.FALCON: ModelFamilySpec(
        family=ModelFamily.FALCON,
        target_modules=["query_key_value"],
        all_linear_names=[
            "query_key_value", "dense",
            "dense_h_to_4h", "dense_4h_to_h",
        ],
        has_fused_qkv=True,
        config_type_hints=["falcon"],
        module_name_hints=["query_key_value", "dense_h_to_4h"],
    ),
    ModelFamily.GPT_NEOX: ModelFamilySpec(
        family=ModelFamily.GPT_NEOX,
        target_modules=["query_key_value"],
        all_linear_names=[
            "query_key_value", "dense",
            "dense_h_to_4h", "dense_4h_to_h",
        ],
        has_fused_qkv=True,
        config_type_hints=["gpt_neox"],
        module_name_hints=["query_key_value", "dense_h_to_4h"],
    ),
    ModelFamily.BLOOM: ModelFamilySpec(
        family=ModelFamily.BLOOM,
        target_modules=["query_key_value"],
        all_linear_names=[
            "query_key_value", "dense",
            "dense_h_to_4h", "dense_4h_to_h",
        ],
        has_fused_qkv=True,
        config_type_hints=["bloom"],
        module_name_hints=["query_key_value", "dense_h_to_4h"],
    ),
}


def get_model_family_spec(family: ModelFamily) -> ModelFamilySpec | None:
    """Return the spec for a given family, or None if unknown."""
    return MODEL_FAMILY_SPECS.get(family)


def detect_model_family(model: nn.Module) -> ModelFamily:
    """Detect model family from config or module name heuristics.

    Strategy:
    1. Check ``model.config.model_type`` (HuggingFace models).
    2. Fall back to scanning ``named_modules()`` for characteristic names.

    Args:
        model: A PyTorch model (optionally with a ``.config`` attribute).

    Returns:
        Detected ModelFamily, or ``ModelFamily.UNKNOWN``.
    """
    # Strategy 1: HuggingFace config.model_type
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str):
            model_type_lower = model_type.lower()
            for family, spec in MODEL_FAMILY_SPECS.items():
                if model_type_lower in spec.config_type_hints:
                    logger.debug(
                        "Detected model family %s from config.model_type=%s",
                        family.value, model_type,
                    )
                    return family

    # Strategy 2: Module name heuristics (only match if exactly one
    # family's hints are satisfied, to avoid false positives)
    module_names = {name.split(".")[-1] for name, _ in model.named_modules()}

    matches = [
        (family, spec)
        for family, spec in MODEL_FAMILY_SPECS.items()
        if all(hint in module_names for hint in spec.module_name_hints)
    ]

    if len(matches) == 1:
        family, _ = matches[0]
        logger.debug(
            "Detected model family %s from module name heuristics",
            family.value,
        )
        return family

    if len(matches) > 1:
        logger.debug(
            "Multiple model families matched heuristics: %s. "
            "Set config.model_type for accurate detection.",
            [f.value for f, _ in matches],
        )

    logger.debug("Could not detect model family, returning UNKNOWN")
    return ModelFamily.UNKNOWN


def get_target_modules(
    model: nn.Module,
    family: ModelFamily | None = None,
) -> list[str]:
    """Return optimal target modules for the given model.

    If *family* is ``None``, auto-detects from the model. If the family
    is ``UNKNOWN``, scans for all ``nn.Linear`` modules whose names
    contain common attention projection patterns.

    Args:
        model: The model to inspect.
        family: Override the auto-detected family. Pass ``None`` to auto-detect.

    Returns:
        List of module name suffixes suitable for ``AdapterConfig.target_modules``.
    """
    if family is None:
        family = detect_model_family(model)

    spec = MODEL_FAMILY_SPECS.get(family)
    if spec is not None:
        return list(spec.target_modules)

    # Unknown family — scan for common attention projection names
    linear_suffixes: set[str] = set()
    _COMMON_PROJ = {"q_proj", "k_proj", "v_proj", "o_proj", "query_key_value",
                    "query", "key", "value", "dense"}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in _COMMON_PROJ:
                linear_suffixes.add(suffix)

    if linear_suffixes:
        return sorted(linear_suffixes)

    # Last resort: all nn.Linear suffixes
    all_suffixes: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            all_suffixes.add(name.split(".")[-1])

    if all_suffixes:
        logger.warning(
            "Could not identify attention projections. "
            "Returning all Linear layer suffixes: %s",
            sorted(all_suffixes),
        )
        return sorted(all_suffixes)

    return ["q_proj", "v_proj"]  # absolute fallback
