"""
Tests for SpeculationEngine validation

Verifies that:
- PROMPT_LOOKUP produces valid generate() kwargs
- DRAFT_MODEL validates draft_model_name (raises ValueError when missing)
- EAGLE/MEDUSA/LAYER_SKIP are not in the matrix and fall back gracefully
"""

import pytest

from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture
from torchbridge.inference.speculative.engine import (
    SpeculationConfig,
    SpeculationEngine,
)
from torchbridge.inference.speculative.methods import SpeculativeMethod


class TestSpeculationValidation:
    """Tests for SpeculationEngine get_generation_kwargs() validation."""

    def test_prompt_lookup_produces_valid_kwargs(self):
        """PROMPT_LOOKUP produces prompt_lookup_num_tokens — fully functional."""
        config = SpeculationConfig(
            method=SpeculativeMethod.PROMPT_LOOKUP,
            num_speculative_tokens=7,
        )
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        kwargs = engine.get_generation_kwargs()
        assert kwargs == {"prompt_lookup_num_tokens": 7}

    def test_draft_model_with_name_produces_valid_kwargs(self):
        """DRAFT_MODEL with draft_model_name produces assistant_model kwargs."""
        config = SpeculationConfig(
            method=SpeculativeMethod.DRAFT_MODEL,
            draft_model_name="Qwen/Qwen3-0.6B",
            num_speculative_tokens=3,
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        kwargs = engine.get_generation_kwargs()
        assert kwargs["assistant_model"] == "Qwen/Qwen3-0.6B"
        assert kwargs["num_assistant_tokens"] == 3

    def test_draft_model_without_name_raises(self):
        """DRAFT_MODEL without draft_model_name raises ValueError."""
        config = SpeculationConfig(
            method=SpeculativeMethod.DRAFT_MODEL,
            draft_model_name=None,
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        with pytest.raises(ValueError, match="draft_model_name"):
            engine.get_generation_kwargs()

    def test_eagle_not_in_matrix_falls_back(self):
        """EAGLE is not in the matrix — explicit request falls back to DRAFT_MODEL."""
        config = SpeculationConfig(
            method=SpeculativeMethod.EAGLE,
            draft_model_name="some-eagle-model",
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        # Falls back to DRAFT_MODEL since EAGLE is not in matrix
        assert engine.method == SpeculativeMethod.DRAFT_MODEL
        kwargs = engine.get_generation_kwargs()
        assert kwargs["assistant_model"] == "some-eagle-model"

    def test_medusa_not_in_matrix_falls_back(self):
        """MEDUSA is not in the matrix — explicit request falls back."""
        config = SpeculationConfig(
            method=SpeculativeMethod.MEDUSA,
            draft_model_name="some-medusa-model",
        )
        engine = SpeculationEngine(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
        )
        # Falls back to DRAFT_MODEL since MEDUSA is not in matrix
        assert engine.method == SpeculativeMethod.DRAFT_MODEL

    def test_layer_skip_not_in_matrix_falls_back(self):
        """LAYER_SKIP is not in the matrix — falls back to PROMPT_LOOKUP."""
        config = SpeculationConfig(method=SpeculativeMethod.LAYER_SKIP)
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        assert engine.method == SpeculativeMethod.PROMPT_LOOKUP
        kwargs = engine.get_generation_kwargs()
        assert "prompt_lookup_num_tokens" in kwargs

    def test_disabled_engine_always_empty(self):
        """Disabled engine returns {} regardless of method."""
        config = SpeculationConfig(
            method=SpeculativeMethod.PROMPT_LOOKUP,
            enabled=False,
        )
        engine = SpeculationEngine(config=config, backend=HardwareBackend.CPU)
        assert engine.get_generation_kwargs() == {}
