"""
Tests for Speculative Decoding draft_model_name Validation

Tests that SpeculationEngine validates draft_model_name is non-empty
and non-whitespace before returning generation kwargs.
"""

import pytest

from torchbridge.core.config import HardwareBackend
from torchbridge.inference.speculative.engine import (
    SpeculationConfig,
    SpeculationEngine,
)
from torchbridge.inference.speculative.methods import SpeculativeMethod


def _make_engine(draft_model_name, method=SpeculativeMethod.DRAFT_MODEL):
    """Create a SpeculationEngine with _resolved_method forced to DRAFT_MODEL.

    On CPU, DRAFT_MODEL is not in the compatibility matrix so the constructor
    auto-falls back. We bypass that by setting _resolved_method directly
    to test the validation logic in get_generation_kwargs().
    """
    config = SpeculationConfig(
        method=method,
        draft_model_name=draft_model_name,
        num_speculative_tokens=5,
    )
    engine = SpeculationEngine(
        config=config,
        backend=HardwareBackend.CPU,
        architecture=None,
    )
    # Force resolved method to DRAFT_MODEL to test validation path
    engine._resolved_method = SpeculativeMethod.DRAFT_MODEL
    return engine


class TestDraftModelValidation:
    """Tests for draft_model_name validation in SpeculationEngine."""

    def test_missing_draft_model_name_raises(self):
        """DRAFT_MODEL without draft_model_name should raise ValueError."""
        engine = _make_engine(draft_model_name=None)
        with pytest.raises(ValueError, match="requires draft_model_name"):
            engine.get_generation_kwargs()

    def test_empty_draft_model_name_raises(self):
        """DRAFT_MODEL with empty string should raise ValueError."""
        engine = _make_engine(draft_model_name="")
        with pytest.raises(ValueError, match="requires draft_model_name"):
            engine.get_generation_kwargs()

    def test_whitespace_only_draft_model_name_raises(self):
        """DRAFT_MODEL with whitespace-only name should raise ValueError."""
        engine = _make_engine(draft_model_name="   ")
        with pytest.raises(ValueError, match="cannot be empty or whitespace"):
            engine.get_generation_kwargs()

    def test_valid_draft_model_name_returns_kwargs(self):
        """DRAFT_MODEL with valid name should return kwargs."""
        engine = _make_engine(draft_model_name="Qwen/Qwen3-0.6B")
        kwargs = engine.get_generation_kwargs()
        assert kwargs["assistant_model"] == "Qwen/Qwen3-0.6B"
        assert kwargs["num_assistant_tokens"] == 5

    def test_draft_model_name_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        engine = _make_engine(draft_model_name="  Qwen/Qwen3-0.6B  ")
        kwargs = engine.get_generation_kwargs()
        assert kwargs["assistant_model"] == "Qwen/Qwen3-0.6B"
