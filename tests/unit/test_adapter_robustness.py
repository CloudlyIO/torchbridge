"""
Tests for Adapter Robustness — Negative Scenarios

Tests edge cases found during the 8-dimension audit:
empty strings in target_modules, double injection, heuristic ambiguity,
empty-string validation, and improved warning messages.
"""

import pytest
import torch.nn as nn

from torchbridge.adapters.config import AdapterConfig
from torchbridge.adapters.engine import AdapterEngine
from torchbridge.adapters.model_families import (
    ModelFamily,
    detect_model_family,
)


class _FakeConfig:
    def __init__(self, model_type):
        self.model_type = model_type


class _LlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("llama")
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)


class _AmbiguousModel(nn.Module):
    """Model with gate_proj+up_proj (shared by LLaMA/Mistral/Gemma)."""

    def __init__(self):
        super().__init__()
        # No config → heuristic only
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.gate_proj = nn.Linear(64, 256)
        self.up_proj = nn.Linear(64, 256)


class _NoLinearModel(nn.Module):
    """Model with no nn.Linear modules."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.norm = nn.LayerNorm(64)


class TestEmptyStringValidation:
    """N1: Empty strings in target_modules should be rejected."""

    def test_empty_string_rejected(self):
        """target_modules with empty string should raise ValueError."""
        with pytest.raises(ValueError, match="empty strings"):
            AdapterConfig(target_modules=["q_proj", ""])

    def test_single_empty_string_rejected(self):
        """target_modules with only empty string should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            AdapterConfig(target_modules=[""])

    def test_valid_modules_accepted(self):
        """Normal target_modules should pass validation."""
        config = AdapterConfig(target_modules=["q_proj", "v_proj"])
        assert config.target_modules == ["q_proj", "v_proj"]


class TestHeuristicAmbiguity:
    """R1: Ambiguous heuristics should return UNKNOWN, not false positive."""

    def test_ambiguous_hints_return_unknown(self):
        """gate_proj+up_proj matches LLaMA, Mistral, Gemma → UNKNOWN."""
        model = _AmbiguousModel()
        family = detect_model_family(model)
        assert family == ModelFamily.UNKNOWN

    def test_config_overrides_ambiguity(self):
        """With config.model_type, ambiguous hints are irrelevant."""
        model = _AmbiguousModel()
        model.config = _FakeConfig("mistral")
        family = detect_model_family(model)
        assert family == ModelFamily.MISTRAL

    def test_unique_hint_still_matches(self):
        """Phi has unique hints (fc1, fc2) — should match."""

        class PhiLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 256)
                self.fc2 = nn.Linear(256, 64)

        model = PhiLike()
        family = detect_model_family(model)
        assert family == ModelFamily.PHI


class TestDoubleInjection:
    """R4: Double injection should warn, not silently no-op."""

    def test_double_injection_warns(self):
        """Second inject() should warn about existing adapters."""
        config = AdapterConfig(auto_detect_targets=False)
        engine = AdapterEngine(config)
        model = _LlamaModel()

        result1 = engine.inject(model)
        assert result1.modules_adapted == 2

        result2 = engine.inject(model)
        assert result2.modules_adapted == 0
        assert any("already has" in w for w in result2.warnings)

    def test_double_injection_not_success(self):
        """Second injection should report success=False."""
        config = AdapterConfig(auto_detect_targets=False)
        engine = AdapterEngine(config)
        model = _LlamaModel()

        engine.inject(model)
        result2 = engine.inject(model)
        assert result2.success is False


class TestNoLinearModules:
    """U1: Better warning when model has no Linear layers."""

    def test_no_linear_warns_clearly(self):
        """Should warn about no Linear modules, not just no matches."""
        config = AdapterConfig(auto_detect_targets=False)
        engine = AdapterEngine(config)
        model = _NoLinearModel()

        result = engine.inject(model)
        assert result.modules_adapted == 0
        assert any("no nn.linear" in w.lower() for w in result.warnings)


class TestSecurityNoTrustRemoteCode:
    """SEC1: CLI should not use trust_remote_code=True."""

    def test_no_trust_remote_code_in_detect(self):
        """_show_detect should not pass trust_remote_code=True."""
        import inspect

        from torchbridge.cli.adapter import _show_detect

        source = inspect.getsource(_show_detect)
        assert "trust_remote_code" not in source

    def test_no_trust_remote_code_in_inject(self):
        """_show_inject_dryrun should not pass trust_remote_code=True."""
        import inspect

        from torchbridge.cli.adapter import _show_inject_dryrun

        source = inspect.getsource(_show_inject_dryrun)
        assert "trust_remote_code" not in source
