"""
Tests for Model Family Detection and Target Module Mapping

Tests ModelFamily enum, ModelFamilySpec, MODEL_FAMILY_SPECS registry,
detect_model_family(), get_target_modules(), and get_model_family_spec().
"""

import torch.nn as nn

from torchbridge.adapters.model_families import (
    MODEL_FAMILY_SPECS,
    ModelFamily,
    ModelFamilySpec,
    detect_model_family,
    get_model_family_spec,
    get_target_modules,
)


class TestModelFamily:
    """Tests for ModelFamily enum."""

    def test_all_families_present(self):
        """All expected families should be defined."""
        expected = {
            "llama", "qwen", "mistral", "phi", "gemma",
            "falcon", "gpt_neox", "bloom", "unknown",
        }
        actual = {f.value for f in ModelFamily}
        assert actual == expected

    def test_unknown_is_last_resort(self):
        """UNKNOWN should exist as a fallback value."""
        assert ModelFamily.UNKNOWN.value == "unknown"


class TestModelFamilySpec:
    """Tests for ModelFamilySpec dataclass."""

    def test_spec_is_frozen(self):
        """ModelFamilySpec should be immutable."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.LLAMA]
        import pytest

        with pytest.raises(AttributeError):
            spec.family = ModelFamily.QWEN

    def test_all_specs_have_required_fields(self):
        """Every spec should have non-empty target_modules and config_type_hints."""
        for family, spec in MODEL_FAMILY_SPECS.items():
            assert spec.family == family
            assert len(spec.target_modules) > 0, f"{family}: empty target_modules"
            assert len(spec.all_linear_names) > 0, f"{family}: empty all_linear_names"
            assert len(spec.config_type_hints) > 0, f"{family}: empty config_type_hints"
            assert len(spec.module_name_hints) > 0, f"{family}: empty module_name_hints"

    def test_unknown_not_in_specs(self):
        """UNKNOWN should NOT have a spec entry."""
        assert ModelFamily.UNKNOWN not in MODEL_FAMILY_SPECS


class TestModelFamilySpecs:
    """Tests for the MODEL_FAMILY_SPECS registry."""

    def test_llama_spec(self):
        """LLaMA should target q_proj and v_proj."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.LLAMA]
        assert spec.target_modules == ["q_proj", "v_proj"]
        assert spec.has_fused_qkv is False
        assert "llama" in spec.config_type_hints

    def test_qwen_spec(self):
        """Qwen should target q_proj, k_proj, v_proj."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.QWEN]
        assert "q_proj" in spec.target_modules
        assert "k_proj" in spec.target_modules
        assert "v_proj" in spec.target_modules
        assert "qwen2" in spec.config_type_hints
        assert "qwen3" in spec.config_type_hints

    def test_falcon_has_fused_qkv(self):
        """Falcon should have fused QKV."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.FALCON]
        assert spec.has_fused_qkv is True
        assert "query_key_value" in spec.target_modules

    def test_phi_has_fc_layers(self):
        """Phi should include fc1/fc2 in all_linear_names."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.PHI]
        assert "fc1" in spec.all_linear_names
        assert "fc2" in spec.all_linear_names

    def test_gemma_includes_gemma3(self):
        """Gemma spec should include gemma3 hint."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.GEMMA]
        assert "gemma3" in spec.config_type_hints

    def test_gpt_neox_fused(self):
        """GPT-NeoX should have fused QKV."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.GPT_NEOX]
        assert spec.has_fused_qkv is True

    def test_bloom_fused(self):
        """BLOOM should have fused QKV."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.BLOOM]
        assert spec.has_fused_qkv is True

    def test_mistral_includes_mixtral(self):
        """Mistral spec should cover Mixtral."""
        spec = MODEL_FAMILY_SPECS[ModelFamily.MISTRAL]
        assert "mixtral" in spec.config_type_hints


class _FakeConfig:
    """Fake HuggingFace config for testing."""

    def __init__(self, model_type: str):
        self.model_type = model_type


class _FakeLlamaModel(nn.Module):
    """Fake LLaMA-like model with characteristic modules."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("llama")
        self.q_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.gate_proj = nn.Linear(64, 64)
        self.up_proj = nn.Linear(64, 64)


class _FakeQwenModel(nn.Module):
    """Fake Qwen model."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("qwen2")
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)


class _FakeFalconModel(nn.Module):
    """Fake Falcon model with fused QKV."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("falcon")
        self.query_key_value = nn.Linear(64, 192)
        self.dense = nn.Linear(64, 64)


class _FakeUnknownModel(nn.Module):
    """Model with no config and no recognizable modules."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 64)
        self.linear2 = nn.Linear(64, 64)


class _FakeHeuristicModel(nn.Module):
    """Model with no config but has falcon-like module names."""

    def __init__(self):
        super().__init__()
        self.query_key_value = nn.Linear(64, 192)
        self.dense_h_to_4h = nn.Linear(64, 256)


class TestDetectModelFamily:
    """Tests for detect_model_family()."""

    def test_detect_llama_from_config(self):
        """Should detect LLaMA from config.model_type."""
        model = _FakeLlamaModel()
        assert detect_model_family(model) == ModelFamily.LLAMA

    def test_detect_qwen_from_config(self):
        """Should detect Qwen from config.model_type."""
        model = _FakeQwenModel()
        assert detect_model_family(model) == ModelFamily.QWEN

    def test_detect_falcon_from_config(self):
        """Should detect Falcon from config.model_type."""
        model = _FakeFalconModel()
        assert detect_model_family(model) == ModelFamily.FALCON

    def test_ambiguous_heuristics_return_unknown(self):
        """Ambiguous heuristics (multiple family match) should return UNKNOWN."""
        model = _FakeHeuristicModel()
        family = detect_model_family(model)
        # Falcon/GPT-NeoX/BLOOM all match query_key_value + dense_h_to_4h
        # Multiple matches → UNKNOWN (safety over false positive)
        assert family == ModelFamily.UNKNOWN

    def test_unknown_when_no_match(self):
        """Should return UNKNOWN when nothing matches."""
        model = _FakeUnknownModel()
        assert detect_model_family(model) == ModelFamily.UNKNOWN

    def test_case_insensitive_config_type(self):
        """Config type detection should be case-insensitive."""
        model = nn.Linear(10, 10)
        model.config = _FakeConfig("LLAMA")
        assert detect_model_family(model) == ModelFamily.LLAMA


class TestGetTargetModules:
    """Tests for get_target_modules()."""

    def test_known_family_returns_spec_targets(self):
        """Should return spec targets for known families."""
        model = _FakeLlamaModel()
        targets = get_target_modules(model)
        assert targets == ["q_proj", "v_proj"]

    def test_qwen_family_targets(self):
        """Qwen should return q_proj, k_proj, v_proj."""
        model = _FakeQwenModel()
        targets = get_target_modules(model)
        assert "q_proj" in targets
        assert "k_proj" in targets
        assert "v_proj" in targets

    def test_explicit_family_override(self):
        """Passing explicit family should override detection."""
        model = _FakeLlamaModel()
        targets = get_target_modules(model, family=ModelFamily.FALCON)
        assert targets == ["query_key_value"]

    def test_unknown_scans_for_projections(self):
        """Unknown family should scan for common projection names."""

        class _ProjModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.k_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)

        model = _ProjModel()
        targets = get_target_modules(model, family=ModelFamily.UNKNOWN)
        assert "q_proj" in targets
        assert "k_proj" in targets
        assert "v_proj" in targets

    def test_unknown_fallback_all_linear(self):
        """Unknown family with no common names should return all suffixes."""
        model = _FakeUnknownModel()
        targets = get_target_modules(model, family=ModelFamily.UNKNOWN)
        assert "linear1" in targets
        assert "linear2" in targets


class TestGetModelFamilySpec:
    """Tests for get_model_family_spec()."""

    def test_known_family_returns_spec(self):
        """Should return spec for known families."""
        spec = get_model_family_spec(ModelFamily.LLAMA)
        assert isinstance(spec, ModelFamilySpec)
        assert spec.family == ModelFamily.LLAMA

    def test_unknown_returns_none(self):
        """Should return None for UNKNOWN."""
        assert get_model_family_spec(ModelFamily.UNKNOWN) is None
