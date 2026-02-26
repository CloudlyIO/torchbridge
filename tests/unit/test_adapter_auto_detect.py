"""
Tests for Adapter Engine Auto-Detection Integration

Tests that AdapterEngine correctly auto-detects model family
and uses appropriate target modules during injection.
"""

import torch.nn as nn

from torchbridge.adapters.config import AdapterConfig
from torchbridge.adapters.engine import AdapterEngine
from torchbridge.adapters.layers import LoRALinear


class _FakeConfig:
    """Fake HuggingFace config."""

    def __init__(self, model_type: str):
        self.model_type = model_type


class _LlamaLikeModel(nn.Module):
    """Fake LLaMA-like model with attention projections."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("llama")
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)
        self.gate_proj = nn.Linear(64, 256)
        self.up_proj = nn.Linear(64, 256)
        self.down_proj = nn.Linear(256, 64)


class _QwenLikeModel(nn.Module):
    """Fake Qwen model — should auto-detect q_proj, k_proj, v_proj."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("qwen3")
        self.q_proj = nn.Linear(64, 64)
        self.k_proj = nn.Linear(64, 64)
        self.v_proj = nn.Linear(64, 64)
        self.o_proj = nn.Linear(64, 64)
        self.gate_proj = nn.Linear(64, 256)
        self.up_proj = nn.Linear(64, 256)


class _FalconLikeModel(nn.Module):
    """Fake Falcon model — fused QKV."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("falcon")
        self.query_key_value = nn.Linear(64, 192)
        self.dense = nn.Linear(64, 64)
        self.dense_h_to_4h = nn.Linear(64, 256)
        self.dense_4h_to_h = nn.Linear(256, 64)


class TestAdapterAutoDetect:
    """Tests for auto-detection in AdapterEngine.inject()."""

    def test_llama_auto_detect_targets(self):
        """LLaMA model should auto-detect q_proj, v_proj as targets."""
        config = AdapterConfig()  # defaults: auto_detect_targets=True
        engine = AdapterEngine(config)
        model = _LlamaLikeModel()
        result = engine.inject(model)
        assert result.success
        # q_proj and v_proj should be adapted
        assert result.modules_adapted == 2

    def test_qwen_auto_detect_targets(self):
        """Qwen model should auto-detect q_proj, k_proj, v_proj as targets."""
        config = AdapterConfig()
        engine = AdapterEngine(config)
        model = _QwenLikeModel()
        result = engine.inject(model)
        assert result.success
        # Qwen targets q_proj, k_proj, v_proj (3 modules)
        assert result.modules_adapted == 3

    def test_falcon_auto_detect_targets(self):
        """Falcon model should auto-detect query_key_value as target."""
        config = AdapterConfig()
        engine = AdapterEngine(config)
        model = _FalconLikeModel()
        result = engine.inject(model)
        assert result.success
        assert result.modules_adapted == 1
        assert isinstance(model.query_key_value, LoRALinear)

    def test_auto_detect_disabled(self):
        """When auto_detect_targets=False, should use config.target_modules."""
        config = AdapterConfig(auto_detect_targets=False)
        engine = AdapterEngine(config)
        model = _FalconLikeModel()
        result = engine.inject(model)
        # Default target_modules is ["q_proj", "v_proj"], Falcon has neither
        assert result.modules_adapted == 0

    def test_explicit_targets_respected_over_auto(self):
        """Explicit non-default target_modules should NOT be overridden."""
        config = AdapterConfig(target_modules=["dense"])
        engine = AdapterEngine(config)
        model = _FalconLikeModel()
        result = engine.inject(model)
        # User explicitly set ["dense"], so auto-detect should NOT override
        assert result.modules_adapted == 1
        assert isinstance(model.dense, LoRALinear)
        # query_key_value should NOT be adapted
        assert not isinstance(model.query_key_value, LoRALinear)

    def test_auto_detect_with_unknown_model(self):
        """Unknown model should scan for common projection names."""

        class _UnknownProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(64, 64)
                self.v_proj = nn.Linear(64, 64)
                self.other = nn.Linear(64, 64)

        config = AdapterConfig()
        engine = AdapterEngine(config)
        model = _UnknownProj()
        result = engine.inject(model)
        assert result.success
        # Should find q_proj and v_proj via scan
        assert result.modules_adapted == 2

    def test_effective_targets_stored(self):
        """Engine should store effective targets after inject."""
        config = AdapterConfig()
        engine = AdapterEngine(config)
        model = _QwenLikeModel()
        engine.inject(model)
        assert hasattr(engine, "_effective_targets")
        assert "k_proj" in engine._effective_targets

    def test_config_to_dict_includes_auto_detect(self):
        """AdapterConfig.to_dict should include auto_detect_targets."""
        config = AdapterConfig(auto_detect_targets=False)
        d = config.to_dict()
        assert "auto_detect_targets" in d
        assert d["auto_detect_targets"] is False

    def test_auto_detect_true_by_default(self):
        """auto_detect_targets should default to True."""
        config = AdapterConfig()
        assert config.auto_detect_targets is True

    def test_adapter_result_reports_correct_params(self):
        """AdapterResult should have correct param counts after auto-detect."""
        config = AdapterConfig(rank=8)
        engine = AdapterEngine(config)
        model = _LlamaLikeModel()
        result = engine.inject(model)
        assert result.trainable_params > 0
        assert result.total_params > result.trainable_params
        assert 0 < result.trainable_ratio < 1
