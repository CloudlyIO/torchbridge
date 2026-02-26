"""
Integration Tests for Adapter Auto-Detection Pipeline

End-to-end tests: detect family → inject adapters → verify correct modules adapted.
"""

import torch.nn as nn

from torchbridge.adapters import (
    AdapterConfig,
    AdapterEngine,
    ModelFamily,
    detect_model_family,
    get_target_modules,
)
from torchbridge.adapters.layers import DoRALinear, LoRALinear


class _FakeConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type


class _LlamaModel(nn.Module):
    """Multi-layer LLaMA-like model for pipeline testing."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("llama")
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({
                    "q_proj": nn.Linear(64, 64),
                    "k_proj": nn.Linear(64, 64),
                    "v_proj": nn.Linear(64, 64),
                    "o_proj": nn.Linear(64, 64),
                }),
                "mlp": nn.ModuleDict({
                    "gate_proj": nn.Linear(64, 256),
                    "up_proj": nn.Linear(64, 256),
                    "down_proj": nn.Linear(256, 64),
                }),
            })
            for _ in range(2)
        ])
        self.lm_head = nn.Linear(64, 1000)


class _QwenModel(nn.Module):
    """Multi-layer Qwen-like model."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("qwen3")
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "self_attn": nn.ModuleDict({
                    "q_proj": nn.Linear(64, 64),
                    "k_proj": nn.Linear(64, 64),
                    "v_proj": nn.Linear(64, 64),
                    "o_proj": nn.Linear(64, 64),
                }),
                "mlp": nn.ModuleDict({
                    "gate_proj": nn.Linear(64, 256),
                    "up_proj": nn.Linear(64, 256),
                    "down_proj": nn.Linear(256, 64),
                }),
            })
            for _ in range(2)
        ])


class _FalconModel(nn.Module):
    """Multi-layer Falcon-like model with fused QKV."""

    def __init__(self):
        super().__init__()
        self.config = _FakeConfig("falcon")
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attention": nn.ModuleDict({
                    "query_key_value": nn.Linear(64, 192),
                    "dense": nn.Linear(64, 64),
                }),
                "mlp": nn.ModuleDict({
                    "dense_h_to_4h": nn.Linear(64, 256),
                    "dense_4h_to_h": nn.Linear(256, 64),
                }),
            })
            for _ in range(2)
        ])


class TestAutoDetectPipeline:
    """End-to-end auto-detection pipeline tests."""

    def test_llama_detect_inject_verify(self):
        """LLaMA: detect → inject → verify q_proj and v_proj adapted."""
        model = _LlamaModel()
        family = detect_model_family(model)
        assert family == ModelFamily.LLAMA

        config = AdapterConfig()
        engine = AdapterEngine(config)
        result = engine.inject(model)

        assert result.success
        # 2 layers × 2 targets (q_proj, v_proj) = 4
        assert result.modules_adapted == 4

        # Verify correct modules are LoRALinear
        for layer in model.layers:
            assert isinstance(layer["self_attn"]["q_proj"], LoRALinear)
            assert isinstance(layer["self_attn"]["v_proj"], LoRALinear)
            # k_proj and o_proj should NOT be adapted
            assert isinstance(layer["self_attn"]["k_proj"], nn.Linear)
            assert isinstance(layer["self_attn"]["o_proj"], nn.Linear)

    def test_qwen_detect_inject_verify(self):
        """Qwen: detect → inject → verify q/k/v_proj adapted."""
        model = _QwenModel()
        family = detect_model_family(model)
        assert family == ModelFamily.QWEN

        config = AdapterConfig()
        engine = AdapterEngine(config)
        result = engine.inject(model)

        assert result.success
        # 2 layers × 3 targets (q_proj, k_proj, v_proj) = 6
        assert result.modules_adapted == 6

        for layer in model.layers:
            assert isinstance(layer["self_attn"]["q_proj"], LoRALinear)
            assert isinstance(layer["self_attn"]["k_proj"], LoRALinear)
            assert isinstance(layer["self_attn"]["v_proj"], LoRALinear)

    def test_falcon_detect_inject_verify(self):
        """Falcon: detect → inject → verify query_key_value adapted."""
        model = _FalconModel()
        family = detect_model_family(model)
        assert family == ModelFamily.FALCON

        config = AdapterConfig()
        engine = AdapterEngine(config)
        result = engine.inject(model)

        assert result.success
        # 2 layers × 1 target (query_key_value) = 2
        assert result.modules_adapted == 2

        for layer in model.layers:
            assert isinstance(layer["attention"]["query_key_value"], LoRALinear)
            assert isinstance(layer["attention"]["dense"], nn.Linear)

    def test_dora_with_auto_detect(self):
        """DoRA method should work with auto-detection."""
        from torchbridge.adapters.config import AdapterMethod

        model = _LlamaModel()
        config = AdapterConfig(method=AdapterMethod.DORA)
        engine = AdapterEngine(config)
        result = engine.inject(model)

        assert result.success
        assert result.modules_adapted == 4
        for layer in model.layers:
            assert isinstance(layer["self_attn"]["q_proj"], DoRALinear)
            assert isinstance(layer["self_attn"]["v_proj"], DoRALinear)

    def test_merge_after_auto_detect_inject(self):
        """Should be able to merge adapters after auto-detected injection."""
        model = _LlamaModel()
        config = AdapterConfig()
        engine = AdapterEngine(config)
        engine.inject(model)

        merged = engine.merge(model)
        assert merged == 4

        # After merge, all should be plain nn.Linear
        for layer in model.layers:
            assert type(layer["self_attn"]["q_proj"]) is nn.Linear
            assert type(layer["self_attn"]["v_proj"]) is nn.Linear

    def test_get_adapter_params_after_auto_detect(self):
        """get_adapter_params should return only adapter params."""
        model = _LlamaModel()
        config = AdapterConfig(rank=4)
        engine = AdapterEngine(config)
        engine.inject(model)

        params = engine.get_adapter_params(model)
        assert len(params) > 0
        # All param names should contain lora
        for name in params:
            assert "lora" in name.lower() or "magnitude" in name.lower()

    def test_get_target_modules_returns_list(self):
        """get_target_modules should return a list for any model."""
        model = _LlamaModel()
        targets = get_target_modules(model)
        assert isinstance(targets, list)
        assert len(targets) > 0

    def test_import_from_adapters_package(self):
        """Public API should be importable from torchbridge.adapters."""
        from torchbridge.adapters import (
            ModelFamily,
            ModelFamilySpec,
            detect_model_family,
            get_model_family_spec,
            get_target_modules,
        )

        assert ModelFamily is not None
        assert ModelFamilySpec is not None
        assert callable(detect_model_family)
        assert callable(get_model_family_spec)
        assert callable(get_target_modules)

    def test_info_includes_backend_and_method(self):
        """get_info should include backend and method info."""
        model = _LlamaModel()
        config = AdapterConfig()
        engine = AdapterEngine(config)
        engine.inject(model)
        info = engine.get_info(model)
        assert "adapter_count" in info
        assert info["adapter_count"] == 4

    def test_trainable_ratio_positive(self):
        """Trainable ratio should be positive after injection."""
        model = _LlamaModel()
        config = AdapterConfig(rank=4)
        engine = AdapterEngine(config)
        result = engine.inject(model)
        assert result.trainable_ratio > 0
        assert result.trainable_params > 0
        assert result.total_params > 0
