"""Tests for adapter engine — injection, merge, and param lifecycle."""

import torch
import torch.nn as nn

from torchbridge.adapters.config import AdapterConfig, AdapterMethod
from torchbridge.adapters.engine import AdapterEngine, AdapterResult, _model_size_mb
from torchbridge.adapters.layers import DoRALinear, LoRALinear
from torchbridge.core.config import HardwareBackend

# ── Test model ───────────────────────────────────────────────────────────────


class TinyTransformer(nn.Module):
    """Minimal model with named projections for testing adapter injection."""

    def __init__(self, dim=32):
        super().__init__()
        self.embed = nn.Embedding(100, dim)
        self.attn = nn.ModuleDict(
            {
                "q_proj": nn.Linear(dim, dim),
                "k_proj": nn.Linear(dim, dim),
                "v_proj": nn.Linear(dim, dim),
                "o_proj": nn.Linear(dim, dim),
            }
        )
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.embed(x)
        q = self.attn["q_proj"](h)
        k = self.attn["k_proj"](h)
        v = self.attn["v_proj"](h)
        return self.attn["o_proj"](q + k + v) + self.mlp(h)


# ── AdapterResult Tests ─────────────────────────────────────────────────────


class TestAdapterResult:
    """Tests for AdapterResult dataclass."""

    def test_to_dict(self):
        result = AdapterResult(
            success=True,
            method_applied=AdapterMethod.LORA,
            method_requested=AdapterMethod.LORA,
            modules_adapted=2,
            trainable_params=1024,
            total_params=100000,
            trainable_ratio=0.01024,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["method_applied"] == "lora"
        assert d["modules_adapted"] == 2
        assert d["trainable_ratio"] == 0.01024

    def test_to_dict_with_quant_format(self):
        from torchbridge.precision.quantization.formats import QuantizationFormat

        result = AdapterResult(
            success=True,
            method_applied=AdapterMethod.QLORA,
            method_requested=AdapterMethod.QLORA,
            base_quantized=True,
            base_quant_format=QuantizationFormat.INT4_WEIGHT_ONLY,
        )
        d = result.to_dict()
        assert d["base_quant_format"] == "int4_weight_only"

    def test_to_dict_warnings(self):
        result = AdapterResult(
            success=True,
            method_applied=AdapterMethod.LORA,
            method_requested=AdapterMethod.QLORA,
            used_fallback=True,
            warnings=["Fell back from qlora to lora"],
        )
        d = result.to_dict()
        assert d["used_fallback"] is True
        assert len(d["warnings"]) == 1


# ── Helper Tests ─────────────────────────────────────────────────────────────


class TestModelSizeMb:
    """Tests for _model_size_mb helper."""

    def test_returns_positive(self):
        model = nn.Linear(64, 64)
        assert _model_size_mb(model) > 0

    def test_larger_model_larger_size(self):
        small = nn.Linear(32, 32)
        large = nn.Linear(256, 256)
        assert _model_size_mb(large) > _model_size_mb(small)


# ── Inject Tests ─────────────────────────────────────────────────────────────


class TestInject:
    """Tests for AdapterEngine.inject()."""

    def test_injects_lora_into_q_v(self):
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            rank=4,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)

        assert result.success
        assert result.modules_adapted == 2
        assert result.method_applied == AdapterMethod.LORA
        assert isinstance(model.attn["q_proj"], LoRALinear)
        assert isinstance(model.attn["v_proj"], LoRALinear)

    def test_injects_dora(self):
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.DORA,
            rank=4,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)

        assert result.success
        assert isinstance(model.attn["q_proj"], DoRALinear)

    def test_no_match_warns(self):
        model = TinyTransformer()
        config = AdapterConfig(target_modules=["nonexistent"])
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)

        assert not result.success
        assert result.modules_adapted == 0
        assert any("No modules matched" in w for w in result.warnings)

    def test_fallback_on_unsupported(self):
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.TRAINIUM)
        result = engine.inject(model)

        assert result.used_fallback
        assert result.method_applied == AdapterMethod.LORA
        assert result.method_requested == AdapterMethod.QLORA

    def test_trainable_ratio(self):
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            rank=4,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)

        assert 0 < result.trainable_ratio < 1
        assert result.trainable_params < result.total_params

    def test_base_params_frozen(self):
        model = TinyTransformer()
        config = AdapterConfig(target_modules=["q_proj"])
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        assert not model.attn["q_proj"].base_linear.weight.requires_grad

    def test_non_target_modules_unchanged(self):
        model = TinyTransformer()
        config = AdapterConfig(target_modules=["q_proj", "v_proj"])
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        assert isinstance(model.attn["k_proj"], nn.Linear)
        assert isinstance(model.mlp, nn.Linear)

    def test_memory_recorded(self):
        model = TinyTransformer()
        config = AdapterConfig(target_modules=["q_proj"])
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)

        assert result.memory_before_mb > 0
        assert result.memory_after_mb > 0


# ── Merge Tests ──────────────────────────────────────────────────────────────


class TestMerge:
    """Tests for AdapterEngine.merge()."""

    def test_merge_replaces_adapter_with_linear(self):
        model = TinyTransformer()
        config = AdapterConfig(target_modules=["q_proj", "v_proj"])
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        merged_count = engine.merge(model)
        assert merged_count == 2
        assert isinstance(model.attn["q_proj"], nn.Linear)
        assert isinstance(model.attn["v_proj"], nn.Linear)

    def test_merge_preserves_output(self):
        torch.manual_seed(42)
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.LORA, rank=4, target_modules=["q_proj"]
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        x = torch.randint(0, 100, (1, 5))
        model.eval()
        with torch.no_grad():
            before = model(x).clone()

        engine.merge(model)
        with torch.no_grad():
            after = model(x)

        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)

    def test_merge_dora_preserves_output(self):
        torch.manual_seed(42)
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.DORA, rank=4, target_modules=["q_proj"]
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        x = torch.randint(0, 100, (1, 5))
        model.eval()
        with torch.no_grad():
            before = model(x).clone()

        engine.merge(model)
        with torch.no_grad():
            after = model(x)

        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)

    def test_merge_returns_zero_when_no_adapters(self):
        model = TinyTransformer()
        config = AdapterConfig()
        engine = AdapterEngine(config, HardwareBackend.CPU)
        assert engine.merge(model) == 0


# ── Param Save/Load Tests ───────────────────────────────────────────────────


class TestAdapterParams:
    """Tests for adapter parameter extraction and loading."""

    def test_get_adapter_params_only_adapter(self):
        model = TinyTransformer()
        config = AdapterConfig(target_modules=["q_proj"])
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        params = engine.get_adapter_params(model)
        # Only adapter params (lora_A, lora_B) — not embed, mlp, etc.
        for name in params:
            assert "lora_A" in name or "lora_B" in name or "magnitude" in name
        # Embedding should NOT be included even though it's trainable
        assert not any("embed" in name for name in params)
        assert not any("mlp" in name for name in params)

    def test_round_trip(self):
        torch.manual_seed(42)
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.LORA, rank=4, target_modules=["q_proj"]
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        # Save adapter params
        params = engine.get_adapter_params(model)

        # Create a fresh model and inject
        model2 = TinyTransformer()
        torch.manual_seed(42)  # same base weights
        engine2 = AdapterEngine(config, HardwareBackend.CPU)
        engine2.inject(model2)

        # Load saved params
        loaded = engine2.load_adapter_params(model2, params)
        assert loaded > 0

    def test_load_returns_count(self):
        model = TinyTransformer()
        config = AdapterConfig(target_modules=["q_proj"])
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        params = engine.get_adapter_params(model)
        loaded = engine.load_adapter_params(model, params)
        assert loaded == len(params)


# ── Info Tests ───────────────────────────────────────────────────────────────


class TestGetInfo:
    """Tests for AdapterEngine.get_info()."""

    def test_info_before_injection(self):
        model = TinyTransformer()
        config = AdapterConfig()
        engine = AdapterEngine(config, HardwareBackend.CPU)
        info = engine.get_info(model)
        assert info["adapter_count"] == 0
        assert info["adapter_types"] == []

    def test_info_after_lora_injection(self):
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.LORA, target_modules=["q_proj", "v_proj"]
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)
        info = engine.get_info(model)

        assert info["adapter_count"] == 2
        assert info["adapter_types"] == ["lora"]
        assert info["trainable_params"] > 0
        assert info["resolved_method"] == "lora"
        assert info["backend"] == "cpu"

    def test_info_after_dora_injection(self):
        model = TinyTransformer()
        config = AdapterConfig(
            method=AdapterMethod.DORA, target_modules=["q_proj"]
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)
        info = engine.get_info(model)

        assert info["adapter_count"] == 1
        assert info["adapter_types"] == ["dora"]

    def test_info_has_config(self):
        model = TinyTransformer()
        config = AdapterConfig(rank=8, alpha=16.0)
        engine = AdapterEngine(config, HardwareBackend.CPU)
        info = engine.get_info(model)

        assert info["config"]["rank"] == 8
        assert info["config"]["alpha"] == 16.0
