"""Integration tests for adapter training pipeline.

End-to-end flows: inject → train → save → load → merge → deploy.
Cross-backend compatibility, CLI, and import tests.
"""

import json
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from torchbridge.adapters import (
    AdapterConfig,
    AdapterEngine,
    AdapterMethod,
    MultiAdapterManager,
)
from torchbridge.core.config import HardwareBackend

# ── Test model ───────────────────────────────────────────────────────────────


class MiniLM(nn.Module):
    """Minimal transformer-like model for integration tests."""

    def __init__(self, dim=32, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(256, dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(dim, dim),
                        "k_proj": nn.Linear(dim, dim),
                        "v_proj": nn.Linear(dim, dim),
                        "o_proj": nn.Linear(dim, dim),
                        "gate_proj": nn.Linear(dim, dim * 4),
                        "up_proj": nn.Linear(dim, dim * 4),
                        "down_proj": nn.Linear(dim * 4, dim),
                    }
                )
            )
        self.head = nn.Linear(dim, 256)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            q = layer["q_proj"](h)
            k = layer["k_proj"](h)
            v = layer["v_proj"](h)
            h = h + layer["o_proj"](q + k + v)
            g = torch.sigmoid(layer["gate_proj"](h))
            h = h + layer["down_proj"](g * layer["up_proj"](h))
        return self.head(h)


# ── End-to-End Pipeline ─────────────────────────────────────────────────────


class TestEndToEndPipeline:
    """Full adapter lifecycle: inject → train → save → load → merge."""

    def test_lora_pipeline(self):
        torch.manual_seed(42)
        model = MiniLM()
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)

        # Inject
        result = engine.inject(model)
        assert result.success
        assert result.modules_adapted == 4  # 2 layers * 2 modules

        # Simulate training step
        x = torch.randint(0, 256, (2, 8))
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Save adapter params
        params = engine.get_adapter_params(model)
        assert len(params) > 0

        # Info
        info = engine.get_info(model)
        assert info["adapter_count"] == 4
        assert info["adapter_types"] == ["lora"]

        # Merge for deployment
        model.eval()
        with torch.no_grad():
            pre_merge = model(x).clone()

        merged = engine.merge(model)
        assert merged == 4

        with torch.no_grad():
            post_merge = model(x)

        torch.testing.assert_close(post_merge, pre_merge, atol=1e-4, rtol=1e-4)

    def test_dora_pipeline(self):
        torch.manual_seed(42)
        model = MiniLM()
        config = AdapterConfig(
            method=AdapterMethod.DORA,
            rank=4,
            alpha=8.0,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)

        result = engine.inject(model)
        assert result.success
        assert result.method_applied == AdapterMethod.DORA

        # Train step
        x = torch.randint(0, 256, (2, 8))
        loss = model(x).sum()
        loss.backward()

        # Merge preserves output
        model.eval()
        with torch.no_grad():
            pre_merge = model(x).clone()
        engine.merge(model)
        with torch.no_grad():
            post_merge = model(x)
        torch.testing.assert_close(post_merge, pre_merge, atol=1e-4, rtol=1e-4)


# ── Cross-Backend ────────────────────────────────────────────────────────────


class TestCrossBackend:
    """Test adapter injection across all backends (on CPU)."""

    @pytest.mark.parametrize(
        "backend",
        [
            HardwareBackend.CPU,
            HardwareBackend.CUDA,
            HardwareBackend.AMD,
            HardwareBackend.TRAINIUM,
            HardwareBackend.TPU,
        ],
    )
    def test_inject_all_backends(self, backend):
        model = MiniLM(dim=16, n_layers=1)
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            rank=4,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, backend)
        result = engine.inject(model)
        assert result.success

    def test_qlora_fallback_on_unsupported(self):
        # TRAINIUM does not support QLORA; CPU now does (INT8).
        model = MiniLM(dim=16, n_layers=1)
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            target_modules=["q_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.TRAINIUM)
        result = engine.inject(model)
        assert result.used_fallback
        assert result.method_applied == AdapterMethod.LORA


# ── Multi-Adapter Serving ───────────────────────────────────────────────────


class TestMultiAdapterServing:
    """Integration test for multi-adapter serving."""

    def test_two_adapters_switch(self):
        torch.manual_seed(42)
        model = MiniLM(dim=16, n_layers=1)
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            rank=4,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        engine.inject(model)

        # Train adapter A
        x = torch.randint(0, 256, (1, 4))
        model(x).sum().backward()
        params_a = engine.get_adapter_params(model)

        # Reset and train adapter B differently
        for p in model.parameters():
            if p.requires_grad:
                p.data.normal_(0, 0.5)
        params_b = engine.get_adapter_params(model)

        # Serving manager
        mgr = MultiAdapterManager(model, max_loaded=4)
        mgr.load_adapter("adapter_a", params_a)
        mgr.load_adapter("adapter_b", params_b)

        # Switch between adapters
        mgr.activate("adapter_a")
        assert mgr.get_active() == "adapter_a"

        model.eval()
        with torch.no_grad():
            out_a = model(x).clone()

        mgr.activate("adapter_b")
        with torch.no_grad():
            out_b = model(x).clone()

        # Outputs should differ (different adapter weights)
        assert not torch.allclose(out_a, out_b)


# ── Wide Target Modules ─────────────────────────────────────────────────────


class TestWideTargetModules:
    """Test adapter injection with many target modules."""

    def test_all_attention_projections(self):
        model = MiniLM(dim=16, n_layers=2)
        config = AdapterConfig(
            rank=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)
        assert result.modules_adapted == 8  # 4 per layer * 2 layers

    def test_mlp_projections(self):
        model = MiniLM(dim=16, n_layers=1)
        config = AdapterConfig(
            rank=4,
            target_modules=["gate_proj", "up_proj", "down_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)
        assert result.modules_adapted == 3


# ── CLI Tests ────────────────────────────────────────────────────────────────


class TestAdapterCLI:
    """Tests for the adapter CLI subcommand."""

    def test_recommend_ci_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "torchbridge.cli", "adapter", "recommend",
             "--backend", "cuda", "--ci"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["backend"] == "cuda"
        assert data["optimal_method"] in ("qlora", "qdora", "lora", "dora")
        assert isinstance(data["fallback_chain"], list)

    def test_info_ci_json(self):
        result = subprocess.run(
            [sys.executable, "-m", "torchbridge.cli", "adapter", "info", "--ci"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "backends" in data
        assert "model_families" in data
        assert "cuda" in data["backends"]
        assert "trainium" in data["backends"]
        assert "cpu" in data["backends"]
        assert "llama" in data["model_families"]

    def test_recommend_human_output(self):
        result = subprocess.run(
            [sys.executable, "-m", "torchbridge.cli", "adapter", "recommend",
             "--backend", "amd"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Adapter Method Recommendation" in result.stdout

    def test_tb_adapter_entry_point(self):
        from torchbridge.cli.adapter import main as adapter_main

        # Test the standalone entry returns 0
        assert adapter_main(["recommend", "--backend", "cpu", "--ci"]) == 0


# ── Import Tests ─────────────────────────────────────────────────────────────


class TestImports:
    """Verify all public imports work."""

    def test_top_level_imports(self):
        import torchbridge.adapters as adapters

        assert hasattr(adapters, "AdapterCompatibilityMatrix")
        assert hasattr(adapters, "AdapterConfig")
        assert hasattr(adapters, "AdapterEngine")
        assert hasattr(adapters, "AdapterMethod")
        assert hasattr(adapters, "AdapterResult")
        assert hasattr(adapters, "DoRALinear")
        assert hasattr(adapters, "LoRALinear")
        assert hasattr(adapters, "MultiAdapterManager")

    def test_submodule_imports(self):
        import torchbridge.adapters.compatibility as compat
        import torchbridge.adapters.config as config
        import torchbridge.adapters.engine as engine
        import torchbridge.adapters.layers as layers
        import torchbridge.adapters.serving as serving

        assert hasattr(compat, "AdapterCompatibilityMatrix")
        assert hasattr(config, "AdapterConfig")
        assert hasattr(config, "AdapterMethod")
        assert hasattr(config, "InitMethod")
        assert hasattr(engine, "AdapterEngine")
        assert hasattr(engine, "AdapterResult")
        assert hasattr(layers, "DoRALinear")
        assert hasattr(layers, "LoRALinear")
        assert hasattr(serving, "AdapterSlot")
        assert hasattr(serving, "MultiAdapterManager")


# ── Trainable Parameter Ratio ────────────────────────────────────────────────


class TestQLoRAIntegration:
    """Integration tests for QLoRA inject + forward on CPU (INT8)."""

    @pytest.fixture(autouse=True)
    def require_torchao(self):
        pytest.importorskip("torchao")

    def test_qlora_inject_and_forward(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            rank=4,
            alpha=8.0,
            target_modules=["0", "2"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)

        assert result.success
        assert result.modules_adapted == 2
        assert result.base_quantized is True

        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 16)

    def test_qlora_trainable_ratio_low(self):
        torch.manual_seed(42)
        model = MiniLM(dim=64, n_layers=2)
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            rank=4,
            target_modules=["q_proj", "v_proj"],
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)

        assert result.success
        # Adapter params should be a small fraction of total
        assert result.trainable_ratio < 0.15


class TestTrainableRatio:
    """Verify adapter efficiency metrics."""

    def test_low_rank_adds_few_params(self):
        model = MiniLM(dim=128, n_layers=4)
        total_before = sum(p.numel() for p in model.parameters())
        config = AdapterConfig(
            rank=4, target_modules=["q_proj", "v_proj"]
        )
        engine = AdapterEngine(config, HardwareBackend.CPU)
        result = engine.inject(model)

        # Adapter adds very few params relative to model size
        adapter_added = result.total_params - total_before
        adapter_fraction = adapter_added / total_before
        assert adapter_fraction < 0.05  # adapters are < 5% of base

    def test_higher_rank_higher_ratio(self):
        model1 = MiniLM(dim=32, n_layers=1)
        config1 = AdapterConfig(rank=4, target_modules=["q_proj"])
        engine1 = AdapterEngine(config1, HardwareBackend.CPU)
        r1 = engine1.inject(model1)

        model2 = MiniLM(dim=32, n_layers=1)
        config2 = AdapterConfig(rank=32, target_modules=["q_proj"])
        engine2 = AdapterEngine(config2, HardwareBackend.CPU)
        r2 = engine2.inject(model2)

        assert r2.trainable_ratio > r1.trainable_ratio
