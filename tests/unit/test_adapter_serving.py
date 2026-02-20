"""Tests for multi-adapter serving manager."""

import pytest
import torch
import torch.nn as nn

from torchbridge.adapters.config import AdapterConfig, AdapterMethod
from torchbridge.adapters.engine import AdapterEngine
from torchbridge.adapters.layers import LoRALinear
from torchbridge.adapters.serving import AdapterSlot, MultiAdapterManager
from torchbridge.core.config import HardwareBackend

# ── Fixtures ─────────────────────────────────────────────────────────────────


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)

    def forward(self, x):
        return self.q_proj(x) + self.v_proj(x)


def _make_adapted_model():
    """Create a model with LoRA adapters injected."""
    torch.manual_seed(42)
    model = SmallModel()
    config = AdapterConfig(
        method=AdapterMethod.LORA,
        rank=4,
        target_modules=["q_proj", "v_proj"],
    )
    engine = AdapterEngine(config, HardwareBackend.CPU)
    engine.inject(model)
    return model, engine


def _make_adapter_params(seed=0):
    """Create mock adapter params by training a model briefly."""
    torch.manual_seed(seed)
    model, engine = _make_adapted_model()
    # Simulate training by randomizing adapter weights
    for param in model.parameters():
        if param.requires_grad:
            param.data.normal_(0, 0.1)
    return engine.get_adapter_params(model)


# ── AdapterSlot Tests ────────────────────────────────────────────────────────


class TestAdapterSlot:
    """Tests for AdapterSlot dataclass."""

    def test_slot_creation(self):
        params = {"w": torch.randn(4, 4)}
        slot = AdapterSlot(name="a1", params=params, config=AdapterConfig())
        assert slot.name == "a1"
        assert slot.device == "cpu"
        assert slot.last_used > 0


# ── MultiAdapterManager Init ────────────────────────────────────────────────


class TestManagerInit:
    """Tests for MultiAdapterManager construction."""

    def test_init_default(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        assert mgr.loaded_count == 0
        assert mgr.get_active() is None

    def test_invalid_max_loaded(self):
        model, _ = _make_adapted_model()
        with pytest.raises(ValueError, match="max_loaded"):
            MultiAdapterManager(model, max_loaded=0)


# ── Load/Remove Lifecycle ───────────────────────────────────────────────────


class TestLoadRemove:
    """Tests for load_adapter and remove."""

    def test_load_increases_count(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model, max_loaded=4)
        params = _make_adapter_params(seed=1)
        mgr.load_adapter("a1", params)
        assert mgr.loaded_count == 1

    def test_load_multiple(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model, max_loaded=4)
        for i in range(3):
            mgr.load_adapter(f"a{i}", _make_adapter_params(seed=i))
        assert mgr.loaded_count == 3

    def test_remove(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model, max_loaded=4)
        mgr.load_adapter("a1", _make_adapter_params())
        mgr.remove("a1")
        assert mgr.loaded_count == 0

    def test_remove_nonexistent_raises(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        with pytest.raises(KeyError, match="not loaded"):
            mgr.remove("nonexistent")

    def test_remove_active_clears_active(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        mgr.load_adapter("a1", _make_adapter_params())
        mgr.activate("a1")
        mgr.remove("a1")
        assert mgr.get_active() is None

    def test_remove_active_deactivates_weights(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        mgr.load_adapter("a1", _make_adapter_params())
        mgr.activate("a1")
        mgr.remove("a1")

        # Adapter weights should be zeroed after removing active
        for module in model.modules():
            if isinstance(module, LoRALinear):
                assert torch.all(module.lora_A.weight == 0)
                assert torch.all(module.lora_B.weight == 0)


# ── Activate/Deactivate ─────────────────────────────────────────────────────


class TestActivateDeactivate:
    """Tests for activate and deactivate."""

    def test_activate_sets_active(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        mgr.load_adapter("a1", _make_adapter_params())
        mgr.activate("a1")
        assert mgr.get_active() == "a1"

    def test_activate_nonexistent_raises(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        with pytest.raises(KeyError, match="not loaded"):
            mgr.activate("nonexistent")

    def test_deactivate_clears_active(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        mgr.load_adapter("a1", _make_adapter_params())
        mgr.activate("a1")
        mgr.deactivate()
        assert mgr.get_active() is None

    def test_activate_switch(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        mgr.load_adapter("a1", _make_adapter_params(seed=1))
        mgr.load_adapter("a2", _make_adapter_params(seed=2))
        mgr.activate("a1")
        assert mgr.get_active() == "a1"
        mgr.activate("a2")
        assert mgr.get_active() == "a2"

    def test_deactivate_zeros_adapter_weights(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        mgr.load_adapter("a1", _make_adapter_params())
        mgr.activate("a1")
        mgr.deactivate()

        # After deactivation, LoRA weights should be zeroed
        for module in model.modules():
            if isinstance(module, LoRALinear):
                assert torch.all(module.lora_A.weight == 0)
                assert torch.all(module.lora_B.weight == 0)


# ── LRU Eviction ────────────────────────────────────────────────────────────


class TestLRUEviction:
    """Tests for LRU eviction policy."""

    def test_evicts_at_capacity(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model, max_loaded=2)

        mgr.load_adapter("a1", _make_adapter_params(seed=1))
        mgr.load_adapter("a2", _make_adapter_params(seed=2))
        assert mgr.loaded_count == 2

        # Loading a3 should evict the oldest (a1)
        mgr.load_adapter("a3", _make_adapter_params(seed=3))
        assert mgr.loaded_count == 2
        adapters = {a["name"] for a in mgr.list_adapters()}
        assert "a1" not in adapters
        assert "a3" in adapters

    def test_does_not_evict_active(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model, max_loaded=2)

        mgr.load_adapter("a1", _make_adapter_params(seed=1))
        mgr.load_adapter("a2", _make_adapter_params(seed=2))
        mgr.activate("a1")  # a1 is active

        # Loading a3 should evict a2 (not the active a1)
        mgr.load_adapter("a3", _make_adapter_params(seed=3))
        adapters = {a["name"] for a in mgr.list_adapters()}
        assert "a1" in adapters
        assert "a2" not in adapters

    def test_evict_active_when_max_loaded_1(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model, max_loaded=1)

        mgr.load_adapter("a1", _make_adapter_params(seed=1))
        mgr.activate("a1")

        # Loading a2 must evict the active a1 (only slot)
        mgr.load_adapter("a2", _make_adapter_params(seed=2))
        assert mgr.loaded_count == 1
        assert mgr.get_active() is None  # active cleared on eviction
        adapters = {a["name"] for a in mgr.list_adapters()}
        assert "a1" not in adapters
        assert "a2" in adapters

    def test_reloading_same_name_no_eviction(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model, max_loaded=2)

        mgr.load_adapter("a1", _make_adapter_params(seed=1))
        mgr.load_adapter("a2", _make_adapter_params(seed=2))

        # Reloading a1 should overwrite, not evict
        mgr.load_adapter("a1", _make_adapter_params(seed=10))
        assert mgr.loaded_count == 2


# ── List Adapters ────────────────────────────────────────────────────────────


class TestListAdapters:
    """Tests for list_adapters."""

    def test_empty(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        assert mgr.list_adapters() == []

    def test_lists_loaded(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        mgr.load_adapter("a1", _make_adapter_params())
        entries = mgr.list_adapters()
        assert len(entries) == 1
        assert entries[0]["name"] == "a1"
        assert entries[0]["active"] is False
        assert entries[0]["param_count"] > 0
        assert entries[0]["method"] == "lora"

    def test_active_flag(self):
        model, _ = _make_adapted_model()
        mgr = MultiAdapterManager(model)
        mgr.load_adapter("a1", _make_adapter_params())
        mgr.activate("a1")
        entries = mgr.list_adapters()
        assert entries[0]["active"] is True
