"""
Integration tests for adapter injection pipeline.

End-to-end: AdapterConfig → AdapterEngine.inject() → forward pass.
QLoRA tests skipped if torchao is not installed.
These tests FAIL until src/torchbridge/adapters/engine.py is created.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class TestLoRAPipeline:
    """LoRA inject → forward → correct shapes and param counts."""

    def test_lora_inject_and_forward(self):
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16))
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            target_modules=["0", "2"],
            rank=8,
            alpha=16.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)
        result = engine.inject(model)

        x = torch.randn(4, 64)
        out = model(x)
        assert out.shape == (4, 16)
        assert result.layers_modified == 2

    def test_lora_trainable_ratio(self):
        """Adapter params should be ≪ total params for rank=4."""
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend

        model = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64))
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            target_modules=["0", "1"],
            rank=4,
            alpha=8.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)
        result = engine.inject(model)

        ratio = result.trainable_params / result.total_params
        assert ratio < 0.10, (
            f"Trainable ratio {ratio:.3f} too high for rank=4 LoRA — "
            "should be < 10% of total params"
        )


class TestQLoRAPipeline:
    """QLoRA inject → forward → correct shapes and reduced trainable ratio."""

    def test_qlora_inject_and_forward(self):
        pytest.importorskip("torchao")
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend

        model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            target_modules=["0", "1"],
            rank=4,
            alpha=8.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)
        result = engine.inject(model)

        x = torch.randn(2, 64)
        out = model(x)
        assert out.shape == (2, 16)
        assert result.base_quantized is True

    def test_qlora_trainable_ratio_low(self):
        """Only adapter matrices (lora_A, lora_B) should be trainable."""
        pytest.importorskip("torchao")
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend

        # Large enough linear to make the ratio meaningful
        model = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 64))
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            target_modules=["0", "1"],
            rank=4,
            alpha=8.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)
        result = engine.inject(model)

        # The docstring's claim, asserted directly: only the adapter matrices
        # carry gradients. The ratio below is a consequence of this, not a
        # substitute for it.
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable, "nothing is trainable — inject() did not attach adapters"
        assert all("lora_A" in n or "lora_B" in n for n in trainable), (
            f"something other than the adapter matrices is trainable: "
            f"{sorted(n for n in trainable if 'lora_' not in n)}"
        )

        # The 5% threshold this test used to assert was never reachable for
        # this model. Linear(256,128)+Linear(128,64) is 41152 base parameters;
        # rank-4 adapters on both add 1536 + 768 = 2304; 2304/43456 = 5.30%.
        # The bound was picked without doing that arithmetic, and the test was
        # skipped on every run (importorskip("torchao")), so it never failed.
        #
        # The real property is that the ratio shrinks as the model grows —
        # adapter cost is linear in width, base cost quadratic — which is the
        # claim worth defending, and it holds at any size.
        ratio = result.trainable_params / result.total_params
        assert ratio == pytest.approx(2304 / 43456, rel=1e-3), (
            f"unexpected trainable ratio {ratio:.4f}; if the adapter shapes "
            f"changed on purpose, recompute the expected value rather than "
            f"loosening the bound"
        )

        wide = nn.Sequential(nn.Linear(2048, 2048))
        wide_engine = AdapterEngine(
            config=AdapterConfig(
                method=AdapterMethod.QLORA,
                target_modules=["0"],
                rank=4,
                alpha=8.0,
            ),
            backend=HardwareBackend.CPU,
        )
        wide_result = wide_engine.inject(wide)
        wide_ratio = wide_result.trainable_params / wide_result.total_params
        assert wide_ratio < ratio / 10, (
            f"the adapter share should fall away as the model grows: "
            f"{wide_ratio:.5f} at 2048 wide vs {ratio:.4f} at 256"
        )
