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

        ratio = result.trainable_params / result.total_params
        assert ratio < 0.05, (
            f"QLoRA trainable ratio {ratio:.3f} should be < 5% of total params"
        )
