"""
Unit tests for AdapterEngine.inject().

These tests FAIL until src/torchbridge/adapters/engine.py is created.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch.nn as nn


class TestAdapterEngine:
    """AdapterEngine.inject() must produce correct AdapterResult."""

    def test_inject_lora_cpu_sets_base_not_quantized(self):
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine, AdapterResult
        from torchbridge.core.config import HardwareBackend

        model = nn.Sequential(nn.Linear(32, 16), nn.Linear(16, 8))
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            target_modules=["0", "1"],
            rank=4,
            alpha=8.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)
        result = engine.inject(model)

        assert isinstance(result, AdapterResult)
        assert result.base_quantized is False
        assert result.base_quant_format is None

    def test_inject_qlora_cpu_sets_base_quantized(self):
        pytest.importorskip("torchao")
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend
        from torchbridge.precision.quantization.formats import QuantizationFormat

        model = nn.Sequential(nn.Linear(32, 16))
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            target_modules=["0"],
            rank=4,
            alpha=8.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)
        result = engine.inject(model)

        assert result.base_quantized is True
        assert result.base_quant_format == QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS

    def test_inject_qlora_fallback_when_no_torchao(self):
        """When torchao is unavailable, QLoRA must fall back to LoRA."""
        import torchbridge.adapters.layers as layers_mod
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend

        model = nn.Sequential(nn.Linear(32, 16))
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            target_modules=["0"],
            rank=4,
            alpha=8.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)

        with patch.object(layers_mod, "_TORCHAO_AVAILABLE", False):
            result = engine.inject(model)

        assert result.method_applied == AdapterMethod.LORA
        assert result.base_quantized is False

    def test_inject_non_module_raises_type_error(self):
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend

        config = AdapterConfig(method=AdapterMethod.LORA, target_modules=["weight"])
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)

        with pytest.raises(TypeError, match="nn.Module"):
            engine.inject("not_a_module")  # type: ignore[arg-type]

    def test_inject_counts_modified_layers(self):
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend

        # 3 linear layers, 2 match target_modules
        model = nn.Sequential(
            nn.Linear(32, 16),   # "0"
            nn.Linear(16, 8),    # "1"
            nn.Linear(8, 4),     # "2" — not targeted
        )
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            target_modules=["0", "1"],
            rank=4,
            alpha=8.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)
        result = engine.inject(model)

        assert result.layers_modified == 2

    def test_inject_returns_positive_trainable_params(self):
        from torchbridge.adapters.config import AdapterConfig, AdapterMethod
        from torchbridge.adapters.engine import AdapterEngine
        from torchbridge.core.config import HardwareBackend

        model = nn.Sequential(nn.Linear(32, 16))
        config = AdapterConfig(
            method=AdapterMethod.LORA,
            target_modules=["0"],
            rank=4,
            alpha=8.0,
        )
        engine = AdapterEngine(config=config, backend=HardwareBackend.CPU)
        result = engine.inject(model)

        assert result.trainable_params > 0
        assert result.total_params >= result.trainable_params
