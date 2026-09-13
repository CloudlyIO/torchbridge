"""
Unit tests for adapter layer classes (LoRA, DoRA, QLoRA, QDoRA).

QLoRA/QDoRA tests are skipped if torchao is not installed.
These tests FAIL until src/torchbridge/adapters/layers.py is created.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear(in_f: int = 64, out_f: int = 32) -> nn.Linear:
    return nn.Linear(in_f, out_f, bias=False)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------


class TestLoRALinear:
    """LoRALinear: standard low-rank adaptation around nn.Linear."""

    def test_forward_shape_matches_base(self):
        from torchbridge.adapters.layers import LoRALinear

        base = _make_linear(64, 32)
        layer = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 32)

    def test_only_adapter_params_trainable(self):
        from torchbridge.adapters.layers import LoRALinear

        base = _make_linear(64, 32)
        layer = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)

        trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
        frozen = [n for n, p in layer.named_parameters() if not p.requires_grad]

        assert any("lora_A" in n or "lora_B" in n for n in trainable)
        assert any("weight" in n for n in frozen), "base_linear weight must be frozen"

    def test_merge_produces_linear(self):
        from torchbridge.adapters.layers import LoRALinear

        base = _make_linear(64, 32)
        layer = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        merged = layer.merge()
        assert isinstance(merged, nn.Linear)
        assert merged.weight.shape == (32, 64)

    def test_trainable_params_property(self):
        from torchbridge.adapters.layers import LoRALinear

        base = _make_linear(64, 32)
        layer = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        # rank=4: lora_A is (4,64)=256, lora_B is (32,4)=128 → 384 trainable
        assert layer.trainable_params == 256 + 128

    def test_total_params_includes_base(self):
        from torchbridge.adapters.layers import LoRALinear

        base = _make_linear(64, 32)
        layer = LoRALinear(base, rank=4, alpha=8.0, dropout=0.0)
        assert layer.total_params > layer.trainable_params


# ---------------------------------------------------------------------------
# QLoRALinear
# ---------------------------------------------------------------------------


class TestQLoRALinear:
    """QLoRALinear: quantized base + LoRA adapters (requires torchao)."""

    def test_forward_shape_matches_base(self):
        pytest.importorskip("torchao")
        from torchbridge.adapters.layers import QLoRALinear
        from torchbridge.precision.formats import QuantizationFormat

        base = _make_linear(64, 32)
        layer = QLoRALinear(
            base,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 32)

    def test_only_adapter_params_trainable(self):
        pytest.importorskip("torchao")
        from torchbridge.adapters.layers import QLoRALinear
        from torchbridge.precision.formats import QuantizationFormat

        base = _make_linear(64, 32)
        layer = QLoRALinear(
            base,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )
        trainable = [n for n, p in layer.named_parameters() if p.requires_grad]
        assert any("lora_A" in n or "lora_B" in n for n in trainable), (
            "lora_A/lora_B must be trainable"
        )
        # base_linear params must be frozen
        base_trainable = [
            n
            for n, p in layer.named_parameters()
            if p.requires_grad and "lora_A" not in n and "lora_B" not in n
        ]
        assert not base_trainable, (
            f"base_linear params must be frozen: {base_trainable}"
        )

    def test_merge_raises_not_implemented(self):
        pytest.importorskip("torchao")
        from torchbridge.adapters.layers import QLoRALinear
        from torchbridge.precision.formats import QuantizationFormat

        base = _make_linear(64, 32)
        layer = QLoRALinear(
            base,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )
        with pytest.raises(NotImplementedError):
            layer.merge()

    def test_memory_reduced_vs_fp32(self):
        """INT8 base weights should use fewer bytes than FP32."""
        pytest.importorskip("torchao")
        from torchbridge.adapters.layers import LoRALinear, QLoRALinear
        from torchbridge.precision.formats import QuantizationFormat

        base_for_qlora = _make_linear(64, 32)
        base_for_lora = _make_linear(64, 32)

        qlora = QLoRALinear(
            base_for_qlora,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )
        lora = LoRALinear(base_for_lora, rank=4, alpha=8.0, dropout=0.0)

        def _real_bytes(t: torch.Tensor) -> int:
            """Bytes a tensor actually occupies, seeing inside subclasses.

            `untyped_storage().nbytes()` reports the *logical* storage of a
            quantized tensor, not its contents. For an Int8Tensor wrapping a
            64x32 weight it answers 8192 — the float32 size — while the int8
            data underneath is 2048. Measured that way, quantization looks
            free, which is what this test was accidentally asserting.

            `__tensor_flatten__` is the standard subclass protocol and names
            the inner tensors (qdata, scale, zero_point), so it survives
            torchao renaming things again.
            """
            if hasattr(t, "__tensor_flatten__"):
                names, _ = t.__tensor_flatten__()
                return sum(_real_bytes(getattr(t, n)) for n in names)
            return t.untyped_storage().nbytes()

        def _param_bytes(module: nn.Module) -> int:
            return sum(_real_bytes(p.data) for p in module.parameters())

        qlora_bytes = _param_bytes(qlora)
        lora_bytes = _param_bytes(lora)
        assert qlora_bytes < lora_bytes, (
            f"QLoRA ({qlora_bytes} bytes) should use less memory than LoRA ({lora_bytes} bytes)"
        )

    def test_requires_torchao_or_raises(self):
        """Without torchao, QLoRALinear must raise RuntimeError."""
        from unittest.mock import patch

        # Temporarily pretend torchao is unavailable
        import torchbridge.adapters.layers as layers_mod

        with patch.object(layers_mod, "_TORCHAO_AVAILABLE", False):
            from torchbridge.adapters.layers import QLoRALinear
            from torchbridge.precision.formats import QuantizationFormat

            base = _make_linear(64, 32)
            with pytest.raises(RuntimeError, match="torchao"):
                QLoRALinear(
                    base,
                    rank=4,
                    alpha=8.0,
                    dropout=0.0,
                    quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
                )


# ---------------------------------------------------------------------------
# QDoRALinear
# ---------------------------------------------------------------------------


class TestQDoRALinear:
    """QDoRALinear: quantized base + DoRA adapters (requires torchao)."""

    def test_forward_shape_matches_base(self):
        pytest.importorskip("torchao")
        from torchbridge.adapters.layers import QDoRALinear
        from torchbridge.precision.formats import QuantizationFormat

        base = _make_linear(64, 32)
        layer = QDoRALinear(
            base,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )
        x = torch.randn(2, 64)
        out = layer(x)
        assert out.shape == (2, 32)

    def test_only_adapter_params_trainable(self):
        pytest.importorskip("torchao")
        from torchbridge.adapters.layers import QDoRALinear
        from torchbridge.precision.formats import QuantizationFormat

        base = _make_linear(64, 32)
        layer = QDoRALinear(
            base,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )
        trainable_names = [n for n, p in layer.named_parameters() if p.requires_grad]
        assert any(
            "lora_A" in n or "lora_B" in n or "magnitude" in n for n in trainable_names
        )

    def test_merge_raises_not_implemented(self):
        pytest.importorskip("torchao")
        from torchbridge.adapters.layers import QDoRALinear
        from torchbridge.precision.formats import QuantizationFormat

        base = _make_linear(64, 32)
        layer = QDoRALinear(
            base,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )
        with pytest.raises(NotImplementedError):
            layer.merge()
