"""Tests for LoRA and DoRA linear adapter layers."""


import pytest
import torch
import torch.nn as nn

from torchbridge.adapters.config import InitMethod
from torchbridge.adapters.layers import DoRALinear, LoRALinear, QDoRALinear, QLoRALinear
from torchbridge.precision.quantization.formats import QuantizationFormat

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def base_linear():
    """A small nn.Linear for testing."""
    torch.manual_seed(42)
    return nn.Linear(32, 64)


@pytest.fixture
def lora(base_linear):
    """LoRALinear with default init."""
    return LoRALinear(base_linear, rank=4, alpha=8.0)


@pytest.fixture
def dora(base_linear):
    """DoRALinear with default init."""
    return DoRALinear(base_linear, rank=4, alpha=8.0)


# ── LoRALinear Tests ─────────────────────────────────────────────────────────


class TestLoRALinearInit:
    """Tests for LoRALinear construction."""

    def test_shapes(self, base_linear):
        layer = LoRALinear(base_linear, rank=4, alpha=8.0)
        assert layer.lora_A.weight.shape == (4, 32)
        assert layer.lora_B.weight.shape == (64, 4)

    def test_scaling(self, base_linear):
        layer = LoRALinear(base_linear, rank=4, alpha=8.0)
        assert layer._scaling == pytest.approx(2.0)

    def test_base_frozen(self, lora):
        for param in lora.base_linear.parameters():
            assert not param.requires_grad

    def test_adapter_trainable(self, lora):
        assert lora.lora_A.weight.requires_grad
        assert lora.lora_B.weight.requires_grad

    def test_kaiming_init_b_zeros(self, base_linear):
        layer = LoRALinear(
            base_linear, rank=4, alpha=8.0, init_method=InitMethod.KAIMING
        )
        assert torch.all(layer.lora_B.weight == 0)

    def test_gaussian_init_b_zeros(self, base_linear):
        layer = LoRALinear(
            base_linear, rank=4, alpha=8.0, init_method=InitMethod.GAUSSIAN
        )
        assert torch.all(layer.lora_B.weight == 0)
        assert not torch.all(layer.lora_A.weight == 0)

    def test_zeros_init(self, base_linear):
        layer = LoRALinear(
            base_linear, rank=4, alpha=8.0, init_method=InitMethod.ZEROS
        )
        assert torch.all(layer.lora_A.weight == 0)
        assert torch.all(layer.lora_B.weight == 0)

    def test_dropout_identity_when_zero(self, base_linear):
        layer = LoRALinear(base_linear, rank=4, alpha=8.0, dropout=0.0)
        assert isinstance(layer.lora_dropout, nn.Identity)

    def test_dropout_module_when_nonzero(self, base_linear):
        layer = LoRALinear(base_linear, rank=4, alpha=8.0, dropout=0.1)
        assert isinstance(layer.lora_dropout, nn.Dropout)


class TestLoRALinearForward:
    """Tests for LoRALinear forward pass."""

    def test_output_shape(self, lora):
        x = torch.randn(2, 32)
        out = lora(x)
        assert out.shape == (2, 64)

    def test_zeros_init_matches_base(self, base_linear):
        layer = LoRALinear(
            base_linear, rank=4, alpha=8.0, init_method=InitMethod.ZEROS
        )
        x = torch.randn(2, 32)
        expected = base_linear(x)
        actual = layer(x)
        torch.testing.assert_close(actual, expected)

    def test_adapter_adds_to_base(self, base_linear):
        layer = LoRALinear(
            base_linear, rank=4, alpha=8.0, init_method=InitMethod.KAIMING
        )
        x = torch.randn(2, 32)
        base_out = base_linear(x)
        adapter_out = layer(x)
        # Since B is zeros at init, adapter output equals base
        torch.testing.assert_close(adapter_out, base_out)

    def test_gradient_flows_to_adapter(self, lora):
        x = torch.randn(2, 32)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None

    def test_gradient_does_not_flow_to_base(self, lora):
        x = torch.randn(2, 32)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        assert lora.base_linear.weight.grad is None

    def test_batch_dims(self, lora):
        x = torch.randn(3, 5, 32)
        out = lora(x)
        assert out.shape == (3, 5, 64)


class TestLoRALinearMerge:
    """Tests for LoRALinear merge."""

    def test_merge_returns_linear(self, lora):
        merged = lora.merge()
        assert isinstance(merged, nn.Linear)
        assert merged.in_features == 32
        assert merged.out_features == 64

    def test_merge_preserves_bias(self):
        torch.manual_seed(42)
        base = nn.Linear(16, 32, bias=True)
        layer = LoRALinear(base, rank=4, alpha=8.0)
        merged = layer.merge()
        assert merged.bias is not None

    def test_merge_no_bias(self):
        base = nn.Linear(16, 32, bias=False)
        layer = LoRALinear(base, rank=4, alpha=8.0)
        merged = layer.merge()
        assert merged.bias is None

    def test_merge_output_matches(self, lora):
        x = torch.randn(5, 32)
        lora.eval()
        with torch.no_grad():
            expected = lora(x)
        merged = lora.merge()
        merged.eval()
        with torch.no_grad():
            actual = merged(x)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TestLoRALinearParams:
    """Tests for parameter counting."""

    def test_trainable_params(self, base_linear):
        layer = LoRALinear(base_linear, rank=4, alpha=8.0)
        # A: 32*4 = 128, B: 4*64 = 256
        assert layer.trainable_params == 128 + 256

    def test_total_params(self, base_linear):
        layer = LoRALinear(base_linear, rank=4, alpha=8.0)
        base_params = 32 * 64 + 64  # weight + bias
        adapter_params = 128 + 256
        assert layer.total_params == base_params + adapter_params


# ── DoRALinear Tests ─────────────────────────────────────────────────────────


class TestDoRALinearInit:
    """Tests for DoRALinear construction."""

    def test_shapes(self, base_linear):
        layer = DoRALinear(base_linear, rank=4, alpha=8.0)
        assert layer.lora_A.weight.shape == (4, 32)
        assert layer.lora_B.weight.shape == (64, 4)

    def test_magnitude_shape(self, base_linear):
        layer = DoRALinear(base_linear, rank=4, alpha=8.0)
        assert layer.magnitude.shape == (64,)

    def test_magnitude_initialized_from_base(self, base_linear):
        layer = DoRALinear(base_linear, rank=4, alpha=8.0)
        expected = base_linear.weight.norm(dim=1)
        torch.testing.assert_close(layer.magnitude.data, expected)

    def test_base_frozen(self, dora):
        for param in dora.base_linear.parameters():
            assert not param.requires_grad

    def test_magnitude_trainable(self, dora):
        assert dora.magnitude.requires_grad

    def test_adapter_trainable(self, dora):
        assert dora.lora_A.weight.requires_grad
        assert dora.lora_B.weight.requires_grad


class TestDoRALinearForward:
    """Tests for DoRALinear forward pass."""

    def test_output_shape(self, dora):
        x = torch.randn(2, 32)
        out = dora(x)
        assert out.shape == (2, 64)

    def test_gradient_flows_to_adapter(self, dora):
        x = torch.randn(2, 32)
        out = dora(x)
        loss = out.sum()
        loss.backward()
        assert dora.lora_A.weight.grad is not None
        assert dora.lora_B.weight.grad is not None
        assert dora.magnitude.grad is not None

    def test_gradient_does_not_flow_to_base(self, dora):
        x = torch.randn(2, 32)
        out = dora(x)
        loss = out.sum()
        loss.backward()
        assert dora.base_linear.weight.grad is None

    def test_batch_dims(self, dora):
        x = torch.randn(3, 5, 32)
        out = dora(x)
        assert out.shape == (3, 5, 64)


class TestDoRALinearMerge:
    """Tests for DoRALinear merge."""

    def test_merge_returns_linear(self, dora):
        merged = dora.merge()
        assert isinstance(merged, nn.Linear)
        assert merged.in_features == 32
        assert merged.out_features == 64

    def test_merge_output_matches(self, dora):
        x = torch.randn(5, 32)
        dora.eval()
        with torch.no_grad():
            expected = dora(x)
        merged = dora.merge()
        merged.eval()
        with torch.no_grad():
            actual = merged(x)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


class TestDoRALinearParams:
    """Tests for DoRA parameter counting."""

    def test_trainable_params(self, base_linear):
        layer = DoRALinear(base_linear, rank=4, alpha=8.0)
        # A: 32*4=128, B: 4*64=256, magnitude: 64
        assert layer.trainable_params == 128 + 256 + 64

    def test_total_includes_base(self, base_linear):
        layer = DoRALinear(base_linear, rank=4, alpha=8.0)
        base_params = 32 * 64 + 64  # weight + bias
        adapter_params = 128 + 256 + 64
        assert layer.total_params == base_params + adapter_params


# ── QLoRALinear Tests ─────────────────────────────────────────────────────────


class TestQLoRALinear:
    """Tests for QLoRALinear (quantized base + LoRA adapter)."""

    @pytest.fixture(autouse=True)
    def require_torchao(self):
        pytest.importorskip("torchao")

    @pytest.fixture
    def base(self):
        torch.manual_seed(42)
        return nn.Linear(64, 32)

    @pytest.fixture
    def qlora(self, base):
        return QLoRALinear(
            base, rank=4, alpha=8.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )

    def test_forward_shape_matches_base(self, qlora):
        x = torch.randn(2, 64)
        out = qlora(x)
        assert out.shape == (2, 32)

    def test_only_adapter_params_trainable(self, qlora):
        assert qlora.lora_A.weight.requires_grad
        assert qlora.lora_B.weight.requires_grad
        for param in qlora.base_linear.parameters():
            assert not param.requires_grad

    def test_merge_raises_not_implemented(self, qlora):
        with pytest.raises(NotImplementedError, match="quantized base weights"):
            qlora.merge()

    def test_memory_reduced_vs_fp32(self, base):
        torch.manual_seed(42)
        plain = nn.Linear(64, 32)
        lora_plain = LoRALinear(plain, rank=4, alpha=8.0)

        torch.manual_seed(42)
        base_q = nn.Linear(64, 32)
        qlora = QLoRALinear(
            base_q, rank=4, alpha=8.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )

        def size_bytes(m: nn.Module) -> int:
            return sum(
                p.untyped_storage().nbytes()
                for p in m.parameters()
                if hasattr(p, "untyped_storage")
            )

        # QLoRA base should use less storage than FP32 LoRA base
        assert size_bytes(qlora.base_linear) < size_bytes(lora_plain.base_linear)

    def test_batch_dims(self, qlora):
        x = torch.randn(3, 5, 64)
        out = qlora(x)
        assert out.shape == (3, 5, 32)


# ── QDoRALinear Tests ─────────────────────────────────────────────────────────


class TestQDoRALinear:
    """Tests for QDoRALinear (quantized base + DoRA adapter)."""

    @pytest.fixture(autouse=True)
    def require_torchao(self):
        pytest.importorskip("torchao")

    @pytest.fixture
    def base(self):
        torch.manual_seed(42)
        return nn.Linear(64, 32)

    @pytest.fixture
    def qdora(self, base):
        return QDoRALinear(
            base, rank=4, alpha=8.0,
            quant_format=QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
        )

    def test_forward_shape_matches_base(self, qdora):
        x = torch.randn(2, 64)
        out = qdora(x)
        assert out.shape == (2, 32)

    def test_only_adapter_params_trainable(self, qdora):
        assert qdora.lora_A.weight.requires_grad
        assert qdora.lora_B.weight.requires_grad
        assert qdora.magnitude.requires_grad
        for param in qdora.base_linear.parameters():
            assert not param.requires_grad

    def test_merge_raises_not_implemented(self, qdora):
        with pytest.raises(NotImplementedError, match="quantized base weights"):
            qdora.merge()
