"""Tests for adapter configuration."""

import pytest

from torchbridge.adapters.config import (
    AdapterConfig,
    AdapterMethod,
    InitMethod,
)
from torchbridge.precision.formats import QuantizationFormat


class TestAdapterMethod:
    """Tests for AdapterMethod enum."""

    def test_all_values(self):
        assert AdapterMethod.LORA.value == "lora"
        assert AdapterMethod.QLORA.value == "qlora"
        assert AdapterMethod.DORA.value == "dora"
        assert AdapterMethod.QDORA.value == "qdora"

    def test_count(self):
        assert len(AdapterMethod) == 4


class TestInitMethod:
    """Tests for InitMethod enum."""

    def test_all_values(self):
        assert InitMethod.KAIMING.value == "kaiming"
        assert InitMethod.GAUSSIAN.value == "gaussian"
        assert InitMethod.ZEROS.value == "zeros"

    def test_count(self):
        assert len(InitMethod) == 3


class TestAdapterConfig:
    """Tests for AdapterConfig dataclass."""

    def test_defaults(self):
        config = AdapterConfig()
        assert config.method == AdapterMethod.LORA
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.0
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.quantize_base is False
        assert config.base_quant_format is None
        assert config.init_method == InitMethod.KAIMING
        assert config.merge_on_save is False

    def test_custom_values(self):
        config = AdapterConfig(
            method=AdapterMethod.DORA,
            rank=8,
            alpha=16.0,
            dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            init_method=InitMethod.GAUSSIAN,
        )
        assert config.method == AdapterMethod.DORA
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert len(config.target_modules) == 4

    def test_qlora_auto_sets_quantize_base(self):
        config = AdapterConfig(method=AdapterMethod.QLORA)
        assert config.quantize_base is True

    def test_qdora_auto_sets_quantize_base(self):
        config = AdapterConfig(method=AdapterMethod.QDORA)
        assert config.quantize_base is True

    def test_lora_does_not_set_quantize_base(self):
        config = AdapterConfig(method=AdapterMethod.LORA)
        assert config.quantize_base is False

    def test_invalid_rank(self):
        with pytest.raises(ValueError, match="rank"):
            AdapterConfig(rank=0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            AdapterConfig(alpha=-1.0)

    def test_invalid_dropout_negative(self):
        with pytest.raises(ValueError, match="dropout"):
            AdapterConfig(dropout=-0.1)

    def test_invalid_dropout_one(self):
        with pytest.raises(ValueError, match="dropout"):
            AdapterConfig(dropout=1.0)

    def test_empty_target_modules(self):
        with pytest.raises(ValueError, match="target_modules"):
            AdapterConfig(target_modules=[])

    def test_to_dict(self):
        config = AdapterConfig()
        d = config.to_dict()
        assert d["method"] == "lora"
        assert d["rank"] == 16
        assert d["alpha"] == 32.0
        assert d["dropout"] == 0.0
        assert d["quantize_base"] is False
        assert d["base_quant_format"] is None
        assert d["init_method"] == "kaiming"

    def test_to_dict_with_quant_format(self):
        config = AdapterConfig(
            method=AdapterMethod.QLORA,
            base_quant_format=QuantizationFormat.INT4_WEIGHT_ONLY,
        )
        d = config.to_dict()
        assert d["base_quant_format"] == "int4_weight_only"
        assert d["quantize_base"] is True

    def test_rank_one_is_valid(self):
        config = AdapterConfig(rank=1)
        assert config.rank == 1

    def test_high_rank_is_valid(self):
        config = AdapterConfig(rank=256)
        assert config.rank == 256
