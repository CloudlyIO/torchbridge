"""
Tests for the Quantization Engine

Tests auto-format selection, explicit format application, fallback behavior,
result dataclass, and operation without torchao.
"""

from unittest import mock

import pytest
import torch
import torch.nn as nn

from torchbridge.precision.quantization.engine import (
    QuantizationEngine,
    QuantizationResult,
    _model_size_mb,
)
from torchbridge.precision.quantization.formats import QuantizationFormat

# =============================================================================
# Helper fixtures
# =============================================================================


@pytest.fixture
def small_model():
    """Create a small sequential model for testing."""
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )


@pytest.fixture
def engine():
    """Create an engine (auto-detects CPU on test machines)."""
    return QuantizationEngine()


# =============================================================================
# Engine Creation Tests
# =============================================================================


class TestEngineCreation:
    """Tests for QuantizationEngine initialization."""

    def test_creates_without_backend(self):
        """Engine should create without explicit backend."""
        engine = QuantizationEngine()
        assert engine.backend_name in ("cpu", "cuda", "amd", "trainium", "tpu")

    def test_creates_with_mock_backend(self):
        """Engine should accept a backend object."""

        class MockBackend:
            BACKEND_NAME = "nvidia"

        engine = QuantizationEngine(backend=MockBackend())
        assert engine.backend_name == "cuda"

    def test_cpu_backend_detected(self):
        """On CPU-only machines, backend should be 'cpu'."""
        if torch.cuda.is_available():
            pytest.skip("CUDA available, skipping CPU-only test")
        engine = QuantizationEngine()
        assert engine.backend_name == "cpu"

    def test_get_optimal_format(self, engine):
        """get_optimal_format should return a valid QuantizationFormat."""
        fmt = engine.get_optimal_format()
        assert isinstance(fmt, QuantizationFormat)

    def test_get_supported_formats(self, engine):
        """get_supported_formats should return a non-empty list."""
        formats = engine.get_supported_formats()
        assert len(formats) > 0
        for f in formats:
            assert isinstance(f, QuantizationFormat)


# =============================================================================
# Quantization Tests
# =============================================================================


class TestQuantization:
    """Tests for the quantize method."""

    def test_auto_selects_format(self, engine, small_model):
        """Auto strategy should select a format and succeed."""
        result = engine.quantize(small_model, format="auto")
        assert result.success
        assert result.model is not None
        assert result.format_applied != QuantizationFormat.NONE

    def test_explicit_int8_dynamic(self, engine, small_model):
        """Explicit INT8 dynamic should work on CPU."""
        result = engine.quantize(small_model, format="int8_dynamic")
        assert result.success
        assert result.format_applied == QuantizationFormat.INT8_DYNAMIC

    def test_explicit_format_enum(self, engine, small_model):
        """Passing QuantizationFormat enum should work."""
        result = engine.quantize(small_model, format=QuantizationFormat.INT8_DYNAMIC)
        assert result.success
        assert result.format_applied == QuantizationFormat.INT8_DYNAMIC

    def test_unsupported_format_falls_back(self, small_model):
        """Requesting NVFP4 on CPU should fall back with warning."""
        engine = QuantizationEngine()  # CPU
        if torch.cuda.is_available():
            pytest.skip("CUDA available, fallback behavior differs")
        result = engine.quantize(small_model, format="nvfp4")
        assert result.success
        assert result.used_fallback
        assert result.format_applied != QuantizationFormat.NVFP4
        assert len(result.warnings) > 0

    def test_deep_copy_by_default(self, engine, small_model):
        """Quantization should not modify original model by default."""
        original_params = {
            name: p.clone() for name, p in small_model.named_parameters()
        }
        engine.quantize(small_model, format="int8_dynamic")
        for name, p in small_model.named_parameters():
            assert torch.equal(p, original_params[name]), (
                f"Original model modified: {name}"
            )

    def test_in_place_modifies_model(self, engine, small_model):
        """in_place=True should quantize the model directly."""
        result = engine.quantize(small_model, format="int8_dynamic", in_place=True)
        assert result.success

    def test_none_format_returns_unmodified(self, engine, small_model):
        """NONE format should return model unchanged."""
        result = engine.quantize(small_model, format=QuantizationFormat.NONE)
        assert result.success
        assert result.format_applied == QuantizationFormat.NONE


# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestQuantizationResult:
    """Tests for the QuantizationResult dataclass."""

    def test_memory_reduction_pct(self):
        """memory_reduction_pct should calculate correctly."""
        result = QuantizationResult(
            success=True,
            model=None,
            format_applied=QuantizationFormat.INT8_DYNAMIC,
            format_requested=QuantizationFormat.INT8_DYNAMIC,
            memory_before_mb=100.0,
            memory_after_mb=50.0,
        )
        assert result.memory_reduction_pct == 50.0

    def test_memory_reduction_pct_zero_before(self):
        """memory_reduction_pct should be 0 when before is 0."""
        result = QuantizationResult(
            success=True,
            model=None,
            format_applied=QuantizationFormat.INT8_DYNAMIC,
            format_requested=QuantizationFormat.INT8_DYNAMIC,
            memory_before_mb=0.0,
            memory_after_mb=0.0,
        )
        assert result.memory_reduction_pct == 0.0

    def test_to_dict(self):
        """to_dict should include all fields."""
        result = QuantizationResult(
            success=True,
            model=None,
            format_applied=QuantizationFormat.FP8_E4M3,
            format_requested=QuantizationFormat.FP8_E4M3,
            used_fallback=False,
            memory_before_mb=200.0,
            memory_after_mb=100.0,
            warnings=["test warning"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["format_applied"] == "fp8_e4m3"
        assert d["memory_reduction_pct"] == 50.0
        assert "test warning" in d["warnings"]

    def test_default_fields(self):
        """Default fields should have sensible values."""
        result = QuantizationResult(
            success=False,
            model=None,
            format_applied=QuantizationFormat.NONE,
            format_requested=QuantizationFormat.NONE,
        )
        assert result.used_fallback is False
        assert result.fallback_chain == []
        assert result.warnings == []
        assert result.errors == []


# =============================================================================
# Model Size Helper Tests
# =============================================================================


class TestModelSizeMB:
    """Tests for _model_size_mb helper."""

    def test_linear_model_size(self):
        """Model size should be positive and reasonable."""
        model = nn.Linear(100, 100)  # 100*100 + 100 = 10100 params * 4 bytes
        size = _model_size_mb(model)
        assert size > 0
        # ~0.04 MB for a (100, 100) linear
        assert size < 1.0

    def test_larger_model_is_bigger(self):
        """Larger model should have larger size."""
        small = nn.Linear(10, 10)
        large = nn.Linear(1000, 1000)
        assert _model_size_mb(large) > _model_size_mb(small)


# =============================================================================
# torchao Unavailable Tests
# =============================================================================


class TestWithoutTorchAO:
    """Tests for engine behavior when torchao is not available."""

    def test_int8_works_without_torchao(self, small_model):
        """INT8 dynamic should fall back to PyTorch native without torchao."""
        with mock.patch(
            "torchbridge.precision.quantization.engine.TORCHAO_AVAILABLE", False
        ):
            engine = QuantizationEngine()
            result = engine.quantize(small_model, format="int8_dynamic")
            assert result.success

    def test_auto_works_without_torchao(self, small_model):
        """Auto strategy should still work without torchao."""
        with mock.patch(
            "torchbridge.precision.quantization.engine.TORCHAO_AVAILABLE", False
        ):
            engine = QuantizationEngine()
            result = engine.quantize(small_model, format="auto")
            assert result.success
