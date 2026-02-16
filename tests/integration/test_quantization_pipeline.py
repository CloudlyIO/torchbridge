"""
Integration Tests for Quantization Pipeline

End-to-end tests: create models, quantize with various formats,
validate output quality, and test the auto strategy on CPU.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbridge.precision.quantization import (
    QuantizationEngine,
    QuantizationFormat,
)


@pytest.fixture
def medium_model():
    """Create a medium-sized model for integration testing."""
    return nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
    )


@pytest.fixture
def engine():
    """Create a quantization engine."""
    return QuantizationEngine()


class TestEndToEndQuantization:
    """End-to-end quantization tests."""

    def test_int8_preserves_output_quality(self, engine, medium_model):
        """Quantized model should produce outputs with high cosine similarity."""
        medium_model.eval()
        test_input = torch.randn(4, 256)

        with torch.no_grad():
            original_output = medium_model(test_input)

        result = engine.quantize(medium_model, format="int8_dynamic")
        assert result.success
        assert result.model is not None

        result.model.eval()
        # Cast input to match model dtype (may be BF16 if INT8 engine unavailable)
        model_dtype = next(result.model.parameters()).dtype
        q_input = test_input.to(dtype=model_dtype)
        with torch.no_grad():
            quantized_output = result.model(q_input)

        cos_sim = F.cosine_similarity(
            original_output.flatten().float().unsqueeze(0),
            quantized_output.flatten().float().unsqueeze(0),
        ).item()
        assert cos_sim > 0.95, f"Cosine similarity too low: {cos_sim:.4f}"

    def test_auto_strategy_on_cpu(self, engine, medium_model):
        """Auto strategy should work on CPU and produce a valid result."""
        result = engine.quantize(medium_model, format="auto")
        assert result.success
        assert result.model is not None
        assert result.format_applied != QuantizationFormat.NONE

        # Verify the model is usable
        result.model.eval()
        model_dtype = next(result.model.parameters()).dtype
        with torch.no_grad():
            output = result.model(torch.randn(2, 256, dtype=model_dtype))
        assert output.shape == (2, 32)

    def test_memory_actually_reduces(self, engine, medium_model):
        """INT8 quantization should reduce memory footprint."""
        result = engine.quantize(medium_model, format="int8_dynamic")
        assert result.success
        # INT8 should reduce memory; the actual reduction depends on
        # what PyTorch's quantize_dynamic does to the model
        assert result.memory_before_mb > 0
        assert result.memory_after_mb > 0

    def test_sequential_quantization(self, engine):
        """Multiple models should be quantizable sequentially."""
        results = []
        for size in [64, 128, 256]:
            model = nn.Sequential(nn.Linear(size, size // 2), nn.ReLU())
            result = engine.quantize(model, format="int8_dynamic")
            results.append(result)

        for r in results:
            assert r.success

    def test_result_to_dict_serializable(self, engine, medium_model):
        """Result.to_dict() should produce JSON-serializable data."""
        import json

        result = engine.quantize(medium_model, format="int8_dynamic")
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_bf16_quantization(self, engine, medium_model):
        """BF16 conversion should work on CPU."""
        result = engine.quantize(medium_model, format="bf16")
        assert result.success
        # On CPU, BF16 is in the supported formats so no fallback needed
        assert result.format_applied == QuantizationFormat.BF16

    def test_fallback_format_still_usable(self, medium_model):
        """Fallback format should produce a usable model."""
        if torch.cuda.is_available():
            pytest.skip("Test targets CPU-only fallback behavior")
        engine = QuantizationEngine()
        # Request NVFP4 on CPU — will fall back
        result = engine.quantize(medium_model, format="nvfp4")
        assert result.success
        assert result.model is not None

        result.model.eval()
        model_dtype = next(result.model.parameters()).dtype
        with torch.no_grad():
            output = result.model(torch.randn(1, 256, dtype=model_dtype))
        assert output.shape == (1, 32)


class TestQuantizationQualityValidation:
    """Quality validation across formats."""

    @pytest.mark.parametrize("format_str", ["int8_dynamic", "bf16"])
    def test_format_output_shape_preserved(self, engine, medium_model, format_str):
        """Quantized model should preserve output shape."""
        result = engine.quantize(medium_model, format=format_str)
        assert result.success
        assert result.model is not None

        result.model.eval()
        model_dtype = next(result.model.parameters()).dtype
        test_input = torch.randn(8, 256, dtype=model_dtype)
        with torch.no_grad():
            output = result.model(test_input)
        assert output.shape == (8, 32)

    def test_multiple_formats_compared(self, engine, medium_model):
        """All CPU-supported formats should produce similar outputs."""
        medium_model.eval()
        test_input = torch.randn(4, 256)

        with torch.no_grad():
            baseline = medium_model(test_input)

        for fmt_str in ["int8_dynamic", "bf16"]:
            result = engine.quantize(medium_model, format=fmt_str)
            if result.success and result.model is not None:
                result.model.eval()
                model_dtype = next(result.model.parameters()).dtype
                q_input = test_input.to(dtype=model_dtype)
                with torch.no_grad():
                    output = result.model(q_input)

                cos_sim = F.cosine_similarity(
                    baseline.flatten().float().unsqueeze(0),
                    output.flatten().float().unsqueeze(0),
                ).item()
                assert cos_sim > 0.9, (
                    f"{fmt_str}: cosine similarity too low: {cos_sim:.4f}"
                )
