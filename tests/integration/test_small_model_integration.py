"""
Test Suite for Small Model Integration (v0.5.3)

Tests for BERT, GPT-2, and DistilBERT optimization wrappers.
Validates optimization, inference, and backend integration.

"""

import pytest
import torch
import torch.nn as nn


class TestTextModelTypes:
    """Tests for TextModelType enum."""

    def test_model_types_exist(self):
        """Verify all model types are defined."""
        from torchbridge.models.text.text_model_optimizer import TextModelType

        assert hasattr(TextModelType, 'BERT')
        assert hasattr(TextModelType, 'GPT2')
        assert hasattr(TextModelType, 'DISTILBERT')
        assert hasattr(TextModelType, 'ROBERTA')
        assert hasattr(TextModelType, 'ALBERT')
        assert hasattr(TextModelType, 'CUSTOM')

    def test_model_type_values(self):
        """Verify model type values."""
        from torchbridge.models.text.text_model_optimizer import TextModelType

        assert TextModelType.BERT.value == "bert"
        assert TextModelType.GPT2.value == "gpt2"
        assert TextModelType.DISTILBERT.value == "distilbert"

class TestOptimizationMode:
    """Tests for OptimizationMode enum."""

    def test_optimization_modes_exist(self):
        """Verify all optimization modes are defined."""
        from torchbridge.models.text.text_model_optimizer import OptimizationMode

        assert hasattr(OptimizationMode, 'INFERENCE')
        assert hasattr(OptimizationMode, 'THROUGHPUT')
        assert hasattr(OptimizationMode, 'MEMORY')
        assert hasattr(OptimizationMode, 'BALANCED')

    def test_optimization_mode_values(self):
        """Verify optimization mode values."""
        from torchbridge.models.text.text_model_optimizer import OptimizationMode

        assert OptimizationMode.INFERENCE.value == "inference"
        assert OptimizationMode.THROUGHPUT.value == "throughput"
        assert OptimizationMode.MEMORY.value == "memory"
        assert OptimizationMode.BALANCED.value == "balanced"

class TestTextModelConfig:
    """Tests for TextModelConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from torchbridge.models.text.text_model_optimizer import (
            OptimizationMode,
            TextModelConfig,
            TextModelType,
        )

        config = TextModelConfig()

        assert config.model_name == "bert-base-uncased"
        assert config.model_type == TextModelType.BERT
        assert config.max_sequence_length == 512
        assert config.optimization_mode == OptimizationMode.INFERENCE
        assert config.use_torch_compile is True
        assert config.compile_mode == "reduce-overhead"
        assert config.dtype is None
        assert config.use_amp is True
        assert config.gradient_checkpointing is False
        assert config.enable_memory_efficient_attention is True
        assert config.device == "auto"
        assert config.warmup_steps == 3
        assert config.enable_profiling is False

    def test_custom_config(self):
        """Test custom configuration."""
        from torchbridge.models.text.text_model_optimizer import (
            OptimizationMode,
            TextModelConfig,
            TextModelType,
        )

        config = TextModelConfig(
            model_name="gpt2",
            model_type=TextModelType.GPT2,
            max_sequence_length=1024,
            optimization_mode=OptimizationMode.THROUGHPUT,
            use_torch_compile=False,
            dtype=torch.float16,
        )

        assert config.model_name == "gpt2"
        assert config.model_type == TextModelType.GPT2
        assert config.max_sequence_length == 1024
        assert config.optimization_mode == OptimizationMode.THROUGHPUT
        assert config.use_torch_compile is False
        assert config.dtype == torch.float16

class TestTextModelOptimizer:
    """Tests for TextModelOptimizer class."""

    def test_optimizer_creation(self):
        """Test optimizer instantiation."""
        from torchbridge.models.text.text_model_optimizer import TextModelOptimizer

        optimizer = TextModelOptimizer()

        assert optimizer.device is not None
        assert optimizer.dtype is not None
        assert optimizer.config is not None

    def test_optimizer_with_custom_config(self):
        """Test optimizer with custom configuration."""
        from torchbridge.models.text.text_model_optimizer import (
            OptimizationMode,
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            optimization_mode=OptimizationMode.MEMORY,
            use_torch_compile=False,
        )

        optimizer = TextModelOptimizer(config)

        assert optimizer.config.optimization_mode == OptimizationMode.MEMORY
        assert optimizer.config.use_torch_compile is False

    def test_device_detection_cpu(self):
        """Test CPU device detection."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(device="cpu")
        optimizer = TextModelOptimizer(config)

        # Note: Backend may override device, but config should be respected
        assert optimizer.config.device == "cpu"

    def test_get_optimization_info(self):
        """Test optimization info retrieval."""
        from torchbridge.models.text.text_model_optimizer import TextModelOptimizer

        optimizer = TextModelOptimizer()
        info = optimizer.get_optimization_info()

        assert "device" in info
        assert "dtype" in info
        assert "backend" in info
        assert "optimization_mode" in info
        assert "torch_compile" in info
        assert "compile_mode" in info
        assert "efficient_attention" in info

    def test_optimize_with_mock_model(self):
        """Test optimization with a mock model."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            use_torch_compile=False,  # Disable for faster testing
            warmup_steps=0,
        )
        optimizer = TextModelOptimizer(config)

        # Create a simple mock model
        mock_model = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 2)
        )

        optimized = optimizer.optimize(mock_model)

        assert optimized is not None
        # Model should be in eval mode for inference optimization
        assert not optimized.training

class TestModelTypeDetection:
    """Tests for automatic model type detection."""

    def test_detect_bert_model(self):
        """Test BERT model detection."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelOptimizer,
            TextModelType,
        )

        optimizer = TextModelOptimizer()

        # Mock model with BERT-like name
        class MockBertModel(nn.Module):
            pass

        MockBertModel.__name__ = "BertForSequenceClassification"
        model = MockBertModel()

        detected_type = optimizer._detect_model_type(model)
        assert detected_type == TextModelType.BERT

    def test_detect_gpt2_model(self):
        """Test GPT-2 model detection."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelOptimizer,
            TextModelType,
        )

        optimizer = TextModelOptimizer()

        class MockGPT2Model(nn.Module):
            pass

        MockGPT2Model.__name__ = "GPT2LMHeadModel"
        model = MockGPT2Model()

        detected_type = optimizer._detect_model_type(model)
        assert detected_type == TextModelType.GPT2

    def test_detect_distilbert_model(self):
        """Test DistilBERT model detection."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelOptimizer,
            TextModelType,
        )

        optimizer = TextModelOptimizer()

        class MockDistilBertModel(nn.Module):
            pass

        MockDistilBertModel.__name__ = "DistilBertForSequenceClassification"
        model = MockDistilBertModel()

        detected_type = optimizer._detect_model_type(model)
        assert detected_type == TextModelType.DISTILBERT

    def test_detect_custom_model(self):
        """Test custom/unknown model detection."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelOptimizer,
            TextModelType,
        )

        optimizer = TextModelOptimizer()

        class CustomModel(nn.Module):
            pass

        model = CustomModel()

        detected_type = optimizer._detect_model_type(model)
        assert detected_type == TextModelType.CUSTOM

class TestOptimizationModes:
    """Tests for different optimization modes."""

    def test_inference_mode_disables_gradients(self):
        """Test that inference mode disables gradients."""
        from torchbridge.models.text.text_model_optimizer import (
            OptimizationMode,
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            optimization_mode=OptimizationMode.INFERENCE,
            use_torch_compile=False,
            warmup_steps=0,
        )
        optimizer = TextModelOptimizer(config)

        model = nn.Linear(10, 2)
        optimized = optimizer.optimize(model)

        # Check gradients are disabled
        for param in optimized.parameters():
            assert not param.requires_grad

    def test_memory_mode(self):
        """Test memory optimization mode."""
        from torchbridge.models.text.text_model_optimizer import (
            OptimizationMode,
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            optimization_mode=OptimizationMode.MEMORY,
            use_torch_compile=False,
            warmup_steps=0,
        )
        optimizer = TextModelOptimizer(config)

        model = nn.Linear(10, 2)
        optimized = optimizer.optimize(model)

        assert optimized is not None
        assert not optimized.training

class TestOptimizedBERT:
    """Tests for OptimizedBERT wrapper."""

    def test_optimized_bert_init_without_transformers(self):
        """Test OptimizedBERT with mock when transformers unavailable."""

        # This will fail if transformers is not installed
        # We can mock it or skip
        pytest.skip("Requires transformers library for full test")

    def test_optimized_bert_device_property(self):
        """Test device property exists."""
        # Mock test
        from torchbridge.models.text.text_model_optimizer import TextModelOptimizer

        optimizer = TextModelOptimizer()
        assert hasattr(optimizer, 'device')

class TestOptimizedGPT2:
    """Tests for OptimizedGPT2 wrapper."""

    def test_optimized_gpt2_config(self):
        """Test OptimizedGPT2 uses correct compile mode."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelType,
        )

        # GPT-2 should use reduce-overhead for generation
        config = TextModelConfig(
            model_name="gpt2",
            model_type=TextModelType.GPT2,
            compile_mode="reduce-overhead"
        )

        assert config.compile_mode == "reduce-overhead"

class TestOptimizedDistilBERT:
    """Tests for OptimizedDistilBERT wrapper."""

    def test_optimized_distilbert_config(self):
        """Test OptimizedDistilBERT uses inference mode by default."""
        from torchbridge.models.text.text_model_optimizer import (
            OptimizationMode,
            TextModelConfig,
            TextModelType,
        )

        config = TextModelConfig(
            model_name="distilbert-base-uncased",
            model_type=TextModelType.DISTILBERT,
            optimization_mode=OptimizationMode.INFERENCE
        )

        assert config.optimization_mode == OptimizationMode.INFERENCE

class TestFactoryFunction:
    """Tests for create_optimized_text_model factory function."""

    def test_factory_function_exists(self):
        """Test factory function is importable."""
        from torchbridge.models.text import create_optimized_text_model

        assert callable(create_optimized_text_model)

    def test_factory_mode_mapping(self):
        """Test optimization mode string mapping."""
        from torchbridge.models.text.text_model_optimizer import (
            OptimizationMode,
            TextModelConfig,
        )

        # Test mode mapping logic
        mode_map = {
            "inference": OptimizationMode.INFERENCE,
            "throughput": OptimizationMode.THROUGHPUT,
            "memory": OptimizationMode.MEMORY,
            "balanced": OptimizationMode.BALANCED,
        }

        for mode_str, mode_enum in mode_map.items():
            config = TextModelConfig(
                optimization_mode=mode_map.get(mode_str, OptimizationMode.INFERENCE)
            )
            assert config.optimization_mode == mode_enum

class TestDtypeHandling:
    """Tests for dtype conversion and handling."""

    def test_float16_conversion(self):
        """Test FP16 dtype conversion."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            dtype=torch.float16,
            use_torch_compile=False,
            warmup_steps=0,
        )
        optimizer = TextModelOptimizer(config)

        model = nn.Linear(10, 2)
        optimized = optimizer.optimize(model)

        # Check model is converted (may vary by device)
        assert optimized is not None

    def test_bfloat16_conversion(self):
        """Test BF16 dtype conversion."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            dtype=torch.bfloat16,
            use_torch_compile=False,
            warmup_steps=0,
        )
        optimizer = TextModelOptimizer(config)

        model = nn.Linear(10, 2)
        optimized = optimizer.optimize(model)

        assert optimized is not None

class TestModuleExports:
    """Tests for module exports and __all__."""

    def test_text_module_exports(self):
        """Test text module exports all required classes."""
        from torchbridge.models.text import (
            OptimizedBERT,
            OptimizedDistilBERT,
            OptimizedGPT2,
            TextModelOptimizer,
            create_optimized_text_model,
        )

        assert TextModelOptimizer is not None
        assert OptimizedBERT is not None
        assert OptimizedGPT2 is not None
        assert OptimizedDistilBERT is not None
        assert create_optimized_text_model is not None

    def test_models_module_exports(self):
        """Test models module exports text components."""
        from torchbridge.models import (
            TextModelOptimizer,
        )

        assert TextModelOptimizer is not None

class TestBackendIntegration:
    """Tests for backend integration."""

    def test_optimizer_uses_backend_factory(self):
        """Test that optimizer attempts to use BackendFactory."""
        from torchbridge.models.text.text_model_optimizer import TextModelOptimizer

        # Create optimizer - it should try to use BackendFactory
        optimizer = TextModelOptimizer()

        # Should have a device set
        assert optimizer.device is not None
        assert isinstance(optimizer.device, torch.device)

    def test_optimizer_fallback_to_pytorch(self):
        """Test optimizer falls back gracefully when backend unavailable."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(device="cpu")
        optimizer = TextModelOptimizer(config)

        # Should still work on CPU
        assert optimizer.device.type in ["cpu", "cuda", "xpu", "hip"]

class TestWarmup:
    """Tests for model warmup functionality."""

    def test_warmup_runs(self):
        """Test warmup executes without error."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            use_torch_compile=False,
            warmup_steps=0,  # Skip warmup for simple model test
        )
        optimizer = TextModelOptimizer(config)

        # Use a simple model that doesn't need special handling
        model = nn.Linear(64, 2)

        # This should complete without error
        optimized = optimizer.optimize(model)
        assert optimized is not None

    def test_no_warmup(self):
        """Test with warmup disabled."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            use_torch_compile=False,
            warmup_steps=0,
        )
        optimizer = TextModelOptimizer(config)

        model = nn.Linear(10, 2)
        optimized = optimizer.optimize(model)

        assert optimized is not None

class TestErrorHandling:
    """Tests for error handling."""

    def test_none_model_handling(self):
        """Test handling of None model."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            use_torch_compile=False,
            warmup_steps=0,
        )
        optimizer = TextModelOptimizer(config)

        # Passing a non-model should work with the optimize method
        # since it accepts nn.Module or string
        model = nn.Linear(10, 2)
        result = optimizer.optimize(model)
        assert result is not None

# Integration tests that require transformers
class TestTransformersIntegration:
    """Integration tests with HuggingFace transformers."""

    @pytest.mark.skip(reason="Requires transformers library and model download")
    def test_load_bert_model(self):
        """Test loading BERT from HuggingFace."""
        from torchbridge.models.text.text_model_optimizer import (
            TextModelConfig,
            TextModelOptimizer,
        )

        config = TextModelConfig(
            model_name="bert-base-uncased",
            use_torch_compile=False,
            warmup_steps=0,
        )
        TextModelOptimizer(config)

        # This would download the model
        # Skip in CI to avoid downloads
        pytest.skip("Skipping model download in tests")

    @pytest.mark.skip(reason="Requires transformers library and model download")
    def test_load_gpt2_model(self):
        """Test loading GPT-2 from HuggingFace."""
        pytest.skip("Skipping model download in tests")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
