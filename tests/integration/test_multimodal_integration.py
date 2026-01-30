"""Integration tests for multi-modal model optimization (v0.4.15)."""

import pytest
import torch

from kernel_pytorch.models.multimodal import (
    # Base
    CLIPOptimizer,
    CrossModalAttention,
    # LLaVA
    LLaVAOptimizer,
    ModalityType,
    MultiModalOptimizationConfig,
    MultiModalType,
    OptimizationLevel,
    # Whisper
    WhisperOptimizer,
)


class TestMultiModalOptimizationConfig:
    """Test multi-modal optimization configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = MultiModalOptimizationConfig()

        assert config.model_type == MultiModalType.CUSTOM
        assert config.optimization_level == OptimizationLevel.O2
        assert ModalityType.VISION in config.modalities
        assert ModalityType.TEXT in config.modalities

    def test_from_optimization_level(self):
        """Test creating config from optimization level."""
        config = MultiModalOptimizationConfig.from_optimization_level(OptimizationLevel.O3)

        assert config.optimization_level == OptimizationLevel.O3
        assert config.compile_model is True
        assert config.enable_attention_slicing is True

    def test_fp16_bf16_mutual_exclusion(self):
        """Test that FP16 and BF16 cannot be used simultaneously."""
        with pytest.raises(ValueError, match="Cannot use both FP16 and BF16"):
            MultiModalOptimizationConfig(use_fp16=True, use_bf16=True)


class TestCrossModalAttention:
    """Test cross-modal attention layer."""

    def test_forward_pass(self):
        """Test forward pass."""
        layer = CrossModalAttention(dim=256, num_heads=8)

        x_query = torch.randn(2, 10, 256)
        x_context = torch.randn(2, 20, 256)

        output = layer(x_query, x_context)

        assert output.shape == (2, 10, 256)


class TestCLIPOptimizer:
    """Test CLIP optimizer."""

    def test_optimizer_creation(self):
        """Test CLIP optimizer creation."""
        config = MultiModalOptimizationConfig(model_type=MultiModalType.CLIP)
        optimizer = CLIPOptimizer(config)

        assert optimizer.config.model_type == MultiModalType.CLIP
        assert ModalityType.VISION in optimizer.config.modalities
        assert ModalityType.TEXT in optimizer.config.modalities


class TestLLaVAOptimizer:
    """Test LLaVA optimizer."""

    def test_optimizer_creation(self):
        """Test LLaVA optimizer creation."""
        config = MultiModalOptimizationConfig(model_type=MultiModalType.LLAVA)
        optimizer = LLaVAOptimizer(config)

        assert optimizer.config.model_type == MultiModalType.LLAVA


class TestWhisperOptimizer:
    """Test Whisper optimizer."""

    def test_optimizer_creation(self):
        """Test Whisper optimizer creation."""
        config = MultiModalOptimizationConfig(model_type=MultiModalType.WHISPER)
        optimizer = WhisperOptimizer(config)

        assert optimizer.config.model_type == MultiModalType.WHISPER


class TestModuleExports:
    """Test module exports."""

    def test_base_exports(self):
        """Test base module exports."""
        from kernel_pytorch.models import multimodal

        assert hasattr(multimodal, "BaseMultiModalOptimizer")
        assert hasattr(multimodal, "MultiModalOptimizationConfig")
        assert hasattr(multimodal, "CrossModalAttention")

    def test_clip_exports(self):
        """Test CLIP exports."""
        from kernel_pytorch.models import multimodal

        assert hasattr(multimodal, "CLIPOptimizer")
        assert hasattr(multimodal, "CLIPBenchmark")
        assert hasattr(multimodal, "create_clip_optimizer")

    def test_llava_exports(self):
        """Test LLaVA exports."""
        from kernel_pytorch.models import multimodal

        assert hasattr(multimodal, "LLaVAOptimizer")
        assert hasattr(multimodal, "LLaVABenchmark")
        assert hasattr(multimodal, "create_llava_optimizer")

    def test_whisper_exports(self):
        """Test Whisper exports."""
        from kernel_pytorch.models import multimodal

        assert hasattr(multimodal, "WhisperOptimizer")
        assert hasattr(multimodal, "WhisperBenchmark")
        assert hasattr(multimodal, "create_whisper_optimizer")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
