"""
Integration tests for vision model optimization.

Tests cover:
- ResNet optimization
- Vision Transformer optimization
- Stable Diffusion optimization
- Configuration validation
- Optimization correctness
"""

import pytest
import torch
import torch.nn as nn

from torchbridge.models.vision import (
    # Base
    BaseVisionOptimizer,
    OptimizationLevel,
    ResNetBenchmark,
    # ResNet
    ResNetOptimizer,
    StableDiffusionOptimizer,
    VisionModelType,
    VisionOptimizationConfig,
    ViTOptimizer,
    count_parameters,
    create_resnet50_optimized,
    estimate_model_memory,
)

# ============================================================================
# Configuration Tests
# ============================================================================


class TestVisionOptimizationConfig:
    """Test vision optimization configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = VisionOptimizationConfig()

        assert config.model_type == VisionModelType.CUSTOM
        assert config.optimization_level == OptimizationLevel.O2
        assert config.batch_size == 1
        assert config.enable_fusion is True
        assert config.channels_last is True

    def test_from_optimization_level(self):
        """Test creating config from optimization level."""
        config = VisionOptimizationConfig.from_optimization_level(OptimizationLevel.O3)

        assert config.optimization_level == OptimizationLevel.O3
        assert config.enable_fusion is True
        assert config.compile_model is True
        assert config.use_fp16 is True

    def test_fp16_bf16_mutual_exclusion(self):
        """Test that FP16 and BF16 cannot be used simultaneously."""
        with pytest.raises(ValueError, match="Cannot use both FP16 and BF16"):
            VisionOptimizationConfig(use_fp16=True, use_bf16=True)

    def test_cuda_fallback(self):
        """Test CUDA device fallback to CPU when unavailable."""
        config = VisionOptimizationConfig(device="cuda")

        if not torch.cuda.is_available():
            assert config.device == "cpu"
        else:
            assert config.device == "cuda"


# ============================================================================
# Base Classes Tests
# ============================================================================


class TestBaseVisionOptimizer:
    """Test base vision optimizer."""

    def test_precision_string(self):
        """Test precision configuration string."""
        config_fp32 = VisionOptimizationConfig()
        optimizer = ResNetOptimizer(config_fp32)
        assert optimizer._get_precision_string() == "fp32"

        config_fp16 = VisionOptimizationConfig(use_fp16=True, device="cpu")
        optimizer = ResNetOptimizer(config_fp16)
        assert optimizer._get_precision_string() == "fp16"

    def test_optimization_summary(self):
        """Test optimization summary generation."""
        config = VisionOptimizationConfig(
            model_type=VisionModelType.RESNET,
            optimization_level=OptimizationLevel.O2,
            batch_size=32,
        )
        optimizer = ResNetOptimizer(config)

        summary = optimizer.get_optimization_summary()

        assert summary["model_type"] == "resnet"
        assert summary["optimization_level"] == "O2"
        assert summary["batch_size"] == 32
        assert "precision" in summary


class TestUtilityFunctions:
    """Test utility functions."""

    def test_count_parameters(self):
        """Test parameter counting."""
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220 params
            nn.ReLU(),
            nn.Linear(20, 5),   # 20*5 + 5 = 105 params
        )

        total, trainable = count_parameters(model)

        assert total == 325
        assert trainable == 325

    def test_count_parameters_frozen(self):
        """Test parameter counting with frozen parameters."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5),
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        total, trainable = count_parameters(model)

        assert total == 325
        assert trainable == 105  # Only second layer

    def test_estimate_model_memory(self):
        """Test model memory estimation."""
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )

        memory = estimate_model_memory(
            model,
            batch_size=1,
            input_size=(3, 224, 224),
            precision="fp32"
        )

        assert "parameter_memory_mb" in memory
        assert "activation_memory_mb" in memory
        assert "total_inference_mb" in memory
        assert memory["parameter_memory_mb"] > 0

    def test_estimate_model_memory_precision(self):
        """Test memory estimation with different precisions."""
        model = nn.Linear(1000, 1000)

        memory_fp32 = estimate_model_memory(model, precision="fp32")
        memory_fp16 = estimate_model_memory(model, precision="fp16")

        # FP16 should use half the memory of FP32
        assert memory_fp16["parameter_memory_mb"] < memory_fp32["parameter_memory_mb"]


# ============================================================================
# ResNet Tests
# ============================================================================


class TestResNetOptimizer:
    """Test ResNet optimizer."""

    @pytest.fixture
    def simple_resnet(self):
        """Create a simple ResNet-like model for testing."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1000),
        )

    def test_optimizer_creation(self):
        """Test ResNet optimizer creation."""
        config = VisionOptimizationConfig(model_type=VisionModelType.RESNET)
        optimizer = ResNetOptimizer(config)

        assert optimizer.config.model_type == VisionModelType.RESNET
        assert isinstance(optimizer, BaseVisionOptimizer)

    def test_optimize(self, simple_resnet):
        """Test model optimization."""
        config = VisionOptimizationConfig(
            model_type=VisionModelType.RESNET,
            optimization_level=OptimizationLevel.O1,
            device="cpu",
        )
        optimizer = ResNetOptimizer(config)

        model = optimizer.optimize(simple_resnet)

        assert model is not None
        assert len(optimizer.optimizations_applied) > 0

    def test_batch_inference(self, simple_resnet):
        """Test batch inference optimization."""
        config = VisionOptimizationConfig(
            model_type=VisionModelType.RESNET,
            batch_size=4,
            device="cpu",
        )
        optimizer = ResNetOptimizer(config)

        model = optimizer.optimize(simple_resnet)

        # Test inference
        images = torch.randn(8, 3, 224, 224)
        outputs = optimizer.optimize_batch_inference(model, images, batch_size=4)

        assert outputs.shape[0] == 8
        assert outputs.shape[1] == 1000

    def test_operator_fusion(self, simple_resnet):
        """Test operator fusion."""
        config = VisionOptimizationConfig(
            enable_fusion=True,
            device="cpu",
        )
        optimizer = ResNetOptimizer(config)

        optimizer.optimize(simple_resnet)

        assert "operator_fusion" in optimizer.optimizations_applied


class TestResNetBenchmark:
    """Test ResNet benchmark."""

    @pytest.fixture
    def simple_resnet_setup(self):
        """Create optimized ResNet for benchmarking."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1000),
        )

        config = VisionOptimizationConfig(device="cpu")
        optimizer = ResNetOptimizer(config)
        model = optimizer.optimize(model)

        return model, optimizer

    def test_benchmark_inference(self, simple_resnet_setup):
        """Test inference benchmarking."""
        model, optimizer = simple_resnet_setup
        benchmark = ResNetBenchmark(model, optimizer)

        results = benchmark.benchmark_inference(
            batch_size=2,
            num_iterations=5,
            warmup_iterations=2,
            image_size=224,
        )

        assert "total_time_seconds" in results
        assert "throughput_images_per_second" in results
        assert results["batch_size"] == 2

    def test_get_model_info(self, simple_resnet_setup):
        """Test model info retrieval."""
        model, optimizer = simple_resnet_setup
        benchmark = ResNetBenchmark(model, optimizer)

        info = benchmark.get_model_info()

        assert "total_parameters" in info
        assert "memory_estimate" in info
        assert info["total_parameters"] > 0


class TestResNetFactory:
    """Test ResNet factory functions."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA and torchvision"
    )
    def test_create_resnet50_optimized(self):
        """Test creating optimized ResNet-50."""
        try:
            import torchvision  # noqa: F401
        except ImportError:
            pytest.skip("torchvision not installed")

        model, optimizer = create_resnet50_optimized(
            optimization_level=OptimizationLevel.O1,
            batch_size=8,
            device="cpu",
        )

        assert model is not None
        assert isinstance(optimizer, ResNetOptimizer)


# ============================================================================
# ViT Tests
# ============================================================================


class TestViTOptimizer:
    """Test ViT optimizer."""

    @pytest.fixture
    def simple_vit(self):
        """Create a simple ViT-like model for testing."""
        class Transpose(nn.Module):
            def __init__(self, dim0, dim1):
                super().__init__()
                self.dim0 = dim0
                self.dim1 = dim1
            def forward(self, x):
                return x.transpose(self.dim0, self.dim1)

        return nn.Sequential(
            nn.Conv2d(3, 768, kernel_size=16, stride=16),
            nn.Flatten(2),
            Transpose(1, 2),
            nn.Linear(768, 1000),
        )

    def test_optimizer_creation(self):
        """Test ViT optimizer creation."""
        config = VisionOptimizationConfig(model_type=VisionModelType.VIT)
        optimizer = ViTOptimizer(config)

        assert optimizer.config.model_type == VisionModelType.VIT
        assert isinstance(optimizer, BaseVisionOptimizer)

    def test_optimize(self, simple_vit):
        """Test model optimization."""
        config = VisionOptimizationConfig(
            model_type=VisionModelType.VIT,
            optimization_level=OptimizationLevel.O1,
            device="cpu",
        )
        optimizer = ViTOptimizer(config)

        model = optimizer.optimize(simple_vit)

        assert model is not None
        assert len(optimizer.optimizations_applied) > 0

    def test_attention_slicing(self, simple_vit):
        """Test attention slicing application."""
        config = VisionOptimizationConfig(
            enable_attention_slicing=True,
            attention_slice_size=8,
            device="cpu",
        )
        optimizer = ViTOptimizer(config)

        optimizer.optimize(simple_vit)

        # Check if any optimization starts with "attention_slicing"
        # (may include status info like "attention_slicing(0 layers - no compatible attention found)")
        assert any(opt.startswith("attention_slicing") for opt in optimizer.optimizations_applied)


# ============================================================================
# Stable Diffusion Tests
# ============================================================================


class TestStableDiffusionOptimizer:
    """Test Stable Diffusion optimizer."""

    def test_optimizer_creation(self):
        """Test SD optimizer creation."""
        config = VisionOptimizationConfig(model_type=VisionModelType.STABLE_DIFFUSION)
        optimizer = StableDiffusionOptimizer(config)

        assert optimizer.config.model_type == VisionModelType.STABLE_DIFFUSION
        assert isinstance(optimizer, BaseVisionOptimizer)

    def test_config_validation(self):
        """Test configuration validation."""
        config = VisionOptimizationConfig(
            model_type=VisionModelType.STABLE_DIFFUSION,
            enable_attention_slicing=True,
            enable_vae_tiling=True,
            device="cpu",
        )

        optimizer = StableDiffusionOptimizer(config)

        assert optimizer.config.enable_attention_slicing is True
        assert optimizer.config.enable_vae_tiling is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_resnet_full_pipeline(self):
        """Test complete ResNet optimization pipeline."""
        # Create simple model
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        # Optimize
        config = VisionOptimizationConfig(
            model_type=VisionModelType.RESNET,
            optimization_level=OptimizationLevel.O2,
            batch_size=4,
            device="cpu",
            enable_fusion=True,
            channels_last=True,
        )
        optimizer = ResNetOptimizer(config)
        model = optimizer.optimize(model)

        # Run inference
        images = torch.randn(4, 3, 224, 224)
        outputs = optimizer.optimize_batch_inference(model, images)

        assert outputs.shape == (4, 10)
        assert len(optimizer.optimizations_applied) >= 2

    def test_optimization_level_progression(self):
        """Test that higher optimization levels apply more optimizations."""
        nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )

        results = {}

        for level in [OptimizationLevel.O0, OptimizationLevel.O1, OptimizationLevel.O2]:
            config = VisionOptimizationConfig.from_optimization_level(
                level,
                device="cpu"
            )
            optimizer = ResNetOptimizer(config)

            # Create fresh model for each test
            test_model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(16, 10),
            )

            _ = optimizer.optimize(test_model)
            results[level.value] = len(optimizer.optimizations_applied)

        # O0 should have fewer optimizations than O1
        assert results["O0"] <= results["O1"]
        # O1 should have fewer or equal optimizations than O2
        assert results["O1"] <= results["O2"]


# ============================================================================
# Export Tests
# ============================================================================


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_base_exports(self):
        """Test base module exports."""
        from torchbridge.models import vision

        assert hasattr(vision, "BaseVisionOptimizer")
        assert hasattr(vision, "VisionOptimizationConfig")
        assert hasattr(vision, "VisionModelType")
        assert hasattr(vision, "OptimizationLevel")
        assert hasattr(vision, "count_parameters")
        assert hasattr(vision, "estimate_model_memory")

    def test_resnet_exports(self):
        """Test ResNet module exports."""
        from torchbridge.models import vision

        assert hasattr(vision, "ResNetOptimizer")
        assert hasattr(vision, "ResNetBenchmark")
        assert hasattr(vision, "create_resnet_optimizer")
        assert hasattr(vision, "create_resnet50_optimized")
        assert hasattr(vision, "create_resnet152_optimized")

    def test_vit_exports(self):
        """Test ViT module exports."""
        from torchbridge.models import vision

        assert hasattr(vision, "ViTOptimizer")
        assert hasattr(vision, "ViTBenchmark")
        assert hasattr(vision, "create_vit_optimizer")
        assert hasattr(vision, "create_vit_base_optimized")
        assert hasattr(vision, "create_vit_large_optimized")

    def test_diffusion_exports(self):
        """Test Stable Diffusion module exports."""
        from torchbridge.models import vision

        assert hasattr(vision, "StableDiffusionOptimizer")
        assert hasattr(vision, "StableDiffusionBenchmark")
        assert hasattr(vision, "create_stable_diffusion_optimizer")
        assert hasattr(vision, "create_sd_1_5_optimized")
        assert hasattr(vision, "create_sd_2_1_optimized")
        assert hasattr(vision, "create_sdxl_optimized")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
