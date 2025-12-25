"""
Tests for auto-optimization and hardware detection.

Tests Stage 3A: Intelligent Optimization Selection
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from kernel_pytorch.core.hardware_detector import (
    HardwareDetector,
    HardwareProfile,
    HardwareType,
    OptimizationCapability,
    detect_hardware,
    get_optimal_backend,
)
from kernel_pytorch.core.config import (
    KernelPyTorchConfig,
    NVIDIAArchitecture,
    TPUVersion,
)
from kernel_pytorch.core.management import UnifiedManager, get_manager


# Test fixtures
@pytest.fixture
def simple_model():
    """Simple test model."""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return SimpleModel()


@pytest.fixture
def sample_inputs():
    """Sample inputs for testing."""
    return torch.randn(8, 128)


# Hardware Detection Tests
class TestHardwareDetector:
    """Test hardware detection functionality."""

    def test_detector_initialization(self):
        """Test hardware detector initialization."""
        detector = HardwareDetector()
        assert detector._cached_profile is None

    def test_detect_hardware(self):
        """Test basic hardware detection."""
        detector = HardwareDetector()
        profile = detector.detect()

        assert isinstance(profile, HardwareProfile)
        assert profile.hardware_type in [
            HardwareType.NVIDIA_GPU,
            HardwareType.TPU,
            HardwareType.CPU
        ]
        assert profile.device_name is not None
        assert profile.device_count >= 1

    def test_detect_hardware_caching(self):
        """Test that hardware detection is cached."""
        detector = HardwareDetector()

        profile1 = detector.detect()
        profile2 = detector.detect()

        # Should return same cached instance
        assert profile1 is profile2

    def test_detect_hardware_force_redetect(self):
        """Test force redetection."""
        detector = HardwareDetector()

        profile1 = detector.detect()
        profile2 = detector.detect(force_redetect=True)

        # Should be equal but not same instance
        assert profile1.hardware_type == profile2.hardware_type

    def test_cpu_fallback(self):
        """Test CPU fallback when no GPU/TPU available."""
        detector = HardwareDetector()

        with patch('torch.cuda.is_available', return_value=False):
            with patch('kernel_pytorch.core.hardware_detector.HardwareDetector._detect_tpu', return_value=None):
                profile = detector.detect(force_redetect=True)

                assert profile.hardware_type == HardwareType.CPU
                assert profile.device_name == "CPU"
                assert profile.capabilities == []

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_nvidia_gpu_detection(self):
        """Test NVIDIA GPU detection."""
        detector = HardwareDetector()
        profile = detector.detect(force_redetect=True)

        if profile.hardware_type == HardwareType.NVIDIA_GPU:
            assert profile.nvidia_architecture is not None
            assert profile.compute_capability is not None
            assert profile.total_memory_gb > 0
            assert OptimizationCapability.MIXED_PRECISION in profile.capabilities

    def test_get_optimal_backend_nvidia(self):
        """Test optimal backend selection for NVIDIA."""
        detector = HardwareDetector()

        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA A100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.AMPERE,
        )

        backend = detector.get_optimal_backend(profile)
        assert backend == 'nvidia'

    def test_get_optimal_backend_tpu(self):
        """Test optimal backend selection for TPU."""
        detector = HardwareDetector()

        profile = HardwareProfile(
            hardware_type=HardwareType.TPU,
            device_name="TPU v5p",
            device_count=8,
            tpu_version=TPUVersion.V5P,
        )

        backend = detector.get_optimal_backend(profile)
        assert backend == 'tpu'

    def test_get_optimal_backend_cpu(self):
        """Test optimal backend selection for CPU."""
        detector = HardwareDetector()

        profile = HardwareProfile(
            hardware_type=HardwareType.CPU,
            device_name="CPU",
            device_count=1,
        )

        backend = detector.get_optimal_backend(profile)
        assert backend == 'cpu'

    def test_get_recommended_optimization_aggressive(self):
        """Test aggressive optimization recommendation."""
        detector = HardwareDetector()

        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA H100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.HOPPER,
            capabilities=[OptimizationCapability.FP8_TRAINING]
        )

        level = detector.get_recommended_optimization_level(profile)
        assert level == 'aggressive'

    def test_get_recommended_optimization_balanced(self):
        """Test balanced optimization recommendation."""
        detector = HardwareDetector()

        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA A100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.AMPERE,
            capabilities=[OptimizationCapability.TENSOR_CORES]
        )

        level = detector.get_recommended_optimization_level(profile)
        assert level in ['balanced', 'conservative']

    def test_get_recommended_optimization_conservative(self):
        """Test conservative optimization recommendation."""
        detector = HardwareDetector()

        profile = HardwareProfile(
            hardware_type=HardwareType.CPU,
            device_name="CPU",
            device_count=1,
        )

        level = detector.get_recommended_optimization_level(profile)
        assert level == 'conservative'


# Hardware Profile Tests
class TestHardwareProfile:
    """Test hardware profile functionality."""

    def test_profile_initialization(self):
        """Test hardware profile initialization."""
        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="Test GPU",
            device_count=1,
        )

        assert profile.hardware_type == HardwareType.NVIDIA_GPU
        assert profile.device_name == "Test GPU"
        assert profile.device_count == 1
        assert profile.capabilities == []

    def test_profile_capabilities(self):
        """Test capability checking."""
        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="Test GPU",
            device_count=1,
            capabilities=[
                OptimizationCapability.FP8_TRAINING,
                OptimizationCapability.FLASH_ATTENTION_3,
            ]
        )

        assert profile.has_capability(OptimizationCapability.FP8_TRAINING)
        assert profile.has_capability(OptimizationCapability.FLASH_ATTENTION_3)
        assert not profile.has_capability(OptimizationCapability.XLA_COMPILATION)

    def test_is_nvidia_h100_or_better(self):
        """Test H100+ detection."""
        profile_h100 = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA H100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.HOPPER,
        )

        profile_blackwell = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA B100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.BLACKWELL,
        )

        profile_ampere = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA A100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.AMPERE,
        )

        assert profile_h100.is_nvidia_h100_or_better()
        assert profile_blackwell.is_nvidia_h100_or_better()
        assert not profile_ampere.is_nvidia_h100_or_better()

    def test_is_high_end_tpu(self):
        """Test high-end TPU detection."""
        profile_v7 = HardwareProfile(
            hardware_type=HardwareType.TPU,
            device_name="TPU v7",
            device_count=8,
            tpu_version=TPUVersion.V7,
        )

        profile_v4 = HardwareProfile(
            hardware_type=HardwareType.TPU,
            device_name="TPU v4",
            device_count=8,
            tpu_version=TPUVersion.V4,
        )

        assert profile_v7.is_high_end_tpu()
        assert not profile_v4.is_high_end_tpu()

    def test_supports_advanced_optimization(self):
        """Test advanced optimization support detection."""
        profile_advanced = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA H100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.HOPPER,
            capabilities=[OptimizationCapability.FP8_TRAINING]
        )

        profile_basic = HardwareProfile(
            hardware_type=HardwareType.CPU,
            device_name="CPU",
            device_count=1,
        )

        assert profile_advanced.supports_advanced_optimization()
        assert not profile_basic.supports_advanced_optimization()


# Auto-Optimization Tests
class TestAutoOptimization:
    """Test auto-optimization functionality in UnifiedManager."""

    def test_manager_has_auto_optimize(self):
        """Test that manager has auto_optimize method."""
        manager = get_manager()
        assert hasattr(manager, 'auto_optimize')
        assert callable(manager.auto_optimize)

    def test_manager_has_hardware_detector(self):
        """Test that manager has hardware detector."""
        manager = get_manager()
        assert hasattr(manager, 'hardware_detector')
        assert isinstance(manager.hardware_detector, HardwareDetector)

    def test_auto_optimize_basic(self, simple_model, sample_inputs):
        """Test basic auto-optimization."""
        manager = get_manager()

        optimized_model = manager.auto_optimize(
            simple_model,
            sample_inputs=sample_inputs
        )

        # Should return a model
        assert optimized_model is not None

        # Should be able to run inference
        with torch.no_grad():
            output = optimized_model(sample_inputs)
            assert output.shape == (8, 128)

    def test_auto_optimize_caches_hardware_profile(self, simple_model):
        """Test that hardware profile is cached."""
        manager = UnifiedManager()  # New instance

        # First call should detect hardware
        manager.auto_optimize(simple_model)
        profile1 = manager._hardware_profile

        # Second call should use cached profile
        manager.auto_optimize(simple_model)
        profile2 = manager._hardware_profile

        assert profile1 is profile2

    def test_get_hardware_profile(self):
        """Test getting hardware profile."""
        manager = get_manager()
        profile = manager.get_hardware_profile()

        assert isinstance(profile, HardwareProfile)
        assert profile.hardware_type in [
            HardwareType.NVIDIA_GPU,
            HardwareType.TPU,
            HardwareType.CPU
        ]

    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        manager = get_manager()
        recommendations = manager.get_optimization_recommendations()

        assert isinstance(recommendations, dict)
        assert 'hardware_type' in recommendations
        assert 'backend' in recommendations
        assert 'optimization_level' in recommendations
        assert 'capabilities' in recommendations

    def test_auto_optimize_with_custom_optimization_level(self, simple_model):
        """Test auto-optimization with custom level."""
        manager = get_manager()

        optimized_model = manager.auto_optimize(
            simple_model,
            optimization_level='conservative'
        )

        assert optimized_model is not None

    def test_auto_optimize_for_inference(self, simple_model):
        """Test auto-optimization for inference."""
        manager = get_manager()

        optimized_model = manager.auto_optimize(
            simple_model,
            for_inference=True
        )

        assert optimized_model is not None

        # Should be in eval mode
        assert not optimized_model.training


# Convenience Function Tests
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_detect_hardware_function(self):
        """Test detect_hardware convenience function."""
        profile = detect_hardware()

        assert isinstance(profile, HardwareProfile)
        assert profile.hardware_type is not None

    def test_get_optimal_backend_function(self):
        """Test get_optimal_backend convenience function."""
        backend = get_optimal_backend()

        assert isinstance(backend, str)
        assert backend in ['nvidia', 'tpu', 'cpu']


# Integration Tests
class TestAutoOptimizationIntegration:
    """Integration tests for auto-optimization."""

    def test_end_to_end_auto_optimization(self, simple_model, sample_inputs):
        """Test complete end-to-end auto-optimization."""
        # Create fresh manager
        manager = UnifiedManager()

        # Auto-optimize
        optimized_model = manager.auto_optimize(
            simple_model,
            sample_inputs=sample_inputs
        )

        # Verify model works
        with torch.no_grad():
            output = optimized_model(sample_inputs)
            assert output.shape == (8, 128)

        # Get recommendations
        recommendations = manager.get_optimization_recommendations()
        assert recommendations['backend'] in ['nvidia', 'tpu', 'cpu']

    def test_auto_optimization_consistency(self, simple_model):
        """Test that auto-optimization is consistent."""
        manager = get_manager()

        # Optimize twice
        model1 = manager.auto_optimize(simple_model)
        model2 = manager.auto_optimize(simple_model)

        # Should use same backend
        profile = manager.get_hardware_profile()
        backend = manager.hardware_detector.get_optimal_backend(profile)

        assert backend in ['nvidia', 'tpu', 'cpu']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
