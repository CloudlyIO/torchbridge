"""
AMD Backend Tests

Comprehensive test suite for AMD ROCm backend implementation.
Tests cover configuration, backend operations, optimization,
and operator fusion.

Note: Tests are designed to work without actual AMD hardware by using
mocks and CPU fallbacks where appropriate.
"""


import pytest
import torch

# Import AMD backend components
from torchbridge.core.config import AMDArchitecture, AMDConfig


class TestAMDConfig:
    """Tests for AMD configuration."""

    def test_default_config_creation(self):
        """Test default AMD configuration creation."""
        config = AMDConfig()

        # Architecture gets auto-detected in __post_init__, so it may not be AUTO
        assert config.architecture in list(AMDArchitecture)
        assert config.device_id == 0
        assert config.optimization_level in ["conservative", "balanced", "aggressive"]
        assert isinstance(config.enable_matrix_cores, bool)
        assert isinstance(config.enable_mixed_precision, bool)

    def test_cdna2_architecture(self):
        """Test CDNA2 architecture configuration."""
        config = AMDConfig(architecture=AMDArchitecture.CDNA2)

        assert config.architecture == AMDArchitecture.CDNA2
        assert config.architecture.value == "cdna2"

    def test_cdna3_architecture(self):
        """Test CDNA3 architecture configuration."""
        config = AMDConfig(architecture=AMDArchitecture.CDNA3)

        assert config.architecture == AMDArchitecture.CDNA3
        assert config.architecture.value == "cdna3"

    def test_rdna_architectures(self):
        """Test RDNA architecture configurations."""
        for arch in [AMDArchitecture.RDNA2, AMDArchitecture.RDNA3]:
            config = AMDConfig(architecture=arch)
            assert config.architecture == arch

    def test_optimization_levels(self):
        """Test different optimization levels."""
        for level in ["conservative", "balanced", "aggressive"]:
            config = AMDConfig(optimization_level=level)
            assert config.optimization_level == level

    def test_precision_settings(self):
        """Test precision configuration."""
        for precision in ["fp32", "fp16", "bf16"]:
            config = AMDConfig(default_precision=precision)
            assert config.default_precision == precision

    def test_memory_settings(self):
        """Test memory configuration."""
        config = AMDConfig(
            memory_pool_size_gb=16.0,
            enable_memory_pooling=True,
        )

        assert config.memory_pool_size_gb == 16.0
        assert config.enable_memory_pooling is True

    def test_matrix_core_settings(self):
        """Test Matrix Core configuration."""
        # CDNA2/CDNA3 architectures enable matrix cores by default
        config = AMDConfig(architecture=AMDArchitecture.CDNA2)
        assert config.enable_matrix_cores is True

        config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        assert config.enable_matrix_cores is True

        # Consumer GPUs (RDNA) disable matrix cores
        config = AMDConfig(architecture=AMDArchitecture.RDNA3)
        assert config.enable_matrix_cores is False


class TestAMDArchitecture:
    """Tests for AMD architecture enum."""

    def test_all_architectures_exist(self):
        """Test all expected architectures exist."""
        expected = ["AUTO", "CDNA", "CDNA2", "CDNA3", "CDNA4", "RDNA2", "RDNA3"]
        for arch_name in expected:
            assert hasattr(AMDArchitecture, arch_name)

    def test_architecture_values(self):
        """Test architecture enum values."""
        assert AMDArchitecture.AUTO.value == "auto"
        assert AMDArchitecture.CDNA2.value == "cdna2"
        assert AMDArchitecture.CDNA3.value == "cdna3"
        assert AMDArchitecture.CDNA4.value == "cdna4"


class TestAMDExceptions:
    """Tests for AMD exception hierarchy."""

    def test_amd_backend_error(self):
        """Test AMDBackendError exception."""
        from torchbridge.backends.amd.amd_exceptions import AMDBackendError

        error = AMDBackendError("Test error")
        assert "Test error" in str(error)

    def test_rocm_not_available_error(self):
        """Test ROCmNotAvailableError exception."""
        from torchbridge.backends.amd.amd_exceptions import ROCmNotAvailableError

        error = ROCmNotAvailableError("ROCm not found")
        assert "ROCm" in str(error)

    def test_hip_compilation_error(self):
        """Test HIPCompilationError exception."""
        from torchbridge.backends.amd.amd_exceptions import HIPCompilationError

        error = HIPCompilationError("test_kernel", "Compilation failed")
        assert "test_kernel" in str(error)

    def test_rocm_memory_error(self):
        """Test ROCmMemoryError exception."""
        from torchbridge.backends.amd.amd_exceptions import ROCmMemoryError

        error = ROCmMemoryError("allocation", required_mb=1000, available_mb=500)
        assert "1000" in str(error) or "allocation" in str(error)

    def test_matrix_core_error(self):
        """Test MatrixCoreError exception."""
        from torchbridge.backends.amd.amd_exceptions import MatrixCoreError

        error = MatrixCoreError("enable", "cdna3", "Not supported")
        assert "Matrix" in str(error) or "cdna3" in str(error)

    def test_amd_optimization_error(self):
        """Test AMDOptimizationError exception."""
        from torchbridge.backends.amd.amd_exceptions import AMDOptimizationError

        error = AMDOptimizationError("balanced", "Optimization failed")
        assert "balanced" in str(error) or "Optimization" in str(error)


class TestAMDAdapter:
    """Tests for AMD adapter."""

    def test_adapter_creation(self):
        """Test adapter creation."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter

        config = AMDConfig()
        optimizer = AMDAdapter(config)

        assert optimizer.config == config

    def test_optimization_levels(self):
        """Test different optimization levels."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter

        for level in ["conservative", "balanced", "aggressive"]:
            config = AMDConfig(optimization_level=level)
            optimizer = AMDAdapter(config)

            model = torch.nn.Linear(64, 32)
            optimized = optimizer.optimize(model)

            assert optimized is not None

    def test_optimization_result(self):
        """Test optimization result structure."""
        from torchbridge.backends.amd.amd_adapter import (
            AMDAdapter,
        )

        config = AMDConfig()
        optimizer = AMDAdapter(config)

        model = torch.nn.Linear(64, 32)
        optimizer.optimize(model)

        summary = optimizer.get_optimization_summary()
        assert "optimization_level" in summary
        assert "architecture" in summary

    def test_optimization_with_conv_model(self):
        """Test optimization with convolutional model."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter

        config = AMDConfig()
        optimizer = AMDAdapter(config)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        optimized = optimizer.optimize(model)
        assert optimized is not None


class TestAMDBackendIntegration:
    """Integration tests for AMD backend."""

    def test_config_integration(self):
        """Test configuration integration with adapter."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter

        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            optimization_level="aggressive",
            enable_matrix_cores=True,
            enable_mixed_precision=True,
        )

        optimizer = AMDAdapter(config)
        summary = optimizer.get_optimization_summary()

        assert summary["architecture"] == "cdna3"
        assert summary["matrix_cores_enabled"] is True
        assert summary["mixed_precision"] is True


class TestAMDOperatorFusion:
    """Tests for AMD operator fusion."""

    def test_conv_bn_fusion_pattern_detection(self):
        """Test Conv+BatchNorm fusion pattern detection."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter

        config = AMDConfig(enable_operator_fusion=True)
        optimizer = AMDAdapter(config)

        # Create model with Conv+BN pattern
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )
        model.eval()  # Required for fusion

        optimized = optimizer.optimize(model, level="conservative")
        assert optimized is not None

    def test_linear_gelu_fusion_pattern(self):
        """Test Linear+GELU fusion pattern detection."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter

        config = AMDConfig(enable_operator_fusion=True)
        optimizer = AMDAdapter(config)

        model = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
        )

        optimized = optimizer.optimize(model, level="balanced")
        summary = optimizer.get_optimization_summary()

        assert "fused_operations" in summary
        assert optimized is not None

    def test_aggressive_fusion_patterns(self):
        """Test aggressive fusion patterns."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter

        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            enable_operator_fusion=True
        )
        optimizer = AMDAdapter(config)

        # Transformer-like model
        model = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.LayerNorm(256),
            torch.nn.Linear(256, 256),
        )

        optimized = optimizer.optimize(model, level="aggressive")
        assert optimized is not None

    def test_memory_layout_optimization(self):
        """Test memory layout optimization for HBM."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter

        config = AMDConfig()
        optimizer = AMDAdapter(config)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
        )

        optimized = optimizer.optimize(model, level="conservative")
        optimizer.get_optimization_summary()

        assert optimized is not None


class TestAMDBackendEnhanced:
    """Enhanced tests for AMD backend."""

    def test_backend_with_all_architectures(self):
        """Test backend initialization with all architectures."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        for arch in [AMDArchitecture.CDNA2, AMDArchitecture.CDNA3,
                     AMDArchitecture.RDNA2, AMDArchitecture.RDNA3]:
            config = AMDConfig(architecture=arch)
            backend = AMDBackend(config)

            # Should initialize (with CPU fallback if no AMD GPU)
            assert backend is not None
            assert backend.device is not None

    def test_backend_get_device_info(self):
        """Test unified device info method."""
        from torchbridge.backends import DeviceInfo
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig()
        backend = AMDBackend(config)

        info = backend.get_device_info()

        assert isinstance(info, DeviceInfo)
        assert info.backend == "amd"
        assert hasattr(info, 'device_type')
        assert hasattr(info, 'is_available')

    def test_backend_optimize_for_inference(self):
        """Test inference optimization."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig()
        backend = AMDBackend(config)

        model = torch.nn.Linear(64, 32)
        optimized = backend.optimize_for_inference(model)

        assert optimized is not None
        assert not any(p.requires_grad for p in optimized.parameters())

    def test_backend_optimize_for_training(self):
        """Test training optimization."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig()
        backend = AMDBackend(config)

        model = torch.nn.Linear(64, 32)
        optimized = backend.optimize_for_training(model)

        assert optimized is not None
        assert optimized.training

    def test_backend_with_optimizer(self):
        """Test training optimization with optimizer."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig()
        backend = AMDBackend(config)

        model = torch.nn.Linear(64, 32)
        optimizer = torch.optim.Adam(model.parameters())

        result = backend.optimize_for_training(model, optimizer=optimizer)

        assert isinstance(result, tuple)
        assert len(result) == 2


class TestAMDIntegrationV049:
    """Integration tests for AMD improvements."""

    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        from torchbridge.backends.amd.amd_adapter import AMDAdapter
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            optimization_level="aggressive",
            enable_operator_fusion=True,
            enable_matrix_cores=True,
        )

        backend = AMDBackend(config)
        optimizer = AMDAdapter(config)

        model = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.GELU(),
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, 256),
        )

        # Prepare model
        prepared = backend.prepare_model(model)
        assert prepared is not None

        # Optimize model
        optimized = optimizer.optimize(prepared)
        assert optimized is not None

        summary = optimizer.get_optimization_summary()
        assert summary['architecture'] == 'cdna3'
        assert summary['matrix_cores_enabled'] is True


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
