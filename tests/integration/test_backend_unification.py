"""
Tests for Backend Unification (v0.5.3)

This module tests the unified backend interface including:
- BaseBackend abstract class
- BaseOptimizer abstract class
- BackendFactory
- DeviceInfo, OptimizationResult dataclasses
- OptimizationLevel enum

All backends (NVIDIA, AMD, TPU, Intel) should inherit from the base classes
and provide a consistent API.
"""


import pytest
import torch
import torch.nn as nn

from torchbridge.backends import (
    # Exceptions
    BackendFactory,
    BackendType,
    # Base classes
    BaseBackend,
    BaseOptimizer,
    CPUBackend,
    CPUOptimizer,
    # Dataclasses
    DeviceInfo,
    KernelConfig,
    OptimizationLevel,
    OptimizationResult,
    OptimizationStrategy,
    detect_best_backend,
    get_backend,
    get_optimizer,
    list_available_backends,
)

# =============================================================================
# Test Models
# =============================================================================

class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, in_features=128, out_features=64):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class ConvModel(nn.Module):
    """Simple conv model for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# =============================================================================
# OptimizationLevel Tests
# =============================================================================

class TestOptimizationLevel:
    """Tests for OptimizationLevel enum."""

    def test_optimization_levels_exist(self):
        """Test that all optimization levels exist."""
        assert OptimizationLevel.O0 is not None
        assert OptimizationLevel.O1 is not None
        assert OptimizationLevel.O2 is not None
        assert OptimizationLevel.O3 is not None

    def test_optimization_level_values(self):
        """Test optimization level values."""
        assert OptimizationLevel.O0.value == "O0"
        assert OptimizationLevel.O1.value == "O1"
        assert OptimizationLevel.O2.value == "O2"
        assert OptimizationLevel.O3.value == "O3"

    def test_from_string_direct_match(self):
        """Test OptimizationLevel.from_string with direct match."""
        assert OptimizationLevel.from_string("O0") == OptimizationLevel.O0
        assert OptimizationLevel.from_string("O1") == OptimizationLevel.O1
        assert OptimizationLevel.from_string("O2") == OptimizationLevel.O2
        assert OptimizationLevel.from_string("O3") == OptimizationLevel.O3

    def test_from_string_aliases(self):
        """Test OptimizationLevel.from_string with aliases."""
        assert OptimizationLevel.from_string("conservative") == OptimizationLevel.O1
        assert OptimizationLevel.from_string("balanced") == OptimizationLevel.O2
        assert OptimizationLevel.from_string("aggressive") == OptimizationLevel.O3
        assert OptimizationLevel.from_string("debug") == OptimizationLevel.O0

    def test_from_string_case_insensitive(self):
        """Test OptimizationLevel.from_string is case insensitive."""
        assert OptimizationLevel.from_string("o0") == OptimizationLevel.O0
        assert OptimizationLevel.from_string("CONSERVATIVE") == OptimizationLevel.O1
        assert OptimizationLevel.from_string("Balanced") == OptimizationLevel.O2

    def test_from_string_unknown_defaults_to_o2(self):
        """Test unknown level defaults to O2."""
        assert OptimizationLevel.from_string("unknown") == OptimizationLevel.O2


# =============================================================================
# DeviceInfo Tests
# =============================================================================

class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_device_info_creation(self):
        """Test creating a DeviceInfo object."""
        info = DeviceInfo(
            backend="test",
            device_type="cpu",
            device_id=0,
            device_name="Test CPU",
            total_memory_bytes=16 * 1024**3,
            is_available=True
        )

        assert info.backend == "test"
        assert info.device_type == "cpu"
        assert info.device_id == 0
        assert info.device_name == "Test CPU"
        assert info.is_available is True

    def test_device_info_memory_properties(self):
        """Test DeviceInfo memory properties."""
        info = DeviceInfo(
            backend="test",
            device_type="cpu",
            device_id=0,
            device_name="Test",
            total_memory_bytes=16 * 1024**3
        )

        assert info.total_memory_gb == 16.0
        assert info.total_memory_mb == 16 * 1024

    def test_device_info_to_dict(self):
        """Test DeviceInfo.to_dict()."""
        info = DeviceInfo(
            backend="test",
            device_type="cuda:0",
            device_id=0,
            device_name="Test GPU",
            compute_capability="8.0",
            total_memory_bytes=24 * 1024**3,
            is_available=True
        )

        d = info.to_dict()
        assert d['backend'] == "test"
        assert d['device_type'] == "cuda:0"
        assert d['compute_capability'] == "8.0"
        assert d['total_memory_gb'] == 24.0


# =============================================================================
# OptimizationResult Tests
# =============================================================================

class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_optimization_result_creation(self):
        """Test creating an OptimizationResult."""
        model = SimpleModel()
        result = OptimizationResult(
            success=True,
            model=model,
            level=OptimizationLevel.O2,
            optimizations_applied=['opt1', 'opt2']
        )

        assert result.success is True
        assert result.model is model
        assert result.level == OptimizationLevel.O2
        assert len(result.optimizations_applied) == 2

    def test_optimization_result_level_conversion(self):
        """Test OptimizationResult converts string level to enum."""
        model = SimpleModel()
        result = OptimizationResult(
            success=True,
            model=model,
            level="O2"  # String should be converted
        )

        assert result.level == OptimizationLevel.O2

    def test_optimization_result_to_dict(self):
        """Test OptimizationResult.to_dict()."""
        model = SimpleModel()
        result = OptimizationResult(
            success=True,
            model=model,
            level=OptimizationLevel.O3,
            optimizations_applied=['opt1'],
            warnings=['warning1'],
            metrics={'time': 1.5}
        )

        d = result.to_dict()
        assert d['success'] is True
        assert d['level'] == "O3"
        assert 'opt1' in d['optimizations_applied']


# =============================================================================
# CPUBackend Tests
# =============================================================================

class TestCPUBackend:
    """Tests for CPUBackend (concrete BaseBackend implementation)."""

    def test_cpu_backend_creation(self):
        """Test creating a CPUBackend."""
        backend = CPUBackend()

        assert backend is not None
        assert backend.BACKEND_NAME == "cpu"
        assert backend.is_available is True

    def test_cpu_backend_device(self):
        """Test CPUBackend device property."""
        backend = CPUBackend()

        assert backend.device == torch.device("cpu")

    def test_cpu_backend_device_info(self):
        """Test CPUBackend.get_device_info()."""
        backend = CPUBackend()
        info = backend.get_device_info()

        assert isinstance(info, DeviceInfo)
        assert info.backend == "cpu"
        assert info.is_available is True

    def test_cpu_backend_prepare_model(self):
        """Test CPUBackend.prepare_model()."""
        backend = CPUBackend()
        model = SimpleModel()

        prepared = backend.prepare_model(model)

        assert prepared is not None
        assert next(prepared.parameters()).device == torch.device("cpu")

    def test_cpu_backend_optimize_for_inference(self):
        """Test CPUBackend.optimize_for_inference()."""
        backend = CPUBackend()
        model = SimpleModel()

        optimized = backend.optimize_for_inference(model)

        assert optimized is not None
        assert not optimized.training  # Should be in eval mode

    def test_cpu_backend_optimize_for_training(self):
        """Test CPUBackend.optimize_for_training()."""
        backend = CPUBackend()
        model = SimpleModel()

        optimized = backend.optimize_for_training(model)

        assert optimized is not None
        assert optimized.training  # Should be in train mode

    def test_cpu_backend_with_optimizer(self):
        """Test CPUBackend.optimize_for_training with optimizer."""
        backend = CPUBackend()
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        result = backend.optimize_for_training(model, optimizer)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_cpu_backend_to_device(self):
        """Test CPUBackend.to_device()."""
        backend = CPUBackend()
        tensor = torch.randn(10)

        result = backend.to_device(tensor)

        assert result.device == torch.device("cpu")

    def test_cpu_backend_context_manager(self):
        """Test CPUBackend as context manager."""
        with CPUBackend() as backend:
            assert backend.is_available is True


# =============================================================================
# CPUOptimizer Tests
# =============================================================================

class TestCPUOptimizer:
    """Tests for CPUOptimizer."""

    def test_cpu_optimizer_creation(self):
        """Test creating a CPUOptimizer."""
        optimizer = CPUOptimizer()

        assert optimizer is not None
        assert optimizer.OPTIMIZER_NAME == "cpu"

    def test_cpu_optimizer_optimize(self):
        """Test CPUOptimizer.optimize()."""
        optimizer = CPUOptimizer()
        model = SimpleModel()

        optimized_model, result = optimizer.optimize(model, level=OptimizationLevel.O2)

        assert optimized_model is not None
        assert isinstance(result, OptimizationResult)
        assert result.success is True

    def test_cpu_optimizer_optimization_levels(self):
        """Test different optimization levels."""
        optimizer = CPUOptimizer()
        model = SimpleModel()

        for level in [OptimizationLevel.O0, OptimizationLevel.O1, OptimizationLevel.O2, OptimizationLevel.O3]:
            _, result = optimizer.optimize(model, level=level)
            assert result.success is True
            assert result.level == level

    def test_cpu_optimizer_get_available_strategies(self):
        """Test CPUOptimizer.get_available_strategies()."""
        optimizer = CPUOptimizer()
        strategies = optimizer.get_available_strategies()

        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert all(isinstance(s, OptimizationStrategy) for s in strategies)


# =============================================================================
# BackendFactory Tests
# =============================================================================

class TestBackendFactory:
    """Tests for BackendFactory."""

    def test_backend_type_enum(self):
        """Test BackendType enum values."""
        assert BackendType.AUTO.value == "auto"
        assert BackendType.NVIDIA.value == "nvidia"
        assert BackendType.AMD.value == "amd"
        assert BackendType.TPU.value == "tpu"
        assert BackendType.INTEL.value == "intel"
        assert BackendType.CPU.value == "cpu"

    def test_backend_type_from_string(self):
        """Test BackendType.from_string()."""
        assert BackendType.from_string("nvidia") == BackendType.NVIDIA
        assert BackendType.from_string("cuda") == BackendType.NVIDIA
        assert BackendType.from_string("amd") == BackendType.AMD
        assert BackendType.from_string("rocm") == BackendType.AMD
        assert BackendType.from_string("tpu") == BackendType.TPU
        assert BackendType.from_string("xla") == BackendType.TPU
        assert BackendType.from_string("intel") == BackendType.INTEL
        assert BackendType.from_string("xpu") == BackendType.INTEL

    def test_factory_create_cpu_backend(self):
        """Test BackendFactory.create() with CPU."""
        backend = BackendFactory.create(BackendType.CPU)

        assert isinstance(backend, BaseBackend)
        assert isinstance(backend, CPUBackend)

    def test_factory_create_auto(self):
        """Test BackendFactory.create() with AUTO."""
        backend = BackendFactory.create(BackendType.AUTO)

        assert isinstance(backend, BaseBackend)
        # Should return some backend (CPU if no accelerators)

    def test_factory_create_with_string(self):
        """Test BackendFactory.create() with string backend type."""
        backend = BackendFactory.create("cpu")

        assert isinstance(backend, CPUBackend)

    def test_factory_get_available_backends(self):
        """Test BackendFactory.get_available_backends()."""
        available = BackendFactory.get_available_backends()

        assert isinstance(available, list)
        assert BackendType.CPU in available  # CPU is always available

    def test_factory_get_backend_info(self):
        """Test BackendFactory.get_backend_info()."""
        info = BackendFactory.get_backend_info(BackendType.CPU)

        assert isinstance(info, dict)
        assert info['type'] == 'cpu'
        assert info['available'] is True

    def test_factory_get_all_backend_info(self):
        """Test BackendFactory.get_all_backend_info()."""
        all_info = BackendFactory.get_all_backend_info()

        assert isinstance(all_info, dict)
        assert 'cpu' in all_info
        assert 'nvidia' in all_info

    def test_get_backend_function(self):
        """Test get_backend() convenience function."""
        backend = get_backend()

        assert isinstance(backend, BaseBackend)

    def test_get_optimizer_function(self):
        """Test get_optimizer() convenience function."""
        optimizer = get_optimizer()

        assert isinstance(optimizer, BaseOptimizer)

    def test_detect_best_backend(self):
        """Test detect_best_backend() function."""
        best = detect_best_backend()

        assert isinstance(best, BackendType)

    def test_list_available_backends(self):
        """Test list_available_backends() function."""
        available = list_available_backends()

        assert isinstance(available, list)
        assert 'cpu' in available


# =============================================================================
# Backend Inheritance Tests
# =============================================================================

class TestBackendInheritance:
    """Test that all backends properly inherit from BaseBackend."""

    def test_nvidia_backend_inherits_base(self):
        """Test NVIDIABackend inherits from BaseBackend."""
        from torchbridge.backends.nvidia import NVIDIABackend

        backend = NVIDIABackend()
        assert isinstance(backend, BaseBackend)
        assert backend.BACKEND_NAME == "nvidia"

    def test_amd_backend_inherits_base(self):
        """Test AMDBackend inherits from BaseBackend."""
        from torchbridge.backends.amd import AMDBackend

        backend = AMDBackend()
        assert isinstance(backend, BaseBackend)
        assert backend.BACKEND_NAME == "amd"

    def test_tpu_backend_inherits_base(self):
        """Test TPUBackend inherits from BaseBackend."""
        from torchbridge.backends.tpu import TPUBackend

        backend = TPUBackend()
        assert isinstance(backend, BaseBackend)
        assert backend.BACKEND_NAME == "tpu"

    def test_intel_backend_inherits_base(self):
        """Test IntelBackend inherits from BaseBackend."""
        from torchbridge.backends.intel import IntelBackend

        backend = IntelBackend()
        assert isinstance(backend, BaseBackend)
        assert backend.BACKEND_NAME == "intel"


# =============================================================================
# Unified Interface Tests
# =============================================================================

class TestUnifiedInterface:
    """Test that all backends have a consistent interface."""

    @pytest.fixture
    def backends(self):
        """Create instances of all backends."""
        from torchbridge.backends.amd import AMDBackend
        from torchbridge.backends.intel import IntelBackend
        from torchbridge.backends.nvidia import NVIDIABackend
        from torchbridge.backends.tpu import TPUBackend

        return [
            CPUBackend(),
            NVIDIABackend(),
            AMDBackend(),
            TPUBackend(),
            IntelBackend(),
        ]

    def test_all_have_device_property(self, backends):
        """Test all backends have device property."""
        for backend in backends:
            assert hasattr(backend, 'device')
            assert isinstance(backend.device, torch.device)

    def test_all_have_is_available(self, backends):
        """Test all backends have is_available property."""
        for backend in backends:
            assert hasattr(backend, 'is_available')
            assert isinstance(backend.is_available, bool)

    def test_all_have_prepare_model(self, backends):
        """Test all backends have prepare_model method."""
        model = SimpleModel()

        for backend in backends:
            assert hasattr(backend, 'prepare_model')
            prepared = backend.prepare_model(model)
            assert prepared is not None

    def test_all_have_get_device_info(self, backends):
        """Test all backends have get_device_info method."""
        for backend in backends:
            assert hasattr(backend, 'get_device_info')
            info = backend.get_device_info()
            assert isinstance(info, DeviceInfo)

    def test_all_have_optimize_for_inference(self, backends):
        """Test all backends have optimize_for_inference method."""
        SimpleModel()

        for backend in backends:
            assert hasattr(backend, 'optimize_for_inference')
            # Just verify the method exists and is callable

    def test_all_have_optimize_for_training(self, backends):
        """Test all backends have optimize_for_training method."""
        SimpleModel()

        for backend in backends:
            assert hasattr(backend, 'optimize_for_training')
            # Just verify the method exists and is callable

    def test_all_have_synchronize(self, backends):
        """Test all backends have synchronize method."""
        for backend in backends:
            assert hasattr(backend, 'synchronize')
            # Should not raise
            backend.synchronize()

    def test_all_have_empty_cache(self, backends):
        """Test all backends have empty_cache method."""
        for backend in backends:
            assert hasattr(backend, 'empty_cache')
            # Should not raise
            backend.empty_cache()


# =============================================================================
# Optimization Strategy Tests
# =============================================================================

class TestOptimizationStrategy:
    """Tests for OptimizationStrategy dataclass."""

    def test_strategy_creation(self):
        """Test creating an OptimizationStrategy."""
        strategy = OptimizationStrategy(
            name='test_opt',
            description='Test optimization',
            applicable_levels=[OptimizationLevel.O2, OptimizationLevel.O3],
            speedup_estimate=1.5
        )

        assert strategy.name == 'test_opt'
        assert OptimizationLevel.O2 in strategy.applicable_levels

    def test_strategy_is_applicable(self):
        """Test OptimizationStrategy.is_applicable()."""
        strategy = OptimizationStrategy(
            name='test',
            description='Test',
            applicable_levels=[OptimizationLevel.O2, OptimizationLevel.O3]
        )

        assert strategy.is_applicable(OptimizationLevel.O2) is True
        assert strategy.is_applicable(OptimizationLevel.O3) is True
        assert strategy.is_applicable(OptimizationLevel.O0) is False
        assert strategy.is_applicable(OptimizationLevel.O1) is False


# =============================================================================
# KernelConfig Tests
# =============================================================================

class TestKernelConfig:
    """Tests for KernelConfig dataclass."""

    def test_kernel_config_creation(self):
        """Test creating a KernelConfig."""
        config = KernelConfig(
            algorithm='auto',
            tile_sizes=(32, 32, 32),
            num_warps=4
        )

        assert config.algorithm == 'auto'
        assert config.tile_sizes == (32, 32, 32)
        assert config.num_warps == 4

    def test_kernel_config_to_dict(self):
        """Test KernelConfig.to_dict()."""
        config = KernelConfig(
            algorithm='custom',
            tile_sizes=(64, 64),
            use_tensor_cores=True
        )

        d = config.to_dict()
        assert d['algorithm'] == 'custom'
        assert d['tile_sizes'] == (64, 64)
        assert d['use_tensor_cores'] is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the unified backend system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow: create backend, prepare model, optimize."""
        # Get backend
        backend = get_backend()
        assert backend is not None

        # Create model
        model = SimpleModel()

        # Prepare model
        prepared = backend.prepare_model(model)
        assert prepared is not None

        # Get device info
        info = backend.get_device_info()
        assert info is not None

        # Optimize for inference
        optimized = backend.optimize_for_inference(prepared)
        assert optimized is not None

        # Get memory stats
        stats = backend.get_memory_stats()
        assert isinstance(stats, dict)

    def test_factory_to_optimizer_workflow(self):
        """Test workflow using factory for both backend and optimizer."""
        # Create backend and optimizer
        backend_type = detect_best_backend()
        get_backend(backend_type)
        optimizer = get_optimizer(backend_type)

        # Create and optimize model
        model = SimpleModel()
        optimized_model, result = optimizer.optimize(model, level=OptimizationLevel.O2)

        assert optimized_model is not None
        assert result.success is True

    def test_multiple_backends_same_model(self):
        """Test same model on multiple backends."""
        model = SimpleModel()

        # CPU backend
        cpu_backend = CPUBackend()
        cpu_prepared = cpu_backend.prepare_model(model)

        # Create fresh model for each backend
        from torchbridge.backends.nvidia import NVIDIABackend

        nvidia_backend = NVIDIABackend()
        nvidia_prepared = nvidia_backend.prepare_model(SimpleModel())

        # Both should work without error
        assert cpu_prepared is not None
        assert nvidia_prepared is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
