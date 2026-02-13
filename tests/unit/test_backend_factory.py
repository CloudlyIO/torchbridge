"""
Tests for Backend Factory and Core Backend Infrastructure

This module tests:
- BackendType enum and string conversion
- BackendFactory creation, auto-selection, and registration
- CPUBackend concrete implementation
- CPUAdapter concrete implementation
- DeviceInfo, OptimizationResult, OperationKernelConfig dataclasses
- OptimizationStrategy applicability
- Convenience functions (get_backend, get_optimizer, detect_best_backend, list_available_backends)
"""

import pytest
import torch
import torch.nn as nn

from torchbridge.backends.backend_factory import (
    BackendFactory,
    BackendType,
    detect_best_backend,
    get_backend,
    get_optimizer,
    list_available_backends,
)
from torchbridge.backends.base_adapter import (
    BaseAdapter,
    CPUAdapter,
    OperationKernelConfig,
    OptimizationStrategy,
)
from torchbridge.backends.base_backend import (
    BaseBackend,
    CPUBackend,
    DeviceInfo,
    OptimizationLevel,
    OptimizationResult,
)

# =============================================================================
# BackendType Enum Tests
# =============================================================================


class TestBackendType:
    """Tests for BackendType enum."""

    def test_all_members_exist(self):
        """All expected backend types should be defined."""
        assert BackendType.AUTO.value == "auto"
        assert BackendType.NVIDIA.value == "nvidia"
        assert BackendType.AMD.value == "amd"
        assert BackendType.TPU.value == "tpu"
        assert BackendType.TRAINIUM.value == "trainium"
        assert BackendType.CPU.value == "cpu"

    def test_from_string_exact_match(self):
        """from_string should handle exact backend names."""
        assert BackendType.from_string("nvidia") == BackendType.NVIDIA
        assert BackendType.from_string("amd") == BackendType.AMD
        assert BackendType.from_string("tpu") == BackendType.TPU
        assert BackendType.from_string("trainium") == BackendType.TRAINIUM
        assert BackendType.from_string("cpu") == BackendType.CPU

    def test_from_string_aliases(self):
        """from_string should handle common aliases."""
        assert BackendType.from_string("cuda") == BackendType.NVIDIA
        assert BackendType.from_string("rocm") == BackendType.AMD
        assert BackendType.from_string("hip") == BackendType.AMD
        assert BackendType.from_string("xla") == BackendType.TPU
        assert BackendType.from_string("neuron") == BackendType.TRAINIUM
        assert BackendType.from_string("trn") == BackendType.TRAINIUM

    def test_from_string_case_insensitive(self):
        """from_string should be case-insensitive."""
        assert BackendType.from_string("NVIDIA") == BackendType.NVIDIA
        assert BackendType.from_string("Cuda") == BackendType.NVIDIA

    def test_from_string_unknown_falls_back_to_cpu(self):
        """Unknown backend names should fall back to CPU."""
        assert BackendType.from_string("unknown") == BackendType.CPU
        assert BackendType.from_string("nonexistent") == BackendType.CPU


# =============================================================================
# BackendFactory Tests
# =============================================================================


class TestBackendFactory:
    """Tests for BackendFactory."""

    def test_create_cpu_backend(self):
        """Creating a CPU backend should return a CPUBackend instance."""
        backend = BackendFactory.create("cpu")
        assert isinstance(backend, CPUBackend)
        assert isinstance(backend, BaseBackend)

    def test_create_cpu_from_enum(self):
        """Creating from BackendType enum should work."""
        backend = BackendFactory.create(BackendType.CPU)
        assert isinstance(backend, CPUBackend)

    def test_create_auto_returns_backend(self):
        """Auto-selection should return a valid BaseBackend subclass."""
        backend = BackendFactory.create(BackendType.AUTO)
        assert isinstance(backend, BaseBackend)

    def test_create_auto_from_string(self):
        """Auto-selection via string should return a valid backend."""
        backend = BackendFactory.create("auto")
        assert isinstance(backend, BaseBackend)

    def test_get_available_backends_includes_cpu(self):
        """CPU should always be in the available backends list."""
        available = BackendFactory.get_available_backends()
        assert BackendType.CPU in available

    def test_get_available_backends_returns_list(self):
        """get_available_backends should return a list of BackendType."""
        available = BackendFactory.get_available_backends()
        assert isinstance(available, list)
        for bt in available:
            assert isinstance(bt, BackendType)

    def test_backend_priority_ordering(self):
        """Backend priorities should be correctly defined."""
        assert BackendFactory._priority[BackendType.NVIDIA] == 100
        assert BackendFactory._priority[BackendType.AMD] == 90
        assert BackendFactory._priority[BackendType.TRAINIUM] == 88
        assert BackendFactory._priority[BackendType.TPU] == 85
        assert BackendFactory._priority[BackendType.CPU] == 0

    def test_register_backend(self):
        """Registering a backend should make it available."""
        # Save original state
        original = BackendFactory._backends.copy()
        try:
            BackendFactory.register_backend(
                BackendType.CPU,
                CPUBackend,
                optimizer_class=CPUAdapter,
                priority=10,
            )
            assert BackendType.CPU in BackendFactory._backends
            assert BackendFactory._priority[BackendType.CPU] == 10
        finally:
            # Restore original state
            BackendFactory._backends = original
            BackendFactory._priority[BackendType.CPU] = 0

    def test_get_backend_info_cpu(self):
        """get_backend_info for CPU should show available=True."""
        info = BackendFactory.get_backend_info(BackendType.CPU)
        assert isinstance(info, dict)
        assert info["available"] is True
        assert info["type"] == "cpu"
        assert info["priority"] == 0

    def test_get_all_backend_info(self):
        """get_all_backend_info should return info for all non-AUTO backends."""
        all_info = BackendFactory.get_all_backend_info()
        assert isinstance(all_info, dict)
        assert "cpu" in all_info
        assert "nvidia" in all_info
        assert "auto" not in all_info

    def test_create_optimizer_cpu(self):
        """Creating CPU optimizer should return a CPUAdapter."""
        optimizer = BackendFactory.create_optimizer("cpu")
        assert isinstance(optimizer, CPUAdapter)
        assert isinstance(optimizer, BaseAdapter)


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_backend_returns_base_backend(self):
        """get_backend should return a BaseBackend."""
        backend = get_backend("cpu")
        assert isinstance(backend, BaseBackend)

    def test_get_optimizer_returns_base_adapter(self):
        """get_optimizer should return a BaseAdapter."""
        optimizer = get_optimizer("cpu")
        assert isinstance(optimizer, BaseAdapter)

    def test_detect_best_backend_returns_backend_type(self):
        """detect_best_backend should return a BackendType."""
        best = detect_best_backend()
        assert isinstance(best, BackendType)
        assert best != BackendType.AUTO  # AUTO should be resolved

    def test_list_available_backends_returns_strings(self):
        """list_available_backends should return list of strings."""
        available = list_available_backends()
        assert isinstance(available, list)
        assert "cpu" in available
        for name in available:
            assert isinstance(name, str)


# =============================================================================
# CPUBackend Tests
# =============================================================================


class TestCPUBackend:
    """Tests for CPUBackend concrete implementation."""

    @pytest.fixture
    def backend(self):
        """Create a CPUBackend instance."""
        return CPUBackend()

    def test_backend_name(self, backend):
        """CPUBackend should report correct name."""
        assert backend.BACKEND_NAME == "cpu"

    def test_is_available(self, backend):
        """CPU should always be available."""
        assert backend.is_available is True

    def test_device_is_cpu(self, backend):
        """Device should be CPU."""
        assert backend.device == torch.device("cpu")

    def test_device_count(self, backend):
        """CPU backend should report 1 device."""
        assert backend.device_count == 1

    def test_get_device_info(self, backend):
        """get_device_info should return a DeviceInfo."""
        info = backend.get_device_info()
        assert isinstance(info, DeviceInfo)
        assert info.backend == "cpu"
        assert info.is_available is True

    def test_prepare_model(self, backend):
        """prepare_model should return the model."""
        model = nn.Linear(10, 5)
        prepared = backend.prepare_model(model)
        assert isinstance(prepared, nn.Module)

    def test_optimize_for_inference(self, backend):
        """optimize_for_inference should return the model."""
        model = nn.Linear(10, 5)
        optimized = backend.optimize_for_inference(model)
        assert isinstance(optimized, nn.Module)

    def test_optimize_for_training(self, backend):
        """optimize_for_training should return the model."""
        model = nn.Linear(10, 5)
        optimized = backend.optimize_for_training(model)
        assert isinstance(optimized, nn.Module)

    def test_synchronize_no_error(self, backend):
        """synchronize should not raise."""
        backend.synchronize()  # Should be a no-op for CPU

    def test_empty_cache_no_error(self, backend):
        """empty_cache should not raise."""
        backend.empty_cache()  # Should be a no-op for CPU

    def test_get_memory_stats(self, backend):
        """get_memory_stats should return stats."""
        stats = backend.get_memory_stats()
        assert isinstance(stats, dict) or stats is not None

    def test_to_device(self, backend):
        """to_device should move tensor to CPU."""
        tensor = torch.randn(3, 3)
        moved = backend.to_device(tensor)
        assert moved.device.type == "cpu"


# =============================================================================
# CPUAdapter Tests
# =============================================================================


class TestCPUAdapter:
    """Tests for CPUAdapter concrete implementation."""

    @pytest.fixture
    def adapter(self):
        """Create a CPUAdapter instance."""
        return CPUAdapter()

    def test_adapter_name(self, adapter):
        """CPUAdapter should have correct name."""
        assert adapter.ADAPTER_NAME == "cpu"

    def test_device_is_cpu(self, adapter):
        """Default device should be CPU."""
        assert adapter.device == torch.device("cpu")

    def test_optimize_returns_result(self, adapter):
        """optimize should return model and OptimizationResult tuple."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        result = adapter.optimize(model)
        # Result is (model, OptimizationResult) or just a model depending on impl
        assert result is not None

    def test_get_available_strategies(self, adapter):
        """get_available_strategies should return a list."""
        strategies = adapter.get_available_strategies()
        assert isinstance(strategies, list)
        for s in strategies:
            assert isinstance(s, OptimizationStrategy)


# =============================================================================
# DeviceInfo Dataclass Tests
# =============================================================================


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass."""

    def test_creation(self):
        """DeviceInfo should be created with required fields."""
        info = DeviceInfo(
            backend="cpu",
            device_type="cpu",
            device_id=0,
            device_name="CPU",
        )
        assert info.backend == "cpu"
        assert info.device_id == 0
        assert info.is_available is True

    def test_memory_conversion(self):
        """Memory conversion properties should work."""
        info = DeviceInfo(
            backend="test",
            device_type="test:0",
            device_id=0,
            device_name="Test",
            total_memory_bytes=8 * 1024**3,  # 8 GB
        )
        assert info.total_memory_gb == 8.0
        assert info.total_memory_mb == 8 * 1024

    def test_to_dict(self):
        """to_dict should return a complete dictionary."""
        info = DeviceInfo(
            backend="nvidia",
            device_type="cuda:0",
            device_id=0,
            device_name="RTX 5090",
            compute_capability="12.0",
        )
        d = info.to_dict()
        assert d["backend"] == "nvidia"
        assert d["compute_capability"] == "12.0"
        assert "total_memory_gb" in d


# =============================================================================
# OptimizationResult Dataclass Tests
# =============================================================================


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_creation(self):
        """OptimizationResult should be created with required fields."""
        model = nn.Linear(10, 5)
        result = OptimizationResult(
            success=True,
            model=model,
            level=OptimizationLevel.O2,
        )
        assert result.success is True
        assert result.level == OptimizationLevel.O2
        assert result.optimizations_applied == []
        assert result.warnings == []

    def test_string_level_conversion(self):
        """String level should be converted to enum."""
        model = nn.Linear(10, 5)
        result = OptimizationResult(
            success=True,
            model=model,
            level="O2",
        )
        assert isinstance(result.level, OptimizationLevel)

    def test_to_dict(self):
        """to_dict should include all fields."""
        model = nn.Linear(10, 5)
        result = OptimizationResult(
            success=True,
            model=model,
            level=OptimizationLevel.O1,
            optimizations_applied=["fuse_ops"],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert "fuse_ops" in d["optimizations_applied"]


# =============================================================================
# OperationKernelConfig Tests
# =============================================================================


class TestOperationKernelConfig:
    """Tests for OperationKernelConfig dataclass."""

    def test_defaults(self):
        """Default values should be sensible."""
        config = OperationKernelConfig()
        assert config.algorithm == "auto"
        assert config.num_warps == 4
        assert config.use_tensor_cores is True

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = OperationKernelConfig(
            algorithm="matmul",
            tile_sizes=(64, 64, 64),
            num_warps=8,
            use_tensor_cores=False,
        )
        assert config.algorithm == "matmul"
        assert config.tile_sizes == (64, 64, 64)
        assert config.num_warps == 8

    def test_to_dict(self):
        """to_dict should include all fields."""
        config = OperationKernelConfig(extra_params={"key": "value"})
        d = config.to_dict()
        assert d["algorithm"] == "auto"
        assert d["key"] == "value"


# =============================================================================
# OptimizationStrategy Tests
# =============================================================================


class TestOptimizationStrategy:
    """Tests for OptimizationStrategy dataclass."""

    def test_applicability(self):
        """is_applicable should check against applicable_levels."""
        strategy = OptimizationStrategy(
            name="test",
            description="Test strategy",
            applicable_levels=[OptimizationLevel.O2, OptimizationLevel.O3],
        )
        assert strategy.is_applicable(OptimizationLevel.O2) is True
        assert strategy.is_applicable(OptimizationLevel.O3) is True
        assert strategy.is_applicable(OptimizationLevel.O1) is False

    def test_defaults(self):
        """Default values should indicate no change."""
        strategy = OptimizationStrategy(
            name="noop",
            description="No-op",
            applicable_levels=[],
        )
        assert strategy.speedup_estimate == 1.0
        assert strategy.memory_impact == 1.0
        assert strategy.precision_impact == "none"
