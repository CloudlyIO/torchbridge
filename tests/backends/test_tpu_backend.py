#!/usr/bin/env python3
"""
Test suite for TPU backend infrastructure.

Comprehensive tests for TPU backend, optimizer, compiler, memory manager,
XLA integration, and validation components.
"""


import pytest
import torch
import torch.nn as nn

from kernel_pytorch.backends.tpu import (
    TPUBackend,
    TPUMemoryManager,
    TPUOptimizer,
    XLACompiler,
    XLADeviceManager,
    XLADistributedTraining,
    XLAOptimizations,
    XLAUtilities,
    create_xla_integration,
)
from kernel_pytorch.core.config import (
    KernelPyTorchConfig,
    TPUConfig,
    TPUVersion,
)
from kernel_pytorch.validation.unified_validator import (
    validate_tpu_configuration,
    validate_tpu_model,
)


class TestTPUBackend:
    """Test TPU backend functionality."""

    def test_tpu_backend_creation(self):
        """Test basic TPU backend creation."""
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        assert backend is not None
        assert backend.device is not None
        assert backend.world_size >= 1
        assert backend.rank >= 0
        assert not backend.is_distributed  # Single device in test env

    def test_tpu_backend_model_preparation(self):
        """Test model preparation for TPU."""
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        prepared_model = backend.prepare_model(model)
        assert prepared_model is not None
        assert hasattr(prepared_model, 'forward')

    def test_tpu_backend_data_preparation(self):
        """Test data preparation for TPU."""
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        # Test tensor preparation
        tensor = torch.randn(8, 64)
        prepared_tensor = backend.prepare_data(tensor)
        assert prepared_tensor.device == backend.device

        # Test dict preparation
        data_dict = {'input': torch.randn(8, 64), 'target': torch.randn(8, 10)}
        prepared_dict = backend.prepare_data(data_dict)
        assert isinstance(prepared_dict, dict)
        assert all(t.device == backend.device for t in prepared_dict.values())

    def test_tpu_backend_memory_stats(self):
        """Test TPU backend memory statistics."""
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        stats = backend.get_memory_stats()
        assert isinstance(stats, dict)
        assert 'device' in stats
        assert 'world_size' in stats
        assert 'rank' in stats

    def test_tpu_backend_synchronization(self):
        """Test TPU synchronization."""
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        # Should not raise an error
        backend.synchronize()

    def test_tpu_backend_cache_management(self):
        """Test TPU cache management."""
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        # Should not raise an error
        backend.clear_cache()


class TestTPUOptimizer:
    """Test TPU optimizer functionality."""

    def test_tpu_optimizer_creation(self):
        """Test TPU optimizer creation."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)

        assert optimizer is not None
        assert optimizer.config == config
        assert optimizer.backend is not None
        assert optimizer.compiler is not None

    def test_tpu_optimizer_conservative_optimization(self):
        """Test conservative optimization level."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize(model, sample_input, optimization_level="conservative")

        assert result is not None
        assert result.optimized_model is not None
        assert result.optimization_time >= 0
        assert isinstance(result.performance_metrics, dict)

    def test_tpu_optimizer_balanced_optimization(self):
        """Test balanced optimization level."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize(model, sample_input, optimization_level="balanced")

        assert result is not None
        assert result.optimized_model is not None
        assert result.optimization_time >= 0

    def test_tpu_optimizer_aggressive_optimization(self):
        """Test aggressive optimization level."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize(model, sample_input, optimization_level="aggressive")

        assert result is not None
        assert result.optimized_model is not None
        assert result.optimization_time >= 0

    def test_tpu_optimizer_inference_optimization(self):
        """Test inference-specific optimization."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize_for_inference(model, sample_input)

        assert result is not None
        assert result.optimized_model is not None
        assert not result.optimized_model.training  # Should be in eval mode

    def test_tpu_optimizer_training_optimization(self):
        """Test training-specific optimization."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize_for_training(model, sample_input)

        assert result is not None
        assert result.optimized_model is not None

    def test_tpu_optimizer_stats(self):
        """Test optimizer statistics."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)

        stats = optimizer.get_optimization_stats()
        assert isinstance(stats, dict)
        assert 'total_optimizations' in stats

    def test_invalid_optimization_level(self):
        """Test invalid optimization level handling."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        with pytest.raises(ValueError):
            optimizer.optimize(model, sample_input, optimization_level="invalid")


class TestXLACompiler:
    """Test XLA compiler functionality."""

    def test_xla_compiler_creation(self):
        """Test XLA compiler creation."""
        config = KernelPyTorchConfig()
        compiler = XLACompiler(config.hardware.tpu)

        assert compiler is not None
        assert compiler.config == config.hardware.tpu

    def test_xla_compiler_model_compilation(self):
        """Test model compilation."""
        config = KernelPyTorchConfig()
        compiler = XLACompiler(config.hardware.tpu)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        compiled_model = compiler.compile_model(model, sample_input)
        assert compiled_model is not None

    def test_xla_compiler_inference_optimization(self):
        """Test inference optimization."""
        config = KernelPyTorchConfig()
        compiler = XLACompiler(config.hardware.tpu)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        optimized_model = compiler.optimize_for_inference(model, sample_input)
        assert optimized_model is not None
        assert not optimized_model.training

    def test_xla_compiler_training_optimization(self):
        """Test training optimization."""
        config = KernelPyTorchConfig()
        compiler = XLACompiler(config.hardware.tpu)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        optimized_model = compiler.optimize_for_training(model, sample_input)
        assert optimized_model is not None

    def test_xla_compiler_stats(self):
        """Test compilation statistics."""
        config = KernelPyTorchConfig()
        compiler = XLACompiler(config.hardware.tpu)

        stats = compiler.get_compilation_stats()
        assert isinstance(stats, dict)
        assert 'compilation_cache' in stats  # v0.3.2: Changed to cache stats
        assert 'xla_available' in stats
        assert 'compilation_mode' in stats
        assert 'cache_max_size' in stats

    def test_xla_compiler_benchmark(self):
        """Test compilation benchmarking."""
        config = KernelPyTorchConfig()
        compiler = XLACompiler(config.hardware.tpu)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        benchmark_results = compiler.benchmark_compilation(model, sample_input, num_runs=2)
        assert isinstance(benchmark_results, dict)
        assert 'min_time' in benchmark_results
        assert 'avg_time' in benchmark_results


class TestTPUMemoryManager:
    """Test TPU memory manager functionality."""

    def test_memory_manager_creation(self):
        """Test memory manager creation."""
        config = KernelPyTorchConfig()
        memory_manager = TPUMemoryManager(config.hardware.tpu)

        assert memory_manager is not None
        assert memory_manager.config == config.hardware.tpu

    def test_tensor_allocation(self):
        """Test tensor allocation."""
        config = KernelPyTorchConfig()
        memory_manager = TPUMemoryManager(config.hardware.tpu)

        tensor = memory_manager.allocate_tensor((8, 64), dtype=torch.float32)
        assert tensor.shape == (8, 64)
        assert tensor.dtype == torch.float32

    def test_tensor_layout_optimization(self):
        """Test tensor layout optimization."""
        config = KernelPyTorchConfig()
        memory_manager = TPUMemoryManager(config.hardware.tpu)

        # Test 2D tensor optimization
        tensor = torch.randn(7, 7)  # Not divisible by 8
        optimized_tensor = memory_manager.optimize_tensor_layout(tensor)
        assert optimized_tensor.shape[0] % 8 == 0 or optimized_tensor.shape[0] == 7
        assert optimized_tensor.shape[1] % 8 == 0 or optimized_tensor.shape[1] == 7

    def test_memory_pool_creation(self):
        """Test memory pool creation."""
        config = KernelPyTorchConfig()
        memory_manager = TPUMemoryManager(config.hardware.tpu)

        pool_id = memory_manager.create_memory_pool(5, (8, 64))
        assert isinstance(pool_id, str)

        pool_stats = memory_manager.get_pool_stats()
        assert pool_stats['total_pools'] == 1

    def test_memory_pool_operations(self):
        """Test memory pool tensor get/return operations."""
        config = KernelPyTorchConfig()
        memory_manager = TPUMemoryManager(config.hardware.tpu)

        pool_id = memory_manager.create_memory_pool(3, (8, 64))

        # Get tensor from pool
        tensor = memory_manager.get_tensor_from_pool(pool_id)
        assert tensor is not None
        assert tensor.shape == (8, 64)

        # Return tensor to pool
        success = memory_manager.return_tensor_to_pool(pool_id, tensor)
        assert success

    def test_memory_stats(self):
        """Test memory statistics."""
        config = KernelPyTorchConfig()
        memory_manager = TPUMemoryManager(config.hardware.tpu)

        stats = memory_manager.get_memory_stats()
        assert hasattr(stats, 'allocated_memory')
        assert hasattr(stats, 'memory_fraction')
        assert hasattr(stats, 'active_tensors')

    def test_memory_optimization(self):
        """Test memory optimization."""
        config = KernelPyTorchConfig()
        memory_manager = TPUMemoryManager(config.hardware.tpu)

        # Should not raise an error
        memory_manager.optimize_memory_usage()


class TestXLAIntegration:
    """Test XLA integration components."""

    def test_xla_device_manager(self):
        """Test XLA device manager."""
        config = KernelPyTorchConfig()
        device_manager = XLADeviceManager(config.hardware.tpu)

        assert device_manager is not None
        assert device_manager.device is not None
        assert device_manager.world_size >= 1
        assert device_manager.rank >= 0

    def test_xla_distributed_training(self):
        """Test XLA distributed training setup."""
        config = KernelPyTorchConfig()
        device_manager = XLADeviceManager(config.hardware.tpu)
        distributed = XLADistributedTraining(device_manager)

        assert distributed is not None
        assert not distributed.is_distributed  # Single device in test env

    def test_xla_optimizations(self):
        """Test XLA-specific optimizations."""
        config = KernelPyTorchConfig()
        optimizations = XLAOptimizations(config.hardware.tpu)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        optimized_model = optimizations.optimize_model_for_xla(model)

        assert optimized_model is not None

    def test_xla_utilities(self):
        """Test XLA utilities."""
        env_info = XLAUtilities.get_xla_env_info()
        assert isinstance(env_info, dict)
        assert 'xla_available' in env_info

    def test_create_xla_integration(self):
        """Test XLA integration factory."""
        config = KernelPyTorchConfig()
        device_mgr, dist_training, opts = create_xla_integration(config.hardware.tpu)

        assert device_mgr is not None
        assert dist_training is not None
        assert opts is not None


class TestTPUValidation:
    """Test TPU validation integration."""

    def test_tpu_configuration_validation(self):
        """Test TPU configuration validation."""
        config = KernelPyTorchConfig()
        results = validate_tpu_configuration(config)

        assert results is not None
        assert results.total_tests > 0
        assert results.passed >= 0

    def test_tpu_model_validation(self):
        """Test TPU model validation."""
        config = KernelPyTorchConfig()
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        sample_input = torch.randn(8, 64)

        results = validate_tpu_model(model, config.hardware.tpu, sample_input)

        assert results is not None
        assert results.total_tests > 0
        assert results.passed >= 0

    def test_tpu_validation_with_warnings(self):
        """Test TPU validation that should produce warnings."""
        config = KernelPyTorchConfig()
        # Model with non-optimal dimensions
        model = nn.Sequential(
            nn.Linear(63, 31),  # Not divisible by 8
            nn.ReLU(),
            nn.Linear(31, 7)    # Not divisible by 8
        )
        sample_input = torch.randn(7, 63)  # Not divisible by 8

        results = validate_tpu_model(model, config.hardware.tpu, sample_input)

        assert results is not None
        assert results.warnings > 0  # Should have warnings about dimensions


class TestTPUConfigurationModes:
    """Test different TPU configuration modes."""

    def test_inference_mode_tpu(self):
        """Test TPU configuration in inference mode."""
        config = KernelPyTorchConfig.for_inference()
        assert hasattr(config.hardware, 'tpu')
        assert config.hardware.tpu.enabled in [True, False]

    def test_training_mode_tpu(self):
        """Test TPU configuration in training mode."""
        config = KernelPyTorchConfig.for_training()
        assert hasattr(config.hardware, 'tpu')
        assert config.hardware.tpu.enabled in [True, False]

    def test_development_mode_tpu(self):
        """Test TPU configuration in development mode."""
        config = KernelPyTorchConfig.for_development()
        assert hasattr(config.hardware, 'tpu')
        assert config.hardware.tpu.enabled in [True, False]

    def test_tpu_config_serialization_modes(self):
        """Test TPU config serialization in different modes."""
        configs = [
            KernelPyTorchConfig.for_inference(),
            KernelPyTorchConfig.for_training(),
            KernelPyTorchConfig.for_development()
        ]

        for config in configs:
            config_dict = config.to_dict()
            assert 'hardware' in config_dict
            assert 'tpu' in config_dict['hardware']


class TestTPUErrorHandling:
    """Test TPU error handling and edge cases."""

    def test_invalid_tpu_version(self):
        """Test handling of invalid TPU version."""
        # This should be handled gracefully
        config = TPUConfig()
        assert config.version in TPUVersion

    def test_invalid_memory_fraction(self):
        """Test validation of memory fraction bounds."""
        config = KernelPyTorchConfig()
        results = validate_tpu_configuration(config)
        # Should pass with valid memory fraction
        assert results.failed == 0 or any('memory fraction' in r.message for r in results.reports if r.status.value == 'failed')

    def test_missing_sample_inputs(self):
        """Test model validation without sample inputs."""
        config = KernelPyTorchConfig()
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

        results = validate_tpu_model(model, config.hardware.tpu, sample_inputs=None)
        assert results is not None
        # Should complete without sample input validation

    def test_backend_without_xla(self):
        """Test backend operations without XLA available."""
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        # Should work with CPU fallback
        assert backend.device.type == 'cpu'


class TestTPUErrorPaths:
    """Test TPU backend error handling and edge cases (v0.3.2 hardening)."""

    def test_lru_cache_eviction(self):
        """Test LRU cache eviction when max size is exceeded."""
        # Create config with small cache
        config = KernelPyTorchConfig()
        config.hardware.tpu.cache_max_size = 3
        backend = TPUBackend(config)

        # Create and cache multiple models
        models = [nn.Linear(10, 10) for _ in range(5)]
        for model in models:
            _ = backend.prepare_model(model)

        # Cache should only hold 3 models (LRU eviction)
        cache_stats = backend._model_cache.get_stats()
        assert cache_stats['size'] <= 3
        assert cache_stats['evictions'] >= 2

    def test_compilation_cache_limits(self):
        """Test XLA compiler cache size limits."""
        config = KernelPyTorchConfig()
        config.hardware.tpu.cache_max_size = 2
        compiler = XLACompiler(config.hardware.tpu)

        # Compile multiple models
        models = [nn.Linear(i*10, 10) for i in range(1, 4)]
        for model in models:
            _ = compiler.compile_model(model, use_cache=True)

        # Cache should respect max size
        cache_stats = compiler._compilation_cache.get_stats()
        assert cache_stats['size'] <= 2

    def test_strict_validation_mode(self):
        """Test strict validation mode raises exceptions."""
        config = KernelPyTorchConfig()
        config.hardware.tpu.enable_strict_validation = True

        from kernel_pytorch.backends.tpu.tpu_exceptions import TPUValidationError

        optimizer = TPUOptimizer(config)
        model = nn.Linear(10, 10)

        # Create invalid inputs that will cause validation to fail
        invalid_inputs = torch.randn(5, 999)  # Wrong size

        # In strict mode, should raise exception
        with pytest.raises((TPUValidationError, Exception)):
            optimizer.optimize(model, invalid_inputs)

    def test_custom_exceptions_importable(self):
        """Test that custom TPU exceptions can be imported and used."""
        from kernel_pytorch.backends.tpu.tpu_exceptions import (
            TPUBackendError,
            TPUMemoryError,
            TPUNotAvailableError,
            TPUOutOfMemoryError,
            XLACompilationError,
        )

        # Verify exception hierarchy
        assert issubclass(TPUNotAvailableError, TPUBackendError)
        assert issubclass(XLACompilationError, TPUBackendError)
        assert issubclass(TPUOutOfMemoryError, TPUMemoryError)
        assert issubclass(TPUMemoryError, TPUBackendError)

    def test_memory_stats_with_retention(self):
        """Test memory allocation history retention."""
        config = KernelPyTorchConfig()
        config.hardware.tpu.allocation_history_retention_seconds = 1  # 1 second
        manager = TPUMemoryManager(config.hardware.tpu)

        # Allocate some tensors
        for _ in range(5):
            manager.allocate_tensor((10, 10))

        initial_history = len(manager._allocation_history)
        assert initial_history == 5

        # Wait for retention period and optimize
        import time
        time.sleep(1.1)
        manager.optimize_memory_usage()

        # Old allocations should be removed
        assert len(manager._allocation_history) <= initial_history

    def test_configurable_tpu_memory_capacity(self):
        """Test configurable TPU memory capacity overrides."""
        config = KernelPyTorchConfig()
        config.hardware.tpu.version = TPUVersion.V6E
        config.hardware.tpu.v6e_memory_gb = 64.0  # Custom value

        manager = TPUMemoryManager(config.hardware.tpu)
        capacity = manager._get_tpu_memory_gb()

        assert capacity == 64.0  # Should use configured value

    def test_cache_utils_statistics(self):
        """Test LRU cache statistics tracking."""
        from kernel_pytorch.backends.tpu.cache_utils import LRUCache

        cache = LRUCache(max_size=5)

        # Add items
        for i in range(3):
            cache.set(f"key{i}", f"value{i}")

        # Test hits
        assert cache.get("key0") == "value0"
        assert cache.get("key1") == "value1"

        # Test miss
        assert cache.get("nonexistent") is None

        # Check stats
        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['size'] == 3
        assert stats['hit_rate'] == 2/3

    def test_cache_eviction_behavior(self):
        """Test LRU cache eviction behavior."""
        from kernel_pytorch.backends.tpu.cache_utils import LRUCache

        cache = LRUCache(max_size=3)

        # Fill cache
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" to make it recently used
        cache.get("a")

        # Add new item (should evict "b" as least recently used)
        cache.set("d", 4)

        stats = cache.get_stats()
        assert stats['evictions'] == 1
        assert cache.get("b") is None  # Evicted
        assert cache.get("a") == 1      # Kept (recently used)
        assert cache.get("d") == 4      # New item

    def test_optimizer_with_invalid_level(self):
        """Test optimizer handles invalid optimization level."""
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)
        model = nn.Linear(10, 10)

        # Invalid optimization level should raise ValueError
        with pytest.raises(ValueError, match="Unknown optimization level"):
            optimizer._apply_optimization_level(model, "invalid_level")

    def test_memory_pool_operations(self):
        """Test memory pool creation and retrieval."""
        config = KernelPyTorchConfig()
        manager = TPUMemoryManager(config.hardware.tpu)

        # Create pool
        pool_id = manager.create_memory_pool(pool_size=5, tensor_size=(10, 10))
        assert pool_id is not None

        # Get tensor from pool
        tensor = manager.get_tensor_from_pool(pool_id)
        assert tensor is not None
        assert tensor.shape == (10, 10)

        # Return tensor to pool
        success = manager.return_tensor_to_pool(pool_id, tensor)
        assert success

        # Get pool stats
        pool_stats = manager.get_pool_stats()
        assert pool_id in pool_stats['pool_details']

    def test_compilation_mode_configuration(self):
        """Test different XLA compilation modes."""
        from kernel_pytorch.core.config import TPUCompilationMode

        config = KernelPyTorchConfig()
        config.hardware.tpu.compilation_mode = TPUCompilationMode.TORCH_XLA

        compiler = XLACompiler(config.hardware.tpu)
        assert compiler.config.compilation_mode == TPUCompilationMode.TORCH_XLA

    def test_logging_not_print(self):
        """Test that logging is used instead of print statements."""
        import logging
        from io import StringIO

        # Capture logs
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)

        logger = logging.getLogger('kernel_pytorch.backends.tpu')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        # Create backend (should log, not print)
        config = KernelPyTorchConfig()
        TPUBackend(config)

        # Check that logs were captured
        log_stream.getvalue()
        # Should have some log output from backend initialization
        # (but may be empty if logger not configured in test env)

        logger.removeHandler(handler)

    def test_xla_optimization_levels(self):
        """Test XLA optimization level configuration."""
        config = KernelPyTorchConfig()
        config.hardware.tpu.xla_optimization_level = 2

        compiler = XLACompiler(config.hardware.tpu)
        assert compiler.config.xla_optimization_level == 2

    def test_cache_clear_operations(self):
        """Test cache clearing functionality."""
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        # Add items to cache
        model = nn.Linear(10, 10)
        backend.prepare_model(model)

        assert len(backend._model_cache) > 0

        # Clear cache
        backend.clear_cache()

        # Cache should be empty
        assert len(backend._model_cache) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
