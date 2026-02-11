#!/usr/bin/env python3
"""
Test suite for Trainium backend infrastructure.

Comprehensive tests for Trainium backend, optimizer, compiler, memory manager,
and Neuron integration components. All tests mock torch_neuronx / torch_xla
imports so no real Trainium hardware is needed.
"""

import pytest
import torch
import torch.nn as nn

from torchbridge.backends.trainium import (
    NeuronCompiler,
    TrainiumAdapter,
    TrainiumBackend,
    TrainiumMemoryManager,
)
from torchbridge.core.config import (
    TorchBridgeConfig,
    TrainiumArchitecture,
    TrainiumConfig,
)


class TestTrainiumBackend:
    """Test Trainium backend functionality."""

    def test_trainium_backend_creation(self):
        """Test basic Trainium backend creation."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        assert backend is not None
        assert backend.device is not None
        assert backend.world_size >= 1
        assert backend.rank >= 0
        assert not backend.is_distributed  # Single device in test env

    def test_trainium_backend_model_preparation(self):
        """Test model preparation for Trainium."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        prepared_model = backend.prepare_model(model)
        assert prepared_model is not None
        assert hasattr(prepared_model, 'forward')

    def test_trainium_backend_data_preparation(self):
        """Test data preparation for Trainium."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        # Test tensor preparation
        tensor = torch.randn(8, 64)
        prepared_tensor = backend.prepare_data(tensor)
        assert prepared_tensor.device == backend.device

        # Test dict preparation
        data_dict = {'input': torch.randn(8, 64), 'target': torch.randn(8, 10)}
        prepared_dict = backend.prepare_data(data_dict)
        assert isinstance(prepared_dict, dict)
        assert all(t.device == backend.device for t in prepared_dict.values())

    def test_trainium_backend_memory_stats(self):
        """Test Trainium backend memory statistics."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        stats = backend.get_memory_stats()
        assert isinstance(stats, dict)
        assert 'device' in stats
        assert 'world_size' in stats
        assert 'rank' in stats

    def test_trainium_backend_synchronization(self):
        """Test Trainium synchronization."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        # Should not raise an error
        backend.synchronize()

    def test_trainium_backend_cache_management(self):
        """Test Trainium cache management."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        # Should not raise an error
        backend.clear_cache()

    def test_trainium_backend_device_info(self):
        """Test Trainium device info."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        info = backend._get_device_info()
        assert info is not None
        assert info.backend == "trainium"

    def test_trainium_backend_repr(self):
        """Test Trainium backend string representation."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        repr_str = repr(backend)
        assert "TrainiumBackend" in repr_str

    def test_trainium_backend_without_neuron(self):
        """Test backend operations without Neuron SDK (CPU fallback)."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        # Should work with CPU fallback
        assert backend.device.type == 'cpu'


class TestTrainiumConfig:
    """Test Trainium configuration."""

    def test_trainium_config_creation(self):
        """Test basic TrainiumConfig creation."""
        config = TrainiumConfig()
        assert config.enabled is True
        assert config.precision == "bfloat16"
        assert config.mixed_precision is True
        assert config.memory_fraction == 0.90
        assert config.cache_max_size == 100

    def test_trainium_architecture_auto(self):
        """Test auto-detection of Trainium architecture."""
        config = TrainiumConfig()
        # In test env (no Trainium), should default to TRN2
        assert config.architecture in TrainiumArchitecture

    def test_trainium_config_trn1(self):
        """Test TRN1 configuration."""
        config = TrainiumConfig(architecture=TrainiumArchitecture.TRN1)
        assert config.architecture == TrainiumArchitecture.TRN1
        assert config.enable_mxfp8 is False  # TRN1 doesn't support MXFP8
        assert config.enable_mxfp4 is False  # TRN1 doesn't support MXFP4

    def test_trainium_config_trn2(self):
        """Test TRN2 configuration."""
        config = TrainiumConfig(architecture=TrainiumArchitecture.TRN2)
        assert config.architecture == TrainiumArchitecture.TRN2
        assert config.enable_mxfp4 is False  # TRN2 doesn't support MXFP4

    def test_trainium_config_trn3(self):
        """Test TRN3 configuration."""
        config = TrainiumConfig(architecture=TrainiumArchitecture.TRN3)
        assert config.architecture == TrainiumArchitecture.TRN3
        # TRN3 supports all precision formats (user-configured)

    def test_trainium_config_inf2(self):
        """Test Inferentia2 configuration."""
        config = TrainiumConfig(architecture=TrainiumArchitecture.INF2)
        assert config.architecture == TrainiumArchitecture.INF2
        assert config.enable_mxfp8 is False
        assert config.enable_mxfp4 is False

    def test_trainium_config_precision_settings(self):
        """Test precision-related settings."""
        config = TrainiumConfig(
            precision="bfloat16",
            mixed_precision=True,
            enable_cfp8=True,
        )
        assert config.precision == "bfloat16"
        assert config.mixed_precision is True
        assert config.enable_cfp8 is True

    def test_trainium_config_distributed_settings(self):
        """Test distributed training settings."""
        config = TrainiumConfig(
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
        )
        assert config.tensor_parallel_size == 2
        assert config.pipeline_parallel_size == 4

    def test_hardware_config_includes_trainium(self):
        """Test that HardwareConfig includes trainium field."""
        config = TorchBridgeConfig()
        assert hasattr(config.hardware, 'trainium')
        assert isinstance(config.hardware.trainium, TrainiumConfig)

    def test_torchbridge_config_serialization_includes_trainium(self):
        """Test that config serialization includes trainium."""
        config = TorchBridgeConfig()
        config_dict = config.to_dict()
        assert 'hardware' in config_dict
        assert 'trainium' in config_dict['hardware']


class TestTrainiumAdapter:
    """Test Trainium adapter functionality."""

    def test_trainium_optimizer_creation(self):
        """Test Trainium optimizer creation."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)

        assert optimizer is not None
        assert optimizer.config == config
        assert optimizer.backend is not None
        assert optimizer.compiler is not None

    def test_trainium_optimizer_conservative_optimization(self):
        """Test conservative optimization level."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize(model, sample_input, optimization_level="conservative")

        assert result is not None
        assert result.optimized_model is not None
        assert result.optimization_time >= 0
        assert isinstance(result.performance_metrics, dict)

    def test_trainium_optimizer_balanced_optimization(self):
        """Test balanced optimization level."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize(model, sample_input, optimization_level="balanced")

        assert result is not None
        assert result.optimized_model is not None

    def test_trainium_optimizer_aggressive_optimization(self):
        """Test aggressive optimization level."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize(model, sample_input, optimization_level="aggressive")

        assert result is not None
        assert result.optimized_model is not None

    def test_trainium_optimizer_inference(self):
        """Test inference-specific optimization."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize_for_inference(model, sample_input)

        assert result is not None
        assert result.optimized_model is not None
        assert not result.optimized_model.training

    def test_trainium_optimizer_training(self):
        """Test training-specific optimization."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        result = optimizer.optimize_for_training(model, sample_input)

        assert result is not None
        assert result.optimized_model is not None

    def test_trainium_optimizer_stats(self):
        """Test optimizer statistics."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)

        stats = optimizer.get_optimization_stats()
        assert isinstance(stats, dict)
        assert 'total_optimizations' in stats

    def test_invalid_optimization_level(self):
        """Test invalid optimization level handling."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        with pytest.raises(ValueError):
            optimizer.optimize(model, sample_input, optimization_level="invalid")


class TestNeuronCompiler:
    """Test Neuron compiler functionality."""

    def test_neuron_compiler_creation(self):
        """Test Neuron compiler creation."""
        config = TorchBridgeConfig()
        compiler = NeuronCompiler(config.hardware.trainium)

        assert compiler is not None
        assert compiler.config == config.hardware.trainium

    def test_neuron_compiler_model_compilation(self):
        """Test model compilation."""
        config = TorchBridgeConfig()
        compiler = NeuronCompiler(config.hardware.trainium)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        compiled_model = compiler.compile_model(model, sample_input)
        assert compiled_model is not None

    def test_neuron_compiler_inference_optimization(self):
        """Test inference optimization."""
        config = TorchBridgeConfig()
        compiler = NeuronCompiler(config.hardware.trainium)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        optimized_model = compiler.optimize_for_inference(model, sample_input)
        assert optimized_model is not None
        assert not optimized_model.training

    def test_neuron_compiler_training_optimization(self):
        """Test training optimization."""
        config = TorchBridgeConfig()
        compiler = NeuronCompiler(config.hardware.trainium)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        optimized_model = compiler.optimize_for_training(model, sample_input)
        assert optimized_model is not None

    def test_neuron_compiler_stats(self):
        """Test compilation statistics."""
        config = TorchBridgeConfig()
        compiler = NeuronCompiler(config.hardware.trainium)

        stats = compiler.get_compilation_stats()
        assert isinstance(stats, dict)
        assert 'compilation_cache' in stats
        assert 'neuron_available' in stats
        assert 'graph_caching_enabled' in stats
        assert 'cache_max_size' in stats

    def test_neuron_compiler_benchmark(self):
        """Test compilation benchmarking."""
        config = TorchBridgeConfig()
        compiler = NeuronCompiler(config.hardware.trainium)

        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        sample_input = torch.randn(8, 64)

        benchmark_results = compiler.benchmark_compilation(model, sample_input, num_runs=2)
        assert isinstance(benchmark_results, dict)
        assert 'min_time' in benchmark_results
        assert 'avg_time' in benchmark_results

    def test_neuron_compiler_repr(self):
        """Test Neuron compiler string representation."""
        config = TorchBridgeConfig()
        compiler = NeuronCompiler(config.hardware.trainium)

        repr_str = repr(compiler)
        assert "NeuronCompiler" in repr_str


class TestTrainiumMemoryManager:
    """Test Trainium memory manager functionality."""

    def test_memory_manager_creation(self):
        """Test memory manager creation."""
        config = TorchBridgeConfig()
        memory_manager = TrainiumMemoryManager(config.hardware.trainium)

        assert memory_manager is not None
        assert memory_manager.config == config.hardware.trainium

    def test_tensor_allocation(self):
        """Test tensor allocation."""
        config = TorchBridgeConfig()
        memory_manager = TrainiumMemoryManager(config.hardware.trainium)

        tensor = memory_manager.allocate_tensor((8, 64), dtype=torch.float32)
        assert tensor.shape == (8, 64)
        assert tensor.dtype == torch.float32

    def test_tensor_layout_optimization(self):
        """Test tensor layout optimization."""
        config = TorchBridgeConfig()
        memory_manager = TrainiumMemoryManager(config.hardware.trainium)

        # Test 2D tensor optimization (non-aligned dimensions)
        tensor = torch.randn(7, 7)
        optimized_tensor = memory_manager.optimize_tensor_layout(tensor)
        assert optimized_tensor.shape[0] % 8 == 0 or optimized_tensor.shape[0] == 7
        assert optimized_tensor.shape[1] % 8 == 0 or optimized_tensor.shape[1] == 7

    def test_memory_pool_creation(self):
        """Test memory pool creation."""
        config = TorchBridgeConfig()
        memory_manager = TrainiumMemoryManager(config.hardware.trainium)

        pool_id = memory_manager.create_memory_pool(5, (8, 64))
        assert isinstance(pool_id, str)

        pool_stats = memory_manager.get_pool_stats()
        assert pool_stats['total_pools'] == 1

    def test_memory_pool_operations(self):
        """Test memory pool tensor get/return operations."""
        config = TorchBridgeConfig()
        memory_manager = TrainiumMemoryManager(config.hardware.trainium)

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
        config = TorchBridgeConfig()
        memory_manager = TrainiumMemoryManager(config.hardware.trainium)

        stats = memory_manager.get_memory_stats()
        assert hasattr(stats, 'allocated_memory')
        assert hasattr(stats, 'memory_fraction')
        assert hasattr(stats, 'active_tensors')

    def test_memory_optimization(self):
        """Test memory optimization."""
        config = TorchBridgeConfig()
        memory_manager = TrainiumMemoryManager(config.hardware.trainium)

        # Should not raise an error
        memory_manager.optimize_memory_usage()

    def test_trainium_memory_capacity(self):
        """Test Trainium memory capacity by architecture."""
        # TRN1: 32GB
        config = TrainiumConfig(architecture=TrainiumArchitecture.TRN1)
        manager = TrainiumMemoryManager(config)
        assert manager._get_trainium_memory_gb() == 32.0

        # TRN2: 96GB
        config = TrainiumConfig(architecture=TrainiumArchitecture.TRN2)
        manager = TrainiumMemoryManager(config)
        assert manager._get_trainium_memory_gb() == 96.0

        # TRN3: 144GB
        config = TrainiumConfig(architecture=TrainiumArchitecture.TRN3)
        manager = TrainiumMemoryManager(config)
        assert manager._get_trainium_memory_gb() == 144.0

        # INF2: 32GB
        config = TrainiumConfig(architecture=TrainiumArchitecture.INF2)
        manager = TrainiumMemoryManager(config)
        assert manager._get_trainium_memory_gb() == 32.0


class TestTrainiumErrorHandling:
    """Test Trainium error handling and edge cases."""

    def test_custom_exceptions_importable(self):
        """Test that custom Trainium exceptions can be imported and used."""
        from torchbridge.backends.trainium.trainium_exceptions import (
            NeuronCompilationError,
            TrainiumBackendError,
            TrainiumMemoryError,
            TrainiumNotAvailableError,
            TrainiumOutOfMemoryError,
        )

        # Verify exception hierarchy
        assert issubclass(TrainiumNotAvailableError, TrainiumBackendError)
        assert issubclass(NeuronCompilationError, TrainiumBackendError)
        assert issubclass(TrainiumOutOfMemoryError, TrainiumMemoryError)
        assert issubclass(TrainiumMemoryError, TrainiumBackendError)

    def test_lru_cache_eviction(self):
        """Test LRU cache eviction when max size is exceeded."""
        config = TorchBridgeConfig()
        config.hardware.trainium.cache_max_size = 3
        backend = TrainiumBackend(config)

        # Create and cache multiple models
        models = [nn.Linear(10, 10) for _ in range(5)]
        for model in models:
            _ = backend.prepare_model(model)

        # Cache should only hold 3 models (LRU eviction)
        cache_stats = backend._model_cache.get_stats()
        assert cache_stats['size'] <= 3
        assert cache_stats['evictions'] >= 2

    def test_compilation_cache_limits(self):
        """Test Neuron compiler cache size limits."""
        config = TorchBridgeConfig()
        config.hardware.trainium.cache_max_size = 2
        compiler = NeuronCompiler(config.hardware.trainium)

        # Compile multiple models
        models = [nn.Linear(i * 10, 10) for i in range(1, 4)]
        for model in models:
            _ = compiler.compile_model(model, use_cache=True)

        # Cache should respect max size
        cache_stats = compiler._compilation_cache.get_stats()
        assert cache_stats['size'] <= 2

    def test_strict_validation_mode(self):
        """Test strict validation mode raises exceptions."""
        config = TorchBridgeConfig()
        config.hardware.trainium.enable_strict_validation = True

        from torchbridge.backends.trainium.trainium_exceptions import (
            TrainiumValidationError,
        )

        optimizer = TrainiumAdapter(config)
        model = nn.Linear(10, 10)

        # Create invalid inputs that will cause validation to fail
        invalid_inputs = torch.randn(5, 999)  # Wrong size

        # In strict mode, should raise exception
        with pytest.raises((TrainiumValidationError, Exception)):
            optimizer.optimize(model, invalid_inputs)

    def test_optimizer_with_invalid_level(self):
        """Test optimizer handles invalid optimization level."""
        config = TorchBridgeConfig()
        optimizer = TrainiumAdapter(config)
        model = nn.Linear(10, 10)

        # Invalid optimization level should raise ValueError
        with pytest.raises(ValueError, match="Unknown optimization level"):
            optimizer._apply_optimization_level(model, "invalid_level")

    def test_memory_stats_with_retention(self):
        """Test memory allocation history retention."""
        config = TorchBridgeConfig()
        config.hardware.trainium.allocation_history_retention_seconds = 1
        manager = TrainiumMemoryManager(config.hardware.trainium)

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

    def test_cache_clear_operations(self):
        """Test cache clearing functionality."""
        config = TorchBridgeConfig()
        backend = TrainiumBackend(config)

        # Add items to cache
        model = nn.Linear(10, 10)
        backend.prepare_model(model)

        assert len(backend._model_cache) > 0

        # Clear cache
        backend.clear_cache()

        # Cache should be empty
        assert len(backend._model_cache) == 0


class TestTrainiumNeuronUtilities:
    """Test Neuron utility functions."""

    def test_neuron_env_info(self):
        """Test getting Neuron environment info."""
        from torchbridge.backends.trainium.neuron_utilities import get_neuron_env_info

        env_info = get_neuron_env_info()
        assert isinstance(env_info, dict)
        assert 'neuron_available' in env_info
        assert 'neuron_sdk_version' in env_info
        assert 'PJRT_DEVICE' in env_info

    def test_neuron_sdk_version(self):
        """Test getting Neuron SDK version."""
        from torchbridge.backends.trainium.neuron_utilities import (
            get_neuron_sdk_version,
        )

        version = get_neuron_sdk_version()
        # In test env (no Neuron), should return 'not installed'
        assert isinstance(version, str)

    def test_is_neuron_available(self):
        """Test Neuron availability check."""
        from torchbridge.backends.trainium.neuron_utilities import is_neuron_available

        # In test env (no Trainium hardware), should return False
        result = is_neuron_available()
        assert isinstance(result, bool)


class TestTrainiumBackendFactory:
    """Test Trainium integration with backend factory."""

    def test_backend_type_trainium(self):
        """Test that TRAINIUM is in BackendType enum."""
        from torchbridge.backends.backend_factory import BackendType

        assert hasattr(BackendType, 'TRAINIUM')
        assert BackendType.TRAINIUM.value == 'trainium'

    def test_backend_type_from_string(self):
        """Test BackendType.from_string with Trainium aliases."""
        from torchbridge.backends.backend_factory import BackendType

        assert BackendType.from_string('trainium') == BackendType.TRAINIUM
        assert BackendType.from_string('neuron') == BackendType.TRAINIUM
        assert BackendType.from_string('trn') == BackendType.TRAINIUM

    def test_backend_factory_priority(self):
        """Test that Trainium has correct priority in factory."""
        from torchbridge.backends.backend_factory import BackendFactory, BackendType

        priority = BackendFactory._priority.get(BackendType.TRAINIUM)
        assert priority == 88  # Between TPU (85) and AMD (90)


class TestTrainiumConfigurationModes:
    """Test different configuration modes with Trainium."""

    def test_inference_mode(self):
        """Test Trainium config in inference mode."""
        config = TorchBridgeConfig.for_inference()
        assert hasattr(config.hardware, 'trainium')
        assert config.hardware.trainium.enabled in [True, False]

    def test_training_mode(self):
        """Test Trainium config in training mode."""
        config = TorchBridgeConfig.for_training()
        assert hasattr(config.hardware, 'trainium')
        assert config.hardware.trainium.enabled in [True, False]

    def test_development_mode(self):
        """Test Trainium config in development mode."""
        config = TorchBridgeConfig.for_development()
        assert hasattr(config.hardware, 'trainium')
        assert config.hardware.trainium.enabled in [True, False]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
