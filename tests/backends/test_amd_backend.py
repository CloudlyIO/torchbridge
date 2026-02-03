"""
AMD Backend Tests (v0.4.9)

Comprehensive test suite for AMD ROCm backend implementation.
Tests cover configuration, backend operations, optimization, compilation,
memory management, and utilities.

v0.4.9 Additions:
- Operator fusion tests (Conv+BN, Linear+GELU, aggressive fusion)
- HIP compilation tests with simulation mode
- Memory layout optimization tests
- Enhanced integration tests

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
        expected = ["AUTO", "CDNA", "CDNA2", "CDNA3", "RDNA2", "RDNA3"]
        for arch_name in expected:
            assert hasattr(AMDArchitecture, arch_name)

    def test_architecture_values(self):
        """Test architecture enum values."""
        assert AMDArchitecture.AUTO.value == "auto"
        assert AMDArchitecture.CDNA2.value == "cdna2"
        assert AMDArchitecture.CDNA3.value == "cdna3"


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


class TestAMDOptimizer:
    """Tests for AMD optimizer."""

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig()
        optimizer = AMDOptimizer(config)

        assert optimizer.config == config

    def test_optimization_levels(self):
        """Test different optimization levels."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        for level in ["conservative", "balanced", "aggressive"]:
            config = AMDConfig(optimization_level=level)
            optimizer = AMDOptimizer(config)

            model = torch.nn.Linear(64, 32)
            optimized = optimizer.optimize(model)

            assert optimized is not None

    def test_optimization_result(self):
        """Test optimization result structure."""
        from torchbridge.backends.amd.amd_optimizer import (
            AMDOptimizer,
        )

        config = AMDConfig()
        optimizer = AMDOptimizer(config)

        model = torch.nn.Linear(64, 32)
        optimizer.optimize(model)

        summary = optimizer.get_optimization_summary()
        assert "optimization_level" in summary
        assert "architecture" in summary

    def test_optimization_with_conv_model(self):
        """Test optimization with convolutional model."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig()
        optimizer = AMDOptimizer(config)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        optimized = optimizer.optimize(model)
        assert optimized is not None


class TestROCmCompiler:
    """Tests for ROCm compiler."""

    def test_compiler_creation(self):
        """Test compiler creation."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig()
        compiler = ROCmCompiler(config)

        assert compiler.config == config

    def test_compile_kernel(self):
        """Test kernel compilation."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig()
        compiler = ROCmCompiler(config)

        source = "__global__ void test_kernel() {}"
        kernel = compiler.compile_kernel(source, "test_kernel")

        assert kernel.name == "test_kernel"
        assert kernel.architecture == config.architecture

    def test_compilation_cache(self):
        """Test compilation cache."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig()
        compiler = ROCmCompiler(config)

        source = "__global__ void cached_kernel() {}"

        # First compilation
        compiler.compile_kernel(source, "cached_kernel")

        # Second compilation should hit cache
        compiler.compile_kernel(source, "cached_kernel")

        stats = compiler.get_compilation_stats()
        assert stats["cache_hits"] >= 1

    def test_compilation_stats(self):
        """Test compilation statistics."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig()
        compiler = ROCmCompiler(config)

        stats = compiler.get_compilation_stats()

        assert "total_compilations" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats

    def test_clear_cache(self):
        """Test cache clearing."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig()
        compiler = ROCmCompiler(config)

        compiler.compile_kernel("__global__ void k() {}", "k")
        compiler.clear_cache()

        stats = compiler.get_compilation_stats()
        assert stats["cache_size"] == 0


class TestAMDMemoryManager:
    """Tests for AMD memory manager."""

    @pytest.fixture
    def memory_manager(self):
        """Create memory manager for tests."""
        from torchbridge.backends.amd.memory_manager import AMDMemoryManager

        config = AMDConfig(memory_pool_size_gb=1.0)
        return AMDMemoryManager(config, device_id=0)

    def test_memory_manager_creation(self):
        """Test memory manager creation."""
        from torchbridge.backends.amd.memory_manager import AMDMemoryManager

        config = AMDConfig()
        manager = AMDMemoryManager(config)

        assert manager.config == config
        assert manager.device_id == 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA/ROCm not available",
    )
    def test_allocate_tensor(self, memory_manager):
        """Test tensor allocation."""
        tensor = memory_manager.allocate_tensor(
            shape=(64, 64),
            dtype=torch.float32,
            purpose="test",
        )

        assert tensor.shape == (64, 64)
        assert tensor.dtype == torch.float32

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA/ROCm not available",
    )
    def test_memory_stats(self, memory_manager):
        """Test memory statistics."""
        stats = memory_manager.get_memory_stats()

        if isinstance(stats, dict):
            assert "total_mb" in stats
            assert "allocated_mb" in stats
            assert "free_mb" in stats
        else:
            assert hasattr(stats, "total_mb")
            assert hasattr(stats, "allocated_mb")
            assert hasattr(stats, "free_mb")

    def test_allocation_summary(self, memory_manager):
        """Test allocation summary."""
        summary = memory_manager.get_allocation_summary()
        assert isinstance(summary, dict)


class TestHIPUtilities:
    """Tests for HIP utilities."""

    @pytest.fixture
    def hip_utils(self):
        """Create HIP utilities for tests."""
        from torchbridge.backends.amd.hip_utilities import HIPUtilities

        config = AMDConfig(enable_profiling=True)
        return HIPUtilities(config)

    def test_utilities_creation(self):
        """Test utilities creation."""
        from torchbridge.backends.amd.hip_utilities import HIPUtilities

        config = AMDConfig()
        utils = HIPUtilities(config)

        assert utils.config == config

    def test_create_stream(self, hip_utils):
        """Test stream creation."""
        stream = hip_utils.create_stream("test_stream")

        assert stream.name == "test_stream"
        assert hip_utils.get_stream("test_stream") is not None

    def test_create_event(self, hip_utils):
        """Test event creation."""
        event = hip_utils.create_event("test_event")

        assert event.name == "test_event"

    def test_profiling_disabled(self):
        """Test profiling when disabled."""
        from torchbridge.backends.amd.hip_utilities import HIPUtilities

        config = AMDConfig(enable_profiling=False)
        utils = HIPUtilities(config)

        with utils.profile_region("test"):
            pass

        data = utils.get_profiling_data()
        assert len(data) == 0

    def test_profiling_enabled(self, hip_utils):
        """Test profiling when enabled."""
        with hip_utils.profile_region("test_region"):
            # Simulate some work
            _ = torch.randn(100, 100)

        data = hip_utils.get_profiling_data()
        assert len(data) >= 1
        assert data[0]["name"] == "test_region"

    def test_profiling_summary(self, hip_utils):
        """Test profiling summary."""
        with hip_utils.profile_region("region1"):
            pass
        with hip_utils.profile_region("region2"):
            pass

        summary = hip_utils.get_profiling_summary()

        assert summary["total_regions"] >= 2
        assert "regions" in summary

    def test_clear_profiling_data(self, hip_utils):
        """Test clearing profiling data."""
        with hip_utils.profile_region("test"):
            pass

        hip_utils.clear_profiling_data()
        assert len(hip_utils.get_profiling_data()) == 0

    def test_device_properties(self, hip_utils):
        """Test getting device properties."""
        props = hip_utils.get_device_properties()

        assert "available" in props
        if props["available"]:
            assert "name" in props
            assert "total_memory_gb" in props

    def test_cleanup(self, hip_utils):
        """Test cleanup."""
        hip_utils.create_stream("test")
        hip_utils.create_event("test")

        hip_utils.cleanup()

        assert len(hip_utils._streams) == 0
        assert len(hip_utils._events) == 0


class TestAMDBackendIntegration:
    """Integration tests for AMD backend."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA/ROCm not available",
    )
    def test_full_pipeline(self):
        """Test full optimization pipeline."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer
        from torchbridge.backends.amd.hip_utilities import HIPUtilities

        config = AMDConfig(
            architecture=AMDArchitecture.CDNA2,
            optimization_level="balanced",
            enable_profiling=True,
        )

        optimizer = AMDOptimizer(config)
        utils = HIPUtilities(config)

        model = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 256),
        )

        with utils.profile_region("optimization"):
            optimized = optimizer.optimize(model)

        assert optimized is not None

        summary = utils.get_profiling_summary()
        assert summary["total_regions"] >= 1

    def test_config_integration(self):
        """Test configuration integration with optimizer."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            optimization_level="aggressive",
            enable_matrix_cores=True,
            enable_mixed_precision=True,
        )

        optimizer = AMDOptimizer(config)
        summary = optimizer.get_optimization_summary()

        assert summary["architecture"] == "cdna3"
        assert summary["matrix_cores_enabled"] is True
        assert summary["mixed_precision"] is True


class TestLRUCache:
    """Tests for LRU cache implementation."""

    def test_lru_cache_basic(self):
        """Test basic LRU cache operations."""
        from torchbridge.backends.amd.rocm_compiler import LRUCache

        cache = LRUCache(max_size=3)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_cache_eviction(self):
        """Test LRU cache eviction."""
        from torchbridge.backends.amd.rocm_compiler import LRUCache

        cache = LRUCache(max_size=2)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # Should evict "a"

        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_cache_update(self):
        """Test LRU cache update behavior."""
        from torchbridge.backends.amd.rocm_compiler import LRUCache

        cache = LRUCache(max_size=2)

        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # Access "a" to make it recently used
        cache.set("c", 3)  # Should evict "b"

        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_lru_cache_clear(self):
        """Test LRU cache clear."""
        from torchbridge.backends.amd.rocm_compiler import LRUCache

        cache = LRUCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)

        cache.clear()

        assert len(cache) == 0
        assert cache.get("a") is None


class TestAMDOperatorFusion:
    """Tests for AMD operator fusion (v0.4.9)."""

    def test_conv_bn_fusion_pattern_detection(self):
        """Test Conv+BatchNorm fusion pattern detection."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig(enable_operator_fusion=True)
        optimizer = AMDOptimizer(config)

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
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig(enable_operator_fusion=True)
        optimizer = AMDOptimizer(config)

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
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            enable_operator_fusion=True
        )
        optimizer = AMDOptimizer(config)

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
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig()
        optimizer = AMDOptimizer(config)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, padding=1),
        )

        optimized = optimizer.optimize(model, level="conservative")
        optimizer.get_optimization_summary()

        assert optimized is not None


class TestHIPCompilationEnhanced:
    """Enhanced tests for HIP compilation (v0.4.9)."""

    def test_simulated_compilation(self):
        """Test simulated compilation without ROCm."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig()
        compiler = ROCmCompiler(config)

        source = """
        __global__ void vector_add(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        """

        kernel = compiler.compile_kernel(source, "vector_add")

        assert kernel.name == "vector_add"
        assert kernel.binary is not None
        assert kernel.compile_time_ms >= 0

    def test_compilation_with_different_opt_levels(self):
        """Test compilation with different optimization levels."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        for level in ["conservative", "balanced", "aggressive"]:
            config = AMDConfig(optimization_level=level)
            compiler = ROCmCompiler(config)

            source = "__global__ void test() {}"
            kernel = compiler.compile_kernel(source, "test", level)

            assert kernel.optimization_flags is not None
            assert len(kernel.optimization_flags) > 0

    def test_gpu_target_mapping(self):
        """Test GPU target mapping for different architectures."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        arch_targets = {
            AMDArchitecture.CDNA2: "gfx90a",
            AMDArchitecture.CDNA3: "gfx940",
            AMDArchitecture.RDNA2: "gfx1030",
            AMDArchitecture.RDNA3: "gfx1100",
        }

        for arch, expected_target in arch_targets.items():
            config = AMDConfig(architecture=arch)
            compiler = ROCmCompiler(config)
            target = compiler._get_gpu_target()
            assert target == expected_target

    def test_precompile_standard_kernels(self):
        """Test precompilation of standard kernels."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig()
        compiler = ROCmCompiler(config)

        # Should complete without errors
        compiler.precompile_standard_kernels()

        stats = compiler.get_compilation_stats()
        assert stats["total_compilations"] >= 0


class TestAMDBackendEnhanced:
    """Enhanced tests for AMD backend (v0.4.9)."""

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


class TestAMDMemoryManagerEnhanced:
    """Enhanced tests for AMD memory manager (v0.4.9)."""

    def test_memory_manager_defragmentation(self):
        """Test memory defragmentation."""
        from torchbridge.backends.amd.memory_manager import AMDMemoryManager

        config = AMDConfig(memory_pool_size_gb=1.0)
        manager = AMDMemoryManager(config)

        # Should complete without errors
        manager.defragment()

        stats = manager.get_memory_stats()
        assert stats['defrag_count'] >= 1

    def test_memory_stats_dataclass(self):
        """Test AMD memory stats dataclass."""
        from torchbridge.backends.amd.memory_manager import AMDMemoryManager

        config = AMDConfig()
        manager = AMDMemoryManager(config)

        stats = manager.get_amd_memory_stats()

        # Check dataclass fields
        assert hasattr(stats, 'total_mb')
        assert hasattr(stats, 'allocated_mb')
        assert hasattr(stats, 'fragmentation_percent')

    def test_memory_manager_cleanup(self):
        """Test memory manager cleanup."""
        from torchbridge.backends.amd.memory_manager import AMDMemoryManager

        config = AMDConfig()
        manager = AMDMemoryManager(config)

        # Should complete without errors
        manager.cleanup()

    def test_tensor_size_estimation(self):
        """Test tensor size estimation."""
        from torchbridge.backends.amd.memory_manager import AMDMemoryManager

        config = AMDConfig()
        manager = AMDMemoryManager(config)

        # 1024 x 1024 float32 = 4MB
        size = manager.estimate_tensor_size((1024, 1024), torch.float32)
        expected = 1024 * 1024 * 4  # 4 bytes per float32

        assert size == expected


class TestAMDIntegrationV049:
    """Integration tests for v0.4.9 AMD improvements."""

    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        from torchbridge.backends.amd.amd_backend import AMDBackend
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig(
            architecture=AMDArchitecture.CDNA3,
            optimization_level="aggressive",
            enable_operator_fusion=True,
            enable_matrix_cores=True,
        )

        backend = AMDBackend(config)
        optimizer = AMDOptimizer(config)

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

    def test_optimizer_and_compiler_integration(self):
        """Test optimizer and compiler work together."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig(optimization_level="balanced")

        optimizer = AMDOptimizer(config)
        compiler = ROCmCompiler(config)

        # Compile a kernel
        kernel = compiler.compile_kernel(
            "__global__ void test() {}",
            "test"
        )
        assert kernel is not None

        # Optimize a model
        model = torch.nn.Linear(64, 32)
        optimized = optimizer.optimize(model)
        assert optimized is not None

    def test_memory_manager_and_backend_integration(self):
        """Test memory manager and backend work together."""
        from torchbridge.backends.amd.amd_backend import AMDBackend
        from torchbridge.backends.amd.memory_manager import AMDMemoryManager

        config = AMDConfig()

        backend = AMDBackend(config)
        mem_manager = AMDMemoryManager(config)

        # Memory stats should be available regardless of device
        stats = mem_manager.get_memory_stats()
        assert stats is not None

        # Check that device types are valid
        assert backend.device.type in ['cpu', 'cuda', 'hip']
        assert mem_manager._get_device().type in ['cpu', 'cuda', 'hip']


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
