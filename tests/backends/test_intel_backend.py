"""
Tests for Intel XPU Backend

Tests cover:
- Device detection and management
- Memory management
- Model optimization
- IPEX integration (when available)
"""


import pytest
import torch
import torch.nn as nn

# Import Intel backend components
from torchbridge.backends.intel import (
    IPEX_AVAILABLE,
    XPU_AVAILABLE,
    IntelBackend,
    IntelKernelOptimizer,
    IntelMemoryManager,
    IntelOptimizationLevel,
    IntelOptimizer,
    OptimizationResult,
    XPUDeviceInfo,
    XPUDeviceManager,
    XPUOptimizations,
    get_ipex_version,
    is_ipex_available,
    is_xpu_available,
)
from torchbridge.backends.intel.intel_exceptions import (
    IPEXNotInstalledError,
    OneDNNError,
    XPUDeviceError,
    XPUNotAvailableError,
    XPUOptimizationError,
    XPUOutOfMemoryError,
)
from torchbridge.core.config import (
    HardwareBackend,
    HardwareConfig,
    IntelArchitecture,
    IntelConfig,
    TorchBridgeConfig,
)

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )


@pytest.fixture
def conv_model():
    """Create a convolutional model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
    )


@pytest.fixture
def intel_config():
    """Create Intel configuration for testing."""
    return IntelConfig(
        enabled=True,
        architecture=IntelArchitecture.DG2,
        optimization_level="balanced",
    )


# ============================================================================
# Test Configuration
# ============================================================================

class TestIntelConfig:
    """Test Intel configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IntelConfig()
        assert config.enabled is True
        assert config.ipex_enabled is True
        assert config.onednn_enabled is True
        assert config.enable_mixed_precision is True

    def test_architecture_enum(self):
        """Test IntelArchitecture enum values."""
        assert IntelArchitecture.AUTO.value == "auto"
        assert IntelArchitecture.PVC.value == "pvc"
        assert IntelArchitecture.DG2.value == "dg2"
        assert IntelArchitecture.FLEX.value == "flex"
        assert IntelArchitecture.INTEGRATED.value == "integrated"

    def test_pvc_configuration(self):
        """Test PVC (Ponte Vecchio) configuration."""
        config = IntelConfig(architecture=IntelArchitecture.PVC)
        assert config.allow_bf16 is True
        assert config.enable_amx is True

    def test_dg2_configuration(self):
        """Test DG2 (Arc) configuration."""
        config = IntelConfig(architecture=IntelArchitecture.DG2)
        assert config.allow_bf16 is True
        assert config.enable_amx is False

    def test_integrated_configuration(self):
        """Test integrated graphics configuration."""
        config = IntelConfig(architecture=IntelArchitecture.INTEGRATED)
        assert config.allow_bf16 is False
        assert config.enable_amx is False
        assert config.max_memory_fraction == 0.5


class TestHardwareConfigWithIntel:
    """Test HardwareConfig with Intel support."""

    def test_intel_config_field(self):
        """Test Intel config is in HardwareConfig."""
        config = HardwareConfig()
        assert hasattr(config, 'intel')
        assert isinstance(config.intel, IntelConfig)

    def test_intel_backend_enum(self):
        """Test INTEL backend enum value."""
        assert HardwareBackend.INTEL.value == "intel"


# ============================================================================
# Test Exceptions
# ============================================================================

class TestIntelExceptions:
    """Test Intel backend exceptions."""

    def test_xpu_not_available_error(self):
        """Test XPUNotAvailableError."""
        error = XPUNotAvailableError()
        assert "Intel XPU" in str(error)
        assert "IPEX" in str(error)

    def test_ipex_not_installed_error(self):
        """Test IPEXNotInstalledError."""
        error = IPEXNotInstalledError()
        assert "IPEX" in str(error)
        assert "pip install" in str(error)

    def test_xpu_device_error(self):
        """Test XPUDeviceError."""
        error = XPUDeviceError("Device error", device_id=1)
        assert error.device_id == 1
        assert "Device error" in str(error)

    def test_xpu_out_of_memory_error(self):
        """Test XPUOutOfMemoryError."""
        error = XPUOutOfMemoryError(
            requested_bytes=1024 * 1024 * 1024,  # 1GB
            available_bytes=512 * 1024 * 1024,   # 512MB
            device_id=0
        )
        assert error.requested_bytes == 1024 * 1024 * 1024
        assert "xpu:0" in str(error)

    def test_onednn_error(self):
        """Test OneDNNError."""
        error = OneDNNError("Primitive error", operation="conv2d")
        assert error.operation == "conv2d"
        assert "oneDNN" in str(error)

    def test_xpu_optimization_error(self):
        """Test XPUOptimizationError."""
        error = XPUOptimizationError(
            "Fusion failed",
            optimization_type="fusion"
        )
        assert "Fusion failed" in str(error)


# ============================================================================
# Test Utilities
# ============================================================================

class TestXPUUtilities:
    """Test XPU utility functions."""

    def test_is_xpu_available_returns_bool(self):
        """Test is_xpu_available returns boolean."""
        result = is_xpu_available()
        assert isinstance(result, bool)

    def test_is_ipex_available_returns_bool(self):
        """Test is_ipex_available returns boolean."""
        result = is_ipex_available()
        assert isinstance(result, bool)

    def test_get_ipex_version(self):
        """Test get_ipex_version returns string or None."""
        result = get_ipex_version()
        assert result is None or isinstance(result, str)

    def test_xpu_available_constant(self):
        """Test XPU_AVAILABLE constant is boolean."""
        assert isinstance(XPU_AVAILABLE, bool)

    def test_ipex_available_constant(self):
        """Test IPEX_AVAILABLE constant is boolean."""
        assert isinstance(IPEX_AVAILABLE, bool)


class TestXPUDeviceInfo:
    """Test XPUDeviceInfo dataclass."""

    def test_device_info_creation(self):
        """Test creating XPUDeviceInfo."""
        info = XPUDeviceInfo(
            device_id=0,
            name="Intel Arc A770",
            total_memory=16 * 1024**3,
            driver_version="1.0.0",
            compute_capability=(1, 0),
            supports_amx=False,
            supports_fp16=True,
            supports_bf16=True,
            max_compute_units=512,
            device_type="consumer"
        )
        assert info.device_id == 0
        assert info.name == "Intel Arc A770"
        assert info.device_type == "consumer"
        assert info.supports_fp16 is True


# ============================================================================
# Test Device Manager
# ============================================================================

class TestXPUDeviceManager:
    """Test XPU device manager."""

    def test_device_manager_creation(self):
        """Test creating device manager."""
        manager = XPUDeviceManager()
        assert manager is not None
        assert manager.device_id == 0

    def test_device_count_property(self):
        """Test device_count property."""
        manager = XPUDeviceManager()
        count = manager.device_count
        assert isinstance(count, int)
        assert count >= 0

    def test_get_device_fallback(self):
        """Test get_device falls back to CPU when XPU unavailable."""
        manager = XPUDeviceManager()
        if not XPU_AVAILABLE:
            device = manager.get_device()
            assert device.type == "cpu"


# ============================================================================
# Test Memory Manager
# ============================================================================

class TestIntelMemoryManager:
    """Test Intel memory manager."""

    def test_memory_manager_creation(self):
        """Test creating memory manager."""
        manager = IntelMemoryManager(device_id=0)
        assert manager is not None
        assert manager._device_id == 0

    def test_get_device_fallback(self):
        """Test _get_device falls back to CPU."""
        manager = IntelMemoryManager(device_id=0)
        device = manager._get_device()
        if not XPU_AVAILABLE:
            assert device.type == "cpu"

    def test_optimal_alignment(self):
        """Test optimal memory alignment."""
        manager = IntelMemoryManager(device_id=0)
        alignment = manager._get_optimal_alignment()
        assert alignment == 64  # Intel XPU optimal alignment

    def test_memory_stats(self):
        """Test getting memory statistics."""
        manager = IntelMemoryManager(device_id=0)
        stats = manager.get_memory_stats()
        assert hasattr(stats, 'allocated_bytes')
        assert hasattr(stats, 'total_bytes')
        assert hasattr(stats, 'free_bytes')

    def test_memory_summary(self):
        """Test memory summary string."""
        manager = IntelMemoryManager(device_id=0)
        summary = manager.get_memory_summary()
        assert isinstance(summary, str)
        assert "Intel XPU" in summary


# ============================================================================
# Test Backend
# ============================================================================

class TestIntelBackend:
    """Test Intel backend."""

    def test_backend_creation(self):
        """Test creating Intel backend."""
        backend = IntelBackend()
        assert backend is not None

    def test_backend_with_config(self):
        """Test creating backend with config."""
        config = TorchBridgeConfig()
        backend = IntelBackend(config=config)
        assert backend.config is not None

    def test_device_property(self):
        """Test device property."""
        backend = IntelBackend()
        device = backend.device
        assert isinstance(device, torch.device)

    def test_devices_property(self):
        """Test devices property."""
        backend = IntelBackend()
        devices = backend.devices
        assert isinstance(devices, list)

    def test_is_xpu_available_property(self):
        """Test is_xpu_available property."""
        backend = IntelBackend()
        result = backend.is_xpu_available
        assert isinstance(result, bool)

    def test_get_device_info(self):
        """Test get_device_info."""
        from torchbridge.backends import DeviceInfo
        backend = IntelBackend()
        info = backend.get_device_info()
        assert isinstance(info, DeviceInfo)
        assert info.backend == 'intel'

        # Also test legacy dict format
        info_dict = backend.get_device_info_dict()
        assert isinstance(info_dict, dict)
        assert info_dict['backend'] == 'intel'
        assert 'xpu_available' in info_dict
        assert 'ipex_available' in info_dict

    def test_get_memory_stats(self):
        """Test get_memory_stats."""
        backend = IntelBackend()
        stats = backend.get_memory_stats()
        assert isinstance(stats, dict)
        assert 'allocated' in stats
        assert 'device' in stats

    def test_prepare_model_none(self, simple_model):
        """Test prepare_model with None model."""
        backend = IntelBackend()
        result = backend.prepare_model(None)
        assert result is None

    def test_prepare_model_cpu_fallback(self, simple_model):
        """Test prepare_model falls back to CPU."""
        backend = IntelBackend()
        if not backend.is_xpu_available:
            result = backend.prepare_model(simple_model)
            assert result is simple_model

    def test_synchronize(self):
        """Test synchronize method."""
        backend = IntelBackend()
        # Should not raise even if XPU unavailable
        backend.synchronize()

    def test_empty_cache(self):
        """Test empty_cache method."""
        backend = IntelBackend()
        # Should not raise even if XPU unavailable
        backend.empty_cache()

    def test_memory_summary(self):
        """Test get_memory_summary."""
        backend = IntelBackend()
        summary = backend.get_memory_summary()
        assert isinstance(summary, str)


# ============================================================================
# Test Optimizer
# ============================================================================

class TestIntelOptimizer:
    """Test Intel optimizer."""

    def test_optimizer_creation(self):
        """Test creating optimizer."""
        optimizer = IntelOptimizer()
        assert optimizer is not None
        assert optimizer.optimization_level == IntelOptimizationLevel.O1

    def test_optimization_levels(self):
        """Test optimization level enum."""
        assert IntelOptimizationLevel.O0.value == "O0"
        assert IntelOptimizationLevel.O1.value == "O1"
        assert IntelOptimizationLevel.O2.value == "O2"
        assert IntelOptimizationLevel.O3.value == "O3"

    def test_optimize_o0(self, simple_model):
        """Test O0 (no optimization)."""
        optimizer = IntelOptimizer(optimization_level=IntelOptimizationLevel.O0)
        model, result = optimizer.optimize(simple_model)
        assert result.success is True
        assert len(result.optimizations_applied) == 0

    def test_optimize_o1(self, simple_model):
        """Test O1 (standard optimization)."""
        optimizer = IntelOptimizer(optimization_level=IntelOptimizationLevel.O1)
        model, result = optimizer.optimize(simple_model)
        assert result.success is True
        assert isinstance(result.optimizations_applied, list)

    def test_optimize_returns_result(self, simple_model):
        """Test optimize returns OptimizationResult."""
        optimizer = IntelOptimizer()
        model, result = optimizer.optimize(simple_model)
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, 'success')
        assert hasattr(result, 'optimizations_applied')
        assert hasattr(result, 'warnings')
        assert hasattr(result, 'metrics')


class TestIntelKernelOptimizer:
    """Test Intel kernel optimizer."""

    def test_kernel_optimizer_creation(self):
        """Test creating kernel optimizer."""
        optimizer = IntelKernelOptimizer()
        assert optimizer is not None
        assert optimizer.device_type == "auto"

    def test_gemm_config(self):
        """Test GEMM configuration."""
        optimizer = IntelKernelOptimizer()
        config = optimizer.get_optimal_gemm_config(1024, 1024, 1024)
        assert isinstance(config, dict)
        assert 'algorithm' in config
        assert 'tile_m' in config

    def test_gemm_config_bf16(self):
        """Test GEMM configuration for BF16."""
        optimizer = IntelKernelOptimizer()
        config = optimizer.get_optimal_gemm_config(
            1024, 1024, 1024,
            dtype=torch.bfloat16
        )
        assert config.get('use_amx') is True

    def test_conv_config(self):
        """Test convolution configuration."""
        optimizer = IntelKernelOptimizer()
        config = optimizer.get_optimal_conv_config(64, 128, (3, 3))
        assert isinstance(config, dict)
        assert 'algorithm' in config
        assert config['algorithm'] == 'winograd'  # 3x3 uses Winograd

    def test_attention_config(self):
        """Test attention configuration."""
        optimizer = IntelKernelOptimizer()
        config = optimizer.get_optimal_attention_config(
            seq_len=2048,
            head_dim=64,
            num_heads=8
        )
        assert isinstance(config, dict)
        assert 'algorithm' in config

    def test_attention_config_long_sequence(self):
        """Test attention configuration for long sequences."""
        optimizer = IntelKernelOptimizer()
        config = optimizer.get_optimal_attention_config(
            seq_len=8192,
            head_dim=64,
            num_heads=8
        )
        assert config['algorithm'] in ['chunked', 'memory_efficient']
        assert config['chunk_size'] is not None


# ============================================================================
# Test Optimizations Helper
# ============================================================================

class TestXPUOptimizations:
    """Test XPU optimizations helper."""

    def test_optimizations_creation(self):
        """Test creating optimizations helper."""
        opts = XPUOptimizations()
        assert opts is not None

    def test_get_optimal_dtype(self):
        """Test getting optimal dtype."""
        opts = XPUOptimizations()
        dtype = opts.get_optimal_dtype()
        assert dtype in [torch.float32, torch.float16, torch.bfloat16]

    def test_optimize_model_for_inference(self, simple_model):
        """Test optimize_model_for_inference."""
        opts = XPUOptimizations()
        model = opts.optimize_model_for_inference(simple_model)
        assert model is not None
        # Model should still work
        x = torch.randn(2, 256)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 256)


# ============================================================================
# Test Integration
# ============================================================================

class TestIntelBackendIntegration:
    """Test Intel backend integration."""

    def test_backend_model_workflow(self, simple_model):
        """Test complete model workflow."""
        backend = IntelBackend()

        # Prepare model
        backend.prepare_model(simple_model)

        # Get device info
        info = backend.get_device_info()
        assert info.backend == 'intel'

        # Check memory
        stats = backend.get_memory_stats()
        assert 'allocated' in stats

    def test_config_to_backend_workflow(self):
        """Test config -> backend workflow."""
        # Create config
        config = TorchBridgeConfig()

        # Create backend with config
        backend = IntelBackend(config=config)

        # Verify backend uses config
        assert backend.config is config

    def test_optimizer_integration(self, simple_model):
        """Test optimizer integration with backend."""
        backend = IntelBackend()
        optimizer = IntelOptimizer(optimization_level=IntelOptimizationLevel.O1)

        # Optimize model
        optimized_model, result = optimizer.optimize(simple_model)

        # Prepare with backend
        prepared_model = backend.prepare_model(optimized_model)

        # Model should still work
        x = torch.randn(2, 256)
        with torch.no_grad():
            output = prepared_model(x)
        assert output.shape == (2, 256)


# ============================================================================
# Conditional XPU Tests (only run if XPU is available)
# ============================================================================

@pytest.mark.skipif(not XPU_AVAILABLE, reason="Intel XPU not available")
class TestIntelBackendWithXPU:
    """Tests that require actual XPU hardware."""

    def test_device_is_xpu(self):
        """Test device is XPU when available."""
        backend = IntelBackend()
        assert backend.device.type == "xpu"

    def test_model_to_xpu(self, simple_model):
        """Test moving model to XPU."""
        backend = IntelBackend()
        model = backend.prepare_model(simple_model)
        assert next(model.parameters()).device.type == "xpu"

    def test_tensor_allocation(self):
        """Test tensor allocation on XPU."""
        backend = IntelBackend()
        if backend.memory_manager:
            tensor = backend.memory_manager.allocate_tensor(
                shape=(64, 64),
                dtype=torch.float32,
                purpose="test"
            )
            assert tensor.device.type == "xpu"
            assert tensor.shape == (64, 64)


@pytest.mark.skipif(not IPEX_AVAILABLE, reason="IPEX not available")
class TestIntelBackendWithIPEX:
    """Tests that require IPEX."""

    def test_ipex_optimization(self, simple_model):
        """Test IPEX optimization."""
        backend = IntelBackend()
        model = backend.optimize_for_inference(simple_model)
        assert model is not None

    def test_ipex_version(self):
        """Test IPEX version is available."""
        version = get_ipex_version()
        assert version is not None
        assert isinstance(version, str)
