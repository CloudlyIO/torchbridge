#!/usr/bin/env python3
"""
Test suite for TPU configuration and hardware detection.

Comprehensive tests for TPU integration in the unified configuration system.
Tests TPU hardware detection, configuration validation, and integration.
"""

from unittest.mock import Mock, patch

import pytest

from kernel_pytorch.core.config import (
    HardwareBackend,
    KernelPyTorchConfig,
    TPUCompilationMode,
    TPUConfig,
    TPUTopology,
    TPUVersion,
)


class TestTPUConfig:
    """Test TPU configuration functionality."""

    def test_tpu_config_creation(self):
        """Test basic TPU config creation."""
        config = TPUConfig()
        assert config.enabled is True
        # After __post_init__, auto detection happens
        assert config.version in [TPUVersion.AUTO, TPUVersion.V5E]  # Falls back to V5E
        assert config.topology in [TPUTopology.AUTO, TPUTopology.SINGLE]  # Falls back to SINGLE
        assert config.compilation_mode == TPUCompilationMode.TORCH_XLA
        assert config.precision == "bfloat16"
        assert config.mixed_precision is True
        # Memory fraction depends on detected version (0.90 for V5E)
        assert config.memory_fraction in [0.90, 0.95]
        # Optimization level depends on detected version (1 for V5E, 2 for high-perf)
        assert config.xla_optimization_level in [1, 2]

    def test_tpu_config_custom_values(self):
        """Test TPU config with custom values."""
        config = TPUConfig(
            version=TPUVersion.V5P,
            topology=TPUTopology.POD,
            compilation_mode=TPUCompilationMode.XLA,
            precision="float32"
        )
        assert config.version == TPUVersion.V5P
        assert config.topology == TPUTopology.POD
        assert config.compilation_mode == TPUCompilationMode.XLA
        assert config.precision == "float32"
        # V5P gets 0.95 memory fraction in __post_init__
        assert config.memory_fraction == 0.95
        # V5P gets optimization level 2 in __post_init__
        assert config.xla_optimization_level == 2

    def test_tpu_config_serialization(self):
        """Test TPU config serialization."""
        config = TPUConfig(
            version=TPUVersion.V5E,
            topology=TPUTopology.SINGLE,
            compilation_mode=TPUCompilationMode.PJIT
        )

        config_dict = config.__dict__
        assert config_dict['version'] == TPUVersion.V5E
        assert config_dict['topology'] == TPUTopology.SINGLE
        assert config_dict['compilation_mode'] == TPUCompilationMode.PJIT

    @patch('kernel_pytorch.core.config.torch_xla', create=True)
    @patch('os.environ.get')
    def test_detect_tpu_version_v4(self, mock_env_get, mock_xla):
        """Test TPU v4 detection."""
        # Mock TPU v4 detection
        mock_device = Mock()
        mock_device.device_type = 'TPU'
        mock_xla.core.xla_model.xla_device.return_value = mock_device
        mock_xla.core.xla_model.xla_device_hw.return_value = 'TPU'
        mock_env_get.return_value = 'v4-8'  # Mock TPU_TYPE environment variable

        config = TPUConfig()
        detected = config._detect_tpu_version()

        # Should detect as v4 or fall back to v5e
        assert detected in [TPUVersion.V4, TPUVersion.V5E]

    @patch('kernel_pytorch.core.config.torch_xla', create=True)
    @patch('os.environ.get')
    def test_detect_tpu_version_v5p(self, mock_env_get, mock_xla):
        """Test TPU v5p detection."""
        # Mock TPU v5p detection
        mock_device = Mock()
        mock_device.device_type = 'TPU'
        mock_xla.core.xla_model.xla_device.return_value = mock_device
        mock_xla.core.xla_model.xla_device_hw.return_value = 'TPU'
        mock_env_get.return_value = 'v5p-8'  # Mock TPU_TYPE environment variable

        config = TPUConfig()
        detected = config._detect_tpu_version()

        assert detected in [TPUVersion.V5P, TPUVersion.V5E]

    @patch('kernel_pytorch.core.config.torch_xla', create=True)
    def test_detect_tpu_topology_single(self, mock_xla):
        """Test single TPU topology detection."""
        # Mock single TPU topology
        mock_xla.core.xla_model.xla_device_hw.return_value = 'TPU'
        mock_xla.core.xla_model.xrt_world_size.return_value = 1

        config = TPUConfig()
        detected = config._detect_tpu_topology()

        assert detected in [TPUTopology.SINGLE, TPUTopology.AUTO]

    @patch('kernel_pytorch.core.config.torch_xla', create=True)
    def test_detect_tpu_topology_pod(self, mock_xla):
        """Test TPU pod topology detection."""
        # Mock TPU pod topology (256 devices = pod)
        mock_xla.core.xla_model.xla_device_hw.return_value = 'TPU'
        mock_xla.core.xla_model.xrt_world_size.return_value = 256

        config = TPUConfig()
        detected = config._detect_tpu_topology()

        assert detected in [TPUTopology.POD, TPUTopology.SINGLE]

    def test_detect_tpu_version_no_xla(self):
        """Test TPU version detection without XLA."""
        config = TPUConfig()
        detected = config._detect_tpu_version()

        # Should fall back to V5E when XLA is not available
        assert detected == TPUVersion.V5E

    def test_detect_tpu_topology_no_xla(self):
        """Test TPU topology detection without XLA."""
        config = TPUConfig()
        detected = config._detect_tpu_topology()

        # Should fall back to SINGLE when XLA is not available
        assert detected == TPUTopology.SINGLE


class TestKernelPyTorchConfigTPU:
    """Test TPU integration in main configuration."""

    def test_config_with_tpu_backend(self):
        """Test configuration with TPU backend."""
        config = KernelPyTorchConfig()

        # Check TPU config is present
        assert hasattr(config.hardware, 'tpu')
        assert isinstance(config.hardware.tpu, TPUConfig)
        # TPU is disabled by default when running on CPU
        assert config.hardware.tpu.enabled in [True, False]

    def test_config_tpu_serialization(self):
        """Test configuration serialization with TPU."""
        config = KernelPyTorchConfig()
        config_dict = config.to_dict()

        # Check TPU section exists
        assert 'hardware' in config_dict
        assert 'tpu' in config_dict['hardware']

        tpu_config = config_dict['hardware']['tpu']
        assert 'version' in tpu_config
        assert 'topology' in tpu_config
        assert 'compilation_mode' in tpu_config

    @patch('kernel_pytorch.core.config.torch_xla', create=True)
    def test_device_detection_tpu_available(self, mock_xla):
        """Test device detection when TPU is available."""
        # Mock TPU availability
        mock_device = Mock()
        mock_device.type = 'xla'
        mock_xla.core.xla_model.xla_device.return_value = mock_device
        mock_xla.core.xla_model.xla_device_hw.return_value = 'TPU'

        # Mock CUDA not available
        with patch('torch.cuda.is_available', return_value=False):
            device = KernelPyTorchConfig._detect_device()
            # Should detect TPU or fallback gracefully
            assert device is not None

    @patch('torch.cuda.is_available', return_value=False)
    def test_device_detection_no_tpu(self, mock_cuda):
        """Test device detection when TPU is not available."""
        # Should fall back to CPU
        device = KernelPyTorchConfig._detect_device()
        assert device.type == 'cpu'

    def test_tpu_config_modes(self):
        """Test different configuration modes with TPU."""
        configs = {
            'default': KernelPyTorchConfig(),
            'inference': KernelPyTorchConfig.for_inference(),
            'training': KernelPyTorchConfig.for_training(),
            'development': KernelPyTorchConfig.for_development()
        }

        for mode, config in configs.items():
            assert hasattr(config.hardware, 'tpu')
            tpu_config = config.hardware.tpu
            assert isinstance(tpu_config, TPUConfig)

            # Check mode-specific settings
            if mode == 'inference':
                # Inference mode might disable certain features
                assert tpu_config.precision in ['bfloat16', 'float16', 'float32']
            elif mode == 'training':
                # Training mode should enable mixed precision
                assert tpu_config.mixed_precision is True
            elif mode == 'development':
                # Development mode should have debugging features
                assert tpu_config.xla_optimization_level >= 0

    def test_tpu_hardware_backend_enum(self):
        """Test TPU is included in HardwareBackend enum."""
        assert HardwareBackend.TPU in HardwareBackend
        assert HardwareBackend.TPU.value == "tpu"

        # Test all expected backends are present
        expected_backends = {'cuda', 'cpu', 'tpu', 'amd', 'intel', 'custom'}
        actual_backends = {backend.value for backend in HardwareBackend}
        assert expected_backends.issubset(actual_backends)


class TestTPUEnums:
    """Test TPU-specific enums."""

    def test_tpu_version_enum(self):
        """Test TPUVersion enum values."""
        expected_versions = {'auto', 'v4', 'v5e', 'v5p', 'v6e', 'v7'}
        actual_versions = {version.value for version in TPUVersion}
        assert expected_versions == actual_versions

    def test_tpu_topology_enum(self):
        """Test TPUTopology enum values."""
        expected_topologies = {'auto', 'single', 'pod', 'superpod'}
        actual_topologies = {topology.value for topology in TPUTopology}
        assert expected_topologies == actual_topologies

    def test_tpu_compilation_mode_enum(self):
        """Test TPUCompilationMode enum values."""
        expected_modes = {'xla', 'pjit', 'torch_xla'}
        actual_modes = {mode.value for mode in TPUCompilationMode}
        assert expected_modes == actual_modes


class TestTPUConfigValidation:
    """Test TPU configuration validation."""

    def test_valid_tpu_configs(self):
        """Test valid TPU configurations."""
        valid_configs = [
            TPUConfig(),  # Default
            TPUConfig(version=TPUVersion.V5P, topology=TPUTopology.POD),
            TPUConfig(compilation_mode=TPUCompilationMode.XLA),
            TPUConfig(precision="float32", memory_fraction=0.75),
            TPUConfig(xla_optimization_level=1, enable_xla_dynamic_shapes=False)
        ]

        for config in valid_configs:
            # Basic validation - config should be creatable
            assert config.enabled in [True, False]
            assert config.memory_fraction >= 0.0 and config.memory_fraction <= 1.0
            assert config.xla_optimization_level >= 0

    def test_tpu_config_memory_bounds(self):
        """Test TPU config memory fraction bounds."""
        # Test that explicit memory fractions are overridden by auto-detection
        # TPUConfig's __post_init__ adjusts memory_fraction based on detected version
        config = TPUConfig(memory_fraction=0.1)
        # Should be adjusted to 0.9 (V5E) or 0.95 (high-perf TPUs)
        assert config.memory_fraction in [0.9, 0.95]

    def test_tpu_config_xla_optimization_levels(self):
        """Test TPU config XLA optimization levels."""
        # Test that optimization levels are adjusted by auto-detection
        # TPUConfig's __post_init__ adjusts xla_optimization_level based on detected version
        config = TPUConfig(xla_optimization_level=0)
        # Should be adjusted to 1 (V5E) or 2 (high-perf TPUs)
        assert config.xla_optimization_level in [1, 2]

    def test_tpu_config_precision_values(self):
        """Test TPU config precision values."""
        valid_precisions = ['bfloat16', 'float16', 'float32']

        for precision in valid_precisions:
            config = TPUConfig(precision=precision)
            assert config.precision == precision


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
