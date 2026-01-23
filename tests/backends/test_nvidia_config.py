"""
Test NVIDIA Configuration Integration

Tests the new NVIDIA configuration system in KernelPyTorchConfig.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from kernel_pytorch.core.config import (
    KernelPyTorchConfig,
    NVIDIAConfig,
    NVIDIAArchitecture,
    HardwareBackend
)


class TestNVIDIAConfig:
    """Test NVIDIA configuration functionality."""

    @patch('torch.cuda.is_available', return_value=False)
    def test_nvidia_config_creation(self, mock_cuda):
        """Test basic NVIDIA config creation."""
        config = NVIDIAConfig()
        assert config.enabled is True
        # Architecture gets auto-detected in __post_init__, so check the detected value
        assert config.architecture in [NVIDIAArchitecture.AUTO, NVIDIAArchitecture.PASCAL]
        assert config.flash_attention_version == "3"

    def test_nvidia_architecture_detection_no_cuda(self):
        """Test architecture detection when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            config = NVIDIAConfig()
            assert config.architecture == NVIDIAArchitecture.PASCAL

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_nvidia_architecture_detection_h100(self, mock_props, mock_cuda):
        """Test H100 architecture detection."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA H100 80GB HBM3"
        mock_device_props.major = 9
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.HOPPER
        assert config.fp8_enabled is True
        assert config.tensor_core_version == 4

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_nvidia_architecture_detection_a100(self, mock_props, mock_cuda):
        """Test A100 architecture detection."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA A100-PCIE-40GB"
        mock_device_props.major = 8
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.AMPERE
        assert config.fp8_enabled is False  # A100 doesn't support FP8
        assert config.tensor_core_version == 3

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_nvidia_config_fp8_settings(self, mock_props, mock_cuda):
        """Test FP8 configuration settings."""
        # Mock H100 device to enable FP8
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA H100 80GB HBM3"
        mock_device_props.major = 9
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.fp8_enabled is True  # Should be enabled for H100
        assert config.fp8_recipe == "DelayedScaling"

    def test_nvidia_config_memory_settings(self):
        """Test memory configuration settings."""
        config = NVIDIAConfig(
            memory_pool_enabled=True,
            memory_fraction=0.90,
            kernel_fusion_enabled=True
        )
        assert config.memory_pool_enabled is True
        assert config.memory_fraction == 0.90
        assert config.kernel_fusion_enabled is True


class TestKernelPyTorchConfigNVIDIA:
    """Test NVIDIA integration in main configuration."""

    def test_kernelpytorch_config_nvidia_integration(self):
        """Test NVIDIA config integration in main config."""
        config = KernelPyTorchConfig()
        assert hasattr(config.hardware, 'nvidia')
        assert isinstance(config.hardware.nvidia, NVIDIAConfig)

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_hardware_config_cuda_auto_enable(self, mock_props, mock_cuda):
        """Test NVIDIA auto-enable when CUDA is available."""
        # Mock device properties to avoid CUDA calls
        mock_device_props = MagicMock()
        mock_device_props.total_memory = 85899345920  # 80GB in bytes
        mock_props.return_value = mock_device_props

        config = KernelPyTorchConfig()
        assert config.hardware.backend == HardwareBackend.CUDA
        assert config.hardware.nvidia.enabled is True

    @patch('torch.cuda.is_available', return_value=False)
    def test_hardware_config_cpu_fallback(self, mock_cuda):
        """Test CPU fallback when CUDA is not available."""
        config = KernelPyTorchConfig()
        # Should still create NVIDIA config but adapt for CPU
        assert hasattr(config.hardware, 'nvidia')
        assert config.device.type == "cpu"

    def test_config_modes_nvidia_settings(self):
        """Test different config modes preserve NVIDIA settings."""
        # Inference mode
        inference_config = KernelPyTorchConfig.for_inference()
        assert hasattr(inference_config.hardware, 'nvidia')

        # Training mode
        training_config = KernelPyTorchConfig.for_training()
        assert hasattr(training_config.hardware, 'nvidia')

        # Development mode
        dev_config = KernelPyTorchConfig.for_development()
        assert hasattr(dev_config.hardware, 'nvidia')

    def test_config_update_nvidia_settings(self):
        """Test updating NVIDIA settings through config update."""
        config = KernelPyTorchConfig()

        # Should be able to access and modify NVIDIA settings
        config.hardware.nvidia.fp8_enabled = False
        config.hardware.nvidia.flash_attention_version = "2"

        assert config.hardware.nvidia.fp8_enabled is False
        assert config.hardware.nvidia.flash_attention_version == "2"

    def test_config_to_dict_includes_nvidia(self):
        """Test config serialization includes NVIDIA settings."""
        config = KernelPyTorchConfig()
        config_dict = config.to_dict()

        assert 'hardware' in config_dict
        assert 'nvidia' in config_dict['hardware']
        assert 'fp8_enabled' in config_dict['hardware']['nvidia']


if __name__ == "__main__":
    pytest.main([__file__])