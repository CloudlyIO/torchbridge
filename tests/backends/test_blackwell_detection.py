"""
Tests for NVIDIA Blackwell GPU Detection

Tests detection of Blackwell data center (sm_100, cc 10.0) and
consumer (sm_120, cc 12.0) GPUs across all detection layers:
- NVIDIAConfig architecture detection
- HardwareDetector compute capability detection
- NVIDIAAdapter vendor adapter generation mapping
- FP4 precision capability detection
"""

from unittest.mock import MagicMock, patch

import pytest

from torchbridge.core.config import (
    NVIDIAArchitecture,
    NVIDIAConfig,
    PrecisionFormat,
)
from torchbridge.core.hardware_detector import (
    HardwareDetector,
    HardwareProfile,
    HardwareType,
    OptimizationCapability,
)


class TestBlackwellDCDetection:
    """Test Blackwell Data Center (B100/B200/GB200) detection — sm_100, cc 10.0."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_detect_b100_by_name(self, mock_props, mock_cuda):
        """Test B100 detection via device name."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA B100"
        mock_device_props.major = 10
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.BLACKWELL_DC

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_detect_b200_by_name(self, mock_props, mock_cuda):
        """Test B200 detection via device name."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA B200"
        mock_device_props.major = 10
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.BLACKWELL_DC

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_detect_gb200_by_name(self, mock_props, mock_cuda):
        """Test GB200 (dual-die) detection via device name."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA GB200"
        mock_device_props.major = 10
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.BLACKWELL_DC

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_detect_blackwell_dc_by_compute_capability(self, mock_props, mock_cuda):
        """Test Blackwell DC detection via compute capability 10.0 fallback."""
        mock_device_props = MagicMock()
        mock_device_props.name = "Unknown NVIDIA GPU"
        mock_device_props.major = 10
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.BLACKWELL_DC

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_blackwell_dc_fp8_enabled(self, mock_props, mock_cuda):
        """Test that FP8 is enabled on Blackwell DC."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA B200"
        mock_device_props.major = 10
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.fp8_enabled is True

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_blackwell_dc_tensor_core_v5(self, mock_props, mock_cuda):
        """Test that Blackwell DC gets Tensor Core v5."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA B200"
        mock_device_props.major = 10
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.tensor_core_version == 5


class TestBlackwellConsumerDetection:
    """Test Blackwell Consumer (RTX 5090/5080) detection — sm_120, cc 12.0."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_detect_rtx5090_by_name(self, mock_props, mock_cuda):
        """Test RTX 5090 detection via device name."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA GeForce RTX 5090"
        mock_device_props.major = 12
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.BLACKWELL_CONSUMER

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_detect_rtx5080_by_name(self, mock_props, mock_cuda):
        """Test RTX 5080 detection via device name."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA GeForce RTX 5080"
        mock_device_props.major = 12
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.BLACKWELL_CONSUMER

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_detect_consumer_by_compute_capability(self, mock_props, mock_cuda):
        """Test Blackwell Consumer detection via compute capability 12.0 fallback."""
        mock_device_props = MagicMock()
        mock_device_props.name = "Unknown NVIDIA GPU"
        mock_device_props.major = 12
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.architecture == NVIDIAArchitecture.BLACKWELL_CONSUMER

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_consumer_fp8_enabled(self, mock_props, mock_cuda):
        """Test that FP8 is enabled on Blackwell Consumer."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA GeForce RTX 5090"
        mock_device_props.major = 12
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.fp8_enabled is True

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_consumer_tensor_core_v5(self, mock_props, mock_cuda):
        """Test that Blackwell Consumer gets Tensor Core v5."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA GeForce RTX 5090"
        mock_device_props.major = 12
        mock_device_props.minor = 0
        mock_props.return_value = mock_device_props

        config = NVIDIAConfig()
        assert config.tensor_core_version == 5


class TestHardwareDetectorBlackwell:
    """Test HardwareDetector for Blackwell compute capabilities."""

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    @patch('torchbridge.core.hardware_detector.HardwareDetector._detect_amd_gpu', return_value=None)
    def test_detector_blackwell_dc(self, mock_amd, mock_props, mock_count, mock_cuda):
        """Test HardwareDetector identifies Blackwell DC (cc 10.0)."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA B200"
        mock_device_props.major = 10
        mock_device_props.minor = 0
        mock_device_props.total_memory = 192 * 1024**3
        mock_device_props.multi_processor_count = 132
        mock_props.return_value = mock_device_props

        detector = HardwareDetector()
        profile = detector.detect(force_redetect=True)

        assert profile.hardware_type == HardwareType.NVIDIA_GPU
        assert profile.nvidia_architecture == NVIDIAArchitecture.BLACKWELL_DC
        assert profile.compute_capability == (10, 0)
        assert OptimizationCapability.FP8_TRAINING in profile.capabilities
        assert OptimizationCapability.FLASH_ATTENTION_3 in profile.capabilities

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=1)
    @patch('torch.cuda.get_device_properties')
    @patch('torchbridge.core.hardware_detector.HardwareDetector._detect_amd_gpu', return_value=None)
    def test_detector_blackwell_consumer(self, mock_amd, mock_props, mock_count, mock_cuda):
        """Test HardwareDetector identifies Blackwell Consumer (cc 12.0)."""
        mock_device_props = MagicMock()
        mock_device_props.name = "NVIDIA GeForce RTX 5090"
        mock_device_props.major = 12
        mock_device_props.minor = 0
        mock_device_props.total_memory = 32 * 1024**3
        mock_device_props.multi_processor_count = 170
        mock_props.return_value = mock_device_props

        detector = HardwareDetector()
        profile = detector.detect(force_redetect=True)

        assert profile.hardware_type == HardwareType.NVIDIA_GPU
        assert profile.nvidia_architecture == NVIDIAArchitecture.BLACKWELL_CONSUMER
        assert profile.compute_capability == (12, 0)


class TestBlackwellIsH100OrBetter:
    """Test that Blackwell is recognized as H100-or-better tier."""

    def test_blackwell_dc_is_h100_or_better(self):
        """Blackwell DC should be recognized as H100 or better."""
        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA B200",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.BLACKWELL_DC,
        )
        assert profile.is_nvidia_h100_or_better()

    def test_blackwell_consumer_is_h100_or_better(self):
        """Blackwell Consumer should be recognized as H100 or better."""
        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA RTX 5090",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.BLACKWELL_CONSUMER,
        )
        assert profile.is_nvidia_h100_or_better()

    def test_hopper_is_h100_or_better(self):
        """Hopper should still be recognized as H100 or better."""
        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA H100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.HOPPER,
        )
        assert profile.is_nvidia_h100_or_better()

    def test_ampere_is_not_h100_or_better(self):
        """Ampere should NOT be recognized as H100 or better."""
        profile = HardwareProfile(
            hardware_type=HardwareType.NVIDIA_GPU,
            device_name="NVIDIA A100",
            device_count=1,
            nvidia_architecture=NVIDIAArchitecture.AMPERE,
        )
        assert not profile.is_nvidia_h100_or_better()


class TestBlackwellEnumValues:
    """Test the enum values for Blackwell architectures."""

    def test_blackwell_dc_enum_value(self):
        """BLACKWELL_DC should have value 'blackwell_dc'."""
        assert NVIDIAArchitecture.BLACKWELL_DC.value == "blackwell_dc"

    def test_blackwell_consumer_enum_value(self):
        """BLACKWELL_CONSUMER should have value 'blackwell_consumer'."""
        assert NVIDIAArchitecture.BLACKWELL_CONSUMER.value == "blackwell_consumer"

    def test_no_legacy_blackwell_enum(self):
        """Old BLACKWELL enum should no longer exist."""
        assert not hasattr(NVIDIAArchitecture, 'BLACKWELL')


@pytest.mark.blackwell
class TestBlackwellPrecisionSupport:
    """Test precision support on Blackwell architectures."""

    def test_fp4_enum_exists(self):
        """FP4 precision format should exist in enum."""
        assert PrecisionFormat.FP4.value == "fp4"

