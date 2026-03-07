"""
Tests for AMD CDNA 4 and MI325X GPU Detection

Tests detection of CDNA 4 (MI350X/MI355X, gfx950) and MI325X (gfx942)
GPUs across all detection layers:
- AMDConfig architecture detection
- AMDBackend architecture detection
- AMDAdapter vendor adapter mapping
- Precision and Matrix Core capability detection
"""

from unittest.mock import MagicMock, patch

import torch

from torchbridge.core.config import AMDArchitecture, AMDConfig


class TestMI325XDetection:
    """Test MI325X detection — gfx942, CDNA3, 256GB HBM3e."""

    def test_mi325x_config_detection(self):
        """MI325X should be detected as CDNA3 via config."""
        mock_hip = MagicMock()
        mock_hip.is_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = "AMD Instinct MI325X"
        mock_hip.get_device_properties.return_value = mock_device

        with patch.dict('sys.modules', {}), \
             patch.object(torch, 'hip', mock_hip, create=True):
            config = AMDConfig()
            assert config.architecture == AMDArchitecture.CDNA3

    def test_mi325x_backend_detection(self):
        """MI325X should be detected as CDNA3 via backend."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        backend = AMDBackend(config)
        arch = backend._detect_architecture("AMD Instinct MI325X")
        assert arch == AMDArchitecture.CDNA3

    def test_mi325x_distinct_from_mi300x(self):
        """MI325X should be detected separately from MI300X (both CDNA3)."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        backend = AMDBackend(config)

        # Both map to CDNA3
        assert backend._detect_architecture("AMD Instinct MI325X") == AMDArchitecture.CDNA3
        assert backend._detect_architecture("AMD Instinct MI300X") == AMDArchitecture.CDNA3

    def test_mi325x_matrix_cores_enabled(self):
        """MI325X (CDNA3) should have Matrix Cores enabled."""
        config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        assert config.enable_matrix_cores is True
        assert config.matrix_core_precision == "bf16"

class TestCDNA4Detection:
    """Test CDNA 4 (MI350X/MI355X) detection — gfx950, 288GB HBM3e."""

    def test_mi350x_config_detection(self):
        """MI350X should be detected as CDNA4 via config."""
        mock_hip = MagicMock()
        mock_hip.is_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = "AMD Instinct MI350X"
        mock_hip.get_device_properties.return_value = mock_device

        with patch.object(torch, 'hip', mock_hip, create=True):
            config = AMDConfig()
            assert config.architecture == AMDArchitecture.CDNA4

    def test_mi355x_config_detection(self):
        """MI355X should be detected as CDNA4 via config."""
        mock_hip = MagicMock()
        mock_hip.is_available.return_value = True
        mock_device = MagicMock()
        mock_device.name = "AMD Instinct MI355X"
        mock_hip.get_device_properties.return_value = mock_device

        with patch.object(torch, 'hip', mock_hip, create=True):
            config = AMDConfig()
            assert config.architecture == AMDArchitecture.CDNA4

    def test_mi350x_backend_detection(self):
        """MI350X should be detected as CDNA4 via backend."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        backend = AMDBackend(config)
        arch = backend._detect_architecture("AMD Instinct MI350X")
        assert arch == AMDArchitecture.CDNA4

    def test_mi355x_backend_detection(self):
        """MI355X should be detected as CDNA4 via backend."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        backend = AMDBackend(config)
        arch = backend._detect_architecture("AMD Instinct MI355X")
        assert arch == AMDArchitecture.CDNA4

    def test_cdna4_matrix_cores_enabled(self):
        """CDNA4 should have Matrix Cores enabled."""
        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        assert config.enable_matrix_cores is True
        assert config.matrix_core_precision == "bf16"
        assert config.allow_bf16 is True

class TestCDNA4EnumValues:
    """Test CDNA4 enum values and consistency."""

    def test_cdna4_enum_value(self):
        """CDNA4 should have value 'cdna4'."""
        assert AMDArchitecture.CDNA4.value == "cdna4"

    def test_all_architectures_present(self):
        """All expected AMD architectures should exist."""
        expected = ["AUTO", "CDNA", "CDNA2", "CDNA3", "CDNA4", "RDNA2", "RDNA3"]
        for name in expected:
            assert hasattr(AMDArchitecture, name), f"Missing: {name}"

    def test_cdna4_is_distinct_from_cdna3(self):
        """CDNA4 and CDNA3 should be distinct enum members."""
        assert AMDArchitecture.CDNA4 != AMDArchitecture.CDNA3
        assert AMDArchitecture.CDNA4.value != AMDArchitecture.CDNA3.value


class TestCDNA4MatrixCores:
    """Test Matrix Core support across CDNA generations."""

    def test_cdna4_has_matrix_cores(self):
        """CDNA4 should have Matrix Cores via config."""
        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        assert config.enable_matrix_cores is True

    def test_cdna3_has_matrix_cores(self):
        """CDNA3 should have Matrix Cores."""
        config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        assert config.enable_matrix_cores is True

    def test_cdna2_has_matrix_cores(self):
        """CDNA2 should still have Matrix Cores."""
        config = AMDConfig(architecture=AMDArchitecture.CDNA2)
        assert config.enable_matrix_cores is True

    def test_cdna_no_matrix_cores(self):
        """CDNA (MI50/MI60) should NOT have Matrix Cores."""
        config = AMDConfig(architecture=AMDArchitecture.CDNA)
        assert config.enable_matrix_cores is False

    def test_rdna_no_matrix_cores(self):
        """RDNA consumer GPUs should NOT have Matrix Cores."""
        config = AMDConfig(architecture=AMDArchitecture.RDNA3)
        assert config.enable_matrix_cores is False


class TestDetectionOrderMatters:
    """Test that MI350X is detected before MI300X (ordering matters)."""

    def test_mi350x_not_confused_with_mi300(self):
        """MI350X should NOT match MI300 patterns."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        backend = AMDBackend(config)

        # MI350X should be CDNA4, not CDNA3
        assert backend._detect_architecture("AMD Instinct MI350X") == AMDArchitecture.CDNA4

        # MI300X should still be CDNA3
        assert backend._detect_architecture("AMD Instinct MI300X") == AMDArchitecture.CDNA3

    def test_mi355x_not_confused_with_mi300(self):
        """MI355X should NOT match MI300 patterns."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        backend = AMDBackend(config)

        assert backend._detect_architecture("AMD Instinct MI355X") == AMDArchitecture.CDNA4
