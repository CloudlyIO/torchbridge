"""
Tests for AMD CDNA 4 and MI325X GPU Detection

Tests detection of CDNA 4 (MI350X/MI355X, gfx950) and MI325X (gfx942)
GPUs across all detection layers:
- AMDConfig architecture detection
- AMDBackend architecture detection
- AMDAdapter vendor adapter mapping
- ROCm compiler GPU target mapping
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

    def test_mi325x_compiler_target(self):
        """MI325X should compile to gfx942 target."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        compiler = ROCmCompiler(config)
        assert compiler._get_gpu_target() == "gfx942"


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

    def test_cdna4_compiler_target(self):
        """CDNA4 should compile to gfx950 target."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        compiler = ROCmCompiler(config)
        assert compiler._get_gpu_target() == "gfx950"


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


class TestCDNA4FP8Support:
    """Test FP8 support on CDNA3 and CDNA4."""

    def test_cdna4_fp8_optimizer(self):
        """CDNA4 optimizer should enable FP8 quantization."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        optimizer = AMDOptimizer(config)

        # FP8 should not raise a warning for CDNA4
        model = torch.nn.Linear(64, 32)
        result = optimizer._prepare_fp8_quantization(model)
        # Returns True on CDNA4 (FP8 supported)
        assert result is True

    def test_cdna3_fp8_optimizer(self):
        """CDNA3 optimizer should also enable FP8 quantization."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        optimizer = AMDOptimizer(config)
        model = torch.nn.Linear(64, 32)
        result = optimizer._prepare_fp8_quantization(model)
        assert result is True

    def test_cdna2_no_fp8(self):
        """CDNA2 should NOT support FP8."""
        from torchbridge.backends.amd.amd_optimizer import AMDOptimizer

        config = AMDConfig(architecture=AMDArchitecture.CDNA2)
        optimizer = AMDOptimizer(config)
        model = torch.nn.Linear(64, 32)
        result = optimizer._prepare_fp8_quantization(model)
        assert result is False


class TestCDNA4CompilerFlags:
    """Test HIP compiler flags for CDNA4."""

    def test_cdna4_matrix_core_flags(self):
        """CDNA4 should get Matrix Core compiler flags."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig(architecture=AMDArchitecture.CDNA4)
        compiler = ROCmCompiler(config)
        flags = compiler._get_optimization_flags("balanced")

        assert "--amdgpu-target=gfx950" in flags
        assert "-mwavefrontsize64" in flags
        assert "-mcumode" in flags

    def test_cdna3_compiler_flags(self):
        """CDNA3 should also get Matrix Core compiler flags."""
        from torchbridge.backends.amd.rocm_compiler import ROCmCompiler

        config = AMDConfig(architecture=AMDArchitecture.CDNA3)
        compiler = ROCmCompiler(config)
        flags = compiler._get_optimization_flags("aggressive")

        assert "--amdgpu-target=gfx942" in flags
        assert "-mwavefrontsize64" in flags


class TestVendorAdapterCDNA4:
    """Test AMDAdapter vendor adapter for CDNA4."""

    def test_gfx950_architecture_mapping(self):
        """Test vendor adapter maps gfx950 to CDNA4."""
        from torchbridge.hardware.abstraction.vendor_adapters import AMDAdapter

        adapter = AMDAdapter()
        assert 'gfx950' in adapter.gpu_architectures
        assert adapter.gpu_architectures['gfx950'] == 'CDNA4'

    def test_gfx942_architecture_mapping(self):
        """Test vendor adapter maps gfx942 to CDNA3."""
        from torchbridge.hardware.abstraction.vendor_adapters import AMDAdapter

        adapter = AMDAdapter()
        assert 'gfx942' in adapter.gpu_architectures
        assert adapter.gpu_architectures['gfx942'] == 'CDNA3'

    def test_gfx950_flops_estimate(self):
        """Test MI350X peak FLOPS estimate."""
        from torchbridge.hardware.abstraction.vendor_adapters import AMDAdapter

        adapter = AMDAdapter()
        flops = adapter._estimate_amd_peak_flops({'compute_capability': 'gfx950'})
        assert flops == 240e12  # 240 TF FP32

    def test_gfx942_flops_estimate(self):
        """Test MI325X peak FLOPS estimate."""
        from torchbridge.hardware.abstraction.vendor_adapters import AMDAdapter

        adapter = AMDAdapter()
        flops = adapter._estimate_amd_peak_flops({'compute_capability': 'gfx942'})
        assert flops == 165e12  # 165 TF FP32

    def test_gfx950_bandwidth_estimate(self):
        """Test MI350X memory bandwidth estimate."""
        from torchbridge.hardware.abstraction.vendor_adapters import AMDAdapter

        adapter = AMDAdapter()
        bw = adapter._estimate_amd_memory_bandwidth({'compute_capability': 'gfx950'})
        assert bw == 8000  # 8 TB/s HBM3e

    def test_gfx942_bandwidth_estimate(self):
        """Test MI325X memory bandwidth estimate."""
        from torchbridge.hardware.abstraction.vendor_adapters import AMDAdapter

        adapter = AMDAdapter()
        bw = adapter._estimate_amd_memory_bandwidth({'compute_capability': 'gfx942'})
        assert bw == 6000  # 6 TB/s HBM3e

    def test_gfx950_has_matrix_cores(self):
        """Test CDNA4 has Matrix Cores."""
        from torchbridge.hardware.abstraction.vendor_adapters import AMDAdapter

        adapter = AMDAdapter()
        assert adapter._has_matrix_cores({'compute_capability': 'gfx950'}) is True

    def test_gfx950_infinity_fabric(self):
        """Test CDNA4 uses Infinity Fabric."""
        from torchbridge.hardware.abstraction.vendor_adapters import AMDAdapter

        adapter = AMDAdapter()
        assert adapter._get_amd_interconnect_type({'compute_capability': 'gfx950'}) == "Infinity Fabric"


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
