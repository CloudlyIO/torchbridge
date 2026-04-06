"""
Tests for Quantization Compatibility Matrix

Parametrized tests for every (backend, architecture) -> optimal format mapping,
fallback chain behavior, and format support queries.
"""

import pytest

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.precision.quantization.compatibility import (
    QuantizationCompatibilityMatrix,
)
from torchbridge.precision.quantization.formats import QuantizationFormat

# =============================================================================
# Optimal Format Tests
# =============================================================================


class TestOptimalFormat:
    """Tests for get_optimal_format."""

    @pytest.mark.parametrize(
        "backend, arch, expected",
        [
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.BLACKWELL_DC,
                QuantizationFormat.NVFP4,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.BLACKWELL_CONSUMER,
                QuantizationFormat.FP8_E4M3,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.HOPPER,
                QuantizationFormat.FP8_E4M3,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.AMPERE,
                QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.ADA,
                QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.TURING,
                QuantizationFormat.INT8_DYNAMIC,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.VOLTA,
                QuantizationFormat.INT8_DYNAMIC,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.PASCAL,
                QuantizationFormat.INT8_DYNAMIC,
            ),
            (HardwareBackend.AMD, AMDArchitecture.CDNA4, QuantizationFormat.FP8_E4M3),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3, QuantizationFormat.FP8_E4M3),
            (
                HardwareBackend.AMD,
                AMDArchitecture.CDNA2,
                QuantizationFormat.INT8_DYNAMIC,
            ),
            (
                HardwareBackend.AMD,
                AMDArchitecture.RDNA3,
                QuantizationFormat.INT8_DYNAMIC,
            ),
            (
                HardwareBackend.TRAINIUM,
                TrainiumArchitecture.TRN2,
                QuantizationFormat.FP8_E4M3,
            ),
            (
                HardwareBackend.TRAINIUM,
                TrainiumArchitecture.TRN3,
                QuantizationFormat.FP8_E4M3,
            ),
            (
                HardwareBackend.TRAINIUM,
                TrainiumArchitecture.TRN1,
                QuantizationFormat.BF16,
            ),
            (
                HardwareBackend.TRAINIUM,
                TrainiumArchitecture.INF2,
                QuantizationFormat.BF16,
            ),
            (HardwareBackend.TPU, TPUVersion.V7, QuantizationFormat.FP8_E4M3),
            (HardwareBackend.TPU, TPUVersion.V5E, QuantizationFormat.FP8_E4M3),
            (HardwareBackend.TPU, TPUVersion.V4, QuantizationFormat.BF16),
            (HardwareBackend.CPU, None, QuantizationFormat.INT8_DYNAMIC),
        ],
    )
    def test_optimal_format(self, backend, arch, expected):
        """Optimal format should match the compatibility matrix."""
        result = QuantizationCompatibilityMatrix.get_optimal_format(backend, arch)
        assert result == expected, (
            f"Expected {expected.value} for {backend.value}/{arch}, got {result.value}"
        )


# =============================================================================
# Supported Formats Tests
# =============================================================================


class TestSupportedFormats:
    """Tests for get_supported_formats."""

    def test_all_backends_have_formats(self):
        """Every backend should have at least one supported format."""
        backends = [
            (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
            (HardwareBackend.TPU, TPUVersion.V5E),
            (HardwareBackend.CPU, None),
        ]
        for backend, arch in backends:
            formats = QuantizationCompatibilityMatrix.get_supported_formats(
                backend, arch
            )
            assert len(formats) > 0, f"No formats for {backend.value}"

    def test_optimal_is_first(self):
        """The first format in supported list should be the optimal one."""
        for backend, arch in [
            (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3),
        ]:
            supported = QuantizationCompatibilityMatrix.get_supported_formats(
                backend, arch
            )
            optimal = QuantizationCompatibilityMatrix.get_optimal_format(backend, arch)
            assert supported[0] == optimal

    def test_blackwell_dc_has_nvfp4(self):
        """Blackwell DC should support NVFP4."""
        formats = QuantizationCompatibilityMatrix.get_supported_formats(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC
        )
        assert QuantizationFormat.NVFP4 in formats

    def test_cpu_has_int8_int4_bf16(self):
        """CPU should support INT8, INT4, and BF16."""
        formats = QuantizationCompatibilityMatrix.get_supported_formats(
            HardwareBackend.CPU
        )
        assert QuantizationFormat.INT8_DYNAMIC in formats
        assert QuantizationFormat.INT4_WEIGHT_ONLY in formats
        assert QuantizationFormat.BF16 in formats

    def test_null_architecture_uses_default(self):
        """None architecture should fall back to a default."""
        formats = QuantizationCompatibilityMatrix.get_supported_formats(
            HardwareBackend.CUDA, None
        )
        assert len(formats) > 0

    def test_auto_architecture_uses_default(self):
        """AUTO architecture should fall back to a default."""
        formats = QuantizationCompatibilityMatrix.get_supported_formats(
            HardwareBackend.CUDA, NVIDIAArchitecture.AUTO
        )
        assert len(formats) > 0


# =============================================================================
# Fallback Chain Tests
# =============================================================================


class TestFallbackChain:
    """Tests for get_fallback_chain."""

    def test_supported_format_returns_itself(self):
        """Requesting a supported format should return just that format."""
        chain = QuantizationCompatibilityMatrix.get_fallback_chain(
            QuantizationFormat.FP8_E4M3,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.HOPPER,
        )
        assert chain == [QuantizationFormat.FP8_E4M3]

    def test_nvfp4_on_amd_falls_back(self):
        """NVFP4 on AMD should fall back to AMD's supported formats."""
        chain = QuantizationCompatibilityMatrix.get_fallback_chain(
            QuantizationFormat.NVFP4,
            HardwareBackend.AMD,
            AMDArchitecture.CDNA3,
        )
        assert QuantizationFormat.NVFP4 not in chain
        assert len(chain) > 0
        assert chain[0] == QuantizationFormat.FP8_E4M3

    def test_fp8_on_cpu_falls_back(self):
        """FP8 on CPU should fall back to INT8/INT4."""
        chain = QuantizationCompatibilityMatrix.get_fallback_chain(
            QuantizationFormat.FP8_E4M3,
            HardwareBackend.CPU,
        )
        assert QuantizationFormat.FP8_E4M3 not in chain
        assert QuantizationFormat.INT8_DYNAMIC in chain

    def test_fallback_chain_non_empty(self):
        """Fallback chains should never be empty."""
        for backend in HardwareBackend:
            if backend == HardwareBackend.CUSTOM:
                continue
            chain = QuantizationCompatibilityMatrix.get_fallback_chain(
                QuantizationFormat.NVFP4, backend
            )
            assert len(chain) > 0, f"Empty fallback for {backend.value}"


# =============================================================================
# is_format_supported Tests
# =============================================================================


class TestIsFormatSupported:
    """Tests for is_format_supported."""

    def test_fp8_on_hopper(self):
        """FP8 should be supported on Hopper."""
        assert QuantizationCompatibilityMatrix.is_format_supported(
            QuantizationFormat.FP8_E4M3,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.HOPPER,
        )

    def test_nvfp4_not_on_ampere(self):
        """NVFP4 should NOT be supported on Ampere."""
        assert not QuantizationCompatibilityMatrix.is_format_supported(
            QuantizationFormat.NVFP4,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.AMPERE,
        )

    def test_fp8_on_cdna3(self):
        """FP8 should be supported on CDNA3."""
        assert QuantizationCompatibilityMatrix.is_format_supported(
            QuantizationFormat.FP8_E4M3,
            HardwareBackend.AMD,
            AMDArchitecture.CDNA3,
        )

    def test_fp8_not_on_cdna2(self):
        """FP8 should NOT be supported on CDNA2."""
        assert not QuantizationCompatibilityMatrix.is_format_supported(
            QuantizationFormat.FP8_E4M3,
            HardwareBackend.AMD,
            AMDArchitecture.CDNA2,
        )

    def test_bf16_on_trn1(self):
        """BF16 should be supported on TRN1."""
        assert QuantizationCompatibilityMatrix.is_format_supported(
            QuantizationFormat.BF16,
            HardwareBackend.TRAINIUM,
            TrainiumArchitecture.TRN1,
        )

    def test_int8_on_cpu(self):
        """INT8 should be supported on CPU."""
        assert QuantizationCompatibilityMatrix.is_format_supported(
            QuantizationFormat.INT8_DYNAMIC,
            HardwareBackend.CPU,
        )

    def test_nvfp4_not_on_cpu(self):
        """NVFP4 should NOT be supported on CPU."""
        assert not QuantizationCompatibilityMatrix.is_format_supported(
            QuantizationFormat.NVFP4,
            HardwareBackend.CPU,
        )
