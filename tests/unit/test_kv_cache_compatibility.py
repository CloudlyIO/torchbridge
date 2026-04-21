"""
Tests for KV-Cache Dtype Compatibility Matrix

Parametrized tests for every (backend, architecture) -> optimal dtype mapping,
fallback chain behavior, and dtype support queries.
"""

import pytest

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.models.kv.cache_compatibility import (
    KVCacheCompatibilityMatrix,
)
from torchbridge.models.kv.cache_dtype import KVCacheDtype

# =============================================================================
# Optimal Dtype Tests
# =============================================================================


class TestOptimalDtype:
    """Tests for get_optimal_dtype."""

    @pytest.mark.parametrize(
        "backend, arch, expected",
        [
            (HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC, KVCacheDtype.NVFP4),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.BLACKWELL_CONSUMER,
                KVCacheDtype.FP8_E4M3,
            ),
            (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER, KVCacheDtype.FP8_E4M3),
            (HardwareBackend.CUDA, NVIDIAArchitecture.ADA, KVCacheDtype.FP8_E4M3),
            (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE, KVCacheDtype.BF16),
            (HardwareBackend.CUDA, NVIDIAArchitecture.TURING, KVCacheDtype.FP16),
            (HardwareBackend.CUDA, NVIDIAArchitecture.VOLTA, KVCacheDtype.FP16),
            (HardwareBackend.CUDA, NVIDIAArchitecture.PASCAL, KVCacheDtype.FP16),
            (HardwareBackend.AMD, AMDArchitecture.CDNA4, KVCacheDtype.FP8_E4M3),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3, KVCacheDtype.FP8_E4M3),
            (HardwareBackend.AMD, AMDArchitecture.CDNA2, KVCacheDtype.BF16),
            (HardwareBackend.AMD, AMDArchitecture.CDNA, KVCacheDtype.FP16),
            (HardwareBackend.AMD, AMDArchitecture.RDNA3, KVCacheDtype.BF16),
            (HardwareBackend.AMD, AMDArchitecture.RDNA2, KVCacheDtype.FP16),
            (HardwareBackend.AMD, AMDArchitecture.RDNA1, KVCacheDtype.FP16),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN3, KVCacheDtype.BF16),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2, KVCacheDtype.BF16),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN1, KVCacheDtype.BF16),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.INF2, KVCacheDtype.BF16),
            (HardwareBackend.TPU, TPUVersion.V7, KVCacheDtype.BF16),
            (HardwareBackend.TPU, TPUVersion.V5E, KVCacheDtype.BF16),
            (HardwareBackend.TPU, TPUVersion.V4, KVCacheDtype.BF16),
            (HardwareBackend.CPU, None, KVCacheDtype.PASSTHROUGH),
        ],
    )
    def test_optimal_dtype(self, backend, arch, expected):
        """Optimal dtype should match the compatibility matrix."""
        result = KVCacheCompatibilityMatrix.get_optimal_dtype(backend, arch)
        assert result == expected, (
            f"Expected {expected.value} for {backend.value}/{arch}, got {result.value}"
        )


# =============================================================================
# Supported Dtypes Tests
# =============================================================================


class TestSupportedDtypes:
    """Tests for get_supported_dtypes."""

    def test_all_backends_have_dtypes(self):
        """Every backend should have at least one supported dtype."""
        backends = [
            (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2),
            (HardwareBackend.TPU, TPUVersion.V5E),
            (HardwareBackend.CPU, None),
        ]
        for backend, arch in backends:
            dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(backend, arch)
            assert len(dtypes) > 0, f"No dtypes for {backend.value}"

    def test_optimal_is_first(self):
        """The first dtype in supported list should be the optimal one."""
        for backend, arch in [
            (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3),
        ]:
            supported = KVCacheCompatibilityMatrix.get_supported_dtypes(backend, arch)
            optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(backend, arch)
            assert supported[0] == optimal

    def test_blackwell_dc_has_nvfp4(self):
        """Blackwell DC should support NVFP4."""
        dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC
        )
        assert KVCacheDtype.NVFP4 in dtypes

    def test_cpu_has_passthrough(self):
        """CPU should support PASSTHROUGH."""
        dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(HardwareBackend.CPU)
        assert KVCacheDtype.PASSTHROUGH in dtypes

    def test_null_architecture_uses_default(self):
        """None architecture should fall back to a default."""
        dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(
            HardwareBackend.CUDA, None
        )
        assert len(dtypes) > 0

    def test_auto_architecture_uses_default(self):
        """AUTO architecture should fall back to a default."""
        dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(
            HardwareBackend.CUDA, NVIDIAArchitecture.AUTO
        )
        assert len(dtypes) > 0


# =============================================================================
# Fallback Chain Tests
# =============================================================================


class TestFallbackChain:
    """Tests for get_fallback_chain."""

    def test_supported_dtype_returns_itself(self):
        """Requesting a supported dtype should return just that dtype."""
        chain = KVCacheCompatibilityMatrix.get_fallback_chain(
            KVCacheDtype.FP8_E4M3,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.HOPPER,
        )
        assert chain == [KVCacheDtype.FP8_E4M3]

    def test_nvfp4_on_amd_falls_back(self):
        """NVFP4 on AMD should fall back to AMD's supported dtypes."""
        chain = KVCacheCompatibilityMatrix.get_fallback_chain(
            KVCacheDtype.NVFP4,
            HardwareBackend.AMD,
            AMDArchitecture.CDNA3,
        )
        assert KVCacheDtype.NVFP4 not in chain
        assert len(chain) > 0
        assert chain[0] == KVCacheDtype.FP8_E4M3

    def test_fp8_on_cpu_falls_back(self):
        """FP8 on CPU should fall back to PASSTHROUGH."""
        chain = KVCacheCompatibilityMatrix.get_fallback_chain(
            KVCacheDtype.FP8_E4M3,
            HardwareBackend.CPU,
        )
        assert KVCacheDtype.FP8_E4M3 not in chain
        assert KVCacheDtype.PASSTHROUGH in chain

    def test_nvfp4_on_ampere_falls_back(self):
        """NVFP4 on Ampere should fall back."""
        chain = KVCacheCompatibilityMatrix.get_fallback_chain(
            KVCacheDtype.NVFP4,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.AMPERE,
        )
        assert KVCacheDtype.NVFP4 not in chain
        assert chain[0] == KVCacheDtype.BF16

    def test_fallback_chain_non_empty(self):
        """Fallback chains should never be empty."""
        for backend in HardwareBackend:
            if backend == HardwareBackend.CUSTOM:
                continue
            chain = KVCacheCompatibilityMatrix.get_fallback_chain(
                KVCacheDtype.NVFP4, backend
            )
            assert len(chain) > 0, f"Empty fallback for {backend.value}"


# =============================================================================
# is_dtype_supported Tests
# =============================================================================


class TestIsDtypeSupported:
    """Tests for is_dtype_supported."""

    def test_fp8_on_hopper(self):
        """FP8 should be supported on Hopper."""
        assert KVCacheCompatibilityMatrix.is_dtype_supported(
            KVCacheDtype.FP8_E4M3,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.HOPPER,
        )

    def test_nvfp4_not_on_ampere(self):
        """NVFP4 should NOT be supported on Ampere."""
        assert not KVCacheCompatibilityMatrix.is_dtype_supported(
            KVCacheDtype.NVFP4,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.AMPERE,
        )

    def test_fp8_on_cdna3(self):
        """FP8 should be supported on CDNA3."""
        assert KVCacheCompatibilityMatrix.is_dtype_supported(
            KVCacheDtype.FP8_E4M3,
            HardwareBackend.AMD,
            AMDArchitecture.CDNA3,
        )

    def test_fp8_not_on_cdna2(self):
        """FP8 should NOT be supported on CDNA2."""
        assert not KVCacheCompatibilityMatrix.is_dtype_supported(
            KVCacheDtype.FP8_E4M3,
            HardwareBackend.AMD,
            AMDArchitecture.CDNA2,
        )

    def test_bf16_on_trn1(self):
        """BF16 should be supported on TRN1."""
        assert KVCacheCompatibilityMatrix.is_dtype_supported(
            KVCacheDtype.BF16,
            HardwareBackend.TRAINIUM,
            TrainiumArchitecture.TRN1,
        )

    def test_passthrough_on_cpu(self):
        """PASSTHROUGH should be supported on CPU."""
        assert KVCacheCompatibilityMatrix.is_dtype_supported(
            KVCacheDtype.PASSTHROUGH,
            HardwareBackend.CPU,
        )

    def test_nvfp4_not_on_cpu(self):
        """NVFP4 should NOT be supported on CPU."""
        assert not KVCacheCompatibilityMatrix.is_dtype_supported(
            KVCacheDtype.NVFP4,
            HardwareBackend.CPU,
        )


# =============================================================================
# Matrix Coverage Tests
# =============================================================================


class TestMatrixCoverage:
    """Tests for overall matrix coverage and consistency."""

    def test_all_nvidia_architectures_covered(self):
        """All non-AUTO NVIDIA architectures should be in the matrix."""
        for arch in NVIDIAArchitecture:
            if arch == NVIDIAArchitecture.AUTO:
                continue
            dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(
                HardwareBackend.CUDA, arch
            )
            assert len(dtypes) > 0, f"No dtypes for CUDA/{arch.value}"

    def test_all_amd_architectures_covered(self):
        """All non-AUTO AMD architectures should be in the matrix."""
        for arch in AMDArchitecture:
            if arch == AMDArchitecture.AUTO:
                continue
            dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(
                HardwareBackend.AMD, arch
            )
            assert len(dtypes) > 0, f"No dtypes for AMD/{arch.value}"

    def test_all_trainium_architectures_covered(self):
        """All non-AUTO Trainium architectures should be in the matrix."""
        for arch in TrainiumArchitecture:
            if arch == TrainiumArchitecture.AUTO:
                continue
            dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(
                HardwareBackend.TRAINIUM, arch
            )
            assert len(dtypes) > 0, f"No dtypes for TRAINIUM/{arch.value}"

    def test_all_tpu_versions_covered(self):
        """All non-AUTO TPU versions should be in the matrix."""
        for ver in TPUVersion:
            if ver == TPUVersion.AUTO:
                continue
            dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(
                HardwareBackend.TPU, ver
            )
            assert len(dtypes) > 0, f"No dtypes for TPU/{ver.value}"

    def test_custom_backend_falls_back_to_cpu(self):
        """CUSTOM backend should fall back to CPU dtypes."""
        dtypes = KVCacheCompatibilityMatrix.get_supported_dtypes(HardwareBackend.CUSTOM)
        assert KVCacheDtype.PASSTHROUGH in dtypes
