"""
Tests for Speculative Decoding Compatibility Matrix

Parametrized tests for every (backend, architecture) -> optimal method mapping,
fallback chain behavior, and method support queries.
"""

import pytest

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.inference.speculative.compatibility import (
    SpeculationCompatibilityMatrix,
)
from torchbridge.inference.speculative.methods import SpeculativeMethod

# =============================================================================
# Optimal Method Tests
# =============================================================================


class TestOptimalMethod:
    """Tests for get_optimal_method."""

    @pytest.mark.parametrize(
        "backend, arch, expected",
        [
            (HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC, SpeculativeMethod.EAGLE),
            (HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_CONSUMER, SpeculativeMethod.EAGLE),
            (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER, SpeculativeMethod.EAGLE),
            (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.CUDA, NVIDIAArchitecture.ADA, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.CUDA, NVIDIAArchitecture.TURING, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.CUDA, NVIDIAArchitecture.VOLTA, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.CUDA, NVIDIAArchitecture.PASCAL, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.AMD, AMDArchitecture.CDNA4, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.AMD, AMDArchitecture.CDNA2, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.AMD, AMDArchitecture.CDNA, SpeculativeMethod.PROMPT_LOOKUP),
            (HardwareBackend.AMD, AMDArchitecture.RDNA3, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.AMD, AMDArchitecture.RDNA2, SpeculativeMethod.PROMPT_LOOKUP),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN3, SpeculativeMethod.LAYER_SKIP),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2, SpeculativeMethod.LAYER_SKIP),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN1, SpeculativeMethod.PROMPT_LOOKUP),
            (HardwareBackend.TRAINIUM, TrainiumArchitecture.INF2, SpeculativeMethod.PROMPT_LOOKUP),
            (HardwareBackend.TPU, TPUVersion.V7, SpeculativeMethod.LAYER_SKIP),
            (HardwareBackend.TPU, TPUVersion.V6E, SpeculativeMethod.LAYER_SKIP),
            (HardwareBackend.TPU, TPUVersion.V5P, SpeculativeMethod.LAYER_SKIP),
            (HardwareBackend.TPU, TPUVersion.V5E, SpeculativeMethod.PROMPT_LOOKUP),
            (HardwareBackend.TPU, TPUVersion.V4, SpeculativeMethod.PROMPT_LOOKUP),
            (HardwareBackend.CPU, None, SpeculativeMethod.PROMPT_LOOKUP),
        ],
    )
    def test_optimal_method(self, backend, arch, expected):
        """Optimal method should match the compatibility matrix."""
        result = SpeculationCompatibilityMatrix.get_optimal_method(backend, arch)
        assert result == expected, (
            f"Expected {expected.value} for {backend.value}/{arch}, got {result.value}"
        )


# =============================================================================
# Supported Methods Tests
# =============================================================================


class TestSupportedMethods:
    """Tests for get_supported_methods."""

    def test_nvidia_blackwell_all_methods(self):
        """Blackwell DC supports all 5 methods."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC
        )
        assert len(methods) == 5
        assert SpeculativeMethod.EAGLE in methods
        assert SpeculativeMethod.MEDUSA in methods

    def test_nvidia_ampere_no_eagle(self):
        """Ampere does not support EAGLE."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        assert SpeculativeMethod.EAGLE not in methods
        assert SpeculativeMethod.DRAFT_MODEL in methods

    def test_cpu_only_prompt_lookup(self):
        """CPU only supports PROMPT_LOOKUP."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CPU
        )
        assert methods == [SpeculativeMethod.PROMPT_LOOKUP]

    def test_trainium_no_draft_model(self):
        """Trainium TRN2 does not support DRAFT_MODEL."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2
        )
        assert SpeculativeMethod.DRAFT_MODEL not in methods
        assert SpeculativeMethod.LAYER_SKIP in methods

    def test_amd_cdna3_methods(self):
        """CDNA3 supports draft_model, layer_skip, prompt_lookup."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.AMD, AMDArchitecture.CDNA3
        )
        assert SpeculativeMethod.DRAFT_MODEL in methods
        assert SpeculativeMethod.LAYER_SKIP in methods
        assert SpeculativeMethod.PROMPT_LOOKUP in methods
        assert SpeculativeMethod.EAGLE not in methods

    def test_supported_methods_returns_copy(self):
        """get_supported_methods returns a copy, not the original list."""
        methods1 = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CPU
        )
        methods2 = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CPU
        )
        assert methods1 is not methods2


# =============================================================================
# is_method_supported Tests
# =============================================================================


class TestIsMethodSupported:
    """Tests for is_method_supported."""

    def test_eagle_supported_on_hopper(self):
        assert SpeculationCompatibilityMatrix.is_method_supported(
            SpeculativeMethod.EAGLE, HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
        )

    def test_eagle_not_supported_on_ampere(self):
        assert not SpeculationCompatibilityMatrix.is_method_supported(
            SpeculativeMethod.EAGLE, HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )

    def test_prompt_lookup_supported_everywhere(self):
        """PROMPT_LOOKUP is universally supported."""
        for backend in HardwareBackend:
            if backend == HardwareBackend.CUSTOM:
                continue
            assert SpeculationCompatibilityMatrix.is_method_supported(
                SpeculativeMethod.PROMPT_LOOKUP, backend
            ), f"PROMPT_LOOKUP should be supported on {backend.value}"

    def test_medusa_not_on_cpu(self):
        assert not SpeculationCompatibilityMatrix.is_method_supported(
            SpeculativeMethod.MEDUSA, HardwareBackend.CPU
        )


# =============================================================================
# Fallback Chain Tests
# =============================================================================


class TestFallbackChain:
    """Tests for get_fallback_chain."""

    def test_supported_method_returns_singleton(self):
        """If requested method is supported, returns [requested]."""
        chain = SpeculationCompatibilityMatrix.get_fallback_chain(
            SpeculativeMethod.DRAFT_MODEL,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.AMPERE,
        )
        assert chain == [SpeculativeMethod.DRAFT_MODEL]

    def test_unsupported_method_returns_full_chain(self):
        """If requested method is not supported, returns full supported list."""
        chain = SpeculationCompatibilityMatrix.get_fallback_chain(
            SpeculativeMethod.EAGLE,
            HardwareBackend.CPU,
        )
        assert SpeculativeMethod.PROMPT_LOOKUP in chain
        assert SpeculativeMethod.EAGLE not in chain

    def test_fallback_chain_cpu(self):
        """CPU fallback chain for EAGLE should be [PROMPT_LOOKUP]."""
        chain = SpeculationCompatibilityMatrix.get_fallback_chain(
            SpeculativeMethod.EAGLE,
            HardwareBackend.CPU,
        )
        assert chain == [SpeculativeMethod.PROMPT_LOOKUP]


# =============================================================================
# Architecture Resolution Tests
# =============================================================================


class TestArchitectureResolution:
    """Tests for AUTO/None architecture resolution."""

    def test_none_architecture_uses_default(self):
        """None architecture resolves to default (Ampere for CUDA)."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUDA, None
        )
        expected = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        assert methods == expected

    def test_auto_architecture_uses_default(self):
        """AUTO architecture resolves to default."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.AUTO
        )
        expected = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        assert methods == expected

    def test_auto_amd_resolves_to_cdna3(self):
        """AMD AUTO resolves to CDNA3."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.AMD, AMDArchitecture.AUTO
        )
        expected = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.AMD, AMDArchitecture.CDNA3
        )
        assert methods == expected

    def test_auto_trainium_resolves_to_trn2(self):
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.TRAINIUM, TrainiumArchitecture.AUTO
        )
        expected = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2
        )
        assert methods == expected

    def test_auto_tpu_resolves_to_v5e(self):
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.TPU, TPUVersion.AUTO
        )
        expected = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.TPU, TPUVersion.V5E
        )
        assert methods == expected

    def test_custom_backend_falls_back_to_cpu(self):
        """CUSTOM backend falls back to CPU methods."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUSTOM
        )
        assert methods == [SpeculativeMethod.PROMPT_LOOKUP]
