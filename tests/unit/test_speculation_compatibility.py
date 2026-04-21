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
from torchbridge.inference.speculative.methods import (
    SPECULATIVE_METHOD_SPECS,
    SpeculativeMethod,
)

# =============================================================================
# Optimal Method Tests
# =============================================================================


class TestOptimalMethod:
    """Tests for get_optimal_method."""

    @pytest.mark.parametrize(
        "backend, arch, expected",
        [
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.BLACKWELL_DC,
                SpeculativeMethod.DRAFT_MODEL,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.BLACKWELL_CONSUMER,
                SpeculativeMethod.DRAFT_MODEL,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.HOPPER,
                SpeculativeMethod.DRAFT_MODEL,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.AMPERE,
                SpeculativeMethod.DRAFT_MODEL,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.ADA,
                SpeculativeMethod.DRAFT_MODEL,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.TURING,
                SpeculativeMethod.DRAFT_MODEL,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.VOLTA,
                SpeculativeMethod.DRAFT_MODEL,
            ),
            (
                HardwareBackend.CUDA,
                NVIDIAArchitecture.PASCAL,
                SpeculativeMethod.DRAFT_MODEL,
            ),
            (HardwareBackend.AMD, AMDArchitecture.CDNA4, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3, SpeculativeMethod.DRAFT_MODEL),
            (HardwareBackend.AMD, AMDArchitecture.CDNA2, SpeculativeMethod.DRAFT_MODEL),
            (
                HardwareBackend.AMD,
                AMDArchitecture.CDNA,
                SpeculativeMethod.PROMPT_LOOKUP,
            ),
            (HardwareBackend.AMD, AMDArchitecture.RDNA3, SpeculativeMethod.DRAFT_MODEL),
            (
                HardwareBackend.AMD,
                AMDArchitecture.RDNA2,
                SpeculativeMethod.PROMPT_LOOKUP,
            ),
            (
                HardwareBackend.AMD,
                AMDArchitecture.RDNA1,
                SpeculativeMethod.PROMPT_LOOKUP,
            ),
            (
                HardwareBackend.TRAINIUM,
                TrainiumArchitecture.TRN3,
                SpeculativeMethod.PROMPT_LOOKUP,
            ),
            (
                HardwareBackend.TRAINIUM,
                TrainiumArchitecture.TRN2,
                SpeculativeMethod.PROMPT_LOOKUP,
            ),
            (
                HardwareBackend.TRAINIUM,
                TrainiumArchitecture.TRN1,
                SpeculativeMethod.PROMPT_LOOKUP,
            ),
            (
                HardwareBackend.TRAINIUM,
                TrainiumArchitecture.INF2,
                SpeculativeMethod.PROMPT_LOOKUP,
            ),
            (HardwareBackend.TPU, TPUVersion.V7, SpeculativeMethod.PROMPT_LOOKUP),
            (HardwareBackend.TPU, TPUVersion.V6E, SpeculativeMethod.PROMPT_LOOKUP),
            (HardwareBackend.TPU, TPUVersion.V5P, SpeculativeMethod.PROMPT_LOOKUP),
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

    def test_nvidia_blackwell_methods(self):
        """Blackwell DC supports DRAFT_MODEL and PROMPT_LOOKUP."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC
        )
        assert len(methods) == 2
        assert SpeculativeMethod.DRAFT_MODEL in methods
        assert SpeculativeMethod.PROMPT_LOOKUP in methods
        # EAGLE/MEDUSA/LAYER_SKIP excluded — not implemented
        assert SpeculativeMethod.EAGLE not in methods

    def test_nvidia_ampere_methods(self):
        """Ampere supports DRAFT_MODEL and PROMPT_LOOKUP."""
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

    def test_trainium_prompt_lookup_only(self):
        """Trainium TRN2 supports only PROMPT_LOOKUP."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2
        )
        assert methods == [SpeculativeMethod.PROMPT_LOOKUP]

    def test_amd_cdna3_methods(self):
        """CDNA3 supports draft_model and prompt_lookup."""
        methods = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.AMD, AMDArchitecture.CDNA3
        )
        assert SpeculativeMethod.DRAFT_MODEL in methods
        assert SpeculativeMethod.PROMPT_LOOKUP in methods
        assert SpeculativeMethod.EAGLE not in methods
        assert SpeculativeMethod.LAYER_SKIP not in methods

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

    def test_eagle_not_supported_anywhere(self):
        """EAGLE is not in the matrix — not implemented."""
        assert not SpeculationCompatibilityMatrix.is_method_supported(
            SpeculativeMethod.EAGLE, HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
        )
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


# =============================================================================
# get_generate_compatible_methods Tests
# =============================================================================


class TestGetGenerateCompatibleMethods:
    """Tests for get_generate_compatible_methods."""

    def test_all_returned_methods_have_flag_set(self):
        """Every method in the result must have is_generate_compatible=True."""
        gc_methods = SpeculationCompatibilityMatrix.get_generate_compatible_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        assert len(gc_methods) > 0
        for m in gc_methods:
            assert SPECULATIVE_METHOD_SPECS[m].is_generate_compatible is True, (
                f"{m} should have is_generate_compatible=True"
            )

    def test_is_subset_of_supported(self):
        """generate-compatible methods must be a subset of supported methods."""
        supported = SpeculationCompatibilityMatrix.get_supported_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        gc_methods = SpeculationCompatibilityMatrix.get_generate_compatible_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        for m in gc_methods:
            assert m in supported

    def test_cpu_returns_prompt_lookup(self):
        """CPU generate-compatible methods should be [PROMPT_LOOKUP]."""
        gc_methods = SpeculationCompatibilityMatrix.get_generate_compatible_methods(
            HardwareBackend.CPU
        )
        assert gc_methods == [SpeculativeMethod.PROMPT_LOOKUP]

    def test_cuda_includes_draft_model(self):
        """CUDA generate-compatible list includes DRAFT_MODEL."""
        gc_methods = SpeculationCompatibilityMatrix.get_generate_compatible_methods(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        assert SpeculativeMethod.DRAFT_MODEL in gc_methods

    def test_eagle_medusa_layer_skip_never_returned(self):
        """EAGLE, MEDUSA, LAYER_SKIP must never appear in generate-compatible list."""
        excluded = {
            SpeculativeMethod.EAGLE,
            SpeculativeMethod.MEDUSA,
            SpeculativeMethod.LAYER_SKIP,
        }
        test_cases = [
            (HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
            (HardwareBackend.AMD, AMDArchitecture.CDNA3),
            (HardwareBackend.CPU, None),
        ]
        for backend, arch in test_cases:
            gc_methods = SpeculationCompatibilityMatrix.get_generate_compatible_methods(
                backend, arch
            )
            for m in excluded:
                assert m not in gc_methods, (
                    f"{m} must not appear in generate_compatible list for {backend}/{arch}"
                )
