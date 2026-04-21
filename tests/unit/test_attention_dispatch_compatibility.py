"""Tests for AttentionDispatchMatrix compatibility tables."""

import pytest

from torchbridge.attention.dispatch.compatibility import AttentionDispatchMatrix
from torchbridge.attention.dispatch.kernel_types import AttentionKernelType
from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)


class TestOptimalKernel:
    """Verify optimal kernel selection for every (backend, arch) pair."""

    @pytest.mark.parametrize(
        "arch, expected",
        [
            (NVIDIAArchitecture.BLACKWELL_DC, AttentionKernelType.FLEX_ATTENTION),
            (NVIDIAArchitecture.BLACKWELL_CONSUMER, AttentionKernelType.FLEX_ATTENTION),
            (NVIDIAArchitecture.HOPPER, AttentionKernelType.FLEX_ATTENTION),
            (NVIDIAArchitecture.ADA, AttentionKernelType.FLASH_ATTENTION_2),
            (NVIDIAArchitecture.AMPERE, AttentionKernelType.FLASH_ATTENTION_2),
            (NVIDIAArchitecture.TURING, AttentionKernelType.PYTORCH_SDPA),
            (NVIDIAArchitecture.VOLTA, AttentionKernelType.PYTORCH_SDPA),
            (NVIDIAArchitecture.PASCAL, AttentionKernelType.PYTORCH_SDPA),
        ],
    )
    def test_nvidia_optimal(self, arch, expected):
        assert (
            AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.CUDA, arch)[0]
            == expected
        )

    @pytest.mark.parametrize(
        "arch, expected",
        [
            (AMDArchitecture.CDNA4, AttentionKernelType.FLASH_ATTENTION_CK),
            (AMDArchitecture.CDNA3, AttentionKernelType.FLASH_ATTENTION_CK),
            (AMDArchitecture.CDNA2, AttentionKernelType.PYTORCH_SDPA),
            (AMDArchitecture.RDNA3, AttentionKernelType.PYTORCH_SDPA),
            # RDNA2 removed: no kernels supported in standard PyTorch ROCm builds
        ],
    )
    def test_amd_optimal(self, arch, expected):
        assert (
            AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.AMD, arch)[0]
            == expected
        )

    @pytest.mark.parametrize(
        "arch, expected",
        [
            (TrainiumArchitecture.TRN3, AttentionKernelType.NEURONX_SDPA),
            (TrainiumArchitecture.TRN2, AttentionKernelType.NEURONX_SDPA),
            (TrainiumArchitecture.TRN1, AttentionKernelType.PYTORCH_SDPA),
            (TrainiumArchitecture.INF2, AttentionKernelType.PYTORCH_SDPA),
        ],
    )
    def test_trainium_optimal(self, arch, expected):
        assert (
            AttentionDispatchMatrix.get_supported_kernels(
                HardwareBackend.TRAINIUM, arch
            )[0]
            == expected
        )

    @pytest.mark.parametrize(
        "version, expected",
        [
            (TPUVersion.V7, AttentionKernelType.PALLAS_ATTENTION),
            (TPUVersion.V6E, AttentionKernelType.PALLAS_ATTENTION),
            (TPUVersion.V5P, AttentionKernelType.PALLAS_ATTENTION),
            (TPUVersion.V5E, AttentionKernelType.PALLAS_ATTENTION),
            (TPUVersion.V4, AttentionKernelType.PYTORCH_SDPA),
        ],
    )
    def test_tpu_optimal(self, version, expected):
        assert (
            AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.TPU, version)[
                0
            ]
            == expected
        )

    def test_cpu_optimal(self):
        assert (
            AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.CPU)[0]
            == AttentionKernelType.PYTORCH_SDPA
        )


class TestFallbackChain:
    """Verify fallback chain logic."""

    def test_hopper_fallback_from_flex(self):
        chain = AttentionDispatchMatrix.get_fallback_chain(
            AttentionKernelType.FLEX_ATTENTION,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.HOPPER,
        )
        assert chain == [
            AttentionKernelType.FLASH_ATTENTION_3,
            AttentionKernelType.FLASH_ATTENTION_2,
            AttentionKernelType.PYTORCH_SDPA,
        ]

    def test_unsupported_kernel_returns_full_chain(self):
        chain = AttentionDispatchMatrix.get_fallback_chain(
            AttentionKernelType.NEURONX_SDPA,
            HardwareBackend.CUDA,
            NVIDIAArchitecture.HOPPER,
        )
        # NEURONX_SDPA not in NVIDIA chain, so returns all supported kernels
        assert AttentionKernelType.FLEX_ATTENTION in chain

    def test_cpu_fallback_chain_empty(self):
        chain = AttentionDispatchMatrix.get_fallback_chain(
            AttentionKernelType.PYTORCH_SDPA, HardwareBackend.CPU
        )
        assert chain == []

    def test_amd_cdna3_fallback_from_ck(self):
        chain = AttentionDispatchMatrix.get_fallback_chain(
            AttentionKernelType.FLASH_ATTENTION_CK,
            HardwareBackend.AMD,
            AMDArchitecture.CDNA3,
        )
        assert chain == [
            AttentionKernelType.PYTORCH_SDPA,
        ]


class TestKernelSupport:
    """Verify kernel support via get_supported_kernels."""

    def test_flex_on_hopper(self):
        assert (
            AttentionKernelType.FLEX_ATTENTION
            in AttentionDispatchMatrix.get_supported_kernels(
                HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
            )
        )

    def test_flex_not_on_cpu(self):
        assert (
            AttentionKernelType.FLEX_ATTENTION
            not in AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.CPU)
        )

    def test_neuronx_on_trn2(self):
        assert (
            AttentionKernelType.NEURONX_SDPA
            in AttentionDispatchMatrix.get_supported_kernels(
                HardwareBackend.TRAINIUM, TrainiumArchitecture.TRN2
            )
        )

    def test_neuronx_not_on_nvidia(self):
        assert (
            AttentionKernelType.NEURONX_SDPA
            not in AttentionDispatchMatrix.get_supported_kernels(
                HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
            )
        )

    def test_pallas_on_tpu_v5e(self):
        assert (
            AttentionKernelType.PALLAS_ATTENTION
            in AttentionDispatchMatrix.get_supported_kernels(
                HardwareBackend.TPU, TPUVersion.V5E
            )
        )


class TestAllBackendsHaveKernels:
    """Every backend must have at least one supported kernel."""

    @pytest.mark.parametrize("backend", list(HardwareBackend))
    def test_at_least_one_kernel(self, backend):
        if backend == HardwareBackend.CUSTOM:
            pytest.skip("CUSTOM backend has no predefined kernels")
        kernels = AttentionDispatchMatrix.get_supported_kernels(backend)
        assert len(kernels) >= 1
        assert AttentionKernelType.PYTORCH_SDPA in kernels


class TestDefaultArchitectureFallback:
    """get_supported_kernels with architecture=None should use sensible defaults."""

    def test_nvidia_default(self):
        kernels = AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.CUDA)
        assert len(kernels) >= 2  # Ampere default has FA-2 + SDPA

    def test_amd_default(self):
        kernels = AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.AMD)
        assert AttentionKernelType.PYTORCH_SDPA in kernels


class TestRDNA2NoKernelFinding:
    """
    AMD RDNA2 (gfx1011) is unsupported in standard PyTorch ROCm builds.

    Finding Details:
    - GPU: AMD Radeon Pro V520 (gfx1011 architecture)
    - PyTorch: 2.5.1+rocm6.2
    - HIP: 6.2
    - PyTorch ROCm arch list: gfx900, gfx906, gfx908, gfx90a, gfx1030, gfx1100, gfx942
    - Issue: gfx1011 is NOT in the standard PyTorch ROCm build targets
    - All kernels fail with: HIP error: invalid device function

    This test class verifies that our compatibility matrix correctly reflects
    this limitation by marking RDNA2 with an empty kernel list.
    """

    def test_rdna2_has_no_kernels(self):
        """Verify RDNA2 has no supported kernels in the compatibility matrix."""
        kernels = AttentionDispatchMatrix.get_supported_kernels(
            HardwareBackend.AMD, AMDArchitecture.RDNA2
        )
        assert kernels == [], f"Expected empty list for RDNA2, got {kernels}"

    def test_rdna2_pytorch_sdpa_not_listed(self):
        """Verify PYTORCH_SDPA is NOT available for RDNA2."""
        kernels = AttentionDispatchMatrix.get_supported_kernels(
            HardwareBackend.AMD, AMDArchitecture.RDNA2
        )
        assert (
            AttentionKernelType.PYTORCH_SDPA not in kernels
        ), "PYTORCH_SDPA should not be supported on RDNA2 in standard PyTorch builds"

    def test_rdna2_fallback_chain_is_empty(self):
        """Verify fallback chain is empty for RDNA2."""
        chain = AttentionDispatchMatrix.get_fallback_chain(
            AttentionKernelType.PYTORCH_SDPA, HardwareBackend.AMD, AMDArchitecture.RDNA2
        )
        assert chain == [], f"Expected empty fallback chain for RDNA2, got {chain}"
