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
            (AMDArchitecture.RDNA2, AttentionKernelType.PYTORCH_SDPA),
            (AMDArchitecture.RDNA1, AttentionKernelType.PYTORCH_SDPA),
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


class TestRDNA1Constraints:
    """RDNA1 (gfx1010/1011/1012) must not recommend kernels that require rocBLAS."""

    def test_rdna1_only_sdpa(self):
        kernels = AttentionDispatchMatrix.get_supported_kernels(
            HardwareBackend.AMD, AMDArchitecture.RDNA1
        )
        assert kernels == [AttentionKernelType.PYTORCH_SDPA]

    def test_rdna1_no_flash_attn(self):
        kernels = AttentionDispatchMatrix.get_supported_kernels(
            HardwareBackend.AMD, AMDArchitecture.RDNA1
        )
        assert AttentionKernelType.FLASH_ATTENTION_CK not in kernels
        assert AttentionKernelType.FLASH_ATTENTION_2 not in kernels


class TestDefaultArchitectureFallback:
    """get_supported_kernels with architecture=None should use sensible defaults."""

    def test_nvidia_default(self):
        kernels = AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.CUDA)
        assert len(kernels) >= 2  # Ampere default has FA-2 + SDPA

    def test_amd_default(self):
        kernels = AttentionDispatchMatrix.get_supported_kernels(HardwareBackend.AMD)
        assert AttentionKernelType.PYTORCH_SDPA in kernels
