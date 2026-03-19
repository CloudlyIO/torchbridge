"""Tests for AttentionDispatcher."""

from unittest.mock import patch

from torchbridge.attention.dispatch.dispatcher import (
    AttentionDispatcher,
    AttentionDispatchResult,
)
from torchbridge.attention.dispatch.kernel_types import AttentionKernelType
from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)


class TestSelectKernelCPU:
    """Dispatcher on CPU should always return PYTORCH_SDPA."""

    def test_cpu_returns_sdpa(self):
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        result = dispatcher.select_kernel()
        assert result.kernel_type == AttentionKernelType.PYTORCH_SDPA
        assert isinstance(result, AttentionDispatchResult)

    def test_cpu_no_fallback(self):
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        result = dispatcher.select_kernel()
        assert result.used_fallback is False
        assert result.fallback_chain == []

    def test_cpu_implementation_name(self):
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        result = dispatcher.select_kernel()
        assert result.implementation_name == "pytorch_sdpa"


class TestSelectKernelMocked:
    """Test kernel selection with mocked backends."""

    def test_nvidia_hopper_no_flash_falls_to_sdpa(self):
        """On Hopper without FlexAttention or flash_attn, falls to SDPA."""
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            use_benchmark_cache=False,
        )
        with patch.object(dispatcher, "_check_kernel_availability", return_value=False):
            def selective_check(kt):
                if kt == AttentionKernelType.PYTORCH_SDPA:
                    return True
                return False

            dispatcher._check_kernel_availability = selective_check
            result = dispatcher.select_kernel()
            assert result.kernel_type == AttentionKernelType.PYTORCH_SDPA
            assert result.used_fallback is True
            assert len(result.warnings) > 0

    def test_nvidia_ampere_prefers_fa2(self):
        """On Ampere with flash_attn available, should select FA-2."""
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
            use_benchmark_cache=False,
        )

        def check(kt):
            return kt in (
                AttentionKernelType.FLASH_ATTENTION_2,
                AttentionKernelType.PYTORCH_SDPA,
            )

        dispatcher._check_kernel_availability = check
        result = dispatcher.select_kernel()
        assert result.kernel_type == AttentionKernelType.FLASH_ATTENTION_2

    def test_amd_cdna3_prefers_ck(self):
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.AMD,
            architecture=AMDArchitecture.CDNA3,
            use_benchmark_cache=False,
        )

        def check(kt):
            return kt in (
                AttentionKernelType.FLASH_ATTENTION_CK,
                AttentionKernelType.PYTORCH_SDPA,
            )

        dispatcher._check_kernel_availability = check
        result = dispatcher.select_kernel()
        assert result.kernel_type == AttentionKernelType.FLASH_ATTENTION_CK

    def test_trainium_trn2_prefers_neuronx(self):
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.TRAINIUM,
            architecture=TrainiumArchitecture.TRN2,
            use_benchmark_cache=False,
        )

        def check(kt):
            return kt in (
                AttentionKernelType.NEURONX_SDPA,
                AttentionKernelType.PYTORCH_SDPA,
            )

        dispatcher._check_kernel_availability = check
        result = dispatcher.select_kernel()
        assert result.kernel_type == AttentionKernelType.NEURONX_SDPA

    def test_tpu_v5e_prefers_pallas(self):
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.TPU,
            architecture=TPUVersion.V5E,
            use_benchmark_cache=False,
        )

        def check(kt):
            return kt in (
                AttentionKernelType.PALLAS_ATTENTION,
                AttentionKernelType.PYTORCH_SDPA,
            )

        dispatcher._check_kernel_availability = check
        result = dispatcher.select_kernel()
        assert result.kernel_type == AttentionKernelType.PALLAS_ATTENTION


class TestRuntimeAvailabilityChecks:
    """Test _check_kernel_availability."""

    def test_sdpa_always_available(self):
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        assert dispatcher._check_kernel_availability(AttentionKernelType.PYTORCH_SDPA) is True

    def test_flex_attention_check_runs(self):
        dispatcher = AttentionDispatcher(
            backend=HardwareBackend.CPU, use_benchmark_cache=False
        )
        # Should return bool without raising
        result = dispatcher._check_kernel_availability(AttentionKernelType.FLEX_ATTENTION)
        assert isinstance(result, bool)


class TestDispatcherInternalState:
    """Test that backend and architecture are stored correctly on the dispatcher."""

    def test_cpu_backend_stored(self):
        d = AttentionDispatcher(backend=HardwareBackend.CPU, use_benchmark_cache=False)
        assert d._backend == HardwareBackend.CPU
        assert d._backend.value == "cpu"

    def test_nvidia_architecture_stored(self):
        d = AttentionDispatcher(
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.HOPPER,
            use_benchmark_cache=False,
        )
        assert d._architecture == NVIDIAArchitecture.HOPPER
        assert d._architecture.value == "hopper"
