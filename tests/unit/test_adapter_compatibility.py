"""Tests for adapter compatibility matrix."""

from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
from torchbridge.adapters.config import AdapterMethod
from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)
from torchbridge.precision.quantization.formats import QuantizationFormat


class TestGetOptimal:
    """Tests for AdapterCompatibilityMatrix.get_optimal()."""

    def test_nvidia_hopper_prefers_qlora(self):
        method = AdapterCompatibilityMatrix.get_optimal(
            HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
        )
        assert method == AdapterMethod.QLORA

    def test_nvidia_blackwell_dc_prefers_qdora(self):
        method = AdapterCompatibilityMatrix.get_optimal(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC
        )
        assert method == AdapterMethod.QDORA

    def test_nvidia_turing_prefers_lora(self):
        method = AdapterCompatibilityMatrix.get_optimal(
            HardwareBackend.CUDA, NVIDIAArchitecture.TURING
        )
        assert method == AdapterMethod.LORA

    def test_amd_cdna3_prefers_qlora(self):
        method = AdapterCompatibilityMatrix.get_optimal(
            HardwareBackend.AMD, AMDArchitecture.CDNA3
        )
        assert method == AdapterMethod.QLORA

    def test_amd_cdna2_prefers_lora(self):
        method = AdapterCompatibilityMatrix.get_optimal(
            HardwareBackend.AMD, AMDArchitecture.CDNA2
        )
        assert method == AdapterMethod.LORA

    def test_trainium_always_lora(self):
        for arch in [TrainiumArchitecture.TRN3, TrainiumArchitecture.TRN2, None]:
            method = AdapterCompatibilityMatrix.get_optimal(
                HardwareBackend.TRAINIUM, arch
            )
            assert method == AdapterMethod.LORA

    def test_tpu_prefers_lora(self):
        method = AdapterCompatibilityMatrix.get_optimal(
            HardwareBackend.TPU, TPUVersion.V5E
        )
        assert method == AdapterMethod.LORA

    def test_cpu_prefers_lora(self):
        method = AdapterCompatibilityMatrix.get_optimal(HardwareBackend.CPU)
        assert method == AdapterMethod.LORA

    def test_nvidia_default_arch(self):
        method = AdapterCompatibilityMatrix.get_optimal(HardwareBackend.CUDA, None)
        assert method == AdapterMethod.QLORA


class TestGetFallbackChain:
    """Tests for fallback chain ordering."""

    def test_nvidia_hopper_chain_has_four_methods(self):
        chain = AdapterCompatibilityMatrix.get_fallback_chain(
            HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
        )
        assert len(chain) == 4
        assert chain[0] == AdapterMethod.QLORA
        assert AdapterMethod.LORA in chain

    def test_trainium_chain_has_only_lora(self):
        chain = AdapterCompatibilityMatrix.get_fallback_chain(HardwareBackend.TRAINIUM)
        assert chain == [AdapterMethod.LORA]

    def test_cpu_chain(self):
        chain = AdapterCompatibilityMatrix.get_fallback_chain(HardwareBackend.CPU)
        # CPU now supports QLORA (INT8) for testing
        assert AdapterMethod.LORA in chain
        assert AdapterMethod.DORA in chain
        assert AdapterMethod.QLORA in chain

    def test_tpu_chain_includes_dora(self):
        chain = AdapterCompatibilityMatrix.get_fallback_chain(
            HardwareBackend.TPU, TPUVersion.V7
        )
        assert AdapterMethod.DORA in chain

    def test_chains_are_copies(self):
        """Modifying a returned chain should not affect future calls."""
        chain1 = AdapterCompatibilityMatrix.get_fallback_chain(HardwareBackend.TRAINIUM)
        chain1.append(AdapterMethod.QLORA)
        chain2 = AdapterCompatibilityMatrix.get_fallback_chain(HardwareBackend.TRAINIUM)
        assert AdapterMethod.QLORA not in chain2

    def test_get_fallback_chain_unknown_string_backend_does_not_crash(self):
        """Regression v0.5.45: get_fallback_chain() must log warning and return [LORA], not crash."""
        chain = AdapterCompatibilityMatrix.get_fallback_chain("unknown_string_backend")
        assert len(chain) >= 1
        assert chain[0] == AdapterMethod.LORA

    def test_get_optimal_unknown_string_backend_does_not_crash(self):
        """Regression v0.5.45: get_optimal() must return LORA fallback, not crash."""
        result = AdapterCompatibilityMatrix.get_optimal("unknown_string_backend")
        assert result == AdapterMethod.LORA


class TestGetBaseQuantFormat:
    """Tests for base quantization format selection."""

    def test_nvidia_gets_int4(self):
        fmt = AdapterCompatibilityMatrix.get_base_quant_format(HardwareBackend.CUDA)
        assert fmt == QuantizationFormat.INT4_WEIGHT_ONLY

    def test_amd_gets_int4(self):
        fmt = AdapterCompatibilityMatrix.get_base_quant_format(HardwareBackend.AMD)
        assert fmt == QuantizationFormat.INT4_WEIGHT_ONLY

    def test_trainium_gets_none(self):
        fmt = AdapterCompatibilityMatrix.get_base_quant_format(HardwareBackend.TRAINIUM)
        assert fmt is None

    def test_tpu_gets_none(self):
        fmt = AdapterCompatibilityMatrix.get_base_quant_format(HardwareBackend.TPU)
        assert fmt is None

    def test_cpu_gets_int8(self):
        fmt = AdapterCompatibilityMatrix.get_base_quant_format(HardwareBackend.CPU)
        assert fmt == QuantizationFormat.INT8_DYNAMIC_ACTIVATIONS


class TestSupportsMethod:
    """Tests for supports_method()."""

    def test_lora_supported_everywhere(self):
        for backend in HardwareBackend:
            assert AdapterCompatibilityMatrix.supports_method(
                backend, None, AdapterMethod.LORA
            )

    def test_qlora_not_on_trainium(self):
        assert not AdapterCompatibilityMatrix.supports_method(
            HardwareBackend.TRAINIUM, None, AdapterMethod.QLORA
        )

    def test_qlora_on_cpu(self):
        # CPU now supports QLORA (INT8 base for testing)
        assert AdapterCompatibilityMatrix.supports_method(
            HardwareBackend.CPU, None, AdapterMethod.QLORA
        )

    def test_qdora_on_nvidia_blackwell(self):
        assert AdapterCompatibilityMatrix.supports_method(
            HardwareBackend.CUDA,
            NVIDIAArchitecture.BLACKWELL_DC,
            AdapterMethod.QDORA,
        )

    def test_dora_not_on_trainium(self):
        assert not AdapterCompatibilityMatrix.supports_method(
            HardwareBackend.TRAINIUM, None, AdapterMethod.DORA
        )
