"""
Unit tests for CompileCompatibility — the (backend, architecture) → torch.compile mode matrix.
"""

from torchbridge.backends.compile_compatibility import (
    _COMPILE_MODE_MATRIX,
    _DEFAULT_COMPILE_MODE,
    CompileCompatibility,
)
from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
)


class TestCompileModeMatrix:
    """Tests for _COMPILE_MODE_MATRIX structure."""

    def test_matrix_is_nonempty(self):
        assert len(_COMPILE_MODE_MATRIX) > 0

    def test_all_values_are_valid_compile_modes(self):
        valid_modes = {"max-autotune", "reduce-overhead", "default"}
        for key, mode in _COMPILE_MODE_MATRIX.items():
            assert mode in valid_modes, f"Invalid compile mode {mode!r} for key {key}"

    def test_default_mode_is_reduce_overhead(self):
        assert _DEFAULT_COMPILE_MODE == "reduce-overhead"


class TestGetCompileMode:
    """Tests for CompileCompatibility.get_compile_mode()."""

    # --- NVIDIA high-end → max-autotune ---

    def test_hopper_returns_max_autotune(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
        )
        assert mode == "max-autotune"

    def test_blackwell_dc_returns_max_autotune(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC
        )
        assert mode == "max-autotune"

    def test_blackwell_consumer_returns_max_autotune(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_CONSUMER
        )
        assert mode == "max-autotune"

    # --- NVIDIA older → default (reduce-overhead) ---

    def test_ampere_returns_reduce_overhead(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
        )
        assert mode == _DEFAULT_COMPILE_MODE

    def test_ada_returns_reduce_overhead(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.CUDA, NVIDIAArchitecture.ADA
        )
        assert mode == _DEFAULT_COMPILE_MODE

    def test_turing_returns_reduce_overhead(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.CUDA, NVIDIAArchitecture.TURING
        )
        assert mode == _DEFAULT_COMPILE_MODE

    # --- AMD high-end → max-autotune ---

    def test_cdna3_returns_max_autotune(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.AMD, AMDArchitecture.CDNA3
        )
        assert mode == "max-autotune"

    def test_cdna4_returns_max_autotune(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.AMD, AMDArchitecture.CDNA4
        )
        assert mode == "max-autotune"

    # --- AMD older → default ---

    def test_cdna2_returns_reduce_overhead(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.AMD, AMDArchitecture.CDNA2
        )
        assert mode == _DEFAULT_COMPILE_MODE

    def test_cdna1_returns_reduce_overhead(self):
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.AMD, AMDArchitecture.CDNA
        )
        assert mode == _DEFAULT_COMPILE_MODE

    # --- Non-CUDA backends → default ---

    def test_cpu_backend_returns_reduce_overhead(self):
        mode = CompileCompatibility.get_compile_mode(HardwareBackend.CPU, None)
        assert mode == _DEFAULT_COMPILE_MODE

    def test_tpu_backend_returns_reduce_overhead(self):
        mode = CompileCompatibility.get_compile_mode(HardwareBackend.TPU, None)
        assert mode == _DEFAULT_COMPILE_MODE

    def test_trainium_backend_returns_reduce_overhead(self):
        mode = CompileCompatibility.get_compile_mode(HardwareBackend.TRAINIUM, None)
        assert mode == _DEFAULT_COMPILE_MODE

    def test_unknown_arch_returns_default(self):
        """An arch not in the matrix returns the default mode, not an error."""
        mode = CompileCompatibility.get_compile_mode(
            HardwareBackend.CUDA, NVIDIAArchitecture.AUTO
        )
        assert mode == _DEFAULT_COMPILE_MODE

    def test_none_arch_returns_default(self):
        mode = CompileCompatibility.get_compile_mode(HardwareBackend.CUDA, None)
        assert mode == _DEFAULT_COMPILE_MODE


class TestAllEntries:
    """Tests for CompileCompatibility.all_entries()."""

    def test_returns_dict(self):
        assert isinstance(CompileCompatibility.all_entries(), dict)

    def test_returns_copy_not_original(self):
        """Mutations to the returned dict should not affect the internal matrix."""
        entries = CompileCompatibility.all_entries()
        entries.clear()
        assert len(CompileCompatibility.all_entries()) > 0

    def test_contains_hopper(self):
        entries = CompileCompatibility.all_entries()
        assert (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER) in entries

    def test_contains_cdna3(self):
        entries = CompileCompatibility.all_entries()
        assert (HardwareBackend.AMD, AMDArchitecture.CDNA3) in entries
