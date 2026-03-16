"""
Compile Mode Compatibility Matrix

Maps (backend, architecture) → optimal torch.compile mode.

This is the 13th TorchBridge compatibility matrix.  The mode selection
logic was previously duplicated inline inside NVIDIABackend and AMDBackend
(``is_h100 or is_blackwell`` conditionals).  Centralising it here makes it
queryable, testable in isolation, and visible to ``tb-advisor``.

Modes:
  "max-autotune"   — aggressive kernel search; worth the compile overhead on
                     powerful hardware (Hopper, Blackwell, CDNA3/4).
  "reduce-overhead" — fast compile, good runtime; safe default for everything
                      else (Ampere, Ada, CDNA2, CPU, TPU, Trainium).
"""

from torchbridge.core.config import AMDArchitecture, HardwareBackend, NVIDIAArchitecture

# (backend, architecture) → optimal torch.compile mode string
_COMPILE_MODE_MATRIX: dict[tuple, str] = {
    (HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER):             "max-autotune",
    (HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC):       "max-autotune",
    (HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_CONSUMER): "max-autotune",
    (HardwareBackend.AMD,  AMDArchitecture.CDNA3):                 "max-autotune",
    (HardwareBackend.AMD,  AMDArchitecture.CDNA4):                 "max-autotune",
}

_DEFAULT_COMPILE_MODE = "reduce-overhead"


class CompileCompatibility:
    """Query interface for the compile mode compatibility matrix."""

    @staticmethod
    def get_compile_mode(backend: HardwareBackend, arch) -> str:
        """Return the optimal ``torch.compile`` mode for a (backend, arch) pair.

        Falls back to ``_DEFAULT_COMPILE_MODE`` for any combination not
        explicitly listed (older architectures, non-CUDA backends, None arch).

        Args:
            backend: The hardware backend (CUDA, AMD, TPU, etc.)
            arch: The architecture enum value, or None.

        Returns:
            A ``torch.compile`` mode string.
        """
        return _COMPILE_MODE_MATRIX.get((backend, arch), _DEFAULT_COMPILE_MODE)

    @staticmethod
    def all_entries() -> dict[tuple, str]:
        """Return a copy of the full matrix for inspection or advisory output."""
        return dict(_COMPILE_MODE_MATRIX)


__all__ = [
    "CompileCompatibility",
    "_COMPILE_MODE_MATRIX",
    "_DEFAULT_COMPILE_MODE",
]
