"""
Tests for AMD RDNA1 (gfx1010/1011/1012) architecture detection and matrix constraints.

RDNA1 (RX 5000 series) is NOT in the rocBLAS binary distribution. Any BLAS call
(matmul, gemm) core dumps on this hardware. All compatibility matrices must
restrict RDNA1 to rocBLAS-free paths only.
"""

import pytest

from torchbridge.core.config import AMDArchitecture, AMDConfig


_RDNA1_DEVICE_NAMES = [
    "AMD Radeon RX 5700 XT",
    "AMD Radeon RX 5600 XT",
    "AMD Radeon RX 5500 XT",
    "AMD Radeon Pro V520",
    "Radeon RX 5700",
]


class TestRDNA1Detection:
    """RDNA1 device names must be detected as AMDArchitecture.RDNA1."""

    @pytest.mark.parametrize("device_name", _RDNA1_DEVICE_NAMES)
    def test_rdna1_backend_detection(self, device_name):
        from torchbridge.backends.amd.amd_backend import AMDBackend

        backend = AMDBackend(AMDConfig(architecture=AMDArchitecture.RDNA1))
        arch = backend._detect_architecture(device_name)
        assert arch == AMDArchitecture.RDNA1, (
            f"Expected RDNA1 for '{device_name}', got {arch}"
        )

    def test_rdna1_not_misclassified_as_cdna2(self):
        """gfx1011 must not silently fall through to CDNA2 default."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        backend = AMDBackend(AMDConfig(architecture=AMDArchitecture.AUTO))
        arch = backend._detect_architecture("AMD Radeon RX 5500 XT")
        assert arch != AMDArchitecture.CDNA2

    def test_rdna2_not_confused_with_rdna1(self):
        """RX 6000 series (RDNA2) must not be detected as RDNA1."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        backend = AMDBackend(AMDConfig(architecture=AMDArchitecture.AUTO))
        arch = backend._detect_architecture("AMD Radeon RX 6700 XT")
        assert arch == AMDArchitecture.RDNA2

    def test_rdna3_not_confused_with_rdna1(self):
        """RX 7000 series (RDNA3) must not be detected as RDNA1."""
        from torchbridge.backends.amd.amd_backend import AMDBackend

        backend = AMDBackend(AMDConfig(architecture=AMDArchitecture.AUTO))
        arch = backend._detect_architecture("AMD Radeon RX 7900 XTX")
        assert arch == AMDArchitecture.RDNA3


class TestRDNA1EnumExists:
    """AMDArchitecture enum must have an RDNA1 entry."""

    def test_rdna1_in_enum(self):
        assert hasattr(AMDArchitecture, "RDNA1")
        assert AMDArchitecture.RDNA1.value == "rdna1"

    def test_rdna1_distinct_from_rdna2(self):
        assert AMDArchitecture.RDNA1 != AMDArchitecture.RDNA2

    def test_rdna1_distinct_from_cdna2(self):
        assert AMDArchitecture.RDNA1 != AMDArchitecture.CDNA2
