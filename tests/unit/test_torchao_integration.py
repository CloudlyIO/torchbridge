"""Tests for B3 fix: TorchAOBackend backend awareness.

Verifies that is_available_on_backend() correctly gates torchao usage
based on the hardware backend, preventing silent failures on AMD/ROCm.
"""

import inspect

import pytest
import torch

from torchbridge.precision.torchao_integration import (
    TORCHAO_AVAILABLE,
    TorchAOBackend,
)


class TestTorchAOBackendAwareness:
    """B3: is_available_on_backend() exists and behaves correctly per backend."""

    def test_is_available_on_backend_method_exists(self):
        """TorchAOBackend must have is_available_on_backend() staticmethod."""
        assert hasattr(TorchAOBackend, "is_available_on_backend")
        sig = inspect.signature(TorchAOBackend.is_available_on_backend)
        assert "backend" in sig.parameters

    def test_returns_false_when_torchao_not_available(self):
        """When torchao is not installed, always returns False."""
        if TORCHAO_AVAILABLE:
            pytest.skip("torchao is installed; skipping not-available test")
        assert TorchAOBackend.is_available_on_backend("cuda") is False
        assert TorchAOBackend.is_available_on_backend("rocm") is False
        assert TorchAOBackend.is_available_on_backend("cpu") is False

    def test_cuda_backend_matches_torchao_availability(self):
        """CUDA backend availability must match overall torchao availability."""
        result = TorchAOBackend.is_available_on_backend("cuda")
        # If torchao is not installed, result must be False
        if not TORCHAO_AVAILABLE:
            assert result is False
        # If torchao IS installed and we're on CUDA, result should be True
        # (we don't assert True here because CI may not have CUDA)
        assert isinstance(result, bool)

    def test_rocm_backend_requires_hip_pytorch(self):
        """ROCm support requires a PyTorch build with HIP."""
        result = TorchAOBackend.is_available_on_backend("rocm")
        hip_available = getattr(torch.version, "hip", None) is not None
        # If torchao available and HIP build: should be True
        # If torchao unavailable OR non-HIP build: should be False
        if not TORCHAO_AVAILABLE:
            assert result is False
        elif not hip_available:
            assert result is False
        # else: on a ROCm+torchao machine, result should be True (not testing here)

    def test_hip_alias_equals_rocm(self):
        """'hip' and 'amd' are aliases for 'rocm'."""
        r_rocm = TorchAOBackend.is_available_on_backend("rocm")
        r_hip = TorchAOBackend.is_available_on_backend("hip")
        r_amd = TorchAOBackend.is_available_on_backend("amd")
        assert r_rocm == r_hip == r_amd

    def test_cpu_is_supported_when_torchao_is_installed(self):
        """CPU is a supported backend — INT8 dynamic quantization runs there.

        This test used to assert False, with the comment "cpu not in
        ('cuda',)". The code had since grown to `backend in ("cuda", "cpu")`,
        and the test did not notice, because without torchao installed
        `is_available_on_backend` returns False from its first line and the
        assertion passed for a reason that had nothing to do with CPU. The
        suite only skipped on `importorskip("torchao")`, so nobody saw it.

        The code is the half that is right: quantizing an nn.Linear on CPU
        with Int8DynamicActivationInt8WeightConfig produces an Int8Tensor whose
        inner data really is int8, and a forward pass through it works.
        """
        result = TorchAOBackend.is_available_on_backend("cpu")
        assert isinstance(result, bool)
        assert result is TORCHAO_AVAILABLE

    def test_unknown_backend_returns_false(self):
        """The check must reject as well as accept.

        Without this, asserting that CPU is supported would pass just as well
        if the function returned True for everything.
        """
        assert TorchAOBackend.is_available_on_backend("wgpu") is False


class TestRocmDetectionAgreesWithHardwareDetector:
    """The ROCm answer here must match the one hardware_detector gives.

    PR #119 fixed two modules that wrote `torch.version.hip is not None` by
    hand: an empty HIP version string — which has been seen in the wild — made
    those say ROCm while `hardware_detector.is_rocm_build()` said CUDA. One
    question, two answers, depending on which module you asked.

    This file held a third copy that was outside #119's diff. Asserting
    agreement rather than a literal keeps it from drifting back: whatever the
    project decides ROCm means, there is one answer.
    """

    def test_empty_hip_string_is_not_rocm(self):
        """The exact input that split the two answers apart."""
        import torch
        from unittest.mock import patch

        from torchbridge.core.hardware_detector import is_rocm_build
        from torchbridge.precision.torchao_integration import TorchAOBackend

        with patch.object(torch.version, "hip", ""):
            assert is_rocm_build() is False
            assert TorchAOBackend.is_available_on_backend("rocm") is False

    def test_real_hip_version_is_rocm(self):
        """And a genuine ROCm build still reports ROCm.

        Without this, the test above would pass just as well if the check
        always returned False.
        """
        import torch
        from unittest.mock import patch

        from torchbridge.core.hardware_detector import is_rocm_build
        from torchbridge.precision.torchao_integration import TorchAOBackend

        with patch.object(torch.version, "hip", "6.2.41133"):
            assert is_rocm_build() is True
            assert TorchAOBackend.is_available_on_backend("rocm") is TORCHAO_AVAILABLE

    def test_the_two_agree_for_every_hip_value(self):
        """Agreement is the invariant, not any particular verdict."""
        import torch
        from unittest.mock import patch

        from torchbridge.core.hardware_detector import is_rocm_build
        from torchbridge.precision.torchao_integration import TorchAOBackend

        for hip in (None, "", "0", "6.2.41133"):
            with patch.object(torch.version, "hip", hip):
                detector = is_rocm_build()
                backend = TorchAOBackend.is_available_on_backend("rocm")
                # torchao's answer is gated on availability as well, so it can
                # only ever be the detector's answer ANDed with that.
                assert backend is (detector and TORCHAO_AVAILABLE), (
                    f"hip={hip!r}: detector says {detector}, torchao says {backend}"
                )
