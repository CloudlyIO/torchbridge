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

    def test_cpu_returns_false_by_default(self):
        """CPU backend is experimental; should return False unless specifically supported."""
        result = TorchAOBackend.is_available_on_backend("cpu")
        assert isinstance(result, bool)
        # CPU is NOT in the supported list ("cuda" only), so:
        if TORCHAO_AVAILABLE:
            assert result is False  # cpu not in ("cuda",)
        else:
            assert result is False
