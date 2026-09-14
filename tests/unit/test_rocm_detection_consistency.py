"""
Every ROCm question in the codebase must get the same answer.

``torch.version.hip`` is unset on a CUDA build and set on a ROCm one — but an
empty string has been seen in the wild, and the codebase disagreed on what that
means. ``is_rocm_build()`` uses truthiness and calls it CUDA; two call sites
used ``is not None`` and called the same value ROCm. Backend availability and
CK kernel dispatch therefore contradicted the detector on exactly one value.

Nothing here needs a GPU.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from torchbridge.core.hardware_detector import is_rocm_build


@pytest.fixture
def hip(request):
    """Set torch.version.hip to the parametrised value for one test."""
    with patch.object(torch.version, "hip", request.param, create=True):
        yield request.param


class TestTheEmptyStringIsTheDisagreement:
    """The value the two styles answered differently."""

    @pytest.mark.parametrize("hip", ["", None], indirect=True)
    def test_an_empty_or_unset_hip_is_not_rocm(self, hip):
        assert is_rocm_build() is False

    @pytest.mark.parametrize("hip", ["6.2.0", "5.7"], indirect=True)
    def test_a_real_hip_version_is_rocm(self, hip):
        assert is_rocm_build() is True


class TestCallSitesAgreeWithTheDetector:
    """Both sites now ask the shared helper, so they cannot drift from it."""

    @pytest.mark.parametrize("hip", ["", None, "6.2.0"], indirect=True)
    def test_ck_dispatch_matches_is_rocm_build(self, hip):
        pytest.importorskip("flash_attn")
        from torchbridge.attention.dispatch.dispatcher import AttentionDispatcher

        assert AttentionDispatcher._check_flash_attention_ck() == is_rocm_build()

    @pytest.mark.parametrize("hip", ["", None, "6.2.0"], indirect=True)
    def test_amd_availability_matches_is_rocm_build(self, hip):
        """Only meaningful when a CUDA-API device exists; otherwise the check
        short-circuits before the vendor question is asked."""
        from torchbridge.backends.backend_factory import BackendFactory, BackendType

        BackendFactory._availability_checks.pop(BackendType.AMD, None)
        with patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.__config__, "show", return_value="no vendor here"):
                assert BackendFactory._check_amd_available() == is_rocm_build()

    def test_neither_site_reads_torch_version_hip_directly(self):
        """Pins the routing, not just today's answers: a future edit that
        reintroduces a local check fails here rather than drifting quietly."""
        import inspect

        from torchbridge.attention.dispatch import dispatcher
        from torchbridge.backends import backend_factory

        for func in (
            dispatcher.AttentionDispatcher._check_flash_attention_ck,
            backend_factory.BackendFactory._check_amd_available,
        ):
            src = inspect.getsource(func)
            assert "version.hip" not in src, (
                f"{func.__qualname__} tests torch.version.hip itself instead of "
                f"calling is_rocm_build()"
            )
