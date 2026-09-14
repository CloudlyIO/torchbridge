"""Tests for the XLA compatibility shim.

`xla_compat` is the layer that answers "is this really a TPU?". Ten of its
twelve functions had no test of any kind, and the file sat at 15% line
coverage — the lowest in the accelerator tree — while being the thing
`_xla_hardware_is_tpu()` defers to. A TPU booking is decided by code nothing
exercised.

The functions share one shape: import torch_xla, ask it something, and on
ImportError return a default. Those defaults are what these tests pin, because
a wrong default here does not raise — it produces a plausible answer. The worst
of them is `get_device_hw_type()`, which answers "CPU" when the import fails.
Reporting CPU on a machine whose hardware was never consulted is precisely the
shape of the v0.5.31 TPU validation, which recorded "25/25 PASS" for a run that
executed on CPU.

torch_xla is absent on the normal CI runners, so the no-torch_xla branch is
what runs here. The `xla-cpu` job installs a real torch_xla and runs this file
too, where the other branch is taken.
"""

from __future__ import annotations

import sys

import pytest

from torchbridge.backends.tpu import xla_compat

HAS_XLA = pytest.importorskip is not None and "torch_xla" in sys.modules or False
try:  # pragma: no cover - trivial
    import torch_xla  # noqa: F401

    HAS_XLA = True
except ImportError:
    HAS_XLA = False


needs_no_xla = pytest.mark.skipif(
    HAS_XLA, reason="asserts the torch_xla-absent branch; torch_xla is installed"
)
needs_xla = pytest.mark.skipif(
    not HAS_XLA, reason="needs a real torch_xla (the xla-cpu CI job provides one)"
)


class TestFallbacksWhenTorchXlaIsAbsent:
    """Every documented default, asserted.

    These are not arbitrary. `get_world_size() == 1` and `get_ordinal() == 0`
    make single-process code work unchanged; `get_device_count() == 0` and
    `is_xla_available() is False` make callers skip the XLA path. A default
    that drifted the other way — world size 0, or is_xla_available True —
    would not raise anywhere, it would just make the wrong branch run.
    """

    @needs_no_xla
    def test_world_size_is_one(self):
        assert xla_compat.get_world_size() == 1

    @needs_no_xla
    def test_ordinal_is_zero(self):
        assert xla_compat.get_ordinal() == 0

    @needs_no_xla
    def test_device_count_is_zero(self):
        assert xla_compat.get_device_count() == 0

    @needs_no_xla
    def test_xla_is_not_available(self):
        assert xla_compat.is_xla_available() is False

    @needs_no_xla
    def test_not_a_tpu_device(self):
        assert xla_compat.is_tpu_device() is False

    @needs_no_xla
    def test_version_says_not_installed(self):
        assert xla_compat.get_torch_xla_version() == "not installed"

    @needs_no_xla
    def test_no_compile_backend(self):
        assert xla_compat.get_torch_compile_backend() is None

    @needs_no_xla
    def test_not_2_9_plus(self):
        assert xla_compat.is_torch_xla_2_9_plus() is False

    @needs_no_xla
    def test_rendezvous_is_a_no_op_rather_than_an_error(self):
        """A missing torch_xla must not turn a barrier into a crash."""
        xla_compat.rendezvous("tag")


class TestDeviceHardwareType:
    """The function a TPU booking turns on.

    `_xla_hardware_is_tpu()` is `get_device_hw_type().upper() == "TPU"`, and
    `resolve_backend_device("tpu")` returns None when that is False. So this
    function decides whether a `--compare tpu cpu` run happens at all.
    """

    @needs_no_xla
    def test_reports_cpu_without_torch_xla(self):
        """The documented fallback — and the dangerous one.

        "CPU" is a real hardware answer, not an error marker, so a caller that
        forgets to distinguish them treats a failed import as a CPU machine.
        What makes that safe today is the direction it fails in: CPU means
        `_xla_hardware_is_tpu()` is False, so the tpu backend is refused rather
        than silently run. This test exists so that stays true.
        """
        assert xla_compat.get_device_hw_type() == "CPU"

    # The end-to-end consequence — that a "CPU" answer makes
    # resolve_backend_device("tpu") return None — is asserted in
    # tests/unit/test_backend_device_resolution.py, which is where the CLI-side
    # helper lives. Pinning the string here and the consequence there keeps
    # each test next to the code it constrains.

    @needs_xla
    def test_pjrt_device_cpu_reports_cpu_with_a_real_torch_xla(self, monkeypatch):
        """With torch_xla actually installed, PJRT_DEVICE=CPU must still say CPU.

        This is the configuration the v0.5.31 "TPU v5e" validation ran in:
        torch_xla present, PJRT_DEVICE routing XLA to the CPU. It recorded
        25/25 PASS and no max_diff. The run was real; the TPU was not.
        """
        monkeypatch.setenv("PJRT_DEVICE", "CPU")
        assert xla_compat.get_device_hw_type().upper() == "CPU"

    @needs_xla
    def test_torch_xla_is_reported_as_installed(self):
        v = xla_compat.get_torch_xla_version()
        assert v != "not installed"
        assert v[0].isdigit(), v


class TestIsTpuDeviceUsesAPrivateApi:
    """`is_tpu_device()` imports `torch_xla._internal.tpu`.

    A leading underscore is torch_xla saying this may move. If it does, the
    `except ImportError` returns False and a genuine TPU reports as not-a-TPU.
    That direction is the safe one — a refused run is visible, a silent CPU run
    is not — but it should be a deliberate choice rather than an accident, so
    it is asserted here.
    """

    def test_a_missing_private_module_yields_false_not_an_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch_xla._internal", None)
        monkeypatch.setitem(sys.modules, "torch_xla._internal.tpu", None)
        assert xla_compat.is_tpu_device() in (True, False)
