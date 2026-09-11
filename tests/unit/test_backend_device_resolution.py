"""
Unit tests for backend name → torch.device resolution in tb-validate.

The resolver must not collapse ``cuda`` and ``rocm`` onto the same device.
A ROCm build of torch exposes AMD GPUs through the ``cuda`` device type, so
``torch.device("cuda")`` is correct for *both* vendors — which means the name
alone cannot be trusted. Without a ``torch.version.hip`` check, a
``--compare cuda rocm`` run points both halves at the same physical GPU and
reports near-zero divergence with no error.

The build flavour is simulated by patching ``torch.cuda.is_available`` and
``torch.version.hip``; this suite must stay runnable on a CPU-only machine.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch

from torchbridge.cli.validate import resolve_backend_device


@contextmanager
def _torch_build(cuda_available: bool, hip: str | None):
    """Simulate an NVIDIA build, a ROCm build, or a machine with no GPU."""
    with patch.object(torch.cuda, "is_available", return_value=cuda_available):
        with patch.object(torch.version, "hip", hip, create=True):
            yield


NVIDIA = {"cuda_available": True, "hip": None}
ROCM = {"cuda_available": True, "hip": "6.0.32830"}
NO_GPU = {"cuda_available": False, "hip": None}


class TestNvidiaBuild:
    """On an NVIDIA build, only cuda (and gpu) may resolve."""

    def test_cuda_resolves(self):
        with _torch_build(**NVIDIA):
            assert resolve_backend_device("cuda") == torch.device("cuda")

    def test_rocm_is_rejected(self):
        with _torch_build(**NVIDIA):
            assert resolve_backend_device("rocm") is None

    def test_gpu_resolves_to_whatever_the_box_is(self):
        with _torch_build(**NVIDIA):
            assert resolve_backend_device("gpu") == torch.device("cuda")


class TestRocmBuild:
    """On a ROCm build, only rocm (and gpu) may resolve."""

    def test_rocm_resolves(self):
        with _torch_build(**ROCM):
            assert resolve_backend_device("rocm") == torch.device("cuda")

    def test_cuda_is_rejected(self):
        with _torch_build(**ROCM):
            assert resolve_backend_device("cuda") is None

    def test_gpu_resolves_to_whatever_the_box_is(self):
        with _torch_build(**ROCM):
            assert resolve_backend_device("gpu") == torch.device("cuda")


class TestTheBugItself:
    """The pair that silently compared a GPU against itself."""

    @pytest.mark.parametrize("build", [NVIDIA, ROCM], ids=["nvidia", "rocm"])
    def test_cuda_and_rocm_never_both_resolve(self, build):
        with _torch_build(**build):
            a = resolve_backend_device("cuda")
            b = resolve_backend_device("rocm")
        assert not (a is not None and b is not None), (
            "cuda and rocm both resolved on one machine — a --compare cuda rocm "
            "run would point both halves at the same physical GPU"
        )


class TestNoGpu:
    """A CPU-only machine rejects every GPU name, as before."""

    @pytest.mark.parametrize("name", ["cuda", "rocm", "gpu"])
    def test_gpu_names_rejected(self, name):
        with _torch_build(**NO_GPU):
            assert resolve_backend_device(name) is None

    def test_cpu_always_resolves(self):
        with _torch_build(**NO_GPU):
            assert resolve_backend_device("cpu") == torch.device("cpu")


class TestUnchangedBehaviour:
    """Names unrelated to the fix must behave exactly as before."""

    def test_unknown_name_returns_none(self):
        with _torch_build(**NVIDIA):
            assert resolve_backend_device("banana") is None

    def test_name_is_case_insensitive(self):
        with _torch_build(**NVIDIA):
            assert resolve_backend_device("CUDA") == torch.device("cuda")

    def test_trainium_without_neuron_returns_none(self):
        with _torch_build(**NO_GPU):
            with patch.dict("sys.modules", {"torch_neuronx": None}):
                assert resolve_backend_device("trainium") is None


class TestErrorMessage:
    """A refusal must say *why*, since 'not available' misleads here."""

    def test_cuda_on_rocm_box_names_the_build(self):
        from torchbridge.cli.validate import explain_unavailable_backend

        with _torch_build(**ROCM):
            msg = explain_unavailable_backend("cuda")
        assert "ROCm" in msg and "cuda" in msg

    def test_rocm_on_nvidia_box_names_the_build(self):
        from torchbridge.cli.validate import explain_unavailable_backend

        with _torch_build(**NVIDIA):
            msg = explain_unavailable_backend("rocm")
        assert "CUDA" in msg and "rocm" in msg

    def test_vendor_mismatch_points_at_record_replay(self):
        from torchbridge.cli.validate import explain_unavailable_backend

        with _torch_build(**ROCM):
            msg = explain_unavailable_backend("cuda")
        assert "replay" in msg.lower()

    def test_plain_unavailable_keeps_the_old_wording(self):
        from torchbridge.cli.validate import explain_unavailable_backend

        with _torch_build(**NO_GPU):
            msg = explain_unavailable_backend("cuda")
        assert msg == "Backend 'cuda' not available on this machine."


class TestAliasLoophole:
    """``gpu`` is an alias, so it can re-open the same-device hole by another name.

    Refusing ``cuda`` on an AMD build closes ``--compare cuda rocm``, but
    ``--compare rocm gpu`` resolves both halves to the same physical GPU again.
    Two *different* names landing on one device is the bug, whatever the names.
    An explicit self-pair (``cpu cpu``, ``cuda cuda``) is a deliberate control
    run and must keep working.
    """

    def test_cuda_and_gpu_collide_on_nvidia(self):
        from torchbridge.cli.validate import same_device_pair

        with _torch_build(**NVIDIA):
            assert same_device_pair("cuda", "gpu") is True

    def test_rocm_and_gpu_collide_on_rocm(self):
        from torchbridge.cli.validate import same_device_pair

        with _torch_build(**ROCM):
            assert same_device_pair("rocm", "gpu") is True

    def test_explicit_self_pair_is_allowed(self):
        from torchbridge.cli.validate import same_device_pair

        with _torch_build(**NO_GPU):
            assert same_device_pair("cpu", "cpu") is False

    def test_genuinely_different_devices_are_fine(self):
        from torchbridge.cli.validate import same_device_pair

        with _torch_build(**NVIDIA):
            assert same_device_pair("cuda", "cpu") is False

    def test_unresolvable_pair_is_not_flagged_here(self):
        """A missing backend is the caller's existing error, not this check."""
        from torchbridge.cli.validate import same_device_pair

        with _torch_build(**NVIDIA):
            assert same_device_pair("rocm", "gpu") is False


class TestHipEdgeCases:
    """``torch.version.hip`` is not always a clean version string."""

    def test_empty_hip_string_is_not_a_rocm_build(self):
        with _torch_build(cuda_available=True, hip=""):
            assert resolve_backend_device("cuda") == torch.device("cuda")
            assert resolve_backend_device("rocm") is None

    def test_missing_hip_attribute_is_not_a_rocm_build(self):
        with patch.object(torch.cuda, "is_available", return_value=True):
            saved = getattr(torch.version, "hip", None)
            try:
                if hasattr(torch.version, "hip"):
                    del torch.version.hip
                assert resolve_backend_device("cuda") == torch.device("cuda")
                assert resolve_backend_device("rocm") is None
            finally:
                torch.version.hip = saved


class TestCliRefusesAliasCollision:
    """The collision must stop the run, not just be detectable."""

    @staticmethod
    def _args(pair, trace):
        import argparse

        return argparse.Namespace(
            compare=list(pair),
            trace=trace,
            steps=2,
            autoregressive=False,
            model=None,
            input_shape="1,4",
            per_layer=False,
            dtype="float32",
            output=None,
            trace_output=None,
            ci=False,
            verbose=False,
            model_family=None,
            cert=None,
        )

    def test_compare_rocm_gpu_on_amd_exits_nonzero(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        with _torch_build(**ROCM):
            rc = ValidateCommand._run_compare(self._args(("rocm", "gpu"), False))
        assert rc == 1
        assert "same" in capsys.readouterr().out.lower()

    def test_trace_cuda_gpu_on_nvidia_exits_nonzero(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        with _torch_build(**NVIDIA):
            rc = ValidateCommand._run_trace(self._args(("cuda", "gpu"), True))
        assert rc == 1
        assert "same" in capsys.readouterr().out.lower()

    def test_cpu_cpu_control_run_still_works(self):
        from torchbridge.cli.validate import ValidateCommand

        assert ValidateCommand._run_compare(self._args(("cpu", "cpu"), False)) == 0


class TestCiModeErrors:
    """--ci consumers parse stdout as JSON, so a refusal must stay machine-readable."""

    @staticmethod
    def _args(**kw):
        import argparse

        d = {
            "compare": ["cpu", "cpu"],
            "trace": False,
            "steps": 2,
            "autoregressive": False,
            "model": None,
            "input_shape": "1,4",
            "per_layer": False,
            "dtype": "float32",
            "output": None,
            "trace_output": None,
            "ci": True,
            "verbose": False,
            "model_family": None,
            "cert": None,
        }
        d.update(kw)
        return argparse.Namespace(**d)

    @staticmethod
    def _error_of(captured):
        import json

        return json.loads(captured.out.strip().splitlines()[-1])["error"]

    def test_unknown_family_emits_json(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        ValidateCommand._run_compare(self._args(model_family="decoder-larg"))
        assert "decoder-larg" in self._error_of(capsys.readouterr())

    def test_device_collision_emits_json(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        with _torch_build(**ROCM):
            ValidateCommand._run_compare(self._args(compare=["rocm", "gpu"]))
        assert "same device" in self._error_of(capsys.readouterr())

    def test_vendor_mismatch_emits_json(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        with _torch_build(**ROCM):
            ValidateCommand._run_compare(self._args(compare=["cuda", "cpu"]))
        assert "ROCm" in self._error_of(capsys.readouterr())

    def test_trace_path_collision_emits_json(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        with _torch_build(**NVIDIA):
            ValidateCommand._run_trace(self._args(compare=["cuda", "gpu"], trace=True))
        assert "same device" in self._error_of(capsys.readouterr())


class TestOneSharedVendorCheck:
    """The ROCm check belongs in one place, not beside the copies already there.

    The repository already tests ``torch.version.hip`` in eight other files, in
    several different styles. Adding another private copy here means a future
    change to the detection rule has to be found in every one of them.
    """

    def test_the_shared_helper_exists_in_core(self):
        from torchbridge.core.hardware_detector import is_rocm_build

        assert callable(is_rocm_build)

    def test_validate_uses_the_shared_helper(self):
        """Patching the shared helper must change the resolver's answer."""
        with patch(
            "torchbridge.core.hardware_detector.is_rocm_build", return_value=True
        ):
            with patch.object(torch.cuda, "is_available", return_value=True):
                assert resolve_backend_device("cuda") is None
                assert resolve_backend_device("rocm") == torch.device("cuda")

    def test_shared_helper_treats_empty_hip_as_not_rocm(self):
        from torchbridge.core.hardware_detector import is_rocm_build

        with patch.object(torch.version, "hip", "", create=True):
            assert is_rocm_build() is False
        with patch.object(torch.version, "hip", "6.0.32830", create=True):
            assert is_rocm_build() is True


class TestTpuAndXlaNames:
    """The CLI must be able to name a TPU at all.

    Before this, the accepted names were cuda, rocm, gpu, mps, trainium, neuron
    and cpu. A rented TPU could not be addressed, and the workaround — passing
    ``trainium`` — wrote the wrong hardware name into the results file.
    """

    @contextmanager
    def _torch_xla_installed(self, present: bool):
        import sys

        with patch.dict(sys.modules, {"torch_xla": object() if present else None}):
            yield

    def test_tpu_resolves_when_torch_xla_is_present(self):
        with self._torch_xla_installed(True):
            assert resolve_backend_device("tpu") == torch.device("xla")

    def test_xla_resolves_when_torch_xla_is_present(self):
        with self._torch_xla_installed(True):
            assert resolve_backend_device("xla") == torch.device("xla")

    def test_tpu_is_refused_without_torch_xla(self):
        with self._torch_xla_installed(False):
            assert resolve_backend_device("tpu") is None

    def test_name_is_case_insensitive(self):
        with self._torch_xla_installed(True):
            assert resolve_backend_device("TPU") == torch.device("xla")

    def test_tpu_and_trainium_are_flagged_as_one_device(self):
        """On one machine they are the same device, so pairing them is refused.

        Both accelerator paths must be importable for this to be a real pair —
        trainium needs torch_neuronx, tpu needs torch_xla.
        """
        import sys

        from torchbridge.cli.validate import same_device_pair

        with patch.dict(
            sys.modules, {"torch_xla": object(), "torch_neuronx": object()}
        ):
            assert same_device_pair("tpu", "trainium") is True

    def test_tpu_against_cpu_is_a_real_pair(self):
        from torchbridge.cli.validate import same_device_pair

        with self._torch_xla_installed(True):
            assert same_device_pair("tpu", "cpu") is False


class TestNonTraceCompareUsesTheCanonicalKey:
    """--compare and --trace must judge the same chip by the same limit.

    Only MultiStepTracer canonicalised the backend name. The plain --compare
    path looked the raw name up, so tpu, neuron and gpu found no row and took
    the 1.0e-3 safe default. The verdict depended on which alias was typed.
    """

    @pytest.mark.parametrize(
        "alias,canonical",
        [("tpu", "xla"), ("neuron", "trainium"), ("xla", "xla"), ("cuda", "cuda")],
    )
    @pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
    def test_alias_and_canonical_name_get_the_same_tolerance(
        self, alias, canonical, dtype
    ):
        from torchbridge.testing.tolerance_db import ToleranceDB
        from torchbridge.testing.trace_validator import _tolerance_key

        db = ToleranceDB()
        assert db.get(_tolerance_key(alias), dtype).atol == pytest.approx(
            db.get(canonical, dtype).atol
        )

    @pytest.mark.parametrize("alias", ["tpu", "neuron", "gpu"])
    def test_the_aliases_no_longer_land_on_the_fallback(self, alias):
        """Each of these used to return the safe default with source
        'fallback', which is the database saying nobody measured it."""
        from torchbridge.testing.tolerance_db import ToleranceDB
        from torchbridge.testing.trace_validator import _tolerance_key

        entry = ToleranceDB().get(_tolerance_key(alias), "float32")
        assert getattr(entry, "source", None) == "measured"

    def test_the_compare_path_reads_the_canonical_key(self):
        """Pins the wiring, not just the helper: _run_compare must call it."""
        import inspect

        from torchbridge.cli.validate import ValidateCommand

        src = inspect.getsource(ValidateCommand._run_compare)
        assert "_tolerance_key(backend1)" in src
        assert "backend1.lower() if backend1.lower()" not in src
