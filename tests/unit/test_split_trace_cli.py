"""
Unit tests for the record/replay half of tb-validate --trace.

A cross-vendor pair such as ``cuda vs rocm`` cannot run in one process: a torch
install is built against CUDA or ROCm but never both, and both expose GPUs as
device ``cuda``, so one process can only drive one vendor. The split workflow is therefore two commands with
the *same* ``--compare`` pair: the first machine records its half, the second
replays that exact token sequence and compares.

The consequence for device resolution is the point of most of these tests. In
record mode only backend A exists locally; in replay mode only backend B does.
Demanding both would make the workflow impossible on the very machines it is for.
"""

from __future__ import annotations

import argparse
import json

import pytest
import torch

from torchbridge.cli.validate import ValidateCommand


def _args(**kw):
    d = {
        "compare": ["cpu", "cpu"],
        "trace": True,
        "steps": 3,
        "autoregressive": False,
        "model": None,
        "input_shape": "1,4",
        "per_layer": False,
        "dtype": "float32",
        "output": None,
        "trace_output": None,
        "ci": False,
        "verbose": False,
        "model_family": None,
        "cert": None,
        "record": None,
        "replay": None,
        "compare_records": None,
        "strict_env": False,
    }
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture
def saved_model(tmp_path):
    """One model on disk, so record and replay load identical weights.

    The default smoke model is initialised randomly per invocation, so a split
    across two processes would compare two different models. Real runs pass
    --model for the same reason.
    """
    import torch.nn as nn

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    path = tmp_path / "m.pt"
    torch.save(model, path)
    return str(path)


class TestRecord:
    def test_record_writes_a_file_and_succeeds(self, tmp_path):
        out = tmp_path / "a.pt"
        assert ValidateCommand._run_trace(_args(record=str(out))) == 0
        assert out.exists() and out.stat().st_size > 0

    def test_recorded_file_reloads_with_the_right_shape(self, tmp_path):
        from torchbridge.testing.trace_validator import SplitTraceRecord

        out = tmp_path / "a.pt"
        ValidateCommand._run_trace(_args(record=str(out), steps=3))
        rec = SplitTraceRecord.load(str(out))
        assert rec.role == "record"
        assert rec.backend == "cpu"
        assert rec.steps == 3
        assert len(rec.token_inputs) == 3

    def test_record_does_not_need_backend_b_present(self, tmp_path):
        """The other vendor's chip is on the other machine, by definition."""
        out = tmp_path / "a.pt"
        rc = ValidateCommand._run_trace(_args(compare=["cpu", "rocm"], record=str(out)))
        assert rc == 0, "record mode wrongly demanded the remote backend"


class TestReplay:
    def _record(self, tmp_path, model=None, **kw):
        out = tmp_path / "a.pt"
        ValidateCommand._run_trace(_args(record=str(out), model=model, **kw))
        return str(out)

    def test_replay_produces_a_comparison(self, tmp_path, saved_model):
        src = self._record(tmp_path, model=saved_model)
        res = tmp_path / "r.json"
        rc = ValidateCommand._run_trace(
            _args(replay=src, model=saved_model, trace_output=str(res))
        )
        assert rc == 0
        saved = json.loads(res.read_text())
        assert saved["steps"] == 3
        assert saved["backend_a"] == "cpu" and saved["backend_b"] == "cpu"

    def test_cpu_against_cpu_replay_shows_no_divergence(self, tmp_path, saved_model):
        """The control case: same backend, same weights, must agree exactly.

        This is what proves record-and-replay adds no error of its own, so any
        divergence a real cross-vendor run reports comes from the hardware.
        """
        src = self._record(tmp_path, model=saved_model)
        res = tmp_path / "r.json"
        rc = ValidateCommand._run_trace(
            _args(replay=src, model=saved_model, trace_output=str(res))
        )
        assert rc == 0
        saved = json.loads(res.read_text())
        assert saved["first_divergence_step"] is None
        assert saved["max_amplification"] == 1.0
        assert saved["final_passed"] is True

    def test_replay_does_not_need_backend_a_present(self, tmp_path, saved_model):
        """The recorded chip is on the other machine, by definition.

        The record has to *claim* that backend, though. Recording on cpu and
        then replaying under --compare rocm cpu is the failure Copilot found:
        the result takes its names from the file, so it came out labelled
        "cpu vs cpu" while the operator had asked about rocm.
        """
        from torchbridge.testing.trace_validator import SplitTraceRecord

        src = self._record(tmp_path, model=saved_model)
        leader = SplitTraceRecord.load(src)
        leader.backend = "rocm"  # as a real AMD machine would have written it
        remote = str(tmp_path / "rocm.pt")
        leader.save(remote)

        rc = ValidateCommand._run_trace(
            _args(compare=["rocm", "cpu"], replay=remote, model=saved_model)
        )
        assert rc == 0, "replay mode wrongly demanded the recording backend"

    def test_replay_refuses_a_record_from_a_different_backend(
        self, tmp_path, saved_model
    ):
        """The result is labelled from the record, so a mismatched pair answers
        a question the operator did not ask — and exits 0 doing it."""
        src = self._record(tmp_path, model=saved_model)  # recorded on cpu
        rc = ValidateCommand._run_trace(
            _args(compare=["rocm", "cpu"], replay=src, model=saved_model)
        )
        assert rc == 1

    def test_replay_refuses_a_follower_as_its_input(self, tmp_path, saved_model):
        """Replaying a replay yields a second follower, and the comparison then
        has no recorded half at all."""
        from torchbridge.testing.trace_validator import SplitTraceRecord

        src = self._record(tmp_path, model=saved_model)
        follower = tmp_path / "b.pt"
        ValidateCommand._run_trace(
            _args(replay=src, record=str(follower), model=saved_model)
        )
        assert SplitTraceRecord.load(str(follower)).role == "replay"

        rc = ValidateCommand._run_trace(_args(replay=str(follower), model=saved_model))
        assert rc == 1

    def test_different_weights_are_reported_not_hidden(self, tmp_path, caplog):
        """Two halves from different models must not pass silently.

        Each record stores a fingerprint of the weights. Without that check, a
        mismatched checkpoint would look like hardware divergence.
        """
        import logging

        src = self._record(tmp_path)  # random weights
        with caplog.at_level(logging.WARNING):
            ValidateCommand._run_trace(_args(replay=src))  # different random weights
        joined = " ".join(r.message.lower() for r in caplog.records)
        assert "fingerprint" in joined or "mismatch" in joined

    def test_replay_records_the_tolerance_used(self, tmp_path):
        src = self._record(tmp_path)
        res = tmp_path / "r.json"
        ValidateCommand._run_trace(
            _args(replay=src, trace_output=str(res), model_family="decoder-large")
        )
        saved = json.loads(res.read_text())
        assert saved["model_family"] == "decoder-large"
        assert saved["atol"] is not None


class TestCompareRecordsOffline:
    def test_a_record_and_its_saved_replay_compare_offline(self, tmp_path, saved_model):
        """--replay with --record keeps the second half, so it can be re-checked.

        Rented machines are destroyed straight after their half is taken, so
        keeping both halves is what allows a later re-comparison — for instance
        under a different tolerance — without paying for the hardware again.
        """
        a, b = tmp_path / "a.pt", tmp_path / "b.pt"
        ValidateCommand._run_trace(_args(record=str(a), model=saved_model))
        ValidateCommand._run_trace(
            _args(replay=str(a), record=str(b), model=saved_model)
        )
        assert b.exists(), "--replay with --record did not save the second half"

        res = tmp_path / "r.json"
        rc = ValidateCommand._run_trace(
            _args(compare_records=[str(a), str(b)], trace_output=str(res))
        )
        assert rc == 0
        assert json.loads(res.read_text())["final_passed"] is True


class TestErrors:
    def test_missing_record_file_fails_cleanly(self, tmp_path, capsys):
        rc = ValidateCommand._run_trace(_args(replay=str(tmp_path / "nope.pt")))
        assert rc == 1
        assert "nope.pt" in capsys.readouterr().out

    def test_compare_records_needs_a_record_and_its_replay(self, tmp_path, capsys):
        """Two independent records are not a pair — they saw different inputs.

        The exactness of the split rests on the follower being fed the leader's
        recorded tokens. Two separate recordings share nothing, so pairing them
        must be refused rather than producing a meaningless number.
        """
        a, b = tmp_path / "a.pt", tmp_path / "b.pt"
        ValidateCommand._run_trace(_args(record=str(a)))
        ValidateCommand._run_trace(_args(record=str(b)))
        rc = ValidateCommand._run_trace(_args(compare_records=[str(a), str(b)]))
        assert rc == 1
        out = capsys.readouterr().out
        assert "replay" in out, out

    def test_ci_mode_errors_stay_json(self, tmp_path, capsys):
        ValidateCommand._run_trace(_args(replay=str(tmp_path / "nope.pt"), ci=True))
        assert "error" in json.loads(capsys.readouterr().out.strip())


class TestFlagsRegistered:
    def test_record_replay_flags_exist(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        ValidateCommand.register(sub)
        a = parser.parse_args(
            ["validate", "--compare", "cuda", "rocm", "--trace", "--record", "x.pt"]
        )
        assert a.record == "x.pt"

    def test_replay_flag_parses(self):
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        ValidateCommand.register(sub)
        a = parser.parse_args(
            ["validate", "--compare", "cuda", "rocm", "--trace", "--replay", "x.pt"]
        )
        assert a.replay == "x.pt"


class TestRecordFilesAreLoadedSafely:
    """A record file arrives from another machine, so it is untrusted input.

    ``torch.load`` with ``weights_only=False`` unpickles arbitrary objects, which
    means a tampered record could execute code on the machine reading it. The
    split workflow's whole point is copying a file off a rented box, so that is
    exactly the threat model. The payload is tensors, strings, bools and ints —
    all of which ``weights_only=True`` supports — so nothing is given up.
    """

    def test_loader_does_not_unpickle_arbitrary_objects(self):
        import inspect

        from torchbridge.testing import trace_validator

        source = inspect.getsource(trace_validator.SplitTraceRecord.load)
        assert "weights_only=True" in source, (
            "record files come from another machine; loading them with "
            "weights_only=False allows code execution from a tampered file"
        )

    def test_a_saved_record_still_round_trips(self, tmp_path):
        from torchbridge.testing.trace_validator import SplitTraceRecord

        rec = SplitTraceRecord(
            backend="cuda", dtype="float32", autoregressive=True, is_lm=False
        )
        rec.token_inputs = [torch.zeros(1, 4)]
        rec.outputs = [torch.ones(1, 4)]
        rec.env = {"torch": "2.11.0", "model_fingerprint": "abc"}
        path = tmp_path / "r.pt"
        rec.save(str(path))

        back = SplitTraceRecord.load(str(path))
        assert back.backend == "cuda"
        assert back.steps == 1
        assert torch.equal(back.outputs[0], torch.ones(1, 4))
        assert back.env["model_fingerprint"] == "abc"


class TestEnvValuesArePlainStrings:
    """``_capture_env`` is annotated ``dict[str, str]`` and must hold real strings.

    ``torch.__version__`` is a ``TorchVersion`` object, not a ``str``. Storing it
    raw both breaks the annotation and makes the saved record unloadable under
    ``weights_only=True``, because that mode refuses unknown globals.
    """

    def test_every_captured_value_is_a_str(self):
        import torch.nn as nn

        from torchbridge.testing.trace_validator import _capture_env

        env = _capture_env(torch.device("cpu"), nn.Linear(2, 2))
        wrong = {k: type(v).__name__ for k, v in env.items() if type(v) is not str}
        assert not wrong, wrong

    def test_torch_version_is_captured_as_text(self):
        from torchbridge.testing.trace_validator import _capture_env

        env = _capture_env(torch.device("cpu"))
        assert env["torch"] == str(torch.__version__)
        assert type(env["torch"]) is str


class TestDtypeMismatchIsNotSilent:
    """Two halves recorded under different dtypes must not compare quietly.

    ``compare_records()`` looks the tolerance up from ``record_a.dtype``, so a
    mismatched pair is judged by the *recording* half's limit while half the
    data came from a different precision. That is the same silent-wrong-answer
    shape the vendor check exists to remove, so it raises.

    ``replay()``'s warning also said the lookup would use the replay dtype,
    which was simply untrue.
    """

    def _pair(self, dtype_a, dtype_b):
        from torchbridge.testing.trace_validator import SplitTraceRecord

        a = SplitTraceRecord(
            backend="cuda",
            dtype=dtype_a,
            autoregressive=False,
            is_lm=False,
            role="record",
        )
        b = SplitTraceRecord(
            backend="rocm",
            dtype=dtype_b,
            autoregressive=False,
            is_lm=False,
            role="replay",
        )
        for rec in (a, b):
            rec.token_inputs = [torch.zeros(1, 4)]
            rec.outputs = [torch.zeros(1, 4)]
            rec.env = {"torch": "2.0", "transformers": "4.0", "model_fingerprint": "x"}
        return a, b

    def test_mismatched_dtypes_are_refused(self):
        from torchbridge.testing.trace_validator import compare_records

        with pytest.raises(ValueError, match="dtype"):
            compare_records(*self._pair("float32", "bfloat16"))

    def test_matching_dtypes_still_compare(self):
        from torchbridge.testing.trace_validator import compare_records

        assert compare_records(*self._pair("float32", "float32")).final_passed is True

    def test_replay_warning_names_the_recorded_dtype(self):
        """The message must not claim the replay dtype decides the tolerance."""
        import inspect

        from torchbridge.testing import trace_validator

        source = inspect.getsource(trace_validator.MultiStepTracer.replay)
        assert "use the replay dtype" not in source, (
            "the tolerance comes from record_a.dtype, so this message is wrong"
        )


class TestSplitModeDeviceContract:
    """The tracer is annotated ``torch.device``, so it must not receive None.

    In record mode backend B is absent locally and in replay mode backend A is,
    which is the whole point of splitting. Passing the missing side through as
    None violates the constructor's own contract and only works today because
    nothing touches the unused half.
    """

    @staticmethod
    def _args(**kw):
        import argparse

        d = {
            "compare": ["cpu", "rocm"],
            "trace": True,
            "steps": 2,
            "autoregressive": False,
            "model": None,
            "input_shape": "1,4",
            "per_layer": False,
            "dtype": "float32",
            "output": None,
            "trace_output": None,
            "ci": False,
            "verbose": False,
            "model_family": None,
            "cert": None,
            "record": None,
            "replay": None,
            "compare_records": None,
        }
        d.update(kw)
        return argparse.Namespace(**d)

    def test_record_mode_never_passes_none_as_a_device(self, tmp_path):
        from unittest.mock import patch

        from torchbridge.cli.validate import ValidateCommand
        from torchbridge.testing import trace_validator

        seen = {}
        real = trace_validator.MultiStepTracer

        def _spy(*a, **kw):
            seen.update(kw)
            return real(*a, **kw)

        with patch.object(trace_validator, "MultiStepTracer", _spy):
            ValidateCommand._run_trace(self._args(record=str(tmp_path / "a.pt")))

        assert seen.get("device_a") is not None
        assert seen.get("device_b") is not None, (
            "backend B is absent in record mode, but the tracer still needs a device"
        )


class TestFailedRecordingIsNotReportedAsSuccess:
    """record() returns whatever it managed before an exception stopped it.

    Saving that and exiting 0 tells the operator the recording worked, while the
    next stage rejects the file for having no steps. On a rented machine that is
    a paid booking spent on an artifact nobody can use, discovered later.
    """

    @staticmethod
    def _break_after(n_steps):
        """A tracer whose record() stops after n_steps, as a failing model would."""
        from torchbridge.testing import trace_validator

        real = trace_validator.MultiStepTracer

        class _Short(real):  # type: ignore[misc,valid-type]
            def record(self, input_ids, steps, autoregressive=False):
                # Truncate rather than ask for fewer steps: record() rejects
                # steps=0 up front, so requesting it would exercise argument
                # validation instead of the "inference died partway" path this
                # stands in for.
                rec = super().record(
                    input_ids=input_ids, steps=steps, autoregressive=autoregressive
                )
                del rec.outputs[n_steps:]
                del rec.token_inputs[n_steps:]
                return rec

        return _Short

    def test_empty_recording_fails_and_writes_nothing(self, tmp_path):
        from unittest.mock import patch

        from torchbridge.testing import trace_validator

        out = tmp_path / "a.pt"
        with patch.object(trace_validator, "MultiStepTracer", self._break_after(0)):
            rc = ValidateCommand._run_trace(_args(record=str(out), steps=3))

        assert rc == 1
        assert not out.exists(), "a failed recording must not leave an artifact"

    def test_short_recording_fails_and_writes_nothing(self, tmp_path):
        """A partial record would compare fewer steps than asked for, and the
        result would not say so."""
        from unittest.mock import patch

        from torchbridge.testing import trace_validator

        out = tmp_path / "a.pt"
        with patch.object(trace_validator, "MultiStepTracer", self._break_after(1)):
            rc = ValidateCommand._run_trace(_args(record=str(out), steps=3))

        assert rc == 1
        assert not out.exists()

    def test_complete_recording_still_succeeds(self, tmp_path):
        out = tmp_path / "a.pt"
        assert ValidateCommand._run_trace(_args(record=str(out), steps=3)) == 0
        assert out.exists()


class TestRecordedFamilyTravelsWithTheArtifact:
    """compare_records() runs offline, possibly on a third machine. If the
    family is not in the file it falls back to the coarse tolerance row, which
    is the bug the whole tolerance change exists to prevent."""

    def test_family_is_written_into_the_record(self, tmp_path):
        from torchbridge.testing.trace_validator import SplitTraceRecord

        out = tmp_path / "a.pt"
        ValidateCommand._run_trace(
            _args(record=str(out), model_family="decoder-large", steps=2)
        )
        assert SplitTraceRecord.load(str(out)).model_family == "decoder-large"

    def test_offline_compare_uses_the_recorded_family(self, tmp_path):
        """No --model-family on the compare host, yet the recorded one is used."""
        from torchbridge.testing.trace_validator import (
            SplitTraceRecord,
            compare_records,
        )

        a = tmp_path / "a.pt"
        b = tmp_path / "b.pt"
        ValidateCommand._run_trace(
            _args(record=str(a), model_family="decoder-large", steps=2)
        )
        ValidateCommand._run_trace(
            _args(replay=str(a), record=str(b), model_family="decoder-large", steps=2)
        )

        result = compare_records(
            SplitTraceRecord.load(str(a)), SplitTraceRecord.load(str(b))
        )
        assert result.model_family == "decoder-large"

    def test_explicit_family_still_overrides_the_record(self, tmp_path):
        from torchbridge.testing.trace_validator import (
            SplitTraceRecord,
            compare_records,
        )

        a = tmp_path / "a.pt"
        b = tmp_path / "b.pt"
        ValidateCommand._run_trace(
            _args(record=str(a), model_family="decoder-large", steps=2)
        )
        ValidateCommand._run_trace(
            _args(replay=str(a), record=str(b), model_family="decoder-large", steps=2)
        )

        result = compare_records(
            SplitTraceRecord.load(str(a)),
            SplitTraceRecord.load(str(b)),
            model_family="decoder-small",
        )
        assert result.model_family == "decoder-small"


class TestFingerprintDoesNotCopyTheWholeModel:
    """The docstring promises endpoint sampling. Converting the full parameter
    to float32 on the host first would copy the entire model — ~2.5 GB for an
    8B checkpoint's embedding alone — twice per trace."""

    def test_only_the_sample_is_converted_and_moved(self):
        from torchbridge.testing.trace_validator import _model_fingerprint

        big = torch.nn.Linear(2048, 512, bias=False).to(torch.bfloat16)
        moved = []

        real_float = torch.Tensor.float

        def _spy_float(self, *a, **kw):
            moved.append(self.numel())
            return real_float(self, *a, **kw)

        torch.Tensor.float = _spy_float  # type: ignore[method-assign]
        try:
            _model_fingerprint(big)
        finally:
            torch.Tensor.float = real_float  # type: ignore[method-assign]

        assert moved, "the fingerprint must still convert its sample"
        assert max(moved) <= 16, (
            f"converted a tensor of {max(moved)} elements; only the 16-value "
            f"sample should be converted, not the whole parameter"
        )


class TestTrajectoryIsNormalisedBeforeTheLoop:
    """The running trajectory is grown with ``next_token.cpu()``, so a caller
    passing ``input_ids`` already on the accelerator hit a device mismatch on
    the first append. It is now detached and moved to CPU up front.

    The device half of that cannot be reproduced here: it needs two device
    types, and this machine has only CPU, where the buggy and fixed versions
    behave identically. The detach half is checkable and is checked below; the
    rest rests on reading the code, and is recorded as such rather than covered
    by a test that would pass either way.
    """

    @staticmethod
    def _tracer(**kw):
        from torchbridge.testing.trace_validator import MultiStepTracer

        d = {
            "model": torch.nn.Linear(4, 4),
            "device_a": torch.device("cpu"),
            "device_b": torch.device("cpu"),
            "backend_a": "cpu",
            "backend_b": "cpu",
            "dtype": "float32",
            "is_lm": False,
        }
        d.update(kw)
        return MultiStepTracer(**d)

    def test_recorded_trajectory_carries_no_autograd_history(self):
        """clone() alone keeps grad_fn, so the record would pickle a piece of
        the caller's graph into an artifact meant to hold plain tensors."""
        given = torch.randn(1, 4, requires_grad=True)
        rec = self._tracer().record(input_ids=given, steps=2, autoregressive=True)
        assert all(not t.requires_grad for t in rec.token_inputs)
        assert all(t.grad_fn is None for t in rec.token_inputs)

    def test_the_caller_s_tensor_is_not_mutated(self):
        given = torch.randn(1, 4)
        before = given.clone()
        self._tracer().record(input_ids=given, steps=2, autoregressive=True)
        assert torch.equal(given, before)

    def test_recorded_trajectory_is_cpu(self):
        rec = self._tracer().record(
            input_ids=torch.randn(1, 4), steps=2, autoregressive=True
        )
        assert all(t.device.type == "cpu" for t in rec.token_inputs)


class TestReplayCommandCarriesTheRunSettings:
    """The printed command is what the operator runs on the second machine.

    Written out by hand it omitted --model, so following it verbatim built a
    freshly initialised smoke model there. The fingerprint mismatch only warns,
    so a verdict came out of two different sets of weights.
    """

    def test_the_model_is_included(self, tmp_path, saved_model, capsys):
        ValidateCommand._run_trace(
            _args(record=str(tmp_path / "a.pt"), model=saved_model)
        )
        assert f"--model {saved_model}" in capsys.readouterr().out

    def test_family_dtype_and_shape_are_included(self, tmp_path, saved_model, capsys):
        ValidateCommand._run_trace(
            _args(
                record=str(tmp_path / "a.pt"),
                model=saved_model,
                model_family="decoder-large",
                dtype="float32",
                input_shape="1,4",
                autoregressive=False,
            )
        )
        out = capsys.readouterr().out
        assert "--model-family decoder-large" in out
        assert "--dtype float32" in out
        assert "--input-shape 1,4" in out

    def test_autoregressive_is_carried_over(self, tmp_path, saved_model, capsys):
        ValidateCommand._run_trace(
            _args(record=str(tmp_path / "a.pt"), model=saved_model, autoregressive=True)
        )
        assert "--autoregressive" in capsys.readouterr().out

    def test_a_smoke_model_recording_says_it_is_not_a_measurement(
        self, tmp_path, capsys
    ):
        ValidateCommand._run_trace(_args(record=str(tmp_path / "a.pt")))
        out = capsys.readouterr().out
        assert "no --model was given" in out
        assert "measure the weights rather than the backends" in out


class TestSplitFlagsAreDispatchedCorrectly:
    """The split flags are read only inside the trace path, so reaching
    execute() without --trace dropped them without a word."""

    def test_compare_records_needs_no_backend_pair(self, tmp_path, saved_model):
        """The documented command. Both backend names come from the files, so
        requiring --compare made it impossible to run."""
        import argparse

        a = tmp_path / "a.pt"
        b = tmp_path / "b.pt"
        ValidateCommand._run_trace(_args(record=str(a), model=saved_model))
        ValidateCommand._run_trace(
            _args(replay=str(a), record=str(b), model=saved_model)
        )

        rc = ValidateCommand.execute(
            argparse.Namespace(
                compare=None,
                trace=False,
                compare_records=[str(a), str(b)],
                ci=False,
                verbose=False,
                model_family=None,
                output=None,
                trace_output=None,
                level="standard",
            )
        )
        assert rc == 0

    @pytest.mark.parametrize("flag", ["record", "replay"])
    def test_a_split_flag_without_trace_is_refused(self, flag, tmp_path, capsys):
        import argparse

        rc = ValidateCommand.execute(
            argparse.Namespace(
                **{
                    "compare": None,
                    "trace": False,
                    "compare_records": None,
                    flag: str(tmp_path / "x.pt"),
                    "ci": False,
                    "verbose": False,
                    "level": "standard",
                }
            )
        )
        assert rc == 1
        assert "split trace" in capsys.readouterr().out


class TestSplitDispatchDoesNotFireOnNonValues:
    """The split flags are shape-checked, not truth-tested.

    `execute()` is called across the suite with a MagicMock as args, where
    every attribute is a truthy Mock. A bare `if getattr(args, ...)` therefore
    fired on all of them and sent ordinary validation runs into the offline
    record comparison. Eight pre-existing CLI tests broke that way, and only
    the full CI command surfaced it — `tests/unit` alone does not reach them.
    """

    @staticmethod
    def _plain_args(**kw):
        import argparse

        d = {
            "level": "quick",
            "model": None,
            "output": None,
            "ci": True,
            "verbose": False,
            "compare": None,
            "trace": False,
            "compare_records": None,
            "record": None,
            "replay": None,
        }
        d.update(kw)
        return argparse.Namespace(**d)

    def test_a_mock_args_object_does_not_reach_compare_records(self):
        from unittest.mock import MagicMock

        from torchbridge.cli.validate import ValidateCommand

        args = MagicMock()
        args.level = "quick"
        args.model = None
        args.output = None
        args.ci = True
        args.verbose = False

        # Must not raise: the old truth test unpacked the Mock as a file pair.
        assert ValidateCommand.execute(args) in (0, 1, 2)

    def test_a_single_path_is_not_treated_as_a_record_pair(self):
        """--compare-records is nargs=2. One value is not a pair."""
        from torchbridge.cli.validate import ValidateCommand

        rc = ValidateCommand.execute(self._plain_args(compare_records=["only_one.pt"]))
        assert rc in (0, 1, 2)

    def test_a_real_pair_still_dispatches(self, tmp_path, saved_model):
        """The behaviour the shape check must not cost us."""
        from torchbridge.cli.validate import ValidateCommand

        a, b = tmp_path / "a.pt", tmp_path / "b.pt"
        ValidateCommand._run_trace(_args(record=str(a), model=saved_model))
        ValidateCommand._run_trace(
            _args(replay=str(a), record=str(b), model=saved_model)
        )

        assert (
            ValidateCommand.execute(
                self._plain_args(compare_records=[str(a), str(b)], ci=True)
            )
            == 0
        )

    def test_a_non_string_record_flag_is_not_refused(self):
        """The refusal is for a real path the user typed, not for a stand-in."""
        from unittest.mock import MagicMock

        from torchbridge.cli.validate import ValidateCommand

        args = MagicMock()
        args.level = "quick"
        args.model = None
        args.output = None
        args.ci = True
        args.verbose = False
        args.compare = None
        args.trace = False
        args.compare_records = None

        assert ValidateCommand.execute(args) in (0, 1, 2)

    def test_a_real_record_path_without_trace_is_still_refused(self, tmp_path):
        """The guard itself must keep working."""
        from torchbridge.cli.validate import ValidateCommand

        rc = ValidateCommand.execute(self._plain_args(record=str(tmp_path / "x.pt")))
        assert rc == 1


class TestStrictEnv:
    """--strict-env turns the environment warning into a refusal.

    Without it a split-trace comparison whose halves carry *provably* different
    weights still reports a verdict: the fingerprint mismatch is logged and the
    run continues. That is defensible as a default — the fingerprint samples
    each parameter's ends, so silence is not proof of a match, and a project
    may well want a number out of a slightly mismatched pair.

    It is not defensible for a figure that goes in a paper. A mistyped --model
    on the second machine produces exactly this state, and the result measures
    the two random initialisations rather than the two backends. The flag is
    what lets the operator of a cross-vendor run say "refuse instead".
    """

    @staticmethod
    def _mismatched_pair(tmp_path):
        """A record/replay pair whose halves ran different weights.

        Built through the real record/replay path rather than by hand, so the
        roles, backends and step counts are whatever the CLI actually writes —
        only the fingerprints differ, which is the condition under test.
        """
        from torchbridge.cli.validate import ValidateCommand

        lead = tmp_path / "lead.rec"
        follow = tmp_path / "follow.rec"
        # No --model, so each invocation initialises its own smoke model: the
        # weights differ, and the fingerprints record that they differ.
        assert ValidateCommand.execute(_args(record=str(lead))) == 0
        # The replay's own exit code is not the subject here and is expected to
        # be 1: two different models diverge far past any tolerance. What this
        # helper needs is the pair of files.
        ValidateCommand.execute(_args(replay=str(lead), record=str(follow)))
        assert lead.exists() and follow.exists()
        return lead, follow

    def test_mismatch_is_only_a_warning_by_default(self, tmp_path, caplog):
        lead, follow = self._mismatched_pair(tmp_path)
        rc = ValidateCommand.execute(
            _args(compare=None, trace=False, compare_records=[str(lead), str(follow)])
        )
        # A verdict is still produced — that is the behaviour the flag exists
        # to override, so it has to be asserted, not assumed.
        assert rc in (0, 1)
        assert any(
            "different environments" in r.message for r in caplog.records
        ), "the mismatch must at least be reported"

    def test_strict_env_refuses_the_same_pair(self, tmp_path, capsys):
        lead, follow = self._mismatched_pair(tmp_path)
        rc = ValidateCommand.execute(
            _args(
                compare=None,
                trace=False,
                compare_records=[str(lead), str(follow)],
                strict_env=True,
            )
        )
        assert rc == 1
        out = capsys.readouterr().out
        assert "different environments" in out
        assert "model_fingerprint" in out

    def test_strict_env_does_not_refuse_a_matched_pair(self, tmp_path, saved_model):
        """The flag must reject mismatches, not everything.

        A test that only checked the refusal would pass just as well if
        --strict-env made every comparison fail.
        """
        from torchbridge.cli.validate import ValidateCommand

        lead = tmp_path / "lead.rec"
        follow = tmp_path / "follow.rec"
        # Same --model on both halves: identical weights, identical fingerprint.
        assert (
            ValidateCommand.execute(_args(model=saved_model, record=str(lead))) == 0
        )
        ValidateCommand.execute(
            _args(model=saved_model, replay=str(lead), record=str(follow))
        )
        assert lead.exists() and follow.exists()
        rc = ValidateCommand.execute(
            _args(
                compare=None,
                trace=False,
                compare_records=[str(lead), str(follow)],
                strict_env=True,
            )
        )
        assert rc == 0
