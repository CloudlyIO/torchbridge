"""
Unit tests for model-family resolution in tb-validate.

The tolerance database is keyed on (model_family, backend, dtype), and the trace
path was sending only two of the three. That silently applied the strictest
limit to every model. These tests pin the three ways that must not happen again:
the family reaches the lookup, a mistyped family is refused rather than absorbed,
and an omitted one is derived from the model rather than left to memory.

Nothing here needs a GPU.
"""

from __future__ import annotations

import pytest
import torch


class TestUnknownModelFamilyIsRefused:
    """A mistyped family silently selects the coarse table — the same class of
    bug as the device collision, so it must be refused rather than absorbed.

    ``ToleranceDB.get`` falls back to the ``(backend, dtype)`` row for any
    family it does not know, and returns no signal that it did. On a
    decoder-large run that means a 4x-too-strict atol and a spurious failure.
    """

    @staticmethod
    def _args(family):
        import argparse

        return argparse.Namespace(
            compare=["cpu", "cpu"],
            trace=False,
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
            model_family=family,
            cert=None,
        )

    def test_typo_is_refused(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        rc = ValidateCommand._run_compare(self._args("decoder-larg"))
        out = capsys.readouterr().out.lower()
        assert rc == 1
        assert "decoder-larg" in out

    def test_error_lists_the_valid_families(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        ValidateCommand._run_compare(self._args("nonsense"))
        assert "decoder-large" in capsys.readouterr().out

    def test_a_real_family_is_accepted(self):
        from torchbridge.cli.validate import ValidateCommand

        assert ValidateCommand._run_compare(self._args("decoder-large")) == 0

    def test_no_family_is_still_fine(self):
        from torchbridge.cli.validate import ValidateCommand

        assert ValidateCommand._run_compare(self._args(None)) == 0


class TestModelFamilyIsInferred:
    """Leaving --model-family off must not silently apply the wrong limit.

    ToleranceDB documents its own size boundaries: under 2B parameters is
    decoder-small, 2B to 20B is decoder-medium, above that decoder-large. A
    loaded model carries its own parameter count, so the family can be worked
    out instead of typed. Relying on the operator to remember a flag is how the
    original bug reaches production again.
    """

    @staticmethod
    def _model(n_params):
        """A stand-in reporting a parameter count without allocating it.

        A real 70B model is 280GB of float32, so the count is reported rather
        than built. ``infer_model_family`` only ever sums ``numel()``.
        """

        class _Param:
            def numel(self):
                return n_params

        class _Model:
            def parameters(self):
                yield _Param()

        return _Model()

    def test_small_model_infers_decoder_small(self):
        from torchbridge.cli.validate import infer_model_family

        assert infer_model_family(self._model(600_000_000)) == "decoder-small"

    def test_eight_billion_infers_decoder_medium(self):
        from torchbridge.cli.validate import infer_model_family

        assert infer_model_family(self._model(8_000_000_000)) == "decoder-medium"

    def test_seventy_billion_infers_decoder_large(self):
        from torchbridge.cli.validate import infer_model_family

        assert infer_model_family(self._model(70_000_000_000)) == "decoder-large"

    def test_documented_boundaries(self):
        """2B and 20B are the documented edges; check both sides of each."""
        from torchbridge.cli.validate import infer_model_family

        cases = {
            1_999_999_999: "decoder-small",
            2_000_000_000: "decoder-medium",
            20_000_000_000: "decoder-medium",
            20_000_000_001: "decoder-large",
        }
        for n, expected in cases.items():
            assert infer_model_family(self._model(n)) == expected, n

    def test_inferred_family_is_a_real_database_entry(self):
        """Whatever is inferred must survive validate_model_family."""
        from torchbridge.cli.validate import infer_model_family, validate_model_family

        for n in (600_000_000, 8_000_000_000, 70_000_000_000):
            fam = infer_model_family(self._model(n))
            assert validate_model_family(fam) is None, fam

    def test_returns_none_when_it_cannot_count(self):
        """No parameters means no basis to guess; the caller decides what to do."""
        from torchbridge.cli.validate import infer_model_family

        class _Empty:
            def parameters(self):
                return iter(())

        assert infer_model_family(_Empty()) is None


class TestCliUsesTheInferredFamily:
    """Inference is worthless if the command line does not apply it."""

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
            "ci": False,
            "verbose": False,
            "model_family": None,
            "cert": None,
        }
        d.update(kw)
        return argparse.Namespace(**d)

    def test_trace_records_an_inferred_family_when_none_given(self, tmp_path):
        import json

        from torchbridge.cli.validate import ValidateCommand

        out = tmp_path / "t.json"
        rc = ValidateCommand._run_trace(self._args(trace=True, trace_output=str(out)))
        assert rc == 0
        saved = json.loads(out.read_text())
        assert saved["model_family"] is not None, (
            "no family was recorded, so the run silently used the coarse table"
        )

    def test_an_explicit_family_is_not_overridden(self, tmp_path):
        import json

        from torchbridge.cli.validate import ValidateCommand

        out = tmp_path / "t.json"
        ValidateCommand._run_trace(
            self._args(trace=True, trace_output=str(out), model_family="decoder-large")
        )
        assert json.loads(out.read_text())["model_family"] == "decoder-large"

    def test_the_user_is_told_what_was_inferred(self, capsys):
        from torchbridge.cli.validate import ValidateCommand

        ValidateCommand._run_trace(self._args(trace=True))
        assert "decoder-small" in capsys.readouterr().out


class TestInferenceNoteRespectsCiMode:
    """--ci output must stay parseable; the inference note is human-facing only."""

    @staticmethod
    def _args(ci, trace):
        import argparse

        return argparse.Namespace(
            compare=["cpu", "cpu"],
            trace=trace,
            steps=2,
            autoregressive=False,
            model=None,
            input_shape="1,4",
            per_layer=False,
            dtype="float32",
            output=None,
            trace_output=None,
            ci=ci,
            verbose=False,
            model_family=None,
            cert=None,
        )

    def test_compare_ci_output_is_still_pure_json(self, capsys):
        import json

        from torchbridge.cli.validate import ValidateCommand

        ValidateCommand._run_compare(self._args(ci=True, trace=False))
        json.loads(capsys.readouterr().out.strip())

    def test_trace_ci_output_is_still_pure_json(self, capsys):
        import json

        from torchbridge.cli.validate import ValidateCommand

        ValidateCommand._run_trace(self._args(ci=True, trace=True))
        json.loads(capsys.readouterr().out.strip())


class _Config:
    """Stand-in for a HuggingFace config, carrying only the attributes read."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _ModelWithConfig(torch.nn.Module):
    def __init__(self, config, params: int = 1_000):
        super().__init__()
        self.config = config
        self.w = torch.nn.Parameter(torch.zeros(params))


class TestArchitectureGuard:
    """A parameter count separates decoder sizes and nothing else.

    An encoder of the same size belongs in the ``encoder`` row and a
    vision-language model in ``vision-language``, both tighter than the decoder
    rows; an MoE model's row turns on active rather than total parameters. So a
    count is not evidence for those, and inferring from it would reintroduce the
    wrong-tolerance bug this PR exists to fix.
    """

    def test_encoder_decoder_is_not_inferred(self):
        from torchbridge.cli.validate import infer_model_family, non_decoder_trait

        model = _ModelWithConfig(_Config(is_encoder_decoder=True))
        assert infer_model_family(model) is None
        assert non_decoder_trait(model) == "encoder-decoder"

    def test_vision_language_is_not_inferred(self):
        from torchbridge.cli.validate import infer_model_family, non_decoder_trait

        model = _ModelWithConfig(_Config(vision_config=_Config(hidden_size=8)))
        assert infer_model_family(model) is None
        assert non_decoder_trait(model) == "vision-language"

    @pytest.mark.parametrize(
        "attr", ["num_experts", "num_local_experts", "n_routed_experts"]
    )
    def test_mixture_of_experts_is_not_inferred(self, attr):
        from torchbridge.cli.validate import infer_model_family, non_decoder_trait

        model = _ModelWithConfig(_Config(**{attr: 64}))
        assert infer_model_family(model) is None
        assert non_decoder_trait(model) == "mixture-of-experts"

    def test_dense_decoder_config_still_infers(self):
        from torchbridge.cli.validate import infer_model_family, non_decoder_trait

        model = _ModelWithConfig(_Config(is_encoder_decoder=False, num_experts=0))
        assert non_decoder_trait(model) is None
        assert infer_model_family(model) == "decoder-small"

    def test_bare_module_without_config_still_infers(self):
        """A traced model or a test stand-in carries no config. Refusing those
        would disable inference for every non-HuggingFace model."""
        from torchbridge.cli.validate import infer_model_family, non_decoder_trait

        model = torch.nn.Linear(4, 4)
        assert non_decoder_trait(model) is None
        assert infer_model_family(model) == "decoder-small"

    def test_refusal_is_reported_not_silent(self):
        """The coarse row is still used, but the run has to say so. Silence is
        the original bug in a different disguise."""
        import argparse

        from torchbridge.cli.validate import resolve_family_for_run

        model = _ModelWithConfig(_Config(is_encoder_decoder=True))
        family, note = resolve_family_for_run(
            argparse.Namespace(model_family=None), model
        )
        assert family is None
        assert note is not None
        assert "encoder-decoder" in note
        assert "--model-family" in note

    def test_explicit_family_still_wins_over_the_guard(self):
        """The operator may know what the config cannot show."""
        import argparse

        from torchbridge.cli.validate import resolve_family_for_run

        model = _ModelWithConfig(_Config(num_experts=64))
        family, note = resolve_family_for_run(
            argparse.Namespace(model_family="deepseek_v4"), model
        )
        assert family == "deepseek_v4"
        assert note is None


class TestResultConstructorContract:
    """The new provenance fields must not shift the positional signature.

    ``TraceValidationResult`` is public and was constructible positionally with
    ``step_results`` sixth. Inserting the new fields ahead of it would bind a
    caller's step list to ``model_family`` and produce a result that looks valid.
    """

    def test_sixth_positional_is_still_step_results(self):
        from torchbridge.testing.trace_validator import (
            TraceStepResult,
            TraceValidationResult,
        )

        step = TraceStepResult(
            step=1,
            max_diff=0.0,
            cosine_sim=1.0,
            within_tolerance=True,
            cumulative_amplification=1.0,
        )
        # Exactly how a caller written against the previous release builds one.
        result = TraceValidationResult("cpu", "cuda", 1, "float32", True, [step])
        assert result.step_results == [step]
        assert result.model_family is None
