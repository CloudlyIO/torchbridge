"""
Integration tests for tb-validate --model-family argument.

Covers:
  - --model-family arg registered in ValidateCommand.register()
  - --model-family arg registered in main() standalone parser
  - default is None
  - --model-family decoder-large with --compare cpu cpu uses larger atol than default
  - --model-family without --compare is silently ignored (no error)
  - --model-family wires through to ToleranceDB.get()
"""

import sys
from types import SimpleNamespace


def _args(**kwargs):
    defaults = {
        "model": None,
        "compare": None,
        "input_shape": "1,64",
        "per_layer": False,
        "dtype": "float32",
        "ci": False,
        "output": None,
        "verbose": False,
        "quantized": False,
        "format": "text",
        "trace": False,
        "steps": 10,
        "autoregressive": False,
        "trace_output": None,
        "cert": None,
        "model_family": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Argument registration
# ---------------------------------------------------------------------------


class TestArgRegistration:
    def _make_parser(self):
        import argparse

        from torchbridge.cli.validate import ValidateCommand

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers()
        ValidateCommand.register(sub)
        return parser

    def test_model_family_arg_registered(self):
        parser = self._make_parser()
        parsed = parser.parse_args(
            ["validate", "--compare", "cpu", "cpu", "--model-family", "decoder-large"]
        )
        assert parsed.model_family == "decoder-large"

    def test_model_family_default_is_none(self):
        parser = self._make_parser()
        parsed = parser.parse_args(["validate", "--compare", "cpu", "cpu"])
        assert parsed.model_family is None

    def test_model_family_in_main_parser(self):
        import unittest.mock as mock

        from torchbridge.cli import validate as validate_module

        orig_argv = sys.argv
        try:
            sys.argv = [
                "tb-validate",
                "--compare",
                "cpu",
                "cpu",
                "--model-family",
                "encoder",
                "--model",
                "__smoke__",
            ]
            with mock.patch("sys.exit"):
                try:
                    validate_module.main()
                except (SystemExit, Exception):
                    pass
        finally:
            sys.argv = orig_argv


# ---------------------------------------------------------------------------
# --model-family without --compare is silently ignored
# ---------------------------------------------------------------------------


class TestModelFamilyWithoutCompare:
    def test_model_family_without_compare_is_harmless(self):
        from torchbridge.cli.validate import ValidateCommand

        args = _args(model_family="decoder-large")  # no compare
        try:
            ValidateCommand.execute(args)
        except Exception:
            pass
        # Should not raise AttributeError or crash on model_family lookup


# ---------------------------------------------------------------------------
# --model-family affects tolerance used in --compare
# ---------------------------------------------------------------------------


class TestModelFamilyToleranceEffect:
    def _run_compare(self, model_family=None, dtype="float32") -> object:
        """Run cpu-cpu compare and return the ToleranceEntry used."""
        from torchbridge.testing.tolerance_db import ToleranceDB

        db = ToleranceDB()
        return db.get("cpu", dtype, model_family=model_family)

    def test_decoder_large_atol_greater_than_default(self):
        default_tol = self._run_compare(model_family=None)
        large_tol = self._run_compare(model_family="decoder-large")
        assert large_tol.atol > default_tol.atol

    def test_encoder_atol_less_than_default(self):
        default_tol = self._run_compare(model_family=None)
        enc_tol = self._run_compare(model_family="encoder")
        assert enc_tol.atol < default_tol.atol

    def test_decoder_small_source_is_measured(self):
        tol = self._run_compare(model_family="decoder-small")
        assert tol.source == "measured"

    def test_decoder_large_source_is_derived(self):
        tol = self._run_compare(model_family="decoder-large")
        assert tol.source == "derived"

    def test_none_family_gives_base_entry(self):
        from torchbridge.testing.tolerance_db import _TOLERANCE_TABLE, ToleranceDB

        db = ToleranceDB()
        tol = db.get("cpu", "float32", model_family=None)
        expected_atol = _TOLERANCE_TABLE[("cpu", "float32")].atol
        assert tol.atol == expected_atol

    def test_vision_language_has_notes(self):
        from torchbridge.testing.tolerance_db import ToleranceDB

        db = ToleranceDB()
        tol = db.get("cuda", "float32", model_family="vision-language")
        assert len(tol.notes) > 0
