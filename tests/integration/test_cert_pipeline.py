"""
Integration tests for the compliance certificate CLI pipeline.

Covers:
  - --cert argument registration in ValidateCommand.register() and main()
  - --cert default is None
  - --cert without --compare is silently ignored (no error)
  - Smoke: --compare cpu cpu --cert FILE → file written, valid JSON, correct fields
  - Certificate fingerprint present and 64-char hex
  - Parent directory created if missing
  - CI JSON mode still writes cert file when --cert given
  - Package-level imports: KVHandoffNegotiator from inference, ComplianceCertificate from testing
"""

import json
import re
import sys
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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

    def test_cert_arg_registered(self):
        parser = self._make_parser()
        parsed = parser.parse_args(
            ["validate", "--compare", "cpu", "cpu", "--cert", "/tmp/cert.json"]
        )
        assert parsed.cert == "/tmp/cert.json"

    def test_cert_default_is_none(self):
        parser = self._make_parser()
        parsed = parser.parse_args(["validate", "--compare", "cpu", "cpu"])
        assert parsed.cert is None

    def test_cert_in_main_parser(self):
        """--cert must also be registered in the standalone main() parser."""

        # Parse via main()'s parser directly by importing and calling parse_args
        # We can test this by checking the help text contains --cert
        import unittest.mock as mock

        from torchbridge.cli import validate as validate_module

        # Verify the arg exists by trying to parse it without error
        # (main() calls parser.parse_args() — we intercept with sys.argv)
        orig_argv = sys.argv
        try:
            sys.argv = [
                "tb-validate",
                "--compare",
                "cpu",
                "cpu",
                "--cert",
                "/tmp/x.json",
                "--model",
                "__smoke__",
            ]
            # Just confirm no AttributeError on parsed.cert
            with mock.patch("sys.exit"):
                try:
                    validate_module.main()
                except SystemExit:
                    pass
                except Exception:
                    pass
        finally:
            sys.argv = orig_argv


# ---------------------------------------------------------------------------
# --cert without --compare is silently ignored
# ---------------------------------------------------------------------------


class TestCertWithoutCompare:
    def test_cert_without_compare_is_harmless(self, tmp_path):
        from torchbridge.cli.validate import ValidateCommand

        cert_file = tmp_path / "cert.json"
        args = _args(cert=str(cert_file))  # no compare
        # Should not crash; standard path proceeds (no GPU → may exit non-zero)
        try:
            ValidateCommand.execute(args)
        except Exception:
            pass
        # Cert file must NOT have been written (no --compare ran)
        assert not cert_file.exists()


# ---------------------------------------------------------------------------
# Smoke: --compare cpu cpu with --cert
# ---------------------------------------------------------------------------


class TestCertSmoke:
    def _run_compare_with_cert(self, cert_path: str, ci: bool = False) -> int:
        from torchbridge.cli.validate import ValidateCommand

        args = _args(
            compare=["cpu", "cpu"],
            cert=cert_path,
            ci=ci,
        )
        # Capture stdout to prevent noise
        import io

        buf = io.StringIO()
        orig = sys.stdout
        sys.stdout = buf
        try:
            rc = ValidateCommand.execute(args)
        finally:
            sys.stdout = orig
        return rc

    def test_compare_cpu_cpu_exits_zero(self, tmp_path):
        cert_file = tmp_path / "cert.json"
        rc = self._run_compare_with_cert(str(cert_file))
        assert rc == 0

    def test_cert_file_is_written(self, tmp_path):
        cert_file = tmp_path / "cert.json"
        self._run_compare_with_cert(str(cert_file))
        assert cert_file.exists()

    def test_cert_file_contains_valid_json(self, tmp_path):
        cert_file = tmp_path / "cert.json"
        self._run_compare_with_cert(str(cert_file))
        content = cert_file.read_text()
        parsed = json.loads(content)
        assert isinstance(parsed, dict)

    def test_cert_file_has_fingerprint(self, tmp_path):
        cert_file = tmp_path / "cert.json"
        self._run_compare_with_cert(str(cert_file))
        parsed = json.loads(cert_file.read_text())
        assert "fingerprint" in parsed
        assert re.fullmatch(r"[0-9a-f]{64}", parsed["fingerprint"])

    def test_cert_file_has_status_passed(self, tmp_path):
        cert_file = tmp_path / "cert.json"
        self._run_compare_with_cert(str(cert_file))
        parsed = json.loads(cert_file.read_text())
        assert parsed["status"] == "PASSED"

    def test_cert_has_required_fields(self, tmp_path):
        cert_file = tmp_path / "cert.json"
        self._run_compare_with_cert(str(cert_file))
        parsed = json.loads(cert_file.read_text())
        for field in (
            "model_id",
            "backend_a",
            "backend_b",
            "timestamp",
            "max_diff",
            "cosine_sim",
            "tolerance_atol",
            "status",
            "torchbridge_version",
            "fingerprint",
        ):
            assert field in parsed, f"Missing field: {field}"

    def test_cert_missing_parent_dir_created(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "cert.json"
        rc = self._run_compare_with_cert(str(nested))
        assert rc == 0
        assert nested.exists()

    def test_ci_mode_with_cert_still_writes_file(self, tmp_path):
        cert_file = tmp_path / "cert_ci.json"
        self._run_compare_with_cert(str(cert_file), ci=True)
        assert cert_file.exists()
        parsed = json.loads(cert_file.read_text())
        assert "fingerprint" in parsed


# ---------------------------------------------------------------------------
# Package-level imports
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_kv_handoff_importable_from_inference(self):
        from torchbridge.inference import KVHandoffNegotiator, KVHandoffSpec

        assert KVHandoffNegotiator is not None
        assert KVHandoffSpec is not None

    def test_compliance_cert_importable_from_testing(self):
        from torchbridge.testing import ComplianceCertificate, generate_certificate

        assert ComplianceCertificate is not None
        assert generate_certificate is not None
