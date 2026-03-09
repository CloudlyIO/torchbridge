"""
Unit tests for torchbridge.testing.compliance_cert.

Covers:
  - ComplianceCertificate field presence
  - status "PASSED" / "FAILED" derivation
  - timestamp is ISO 8601
  - torchbridge_version is a non-empty string
  - Fingerprint is a 64-char hex string (SHA256)
  - Fingerprint is deterministic (same inputs → same fingerprint)
  - Changing any key field changes the fingerprint
  - JSON serialisation round-trips correctly
  - Edge cases: max_diff=0.0, very small floats, model_id with path separators
  - Package-level import from torchbridge.testing
"""

import json
import re

import pytest

from torchbridge.testing.compliance_cert import (
    _compute_fingerprint,
    generate_certificate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cert(**kwargs):
    defaults = {
        "model_id": "Qwen/Qwen3-0.6B",
        "backend_a": "cuda",
        "backend_b": "rocm",
        "max_diff": 2.1e-6,
        "cosine_sim": 1.0,
        "tolerance_atol": 1e-3,
        "passed": True,
    }
    defaults.update(kwargs)
    return generate_certificate(**defaults)


# ---------------------------------------------------------------------------
# Field presence
# ---------------------------------------------------------------------------

class TestCertificateFields:
    def test_cert_has_model_id(self):
        c = _cert()
        assert c.model_id == "Qwen/Qwen3-0.6B"

    def test_cert_has_backend_a_lowercase(self):
        c = _cert(backend_a="CUDA")
        assert c.backend_a == "cuda"

    def test_cert_has_backend_b_lowercase(self):
        c = _cert(backend_b="ROCm")
        assert c.backend_b == "rocm"

    def test_cert_has_timestamp(self):
        c = _cert()
        assert isinstance(c.timestamp, str)
        assert len(c.timestamp) > 0

    def test_cert_has_max_diff(self):
        c = _cert(max_diff=1.5e-5)
        assert c.max_diff == 1.5e-5

    def test_cert_has_cosine_sim(self):
        c = _cert(cosine_sim=0.999)
        assert c.cosine_sim == 0.999

    def test_cert_has_tolerance_atol(self):
        c = _cert(tolerance_atol=1e-4)
        assert c.tolerance_atol == 1e-4

    def test_cert_has_torchbridge_version(self):
        c = _cert()
        assert isinstance(c.torchbridge_version, str)
        assert len(c.torchbridge_version) > 0

    def test_cert_has_fingerprint(self):
        c = _cert()
        assert isinstance(c.fingerprint, str)
        assert len(c.fingerprint) > 0


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------

class TestStatusDerivation:
    def test_status_passed_for_true(self):
        c = _cert(passed=True)
        assert c.status == "PASSED"

    def test_status_failed_for_false(self):
        c = _cert(passed=False)
        assert c.status == "FAILED"


# ---------------------------------------------------------------------------
# Timestamp format
# ---------------------------------------------------------------------------

class TestTimestamp:
    def test_timestamp_is_iso8601(self):
        c = _cert()
        # ISO 8601 UTC includes 'T' separator and '+00:00' or 'Z'
        assert "T" in c.timestamp
        assert "+" in c.timestamp or c.timestamp.endswith("Z")

    def test_timestamp_is_string(self):
        c = _cert()
        assert isinstance(c.timestamp, str)


# ---------------------------------------------------------------------------
# Fingerprint correctness
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_fingerprint_is_64_char_hex(self):
        c = _cert()
        assert re.fullmatch(r"[0-9a-f]{64}", c.fingerprint)

    def test_same_inputs_same_fingerprint(self):
        fp1 = _compute_fingerprint("m", "cuda", "rocm", 1e-6, 1e-3, "PASSED")
        fp2 = _compute_fingerprint("m", "cuda", "rocm", 1e-6, 1e-3, "PASSED")
        assert fp1 == fp2

    def test_different_status_different_fingerprint(self):
        fp_pass = _compute_fingerprint("m", "cuda", "rocm", 1e-6, 1e-3, "PASSED")
        fp_fail = _compute_fingerprint("m", "cuda", "rocm", 1e-6, 1e-3, "FAILED")
        assert fp_pass != fp_fail

    def test_different_max_diff_different_fingerprint(self):
        fp1 = _compute_fingerprint("m", "cuda", "rocm", 1e-6, 1e-3, "PASSED")
        fp2 = _compute_fingerprint("m", "cuda", "rocm", 2e-6, 1e-3, "PASSED")
        assert fp1 != fp2

    def test_different_model_different_fingerprint(self):
        fp1 = _compute_fingerprint("model-a", "cuda", "rocm", 1e-6, 1e-3, "PASSED")
        fp2 = _compute_fingerprint("model-b", "cuda", "rocm", 1e-6, 1e-3, "PASSED")
        assert fp1 != fp2

    def test_different_backends_different_fingerprint(self):
        fp1 = _compute_fingerprint("m", "cuda", "rocm", 1e-6, 1e-3, "PASSED")
        fp2 = _compute_fingerprint("m", "cuda", "cpu", 1e-6, 1e-3, "PASSED")
        assert fp1 != fp2

    def test_cert_fingerprint_matches_compute(self):
        c = _cert(model_id="test-model", backend_a="cuda", backend_b="rocm",
                  max_diff=2.1e-6, tolerance_atol=1e-3, passed=True)
        expected = _compute_fingerprint(
            "test-model", "cuda", "rocm", 2.1e-6, 1e-3, "PASSED"
        )
        assert c.fingerprint == expected


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_to_dict_is_json_serialisable(self):
        c = _cert()
        serialised = json.dumps(c.to_dict())
        assert len(serialised) > 0

    def test_to_json_parses_correctly(self):
        c = _cert()
        parsed = json.loads(c.to_json())
        assert parsed["status"] == "PASSED"
        assert "fingerprint" in parsed

    def test_to_dict_has_all_required_fields(self):
        c = _cert()
        d = c.to_dict()
        for field in ("model_id", "backend_a", "backend_b", "timestamp",
                      "max_diff", "cosine_sim", "tolerance_atol",
                      "status", "torchbridge_version", "fingerprint"):
            assert field in d, f"Missing field: {field}"

    def test_to_json_fingerprint_survives_roundtrip(self):
        c = _cert()
        parsed = json.loads(c.to_json())
        assert parsed["fingerprint"] == c.fingerprint


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_max_diff_zero_is_valid(self):
        c = _cert(max_diff=0.0, passed=True)
        assert c.max_diff == 0.0
        assert c.status == "PASSED"

    def test_very_small_max_diff_preserved(self):
        c = _cert(max_diff=1e-12)
        assert c.max_diff == 1e-12

    def test_model_id_with_path_separators(self):
        c = _cert(model_id="/path/to/my/model.pt")
        assert c.model_id == "/path/to/my/model.pt"

    def test_failed_cert_json_serialisable(self):
        c = _cert(max_diff=0.5, passed=False)
        serialised = json.dumps(c.to_dict())
        assert "FAILED" in serialised


# ---------------------------------------------------------------------------
# Input validation (non-finite floats)
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_inf_max_diff_raises(self):
        with pytest.raises(ValueError, match="max_diff"):
            _cert(max_diff=float("inf"))

    def test_nan_max_diff_raises(self):
        with pytest.raises(ValueError, match="max_diff"):
            _cert(max_diff=float("nan"))

    def test_inf_cosine_sim_raises(self):
        with pytest.raises(ValueError, match="cosine_sim"):
            _cert(cosine_sim=float("inf"))

    def test_nan_cosine_sim_raises(self):
        with pytest.raises(ValueError, match="cosine_sim"):
            _cert(cosine_sim=float("nan"))

    def test_inf_tolerance_atol_raises(self):
        with pytest.raises(ValueError, match="tolerance_atol"):
            _cert(tolerance_atol=float("inf"))

    def test_nan_tolerance_atol_raises(self):
        with pytest.raises(ValueError, match="tolerance_atol"):
            _cert(tolerance_atol=float("nan"))

    def test_negative_inf_max_diff_raises(self):
        with pytest.raises(ValueError, match="max_diff"):
            _cert(max_diff=float("-inf"))


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_importable_from_torchbridge_testing(self):
        from torchbridge.testing import ComplianceCertificate, generate_certificate
        assert ComplianceCertificate is not None
        assert generate_certificate is not None
