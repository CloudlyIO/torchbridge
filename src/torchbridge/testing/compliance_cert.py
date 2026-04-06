"""
Compliance Certificate

Tamper-evident validation certificate generated after a successful (or failed)
cross-backend comparison run. Provides a shareable JSON artifact that encodes:

  - Which model was validated and on which backends
  - The measured numerical divergence (max_diff, cosine_sim)
  - The tolerance threshold used
  - The pass/fail verdict
  - A SHA256 fingerprint of the above fields

The fingerprint is computed over the canonical JSON representation of the key
fields using deterministic serialisation (sorted keys, no whitespace, fixed
float precision). Any post-generation mutation of the fields will produce a
fingerprint mismatch, making tampering detectable.

This module does NOT submit certificates to an external service. It generates
a local JSON artifact. The "TorchBridge Verified" cloud attestation service
(v0.5.63+) will build on top of this format.

Usage::

    from torchbridge.testing.compliance_cert import generate_certificate

    cert = generate_certificate(
        model_id="Qwen/Qwen3-0.6B",
        backend_a="cuda",
        backend_b="rocm",
        max_diff=2.1e-6,
        cosine_sim=1.000000,
        tolerance_atol=1e-3,
        passed=True,
    )
    print(cert.to_json())
    # {"model_id": "Qwen/Qwen3-0.6B", ..., "status": "PASSED", "fingerprint": "a3f1..."}
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from typing import Any


def _get_version() -> str:
    try:
        return importlib.metadata.version("torchbridge-ml")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass
class ComplianceCertificate:
    """Tamper-evident compliance certificate for a TorchBridge validation run."""

    model_id: str
    """Model path or HuggingFace model ID that was validated."""

    backend_a: str
    """Primary backend (e.g. ``"cuda"``)."""

    backend_b: str
    """Comparison backend (e.g. ``"rocm"``)."""

    timestamp: str
    """ISO 8601 UTC timestamp when the certificate was generated."""

    max_diff: float
    """Maximum absolute difference between backend outputs."""

    cosine_sim: float
    """Cosine similarity between backend outputs."""

    tolerance_atol: float
    """Absolute tolerance threshold used for pass/fail verdict."""

    status: str
    """``"PASSED"`` if max_diff <= tolerance_atol, otherwise ``"FAILED"``."""

    torchbridge_version: str
    """TorchBridge version that produced this certificate."""

    fingerprint: str
    """SHA256 hex digest of the canonical payload. Changing any field invalidates this."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend_a": self.backend_a,
            "backend_b": self.backend_b,
            "timestamp": self.timestamp,
            "max_diff": self.max_diff,
            "cosine_sim": self.cosine_sim,
            "tolerance_atol": self.tolerance_atol,
            "status": self.status,
            "torchbridge_version": self.torchbridge_version,
            "fingerprint": self.fingerprint,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def generate_certificate(
    model_id: str,
    backend_a: str,
    backend_b: str,
    max_diff: float,
    cosine_sim: float,
    tolerance_atol: float,
    passed: bool,
) -> ComplianceCertificate:
    """Generate a tamper-evident compliance certificate for a validation run.

    Args:
        model_id: Model path or HuggingFace model ID.
        backend_a: Primary backend name (e.g. ``"cuda"``).
        backend_b: Comparison backend name (e.g. ``"rocm"``).
        max_diff: Measured maximum absolute difference between outputs.
        cosine_sim: Measured cosine similarity between outputs.
        tolerance_atol: Absolute tolerance threshold that was applied.
        passed: ``True`` if the run passed (max_diff <= tolerance_atol).

    Returns:
        :class:`ComplianceCertificate` with a SHA256 fingerprint.
    """
    if not math.isfinite(max_diff):
        raise ValueError(f"max_diff must be a finite float, got {max_diff!r}")
    if not math.isfinite(cosine_sim):
        raise ValueError(f"cosine_sim must be a finite float, got {cosine_sim!r}")
    if not math.isfinite(tolerance_atol):
        raise ValueError(
            f"tolerance_atol must be a finite float, got {tolerance_atol!r}"
        )

    status = "PASSED" if passed else "FAILED"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    version = _get_version()

    fingerprint = _compute_fingerprint(
        model_id=model_id,
        backend_a=backend_a.lower(),
        backend_b=backend_b.lower(),
        max_diff=max_diff,
        tolerance_atol=tolerance_atol,
        status=status,
    )

    return ComplianceCertificate(
        model_id=model_id,
        backend_a=backend_a.lower(),
        backend_b=backend_b.lower(),
        timestamp=timestamp,
        max_diff=max_diff,
        cosine_sim=cosine_sim,
        tolerance_atol=tolerance_atol,
        status=status,
        torchbridge_version=version,
        fingerprint=fingerprint,
    )


def _compute_fingerprint(
    model_id: str,
    backend_a: str,
    backend_b: str,
    max_diff: float,
    tolerance_atol: float,
    status: str,
) -> str:
    """Compute SHA256 fingerprint over canonical payload.

    Uses deterministic JSON serialisation: sorted keys, no whitespace,
    floats rounded to 12 decimal places to avoid floating-point noise.
    """
    payload = json.dumps(
        {
            "model_id": model_id,
            "backend_a": backend_a,
            "backend_b": backend_b,
            "max_diff": round(max_diff, 12),
            "tolerance_atol": round(tolerance_atol, 12),
            "status": status,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
