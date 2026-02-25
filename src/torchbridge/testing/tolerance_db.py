"""
Empirical Tolerance Database

Per-(op, backend, dtype) numerical tolerances seeded from cloud validation
results (v0.5.31). Used by CrossBackendTestSuite and DivergenceTracer to
decide pass/fail for cross-backend output comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TolerancePair:
    """Absolute tolerance (atol) and relative tolerance (rtol) for a comparison."""

    atol: float
    rtol: float


# ── Empirical tolerances from cloud validation (v0.5.31 reports) ─────────────
#
# Key: (backend_name: str, dtype_name: str)
# backend_name matches HardwareBackend.value ("cuda", "rocm", "mps", "xla", "cpu")
# dtype_name is the PyTorch dtype string ("float32", "float16", "bfloat16")
#
# These values were derived from the worst-case max-diff observed across 8 platforms
# during real-hardware validation with Qwen3-0.6B.

_TOLERANCE_TABLE: dict[tuple[str, str], TolerancePair] = {
    # CUDA (NVIDIA) — tight tolerances; exact parity expected
    ("cuda", "float32"): TolerancePair(atol=1e-4, rtol=1e-5),
    ("cuda", "float16"): TolerancePair(atol=1e-3, rtol=1e-3),
    ("cuda", "bfloat16"): TolerancePair(atol=1e-2, rtol=1e-3),
    # ROCm (AMD) — slightly looser; SDPA flash attention divergence
    ("rocm", "float32"): TolerancePair(atol=1e-3, rtol=1e-4),
    ("rocm", "float16"): TolerancePair(atol=2e-3, rtol=1e-3),
    ("rocm", "bfloat16"): TolerancePair(atol=2e-2, rtol=1e-3),
    # MPS (Apple Silicon)
    ("mps", "float32"): TolerancePair(atol=1e-4, rtol=1e-5),
    ("mps", "float16"): TolerancePair(atol=1e-3, rtol=1e-3),
    ("mps", "bfloat16"): TolerancePair(atol=1e-2, rtol=1e-3),
    # XLA (TPU) — very loose absolute; XLA reorders FP ops
    ("xla", "float32"): TolerancePair(atol=0.5, rtol=1e-2),
    ("xla", "bfloat16"): TolerancePair(atol=0.5, rtol=1e-2),
    # CPU — reference backend; very tight
    ("cpu", "float32"): TolerancePair(atol=1e-6, rtol=1e-6),
    ("cpu", "float16"): TolerancePair(atol=1e-4, rtol=1e-4),
    ("cpu", "bfloat16"): TolerancePair(atol=1e-3, rtol=1e-4),
    # Trainium / Inferentia2 (PrivateUse1 / CPU-backed)
    ("trainium", "float32"): TolerancePair(atol=1e-4, rtol=1e-5),
    ("trainium", "bfloat16"): TolerancePair(atol=1e-2, rtol=1e-3),
}

# Default fallback if key not found
_DEFAULT_TOLERANCE = TolerancePair(atol=1e-3, rtol=1e-4)


class ToleranceDB:
    """Look up empirical tolerances by backend and dtype.

    Example::

        db = ToleranceDB()
        tol = db.get("cuda", "float16")
        assert tol.atol == 1e-3
    """

    def __init__(
        self, extra: dict[tuple[str, str], TolerancePair] | None = None
    ) -> None:
        self._table: dict[tuple[str, str], TolerancePair] = dict(_TOLERANCE_TABLE)
        if extra:
            self._table.update(extra)

    def get(self, backend: str, dtype: str) -> TolerancePair:
        """Return tolerances for the given backend and dtype.

        Args:
            backend: Backend string (e.g. ``"cuda"``, ``"rocm"``, ``"cpu"``).
            dtype: PyTorch dtype as string (e.g. ``"float32"``, ``"bfloat16"``).

        Returns:
            :class:`TolerancePair` with ``atol`` and ``rtol``.
        """
        return self._table.get((backend.lower(), dtype.lower()), _DEFAULT_TOLERANCE)

    def register(self, backend: str, dtype: str, atol: float, rtol: float) -> None:
        """Register a custom tolerance for (backend, dtype)."""
        self._table[(backend.lower(), dtype.lower())] = TolerancePair(atol=atol, rtol=rtol)

    def all_backends(self) -> list[str]:
        """Return all backend names that have registered tolerances."""
        return sorted({b for b, _ in self._table})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full tolerance table to a JSON-compatible dict."""
        return {
            f"{b}/{d}": {"atol": t.atol, "rtol": t.rtol}
            for (b, d), t in sorted(self._table.items())
        }
