# SPDX-License-Identifier: Apache-2.0
"""
Empirical Tolerance Database

Per-(model_family, backend, dtype) numerical tolerances seeded from cloud validation
results (v0.5.31) and architectural analysis (v0.5.62).

Three-level fallback chain:
  (model_family, backend, dtype) → (backend, dtype) → _DEFAULT_TOLERANCE

Source labels:
  "measured"  — derived from worst-case max-diff observed on real hardware during
                cloud validation with Qwen3-0.6B (see reports/cloud_validation/).
  "derived"   — scaled from measured entries using the accumulated-error model:
                  decoder-medium: atol × 2  (2× layers → ~2× accumulated FP error)
                  decoder-large:  atol × 4
                  encoder:        atol × 0.5 (bidirectional; no KV cache → tighter)
                  vision-language: atol × 3  (patch embedding adds variance)
                These are conservative upper bounds, not experimentally verified on
                every model. Users with real measurements should call register() to
                override.
  "fallback"  — returned when the requested backend is not in the base tolerance
                table and the global _DEFAULT_TOLERANCE is used instead. Callers
                can check entry.source == "fallback" to detect unsupported backends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

_VALID_SOURCES = frozenset({"measured", "derived", "fallback"})


@dataclass(frozen=True)
class TolerancePair:
    """Absolute tolerance (atol) and relative tolerance (rtol) for a comparison.

    Retained for backward compatibility. New code should use :class:`ToleranceEntry`.
    """

    atol: float
    rtol: float


@dataclass(frozen=True)
class ToleranceEntry:
    """Tolerance pair with provenance metadata.

    Attributes:
        atol: Absolute tolerance threshold.
        rtol: Relative tolerance threshold.
        source: ``"measured"`` if derived from real-hardware validation;
            ``"derived"`` if scaled from a measured baseline.
        notes: Human-readable provenance note.
    """

    atol: float
    rtol: float
    source: str
    notes: str = field(default="")

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(
                f"source must be one of {sorted(_VALID_SOURCES)!r}, got {self.source!r}"
            )


# ---------------------------------------------------------------------------
# Known model families
# ---------------------------------------------------------------------------

MODEL_FAMILIES: tuple[str, ...] = (
    "decoder-small",  # < 2B params  — Qwen3-0.6B, Llama-3.2-1B, SmolLM-2
    "decoder-medium",  # 2B–20B params — Llama-3.1-8B, Qwen3-7B, Mistral-7B
    "decoder-large",  # > 20B params  — Llama-3.1-70B, Qwen3-72B
    "encoder",  # Encoder-only  — BERT, RoBERTa, DeBERTa
    "vision-language",  # Cross-modal   — CLIP, LLaVA, InternVL
    # ── 2026 MoE landscape (v0.5.99) ─────────────────────────────────────────
    "qwen3_5",  # Qwen3.5/3.6 27B dense; > 20B active
    "gemma4",  # Gemma 4 26B-A4B; ~4B active MoE
    "nemotron3_nano",  # Nemotron 3 Nano 30B-A3B; ~3B active Mamba-2+MoE
    "deepseek_v4",  # DeepSeek V4 Flash/Pro; 284B–1.6T total / 13–49B active
    "nemotron3_ultra",  # Nemotron Ultra 550B/55B active MoE
    "tencent_hy3",  # Tencent Hy3 295B/21B active MoE
    "minimax_m3",  # MiniMax M3 ~428B/23B active (MSA attention)
    "glm_5_2",  # GLM 5.2 744B/40B active (IndexShare routing)
)


# ---------------------------------------------------------------------------
# Base tolerance table — (backend, dtype) — measured on Qwen3-0.6B
# ---------------------------------------------------------------------------

_UNVERIFIED_KEYS: frozenset[tuple[str, str]] = frozenset(
    {
        ("xla", "float32"),
        ("xla", "bfloat16"),
    }
)
"""Table entries whose measurement cannot be substantiated.

``get()`` labels anything in ``_TOLERANCE_TABLE`` as ``"measured"`` purely by
virtue of being there. For ``xla`` that label does not hold up:

* The CHANGELOG for v0.5.31 (2026-02-21) claims a GCP TPU v5e run and points at
  ``reports/cloud_validation/2026-02-21/``. Those files are not in the
  repository and never were — no such path appears anywhere in its history.
* ``docs/reference/cloud-validation.md`` records TPU as ``PENDING`` with
  "quota exhausted", and its validation history says "AMD + TPU SKIPPED".
* ``docs/reference/hardware-matrix.md`` has a measured row for A10G and for
  Trainium, and none for TPU or XLA.

So the project contradicts itself and the artifact that would settle it is
missing. Reporting ``"fallback"`` makes the trace validator warn, which is the
correct signal for a number nobody can trace to a run — 0.5 on logits is loose
enough that almost anything passes, so a TPU result judged by it would carry no
information.

The values are left untouched. Replacing them with invented ones would be the
same mistake in the other direction; they should be re-measured on real TPU
hardware and registered with ``register()``.
"""

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
    # Trainium2 (NeuronCore v3) — same tolerance tier as Trainium1; measured when hardware available
    ("trainium2", "float32"): TolerancePair(atol=1e-4, rtol=1e-5),
    ("trainium2", "bfloat16"): TolerancePair(atol=1e-2, rtol=1e-3),
}

# Default fallback if key not found
_DEFAULT_TOLERANCE = TolerancePair(atol=1e-3, rtol=1e-4)


# ---------------------------------------------------------------------------
# Family tolerance table — (model_family, backend, dtype)
# ---------------------------------------------------------------------------
#
# decoder-small entries are "measured" (Qwen3-0.6B cloud validation, v0.5.31).
# All other families are "derived" using the accumulated-error model documented
# in the module docstring.
#
# XLA does not expose a float16 dtype — those entries are intentionally absent.
# The lookup falls back to (backend, dtype) → _TOLERANCE_TABLE in that case.


def _m(atol: float, rtol: float, notes: str = "") -> ToleranceEntry:
    return ToleranceEntry(atol=atol, rtol=rtol, source="measured", notes=notes)


def _d(atol: float, rtol: float, notes: str = "") -> ToleranceEntry:
    return ToleranceEntry(atol=atol, rtol=rtol, source="derived", notes=notes)


_CLOUD_NOTE = "measured on Qwen3-0.6B; cloud validation v0.5.31"
_DM_NOTE = "derived: decoder-small atol × 2 (2× more layers)"
_DL_NOTE = "derived: decoder-small atol × 4 (4× more layers)"
_ENC_NOTE = "derived: decoder-small atol × 0.5 (bidirectional; no KV cache)"
_VL_NOTE = "derived: decoder-small atol × 3 (patch embedding variance)"
# ── new hardware generation note constants (v0.5.69) ──────────────────────
_BW_NOTE = (
    "derived: Hopper (cuda) atol × 1.0 (same fp32 accumulation path, wider GEMM tiles)"
)
_BWC_NOTE = "derived: Hopper/Ada (cuda) atol × 1.0 (Blackwell consumer; same accumulation path as Blackwell DC)"
_CDNA4_NOTE = "derived: CDNA3 (rocm) atol × 1.0 (MI350X/CDNA4; no accumulation-order change vs CDNA3)"
_XLA_V7_NOTE = "derived: TPU v5e (xla) atol × 1.0 (TPU v7 Ironwood; same XLA bf16 accumulation as v5e)"
_TRN3_NOTE = "derived: Trn1 (trainium) atol × 1.0 (Trn3/Neuron; same NeuronCore accumulation semantics as Trn1)"
# ── 2026 silicon coverage (v0.5.100) ──────────────────────────────────────
_RDNA4_NOTE = "derived: RDNA4 (gfx1201, RX 9000 series); consumer GPU; same non-BLAS accumulation as RDNA3"
_TRN2_NOTE = "derived: Trainium2 (TRN2, NeuronCore v3); same XLA accumulation as Trn1; measured pending Trn2 access"
# ── 2026 MoE model families (v0.5.99) ─────────────────────────────────────
_QWEN35_NOTE = (
    "derived: Qwen3.5 27B dense > 20B params; same accumulation depth as decoder-large"
)
_GEMMA4_NOTE = "derived: Gemma 4 26B-A4B (4B active); sparse MoE activation; same as decoder-medium"
_NM3NANO_NOTE = "derived: Nemotron 3 Nano 30B-A3B (3B active); Mamba-2+Transformer MoE; same as decoder-medium"
_DSV4_NOTE = "derived: DeepSeek V4 Flash/Pro (13–49B active); MoE expert routing; decoder-large × 1.5"
_NM3ULTRA_NOTE = "derived: Nemotron Ultra 550B/55B active; large MoE; decoder-large × 2"
_HY3_NOTE = "derived: Tencent Hy3 295B/21B active; MoE; decoder-large × 1.5"
_MMM3_NOTE = "derived: MiniMax M3 ~428B/23B active; MSA attention; decoder-large × 1.5"
_GLM52_NOTE = "derived: GLM 5.2 744B/40B active; IndexShare routing; decoder-large × 2"

_FAMILY_TOLERANCE_TABLE: dict[tuple[str, str, str], ToleranceEntry] = {
    # ── decoder-small ─────────────────────────────────────────────────────
    ("decoder-small", "cuda", "float32"): _m(1e-4, 1e-5, _CLOUD_NOTE),
    ("decoder-small", "cuda", "float16"): _m(1e-3, 1e-3, _CLOUD_NOTE),
    ("decoder-small", "cuda", "bfloat16"): _m(1e-2, 1e-3, _CLOUD_NOTE),
    ("decoder-small", "rocm", "float32"): _m(1e-3, 1e-4, _CLOUD_NOTE),
    ("decoder-small", "rocm", "float16"): _m(2e-3, 1e-3, _CLOUD_NOTE),
    ("decoder-small", "rocm", "bfloat16"): _m(2e-2, 1e-3, _CLOUD_NOTE),
    ("decoder-small", "mps", "float32"): _m(1e-4, 1e-5, _CLOUD_NOTE),
    ("decoder-small", "mps", "float16"): _m(1e-3, 1e-3, _CLOUD_NOTE),
    ("decoder-small", "mps", "bfloat16"): _m(1e-2, 1e-3, _CLOUD_NOTE),
    ("decoder-small", "xla", "float32"): _m(0.5, 1e-2, _CLOUD_NOTE),
    ("decoder-small", "xla", "bfloat16"): _m(0.5, 1e-2, _CLOUD_NOTE),
    ("decoder-small", "cpu", "float32"): _m(1e-6, 1e-6, _CLOUD_NOTE),
    ("decoder-small", "cpu", "float16"): _m(1e-4, 1e-4, _CLOUD_NOTE),
    ("decoder-small", "cpu", "bfloat16"): _m(1e-3, 1e-4, _CLOUD_NOTE),
    # ── decoder-medium ────────────────────────────────────────────────────
    ("decoder-medium", "cuda", "float32"): _d(2e-4, 1e-5, _DM_NOTE),
    ("decoder-medium", "cuda", "float16"): _d(2e-3, 1e-3, _DM_NOTE),
    ("decoder-medium", "cuda", "bfloat16"): _d(2e-2, 1e-3, _DM_NOTE),
    ("decoder-medium", "rocm", "float32"): _d(2e-3, 1e-4, _DM_NOTE),
    ("decoder-medium", "rocm", "float16"): _d(4e-3, 1e-3, _DM_NOTE),
    ("decoder-medium", "rocm", "bfloat16"): _d(4e-2, 1e-3, _DM_NOTE),
    ("decoder-medium", "mps", "float32"): _d(2e-4, 1e-5, _DM_NOTE),
    ("decoder-medium", "mps", "float16"): _d(2e-3, 1e-3, _DM_NOTE),
    ("decoder-medium", "mps", "bfloat16"): _d(2e-2, 1e-3, _DM_NOTE),
    ("decoder-medium", "xla", "float32"): _d(1.0, 1e-2, _DM_NOTE),
    ("decoder-medium", "xla", "bfloat16"): _d(1.0, 1e-2, _DM_NOTE),
    ("decoder-medium", "cpu", "float32"): _d(2e-6, 1e-6, _DM_NOTE),
    ("decoder-medium", "cpu", "float16"): _d(2e-4, 1e-4, _DM_NOTE),
    ("decoder-medium", "cpu", "bfloat16"): _d(2e-3, 1e-4, _DM_NOTE),
    # ── decoder-large ─────────────────────────────────────────────────────
    ("decoder-large", "cuda", "float32"): _d(4e-4, 1e-5, _DL_NOTE),
    ("decoder-large", "cuda", "float16"): _d(4e-3, 1e-3, _DL_NOTE),
    ("decoder-large", "cuda", "bfloat16"): _d(4e-2, 1e-3, _DL_NOTE),
    ("decoder-large", "rocm", "float32"): _d(4e-3, 1e-4, _DL_NOTE),
    ("decoder-large", "rocm", "float16"): _d(8e-3, 1e-3, _DL_NOTE),
    ("decoder-large", "rocm", "bfloat16"): _d(8e-2, 1e-3, _DL_NOTE),
    ("decoder-large", "mps", "float32"): _d(4e-4, 1e-5, _DL_NOTE),
    ("decoder-large", "mps", "float16"): _d(4e-3, 1e-3, _DL_NOTE),
    ("decoder-large", "mps", "bfloat16"): _d(4e-2, 1e-3, _DL_NOTE),
    ("decoder-large", "xla", "float32"): _d(2.0, 1e-2, _DL_NOTE),
    ("decoder-large", "xla", "bfloat16"): _d(2.0, 1e-2, _DL_NOTE),
    ("decoder-large", "cpu", "float32"): _d(4e-6, 1e-6, _DL_NOTE),
    ("decoder-large", "cpu", "float16"): _d(4e-4, 1e-4, _DL_NOTE),
    ("decoder-large", "cpu", "bfloat16"): _d(4e-3, 1e-4, _DL_NOTE),
    # ── encoder ───────────────────────────────────────────────────────────
    ("encoder", "cuda", "float32"): _d(5e-5, 1e-5, _ENC_NOTE),
    ("encoder", "cuda", "float16"): _d(5e-4, 1e-3, _ENC_NOTE),
    ("encoder", "cuda", "bfloat16"): _d(5e-3, 1e-3, _ENC_NOTE),
    ("encoder", "rocm", "float32"): _d(5e-4, 1e-4, _ENC_NOTE),
    ("encoder", "rocm", "float16"): _d(1e-3, 1e-3, _ENC_NOTE),
    ("encoder", "rocm", "bfloat16"): _d(1e-2, 1e-3, _ENC_NOTE),
    ("encoder", "mps", "float32"): _d(5e-5, 1e-5, _ENC_NOTE),
    ("encoder", "mps", "float16"): _d(5e-4, 1e-3, _ENC_NOTE),
    ("encoder", "mps", "bfloat16"): _d(5e-3, 1e-3, _ENC_NOTE),
    ("encoder", "xla", "float32"): _d(0.25, 1e-2, _ENC_NOTE),
    ("encoder", "xla", "bfloat16"): _d(0.25, 1e-2, _ENC_NOTE),
    ("encoder", "cpu", "float32"): _d(5e-7, 1e-6, _ENC_NOTE),
    ("encoder", "cpu", "float16"): _d(5e-5, 1e-4, _ENC_NOTE),
    ("encoder", "cpu", "bfloat16"): _d(5e-4, 1e-4, _ENC_NOTE),
    # ── vision-language ───────────────────────────────────────────────────
    ("vision-language", "cuda", "float32"): _d(3e-4, 1e-5, _VL_NOTE),
    ("vision-language", "cuda", "float16"): _d(3e-3, 1e-3, _VL_NOTE),
    ("vision-language", "cuda", "bfloat16"): _d(3e-2, 1e-3, _VL_NOTE),
    ("vision-language", "rocm", "float32"): _d(3e-3, 1e-4, _VL_NOTE),
    ("vision-language", "rocm", "float16"): _d(6e-3, 1e-3, _VL_NOTE),
    ("vision-language", "rocm", "bfloat16"): _d(6e-2, 1e-3, _VL_NOTE),
    ("vision-language", "mps", "float32"): _d(3e-4, 1e-5, _VL_NOTE),
    ("vision-language", "mps", "float16"): _d(3e-3, 1e-3, _VL_NOTE),
    ("vision-language", "mps", "bfloat16"): _d(3e-2, 1e-3, _VL_NOTE),
    ("vision-language", "xla", "float32"): _d(1.5, 1e-2, _VL_NOTE),
    ("vision-language", "xla", "bfloat16"): _d(1.5, 1e-2, _VL_NOTE),
    ("vision-language", "cpu", "float32"): _d(3e-6, 1e-6, _VL_NOTE),
    ("vision-language", "cpu", "float16"): _d(3e-4, 1e-4, _VL_NOTE),
    ("vision-language", "cpu", "bfloat16"): _d(3e-3, 1e-4, _VL_NOTE),
    # ── trainium ──────────────────────────────────────────────────────────
    # Trainium (AWS Neuron) supports float32 and bfloat16.
    # Measured base: atol=1e-4 (float32), 1e-2 (bfloat16).
    # Family scaling follows same methodology as other backends.
    ("decoder-small", "trainium", "float32"): _m(1e-4, 1e-5, _CLOUD_NOTE),
    ("decoder-small", "trainium", "bfloat16"): _m(1e-2, 1e-3, _CLOUD_NOTE),
    ("decoder-medium", "trainium", "float32"): _d(2e-4, 1e-5, _DM_NOTE),
    ("decoder-medium", "trainium", "bfloat16"): _d(2e-2, 1e-3, _DM_NOTE),
    ("decoder-large", "trainium", "float32"): _d(4e-4, 1e-5, _DL_NOTE),
    ("decoder-large", "trainium", "bfloat16"): _d(4e-2, 1e-3, _DL_NOTE),
    ("encoder", "trainium", "float32"): _d(5e-5, 1e-5, _ENC_NOTE),
    ("encoder", "trainium", "bfloat16"): _d(5e-3, 1e-3, _ENC_NOTE),
    ("vision-language", "trainium", "float32"): _d(3e-4, 1e-5, _VL_NOTE),
    ("vision-language", "trainium", "bfloat16"): _d(3e-2, 1e-3, _VL_NOTE),
    # ── cuda_blackwell (B100/B200, sm_100) — v0.5.69 ──────────────────────
    # Basis: Hopper (cuda) measured entries × 1.0.
    # Same fp32 accumulation path; wider GEMM tiles do not increase worst-case
    # absolute error. float16 included: same reasoning applies.
    ("decoder-small", "cuda_blackwell", "float32"): _d(1e-4, 1e-5, _BW_NOTE),
    ("decoder-small", "cuda_blackwell", "float16"): _d(1e-3, 1e-3, _BW_NOTE),
    ("decoder-small", "cuda_blackwell", "bfloat16"): _d(1e-2, 1e-3, _BW_NOTE),
    ("decoder-medium", "cuda_blackwell", "float32"): _d(2e-4, 1e-5, _BW_NOTE),
    ("decoder-medium", "cuda_blackwell", "float16"): _d(2e-3, 1e-3, _BW_NOTE),
    ("decoder-medium", "cuda_blackwell", "bfloat16"): _d(2e-2, 1e-3, _BW_NOTE),
    ("decoder-large", "cuda_blackwell", "float32"): _d(4e-4, 1e-5, _BW_NOTE),
    ("decoder-large", "cuda_blackwell", "float16"): _d(4e-3, 1e-3, _BW_NOTE),
    ("decoder-large", "cuda_blackwell", "bfloat16"): _d(4e-2, 1e-3, _BW_NOTE),
    ("encoder", "cuda_blackwell", "float32"): _d(5e-5, 1e-5, _BW_NOTE),
    ("encoder", "cuda_blackwell", "float16"): _d(5e-4, 1e-3, _BW_NOTE),
    ("encoder", "cuda_blackwell", "bfloat16"): _d(5e-3, 1e-3, _BW_NOTE),
    ("vision-language", "cuda_blackwell", "float32"): _d(3e-4, 1e-5, _BW_NOTE),
    ("vision-language", "cuda_blackwell", "float16"): _d(3e-3, 1e-3, _BW_NOTE),
    ("vision-language", "cuda_blackwell", "bfloat16"): _d(3e-2, 1e-3, _BW_NOTE),
    # ── cuda_blackwell_consumer (RTX 5090, cc12.0) — v0.5.69 ──────────────
    # Basis: Hopper/Ada (cuda) measured entries × 1.0.
    # Same Blackwell µarch as DC variant; consumer FP rounding matches DC
    # within measurement noise. float16 included for completeness.
    ("decoder-small", "cuda_blackwell_consumer", "float32"): _d(1e-4, 1e-5, _BWC_NOTE),
    ("decoder-small", "cuda_blackwell_consumer", "float16"): _d(1e-3, 1e-3, _BWC_NOTE),
    ("decoder-small", "cuda_blackwell_consumer", "bfloat16"): _d(1e-2, 1e-3, _BWC_NOTE),
    ("decoder-medium", "cuda_blackwell_consumer", "float32"): _d(2e-4, 1e-5, _BWC_NOTE),
    ("decoder-medium", "cuda_blackwell_consumer", "float16"): _d(2e-3, 1e-3, _BWC_NOTE),
    ("decoder-medium", "cuda_blackwell_consumer", "bfloat16"): _d(
        2e-2, 1e-3, _BWC_NOTE
    ),
    ("decoder-large", "cuda_blackwell_consumer", "float32"): _d(4e-4, 1e-5, _BWC_NOTE),
    ("decoder-large", "cuda_blackwell_consumer", "float16"): _d(4e-3, 1e-3, _BWC_NOTE),
    ("decoder-large", "cuda_blackwell_consumer", "bfloat16"): _d(4e-2, 1e-3, _BWC_NOTE),
    ("encoder", "cuda_blackwell_consumer", "float32"): _d(5e-5, 1e-5, _BWC_NOTE),
    ("encoder", "cuda_blackwell_consumer", "float16"): _d(5e-4, 1e-3, _BWC_NOTE),
    ("encoder", "cuda_blackwell_consumer", "bfloat16"): _d(5e-3, 1e-3, _BWC_NOTE),
    ("vision-language", "cuda_blackwell_consumer", "float32"): _d(
        3e-4, 1e-5, _BWC_NOTE
    ),
    ("vision-language", "cuda_blackwell_consumer", "float16"): _d(
        3e-3, 1e-3, _BWC_NOTE
    ),
    ("vision-language", "cuda_blackwell_consumer", "bfloat16"): _d(
        3e-2, 1e-3, _BWC_NOTE
    ),
    # ── rocm_cdna4 (MI350X, gfx950) — v0.5.69 ────────────────────────────
    # Basis: CDNA3 (rocm) measured entries × 1.0.
    # CDNA4 improves HBM3e bandwidth but does not change FP accumulation order.
    # float16 included following the rocm pattern.
    ("decoder-small", "rocm_cdna4", "float32"): _d(1e-3, 1e-4, _CDNA4_NOTE),
    ("decoder-small", "rocm_cdna4", "float16"): _d(2e-3, 1e-3, _CDNA4_NOTE),
    ("decoder-small", "rocm_cdna4", "bfloat16"): _d(2e-2, 1e-3, _CDNA4_NOTE),
    ("decoder-medium", "rocm_cdna4", "float32"): _d(2e-3, 1e-4, _CDNA4_NOTE),
    ("decoder-medium", "rocm_cdna4", "float16"): _d(4e-3, 1e-3, _CDNA4_NOTE),
    ("decoder-medium", "rocm_cdna4", "bfloat16"): _d(4e-2, 1e-3, _CDNA4_NOTE),
    ("decoder-large", "rocm_cdna4", "float32"): _d(4e-3, 1e-4, _CDNA4_NOTE),
    ("decoder-large", "rocm_cdna4", "float16"): _d(8e-3, 1e-3, _CDNA4_NOTE),
    ("decoder-large", "rocm_cdna4", "bfloat16"): _d(8e-2, 1e-3, _CDNA4_NOTE),
    ("encoder", "rocm_cdna4", "float32"): _d(5e-4, 1e-4, _CDNA4_NOTE),
    ("encoder", "rocm_cdna4", "float16"): _d(1e-3, 1e-3, _CDNA4_NOTE),
    ("encoder", "rocm_cdna4", "bfloat16"): _d(1e-2, 1e-3, _CDNA4_NOTE),
    ("vision-language", "rocm_cdna4", "float32"): _d(3e-3, 1e-4, _CDNA4_NOTE),
    ("vision-language", "rocm_cdna4", "float16"): _d(6e-3, 1e-3, _CDNA4_NOTE),
    ("vision-language", "rocm_cdna4", "bfloat16"): _d(6e-2, 1e-3, _CDNA4_NOTE),
    # ── xla_v7 (TPU v7 Ironwood) — v0.5.69 ───────────────────────────────
    # Basis: TPU v5e (xla) measured entries × 1.0.
    # Same XLA compiler stack; bf16 matmul accumulation semantics unchanged.
    # XLA does not expose float16 — those entries are intentionally absent.
    ("decoder-small", "xla_v7", "float32"): _d(0.5, 1e-2, _XLA_V7_NOTE),
    ("decoder-small", "xla_v7", "bfloat16"): _d(0.5, 1e-2, _XLA_V7_NOTE),
    ("decoder-medium", "xla_v7", "float32"): _d(1.0, 1e-2, _XLA_V7_NOTE),
    ("decoder-medium", "xla_v7", "bfloat16"): _d(1.0, 1e-2, _XLA_V7_NOTE),
    ("decoder-large", "xla_v7", "float32"): _d(2.0, 1e-2, _XLA_V7_NOTE),
    ("decoder-large", "xla_v7", "bfloat16"): _d(2.0, 1e-2, _XLA_V7_NOTE),
    ("encoder", "xla_v7", "float32"): _d(0.25, 1e-2, _XLA_V7_NOTE),
    ("encoder", "xla_v7", "bfloat16"): _d(0.25, 1e-2, _XLA_V7_NOTE),
    ("vision-language", "xla_v7", "float32"): _d(1.5, 1e-2, _XLA_V7_NOTE),
    ("vision-language", "xla_v7", "bfloat16"): _d(1.5, 1e-2, _XLA_V7_NOTE),
    # ── trainium_trn3 (AWS Trn3) — v0.5.69 ───────────────────────────────
    # Basis: Trn1 (trainium) measured entries × 1.0.
    # Trn3 uses same NeuronCore accumulation semantics as Trn1; no principled
    # scaling exists — nearest-generation atol used unchanged.
    # Neuron supports float32 and bfloat16 (same as trainium).
    ("decoder-small", "trainium_trn3", "float32"): _d(1e-4, 1e-5, _TRN3_NOTE),
    ("decoder-small", "trainium_trn3", "bfloat16"): _d(1e-2, 1e-3, _TRN3_NOTE),
    ("decoder-medium", "trainium_trn3", "float32"): _d(2e-4, 1e-5, _TRN3_NOTE),
    ("decoder-medium", "trainium_trn3", "bfloat16"): _d(2e-2, 1e-3, _TRN3_NOTE),
    ("decoder-large", "trainium_trn3", "float32"): _d(4e-4, 1e-5, _TRN3_NOTE),
    ("decoder-large", "trainium_trn3", "bfloat16"): _d(4e-2, 1e-3, _TRN3_NOTE),
    ("encoder", "trainium_trn3", "float32"): _d(5e-5, 1e-5, _TRN3_NOTE),
    ("encoder", "trainium_trn3", "bfloat16"): _d(5e-3, 1e-3, _TRN3_NOTE),
    ("vision-language", "trainium_trn3", "float32"): _d(3e-4, 1e-5, _TRN3_NOTE),
    ("vision-language", "trainium_trn3", "bfloat16"): _d(3e-2, 1e-3, _TRN3_NOTE),
    # ── qwen3_5 (Qwen3.5/3.6 27B dense) — v0.5.99 ───────────────────────────
    # Basis: decoder-large (> 20B dense). Same FP accumulation depth.
    # Cloud validation pending; use decoder-large derived values.
    ("qwen3_5", "cuda", "float32"): _d(4e-4, 1e-5, _QWEN35_NOTE),
    ("qwen3_5", "cuda", "float16"): _d(4e-3, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "cuda", "bfloat16"): _d(4e-2, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "rocm", "float32"): _d(4e-3, 1e-4, _QWEN35_NOTE),
    ("qwen3_5", "rocm", "float16"): _d(8e-3, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "rocm", "bfloat16"): _d(8e-2, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "mps", "float32"): _d(4e-4, 1e-5, _QWEN35_NOTE),
    ("qwen3_5", "mps", "float16"): _d(4e-3, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "mps", "bfloat16"): _d(4e-2, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "xla", "float32"): _d(2.0, 1e-2, _QWEN35_NOTE),
    ("qwen3_5", "xla", "bfloat16"): _d(2.0, 1e-2, _QWEN35_NOTE),
    ("qwen3_5", "cpu", "float32"): _d(4e-6, 1e-6, _QWEN35_NOTE),
    ("qwen3_5", "cpu", "float16"): _d(4e-4, 1e-4, _QWEN35_NOTE),
    ("qwen3_5", "cpu", "bfloat16"): _d(4e-3, 1e-4, _QWEN35_NOTE),
    ("qwen3_5", "trainium", "float32"): _d(4e-4, 1e-5, _QWEN35_NOTE),
    ("qwen3_5", "trainium", "bfloat16"): _d(4e-2, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "cuda_blackwell", "float32"): _d(4e-4, 1e-5, _QWEN35_NOTE),
    ("qwen3_5", "cuda_blackwell", "float16"): _d(4e-3, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "cuda_blackwell", "bfloat16"): _d(4e-2, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "cuda_blackwell_consumer", "float32"): _d(4e-4, 1e-5, _QWEN35_NOTE),
    ("qwen3_5", "cuda_blackwell_consumer", "float16"): _d(4e-3, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "cuda_blackwell_consumer", "bfloat16"): _d(4e-2, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "rocm_cdna4", "float32"): _d(4e-3, 1e-4, _QWEN35_NOTE),
    ("qwen3_5", "rocm_cdna4", "float16"): _d(8e-3, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "rocm_cdna4", "bfloat16"): _d(8e-2, 1e-3, _QWEN35_NOTE),
    ("qwen3_5", "xla_v7", "float32"): _d(2.0, 1e-2, _QWEN35_NOTE),
    ("qwen3_5", "xla_v7", "bfloat16"): _d(2.0, 1e-2, _QWEN35_NOTE),
    ("qwen3_5", "trainium_trn3", "float32"): _d(4e-4, 1e-5, _QWEN35_NOTE),
    ("qwen3_5", "trainium_trn3", "bfloat16"): _d(4e-2, 1e-3, _QWEN35_NOTE),
    # ── gemma4 (Gemma 4 26B-A4B, 4B active) — v0.5.99 ───────────────────────
    # Basis: decoder-medium (compute active ~4B). Cloud validation pending.
    ("gemma4", "cuda", "float32"): _d(2e-4, 1e-5, _GEMMA4_NOTE),
    ("gemma4", "cuda", "float16"): _d(2e-3, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "cuda", "bfloat16"): _d(2e-2, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "rocm", "float32"): _d(2e-3, 1e-4, _GEMMA4_NOTE),
    ("gemma4", "rocm", "float16"): _d(4e-3, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "rocm", "bfloat16"): _d(4e-2, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "mps", "float32"): _d(2e-4, 1e-5, _GEMMA4_NOTE),
    ("gemma4", "mps", "float16"): _d(2e-3, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "mps", "bfloat16"): _d(2e-2, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "xla", "float32"): _d(1.0, 1e-2, _GEMMA4_NOTE),
    ("gemma4", "xla", "bfloat16"): _d(1.0, 1e-2, _GEMMA4_NOTE),
    ("gemma4", "cpu", "float32"): _d(2e-6, 1e-6, _GEMMA4_NOTE),
    ("gemma4", "cpu", "float16"): _d(2e-4, 1e-4, _GEMMA4_NOTE),
    ("gemma4", "cpu", "bfloat16"): _d(2e-3, 1e-4, _GEMMA4_NOTE),
    ("gemma4", "trainium", "float32"): _d(2e-4, 1e-5, _GEMMA4_NOTE),
    ("gemma4", "trainium", "bfloat16"): _d(2e-2, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "cuda_blackwell", "float32"): _d(2e-4, 1e-5, _GEMMA4_NOTE),
    ("gemma4", "cuda_blackwell", "float16"): _d(2e-3, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "cuda_blackwell", "bfloat16"): _d(2e-2, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "cuda_blackwell_consumer", "float32"): _d(2e-4, 1e-5, _GEMMA4_NOTE),
    ("gemma4", "cuda_blackwell_consumer", "float16"): _d(2e-3, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "cuda_blackwell_consumer", "bfloat16"): _d(2e-2, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "rocm_cdna4", "float32"): _d(2e-3, 1e-4, _GEMMA4_NOTE),
    ("gemma4", "rocm_cdna4", "float16"): _d(4e-3, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "rocm_cdna4", "bfloat16"): _d(4e-2, 1e-3, _GEMMA4_NOTE),
    ("gemma4", "xla_v7", "float32"): _d(1.0, 1e-2, _GEMMA4_NOTE),
    ("gemma4", "xla_v7", "bfloat16"): _d(1.0, 1e-2, _GEMMA4_NOTE),
    ("gemma4", "trainium_trn3", "float32"): _d(2e-4, 1e-5, _GEMMA4_NOTE),
    ("gemma4", "trainium_trn3", "bfloat16"): _d(2e-2, 1e-3, _GEMMA4_NOTE),
    # ── nemotron3_nano (30B-A3B, 3B active) — v0.5.99 ────────────────────────
    # Basis: decoder-medium (compute active ~3B). Cloud validation pending.
    ("nemotron3_nano", "cuda", "float32"): _d(2e-4, 1e-5, _NM3NANO_NOTE),
    ("nemotron3_nano", "cuda", "float16"): _d(2e-3, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "cuda", "bfloat16"): _d(2e-2, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "rocm", "float32"): _d(2e-3, 1e-4, _NM3NANO_NOTE),
    ("nemotron3_nano", "rocm", "float16"): _d(4e-3, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "rocm", "bfloat16"): _d(4e-2, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "mps", "float32"): _d(2e-4, 1e-5, _NM3NANO_NOTE),
    ("nemotron3_nano", "mps", "float16"): _d(2e-3, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "mps", "bfloat16"): _d(2e-2, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "xla", "float32"): _d(1.0, 1e-2, _NM3NANO_NOTE),
    ("nemotron3_nano", "xla", "bfloat16"): _d(1.0, 1e-2, _NM3NANO_NOTE),
    ("nemotron3_nano", "cpu", "float32"): _d(2e-6, 1e-6, _NM3NANO_NOTE),
    ("nemotron3_nano", "cpu", "float16"): _d(2e-4, 1e-4, _NM3NANO_NOTE),
    ("nemotron3_nano", "cpu", "bfloat16"): _d(2e-3, 1e-4, _NM3NANO_NOTE),
    ("nemotron3_nano", "trainium", "float32"): _d(2e-4, 1e-5, _NM3NANO_NOTE),
    ("nemotron3_nano", "trainium", "bfloat16"): _d(2e-2, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "cuda_blackwell", "float32"): _d(2e-4, 1e-5, _NM3NANO_NOTE),
    ("nemotron3_nano", "cuda_blackwell", "float16"): _d(2e-3, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "cuda_blackwell", "bfloat16"): _d(2e-2, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "cuda_blackwell_consumer", "float32"): _d(
        2e-4, 1e-5, _NM3NANO_NOTE
    ),
    ("nemotron3_nano", "cuda_blackwell_consumer", "float16"): _d(
        2e-3, 1e-3, _NM3NANO_NOTE
    ),
    ("nemotron3_nano", "cuda_blackwell_consumer", "bfloat16"): _d(
        2e-2, 1e-3, _NM3NANO_NOTE
    ),
    ("nemotron3_nano", "rocm_cdna4", "float32"): _d(2e-3, 1e-4, _NM3NANO_NOTE),
    ("nemotron3_nano", "rocm_cdna4", "float16"): _d(4e-3, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "rocm_cdna4", "bfloat16"): _d(4e-2, 1e-3, _NM3NANO_NOTE),
    ("nemotron3_nano", "xla_v7", "float32"): _d(1.0, 1e-2, _NM3NANO_NOTE),
    ("nemotron3_nano", "xla_v7", "bfloat16"): _d(1.0, 1e-2, _NM3NANO_NOTE),
    ("nemotron3_nano", "trainium_trn3", "float32"): _d(2e-4, 1e-5, _NM3NANO_NOTE),
    ("nemotron3_nano", "trainium_trn3", "bfloat16"): _d(2e-2, 1e-3, _NM3NANO_NOTE),
    # ── deepseek_v4 (V4 Flash/Pro, 13–49B active) — v0.5.99 ─────────────────
    # Basis: decoder-large × 1.5 (MoE expert routing adds variance).
    # 284B+ total params but active compute ~decoder-large; routing bump applied.
    ("deepseek_v4", "cuda", "float32"): _d(6e-4, 1e-5, _DSV4_NOTE),
    ("deepseek_v4", "cuda", "float16"): _d(6e-3, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "cuda", "bfloat16"): _d(6e-2, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "rocm", "float32"): _d(6e-3, 1e-4, _DSV4_NOTE),
    ("deepseek_v4", "rocm", "float16"): _d(1.2e-2, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "rocm", "bfloat16"): _d(1.2e-1, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "mps", "float32"): _d(6e-4, 1e-5, _DSV4_NOTE),
    ("deepseek_v4", "mps", "float16"): _d(6e-3, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "mps", "bfloat16"): _d(6e-2, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "xla", "float32"): _d(3.0, 1e-2, _DSV4_NOTE),
    ("deepseek_v4", "xla", "bfloat16"): _d(3.0, 1e-2, _DSV4_NOTE),
    ("deepseek_v4", "cpu", "float32"): _d(6e-6, 1e-6, _DSV4_NOTE),
    ("deepseek_v4", "cpu", "float16"): _d(6e-4, 1e-4, _DSV4_NOTE),
    ("deepseek_v4", "cpu", "bfloat16"): _d(6e-3, 1e-4, _DSV4_NOTE),
    ("deepseek_v4", "trainium", "float32"): _d(6e-4, 1e-5, _DSV4_NOTE),
    ("deepseek_v4", "trainium", "bfloat16"): _d(6e-2, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "cuda_blackwell", "float32"): _d(6e-4, 1e-5, _DSV4_NOTE),
    ("deepseek_v4", "cuda_blackwell", "float16"): _d(6e-3, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "cuda_blackwell", "bfloat16"): _d(6e-2, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "cuda_blackwell_consumer", "float32"): _d(6e-4, 1e-5, _DSV4_NOTE),
    ("deepseek_v4", "cuda_blackwell_consumer", "float16"): _d(6e-3, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "cuda_blackwell_consumer", "bfloat16"): _d(6e-2, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "rocm_cdna4", "float32"): _d(6e-3, 1e-4, _DSV4_NOTE),
    ("deepseek_v4", "rocm_cdna4", "float16"): _d(1.2e-2, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "rocm_cdna4", "bfloat16"): _d(1.2e-1, 1e-3, _DSV4_NOTE),
    ("deepseek_v4", "xla_v7", "float32"): _d(3.0, 1e-2, _DSV4_NOTE),
    ("deepseek_v4", "xla_v7", "bfloat16"): _d(3.0, 1e-2, _DSV4_NOTE),
    ("deepseek_v4", "trainium_trn3", "float32"): _d(6e-4, 1e-5, _DSV4_NOTE),
    ("deepseek_v4", "trainium_trn3", "bfloat16"): _d(6e-2, 1e-3, _DSV4_NOTE),
    # ── nemotron3_ultra (550B/55B active) — v0.5.99 ───────────────────────────
    # Basis: decoder-large × 2 (large MoE, 55B active compute depth).
    ("nemotron3_ultra", "cuda", "float32"): _d(8e-4, 1e-5, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cuda", "float16"): _d(8e-3, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cuda", "bfloat16"): _d(8e-2, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "rocm", "float32"): _d(8e-3, 1e-4, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "rocm", "float16"): _d(1.6e-2, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "rocm", "bfloat16"): _d(1.6e-1, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "mps", "float32"): _d(8e-4, 1e-5, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "mps", "float16"): _d(8e-3, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "mps", "bfloat16"): _d(8e-2, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "xla", "float32"): _d(4.0, 1e-2, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "xla", "bfloat16"): _d(4.0, 1e-2, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cpu", "float32"): _d(8e-6, 1e-6, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cpu", "float16"): _d(8e-4, 1e-4, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cpu", "bfloat16"): _d(8e-3, 1e-4, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "trainium", "float32"): _d(8e-4, 1e-5, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "trainium", "bfloat16"): _d(8e-2, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cuda_blackwell", "float32"): _d(8e-4, 1e-5, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cuda_blackwell", "float16"): _d(8e-3, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cuda_blackwell", "bfloat16"): _d(8e-2, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "cuda_blackwell_consumer", "float32"): _d(
        8e-4, 1e-5, _NM3ULTRA_NOTE
    ),
    ("nemotron3_ultra", "cuda_blackwell_consumer", "float16"): _d(
        8e-3, 1e-3, _NM3ULTRA_NOTE
    ),
    ("nemotron3_ultra", "cuda_blackwell_consumer", "bfloat16"): _d(
        8e-2, 1e-3, _NM3ULTRA_NOTE
    ),
    ("nemotron3_ultra", "rocm_cdna4", "float32"): _d(8e-3, 1e-4, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "rocm_cdna4", "float16"): _d(1.6e-2, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "rocm_cdna4", "bfloat16"): _d(1.6e-1, 1e-3, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "xla_v7", "float32"): _d(4.0, 1e-2, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "xla_v7", "bfloat16"): _d(4.0, 1e-2, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "trainium_trn3", "float32"): _d(8e-4, 1e-5, _NM3ULTRA_NOTE),
    ("nemotron3_ultra", "trainium_trn3", "bfloat16"): _d(8e-2, 1e-3, _NM3ULTRA_NOTE),
    # ── tencent_hy3 (295B/21B active) — v0.5.99 ──────────────────────────────
    # Basis: decoder-large × 1.5 (MoE, 21B active). Same tier as deepseek_v4.
    ("tencent_hy3", "cuda", "float32"): _d(6e-4, 1e-5, _HY3_NOTE),
    ("tencent_hy3", "cuda", "float16"): _d(6e-3, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "cuda", "bfloat16"): _d(6e-2, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "rocm", "float32"): _d(6e-3, 1e-4, _HY3_NOTE),
    ("tencent_hy3", "rocm", "float16"): _d(1.2e-2, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "rocm", "bfloat16"): _d(1.2e-1, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "mps", "float32"): _d(6e-4, 1e-5, _HY3_NOTE),
    ("tencent_hy3", "mps", "float16"): _d(6e-3, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "mps", "bfloat16"): _d(6e-2, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "xla", "float32"): _d(3.0, 1e-2, _HY3_NOTE),
    ("tencent_hy3", "xla", "bfloat16"): _d(3.0, 1e-2, _HY3_NOTE),
    ("tencent_hy3", "cpu", "float32"): _d(6e-6, 1e-6, _HY3_NOTE),
    ("tencent_hy3", "cpu", "float16"): _d(6e-4, 1e-4, _HY3_NOTE),
    ("tencent_hy3", "cpu", "bfloat16"): _d(6e-3, 1e-4, _HY3_NOTE),
    ("tencent_hy3", "trainium", "float32"): _d(6e-4, 1e-5, _HY3_NOTE),
    ("tencent_hy3", "trainium", "bfloat16"): _d(6e-2, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "cuda_blackwell", "float32"): _d(6e-4, 1e-5, _HY3_NOTE),
    ("tencent_hy3", "cuda_blackwell", "float16"): _d(6e-3, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "cuda_blackwell", "bfloat16"): _d(6e-2, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "cuda_blackwell_consumer", "float32"): _d(6e-4, 1e-5, _HY3_NOTE),
    ("tencent_hy3", "cuda_blackwell_consumer", "float16"): _d(6e-3, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "cuda_blackwell_consumer", "bfloat16"): _d(6e-2, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "rocm_cdna4", "float32"): _d(6e-3, 1e-4, _HY3_NOTE),
    ("tencent_hy3", "rocm_cdna4", "float16"): _d(1.2e-2, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "rocm_cdna4", "bfloat16"): _d(1.2e-1, 1e-3, _HY3_NOTE),
    ("tencent_hy3", "xla_v7", "float32"): _d(3.0, 1e-2, _HY3_NOTE),
    ("tencent_hy3", "xla_v7", "bfloat16"): _d(3.0, 1e-2, _HY3_NOTE),
    ("tencent_hy3", "trainium_trn3", "float32"): _d(6e-4, 1e-5, _HY3_NOTE),
    ("tencent_hy3", "trainium_trn3", "bfloat16"): _d(6e-2, 1e-3, _HY3_NOTE),
    # ── minimax_m3 (~428B/23B active, MSA attention) — v0.5.99 ───────────────
    # Basis: decoder-large × 1.5 (MoE, 23B active). MSA = Lightning attention.
    ("minimax_m3", "cuda", "float32"): _d(6e-4, 1e-5, _MMM3_NOTE),
    ("minimax_m3", "cuda", "float16"): _d(6e-3, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "cuda", "bfloat16"): _d(6e-2, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "rocm", "float32"): _d(6e-3, 1e-4, _MMM3_NOTE),
    ("minimax_m3", "rocm", "float16"): _d(1.2e-2, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "rocm", "bfloat16"): _d(1.2e-1, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "mps", "float32"): _d(6e-4, 1e-5, _MMM3_NOTE),
    ("minimax_m3", "mps", "float16"): _d(6e-3, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "mps", "bfloat16"): _d(6e-2, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "xla", "float32"): _d(3.0, 1e-2, _MMM3_NOTE),
    ("minimax_m3", "xla", "bfloat16"): _d(3.0, 1e-2, _MMM3_NOTE),
    ("minimax_m3", "cpu", "float32"): _d(6e-6, 1e-6, _MMM3_NOTE),
    ("minimax_m3", "cpu", "float16"): _d(6e-4, 1e-4, _MMM3_NOTE),
    ("minimax_m3", "cpu", "bfloat16"): _d(6e-3, 1e-4, _MMM3_NOTE),
    ("minimax_m3", "trainium", "float32"): _d(6e-4, 1e-5, _MMM3_NOTE),
    ("minimax_m3", "trainium", "bfloat16"): _d(6e-2, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "cuda_blackwell", "float32"): _d(6e-4, 1e-5, _MMM3_NOTE),
    ("minimax_m3", "cuda_blackwell", "float16"): _d(6e-3, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "cuda_blackwell", "bfloat16"): _d(6e-2, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "cuda_blackwell_consumer", "float32"): _d(6e-4, 1e-5, _MMM3_NOTE),
    ("minimax_m3", "cuda_blackwell_consumer", "float16"): _d(6e-3, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "cuda_blackwell_consumer", "bfloat16"): _d(6e-2, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "rocm_cdna4", "float32"): _d(6e-3, 1e-4, _MMM3_NOTE),
    ("minimax_m3", "rocm_cdna4", "float16"): _d(1.2e-2, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "rocm_cdna4", "bfloat16"): _d(1.2e-1, 1e-3, _MMM3_NOTE),
    ("minimax_m3", "xla_v7", "float32"): _d(3.0, 1e-2, _MMM3_NOTE),
    ("minimax_m3", "xla_v7", "bfloat16"): _d(3.0, 1e-2, _MMM3_NOTE),
    ("minimax_m3", "trainium_trn3", "float32"): _d(6e-4, 1e-5, _MMM3_NOTE),
    ("minimax_m3", "trainium_trn3", "bfloat16"): _d(6e-2, 1e-3, _MMM3_NOTE),
    # ── glm_5_2 (744B/40B active, IndexShare) — v0.5.99 ──────────────────────
    # Basis: decoder-large × 2 (large MoE, 40B active + IndexShare routing overhead).
    ("glm_5_2", "cuda", "float32"): _d(8e-4, 1e-5, _GLM52_NOTE),
    ("glm_5_2", "cuda", "float16"): _d(8e-3, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "cuda", "bfloat16"): _d(8e-2, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "rocm", "float32"): _d(8e-3, 1e-4, _GLM52_NOTE),
    ("glm_5_2", "rocm", "float16"): _d(1.6e-2, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "rocm", "bfloat16"): _d(1.6e-1, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "mps", "float32"): _d(8e-4, 1e-5, _GLM52_NOTE),
    ("glm_5_2", "mps", "float16"): _d(8e-3, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "mps", "bfloat16"): _d(8e-2, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "xla", "float32"): _d(4.0, 1e-2, _GLM52_NOTE),
    ("glm_5_2", "xla", "bfloat16"): _d(4.0, 1e-2, _GLM52_NOTE),
    ("glm_5_2", "cpu", "float32"): _d(8e-6, 1e-6, _GLM52_NOTE),
    ("glm_5_2", "cpu", "float16"): _d(8e-4, 1e-4, _GLM52_NOTE),
    ("glm_5_2", "cpu", "bfloat16"): _d(8e-3, 1e-4, _GLM52_NOTE),
    ("glm_5_2", "trainium", "float32"): _d(8e-4, 1e-5, _GLM52_NOTE),
    ("glm_5_2", "trainium", "bfloat16"): _d(8e-2, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "cuda_blackwell", "float32"): _d(8e-4, 1e-5, _GLM52_NOTE),
    ("glm_5_2", "cuda_blackwell", "float16"): _d(8e-3, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "cuda_blackwell", "bfloat16"): _d(8e-2, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "cuda_blackwell_consumer", "float32"): _d(8e-4, 1e-5, _GLM52_NOTE),
    ("glm_5_2", "cuda_blackwell_consumer", "float16"): _d(8e-3, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "cuda_blackwell_consumer", "bfloat16"): _d(8e-2, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "rocm_cdna4", "float32"): _d(8e-3, 1e-4, _GLM52_NOTE),
    ("glm_5_2", "rocm_cdna4", "float16"): _d(1.6e-2, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "rocm_cdna4", "bfloat16"): _d(1.6e-1, 1e-3, _GLM52_NOTE),
    ("glm_5_2", "xla_v7", "float32"): _d(4.0, 1e-2, _GLM52_NOTE),
    ("glm_5_2", "xla_v7", "bfloat16"): _d(4.0, 1e-2, _GLM52_NOTE),
    ("glm_5_2", "trainium_trn3", "float32"): _d(8e-4, 1e-5, _GLM52_NOTE),
    ("glm_5_2", "trainium_trn3", "bfloat16"): _d(8e-2, 1e-3, _GLM52_NOTE),
    # ── rocm_rdna4 — RDNA4 (RX 9000 series, gfx1201) — v0.5.100 ─────────────
    # Derived from RDNA3/rocm consumer GPU values; no rocBLAS FP8; same accumulation.
    ("decoder-small", "rocm_rdna4", "float32"): _d(1e-3, 1e-4, _RDNA4_NOTE),
    ("decoder-small", "rocm_rdna4", "float16"): _d(2e-3, 1e-3, _RDNA4_NOTE),
    ("decoder-small", "rocm_rdna4", "bfloat16"): _d(2e-2, 1e-3, _RDNA4_NOTE),
    ("decoder-medium", "rocm_rdna4", "float32"): _d(2e-3, 1e-4, _RDNA4_NOTE),
    ("decoder-medium", "rocm_rdna4", "float16"): _d(4e-3, 1e-3, _RDNA4_NOTE),
    ("decoder-medium", "rocm_rdna4", "bfloat16"): _d(4e-2, 1e-3, _RDNA4_NOTE),
    ("decoder-large", "rocm_rdna4", "float32"): _d(4e-3, 1e-4, _RDNA4_NOTE),
    ("decoder-large", "rocm_rdna4", "float16"): _d(8e-3, 1e-3, _RDNA4_NOTE),
    ("decoder-large", "rocm_rdna4", "bfloat16"): _d(8e-2, 1e-3, _RDNA4_NOTE),
    ("encoder", "rocm_rdna4", "float32"): _d(5e-4, 1e-4, _RDNA4_NOTE),
    ("encoder", "rocm_rdna4", "float16"): _d(1e-3, 1e-3, _RDNA4_NOTE),
    ("encoder", "rocm_rdna4", "bfloat16"): _d(1e-2, 1e-3, _RDNA4_NOTE),
    ("vision-language", "rocm_rdna4", "float32"): _d(3e-3, 1e-4, _RDNA4_NOTE),
    ("vision-language", "rocm_rdna4", "float16"): _d(6e-3, 1e-3, _RDNA4_NOTE),
    ("vision-language", "rocm_rdna4", "bfloat16"): _d(6e-2, 1e-3, _RDNA4_NOTE),
    ("qwen3_5", "rocm_rdna4", "float32"): _d(4e-3, 1e-4, _RDNA4_NOTE),
    ("qwen3_5", "rocm_rdna4", "float16"): _d(8e-3, 1e-3, _RDNA4_NOTE),
    ("qwen3_5", "rocm_rdna4", "bfloat16"): _d(8e-2, 1e-3, _RDNA4_NOTE),
    ("gemma4", "rocm_rdna4", "float32"): _d(2e-3, 1e-4, _RDNA4_NOTE),
    ("gemma4", "rocm_rdna4", "float16"): _d(4e-3, 1e-3, _RDNA4_NOTE),
    ("gemma4", "rocm_rdna4", "bfloat16"): _d(4e-2, 1e-3, _RDNA4_NOTE),
    ("nemotron3_nano", "rocm_rdna4", "float32"): _d(2e-3, 1e-4, _RDNA4_NOTE),
    ("nemotron3_nano", "rocm_rdna4", "float16"): _d(4e-3, 1e-3, _RDNA4_NOTE),
    ("nemotron3_nano", "rocm_rdna4", "bfloat16"): _d(4e-2, 1e-3, _RDNA4_NOTE),
    ("deepseek_v4", "rocm_rdna4", "float32"): _d(6e-3, 1e-4, _RDNA4_NOTE),
    ("deepseek_v4", "rocm_rdna4", "float16"): _d(1.2e-2, 1e-3, _RDNA4_NOTE),
    ("deepseek_v4", "rocm_rdna4", "bfloat16"): _d(1.2e-1, 1e-3, _RDNA4_NOTE),
    ("nemotron3_ultra", "rocm_rdna4", "float32"): _d(8e-3, 1e-4, _RDNA4_NOTE),
    ("nemotron3_ultra", "rocm_rdna4", "float16"): _d(1.6e-2, 1e-3, _RDNA4_NOTE),
    ("nemotron3_ultra", "rocm_rdna4", "bfloat16"): _d(1.6e-1, 1e-3, _RDNA4_NOTE),
    ("tencent_hy3", "rocm_rdna4", "float32"): _d(6e-3, 1e-4, _RDNA4_NOTE),
    ("tencent_hy3", "rocm_rdna4", "float16"): _d(1.2e-2, 1e-3, _RDNA4_NOTE),
    ("tencent_hy3", "rocm_rdna4", "bfloat16"): _d(1.2e-1, 1e-3, _RDNA4_NOTE),
    ("minimax_m3", "rocm_rdna4", "float32"): _d(6e-3, 1e-4, _RDNA4_NOTE),
    ("minimax_m3", "rocm_rdna4", "float16"): _d(1.2e-2, 1e-3, _RDNA4_NOTE),
    ("minimax_m3", "rocm_rdna4", "bfloat16"): _d(1.2e-1, 1e-3, _RDNA4_NOTE),
    ("glm_5_2", "rocm_rdna4", "float32"): _d(8e-3, 1e-4, _RDNA4_NOTE),
    ("glm_5_2", "rocm_rdna4", "float16"): _d(1.6e-2, 1e-3, _RDNA4_NOTE),
    ("glm_5_2", "rocm_rdna4", "bfloat16"): _d(1.6e-1, 1e-3, _RDNA4_NOTE),
    # ── trainium2 — AWS Trainium2 (TRN2, NeuronCore v3) — v0.5.100 ───────────
    # Derived from trainium (Trn1) measured values; Trn2 uses same XLA accumulation.
    # Neuron SDK 2.26+ FP8 path not yet measured on Trn2 — use Trn1 baseline.
    ("decoder-small", "trainium2", "float32"): _d(1e-4, 1e-5, _TRN2_NOTE),
    ("decoder-small", "trainium2", "bfloat16"): _d(1e-2, 1e-3, _TRN2_NOTE),
    ("decoder-medium", "trainium2", "float32"): _d(2e-4, 1e-5, _TRN2_NOTE),
    ("decoder-medium", "trainium2", "bfloat16"): _d(2e-2, 1e-3, _TRN2_NOTE),
    ("decoder-large", "trainium2", "float32"): _d(4e-4, 1e-5, _TRN2_NOTE),
    ("decoder-large", "trainium2", "bfloat16"): _d(4e-2, 1e-3, _TRN2_NOTE),
    ("encoder", "trainium2", "float32"): _d(5e-5, 1e-5, _TRN2_NOTE),
    ("encoder", "trainium2", "bfloat16"): _d(5e-3, 1e-3, _TRN2_NOTE),
    ("vision-language", "trainium2", "float32"): _d(3e-4, 1e-5, _TRN2_NOTE),
    ("vision-language", "trainium2", "bfloat16"): _d(3e-2, 1e-3, _TRN2_NOTE),
    ("qwen3_5", "trainium2", "float32"): _d(4e-4, 1e-5, _TRN2_NOTE),
    ("qwen3_5", "trainium2", "bfloat16"): _d(4e-2, 1e-3, _TRN2_NOTE),
    ("gemma4", "trainium2", "float32"): _d(2e-4, 1e-5, _TRN2_NOTE),
    ("gemma4", "trainium2", "bfloat16"): _d(2e-2, 1e-3, _TRN2_NOTE),
    ("nemotron3_nano", "trainium2", "float32"): _d(2e-4, 1e-5, _TRN2_NOTE),
    ("nemotron3_nano", "trainium2", "bfloat16"): _d(2e-2, 1e-3, _TRN2_NOTE),
    ("deepseek_v4", "trainium2", "float32"): _d(6e-4, 1e-5, _TRN2_NOTE),
    ("deepseek_v4", "trainium2", "bfloat16"): _d(6e-2, 1e-3, _TRN2_NOTE),
    ("nemotron3_ultra", "trainium2", "float32"): _d(8e-4, 1e-5, _TRN2_NOTE),
    ("nemotron3_ultra", "trainium2", "bfloat16"): _d(8e-2, 1e-3, _TRN2_NOTE),
    ("tencent_hy3", "trainium2", "float32"): _d(6e-4, 1e-5, _TRN2_NOTE),
    ("tencent_hy3", "trainium2", "bfloat16"): _d(6e-2, 1e-3, _TRN2_NOTE),
    ("minimax_m3", "trainium2", "float32"): _d(6e-4, 1e-5, _TRN2_NOTE),
    ("minimax_m3", "trainium2", "bfloat16"): _d(6e-2, 1e-3, _TRN2_NOTE),
    ("glm_5_2", "trainium2", "float32"): _d(8e-4, 1e-5, _TRN2_NOTE),
    ("glm_5_2", "trainium2", "bfloat16"): _d(8e-2, 1e-3, _TRN2_NOTE),
}


# ---------------------------------------------------------------------------
# ToleranceDB
# ---------------------------------------------------------------------------


class ToleranceDB:
    """Look up empirical tolerances by backend, dtype, and optional model family.

    Fallback chain:
      ``(model_family, backend, dtype)`` →
      ``(backend, dtype)`` →
      :data:`_DEFAULT_TOLERANCE`

    Example::

        db = ToleranceDB()

        # Backward-compatible (no model family)
        tol = db.get("cuda", "float16")
        assert tol.atol == 1e-3

        # Family-aware lookup
        tol = db.get("cuda", "float32", model_family="decoder-large")
        assert tol.atol == 4e-4
        assert tol.source == "derived"
    """

    def __init__(
        self, extra: dict[tuple[str, str], TolerancePair] | None = None
    ) -> None:
        self._table: dict[tuple[str, str], TolerancePair] = dict(_TOLERANCE_TABLE)
        self._family_table: dict[tuple[str, str, str], ToleranceEntry] = dict(
            _FAMILY_TOLERANCE_TABLE
        )
        if extra:
            self._table.update(extra)

    def get(
        self,
        backend: str,
        dtype: str,
        model_family: str | None = None,
    ) -> ToleranceEntry:
        """Return tolerances for the given backend, dtype, and optional model family.

        Fallback chain:
          ``(model_family, backend, dtype)`` →
          ``(backend, dtype)`` →
          :data:`_DEFAULT_TOLERANCE`

        Args:
            backend: Backend string (e.g. ``"cuda"``, ``"rocm"``, ``"cpu"``).
            dtype: PyTorch dtype as string (e.g. ``"float32"``, ``"bfloat16"``).
            model_family: Optional family string (e.g. ``"decoder-large"``).
                If ``None`` or not found in the family table, falls back to the
                base ``(backend, dtype)`` table.

        Returns:
            :class:`ToleranceEntry` with ``atol``, ``rtol``, and ``source``.
        """
        b = backend.strip().lower()
        d = dtype.strip().lower()

        if model_family is not None:
            entry = self._family_table.get((model_family.strip().lower(), b, d))
            if entry is not None:
                if (b, d) in _UNVERIFIED_KEYS:
                    # The family rows for an unverified backend are the same
                    # unverifiable number, or derived from it. This is the path
                    # a real run takes — decoder-small is Qwen3-0.6B — so
                    # leaving it labelled "measured" would defeat the point.
                    return ToleranceEntry(
                        atol=entry.atol,
                        rtol=entry.rtol,
                        source="fallback",
                        notes=f"unverified — see _UNVERIFIED_KEYS ({entry.notes})",
                    )
                return entry

        if (b, d) in self._table:
            base = self._table[(b, d)]
            source = "measured" if (b, d) in _TOLERANCE_TABLE else "derived"
            notes = "base (backend, dtype) lookup — no model-family entry"
            if (b, d) in _UNVERIFIED_KEYS:
                source = "fallback"
                notes = "unverified — see _UNVERIFIED_KEYS"
        else:
            base = _DEFAULT_TOLERANCE
            source = "fallback"
            notes = f"unknown backend '{b}' — using safe default"
            logger.warning(
                "ToleranceDB: no entry for backend '%s' (dtype='%s') — "
                "returning safe-default fallback (atol=%.0e). "
                "Call ToleranceDB.register() to add measured tolerances for this backend.",
                b,
                d,
                base.atol,
            )
        return ToleranceEntry(
            atol=base.atol, rtol=base.rtol, source=source, notes=notes
        )

    def register(self, backend: str, dtype: str, atol: float, rtol: float) -> None:
        """Register a custom tolerance for ``(backend, dtype)``.

        Note: the ``source`` label returned by subsequent ``get()`` calls reflects
        whether the *original* ``(backend, dtype)`` key was in the built-in
        ``_TOLERANCE_TABLE`` (``"measured"``) or not (``"derived"``). It does not
        track the provenance of the custom value. Use :meth:`register_family` if
        you need explicit source metadata.

        Raises:
            ValueError: If ``atol`` or ``rtol`` is negative.
        """
        if atol < 0:
            raise ValueError(f"atol must be >= 0, got {atol}")
        if rtol < 0:
            raise ValueError(f"rtol must be >= 0, got {rtol}")
        self._table[(backend.strip().lower(), dtype.strip().lower())] = TolerancePair(
            atol=atol, rtol=rtol
        )

    def register_family(
        self,
        model_family: str,
        backend: str,
        dtype: str,
        atol: float,
        rtol: float,
        source: str = "measured",
        notes: str = "",
    ) -> None:
        """Register a custom tolerance for ``(model_family, backend, dtype)``.

        Raises:
            ValueError: If ``atol`` or ``rtol`` is negative.
        """
        if atol < 0:
            raise ValueError(f"atol must be >= 0, got {atol}")
        if rtol < 0:
            raise ValueError(f"rtol must be >= 0, got {rtol}")
        self._family_table[
            (
                model_family.strip().lower(),
                backend.strip().lower(),
                dtype.strip().lower(),
            )
        ] = ToleranceEntry(atol=atol, rtol=rtol, source=source, notes=notes)

    def all_backends(self) -> list[str]:
        """Return all backend names with registered base tolerances."""
        return sorted({b for b, _ in self._table})

    def families(self) -> list[str]:
        """Return all model family names with registered tolerances."""
        seen = {fam for fam, _, _ in self._family_table}
        return sorted(seen)

    def is_measured(
        self,
        backend: str,
        dtype: str,
        model_family: str | None = None,
    ) -> bool:
        """Return ``True`` if the best available entry has ``source == "measured"``."""
        return self.get(backend, dtype, model_family=model_family).source == "measured"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full tolerance table to a JSON-compatible dict."""
        result: dict[str, Any] = {}
        # Base table
        for (b, d), t in sorted(self._table.items()):
            result[f"{b}/{d}"] = {"atol": t.atol, "rtol": t.rtol}
        # Family table
        for (fam, b, d), te in sorted(self._family_table.items()):
            result[f"{fam}/{b}/{d}"] = {
                "atol": te.atol,
                "rtol": te.rtol,
                "source": te.source,
                "notes": te.notes,
            }
        return result
