"""
KV Cache Handoff Spec

Matrix-first physical layout advisor for KV cache tensors handed between
prefill and decode workers in a disaggregated serving fleet.

v0.5.60 told you *what dtype* to use for the handoff.
v0.5.61 tells you *how to lay out the memory*:
  - page_size_tokens: KV cache page size (tokens per page)
  - alignment_bytes:  memory alignment requirement
  - layout:           "separate" (K and V buffers apart) or "interleaved" (K₀V₀K₁V₁…)

This module does NOT implement KV cache memory. Use vLLM, SGLang, or NVIDIA Dynamo
for the runtime. TorchBridge's role: (prefill_hw, decode_hw) → safe layout spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Matrix: (backend, arch) → (page_size_tokens, alignment_bytes, layout)
#
# Sources:
#   CUDA:  cuBLAS requires 128-byte alignment for Hopper/Ampere/Blackwell;
#          Ada (RTX 40-series) works at 64-byte.
#   ROCm:  HIP memory alignment is 64 bytes on all CDNA generations.
#   TPU:   XLA paged-attention kernel uses 32-token pages and interleaved KV.
#   CPU:   No paging constraint; 1-token pages, 64-byte cache-line alignment.
# ---------------------------------------------------------------------------
_KV_HANDOFF_MATRIX: dict[tuple[str, str | None], tuple[int, int, str]] = {
    # (backend, arch) → (page_size_tokens, alignment_bytes, layout)
    ("cuda", "hopper"): (16, 128, "separate"),
    ("cuda", "blackwell"): (16, 128, "separate"),
    ("cuda", "ampere"): (16, 128, "separate"),
    ("cuda", "ada"): (16, 64, "separate"),  # RTX 40-series: 64-byte safe
    ("cuda", None): (16, 128, "separate"),  # unknown NVIDIA → conservative
    ("rocm", "cdna4"): (16, 64, "separate"),  # MI350X: 64-byte HIP alignment
    ("rocm", "cdna3"): (16, 64, "separate"),  # MI300X
    ("rocm", "cdna2"): (16, 64, "separate"),  # MI250X
    ("rocm", None): (16, 64, "separate"),
    ("tpu", None): (32, 128, "interleaved"),  # XLA paged-attn: 32-token pages
    ("cpu", None): (1, 64, "separate"),  # no paging; cache-line alignment
}

# Safe universal default if backend not in matrix
_SAFE_DEFAULT: tuple[int, int, str] = (16, 64, "separate")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class KVHandoffSpec:
    """Physical memory specification for KV cache tensors at the prefill→decode boundary."""

    dtype: str
    """KV tensor dtype agreed for handoff (e.g. ``"float16"``). Comes from v0.5.60 matrix."""

    layout: str
    """Memory layout: ``"separate"`` (K and V in distinct buffers) or
    ``"interleaved"`` (K₀V₀K₁V₁… in a single buffer)."""

    page_size_tokens: int
    """Number of tokens per KV cache page. Must be consistent across roles."""

    alignment_bytes: int
    """Required memory alignment in bytes. Both sides must satisfy this."""

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype": self.dtype,
            "layout": self.layout,
            "page_size_tokens": self.page_size_tokens,
            "alignment_bytes": self.alignment_bytes,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Negotiator
# ---------------------------------------------------------------------------


class KVHandoffNegotiator:
    """Matrix-first negotiator for KV cache handoff physical spec.

    Does NOT implement KV cache memory. Use vLLM/SGLang/Dynamo for the runtime.
    TorchBridge's role: (prefill_hw, decode_hw, dtype) → safe layout intersection.

    Negotiation rules:
    - ``page_size_tokens``: ``max(prefill, decode)`` — decode dominates cache layout
    - ``alignment_bytes``: ``min(prefill, decode)`` — strictest requirement both must meet
    - ``layout``: ``"interleaved"`` only if *both* sides prefer it; otherwise ``"separate"``
    """

    @staticmethod
    def negotiate(
        prefill_backend: str,
        decode_backend: str,
        kv_dtype: str,
        prefill_arch: str | None = None,
        decode_arch: str | None = None,
    ) -> KVHandoffSpec:
        """Negotiate a KV handoff physical spec for a disaggregated fleet pair.

        Args:
            prefill_backend: Backend for prefill workers (``"cuda"``, ``"rocm"``,
                ``"tpu"``, ``"cpu"``).
            decode_backend: Backend for decode workers.
            kv_dtype: Agreed KV dtype from :class:`DisaggregatedFleetAdvisor`
                (e.g. ``"float16"``).
            prefill_arch: Micro-architecture for prefill hardware. ``None`` triggers
                a conservative fallback.
            decode_arch: Micro-architecture for decode hardware.

        Returns:
            :class:`KVHandoffSpec` with the safe intersection of layout requirements.
        """
        prefill_backend = prefill_backend.lower()
        decode_backend = decode_backend.lower()
        if prefill_arch is not None:
            prefill_arch = prefill_arch.lower()
        if decode_arch is not None:
            decode_arch = decode_arch.lower()

        p_page, p_align, p_layout = _lookup_hw_spec(prefill_backend, prefill_arch)
        d_page, d_align, d_layout = _lookup_hw_spec(decode_backend, decode_arch)

        page_size = max(p_page, d_page)
        alignment = min(p_align, d_align)
        layout = (
            "interleaved"
            if (p_layout == "interleaved" and d_layout == "interleaved")
            else "separate"
        )

        notes = _build_notes(
            prefill_backend,
            prefill_arch,
            decode_backend,
            decode_arch,
            p_page,
            d_page,
            p_align,
            d_align,
            p_layout,
            d_layout,
        )

        return KVHandoffSpec(
            dtype=kv_dtype,
            layout=layout,
            page_size_tokens=page_size,
            alignment_bytes=alignment,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lookup_hw_spec(backend: str, arch: str | None) -> tuple[int, int, str]:
    """Return (page_size, alignment, layout) for a backend/arch, with fallback."""
    key = (backend, arch)
    if key in _KV_HANDOFF_MATRIX:
        return _KV_HANDOFF_MATRIX[key]
    # Try (backend, None)
    fallback = (backend, None)
    if fallback in _KV_HANDOFF_MATRIX:
        return _KV_HANDOFF_MATRIX[fallback]
    return _SAFE_DEFAULT


def _build_notes(
    pb: str,
    pa: str | None,
    db: str,
    da: str | None,
    pp: int,
    dp: int,
    pal: int,
    dal: int,
    pl: str,
    dl: str,
) -> list[str]:
    notes: list[str] = []
    if pp != dp:
        winner = "decode" if dp > pp else "prefill"
        notes.append(
            f"Page size negotiated to {max(pp, dp)} tokens "
            f"({winner} side dominates: {pb}/{pa or 'unknown'}={pp}, "
            f"{db}/{da or 'unknown'}={dp})"
        )
    if pal != dal:
        notes.append(
            f"Alignment negotiated to {min(pal, dal)} bytes "
            f"(min of {pb}={pal}, {db}={dal})"
        )
    if pl != dl:
        notes.append(
            f"Layout set to 'separate' — sides differ ({pb}={pl!r}, {db}={dl!r})"
        )
    notes.append(
        "Configure page_size_tokens and alignment_bytes in your vLLM/SGLang runtime config"
    )
    return notes
