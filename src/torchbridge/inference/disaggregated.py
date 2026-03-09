"""
Disaggregated Fleet Config Advisor

Matrix-first configuration advisor for prefill-decode disaggregated LLM serving.

Prefill workers are compute-bound (large activation memory for prompt processing).
Decode workers are memory-bound (KV cache dominates; autoregressive one-token steps).
They often run on different hardware — e.g. NVIDIA Hopper for prefill, AMD CDNA3 for decode.

This module does NOT implement serving. Use vLLM, SGLang, or NVIDIA Dynamo for the runtime.
TorchBridge's role: (prefill_hw, decode_hw, model_params) → optimal per-role config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Matrix 1 — KV dtype per (role, backend, architecture)
#
# Key insight:
#   prefill is compute-bound → use higher-quality dtype (bfloat16 / float16)
#   decode is memory-bound  → use int8 to halve KV cache footprint where stable
# ---------------------------------------------------------------------------
_KV_DTYPE_MATRIX: dict[tuple[str, str, str | None], str] = {
    # --- prefill ---
    ("prefill", "cuda", "hopper"):    "bfloat16",   # Hopper native BF16 tensor cores
    ("prefill", "cuda", "blackwell"): "bfloat16",   # Blackwell: BF16 + FP8 supported
    ("prefill", "cuda", "ampere"):    "float16",    # Ampere: FP16 preferred
    ("prefill", "cuda", "ada"):       "float16",    # Ada (RTX 4090-class)
    ("prefill", "cuda", None):        "float16",    # Unknown NVIDIA → conservative
    ("prefill", "rocm", "cdna4"):     "bfloat16",   # CDNA4 (MI350X): native BF16
    ("prefill", "rocm", "cdna3"):     "bfloat16",   # CDNA3 (MI300X): native BF16
    ("prefill", "rocm", "cdna2"):     "float16",    # CDNA2 (MI250X): FP16
    ("prefill", "rocm", None):        "float16",
    ("prefill", "tpu", None):         "bfloat16",   # TPU: BF16 native
    ("prefill", "cpu", None):         "float32",    # CPU: no HBM pressure

    # --- decode ---
    ("decode", "cuda", "hopper"):     "int8",       # Memory-bound: int8 halves KV size
    ("decode", "cuda", "blackwell"):  "int8",
    ("decode", "cuda", "ampere"):     "int8",
    ("decode", "cuda", "ada"):        "int8",
    ("decode", "cuda", None):         "int8",
    ("decode", "rocm", "cdna4"):      "int8",
    ("decode", "rocm", "cdna3"):      "int8",
    ("decode", "rocm", "cdna2"):      "float16",    # CDNA2: int8 KV less stable
    ("decode", "rocm", None):         "float16",
    ("decode", "tpu", None):          "bfloat16",   # XLA doesn't support int8 KV natively
    ("decode", "cpu", None):          "float32",
}

# ---------------------------------------------------------------------------
# Matrix 2 — KV transfer format between prefill and decode
#
# This is the agreed dtype/layout for KV tensors handed off between roles.
# float16 is the safe cross-vendor common format; same-vendor uses bfloat16.
# ---------------------------------------------------------------------------
_TRANSFER_FORMAT_MATRIX: dict[tuple[str, str], str] = {
    ("cuda", "cuda"): "bfloat16",   # Same vendor: highest-quality common format
    ("cuda", "rocm"): "float16",    # Cross-vendor: float16 is universal
    ("rocm", "cuda"): "float16",
    ("rocm", "rocm"): "bfloat16",
    ("cuda", "cpu"):  "float32",    # CPU can always accept float32
    ("rocm", "cpu"):  "float32",
    ("cpu", "cuda"):  "float32",
    ("cpu", "rocm"):  "float32",
    ("cpu", "cpu"):   "float32",
    ("tpu", "cuda"):  "float16",
    ("tpu", "rocm"):  "float16",
    ("cuda", "tpu"):  "float16",
    ("rocm", "tpu"):  "float16",
    ("tpu", "tpu"):   "bfloat16",
}

# ---------------------------------------------------------------------------
# Matrix 3 — Memory split per role
#
# (kv_cache_fraction, model_fraction)
# Prefill: large activation memory for prompt processing; KV cache is small
# Decode:  KV cache dominates; model weights are static; maximize cache budget
# ---------------------------------------------------------------------------
_MEMORY_SPLIT_MATRIX: dict[str, tuple[float, float]] = {
    "prefill": (0.20, 0.80),   # 20% KV cache, 80% model + activations
    "decode":  (0.80, 0.20),   # 80% KV cache, 20% model weights
}

# Batch size and sequence length heuristics per role
_BATCH_HEURISTICS: dict[str, dict[str, int]] = {
    "prefill": {"base_batch": 8,   "max_seq_len": 32768},
    "decode":  {"base_batch": 128, "max_seq_len": 2048},
}

# Default GPU memory by backend/role when caller doesn't specify
_DEFAULT_MEMORY_GB: dict[str, float] = {
    "cuda": 80.0,   # H100/A100 80GB
    "rocm": 80.0,   # MI300X nominally 192GB, use conservative 80 as floor
    "tpu":  16.0,
    "cpu":  0.0,    # No HBM; budget computation skipped
}

# Human-readable architecture labels for output
_ARCH_LABELS: dict[tuple[str, str | None], str] = {
    ("cuda", "hopper"):    "NVIDIA Hopper",
    ("cuda", "blackwell"): "NVIDIA Blackwell",
    ("cuda", "ampere"):    "NVIDIA Ampere",
    ("cuda", "ada"):       "NVIDIA Ada",
    ("cuda", None):        "NVIDIA (unknown arch)",
    ("rocm", "cdna4"):     "AMD CDNA4",
    ("rocm", "cdna3"):     "AMD CDNA3",
    ("rocm", "cdna2"):     "AMD CDNA2",
    ("rocm", None):        "AMD (unknown arch)",
    ("tpu", None):         "Google TPU",
    ("cpu", None):         "CPU",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DisaggregatedRoleConfig:
    """Configuration for one role (prefill or decode) in a disaggregated fleet."""

    role: str
    """\"prefill\" or \"decode\"."""

    backend: str
    """Backend name, e.g. \"cuda\", \"rocm\", \"tpu\", \"cpu\"."""

    architecture: str | None
    """Micro-architecture, e.g. \"hopper\", \"cdna3\". None if unknown."""

    kv_dtype: str
    """Recommended KV cache dtype: \"bfloat16\", \"float16\", \"int8\", or \"float32\"."""

    kv_cache_budget_gb: float
    """HBM fraction (GB) to allocate to KV cache."""

    max_batch_size: int
    """Recommended maximum batch size for this role."""

    max_seq_len: int
    """Recommended maximum sequence length for this role."""

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "backend": self.backend,
            "architecture": self.architecture,
            "kv_dtype": self.kv_dtype,
            "kv_cache_budget_gb": self.kv_cache_budget_gb,
            "max_batch_size": self.max_batch_size,
            "max_seq_len": self.max_seq_len,
            "notes": list(self.notes),
        }


@dataclass
class DisaggregatedFleetConfig:
    """Complete configuration for a prefill-decode disaggregated serving fleet."""

    model_params: int
    """Total model parameter count."""

    prefill: DisaggregatedRoleConfig
    """Configuration for the prefill role."""

    decode: DisaggregatedRoleConfig
    """Configuration for the decode role."""

    kv_transfer_format: str
    """Agreed format for cross-role KV tensor handoff (prefill → decode)."""

    notes: list[str] = field(default_factory=list)
    """Fleet-level notes (warnings, caveats)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_params": self.model_params,
            "prefill": self.prefill.to_dict(),
            "decode": self.decode.to_dict(),
            "kv_transfer_format": self.kv_transfer_format,
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------

class DisaggregatedFleetAdvisor:
    """Matrix-first config advisor for prefill-decode disaggregated serving.

    Does NOT implement serving. Use vLLM, SGLang, or NVIDIA Dynamo for the runtime.
    TorchBridge's role: (prefill_hw, decode_hw, model_params) → optimal per-role config.
    """

    @staticmethod
    def recommend(
        model_params: int,
        prefill_backend: str,
        decode_backend: str,
        prefill_arch: str | None = None,
        decode_arch: str | None = None,
        prefill_memory_gb: float | None = None,
        decode_memory_gb: float | None = None,
    ) -> DisaggregatedFleetConfig:
        """Generate a hardware-aware config for a disaggregated serving fleet.

        Args:
            model_params: Total model parameter count (e.g. 7_000_000_000).
            prefill_backend: Backend for prefill workers: ``"cuda"``, ``"rocm"``,
                ``"tpu"``, or ``"cpu"``.
            decode_backend: Backend for decode workers.
            prefill_arch: Micro-architecture for prefill hardware (e.g. ``"hopper"``).
                None triggers a conservative fallback.
            decode_arch: Micro-architecture for decode hardware.
            prefill_memory_gb: Total GPU memory (GB) for prefill workers.
                Defaults to a backend-specific heuristic.
            decode_memory_gb: Total GPU memory (GB) for decode workers.

        Returns:
            :class:`DisaggregatedFleetConfig` with per-role configs and
            the agreed KV transfer format.
        """
        prefill_backend = prefill_backend.lower()
        decode_backend = decode_backend.lower()
        if prefill_arch is not None:
            prefill_arch = prefill_arch.lower()
        if decode_arch is not None:
            decode_arch = decode_arch.lower()

        prefill_mem = (
            prefill_memory_gb
            if prefill_memory_gb is not None
            else _DEFAULT_MEMORY_GB.get(prefill_backend, 80.0)
        )
        decode_mem = (
            decode_memory_gb
            if decode_memory_gb is not None
            else _DEFAULT_MEMORY_GB.get(decode_backend, 80.0)
        )

        prefill_cfg = _build_role_config(
            "prefill", prefill_backend, prefill_arch, prefill_mem, model_params
        )
        decode_cfg = _build_role_config(
            "decode", decode_backend, decode_arch, decode_mem, model_params
        )

        transfer_fmt = _TRANSFER_FORMAT_MATRIX.get(
            (prefill_backend, decode_backend), "float16"
        )

        fleet_notes: list[str] = [
            "Use vLLM/SGLang/NVIDIA Dynamo for the serving runtime",
        ]
        if model_params >= 70_000_000_000:
            fleet_notes.append(
                "Large model (≥70B): consider tensor-parallel within each role"
            )

        return DisaggregatedFleetConfig(
            model_params=model_params,
            prefill=prefill_cfg,
            decode=decode_cfg,
            kv_transfer_format=transfer_fmt,
            notes=fleet_notes,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lookup_kv_dtype(role: str, backend: str, arch: str | None) -> tuple[str, str]:
    """Return (kv_dtype, note) for the given role/backend/arch combination."""
    key = (role, backend, arch)
    if key in _KV_DTYPE_MATRIX:
        dtype = _KV_DTYPE_MATRIX[key]
        arch_label = _ARCH_LABELS.get((backend, arch), f"{backend}/{arch}")
        note = _dtype_note(role, dtype, arch_label)
        return dtype, note

    # Fallback: try (role, backend, None)
    fallback_key = (role, backend, None)
    if fallback_key in _KV_DTYPE_MATRIX:
        dtype = _KV_DTYPE_MATRIX[fallback_key]
        note = f"{role} KV dtype {dtype}: unknown arch for {backend}, using conservative default"
        return dtype, note

    # Last resort
    return "float16", f"{role} KV dtype float16: backend {backend!r} not in matrix, using safe default"


def _dtype_note(role: str, dtype: str, arch_label: str) -> str:
    """Generate a human-readable note for a KV dtype choice."""
    if role == "prefill":
        if dtype == "bfloat16":
            return f"Prefill KV dtype bfloat16: {arch_label} native BF16 tensor cores"
        if dtype == "float16":
            return f"Prefill KV dtype float16: {arch_label} FP16 preferred"
        return f"Prefill KV dtype {dtype}: {arch_label}"
    else:  # decode
        if dtype == "int8":
            return "Decode KV dtype int8: memory-bound role; int8 halves KV cache footprint"
        if dtype == "float16":
            return f"Decode KV dtype float16: int8 KV less stable on {arch_label}"
        return f"Decode KV dtype {dtype}: {arch_label}"


def _build_role_config(
    role: str,
    backend: str,
    arch: str | None,
    memory_gb: float,
    model_params: int,
) -> DisaggregatedRoleConfig:
    """Build a DisaggregatedRoleConfig for one role."""
    kv_dtype, dtype_note = _lookup_kv_dtype(role, backend, arch)

    kv_fraction, _model_fraction = _MEMORY_SPLIT_MATRIX.get(role, (0.20, 0.80))
    kv_budget = memory_gb * kv_fraction

    heuristics = _BATCH_HEURISTICS[role]
    # Scale max_batch_size with memory: larger GPU → larger batch, smaller GPU → smaller batch
    memory_scale = memory_gb / 80.0 if memory_gb > 0.0 else 1.0
    max_batch = max(1, int(heuristics["base_batch"] * memory_scale))
    max_seq = heuristics["max_seq_len"]

    notes = [dtype_note]
    if model_params >= 30_000_000_000 and role == "decode":
        notes.append("Large model on decode: monitor KV cache eviction rate")

    return DisaggregatedRoleConfig(
        role=role,
        backend=backend,
        architecture=arch,
        kv_dtype=kv_dtype,
        kv_cache_budget_gb=round(kv_budget, 1),
        max_batch_size=max_batch,
        max_seq_len=max_seq,
        notes=notes,
    )
