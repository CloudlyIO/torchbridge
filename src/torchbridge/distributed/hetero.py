"""
Heterogeneous Cluster Training Configuration Advisor

Advises on FSDP, collective bridge, and partition strategy for mixed
NVIDIA+AMD training clusters. The key matrix is
``(NVIDIAArchitecture, AMDArchitecture) → collective_bridge`` — validated by
the HetCCL paper (arXiv 2601.22585) for Hopper+CDNA3/4 pairings.

TorchBridge produces configuration recommendations; PyTorch Distributed
(torch.distributed, FSDP) performs the actual training.

Example::

    from torchbridge.distributed.hetero import HeterogeneousClusterAdvisor
    from torchbridge.core.config import NVIDIAArchitecture, AMDArchitecture

    cfg = HeterogeneousClusterAdvisor.recommend(
        nvidia_count=4,
        nvidia_arch=NVIDIAArchitecture.HOPPER,
        amd_count=8,
        amd_arch=AMDArchitecture.CDNA3,
        model_params=7_000_000_000,
    )
    print(cfg.collective_bridge)   # "hetccl"
    print(cfg.partition_strategy)  # "vendor_isolated" or "memory_balanced"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Matrix 1 — Collective bridge selection
# ---------------------------------------------------------------------------

#: Maps ``(NVIDIAArchitecture, AMDArchitecture)`` → collective bridge name.
#:
#: "hetccl" — HetCCL (arXiv 2601.22585): validated for Hopper+CDNA3/4 pairings.
#: "ucc"    — Unified Collective Communication: general-purpose fallback for
#:            arch pairs not covered by HetCCL validation.
#: "gloo"   — Gloo: CPU-side fallback only (not recommended for GPU training).
_COLLECTIVE_BRIDGE_MATRIX: dict[tuple, str] = {
    # Hopper + CDNA3/4: validated by HetCCL paper
    (NVIDIAArchitecture.HOPPER, AMDArchitecture.CDNA4): "hetccl",
    (NVIDIAArchitecture.HOPPER, AMDArchitecture.CDNA3): "hetccl",
    (NVIDIAArchitecture.HOPPER, AMDArchitecture.CDNA2): "ucc",
    # Blackwell DC: HetCCL expected (architecture newer than paper)
    (NVIDIAArchitecture.BLACKWELL_DC, AMDArchitecture.CDNA4): "hetccl",
    (NVIDIAArchitecture.BLACKWELL_DC, AMDArchitecture.CDNA3): "hetccl",
    (NVIDIAArchitecture.BLACKWELL_DC, AMDArchitecture.CDNA2): "ucc",
    # Ampere: UCC stable; HetCCL not validated for Ampere
    (NVIDIAArchitecture.AMPERE, AMDArchitecture.CDNA3): "ucc",
    (NVIDIAArchitecture.AMPERE, AMDArchitecture.CDNA2): "ucc",
    # Ada (consumer): UCC
    (NVIDIAArchitecture.ADA, AMDArchitecture.CDNA3): "ucc",
    (NVIDIAArchitecture.ADA, AMDArchitecture.CDNA2): "ucc",
}

#: Fallback bridge when arch pair is not in ``_COLLECTIVE_BRIDGE_MATRIX``.
_COLLECTIVE_BRIDGE_DEFAULT: str = "ucc"

# ---------------------------------------------------------------------------
# Matrix 2 — Partition strategy thresholds
# ---------------------------------------------------------------------------

#: List of ``(min_ratio, strategy)`` pairs sorted descending by ratio.
#: ``ratio = total_amd_memory_gb / total_nvidia_memory_gb``.
#:
#: "memory_balanced" — distribute shards across all GPUs regardless of vendor.
#:   Maximises model capacity when AMD has substantially more memory.
#:   Cost: increased cross-vendor all-reduce traffic (~2× vs isolated).
#: "vendor_isolated" — shard within each vendor group; use cross-vendor only
#:   for gradient synchronisation. Minimises expensive cross-vendor bandwidth.
_PARTITION_THRESHOLDS: list[tuple[float, str]] = [
    (2.0, "memory_balanced"),  # AMD ≥ 2× NVIDIA total memory
    (0.0, "vendor_isolated"),  # balanced or NVIDIA-heavy
]

# ---------------------------------------------------------------------------
# GPU memory reference (GB) — mirrors config.py _GPU_MEMORY_MAP
# ---------------------------------------------------------------------------

_NVIDIA_MEMORY_GB: dict[NVIDIAArchitecture | None, float] = {
    NVIDIAArchitecture.BLACKWELL_DC: 192.0,
    NVIDIAArchitecture.BLACKWELL_CONSUMER: 32.0,
    NVIDIAArchitecture.HOPPER: 80.0,
    NVIDIAArchitecture.ADA: 48.0,
    NVIDIAArchitecture.AMPERE: 80.0,
    NVIDIAArchitecture.TURING: 24.0,
    None: 80.0,
}

_AMD_MEMORY_GB: dict[AMDArchitecture | None, float] = {
    AMDArchitecture.CDNA4: 288.0,
    AMDArchitecture.CDNA3: 192.0,
    AMDArchitecture.CDNA2: 128.0,
    None: 192.0,
}

# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class HeterogeneousClusterConfig:
    """
    Recommended configuration for a mixed NVIDIA+AMD training cluster.

    Produced by :meth:`HeterogeneousClusterAdvisor.recommend`. All fields
    are plain Python scalars — pass them to PyTorch's distributed APIs
    yourself; TorchBridge does not perform distributed training.
    """

    nvidia_count: int
    nvidia_arch: NVIDIAArchitecture | None
    amd_count: int
    amd_arch: AMDArchitecture | None
    model_params: int
    collective_bridge: str
    partition_strategy: str
    nvidia_fsdp_strategy: str
    amd_fsdp_strategy: str
    nvidia_mixed_precision: str
    amd_mixed_precision: str
    estimated_cross_vendor_comm_gb: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "nvidia_count": self.nvidia_count,
            "nvidia_arch": self.nvidia_arch.value if self.nvidia_arch else None,
            "amd_count": self.amd_count,
            "amd_arch": self.amd_arch.value if self.amd_arch else None,
            "model_params": self.model_params,
            "collective_bridge": self.collective_bridge,
            "partition_strategy": self.partition_strategy,
            "nvidia_fsdp_strategy": self.nvidia_fsdp_strategy,
            "amd_fsdp_strategy": self.amd_fsdp_strategy,
            "nvidia_mixed_precision": self.nvidia_mixed_precision,
            "amd_mixed_precision": self.amd_mixed_precision,
            "estimated_cross_vendor_comm_gb": round(
                self.estimated_cross_vendor_comm_gb, 3
            ),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Advisor
# ---------------------------------------------------------------------------


class HeterogeneousClusterAdvisor:
    """
    Config advisor for mixed NVIDIA+AMD training clusters.

    The core logic is two matrix lookups:
    1. ``_COLLECTIVE_BRIDGE_MATRIX[(nvidia_arch, amd_arch)]`` → bridge.
    2. ``_PARTITION_THRESHOLDS`` walked by AMD/NVIDIA memory ratio → strategy.

    All other values (FSDP strategy, mixed precision) are derived from
    existing per-backend matrices in ``fsdp.py``.
    """

    @staticmethod
    def recommend(
        nvidia_count: int,
        nvidia_arch: NVIDIAArchitecture | None,
        amd_count: int,
        amd_arch: AMDArchitecture | None,
        model_params: int,
    ) -> HeterogeneousClusterConfig:
        """
        Recommend a heterogeneous cluster configuration.

        Args:
            nvidia_count: Number of NVIDIA GPUs in the cluster.
            nvidia_arch: NVIDIA GPU architecture (None = unknown).
            amd_count: Number of AMD GPUs in the cluster.
            amd_arch: AMD GPU architecture (None = unknown).
            model_params: Total model parameters.

        Returns:
            :class:`HeterogeneousClusterConfig` with all recommendations.
        """
        notes: list[str] = []

        # 1. Collective bridge lookup
        bridge = _COLLECTIVE_BRIDGE_MATRIX.get(
            (nvidia_arch, amd_arch), _COLLECTIVE_BRIDGE_DEFAULT
        )
        if bridge == "hetccl":
            notes.append(
                f"HetCCL bridge validated for {_arch_label(nvidia_arch, 'nvidia')} "
                f"+ {_arch_label(amd_arch, 'amd')} (arXiv 2601.22585)."
            )
        else:
            notes.append(
                f"UCC bridge recommended for {_arch_label(nvidia_arch, 'nvidia')} "
                f"+ {_arch_label(amd_arch, 'amd')}; HetCCL not validated for this pair."
            )

        # 2. Memory ratio → partition strategy
        nvidia_mem = _NVIDIA_MEMORY_GB.get(nvidia_arch, _NVIDIA_MEMORY_GB[None])
        amd_mem = _AMD_MEMORY_GB.get(amd_arch, _AMD_MEMORY_GB[None])
        total_nvidia_gb = nvidia_mem * nvidia_count
        total_amd_gb = amd_mem * amd_count

        ratio = total_amd_gb / total_nvidia_gb if total_nvidia_gb > 0 else 1.0
        partition_strategy = "vendor_isolated"  # default; overridden by threshold walk
        for threshold, strategy in _PARTITION_THRESHOLDS:
            if ratio >= threshold:
                partition_strategy = strategy
                break

        if partition_strategy == "memory_balanced":
            notes.append(
                f"memory_balanced: AMD total memory ({total_amd_gb:.0f} GB) is "
                f"{ratio:.1f}× NVIDIA ({total_nvidia_gb:.0f} GB) — cross-vendor sharding "
                f"worth the bandwidth penalty."
            )
        else:
            notes.append(
                f"vendor_isolated: NVIDIA {total_nvidia_gb:.0f} GB / AMD {total_amd_gb:.0f} GB "
                f"(ratio {ratio:.1f}×) — isolate shards per vendor to minimise "
                f"cross-vendor all-reduce traffic."
            )

        # 3. Per-vendor FSDP strategy
        nvidia_fsdp = "hybrid_shard" if nvidia_count > 8 else "full_shard"
        amd_fsdp = "hybrid_shard" if amd_count > 8 else "full_shard"

        # 4. Mixed precision from existing FSDP matrix
        nvidia_mp = _lookup_mixed_precision(HardwareBackend.CUDA, nvidia_arch)
        amd_mp = _lookup_mixed_precision(HardwareBackend.AMD, amd_arch)

        # 5. Cross-vendor communication estimate
        bytes_per_param = 2  # BF16
        model_bytes = model_params * bytes_per_param
        if partition_strategy == "memory_balanced":
            # all-gather + reduce-scatter cross-vendor
            cross_comm_gb = 4 * model_bytes / (1024**3)
        else:
            # one reduce-scatter across vendor boundary per step
            cross_comm_gb = 2 * model_bytes / (1024**3)

        notes.append(
            f"Estimated cross-vendor comm: {cross_comm_gb:.2f} GB/step "
            f"({partition_strategy})."
        )

        return HeterogeneousClusterConfig(
            nvidia_count=nvidia_count,
            nvidia_arch=nvidia_arch,
            amd_count=amd_count,
            amd_arch=amd_arch,
            model_params=model_params,
            collective_bridge=bridge,
            partition_strategy=partition_strategy,
            nvidia_fsdp_strategy=nvidia_fsdp,
            amd_fsdp_strategy=amd_fsdp,
            nvidia_mixed_precision=nvidia_mp,
            amd_mixed_precision=amd_mp,
            estimated_cross_vendor_comm_gb=cross_comm_gb,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _lookup_mixed_precision(backend: HardwareBackend, arch) -> str:
    """Return mixed precision string for the given backend+arch from FSDPManager."""
    from torchbridge.distributed.fsdp import FSDPManager

    try:
        mgr = FSDPManager(backend=backend, architecture=arch)
        return mgr.mixed_precision.value
    except Exception:
        return "bf16"


def _arch_label(arch, vendor: str) -> str:
    """Human-readable label for an arch enum value."""
    if arch is None:
        return f"{vendor}/unknown"
    return arch.value
