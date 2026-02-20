"""
Communication Backend Abstraction

Maps hardware backends to optimal collective communication libraries
(NCCL, RCCL, Neuron CC, XLA collectives, Gloo) with support for
advanced features like symmetric memory and FP8 all-reduce.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)

logger = logging.getLogger(__name__)

# Type alias
Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)


class CollectiveBackendType(Enum):
    """Communication backend for distributed collectives."""

    NCCL = "nccl"
    RCCL = "rccl"
    NEURON_CC = "neuron_cc"
    XLA_COLLECTIVES = "xla"
    GLOO = "gloo"


@dataclass(frozen=True)
class CollectiveBackendSpec:
    """Specification for a collective backend."""

    backend: CollectiveBackendType
    display_name: str
    supports_gpu_direct: bool
    supports_fp8_reduce: bool
    supports_symmetric_memory: bool
    description: str


COLLECTIVE_BACKEND_SPECS: dict[CollectiveBackendType, CollectiveBackendSpec] = {
    CollectiveBackendType.NCCL: CollectiveBackendSpec(
        backend=CollectiveBackendType.NCCL,
        display_name="NCCL",
        supports_gpu_direct=True,
        supports_fp8_reduce=True,
        supports_symmetric_memory=True,
        description="NVIDIA Collective Communications Library (CUDA GPUs)",
    ),
    CollectiveBackendType.RCCL: CollectiveBackendSpec(
        backend=CollectiveBackendType.RCCL,
        display_name="RCCL",
        supports_gpu_direct=True,
        supports_fp8_reduce=False,
        supports_symmetric_memory=False,
        description="ROCm Communication Collectives Library (AMD GPUs)",
    ),
    CollectiveBackendType.NEURON_CC: CollectiveBackendSpec(
        backend=CollectiveBackendType.NEURON_CC,
        display_name="Neuron CC",
        supports_gpu_direct=False,
        supports_fp8_reduce=False,
        supports_symmetric_memory=False,
        description="AWS Neuron Collective Compute (Trainium/Inferentia)",
    ),
    CollectiveBackendType.XLA_COLLECTIVES: CollectiveBackendSpec(
        backend=CollectiveBackendType.XLA_COLLECTIVES,
        display_name="XLA Collectives",
        supports_gpu_direct=False,
        supports_fp8_reduce=False,
        supports_symmetric_memory=False,
        description="XLA collective operations (TPU)",
    ),
    CollectiveBackendType.GLOO: CollectiveBackendSpec(
        backend=CollectiveBackendType.GLOO,
        display_name="Gloo",
        supports_gpu_direct=False,
        supports_fp8_reduce=False,
        supports_symmetric_memory=False,
        description="CPU-based collective backend (fallback)",
    ),
}


# ── Compatibility Matrix ────────────────────────────────────────────────────

# Maps HardwareBackend → optimal CollectiveBackendType
_BACKEND_MAP: dict[HardwareBackend, CollectiveBackendType] = {
    HardwareBackend.CUDA: CollectiveBackendType.NCCL,
    HardwareBackend.AMD: CollectiveBackendType.RCCL,
    HardwareBackend.TRAINIUM: CollectiveBackendType.NEURON_CC,
    HardwareBackend.TPU: CollectiveBackendType.XLA_COLLECTIVES,
    HardwareBackend.CPU: CollectiveBackendType.GLOO,
}

# Architectures that support symmetric memory (NCCL 2.27+ on NVLink domains)
_SYMMETRIC_MEMORY_ARCHS: set[NVIDIAArchitecture] = {
    NVIDIAArchitecture.HOPPER,
    NVIDIAArchitecture.BLACKWELL_DC,
    NVIDIAArchitecture.BLACKWELL_CONSUMER,
}

# Architectures that support FP8 all-reduce
_FP8_REDUCE_ARCHS: set[NVIDIAArchitecture] = {
    NVIDIAArchitecture.HOPPER,
    NVIDIAArchitecture.BLACKWELL_DC,
    NVIDIAArchitecture.BLACKWELL_CONSUMER,
    NVIDIAArchitecture.ADA,
}


class CollectiveBackendMatrix:
    """Maps (backend, architecture) to optimal collective configuration."""

    @staticmethod
    def get_optimal_backend(
        hw: HardwareBackend,
        architecture: Architecture = None,
    ) -> CollectiveBackendType:
        """Return the optimal collective backend for the given hardware."""
        return _BACKEND_MAP.get(hw, CollectiveBackendType.GLOO)

    @staticmethod
    def supports_symmetric_memory(
        hw: HardwareBackend,
        architecture: Architecture = None,
    ) -> bool:
        """Check if symmetric memory is available (NCCL 2.27+ on NVLink).

        Symmetric memory provides 9x latency reduction for small messages
        on NVLink-connected domains (Hopper+).
        """
        if hw != HardwareBackend.CUDA:
            return False
        if isinstance(architecture, NVIDIAArchitecture):
            return architecture in _SYMMETRIC_MEMORY_ARCHS
        return False

    @staticmethod
    def supports_fp8_reduce(
        hw: HardwareBackend,
        architecture: Architecture = None,
    ) -> bool:
        """Check if FP8 all-reduce is supported."""
        if hw != HardwareBackend.CUDA:
            return False
        if isinstance(architecture, NVIDIAArchitecture):
            return architecture in _FP8_REDUCE_ARCHS
        return False

    @staticmethod
    def get_torch_backend_string(
        hw: HardwareBackend,
    ) -> str:
        """Return the string expected by torch.distributed.init_process_group()."""
        mapping = {
            CollectiveBackendType.NCCL: "nccl",
            CollectiveBackendType.RCCL: "nccl",  # RCCL uses "nccl" in PyTorch
            CollectiveBackendType.GLOO: "gloo",
        }
        optimal = CollectiveBackendMatrix.get_optimal_backend(hw)
        return mapping.get(optimal, "gloo")

    @staticmethod
    def get_all_backends() -> list[dict[str, Any]]:
        """Return all backend specs as dicts (for CLI matrix display)."""
        return [
            {
                "backend": spec.backend.value,
                "display_name": spec.display_name,
                "gpu_direct": spec.supports_gpu_direct,
                "fp8_reduce": spec.supports_fp8_reduce,
                "symmetric_memory": spec.supports_symmetric_memory,
                "description": spec.description,
            }
            for spec in COLLECTIVE_BACKEND_SPECS.values()
        ]


@dataclass
class CollectiveConfig:
    """Configuration for collective communication."""

    backend: CollectiveBackendType | None = None  # None = auto-select
    symmetric_memory: bool = False
    float8_reduce: bool = False

    def resolve(
        self,
        hw: HardwareBackend,
        architecture: Architecture = None,
    ) -> CollectiveConfig:
        """Resolve auto settings based on hardware."""
        resolved_backend = (
            self.backend
            or CollectiveBackendMatrix.get_optimal_backend(hw, architecture)
        )
        resolved_symmetric = (
            self.symmetric_memory
            or CollectiveBackendMatrix.supports_symmetric_memory(hw, architecture)
        )
        resolved_fp8 = (
            self.float8_reduce
            or CollectiveBackendMatrix.supports_fp8_reduce(hw, architecture)
        )
        return CollectiveConfig(
            backend=resolved_backend,
            symmetric_memory=resolved_symmetric,
            float8_reduce=resolved_fp8,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "backend": self.backend.value if self.backend else "auto",
            "symmetric_memory": self.symmetric_memory,
            "float8_reduce": self.float8_reduce,
        }
