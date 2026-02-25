"""
FSDP Configuration Manager

Backend-aware FSDP configuration that auto-selects sharding strategy,
mixed precision policy, and communication optimizations per hardware.

Note: ``apply()`` uses ``torch.distributed.fsdp.FullyShardedDataParallel``
(the stable FSDP API).  The composable FSDP2 API
(``torch.distributed._composable.fsdp``) is experimental and not yet used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch.nn as nn

from torchbridge.core.config import (
    AMDArchitecture,
    HardwareBackend,
    NVIDIAArchitecture,
    TPUVersion,
    TrainiumArchitecture,
)

logger = logging.getLogger(__name__)

Architecture = (
    NVIDIAArchitecture | AMDArchitecture | TrainiumArchitecture | TPUVersion | None
)


class ShardingStrategy(Enum):
    """FSDP sharding strategies."""

    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    HYBRID_SHARD = "hybrid_shard"
    NO_SHARD = "no_shard"


class MixedPrecisionChoice(Enum):
    """Mixed precision configurations for FSDP."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"


# ── Per-backend optimal settings ────────────────────────────────────────────

# Mixed precision: (backend, architecture) → MixedPrecisionChoice
_MIXED_PRECISION_MAP: dict[
    HardwareBackend, dict[Architecture, MixedPrecisionChoice]
] = {
    HardwareBackend.CUDA: {
        NVIDIAArchitecture.BLACKWELL_DC: MixedPrecisionChoice.FP8,
        NVIDIAArchitecture.BLACKWELL_CONSUMER: MixedPrecisionChoice.BF16,
        NVIDIAArchitecture.HOPPER: MixedPrecisionChoice.BF16,
        NVIDIAArchitecture.ADA: MixedPrecisionChoice.BF16,
        NVIDIAArchitecture.AMPERE: MixedPrecisionChoice.BF16,
        NVIDIAArchitecture.TURING: MixedPrecisionChoice.FP16,
        None: MixedPrecisionChoice.BF16,
    },
    HardwareBackend.AMD: {
        AMDArchitecture.CDNA4: MixedPrecisionChoice.BF16,
        AMDArchitecture.CDNA3: MixedPrecisionChoice.BF16,
        AMDArchitecture.CDNA2: MixedPrecisionChoice.BF16,
        None: MixedPrecisionChoice.BF16,
    },
    HardwareBackend.TRAINIUM: {
        TrainiumArchitecture.TRN3: MixedPrecisionChoice.BF16,
        TrainiumArchitecture.TRN2: MixedPrecisionChoice.BF16,
        None: MixedPrecisionChoice.BF16,
    },
    HardwareBackend.TPU: {
        TPUVersion.V7: MixedPrecisionChoice.BF16,
        TPUVersion.V5E: MixedPrecisionChoice.BF16,
        None: MixedPrecisionChoice.BF16,
    },
    HardwareBackend.CPU: {
        None: MixedPrecisionChoice.FP32,
    },
}

# Sharding strategy: multi-node → hybrid shard
_MULTI_NODE_STRATEGY = ShardingStrategy.HYBRID_SHARD

# Float8 all-gather support
_FLOAT8_ALLGATHER_ARCHS: set[NVIDIAArchitecture] = {
    NVIDIAArchitecture.HOPPER,
    NVIDIAArchitecture.BLACKWELL_DC,
    NVIDIAArchitecture.BLACKWELL_CONSUMER,
}


@dataclass
class FSDPConfig:
    """Configuration for FSDP wrapping."""

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    cpu_offload: bool = False
    mixed_precision: MixedPrecisionChoice | None = None  # None = auto
    backward_prefetch: bool = True
    forward_prefetch: bool = False
    float8_all_gather: bool = False  # auto-enable when supported
    reshard_after_forward: bool = True
    limit_all_gathers: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sharding_strategy": self.sharding_strategy.value,
            "cpu_offload": self.cpu_offload,
            "mixed_precision": self.mixed_precision.value if self.mixed_precision else "auto",
            "backward_prefetch": self.backward_prefetch,
            "forward_prefetch": self.forward_prefetch,
            "float8_all_gather": self.float8_all_gather,
            "reshard_after_forward": self.reshard_after_forward,
            "limit_all_gathers": self.limit_all_gathers,
        }


class FSDPManager:
    """Backend-aware FSDP configuration manager and applier.

    Resolves optimal FSDP settings from hardware backend and architecture,
    and can apply them via :meth:`apply`. Directly wraps ``torch.distributed.fsdp.FSDP``
    when ``apply()`` is called in an initialized distributed environment.

    Args:
        config: FSDP configuration (auto fields resolved on init).
        backend: Hardware backend.
        architecture: Specific architecture for fine-grained selection.
        multi_node: Whether the training spans multiple nodes.
    """

    def __init__(
        self,
        config: FSDPConfig | None = None,
        backend: HardwareBackend = HardwareBackend.CPU,
        architecture: Architecture = None,
        multi_node: bool = False,
    ):
        self._config = config or FSDPConfig()
        self._backend = backend
        self._architecture = architecture
        self._multi_node = multi_node

        # Resolve auto settings
        self._resolved = self._resolve(self._config)

    def _resolve(self, config: FSDPConfig) -> FSDPConfig:
        """Resolve auto settings based on hardware."""
        resolved_mp = config.mixed_precision
        if resolved_mp is None:
            resolved_mp = self._get_optimal_mixed_precision()

        resolved_strategy = config.sharding_strategy
        if self._multi_node and resolved_strategy == ShardingStrategy.FULL_SHARD:
            resolved_strategy = _MULTI_NODE_STRATEGY

        resolved_fp8 = config.float8_all_gather
        if not resolved_fp8:
            resolved_fp8 = self._supports_float8_all_gather()

        return FSDPConfig(
            sharding_strategy=resolved_strategy,
            cpu_offload=config.cpu_offload,
            mixed_precision=resolved_mp,
            backward_prefetch=config.backward_prefetch,
            forward_prefetch=config.forward_prefetch,
            float8_all_gather=resolved_fp8,
            reshard_after_forward=config.reshard_after_forward,
            limit_all_gathers=config.limit_all_gathers,
        )

    def _get_optimal_mixed_precision(self) -> MixedPrecisionChoice:
        """Auto-select mixed precision based on backend and architecture."""
        backend_map = _MIXED_PRECISION_MAP.get(self._backend, {})
        if self._architecture in backend_map:
            return backend_map[self._architecture]
        return backend_map.get(None, MixedPrecisionChoice.FP32)

    def _supports_float8_all_gather(self) -> bool:
        """Check if Float8 all-gather is supported."""
        if self._backend != HardwareBackend.CUDA:
            return False
        if isinstance(self._architecture, NVIDIAArchitecture):
            return self._architecture in _FLOAT8_ALLGATHER_ARCHS
        return False

    @property
    def resolved_config(self) -> FSDPConfig:
        """Return the resolved FSDP config with all auto fields filled."""
        return self._resolved

    @property
    def mixed_precision(self) -> MixedPrecisionChoice:
        """Return the resolved mixed precision choice."""
        return self._resolved.mixed_precision or MixedPrecisionChoice.FP32

    @property
    def sharding_strategy(self) -> ShardingStrategy:
        """Return the resolved sharding strategy."""
        return self._resolved.sharding_strategy

    def apply(self, model: nn.Module) -> nn.Module:
        """Wrap ``model`` with FSDP using the resolved backend-aware configuration.

        Requires an initialized ``torch.distributed`` process group.  Call
        ``torch.distributed.init_process_group()`` before invoking this method.

        Args:
            model: PyTorch module to wrap.

        Returns:
            ``torch.distributed.fsdp.FullyShardedDataParallel``-wrapped module.

        Raises:
            RuntimeError: If ``torch.distributed`` is not initialized.
            ImportError: If ``torch.distributed.fsdp`` is not available.
        """
        import torch
        import torch.distributed as dist
        from torch.distributed.fsdp import BackwardPrefetch, CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy as TorchShardingStrategy

        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed is not initialized. "
                "Call torch.distributed.init_process_group() before apply()."
            )

        # Map TorchBridge ShardingStrategy → torch.distributed.fsdp.ShardingStrategy
        _strategy_map = {
            ShardingStrategy.FULL_SHARD: TorchShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP: TorchShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.HYBRID_SHARD: TorchShardingStrategy.HYBRID_SHARD,
            ShardingStrategy.NO_SHARD: TorchShardingStrategy.NO_SHARD,
        }
        torch_strategy = _strategy_map[self._resolved.sharding_strategy]

        # Build MixedPrecision policy
        mp = self._resolved.mixed_precision
        _dtype_map = {
            MixedPrecisionChoice.FP32: torch.float32,
            MixedPrecisionChoice.FP16: torch.float16,
            MixedPrecisionChoice.BF16: torch.bfloat16,
            # FP8 training uses bfloat16 for reduction (FP8 tensors handled by model)
            MixedPrecisionChoice.FP8: torch.bfloat16,
        }
        mp_dtype = _dtype_map.get(mp, torch.float32)
        mixed_precision_policy = MixedPrecision(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
            buffer_dtype=mp_dtype,
        )

        backward_prefetch_setting = (
            BackwardPrefetch.BACKWARD_PRE if self._resolved.backward_prefetch else None
        )

        cpu_offload_obj = (
            CPUOffload(offload_params=True) if self._resolved.cpu_offload else None
        )

        device_id = torch.cuda.current_device() if torch.cuda.is_available() else None

        logger.info(
            "Applying FSDP: strategy=%s, mixed_precision=%s, cpu_offload=%s",
            self._resolved.sharding_strategy.value,
            mp.value if mp else "auto",
            self._resolved.cpu_offload,
        )

        return FSDP(
            model,
            sharding_strategy=torch_strategy,
            mixed_precision=mixed_precision_policy,
            backward_prefetch=backward_prefetch_setting,
            cpu_offload=cpu_offload_obj,
            device_id=device_id,
            limit_all_gathers=self._resolved.limit_all_gathers,
        )

    def get_info(self) -> dict[str, Any]:
        """Return diagnostic info about the FSDP configuration."""
        return {
            "backend": self._backend.value,
            "architecture": (
                self._architecture.value if self._architecture else None
            ),
            "multi_node": self._multi_node,
            "resolved_config": self._resolved.to_dict(),
            "float8_all_gather_supported": self._supports_float8_all_gather(),
        }
