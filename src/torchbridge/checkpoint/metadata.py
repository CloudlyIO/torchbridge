"""
Checkpoint Metadata & Cross-Backend Portability

Records hardware provenance for each checkpoint and normalizes tensor
dtypes/devices so checkpoints saved on one backend can be loaded on another.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from torchbridge.core.config import HardwareBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointMetadata:
    """Hardware provenance and portability metadata for a checkpoint.

    Written as a JSON sidecar file alongside the checkpoint data.
    """

    checkpoint_id: str
    timestamp: str
    torchbridge_version: str
    pytorch_version: str
    backend: str
    architecture: str | None
    world_size: int
    local_world_size: int
    model_params: int | None = None
    dtype_map: dict[str, str] = field(default_factory=dict)
    device_map: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "torchbridge_version": self.torchbridge_version,
            "pytorch_version": self.pytorch_version,
            "backend": self.backend,
            "architecture": self.architecture,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "model_params": self.model_params,
            "dtype_map": self.dtype_map,
            "device_map": self.device_map,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        """Deserialize from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            timestamp=data["timestamp"],
            torchbridge_version=data["torchbridge_version"],
            pytorch_version=data["pytorch_version"],
            backend=data["backend"],
            architecture=data.get("architecture"),
            world_size=data["world_size"],
            local_world_size=data.get("local_world_size", 1),
            model_params=data.get("model_params"),
            dtype_map=data.get("dtype_map", {}),
            device_map=data.get("device_map", {}),
        )

    def save(self, path: str | Path) -> None:
        """Write metadata to a JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CheckpointMetadata:
        """Read metadata from a JSON file."""
        filepath = Path(path)
        data = json.loads(filepath.read_text())
        return cls.from_dict(data)


# ── Dtype normalization for cross-backend portability ────────────────────────

# FP8 types exist only on specific hardware. Normalize to portable dtypes.
_PORTABLE_DTYPE_MAP: dict[torch.dtype, torch.dtype] = {
    torch.float8_e4m3fn: torch.float16,
    torch.float8_e5m2: torch.float16,
}

# String↔dtype mapping for serialization
_DTYPE_STR_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch.dtype to a string key."""
    # torch.float32 → "torch.float32" → "float32"
    return str(dtype).replace("torch.", "")


def _str_to_dtype(s: str) -> torch.dtype:
    """Convert a string key back to torch.dtype."""
    return _DTYPE_STR_MAP.get(s, torch.float32)


class PortabilityNormalizer:
    """Normalize state dicts for cross-backend checkpoint portability.

    On save: moves all tensors to CPU and converts non-portable dtypes
    (e.g., FP8) to portable equivalents (FP16/BF16). Records the original
    dtype and device in a metadata map so they can be restored on load.

    On load: restores tensors to the target backend's device and converts
    back to the original (or target-appropriate) dtypes.
    """

    @staticmethod
    def normalize_state_dict(
        state_dict: dict[str, Any],
        backend: HardwareBackend,
        architecture: Any = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
        """Normalize a state_dict for portable saving.

        Args:
            state_dict: Model or optimizer state dict.
            backend: Source hardware backend.
            architecture: Source architecture (for metadata only).

        Returns:
            Tuple of (normalized_state_dict, dtype_map, device_map).
            dtype_map records original dtypes for each tensor key.
            device_map records original devices for each tensor key.
        """
        normalized: dict[str, Any] = {}
        dtype_map: dict[str, str] = {}
        device_map: dict[str, str] = {}

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                dtype_map[key] = _dtype_to_str(value.dtype)
                device_map[key] = str(value.device)

                # Move to CPU
                tensor = value.cpu()

                # Normalize non-portable dtypes
                portable_dtype = _PORTABLE_DTYPE_MAP.get(tensor.dtype)
                if portable_dtype is not None:
                    logger.debug(
                        "Normalizing %s: %s → %s",
                        key,
                        tensor.dtype,
                        portable_dtype,
                    )
                    tensor = tensor.to(portable_dtype)

                normalized[key] = tensor
            elif isinstance(value, dict):
                # Recurse into nested dicts (optimizer state groups)
                nested, nested_dtypes, nested_devices = (
                    PortabilityNormalizer.normalize_state_dict(
                        value, backend, architecture
                    )
                )
                normalized[key] = nested
                for nk, nv in nested_dtypes.items():
                    dtype_map[f"{key}.{nk}"] = nv
                for nk, nv in nested_devices.items():
                    device_map[f"{key}.{nk}"] = nv
            else:
                normalized[key] = value

        return normalized, dtype_map, device_map

    @staticmethod
    def restore_state_dict(
        state_dict: dict[str, Any],
        metadata: CheckpointMetadata,
        target_backend: HardwareBackend,
        target_device: torch.device,
    ) -> dict[str, Any]:
        """Restore a normalized state_dict to a target backend.

        Moves tensors to the target device and optionally restores
        original dtypes if supported by the target backend.

        Args:
            state_dict: Normalized state dict (CPU tensors).
            metadata: Checkpoint metadata with dtype/device maps.
            target_backend: Target hardware backend.
            target_device: Target device (e.g., cuda:0, cpu).

        Returns:
            Restored state dict with tensors on target device.
        """
        restored: dict[str, Any] = {}

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                tensor = value

                # Restore original dtype if target supports it
                original_dtype_str = metadata.dtype_map.get(key)
                if original_dtype_str:
                    original_dtype = _str_to_dtype(original_dtype_str)
                    if _backend_supports_dtype(target_backend, original_dtype):
                        tensor = tensor.to(original_dtype)

                # Move to target device
                tensor = tensor.to(target_device)
                restored[key] = tensor
            elif isinstance(value, dict):
                # Build sub-metadata for nested dict
                sub_meta = CheckpointMetadata(
                    checkpoint_id=metadata.checkpoint_id,
                    timestamp=metadata.timestamp,
                    torchbridge_version=metadata.torchbridge_version,
                    pytorch_version=metadata.pytorch_version,
                    backend=metadata.backend,
                    architecture=metadata.architecture,
                    world_size=metadata.world_size,
                    local_world_size=metadata.local_world_size,
                    dtype_map={
                        k.removeprefix(f"{key}."): v
                        for k, v in metadata.dtype_map.items()
                        if k.startswith(f"{key}.")
                    },
                    device_map={
                        k.removeprefix(f"{key}."): v
                        for k, v in metadata.device_map.items()
                        if k.startswith(f"{key}.")
                    },
                )
                restored[key] = PortabilityNormalizer.restore_state_dict(
                    value, sub_meta, target_backend, target_device
                )
            else:
                restored[key] = value

        return restored


# ── Backend dtype support ────────────────────────────────────────────────────

# FP8 dtypes are only natively supported on specific backends
_FP8_BACKENDS: set[HardwareBackend] = {
    HardwareBackend.CUDA,
    HardwareBackend.AMD,
}


def _backend_supports_dtype(
    backend: HardwareBackend, dtype: torch.dtype
) -> bool:
    """Check if a backend supports the given dtype."""
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return backend in _FP8_BACKENDS
    # All standard dtypes are universally supported
    return True
