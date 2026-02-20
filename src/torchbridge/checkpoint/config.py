"""
Checkpoint Configuration

Defines checkpoint storage backend, serialization format, and async/caching
options for the CheckpointManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class StorageBackendType(Enum):
    """Storage backend for checkpoint persistence."""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


class SerializationFormat(Enum):
    """Tensor serialization format for checkpoint data."""

    TORCH = "torch"
    SAFETENSORS = "safetensors"


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint saving and loading.

    Controls storage backend, async behavior, plan caching, cross-backend
    portability, and checkpoint rotation.

    Attributes:
        storage_backend: Where to persist checkpoints.
        storage_path: Base path or URI for checkpoint storage.
        async_save: Use async saving to overlap with training.
        process_based_async: Use process-based async (eliminates GIL contention).
        plan_caching: Cache DCP save plans for faster subsequent saves.
        pinned_memory_staging: Use pinned memory for GPU→CPU staging.
        serialization_format: Tensor serialization format.
        max_checkpoints_to_keep: Auto-delete oldest checkpoints (0 = unlimited).
        normalize_on_save: Normalize dtypes/devices for cross-backend portability.
        include_metadata: Write hardware provenance metadata sidecar.
        io_thread_count: Number of I/O threads for parallel writes.
    """

    storage_backend: StorageBackendType = StorageBackendType.LOCAL
    storage_path: str = "./checkpoints"
    async_save: bool = True
    process_based_async: bool = True
    plan_caching: bool = True
    pinned_memory_staging: bool = True
    serialization_format: SerializationFormat = SerializationFormat.TORCH
    max_checkpoints_to_keep: int = 3
    normalize_on_save: bool = True
    include_metadata: bool = True
    io_thread_count: int = 1

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if self.max_checkpoints_to_keep < 0:
            raise ValueError(
                f"max_checkpoints_to_keep must be >= 0, got "
                f"{self.max_checkpoints_to_keep}"
            )
        if self.io_thread_count < 1:
            raise ValueError(
                f"io_thread_count must be >= 1, got {self.io_thread_count}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "storage_backend": self.storage_backend.value,
            "storage_path": self.storage_path,
            "async_save": self.async_save,
            "process_based_async": self.process_based_async,
            "plan_caching": self.plan_caching,
            "pinned_memory_staging": self.pinned_memory_staging,
            "serialization_format": self.serialization_format.value,
            "max_checkpoints_to_keep": self.max_checkpoints_to_keep,
            "normalize_on_save": self.normalize_on_save,
            "include_metadata": self.include_metadata,
            "io_thread_count": self.io_thread_count,
        }
