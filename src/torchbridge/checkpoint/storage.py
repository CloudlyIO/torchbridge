"""
Storage Backend Factory

Creates PyTorch DCP StorageWriter/StorageReader instances from
CheckpointConfig, with support for local filesystem and cloud
object stores (S3, GCS, Azure) via fsspec.
"""

from __future__ import annotations

import logging
from typing import Any

from torchbridge.checkpoint.config import (
    CheckpointConfig,
    StorageBackendType,
)

logger = logging.getLogger(__name__)


class StorageBackendFactory:
    """Create DCP StorageWriter/StorageReader from CheckpointConfig.

    LOCAL backend uses FileSystemWriter/FileSystemReader directly.
    Cloud backends (S3, GCS, Azure) use FsspecWriter/FsspecReader with
    the appropriate fsspec filesystem, imported lazily.
    """

    @staticmethod
    def create_writer(config: CheckpointConfig) -> Any:
        """Create a DCP-compatible StorageWriter.

        Args:
            config: Checkpoint configuration.

        Returns:
            A StorageWriter instance (FileSystemWriter or FsspecWriter).

        Raises:
            ImportError: If required cloud storage package is not installed.
            RuntimeError: If torch.distributed.checkpoint is unavailable.
        """
        dcp_fs = _import_dcp_filesystem()

        if config.storage_backend == StorageBackendType.LOCAL:
            kwargs: dict[str, Any] = {
                "path": config.storage_path,
                "single_file_per_rank": True,
                "sync_files": True,
                "thread_count": config.io_thread_count,
                "overwrite": True,
            }
            if config.pinned_memory_staging:
                kwargs["cache_staged_state_dict"] = True
            return dcp_fs.FileSystemWriter(**kwargs)

        # Cloud backends use fsspec
        filesystem = _get_fsspec_filesystem(config.storage_backend)
        fsspec_mod = _import_dcp_fsspec()
        return fsspec_mod.FsspecWriter(
            path=config.storage_path,
            storage_options={"filesystem": filesystem},
        )

    @staticmethod
    def create_reader(config: CheckpointConfig, checkpoint_id: str) -> Any:
        """Create a DCP-compatible StorageReader.

        Args:
            config: Checkpoint configuration.
            checkpoint_id: Path or URI of the checkpoint to read.

        Returns:
            A StorageReader instance (FileSystemReader or FsspecReader).

        Raises:
            ImportError: If required cloud storage package is not installed.
            RuntimeError: If torch.distributed.checkpoint is unavailable.
        """
        dcp_fs = _import_dcp_filesystem()

        if config.storage_backend == StorageBackendType.LOCAL:
            return dcp_fs.FileSystemReader(checkpoint_id)

        filesystem = _get_fsspec_filesystem(config.storage_backend)
        fsspec_mod = _import_dcp_fsspec()
        return fsspec_mod.FsspecReader(
            path=checkpoint_id,
            storage_options={"filesystem": filesystem},
        )

    @staticmethod
    def get_writer_info(config: CheckpointConfig) -> dict[str, Any]:
        """Return diagnostic info about the storage configuration."""
        return {
            "backend": config.storage_backend.value,
            "path": config.storage_path,
            "async_save": config.async_save,
            "pinned_memory_staging": config.pinned_memory_staging,
            "io_thread_count": config.io_thread_count,
            "serialization_format": config.serialization_format.value,
        }


# ── Internal helpers ─────────────────────────────────────────────────────────


def _import_dcp_filesystem() -> Any:
    """Import torch.distributed.checkpoint.filesystem."""
    try:
        import torch.distributed.checkpoint.filesystem as dcp_fs

        return dcp_fs
    except ImportError as e:
        raise RuntimeError(
            "torch.distributed.checkpoint is required for checkpoint management. "
            "Ensure PyTorch >= 2.0 is installed."
        ) from e


def _import_dcp_fsspec() -> Any:
    """Import torch.distributed.checkpoint._fsspec_filesystem."""
    try:
        import torch.distributed.checkpoint._fsspec_filesystem as fsspec_mod

        return fsspec_mod
    except ImportError as e:
        raise RuntimeError(
            "FsspecWriter/FsspecReader requires PyTorch >= 2.3 with fsspec support."
        ) from e


_FSSPEC_PACKAGES: dict[StorageBackendType, tuple[str, str]] = {
    StorageBackendType.S3: ("s3fs", "s3fs"),
    StorageBackendType.GCS: ("gcsfs", "gcsfs"),
    StorageBackendType.AZURE: ("adlfs", "adlfs"),
}


def _get_fsspec_filesystem(backend: StorageBackendType) -> Any:
    """Lazy-import and instantiate the fsspec filesystem for a cloud backend.

    Raises:
        ImportError: If the required fsspec package is not installed.
        ValueError: If the backend is not a cloud storage type.
    """
    if backend not in _FSSPEC_PACKAGES:
        raise ValueError(
            f"No fsspec filesystem for backend {backend.value}. "
            f"Supported cloud backends: {', '.join(b.value for b in _FSSPEC_PACKAGES)}"
        )

    package_name, import_name = _FSSPEC_PACKAGES[backend]

    try:
        import importlib

        mod = importlib.import_module(import_name)
    except ImportError as e:
        raise ImportError(
            f"Cloud storage backend '{backend.value}' requires the "
            f"'{package_name}' package. Install it with: "
            f"pip install torchbridge-ml[checkpoint]"
        ) from e

    # Each fsspec package provides a filesystem class
    filesystem_classes: dict[StorageBackendType, str] = {
        StorageBackendType.S3: "S3FileSystem",
        StorageBackendType.GCS: "GCSFileSystem",
        StorageBackendType.AZURE: "AzureBlobFileSystem",
    }

    cls_name = filesystem_classes[backend]
    fs_cls = getattr(mod, cls_name)
    logger.info("Using %s filesystem for checkpoint storage", cls_name)
    return fs_cls()
