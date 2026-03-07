"""
Checkpoint Manager

Thin wrapper around PyTorch Distributed Checkpoint (DCP) with cross-backend
metadata and checkpoint rotation. TorchBridge adds backend-portable metadata
(dtype normalization, backend tagging) and rotation logic; the underlying
serialization is PyTorch DCP.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from torchbridge.checkpoint.config import CheckpointConfig, StorageBackendType
from torchbridge.checkpoint.metadata import (
    CheckpointMetadata,
    PortabilityNormalizer,
)
from torchbridge.checkpoint.storage import StorageBackendFactory
from torchbridge.core.config import HardwareBackend

logger = logging.getLogger(__name__)

_Architecture = Any  # NVIDIAArchitecture | AMDArchitecture | ... | None

_METADATA_FILENAME = "torchbridge_metadata.json"


class CheckpointManager:
    """Backend-aware checkpoint manager wrapping PyTorch DCP.

    Handles async saving, plan caching, storage backend selection,
    cross-backend portability, and checkpoint rotation.

    Args:
        config: Checkpoint configuration.
        backend: Source hardware backend for saves, target for loads.
        architecture: Specific architecture (for metadata and dtype selection).
    """

    def __init__(
        self,
        config: CheckpointConfig | None = None,
        backend: HardwareBackend = HardwareBackend.CPU,
        architecture: _Architecture = None,
    ):
        self._config = config or CheckpointConfig()
        self._backend = backend
        self._architecture = architecture
        self._save_count = 0
        self._planner: Any = None  # Lazy-init DCP planner with caching

    def save(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str | None = None,
        model_params: int | None = None,
    ) -> str:
        """Save a checkpoint.

        Steps:
        1. Generate checkpoint_id if not provided.
        2. Normalize state_dict for cross-backend portability (if enabled).
        3. Save via DCP (async or sync based on config).
        4. Write metadata sidecar file.
        5. Rotate old checkpoints if limit exceeded.

        Args:
            state_dict: Model/optimizer state dict to save.
            checkpoint_id: Explicit checkpoint path/URI. Auto-generated if None.
            model_params: Total model parameters (for metadata).

        Returns:
            The checkpoint_id used for saving.
        """
        if checkpoint_id is None:
            checkpoint_id = self._generate_checkpoint_id()

        logger.info("Saving checkpoint to %s", checkpoint_id)

        # Normalize for portability
        dtype_map: dict[str, str] = {}
        device_map: dict[str, str] = {}
        save_dict = state_dict

        if self._config.normalize_on_save:
            save_dict, dtype_map, device_map = (
                PortabilityNormalizer.normalize_state_dict(
                    state_dict, self._backend, self._architecture
                )
            )

        # Save via DCP
        self._dcp_save(save_dict, checkpoint_id)
        self._save_count += 1

        # Write metadata sidecar
        if self._config.include_metadata:
            metadata = self._build_metadata(
                checkpoint_id, model_params, dtype_map, device_map
            )
            metadata_path = os.path.join(checkpoint_id, _METADATA_FILENAME)
            metadata.save(metadata_path)
            logger.debug("Wrote metadata to %s", metadata_path)

        # Rotate old checkpoints
        if self._config.max_checkpoints_to_keep > 0:
            self._rotate_checkpoints()

        return checkpoint_id

    def load(
        self,
        state_dict: dict[str, Any],
        checkpoint_id: str,
        target_backend: HardwareBackend | None = None,
        target_device: torch.device | None = None,
    ) -> CheckpointMetadata | None:
        """Load a checkpoint with optional cross-backend restoration.

        Steps:
        1. Read metadata sidecar (if available).
        2. Load via DCP (handles resharding automatically).
        3. Restore dtypes/devices for target backend if different from source.

        Args:
            state_dict: Pre-allocated state dict to load into (in-place).
            checkpoint_id: Path or URI of the checkpoint.
            target_backend: Target backend for dtype restoration. Uses
                self._backend if None.
            target_device: Target device. Uses CPU if None.

        Returns:
            CheckpointMetadata from the loaded checkpoint, or None if
            no metadata sidecar was found.
        """
        logger.info("Loading checkpoint from %s", checkpoint_id)

        if target_backend is None:
            target_backend = self._backend
        if target_device is None:
            target_device = torch.device("cpu")

        # Load via DCP
        self._dcp_load(state_dict, checkpoint_id)

        # Read metadata
        metadata = self._read_metadata(checkpoint_id)

        # Cross-backend restoration
        if metadata and metadata.dtype_map:
            restored = PortabilityNormalizer.restore_state_dict(
                state_dict, metadata, target_backend, target_device
            )
            # Update state_dict in-place
            state_dict.update(restored)

        return metadata

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List available checkpoints with metadata.

        Scans the storage path for checkpoint directories containing
        metadata sidecar files.

        Returns:
            List of CheckpointMetadata, sorted by timestamp (newest first).
        """
        if self._config.storage_backend != StorageBackendType.LOCAL:
            logger.warning(
                "list_checkpoints only supports LOCAL storage backend"
            )
            return []

        base = Path(self._config.storage_path)
        if not base.exists():
            return []

        results: list[CheckpointMetadata] = []
        for entry in base.iterdir():
            if not entry.is_dir():
                continue
            meta_path = entry / _METADATA_FILENAME
            if meta_path.exists():
                try:
                    results.append(CheckpointMetadata.load(meta_path))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(
                        "Skipping malformed metadata in %s: %s",
                        meta_path,
                        exc,
                    )

        results.sort(key=lambda m: m.timestamp, reverse=True)
        return results

    def get_info(self) -> dict[str, Any]:
        """Return diagnostic info about the checkpoint configuration."""
        return {
            "backend": self._backend.value,
            "architecture": (
                self._architecture.value if self._architecture else None
            ),
            "config": self._config.to_dict(),
            "save_count": self._save_count,
            "storage_info": StorageBackendFactory.get_writer_info(self._config),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _dcp_save(
        self, state_dict: dict[str, Any], checkpoint_id: str
    ) -> None:
        """Save via PyTorch DCP (async or sync)."""
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError:
            # Fallback: simple torch.save for non-distributed environments
            self._fallback_save(state_dict, checkpoint_id)
            return

        writer = StorageBackendFactory.create_writer(self._config)
        planner = self._get_or_create_planner()

        if self._config.async_save:
            try:
                response = dcp.async_save(
                    state_dict,
                    checkpoint_id=checkpoint_id,
                    storage_writer=writer,
                    planner=planner,
                )
                # Wait for staging to complete (training can resume after this)
                response.result()
                logger.debug("Async save staged for %s", checkpoint_id)
            except (AttributeError, TypeError):
                # async_save not available in this PyTorch version
                logger.debug(
                    "async_save unavailable, falling back to sync save"
                )
                dcp.save(
                    state_dict,
                    checkpoint_id=checkpoint_id,
                    storage_writer=writer,
                    planner=planner,
                )
        else:
            dcp.save(
                state_dict,
                checkpoint_id=checkpoint_id,
                storage_writer=writer,
                planner=planner,
            )

    def _dcp_load(
        self, state_dict: dict[str, Any], checkpoint_id: str
    ) -> None:
        """Load via PyTorch DCP."""
        try:
            import torch.distributed.checkpoint as dcp
        except ImportError:
            self._fallback_load(state_dict, checkpoint_id)
            return

        reader = StorageBackendFactory.create_reader(
            self._config, checkpoint_id
        )
        dcp.load(
            state_dict,
            checkpoint_id=checkpoint_id,
            storage_reader=reader,
        )

    def _fallback_save(
        self, state_dict: dict[str, Any], checkpoint_id: str
    ) -> None:
        """Fallback save using torch.save for single-rank environments."""
        path = Path(checkpoint_id)
        path.mkdir(parents=True, exist_ok=True)
        save_path = path / "checkpoint.pt"
        torch.save(state_dict, save_path)
        logger.info("Saved checkpoint via torch.save to %s", save_path)

    def _fallback_load(
        self, state_dict: dict[str, Any], checkpoint_id: str
    ) -> None:
        """Fallback load using torch.load for single-rank environments."""
        save_path = Path(checkpoint_id) / "checkpoint.pt"
        loaded = torch.load(save_path, weights_only=True)
        state_dict.update(loaded)
        logger.info("Loaded checkpoint via torch.load from %s", save_path)

    def _get_or_create_planner(self) -> Any:
        """Get or create a DCP SavePlanner with optional plan caching."""
        if self._planner is not None:
            return self._planner

        try:
            from torch.distributed.checkpoint.default_planner import (
                DefaultSavePlanner,
            )

            if self._config.plan_caching:
                # Plan caching support varies by PyTorch version
                try:
                    self._planner = DefaultSavePlanner(
                        enable_plan_caching=True
                    )
                    logger.debug("DCP plan caching enabled")
                except TypeError:
                    self._planner = DefaultSavePlanner()
                    logger.debug(
                        "DCP plan caching not supported in this PyTorch version"
                    )
            else:
                self._planner = DefaultSavePlanner()
        except ImportError:
            self._planner = None

        return self._planner

    def _build_metadata(
        self,
        checkpoint_id: str,
        model_params: int | None,
        dtype_map: dict[str, str],
        device_map: dict[str, str],
    ) -> CheckpointMetadata:
        """Build checkpoint metadata with hardware provenance."""
        from torchbridge import __version__

        return CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            torchbridge_version=__version__,
            pytorch_version=torch.__version__,
            backend=self._backend.value,
            architecture=(
                self._architecture.value if self._architecture else None
            ),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            local_world_size=int(os.environ.get("LOCAL_WORLD_SIZE", "1")),
            model_params=model_params,
            dtype_map=dtype_map,
            device_map=device_map,
        )

    def _read_metadata(self, checkpoint_id: str) -> CheckpointMetadata | None:
        """Read metadata sidecar file if it exists."""
        meta_path = Path(checkpoint_id) / _METADATA_FILENAME
        if not meta_path.exists():
            logger.debug("No metadata sidecar found at %s", meta_path)
            return None
        try:
            return CheckpointMetadata.load(meta_path)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse metadata at %s: %s", meta_path, exc)
            return None

    def _rotate_checkpoints(self) -> None:
        """Delete oldest checkpoints exceeding max_checkpoints_to_keep."""
        if self._config.storage_backend != StorageBackendType.LOCAL:
            return  # Rotation only supported for local storage

        checkpoints = self.list_checkpoints()
        max_keep = self._config.max_checkpoints_to_keep

        if len(checkpoints) <= max_keep:
            return

        # Delete oldest (list is sorted newest-first)
        for old in checkpoints[max_keep:]:
            old_path = Path(old.checkpoint_id)
            if old_path.exists() and old_path.is_dir():
                shutil.rmtree(old_path)
                logger.info(
                    "Rotated checkpoint %s (keeping %d most recent)",
                    old.checkpoint_id,
                    max_keep,
                )

    def _generate_checkpoint_id(self) -> str:
        """Generate a timestamped checkpoint ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = f"checkpoint_{timestamp}_{self._save_count:04d}"
        return os.path.join(self._config.storage_path, name)
