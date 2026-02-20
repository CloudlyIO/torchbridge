"""Tests for CheckpointManager."""

import json
from pathlib import Path
from unittest.mock import patch

import torch

from torchbridge.checkpoint.config import CheckpointConfig
from torchbridge.checkpoint.manager import _METADATA_FILENAME, CheckpointManager
from torchbridge.checkpoint.metadata import CheckpointMetadata
from torchbridge.core.config import HardwareBackend


class TestCheckpointManagerInit:
    """Tests for CheckpointManager initialization."""

    def test_default_init(self):
        mgr = CheckpointManager()
        assert mgr._backend == HardwareBackend.CPU
        assert mgr._architecture is None
        assert mgr._save_count == 0

    def test_custom_init(self):
        config = CheckpointConfig(storage_path="/tmp/ckpts")
        mgr = CheckpointManager(
            config=config,
            backend=HardwareBackend.CUDA,
        )
        assert mgr._config.storage_path == "/tmp/ckpts"
        assert mgr._backend == HardwareBackend.CUDA

    def test_get_info(self):
        mgr = CheckpointManager(backend=HardwareBackend.AMD)
        info = mgr.get_info()
        assert info["backend"] == "amd"
        assert info["save_count"] == 0
        assert "config" in info
        assert "storage_info" in info


class TestCheckpointManagerSave:
    """Tests for CheckpointManager.save()."""

    def test_save_generates_checkpoint_id(self, tmp_path):
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            normalize_on_save=False,
            include_metadata=False,
            max_checkpoints_to_keep=0,
        )
        mgr = CheckpointManager(config=config)

        with patch.object(mgr, "_dcp_save") as mock_save:
            ckpt_id = mgr.save({"weight": torch.randn(2, 2)})

        assert ckpt_id.startswith(str(tmp_path))
        assert "checkpoint_" in ckpt_id
        mock_save.assert_called_once()

    def test_save_uses_explicit_id(self, tmp_path):
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            normalize_on_save=False,
            include_metadata=False,
            max_checkpoints_to_keep=0,
        )
        mgr = CheckpointManager(config=config)
        explicit_id = str(tmp_path / "my_ckpt")

        with patch.object(mgr, "_dcp_save"):
            result_id = mgr.save({"w": torch.randn(2)}, checkpoint_id=explicit_id)

        assert result_id == explicit_id

    def test_save_increments_count(self, tmp_path):
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            normalize_on_save=False,
            include_metadata=False,
            max_checkpoints_to_keep=0,
        )
        mgr = CheckpointManager(config=config)
        assert mgr._save_count == 0

        with patch.object(mgr, "_dcp_save"):
            mgr.save({"w": torch.randn(2)})

        assert mgr._save_count == 1

    def test_save_writes_metadata(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt_001"
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            normalize_on_save=False,
            include_metadata=True,
            max_checkpoints_to_keep=0,
        )
        mgr = CheckpointManager(config=config, backend=HardwareBackend.CUDA)

        with patch.object(mgr, "_dcp_save"):
            mgr.save(
                {"w": torch.randn(2)},
                checkpoint_id=str(ckpt_dir),
                model_params=int(7e9),
            )

        meta_path = ckpt_dir / _METADATA_FILENAME
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["backend"] == "cuda"
        assert data["model_params"] == int(7e9)

    def test_save_normalizes_state_dict(self, tmp_path):
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            normalize_on_save=True,
            include_metadata=True,
            max_checkpoints_to_keep=0,
        )
        mgr = CheckpointManager(config=config)

        original = torch.randn(4, 4)
        state = {"weight": original.clone()}

        with patch.object(mgr, "_dcp_save") as mock_save:
            mgr.save(state, checkpoint_id=str(tmp_path / "ckpt"))

        # The normalized dict should have been passed to _dcp_save
        saved_dict = mock_save.call_args[0][0]
        assert saved_dict["weight"].device == torch.device("cpu")


class TestCheckpointManagerLoad:
    """Tests for CheckpointManager.load()."""

    def test_load_returns_metadata(self, tmp_path):
        # Create metadata file
        ckpt_dir = tmp_path / "ckpt_001"
        ckpt_dir.mkdir()
        meta = CheckpointMetadata(
            checkpoint_id=str(ckpt_dir),
            timestamp="2026-02-19T12:00:00+00:00",
            torchbridge_version="0.5.28",
            pytorch_version="2.7.0",
            backend="cuda",
            architecture="hopper",
            world_size=8,
            local_world_size=8,
        )
        meta.save(ckpt_dir / _METADATA_FILENAME)

        config = CheckpointConfig(normalize_on_save=False)
        mgr = CheckpointManager(config=config)

        state = {"w": torch.randn(2)}
        with patch.object(mgr, "_dcp_load"):
            loaded_meta = mgr.load(state, str(ckpt_dir))

        assert loaded_meta is not None
        assert loaded_meta.backend == "cuda"
        assert loaded_meta.world_size == 8

    def test_load_returns_none_without_metadata(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt_no_meta"
        ckpt_dir.mkdir()

        config = CheckpointConfig(normalize_on_save=False)
        mgr = CheckpointManager(config=config)

        state = {"w": torch.randn(2)}
        with patch.object(mgr, "_dcp_load"):
            loaded_meta = mgr.load(state, str(ckpt_dir))

        assert loaded_meta is None


class TestCheckpointManagerList:
    """Tests for CheckpointManager.list_checkpoints()."""

    def test_list_empty_directory(self, tmp_path):
        config = CheckpointConfig(storage_path=str(tmp_path))
        mgr = CheckpointManager(config=config)
        assert mgr.list_checkpoints() == []

    def test_list_nonexistent_directory(self, tmp_path):
        config = CheckpointConfig(storage_path=str(tmp_path / "nonexistent"))
        mgr = CheckpointManager(config=config)
        assert mgr.list_checkpoints() == []

    def test_list_finds_checkpoints(self, tmp_path):
        for i, ts in enumerate(["2026-02-18", "2026-02-19"]):
            ckpt_dir = tmp_path / f"ckpt_{i}"
            ckpt_dir.mkdir()
            meta = CheckpointMetadata(
                checkpoint_id=str(ckpt_dir),
                timestamp=f"{ts}T12:00:00+00:00",
                torchbridge_version="0.5.28",
                pytorch_version="2.7.0",
                backend="cuda",
                architecture=None,
                world_size=1,
                local_world_size=1,
            )
            meta.save(ckpt_dir / _METADATA_FILENAME)

        config = CheckpointConfig(storage_path=str(tmp_path))
        mgr = CheckpointManager(config=config)
        results = mgr.list_checkpoints()

        assert len(results) == 2
        # Should be sorted newest first
        assert "2026-02-19" in results[0].timestamp

    def test_list_skips_malformed_metadata(self, tmp_path):
        # Valid checkpoint
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        meta = CheckpointMetadata(
            checkpoint_id=str(valid_dir),
            timestamp="2026-02-19T12:00:00+00:00",
            torchbridge_version="0.5.28",
            pytorch_version="2.7.0",
            backend="cuda",
            architecture=None,
            world_size=1,
            local_world_size=1,
        )
        meta.save(valid_dir / _METADATA_FILENAME)

        # Malformed checkpoint
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / _METADATA_FILENAME).write_text("not json")

        config = CheckpointConfig(storage_path=str(tmp_path))
        mgr = CheckpointManager(config=config)
        results = mgr.list_checkpoints()

        assert len(results) == 1


class TestCheckpointRotation:
    """Tests for checkpoint rotation."""

    def test_rotation_deletes_oldest(self, tmp_path):
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            max_checkpoints_to_keep=2,
        )
        mgr = CheckpointManager(config=config)

        # Create 3 checkpoint directories with metadata
        for i in range(3):
            ckpt_dir = tmp_path / f"ckpt_{i:04d}"
            ckpt_dir.mkdir()
            meta = CheckpointMetadata(
                checkpoint_id=str(ckpt_dir),
                timestamp=f"2026-02-{17+i:02d}T12:00:00+00:00",
                torchbridge_version="0.5.28",
                pytorch_version="2.7.0",
                backend="cpu",
                architecture=None,
                world_size=1,
                local_world_size=1,
            )
            meta.save(ckpt_dir / _METADATA_FILENAME)

        mgr._rotate_checkpoints()

        remaining = mgr.list_checkpoints()
        assert len(remaining) == 2
        # Oldest should be deleted
        assert not (tmp_path / "ckpt_0000").exists()

    def test_no_rotation_when_under_limit(self, tmp_path):
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            max_checkpoints_to_keep=5,
        )
        mgr = CheckpointManager(config=config)

        ckpt_dir = tmp_path / "ckpt_0000"
        ckpt_dir.mkdir()
        meta = CheckpointMetadata(
            checkpoint_id=str(ckpt_dir),
            timestamp="2026-02-19T12:00:00+00:00",
            torchbridge_version="0.5.28",
            pytorch_version="2.7.0",
            backend="cpu",
            architecture=None,
            world_size=1,
            local_world_size=1,
        )
        meta.save(ckpt_dir / _METADATA_FILENAME)

        mgr._rotate_checkpoints()
        assert ckpt_dir.exists()


class TestFallbackSaveLoad:
    """Tests for torch.save/load fallback when DCP unavailable."""

    def test_fallback_save_creates_file(self, tmp_path):
        mgr = CheckpointManager()
        ckpt_dir = str(tmp_path / "ckpt")
        state = {"weight": torch.randn(4, 4), "bias": torch.randn(4)}

        mgr._fallback_save(state, ckpt_dir)

        save_path = Path(ckpt_dir) / "checkpoint.pt"
        assert save_path.exists()

    def test_fallback_load_restores_data(self, tmp_path):
        mgr = CheckpointManager()
        ckpt_dir = str(tmp_path / "ckpt")
        original = torch.randn(4, 4)
        state = {"weight": original.clone()}

        mgr._fallback_save(state, ckpt_dir)

        loaded_state: dict = {}
        mgr._fallback_load(loaded_state, ckpt_dir)

        assert "weight" in loaded_state
        torch.testing.assert_close(loaded_state["weight"], original)
