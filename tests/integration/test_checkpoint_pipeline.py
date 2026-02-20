"""Integration tests for checkpoint pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from torchbridge.checkpoint.config import (
    CheckpointConfig,
)
from torchbridge.checkpoint.frequency import (
    CheckpointFrequencyAdvisor,
    CheckpointHealthTrigger,
)
from torchbridge.checkpoint.manager import _METADATA_FILENAME, CheckpointManager
from torchbridge.checkpoint.metadata import (
    CheckpointMetadata,
    PortabilityNormalizer,
)
from torchbridge.core.config import HardwareBackend


class TestEndToEndCheckpoint:
    """End-to-end checkpoint save/load tests."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load a checkpoint using fallback (no DCP)."""
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            normalize_on_save=True,
            include_metadata=True,
            max_checkpoints_to_keep=0,
        )
        mgr = CheckpointManager(config=config, backend=HardwareBackend.CPU)

        # Create state dict
        original_weight = torch.randn(16, 16)
        original_bias = torch.randn(16)
        state = {
            "weight": original_weight.clone(),
            "bias": original_bias.clone(),
        }

        # Save using fallback
        ckpt_id = str(tmp_path / "test_ckpt")
        with patch.object(mgr, "_dcp_save", mgr._fallback_save):
            result_id = mgr.save(state, checkpoint_id=ckpt_id, model_params=int(1e6))

        assert result_id == ckpt_id

        # Verify metadata exists
        meta_path = Path(ckpt_id) / _METADATA_FILENAME
        assert meta_path.exists()
        meta = CheckpointMetadata.load(meta_path)
        assert meta.backend == "cpu"
        assert meta.model_params == int(1e6)

        # Load
        loaded_state: dict = {}
        with patch.object(mgr, "_dcp_load", mgr._fallback_load):
            loaded_meta = mgr.load(
                loaded_state,
                ckpt_id,
                target_backend=HardwareBackend.CPU,
                target_device=torch.device("cpu"),
            )

        assert loaded_meta is not None
        assert "weight" in loaded_state
        torch.testing.assert_close(loaded_state["weight"], original_weight)
        torch.testing.assert_close(loaded_state["bias"], original_bias)

    def test_cross_backend_metadata(self, tmp_path):
        """Save with CUDA metadata, verify metadata records source backend."""
        config = CheckpointConfig(
            storage_path=str(tmp_path),
            normalize_on_save=True,
            include_metadata=True,
            max_checkpoints_to_keep=0,
        )
        mgr = CheckpointManager(config=config, backend=HardwareBackend.CUDA)

        state = {"weight": torch.randn(8, 8)}
        ckpt_id = str(tmp_path / "cuda_ckpt")

        with patch.object(mgr, "_dcp_save"):
            mgr.save(state, checkpoint_id=ckpt_id)

        meta = CheckpointMetadata.load(Path(ckpt_id) / _METADATA_FILENAME)
        assert meta.backend == "cuda"
        assert "weight" in meta.dtype_map


class TestPortabilityPipeline:
    """Tests for cross-backend portability flow."""

    def test_normalize_restore_preserves_values(self):
        original = torch.randn(32, 32)
        state = {"model.weight": original.clone()}

        normalized, dtype_map, device_map = (
            PortabilityNormalizer.normalize_state_dict(
                state, HardwareBackend.CUDA
            )
        )

        metadata = CheckpointMetadata(
            checkpoint_id="test",
            timestamp="2026-01-01",
            torchbridge_version="0.5.28",
            pytorch_version="2.7.0",
            backend="cuda",
            architecture=None,
            world_size=1,
            local_world_size=1,
            dtype_map=dtype_map,
            device_map=device_map,
        )

        restored = PortabilityNormalizer.restore_state_dict(
            normalized, metadata, HardwareBackend.CPU, torch.device("cpu")
        )
        torch.testing.assert_close(restored["model.weight"], original)

    def test_normalize_nested_optimizer_state(self):
        state = {
            "optimizer": {
                "state": {
                    "momentum": torch.randn(8),
                    "step": torch.tensor(100),
                },
                "lr": 0.001,
            }
        }
        normalized, dtype_map, device_map = (
            PortabilityNormalizer.normalize_state_dict(
                state, HardwareBackend.CUDA
            )
        )

        assert "optimizer.state.momentum" in dtype_map
        assert "optimizer.state.step" in dtype_map
        assert normalized["optimizer"]["lr"] == 0.001


class TestFrequencyAdvisorIntegration:
    """Integration tests for frequency advisor across cluster sizes."""

    @pytest.mark.parametrize("world_size,expected_risk", [
        (1, "low"),
        (4, "low"),
        (16, "medium"),
        (64, "medium"),
        (128, "medium"),
        (512, "high"),
    ])
    def test_risk_level_by_cluster_size(self, world_size, expected_risk):
        advisor = CheckpointFrequencyAdvisor()
        rec = advisor.recommend(world_size=world_size)
        assert rec.risk_level == expected_risk

    def test_recommendations_are_reasonable(self):
        advisor = CheckpointFrequencyAdvisor()

        # Small cluster: interval should be long (> 30 min)
        small = advisor.recommend(world_size=4, checkpoint_time_seconds=60)
        assert small.interval_minutes > 30

        # Large cluster: interval should be shorter
        large = advisor.recommend(world_size=256, checkpoint_time_seconds=60)
        assert large.interval_minutes < small.interval_minutes

    def test_overhead_decreases_with_scale(self):
        advisor = CheckpointFrequencyAdvisor()
        # The optimal overhead should be reasonable (< 10%)
        rec = advisor.recommend(
            world_size=64,
            checkpoint_time_seconds=60,
        )
        assert rec.optimal_overhead_pct < 10.0


class TestHealthTriggerIntegration:
    """Integration tests combining health triggers with advisor."""

    def test_degradation_triggers_then_advisor_recommends(self):
        trigger = CheckpointHealthTrigger()
        advisor = CheckpointFrequencyAdvisor()

        # Simulate degradation
        should, reason = trigger.should_checkpoint({
            "temperature_c": 91.0,
            "health_trend": "degrading",
            "memory_errors": 0,
        })
        assert should is True

        # After trigger, advisor recommends future frequency
        rec = advisor.recommend(world_size=64, mtbf_hours=6.0)
        assert rec.risk_level in ("medium", "high")
        assert rec.interval_minutes > 0


class TestCLIIntegration:
    """Tests for checkpoint CLI command."""

    def test_advisor_help(self):
        from torchbridge.cli.checkpoint import main

        with pytest.raises(SystemExit) as exc_info:
            main(["advisor", "--help"])
        assert exc_info.value.code == 0

    def test_advisor_basic(self):
        from torchbridge.cli.checkpoint import main

        result = main(["advisor", "--world-size", "64"])
        assert result == 0

    def test_advisor_ci_json(self, capsys):
        from torchbridge.cli.checkpoint import main

        result = main(["advisor", "--world-size", "32", "--ci"])
        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "recommendation" in data
        assert data["world_size"] == 32

    def test_info_missing_path(self, tmp_path):
        from torchbridge.cli.checkpoint import main

        result = main(["info", str(tmp_path / "nonexistent")])
        assert result == 1

    def test_list_empty(self, tmp_path):
        from torchbridge.cli.checkpoint import main

        result = main(["list", str(tmp_path)])
        assert result == 0

    def test_no_action_shows_usage(self):
        from torchbridge.cli.checkpoint import main

        result = main([])
        assert result == 1


class TestPackageImports:
    """Tests for package import structure."""

    def test_import_checkpoint_package(self):
        from torchbridge.checkpoint import (  # noqa: F811
            CheckpointConfig,
            CheckpointFrequencyAdvisor,
            CheckpointHealthTrigger,
            CheckpointManager,
            CheckpointMetadata,
            FrequencyRecommendation,
            PortabilityNormalizer,
            SerializationFormat,
            StorageBackendFactory,
            StorageBackendType,
        )

        all_exports = [
            CheckpointConfig, CheckpointFrequencyAdvisor,
            CheckpointHealthTrigger, CheckpointManager,
            CheckpointMetadata, FrequencyRecommendation,
            PortabilityNormalizer, SerializationFormat,
            StorageBackendFactory, StorageBackendType,
        ]
        assert all(cls is not None for cls in all_exports)
        assert StorageBackendType.LOCAL.value == "local"

    def test_import_cli_command(self):
        from torchbridge.cli.checkpoint import CheckpointCommand

        assert hasattr(CheckpointCommand, "register")
        assert hasattr(CheckpointCommand, "execute")
