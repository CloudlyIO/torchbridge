"""
Tests for FSDPManager.apply() — facade fix

Verifies that apply() exists, has the correct signature, raises RuntimeError
when torch.distributed is not initialized, and maps config values correctly.
"""

import pytest

from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture
from torchbridge.distributed.fsdp2 import (
    FSDPConfig,
    FSDPManager,
    MixedPrecisionChoice,
    ShardingStrategy,
)


class TestFSDPApply:
    """Tests for FSDPManager.apply() method."""

    def test_apply_method_exists(self):
        """FSDPManager has an apply() method."""
        manager = FSDPManager(backend=HardwareBackend.CUDA)
        assert callable(getattr(manager, "apply", None))

    def test_apply_raises_runtime_error_uninitialized(self):
        """apply() raises RuntimeError when torch.distributed is not initialized."""
        import torch
        import torch.nn as nn

        manager = FSDPManager(backend=HardwareBackend.CUDA)
        model = nn.Linear(64, 64)

        # In test environment, distributed is never initialized
        if not torch.distributed.is_initialized():
            with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
                manager.apply(model)

    def test_apply_raises_import_error_without_dist(self, monkeypatch):
        """apply() raises RuntimeError when dist.is_initialized() returns False."""
        import torch.nn as nn

        manager = FSDPManager(backend=HardwareBackend.CUDA)
        model = nn.Linear(32, 32)

        # Force dist.is_initialized to return False
        monkeypatch.setattr(
            "torch.distributed.is_initialized", lambda: False
        )
        with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
            manager.apply(model)

    def test_apply_signature_accepts_model(self):
        """apply() signature accepts a single nn.Module positional argument."""
        import inspect

        manager = FSDPManager(backend=HardwareBackend.CPU)
        sig = inspect.signature(manager.apply)
        params = list(sig.parameters.keys())
        assert "model" in params

    def test_resolved_config_used_in_apply_path(self):
        """FSDPManager resolves config before apply() is called."""
        config = FSDPConfig(
            sharding_strategy=ShardingStrategy.NO_SHARD,
            mixed_precision=MixedPrecisionChoice.FP16,
        )
        manager = FSDPManager(
            config=config,
            backend=HardwareBackend.CUDA,
            architecture=NVIDIAArchitecture.AMPERE,
        )
        resolved = manager.resolved_config
        assert resolved.sharding_strategy == ShardingStrategy.NO_SHARD
        assert resolved.mixed_precision == MixedPrecisionChoice.FP16
