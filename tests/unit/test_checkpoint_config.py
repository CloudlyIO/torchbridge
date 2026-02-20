"""Tests for checkpoint configuration."""

import pytest

from torchbridge.checkpoint.config import (
    CheckpointConfig,
    SerializationFormat,
    StorageBackendType,
)


class TestStorageBackendType:
    """Tests for StorageBackendType enum."""

    def test_all_values(self):
        assert StorageBackendType.LOCAL.value == "local"
        assert StorageBackendType.S3.value == "s3"
        assert StorageBackendType.GCS.value == "gcs"
        assert StorageBackendType.AZURE.value == "azure"

    def test_count(self):
        assert len(StorageBackendType) == 4


class TestSerializationFormat:
    """Tests for SerializationFormat enum."""

    def test_all_values(self):
        assert SerializationFormat.TORCH.value == "torch"
        assert SerializationFormat.SAFETENSORS.value == "safetensors"

    def test_count(self):
        assert len(SerializationFormat) == 2


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_defaults(self):
        config = CheckpointConfig()
        assert config.storage_backend == StorageBackendType.LOCAL
        assert config.storage_path == "./checkpoints"
        assert config.async_save is True
        assert config.process_based_async is True
        assert config.plan_caching is True
        assert config.pinned_memory_staging is True
        assert config.serialization_format == SerializationFormat.TORCH
        assert config.max_checkpoints_to_keep == 3
        assert config.normalize_on_save is True
        assert config.include_metadata is True
        assert config.io_thread_count == 1

    def test_custom_values(self):
        config = CheckpointConfig(
            storage_backend=StorageBackendType.S3,
            storage_path="s3://my-bucket/checkpoints",
            async_save=False,
            process_based_async=False,
            plan_caching=False,
            max_checkpoints_to_keep=5,
            serialization_format=SerializationFormat.SAFETENSORS,
            io_thread_count=4,
        )
        assert config.storage_backend == StorageBackendType.S3
        assert config.storage_path == "s3://my-bucket/checkpoints"
        assert config.async_save is False
        assert config.max_checkpoints_to_keep == 5
        assert config.io_thread_count == 4

    def test_to_dict(self):
        config = CheckpointConfig()
        d = config.to_dict()
        assert d["storage_backend"] == "local"
        assert d["async_save"] is True
        assert d["serialization_format"] == "torch"
        assert d["max_checkpoints_to_keep"] == 3
        assert d["io_thread_count"] == 1

    def test_to_dict_cloud(self):
        config = CheckpointConfig(
            storage_backend=StorageBackendType.GCS,
            storage_path="gs://bucket/ckpts",
        )
        d = config.to_dict()
        assert d["storage_backend"] == "gcs"
        assert d["storage_path"] == "gs://bucket/ckpts"

    def test_invalid_max_checkpoints(self):
        with pytest.raises(ValueError, match="max_checkpoints_to_keep"):
            CheckpointConfig(max_checkpoints_to_keep=-1)

    def test_invalid_io_thread_count(self):
        with pytest.raises(ValueError, match="io_thread_count"):
            CheckpointConfig(io_thread_count=0)

    def test_zero_max_checkpoints_is_valid(self):
        config = CheckpointConfig(max_checkpoints_to_keep=0)
        assert config.max_checkpoints_to_keep == 0

    def test_all_storage_backends_in_to_dict(self):
        for backend in StorageBackendType:
            config = CheckpointConfig(storage_backend=backend)
            d = config.to_dict()
            assert d["storage_backend"] == backend.value
