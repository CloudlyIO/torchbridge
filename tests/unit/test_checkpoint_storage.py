"""Tests for checkpoint storage backend factory."""

from unittest.mock import MagicMock, patch

import pytest

from torchbridge.checkpoint.config import CheckpointConfig, StorageBackendType
from torchbridge.checkpoint.storage import (
    _FSSPEC_PACKAGES,
    StorageBackendFactory,
    _get_fsspec_filesystem,
)


class TestStorageBackendFactory:
    """Tests for StorageBackendFactory."""

    @patch("torchbridge.checkpoint.storage._import_dcp_filesystem")
    def test_create_writer_local(self, mock_dcp):
        mock_fs = MagicMock()
        mock_dcp.return_value = mock_fs

        config = CheckpointConfig(
            storage_backend=StorageBackendType.LOCAL,
            storage_path="/tmp/ckpts",
            io_thread_count=2,
            pinned_memory_staging=True,
        )
        StorageBackendFactory.create_writer(config)

        mock_fs.FileSystemWriter.assert_called_once_with(
            path="/tmp/ckpts",
            single_file_per_rank=True,
            sync_files=True,
            thread_count=2,
            overwrite=True,
            cache_staged_state_dict=True,
        )

    @patch("torchbridge.checkpoint.storage._import_dcp_filesystem")
    def test_create_writer_local_no_pinned(self, mock_dcp):
        mock_fs = MagicMock()
        mock_dcp.return_value = mock_fs

        config = CheckpointConfig(
            storage_backend=StorageBackendType.LOCAL,
            pinned_memory_staging=False,
        )
        StorageBackendFactory.create_writer(config)

        call_kwargs = mock_fs.FileSystemWriter.call_args[1]
        assert "cache_staged_state_dict" not in call_kwargs

    @patch("torchbridge.checkpoint.storage._import_dcp_filesystem")
    def test_create_reader_local(self, mock_dcp):
        mock_fs = MagicMock()
        mock_dcp.return_value = mock_fs

        config = CheckpointConfig(storage_backend=StorageBackendType.LOCAL)
        StorageBackendFactory.create_reader(config, "/tmp/ckpts/ckpt_001")

        mock_fs.FileSystemReader.assert_called_once_with("/tmp/ckpts/ckpt_001")

    @patch("torchbridge.checkpoint.storage._get_fsspec_filesystem")
    @patch("torchbridge.checkpoint.storage._import_dcp_fsspec")
    @patch("torchbridge.checkpoint.storage._import_dcp_filesystem")
    def test_create_writer_s3(self, mock_dcp, mock_fsspec_mod, mock_get_fs):
        mock_filesystem = MagicMock()
        mock_get_fs.return_value = mock_filesystem

        config = CheckpointConfig(
            storage_backend=StorageBackendType.S3,
            storage_path="s3://bucket/ckpts",
        )
        StorageBackendFactory.create_writer(config)

        mock_get_fs.assert_called_once_with(StorageBackendType.S3)
        mock_fsspec_mod.return_value.FsspecWriter.assert_called_once()

    @patch("torchbridge.checkpoint.storage._get_fsspec_filesystem")
    @patch("torchbridge.checkpoint.storage._import_dcp_fsspec")
    @patch("torchbridge.checkpoint.storage._import_dcp_filesystem")
    def test_create_reader_s3(self, mock_dcp, mock_fsspec_mod, mock_get_fs):
        mock_filesystem = MagicMock()
        mock_get_fs.return_value = mock_filesystem

        config = CheckpointConfig(storage_backend=StorageBackendType.S3)
        StorageBackendFactory.create_reader(config, "s3://bucket/ckpts/ckpt_001")

        mock_get_fs.assert_called_once_with(StorageBackendType.S3)
        mock_fsspec_mod.return_value.FsspecReader.assert_called_once()

    def test_get_writer_info(self):
        config = CheckpointConfig(
            storage_backend=StorageBackendType.LOCAL,
            storage_path="/data/ckpts",
            async_save=True,
            pinned_memory_staging=True,
            io_thread_count=4,
        )
        info = StorageBackendFactory.get_writer_info(config)
        assert info["backend"] == "local"
        assert info["path"] == "/data/ckpts"
        assert info["async_save"] is True
        assert info["pinned_memory_staging"] is True
        assert info["io_thread_count"] == 4


class TestFsspecPackages:
    """Tests for fsspec package mapping."""

    def test_s3_package(self):
        assert StorageBackendType.S3 in _FSSPEC_PACKAGES
        pkg, mod = _FSSPEC_PACKAGES[StorageBackendType.S3]
        assert pkg == "s3fs"
        assert mod == "s3fs"

    def test_gcs_package(self):
        assert StorageBackendType.GCS in _FSSPEC_PACKAGES
        pkg, mod = _FSSPEC_PACKAGES[StorageBackendType.GCS]
        assert pkg == "gcsfs"

    def test_azure_package(self):
        assert StorageBackendType.AZURE in _FSSPEC_PACKAGES
        pkg, mod = _FSSPEC_PACKAGES[StorageBackendType.AZURE]
        assert pkg == "adlfs"

    def test_local_not_in_fsspec(self):
        assert StorageBackendType.LOCAL not in _FSSPEC_PACKAGES

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="No fsspec filesystem"):
            _get_fsspec_filesystem(StorageBackendType.LOCAL)

    @patch("importlib.import_module", side_effect=ImportError("no s3fs"))
    def test_missing_package_error_message(self, mock_import):
        with pytest.raises(ImportError, match="torchbridge-ml\\[checkpoint\\]"):
            _get_fsspec_filesystem(StorageBackendType.S3)

    @patch("importlib.import_module")
    def test_successful_filesystem_creation(self, mock_import):
        mock_mod = MagicMock()
        mock_fs_instance = MagicMock()
        mock_mod.S3FileSystem.return_value = mock_fs_instance
        mock_import.return_value = mock_mod

        result = _get_fsspec_filesystem(StorageBackendType.S3)
        assert result == mock_fs_instance
        mock_mod.S3FileSystem.assert_called_once()

    @patch("importlib.import_module")
    def test_gcs_filesystem_creation(self, mock_import):
        mock_mod = MagicMock()
        mock_fs_instance = MagicMock()
        mock_mod.GCSFileSystem.return_value = mock_fs_instance
        mock_import.return_value = mock_mod

        result = _get_fsspec_filesystem(StorageBackendType.GCS)
        assert result == mock_fs_instance

    @patch("importlib.import_module")
    def test_azure_filesystem_creation(self, mock_import):
        mock_mod = MagicMock()
        mock_fs_instance = MagicMock()
        mock_mod.AzureBlobFileSystem.return_value = mock_fs_instance
        mock_import.return_value = mock_mod

        result = _get_fsspec_filesystem(StorageBackendType.AZURE)
        assert result == mock_fs_instance
