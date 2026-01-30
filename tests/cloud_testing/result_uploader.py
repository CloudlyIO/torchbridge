"""
Result Uploader for KernelPyTorch Cloud Testing.

This module provides utilities for uploading test results and benchmarks
to cloud storage (S3, GCS) for persistence and analysis.

Version: 0.3.7
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Base Uploader
# ============================================================================

@dataclass
class UploadResult:
    """Result of an upload operation."""
    success: bool
    path: str
    url: str | None = None
    error_message: str | None = None
    bytes_uploaded: int = 0


class ResultUploader(ABC):
    """Abstract base class for result uploaders."""

    @abstractmethod
    def upload_json(
        self,
        data: dict[str, Any],
        path: str,
        metadata: dict[str, str] | None = None,
    ) -> UploadResult:
        """Upload JSON data."""
        pass

    @abstractmethod
    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        metadata: dict[str, str] | None = None,
    ) -> UploadResult:
        """Upload a file."""
        pass

    @abstractmethod
    def list_results(
        self,
        prefix: str = "",
        limit: int = 100,
    ) -> list[str]:
        """List uploaded results."""
        pass

    @abstractmethod
    def download_json(self, path: str) -> dict[str, Any] | None:
        """Download JSON data."""
        pass


# ============================================================================
# S3 Uploader
# ============================================================================

class S3Uploader(ResultUploader):
    """
    Upload results to AWS S3.

    Example:
        >>> uploader = S3Uploader("my-bucket", prefix="kernelpytorch/results")
        >>> result = uploader.upload_json(
        ...     {"tests_passed": 100, "tests_failed": 0},
        ...     "test-run-123/results.json"
        ... )
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "kernelpytorch/results",
        region: str = "us-west-2",
    ):
        """
        Initialize S3 Uploader.

        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all uploads
            region: AWS region
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self._boto3_available = self._check_boto3()

    def _check_boto3(self) -> bool:
        """Check if boto3 is available."""
        try:
            import boto3  # noqa: F401
            return True
        except ImportError:
            logger.warning("boto3 not installed. S3 uploads will be simulated.")
            return False

    def _get_s3_client(self):
        """Get S3 client."""
        if not self._boto3_available:
            return None
        import boto3
        return boto3.client("s3", region_name=self.region)

    def _get_full_path(self, path: str) -> str:
        """Get full S3 key with prefix."""
        return f"{self.prefix}/{path}" if self.prefix else path

    def upload_json(
        self,
        data: dict[str, Any],
        path: str,
        metadata: dict[str, str] | None = None,
    ) -> UploadResult:
        """Upload JSON data to S3."""
        full_path = self._get_full_path(path)
        json_data = json.dumps(data, indent=2, default=str)

        logger.info(f"Uploading JSON to s3://{self.bucket}/{full_path}")

        if not self._boto3_available:
            # Simulate upload
            return UploadResult(
                success=True,
                path=full_path,
                url=f"s3://{self.bucket}/{full_path}",
                bytes_uploaded=len(json_data),
            )

        try:
            s3 = self._get_s3_client()
            extra_args = {"ContentType": "application/json"}
            if metadata:
                extra_args["Metadata"] = metadata

            s3.put_object(
                Bucket=self.bucket,
                Key=full_path,
                Body=json_data,
                **extra_args,
            )

            return UploadResult(
                success=True,
                path=full_path,
                url=f"s3://{self.bucket}/{full_path}",
                bytes_uploaded=len(json_data),
            )

        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return UploadResult(
                success=False,
                path=full_path,
                error_message=str(e),
            )

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        metadata: dict[str, str] | None = None,
    ) -> UploadResult:
        """Upload a file to S3."""
        local_path = Path(local_path)
        full_path = self._get_full_path(remote_path)

        logger.info(f"Uploading {local_path} to s3://{self.bucket}/{full_path}")

        if not local_path.exists():
            return UploadResult(
                success=False,
                path=full_path,
                error_message=f"Local file not found: {local_path}",
            )

        if not self._boto3_available:
            return UploadResult(
                success=True,
                path=full_path,
                url=f"s3://{self.bucket}/{full_path}",
                bytes_uploaded=local_path.stat().st_size,
            )

        try:
            s3 = self._get_s3_client()
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = metadata

            s3.upload_file(
                str(local_path),
                self.bucket,
                full_path,
                ExtraArgs=extra_args if extra_args else None,
            )

            return UploadResult(
                success=True,
                path=full_path,
                url=f"s3://{self.bucket}/{full_path}",
                bytes_uploaded=local_path.stat().st_size,
            )

        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return UploadResult(
                success=False,
                path=full_path,
                error_message=str(e),
            )

    def list_results(
        self,
        prefix: str = "",
        limit: int = 100,
    ) -> list[str]:
        """List results in S3."""
        full_prefix = self._get_full_path(prefix) if prefix else self.prefix

        if not self._boto3_available:
            return []

        try:
            s3 = self._get_s3_client()
            response = s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=full_prefix,
                MaxKeys=limit,
            )
            return [obj["Key"] for obj in response.get("Contents", [])]

        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []

    def download_json(self, path: str) -> dict[str, Any] | None:
        """Download JSON data from S3."""
        full_path = self._get_full_path(path)

        if not self._boto3_available:
            return None

        try:
            s3 = self._get_s3_client()
            response = s3.get_object(Bucket=self.bucket, Key=full_path)
            return json.loads(response["Body"].read().decode("utf-8"))

        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return None


# ============================================================================
# GCS Uploader
# ============================================================================

class GCSUploader(ResultUploader):
    """
    Upload results to Google Cloud Storage.

    Example:
        >>> uploader = GCSUploader("my-bucket", prefix="kernelpytorch/results")
        >>> result = uploader.upload_json(
        ...     {"tests_passed": 100},
        ...     "test-run-123/results.json"
        ... )
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "kernelpytorch/results",
    ):
        """
        Initialize GCS Uploader.

        Args:
            bucket: GCS bucket name
            prefix: Path prefix for all uploads
        """
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._gcs_available = self._check_gcs()

    def _check_gcs(self) -> bool:
        """Check if google-cloud-storage is available."""
        try:
            from google.cloud import storage  # noqa: F401
            return True
        except ImportError:
            logger.warning("google-cloud-storage not installed. GCS uploads will be simulated.")
            return False

    def _get_gcs_client(self):
        """Get GCS client."""
        if not self._gcs_available:
            return None
        from google.cloud import storage
        return storage.Client()

    def _get_full_path(self, path: str) -> str:
        """Get full GCS path with prefix."""
        return f"{self.prefix}/{path}" if self.prefix else path

    def upload_json(
        self,
        data: dict[str, Any],
        path: str,
        metadata: dict[str, str] | None = None,
    ) -> UploadResult:
        """Upload JSON data to GCS."""
        full_path = self._get_full_path(path)
        json_data = json.dumps(data, indent=2, default=str)

        logger.info(f"Uploading JSON to gs://{self.bucket}/{full_path}")

        if not self._gcs_available:
            return UploadResult(
                success=True,
                path=full_path,
                url=f"gs://{self.bucket}/{full_path}",
                bytes_uploaded=len(json_data),
            )

        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.bucket)
            blob = bucket.blob(full_path)

            if metadata:
                blob.metadata = metadata

            blob.upload_from_string(json_data, content_type="application/json")

            return UploadResult(
                success=True,
                path=full_path,
                url=f"gs://{self.bucket}/{full_path}",
                bytes_uploaded=len(json_data),
            )

        except Exception as e:
            logger.error(f"Failed to upload to GCS: {e}")
            return UploadResult(
                success=False,
                path=full_path,
                error_message=str(e),
            )

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        metadata: dict[str, str] | None = None,
    ) -> UploadResult:
        """Upload a file to GCS."""
        local_path = Path(local_path)
        full_path = self._get_full_path(remote_path)

        logger.info(f"Uploading {local_path} to gs://{self.bucket}/{full_path}")

        if not local_path.exists():
            return UploadResult(
                success=False,
                path=full_path,
                error_message=f"Local file not found: {local_path}",
            )

        if not self._gcs_available:
            return UploadResult(
                success=True,
                path=full_path,
                url=f"gs://{self.bucket}/{full_path}",
                bytes_uploaded=local_path.stat().st_size,
            )

        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.bucket)
            blob = bucket.blob(full_path)

            if metadata:
                blob.metadata = metadata

            blob.upload_from_filename(str(local_path))

            return UploadResult(
                success=True,
                path=full_path,
                url=f"gs://{self.bucket}/{full_path}",
                bytes_uploaded=local_path.stat().st_size,
            )

        except Exception as e:
            logger.error(f"Failed to upload file to GCS: {e}")
            return UploadResult(
                success=False,
                path=full_path,
                error_message=str(e),
            )

    def list_results(
        self,
        prefix: str = "",
        limit: int = 100,
    ) -> list[str]:
        """List results in GCS."""
        full_prefix = self._get_full_path(prefix) if prefix else self.prefix

        if not self._gcs_available:
            return []

        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.bucket)
            blobs = bucket.list_blobs(prefix=full_prefix, max_results=limit)
            return [blob.name for blob in blobs]

        except Exception as e:
            logger.error(f"Failed to list GCS objects: {e}")
            return []

    def download_json(self, path: str) -> dict[str, Any] | None:
        """Download JSON data from GCS."""
        full_path = self._get_full_path(path)

        if not self._gcs_available:
            return None

        try:
            client = self._get_gcs_client()
            bucket = client.bucket(self.bucket)
            blob = bucket.blob(full_path)
            return json.loads(blob.download_as_string().decode("utf-8"))

        except Exception as e:
            logger.error(f"Failed to download from GCS: {e}")
            return None


# ============================================================================
# Convenience Functions
# ============================================================================

def upload_results(
    results: dict[str, Any],
    cloud_provider: str = "auto",
    bucket: str | None = None,
    path: str | None = None,
) -> UploadResult:
    """
    Upload results to cloud storage.

    Args:
        results: Results dictionary to upload
        cloud_provider: "s3", "gcs", or "auto" (detect from environment)
        bucket: Bucket name (auto-detect from environment if None)
        path: Upload path (auto-generate if None)

    Returns:
        UploadResult with upload status
    """
    # Auto-detect provider
    if cloud_provider == "auto":
        if os.environ.get("KERNELPYTORCH_S3_BUCKET"):
            cloud_provider = "s3"
        elif os.environ.get("KERNELPYTORCH_GCS_BUCKET"):
            cloud_provider = "gcs"
        else:
            cloud_provider = "s3"  # Default

    # Get bucket
    if bucket is None:
        if cloud_provider == "s3":
            bucket = os.environ.get("KERNELPYTORCH_S3_BUCKET", "kernelpytorch-results")
        else:
            bucket = os.environ.get("KERNELPYTORCH_GCS_BUCKET", "kernelpytorch-results")

    # Generate path
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = f"results/{timestamp}/results.json"

    # Create uploader and upload
    if cloud_provider == "s3":
        uploader = S3Uploader(bucket)
    else:
        uploader = GCSUploader(bucket)

    return uploader.upload_json(results, path)
