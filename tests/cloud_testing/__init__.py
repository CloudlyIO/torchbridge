"""
Cloud Testing Infrastructure for KernelPyTorch.

This module provides infrastructure for running KernelPyTorch tests and benchmarks
on cloud hardware (AWS EC2, GCP Compute Engine, GCP TPU).

Components:
- aws_test_harness: AWS EC2 test orchestration (NVIDIA P5/P4d, AMD ROCm)
- gcp_test_harness: GCP test orchestration (NVIDIA A3/A2, TPU v5e/v6e)
- result_uploader: Upload results to S3/GCS
- benchmark_database: Store and query benchmark results
"""

from .aws_test_harness import (
    AWSInstanceConfig,
    AWSTestHarness,
    AWSTestResult,
    create_aws_harness,
)
from .benchmark_database import (
    BenchmarkDatabase,
    BenchmarkRecord,
    compare_platforms,
    query_benchmarks,
)
from .gcp_test_harness import (
    GCPInstanceConfig,
    GCPTestHarness,
    GCPTestResult,
    TPUConfig,
    create_gcp_harness,
)
from .result_uploader import (
    GCSUploader,
    ResultUploader,
    S3Uploader,
    upload_results,
)

__all__ = [
    # AWS
    "AWSTestHarness",
    "AWSInstanceConfig",
    "AWSTestResult",
    "create_aws_harness",
    # GCP
    "GCPTestHarness",
    "GCPInstanceConfig",
    "GCPTestResult",
    "TPUConfig",
    "create_gcp_harness",
    # Uploaders
    "ResultUploader",
    "S3Uploader",
    "GCSUploader",
    "upload_results",
    # Database
    "BenchmarkDatabase",
    "BenchmarkRecord",
    "query_benchmarks",
    "compare_platforms",
]
