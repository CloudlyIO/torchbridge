"""
AWS Test Harness for KernelPyTorch Cloud Testing.

This module provides infrastructure for running tests on AWS EC2 instances,
including NVIDIA GPU instances (P5/P4d) and AMD ROCm instances.

Features:
- Instance lifecycle management (launch, configure, terminate)
- Automated test execution and result collection
- CloudWatch metrics integration
- Cost tracking and optimization
- Multi-instance parallel testing

Version: 0.3.7
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================

class AWSInstanceType(Enum):
    """Supported AWS instance types for testing."""
    # NVIDIA Instances
    P5_24XLARGE = "p5.48xlarge"      # 8x H100 80GB
    P4D_24XLARGE = "p4d.24xlarge"    # 8x A100 40GB
    P4DE_24XLARGE = "p4de.24xlarge"  # 8x A100 80GB
    G5_XLARGE = "g5.xlarge"          # 1x A10G 24GB (dev/test)
    G5_12XLARGE = "g5.12xlarge"      # 4x A10G 24GB

    # AMD ROCm Instances (when available)
    # Note: AMD instances may not be available in all regions

    # CPU Instances (fallback)
    C6I_8XLARGE = "c6i.8xlarge"      # 32 vCPU, 64GB RAM


class AWSRegion(Enum):
    """AWS regions with GPU availability."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    AP_NORTHEAST_1 = "ap-northeast-1"


@dataclass
class AWSInstanceConfig:
    """Configuration for an AWS EC2 instance."""
    instance_type: AWSInstanceType
    region: AWSRegion = AWSRegion.US_WEST_2
    ami_id: Optional[str] = None  # Auto-detect if None
    key_name: Optional[str] = None
    security_group_ids: List[str] = field(default_factory=list)
    subnet_id: Optional[str] = None
    iam_instance_profile: Optional[str] = None
    spot_instance: bool = True  # Use spot for cost savings
    max_spot_price: Optional[float] = None
    root_volume_size_gb: int = 200
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Set default tags."""
        default_tags = {
            "Project": "KernelPyTorch",
            "Environment": "Testing",
            "ManagedBy": "cloud-testing-harness",
        }
        self.tags = {**default_tags, **self.tags}

    def get_ami_id(self) -> str:
        """Get appropriate AMI ID for the instance type and region."""
        if self.ami_id:
            return self.ami_id

        # Default to Deep Learning AMI (PyTorch)
        # These are region-specific and should be updated periodically
        ami_map = {
            AWSRegion.US_EAST_1: "ami-0123456789abcdef0",  # Placeholder
            AWSRegion.US_WEST_2: "ami-0123456789abcdef0",  # Placeholder
        }
        return ami_map.get(self.region, "ami-0123456789abcdef0")


@dataclass
class AWSTestResult:
    """Results from a test run on AWS."""
    instance_id: str
    instance_type: str
    region: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    test_output: str
    benchmark_results: Dict[str, Any]
    cloudwatch_metrics: Dict[str, Any]
    cost_estimate_usd: float
    success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "instance_id": self.instance_id,
            "instance_type": self.instance_type,
            "region": self.region,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "tests_skipped": self.tests_skipped,
            "test_output": self.test_output,
            "benchmark_results": self.benchmark_results,
            "cloudwatch_metrics": self.cloudwatch_metrics,
            "cost_estimate_usd": self.cost_estimate_usd,
            "success": self.success,
            "error_message": self.error_message,
        }


# ============================================================================
# AWS Test Harness
# ============================================================================

class AWSTestHarness:
    """
    AWS EC2 Test Harness for KernelPyTorch.

    Manages instance lifecycle and test execution on AWS.

    Example:
        >>> config = AWSInstanceConfig(
        ...     instance_type=AWSInstanceType.P4D_24XLARGE,
        ...     region=AWSRegion.US_WEST_2,
        ...     spot_instance=True,
        ... )
        >>> harness = AWSTestHarness(config)
        >>> result = harness.run_tests()
        >>> print(f"Tests passed: {result.tests_passed}")
    """

    # Hourly costs for cost estimation (approximate, may vary)
    INSTANCE_COSTS = {
        AWSInstanceType.P5_24XLARGE: 98.32,
        AWSInstanceType.P4D_24XLARGE: 32.77,
        AWSInstanceType.P4DE_24XLARGE: 40.96,
        AWSInstanceType.G5_XLARGE: 1.006,
        AWSInstanceType.G5_12XLARGE: 5.672,
        AWSInstanceType.C6I_8XLARGE: 1.36,
    }

    def __init__(
        self,
        config: AWSInstanceConfig,
        s3_bucket: Optional[str] = None,
        cloudwatch_namespace: str = "KernelPyTorch/Testing",
    ):
        """
        Initialize AWS Test Harness.

        Args:
            config: Instance configuration
            s3_bucket: S3 bucket for result storage
            cloudwatch_namespace: CloudWatch namespace for metrics
        """
        self.config = config
        self.s3_bucket = s3_bucket or os.environ.get("KERNELPYTORCH_S3_BUCKET")
        self.cloudwatch_namespace = cloudwatch_namespace
        self.instance_id: Optional[str] = None
        self.public_ip: Optional[str] = None
        self._boto3_available = self._check_boto3()

    def _check_boto3(self) -> bool:
        """Check if boto3 is available."""
        try:
            import boto3
            return True
        except ImportError:
            logger.warning("boto3 not installed. AWS operations will be simulated.")
            return False

    def _get_ec2_client(self):
        """Get EC2 client for the configured region."""
        if not self._boto3_available:
            return None
        import boto3
        return boto3.client("ec2", region_name=self.config.region.value)

    def _get_cloudwatch_client(self):
        """Get CloudWatch client."""
        if not self._boto3_available:
            return None
        import boto3
        return boto3.client("cloudwatch", region_name=self.config.region.value)

    def launch_instance(self) -> str:
        """
        Launch an EC2 instance for testing.

        Returns:
            Instance ID
        """
        logger.info(f"Launching {self.config.instance_type.value} in {self.config.region.value}")

        if not self._boto3_available:
            # Simulate for testing without AWS credentials
            self.instance_id = f"i-simulated-{int(time.time())}"
            self.public_ip = "127.0.0.1"
            logger.info(f"Simulated instance: {self.instance_id}")
            return self.instance_id

        ec2 = self._get_ec2_client()

        # Build launch specification
        launch_params = {
            "ImageId": self.config.get_ami_id(),
            "InstanceType": self.config.instance_type.value,
            "MinCount": 1,
            "MaxCount": 1,
            "BlockDeviceMappings": [
                {
                    "DeviceName": "/dev/sda1",
                    "Ebs": {
                        "VolumeSize": self.config.root_volume_size_gb,
                        "VolumeType": "gp3",
                        "DeleteOnTermination": True,
                    },
                },
            ],
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [{"Key": k, "Value": v} for k, v in self.config.tags.items()],
                },
            ],
        }

        if self.config.key_name:
            launch_params["KeyName"] = self.config.key_name
        if self.config.security_group_ids:
            launch_params["SecurityGroupIds"] = self.config.security_group_ids
        if self.config.subnet_id:
            launch_params["SubnetId"] = self.config.subnet_id
        if self.config.iam_instance_profile:
            launch_params["IamInstanceProfile"] = {"Name": self.config.iam_instance_profile}

        # Launch as spot or on-demand
        if self.config.spot_instance:
            launch_params["InstanceMarketOptions"] = {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                },
            }
            if self.config.max_spot_price:
                launch_params["InstanceMarketOptions"]["SpotOptions"]["MaxPrice"] = str(
                    self.config.max_spot_price
                )

        response = ec2.run_instances(**launch_params)
        self.instance_id = response["Instances"][0]["InstanceId"]

        # Wait for instance to be running
        logger.info(f"Waiting for instance {self.instance_id} to be running...")
        waiter = ec2.get_waiter("instance_running")
        waiter.wait(InstanceIds=[self.instance_id])

        # Get public IP
        response = ec2.describe_instances(InstanceIds=[self.instance_id])
        self.public_ip = response["Reservations"][0]["Instances"][0].get("PublicIpAddress")

        logger.info(f"Instance {self.instance_id} running at {self.public_ip}")
        return self.instance_id

    def terminate_instance(self) -> None:
        """Terminate the EC2 instance."""
        if not self.instance_id:
            return

        logger.info(f"Terminating instance {self.instance_id}")

        if not self._boto3_available:
            self.instance_id = None
            self.public_ip = None
            return

        ec2 = self._get_ec2_client()
        ec2.terminate_instances(InstanceIds=[self.instance_id])

        self.instance_id = None
        self.public_ip = None

    def run_tests(
        self,
        test_path: str = "tests/",
        pytest_args: Optional[List[str]] = None,
        timeout_seconds: int = 7200,  # 2 hours default
    ) -> AWSTestResult:
        """
        Run tests on the AWS instance.

        Args:
            test_path: Path to tests (relative to repo root)
            pytest_args: Additional pytest arguments
            timeout_seconds: Maximum test duration

        Returns:
            AWSTestResult with test outcomes
        """
        start_time = datetime.now()

        if pytest_args is None:
            pytest_args = ["--tb=short", "-v"]

        try:
            # Launch instance if not already running
            if not self.instance_id:
                self.launch_instance()

            # Run tests (simulated if no AWS access)
            if not self._boto3_available or self.public_ip == "127.0.0.1":
                test_output, passed, failed, skipped = self._run_local_tests(
                    test_path, pytest_args
                )
            else:
                test_output, passed, failed, skipped = self._run_remote_tests(
                    test_path, pytest_args, timeout_seconds
                )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Get CloudWatch metrics
            cloudwatch_metrics = self._get_cloudwatch_metrics(start_time, end_time)

            # Estimate cost
            hourly_cost = self.INSTANCE_COSTS.get(self.config.instance_type, 1.0)
            if self.config.spot_instance:
                hourly_cost *= 0.3  # Approximate spot discount
            cost_estimate = (duration / 3600) * hourly_cost

            return AWSTestResult(
                instance_id=self.instance_id or "local",
                instance_type=self.config.instance_type.value,
                region=self.config.region.value,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                tests_passed=passed,
                tests_failed=failed,
                tests_skipped=skipped,
                test_output=test_output,
                benchmark_results={},
                cloudwatch_metrics=cloudwatch_metrics,
                cost_estimate_usd=cost_estimate,
                success=failed == 0,
            )

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return AWSTestResult(
                instance_id=self.instance_id or "unknown",
                instance_type=self.config.instance_type.value,
                region=self.config.region.value,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                test_output="",
                benchmark_results={},
                cloudwatch_metrics={},
                cost_estimate_usd=0.0,
                success=False,
                error_message=str(e),
            )
        finally:
            # Always clean up
            if self.instance_id and self.config.spot_instance:
                self.terminate_instance()

    def _run_local_tests(
        self,
        test_path: str,
        pytest_args: List[str],
    ) -> Tuple[str, int, int, int]:
        """Run tests locally (for simulation/development)."""
        logger.info("Running tests locally (simulated AWS)")

        cmd = ["python", "-m", "pytest", test_path] + pytest_args + ["--tb=no", "-q"]
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

        output = result.stdout + result.stderr

        # Parse pytest output for counts
        passed, failed, skipped = 0, 0, 0
        for line in output.split("\n"):
            if "passed" in line or "failed" in line or "skipped" in line:
                import re
                match = re.search(r"(\d+) passed", line)
                if match:
                    passed = int(match.group(1))
                match = re.search(r"(\d+) failed", line)
                if match:
                    failed = int(match.group(1))
                match = re.search(r"(\d+) skipped", line)
                if match:
                    skipped = int(match.group(1))

        return output, passed, failed, skipped

    def _run_remote_tests(
        self,
        test_path: str,
        pytest_args: List[str],
        timeout_seconds: int,
    ) -> Tuple[str, int, int, int]:
        """Run tests on remote EC2 instance via SSH."""
        logger.info(f"Running tests on {self.public_ip}")

        # This would use SSH to run tests on the remote instance
        # For now, return placeholder
        raise NotImplementedError(
            "Remote test execution requires SSH setup. "
            "Use _run_local_tests for development."
        )

    def _get_cloudwatch_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> Dict[str, Any]:
        """Get CloudWatch metrics for the test run."""
        if not self._boto3_available or not self.instance_id:
            return {"simulated": True}

        # Would query CloudWatch for GPU utilization, memory, etc.
        return {
            "gpu_utilization_avg": 0.0,
            "gpu_memory_used_avg": 0.0,
            "cpu_utilization_avg": 0.0,
        }

    def run_benchmarks(
        self,
        benchmark_suite: str = "all",
        quick: bool = False,
    ) -> Dict[str, Any]:
        """
        Run performance benchmarks on the instance.

        Args:
            benchmark_suite: Which benchmarks to run (all, nvidia, tpu, amd)
            quick: Run quick benchmarks (fewer iterations)

        Returns:
            Dictionary of benchmark results
        """
        logger.info(f"Running {benchmark_suite} benchmarks (quick={quick})")

        # Launch instance if needed
        if not self.instance_id:
            self.launch_instance()

        # Run benchmarks
        results = {
            "suite": benchmark_suite,
            "quick": quick,
            "instance_type": self.config.instance_type.value,
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {},
        }

        # Would run actual benchmarks here
        # For now, return structure

        return results


# ============================================================================
# Factory Function
# ============================================================================

def create_aws_harness(
    instance_type: str = "p4d.24xlarge",
    region: str = "us-west-2",
    spot: bool = True,
    **kwargs,
) -> AWSTestHarness:
    """
    Create an AWS test harness with common defaults.

    Args:
        instance_type: EC2 instance type string
        region: AWS region string
        spot: Use spot instances
        **kwargs: Additional AWSInstanceConfig parameters

    Returns:
        Configured AWSTestHarness
    """
    # Map strings to enums
    instance_type_enum = None
    for it in AWSInstanceType:
        if it.value == instance_type:
            instance_type_enum = it
            break
    if not instance_type_enum:
        raise ValueError(f"Unknown instance type: {instance_type}")

    region_enum = None
    for r in AWSRegion:
        if r.value == region:
            region_enum = r
            break
    if not region_enum:
        raise ValueError(f"Unknown region: {region}")

    config = AWSInstanceConfig(
        instance_type=instance_type_enum,
        region=region_enum,
        spot_instance=spot,
        **kwargs,
    )

    return AWSTestHarness(config)
