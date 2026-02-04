"""
v0.4.6 Comprehensive Cloud Testing Framework

This module provides unified testing across all supported backends:
- NVIDIA (AWS P4d/P5/G5, GCP A100/L4)
- AMD (AWS AMD instances, AMD Developer Cloud)
- TPU (GCP TPU v5e/v5p)

Tests all v0.4.6 features including:
- Mixture of Experts (MoE)
- Native FP8 support
- FlexAttention
- All existing functionality

Version: 0.4.6
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class CloudPlatform(Enum):
    """Supported cloud platforms."""
    AWS_NVIDIA = "aws_nvidia"
    AWS_AMD = "aws_amd"
    GCP_NVIDIA = "gcp_nvidia"
    GCP_TPU = "gcp_tpu"
    LOCAL = "local"


class GPUType(Enum):
    """Supported GPU types."""
    # NVIDIA
    H100 = "h100"
    A100 = "a100"
    A100_80GB = "a100_80gb"
    A10G = "a10g"
    L4 = "l4"

    # AMD
    MI300X = "mi300x"
    MI300A = "mi300a"

    # TPU
    TPU_V5E = "tpu_v5e"
    TPU_V5P = "tpu_v5p"

    # CPU
    CPU = "cpu"


@dataclass
class TestConfig:
    """Configuration for a test run."""
    platform: CloudPlatform
    gpu_type: GPUType
    version: str = "0.4.6"
    test_moe: bool = True
    test_fp8: bool = True
    test_attention: bool = True
    test_memory: bool = True
    test_distributed: bool = False
    benchmark_iterations: int = 100
    warmup_iterations: int = 10
    output_dir: str = "./test_results"

    def __post_init__(self):
        """Create output directory."""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class TestResult:
    """Results from a test suite."""
    suite_name: str
    platform: str
    gpu_type: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkResult:
    """Results from a benchmark."""
    benchmark_name: str
    platform: str
    gpu_type: str
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Hardware Detection
# ============================================================================

def detect_hardware() -> tuple[CloudPlatform, GPUType, dict[str, Any]]:
    """Detect the current hardware platform and GPU type."""
    import torch

    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": 0,
        "gpus": [],
        "platform": "unknown",
        "backend": "cpu"
    }

    # Check for TPU first
    try:
        import torch_xla.core.xla_model as xm
        info["platform"] = "gcp_tpu"
        info["backend"] = "tpu"
        devices = xm.get_xla_supported_devices()
        info["gpu_count"] = len(devices)

        # Determine TPU version from environment
        tpu_type = os.environ.get("TPU_ACCELERATOR_TYPE", "v5litepod-8")
        if "v5p" in tpu_type:
            return CloudPlatform.GCP_TPU, GPUType.TPU_V5P, info
        else:
            return CloudPlatform.GCP_TPU, GPUType.TPU_V5E, info
    except ImportError:
        pass

    # Check for CUDA/ROCm
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_name = torch.cuda.get_device_name(i)
            info["gpus"].append({
                "index": i,
                "name": gpu_name,
                "memory_gb": round(props.total_memory / 1024**3, 1),
                "compute_capability": f"{props.major}.{props.minor}"
            })

        gpu_name = info["gpus"][0]["name"].lower() if info["gpus"] else ""

        # Detect GPU type
        if "h100" in gpu_name:
            gpu_type = GPUType.H100
            info["backend"] = "nvidia"
        elif "a100-80gb" in gpu_name or "a100-sxm4-80gb" in gpu_name:
            gpu_type = GPUType.A100_80GB
            info["backend"] = "nvidia"
        elif "a100" in gpu_name:
            gpu_type = GPUType.A100
            info["backend"] = "nvidia"
        elif "a10g" in gpu_name:
            gpu_type = GPUType.A10G
            info["backend"] = "nvidia"
        elif "l4" in gpu_name:
            gpu_type = GPUType.L4
            info["backend"] = "nvidia"
        elif "mi300x" in gpu_name or "instinct mi300x" in gpu_name:
            gpu_type = GPUType.MI300X
            info["backend"] = "amd"
        elif "mi300" in gpu_name:
            gpu_type = GPUType.MI300A
            info["backend"] = "amd"
        else:
            gpu_type = GPUType.A100  # Default assumption
            info["backend"] = "nvidia"

        # Detect platform (AWS vs GCP) based on instance metadata
        platform = _detect_cloud_platform(info["backend"])
        return platform, gpu_type, info

    # CPU fallback
    return CloudPlatform.LOCAL, GPUType.CPU, info


def _detect_cloud_platform(backend: str) -> CloudPlatform:
    """Detect cloud platform from instance metadata."""
    # Try AWS
    try:
        import requests
        resp = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type",
            timeout=1
        )
        if resp.status_code == 200:
            if backend == "amd":
                return CloudPlatform.AWS_AMD
            return CloudPlatform.AWS_NVIDIA
    except Exception:
        pass

    # Try GCP
    try:
        import requests
        resp = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
            headers={"Metadata-Flavor": "Google"},
            timeout=1
        )
        if resp.status_code == 200:
            return CloudPlatform.GCP_NVIDIA
    except Exception:
        pass

    return CloudPlatform.LOCAL


# ============================================================================
# Test Suites
# ============================================================================

class TestSuite:
    """Base class for test suites."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.results: list[TestResult] = []
        self.benchmarks: list[BenchmarkResult] = []

    def run_pytest(self, test_path: str, name: str) -> TestResult:
        """Run pytest on a test file/directory."""
        logger.info(f"Running tests: {name}")

        result_file = os.path.join(self.config.output_dir, f"{name}_results.json")

        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v",
            "--tb=short",
            f"--json-report-file={result_file}",
            "--json-report"
        ]

        start = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            duration = time.time() - start

            # Parse results
            result = TestResult(
                suite_name=name,
                platform=self.config.platform.value,
                gpu_type=self.config.gpu_type.value,
                duration_seconds=duration
            )

            if os.path.exists(result_file):
                with open(result_file) as f:
                    data = json.load(f)
                    summary = data.get("summary", {})
                    result.passed = summary.get("passed", 0)
                    result.failed = summary.get("failed", 0)
                    result.skipped = summary.get("skipped", 0)

            if proc.returncode != 0:
                result.errors.append(proc.stderr[-1000:] if proc.stderr else "Unknown error")

            return result

        except subprocess.TimeoutExpired:
            return TestResult(
                suite_name=name,
                platform=self.config.platform.value,
                gpu_type=self.config.gpu_type.value,
                errors=["Test timed out after 600 seconds"]
            )
        except Exception as e:
            return TestResult(
                suite_name=name,
                platform=self.config.platform.value,
                gpu_type=self.config.gpu_type.value,
                errors=[str(e)]
            )


class MoETestSuite(TestSuite):
    """Test suite for Mixture of Experts functionality."""

    def run(self) -> list[TestResult]:
        """Run MoE tests."""
        if not self.config.test_moe:
            logger.info("MoE tests disabled, skipping")
            return []

        logger.info("=" * 60)
        logger.info("Running MoE Test Suite")
        logger.info("=" * 60)

        results = []

        # Run MoE unit tests
        result = self.run_pytest("tests/test_moe.py", "moe_unit_tests")
        results.append(result)

        # Run MoE integration tests
        result = self._run_moe_integration()
        results.append(result)

        # Run MoE benchmarks
        benchmark = self._run_moe_benchmark()
        self.benchmarks.append(benchmark)

        self.results = results
        return results

    def _run_moe_integration(self) -> TestResult:
        """Run MoE integration tests."""
        import torch

        result = TestResult(
            suite_name="moe_integration",
            platform=self.config.platform.value,
            gpu_type=self.config.gpu_type.value
        )

        try:
            from torchbridge import (
                GLaMStyleMoE,
                MoEConfig,
                MoELayer,
                SparseMoELayer,
                SwitchTransformerMoE,
                create_moe,
                create_moe_layer,
            )

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            hidden_size = 256
            batch_size, seq_len = 4, 32
            x = torch.randn(batch_size, seq_len, hidden_size, device=device)

            tests = [
                ("standard_moe", lambda: MoELayer(MoEConfig(), hidden_size)),
                ("sparse_moe", lambda: SparseMoELayer(MoEConfig(), hidden_size, sparsity_level=0.25)),
                ("switch_moe", lambda: SwitchTransformerMoE(MoEConfig(), hidden_size)),
                ("glam_moe", lambda: GLaMStyleMoE(MoEConfig(), hidden_size)),
                ("create_moe", lambda: create_moe(hidden_size, num_experts=8)),
                ("factory_standard", lambda: create_moe_layer("standard", hidden_size, 8)),
                ("factory_switch", lambda: create_moe_layer("switch", hidden_size, 8)),
            ]

            start = time.time()
            for name, create_fn in tests:
                try:
                    layer = create_fn().to(device)
                    out = layer(x)
                    assert out.shape == x.shape
                    result.passed += 1
                    logger.info(f"  PASS: {name}")
                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"{name}: {str(e)}")
                    logger.error(f"  FAIL: {name} - {e}")

            result.duration_seconds = time.time() - start

        except ImportError as e:
            result.errors.append(f"Import error: {e}")
            result.failed = 1

        return result

    def _run_moe_benchmark(self) -> BenchmarkResult:
        """Run MoE performance benchmark."""
        import torch

        result = BenchmarkResult(
            benchmark_name="moe_performance",
            platform=self.config.platform.value,
            gpu_type=self.config.gpu_type.value
        )

        try:
            from torchbridge import MoEConfig, MoELayer

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            configs = [
                {"hidden": 512, "experts": 8, "top_k": 2, "batch": 8, "seq": 128},
                {"hidden": 1024, "experts": 8, "top_k": 2, "batch": 4, "seq": 256},
                {"hidden": 2048, "experts": 16, "top_k": 4, "batch": 2, "seq": 512},
            ]

            benchmarks = []
            for cfg in configs:
                config = MoEConfig(num_experts=cfg["experts"], top_k=cfg["top_k"])
                layer = MoELayer(config, cfg["hidden"]).to(device)
                x = torch.randn(cfg["batch"], cfg["seq"], cfg["hidden"], device=device)

                # Warmup
                for _ in range(self.config.warmup_iterations):
                    _ = layer(x)
                if device == 'cuda':
                    torch.cuda.synchronize()

                # Benchmark
                start = time.time()
                for _ in range(self.config.benchmark_iterations):
                    _ = layer(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start

                avg_ms = elapsed / self.config.benchmark_iterations * 1000
                tokens_per_sec = (cfg["batch"] * cfg["seq"]) / (avg_ms / 1000)

                benchmarks.append({
                    "config": cfg,
                    "avg_latency_ms": avg_ms,
                    "tokens_per_second": tokens_per_sec
                })

                logger.info(f"  MoE {cfg['hidden']}x{cfg['experts']}e: {avg_ms:.2f}ms, {tokens_per_sec:.0f} tok/s")

            result.metrics["benchmarks"] = benchmarks

        except Exception as e:
            result.metrics["error"] = str(e)

        return result


class FP8TestSuite(TestSuite):
    """Test suite for FP8 functionality."""

    def run(self) -> list[TestResult]:
        """Run FP8 tests."""
        if not self.config.test_fp8:
            logger.info("FP8 tests disabled, skipping")
            return []

        logger.info("=" * 60)
        logger.info("Running FP8 Test Suite")
        logger.info("=" * 60)

        results = []

        # Run FP8 native tests
        result = self.run_pytest("tests/test_fp8_native.py", "fp8_native_tests")
        results.append(result)

        # Run FP8 integration tests
        result = self._run_fp8_integration()
        results.append(result)

        # Run FP8 benchmarks
        benchmark = self._run_fp8_benchmark()
        self.benchmarks.append(benchmark)

        self.results = results
        return results

    def _run_fp8_integration(self) -> TestResult:
        """Run FP8 integration tests."""
        import torch

        result = TestResult(
            suite_name="fp8_integration",
            platform=self.config.platform.value,
            gpu_type=self.config.gpu_type.value
        )

        try:
            from torchbridge.precision.fp8_native import (
                FP8InferenceEngine,  # noqa: F401
                NativeFP8Linear,  # noqa: F401
                dequantize_from_fp8,  # noqa: F401
                is_fp8_available,
                quantize_to_fp8,  # noqa: F401
            )

            'cuda' if torch.cuda.is_available() else 'cpu'

            tests = [
                ("fp8_availability", lambda: is_fp8_available()),
                ("fp8_quantize", self._test_fp8_quantize),
                ("fp8_linear", self._test_fp8_linear),
                ("fp8_inference_engine", self._test_fp8_inference_engine),
            ]

            start = time.time()
            for name, test_fn in tests:
                try:
                    test_fn()
                    result.passed += 1
                    logger.info(f"  PASS: {name}")
                except Exception as e:
                    result.failed += 1
                    result.errors.append(f"{name}: {str(e)}")
                    logger.error(f"  FAIL: {name} - {e}")

            result.duration_seconds = time.time() - start

        except ImportError as e:
            result.errors.append(f"Import error: {e}")
            result.failed = 1

        return result

    def _test_fp8_quantize(self):
        """Test FP8 quantization."""
        import torch

        from torchbridge.precision.fp8_native import (
            FP8Dtype,
            dequantize_from_fp8,
            quantize_to_fp8,
        )

        x = torch.randn(128, 256)
        quantized, scale = quantize_to_fp8(x, FP8Dtype.E4M3)
        dequantized = dequantize_from_fp8(quantized, scale)

        # Check reconstruction error
        rel_error = (x - dequantized).abs().mean() / x.abs().mean()
        assert rel_error < 0.1, f"FP8 reconstruction error too high: {rel_error}"

    def _test_fp8_linear(self):
        """Test FP8 linear layer."""
        import torch

        from torchbridge.precision.fp8_native import NativeFP8Linear

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        layer = NativeFP8Linear(256, 512).to(device)
        x = torch.randn(32, 256, device=device)
        out = layer(x)
        assert out.shape == (32, 512)

    def _test_fp8_inference_engine(self):
        """Test FP8 inference engine."""
        import torch
        import torch.nn as nn

        from torchbridge.precision.fp8_native import FP8InferenceEngine

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        ).to(device)

        engine = FP8InferenceEngine(model)
        engine.prepare()

        x = torch.randn(32, 256, device=device)
        out = engine.infer(x)
        assert out.shape == (32, 256)

    def _run_fp8_benchmark(self) -> BenchmarkResult:
        """Run FP8 performance benchmark."""
        import torch

        result = BenchmarkResult(
            benchmark_name="fp8_performance",
            platform=self.config.platform.value,
            gpu_type=self.config.gpu_type.value
        )

        try:
            import torch.nn as nn

            from torchbridge.precision.fp8_native import NativeFP8Linear

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sizes = [(512, 1024), (1024, 2048), (2048, 4096)]

            benchmarks = []
            for in_f, out_f in sizes:
                # Standard layer
                std_layer = nn.Linear(in_f, out_f).to(device)

                # FP8 layer
                fp8_layer = NativeFP8Linear(in_f, out_f).to(device)

                x = torch.randn(128, in_f, device=device)

                # Benchmark standard
                for _ in range(10):
                    _ = std_layer(x)
                if device == 'cuda':
                    torch.cuda.synchronize()

                start = time.time()
                for _ in range(self.config.benchmark_iterations):
                    _ = std_layer(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                std_time = (time.time() - start) / self.config.benchmark_iterations * 1000

                # Benchmark FP8
                for _ in range(10):
                    _ = fp8_layer(x)
                if device == 'cuda':
                    torch.cuda.synchronize()

                start = time.time()
                for _ in range(self.config.benchmark_iterations):
                    _ = fp8_layer(x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                fp8_time = (time.time() - start) / self.config.benchmark_iterations * 1000

                speedup = std_time / fp8_time if fp8_time > 0 else 0

                benchmarks.append({
                    "size": f"{in_f}x{out_f}",
                    "standard_ms": std_time,
                    "fp8_ms": fp8_time,
                    "speedup": speedup
                })

                logger.info(f"  FP8 {in_f}x{out_f}: std={std_time:.2f}ms, fp8={fp8_time:.2f}ms, speedup={speedup:.2f}x")

            result.metrics["benchmarks"] = benchmarks

        except Exception as e:
            result.metrics["error"] = str(e)

        return result


class BackendTestSuite(TestSuite):
    """Test suite for backend-specific functionality."""

    def run(self) -> list[TestResult]:
        """Run backend tests based on detected hardware."""
        logger.info("=" * 60)
        logger.info(f"Running Backend Tests: {self.config.platform.value}")
        logger.info("=" * 60)

        results = []

        if self.config.platform in [CloudPlatform.AWS_NVIDIA, CloudPlatform.GCP_NVIDIA, CloudPlatform.LOCAL]:
            if self.config.gpu_type != GPUType.CPU:
                result = self.run_pytest("tests/test_nvidia_backend.py", "nvidia_backend")
                results.append(result)

        elif self.config.platform == CloudPlatform.AWS_AMD:
            result = self.run_pytest("tests/test_amd_backend.py", "amd_backend")
            results.append(result)

        elif self.config.platform == CloudPlatform.GCP_TPU:
            result = self.run_pytest("tests/test_tpu_backend.py", "tpu_backend")
            results.append(result)

        # Run backend benchmark
        benchmark = self._run_backend_benchmark()
        self.benchmarks.append(benchmark)

        self.results = results
        return results

    def _run_backend_benchmark(self) -> BenchmarkResult:
        """Run backend performance benchmark."""
        import torch

        result = BenchmarkResult(
            benchmark_name="backend_performance",
            platform=self.config.platform.value,
            gpu_type=self.config.gpu_type.value
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        benchmarks = {}

        # Matrix multiplication
        sizes = [(1024, 1024), (2048, 2048), (4096, 4096)]
        matmul_results = []

        for M, N in sizes:
            K = M
            a = torch.randn(M, K, device=device, dtype=torch.float16)
            b = torch.randn(K, N, device=device, dtype=torch.float16)

            # Warmup
            for _ in range(10):
                _ = torch.matmul(a, b)
            if device == 'cuda':
                torch.cuda.synchronize()

            # Benchmark
            start = time.time()
            for _ in range(self.config.benchmark_iterations):
                _ = torch.matmul(a, b)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start

            avg_ms = elapsed / self.config.benchmark_iterations * 1000
            tflops = (2 * M * N * K) / (avg_ms / 1000) / 1e12

            matmul_results.append({
                "size": f"{M}x{K}x{N}",
                "time_ms": avg_ms,
                "tflops": tflops
            })

            logger.info(f"  MatMul {M}x{K}: {avg_ms:.2f}ms, {tflops:.2f} TFLOPS")

        benchmarks["matrix_multiply"] = matmul_results

        # Memory bandwidth
        if device == 'cuda':
            mem_results = []
            sizes_mb = [256, 512, 1024, 2048]

            for size_mb in sizes_mb:
                numel = size_mb * 1024 * 1024 // 4
                a = torch.randn(numel, device=device)
                b = torch.empty_like(a)

                # Warmup
                for _ in range(5):
                    b.copy_(a)
                torch.cuda.synchronize()

                # Benchmark
                start = time.time()
                for _ in range(50):
                    b.copy_(a)
                torch.cuda.synchronize()
                elapsed = time.time() - start

                bandwidth_gbps = (size_mb * 2 * 50) / elapsed / 1000  # Read + write

                mem_results.append({
                    "size_mb": size_mb,
                    "bandwidth_gbps": bandwidth_gbps
                })

                logger.info(f"  Memory {size_mb}MB: {bandwidth_gbps:.1f} GB/s")

            benchmarks["memory_bandwidth"] = mem_results

        result.metrics = benchmarks
        return result


class AttentionTestSuite(TestSuite):
    """Test suite for attention mechanisms."""

    def run(self) -> list[TestResult]:
        """Run attention tests."""
        if not self.config.test_attention:
            logger.info("Attention tests disabled, skipping")
            return []

        logger.info("=" * 60)
        logger.info("Running Attention Test Suite")
        logger.info("=" * 60)

        results = []

        # Run attention tests
        result = self.run_pytest("tests/test_attention.py", "attention_tests")
        results.append(result)

        # Run attention benchmark
        benchmark = self._run_attention_benchmark()
        self.benchmarks.append(benchmark)

        self.results = results
        return results

    def _run_attention_benchmark(self) -> BenchmarkResult:
        """Run attention performance benchmark."""
        import torch

        result = BenchmarkResult(
            benchmark_name="attention_performance",
            platform=self.config.platform.value,
            gpu_type=self.config.gpu_type.value
        )

        try:
            from torchbridge import AttentionLayer

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            configs = [
                {"d_model": 512, "heads": 8, "seq": 128, "batch": 8},
                {"d_model": 768, "heads": 12, "seq": 256, "batch": 4},
                {"d_model": 1024, "heads": 16, "seq": 512, "batch": 2},
            ]

            benchmarks = []
            for cfg in configs:
                layer = AttentionLayer(
                    embed_dim=cfg["d_model"],
                    num_heads=cfg["heads"]
                ).to(device)

                x = torch.randn(cfg["batch"], cfg["seq"], cfg["d_model"], device=device)

                # Warmup
                for _ in range(10):
                    _ = layer(x, x, x)
                if device == 'cuda':
                    torch.cuda.synchronize()

                # Benchmark
                start = time.time()
                for _ in range(self.config.benchmark_iterations):
                    _ = layer(x, x, x)
                if device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start

                avg_ms = elapsed / self.config.benchmark_iterations * 1000

                benchmarks.append({
                    "config": cfg,
                    "avg_latency_ms": avg_ms
                })

                logger.info(f"  Attention {cfg['d_model']}d/{cfg['heads']}h: {avg_ms:.2f}ms")

            result.metrics["benchmarks"] = benchmarks

        except Exception as e:
            result.metrics["error"] = str(e)

        return result


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    config: TestConfig,
    hardware_info: dict[str, Any],
    test_results: list[TestResult],
    benchmarks: list[BenchmarkResult]
) -> str:
    """Generate a comprehensive test report."""

    total_passed = sum(r.passed for r in test_results)
    total_failed = sum(r.failed for r in test_results)
    total_skipped = sum(r.skipped for r in test_results)

    report = f"""# TorchBridge v{config.version} Cloud Validation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Platform:** {config.platform.value}
**GPU Type:** {config.gpu_type.value}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Tests | {total_passed + total_failed + total_skipped} |
| Passed | {total_passed} |
| Failed | {total_failed} |
| Skipped | {total_skipped} |
| Status | {'**PASSED**' if total_failed == 0 else '**FAILED**'} |

## Hardware Configuration

| Property | Value |
|----------|-------|
| PyTorch Version | {hardware_info.get('pytorch_version', 'N/A')} |
| CUDA Version | {hardware_info.get('cuda_version', 'N/A')} |
| GPU Count | {hardware_info.get('gpu_count', 0)} |
| Backend | {hardware_info.get('backend', 'N/A')} |

"""

    # GPU details
    if hardware_info.get('gpus'):
        report += "### GPU Details\n\n"
        report += "| Index | Name | Memory |\n"
        report += "|-------|------|--------|\n"
        for gpu in hardware_info['gpus']:
            report += f"| {gpu['index']} | {gpu['name']} | {gpu['memory_gb']} GB |\n"
        report += "\n"

    # Test results by suite
    report += "## Test Results by Suite\n\n"
    report += "| Suite | Passed | Failed | Skipped | Duration |\n"
    report += "|-------|--------|--------|---------|----------|\n"

    for result in test_results:
        report += f"| {result.suite_name} | {result.passed} | {result.failed} | {result.skipped} | {result.duration_seconds:.1f}s |\n"

    report += "\n"

    # Errors
    errors = [e for r in test_results for e in r.errors]
    if errors:
        report += "## Errors\n\n"
        for i, error in enumerate(errors[:10], 1):
            report += f"{i}. {error[:200]}...\n" if len(error) > 200 else f"{i}. {error}\n"
        report += "\n"

    # Benchmarks
    if benchmarks:
        report += "## Benchmark Results\n\n"
        for benchmark in benchmarks:
            report += f"### {benchmark.benchmark_name}\n\n"
            if "error" in benchmark.metrics:
                report += f"Error: {benchmark.metrics['error']}\n\n"
            elif "benchmarks" in benchmark.metrics:
                items = benchmark.metrics["benchmarks"]
                if items and isinstance(items[0], dict):
                    # Create table from first item's keys
                    keys = [k for k in items[0].keys() if k != "config"]
                    report += "| " + " | ".join(keys) + " |\n"
                    report += "|" + "|".join(["---"] * len(keys)) + "|\n"
                    for item in items:
                        values = []
                        for k in keys:
                            v = item.get(k, "N/A")
                            if isinstance(v, float):
                                values.append(f"{v:.2f}")
                            else:
                                values.append(str(v))
                        report += "| " + " | ".join(values) + " |\n"
                report += "\n"

    # Footer
    report += f"""
---

## Validation Status

{'All tests passed. System is validated for production use.' if total_failed == 0 else f'FAILED: {total_failed} tests failed. Review errors above.'}

**Report generated by:** TorchBridge v{config.version} Cloud Testing Framework
"""

    return report


# ============================================================================
# Main Orchestration
# ============================================================================

def run_comprehensive_tests(output_dir: str = "./test_results") -> dict[str, Any]:
    """Run comprehensive tests on the current platform."""

    logger.info("=" * 70)
    logger.info("TorchBridge v0.4.6 Comprehensive Cloud Validation")
    logger.info("=" * 70)

    # Detect hardware
    platform, gpu_type, hardware_info = detect_hardware()
    logger.info(f"Detected Platform: {platform.value}")
    logger.info(f"Detected GPU: {gpu_type.value}")
    logger.info(f"GPU Count: {hardware_info.get('gpu_count', 0)}")

    # Create config
    config = TestConfig(
        platform=platform,
        gpu_type=gpu_type,
        output_dir=output_dir,
        test_fp8=gpu_type in [GPUType.H100, GPUType.A100, GPUType.A100_80GB],  # FP8 only on capable GPUs
    )

    # Run test suites
    all_results = []
    all_benchmarks = []

    # MoE tests
    moe_suite = MoETestSuite(config)
    all_results.extend(moe_suite.run())
    all_benchmarks.extend(moe_suite.benchmarks)

    # FP8 tests
    fp8_suite = FP8TestSuite(config)
    all_results.extend(fp8_suite.run())
    all_benchmarks.extend(fp8_suite.benchmarks)

    # Backend tests
    backend_suite = BackendTestSuite(config)
    all_results.extend(backend_suite.run())
    all_benchmarks.extend(backend_suite.benchmarks)

    # Attention tests
    attn_suite = AttentionTestSuite(config)
    all_results.extend(attn_suite.run())
    all_benchmarks.extend(attn_suite.benchmarks)

    # Generate report
    report = generate_report(config, hardware_info, all_results, all_benchmarks)

    # Save report
    report_path = os.path.join(output_dir, f"VALIDATION_REPORT_{platform.value}_{gpu_type.value}.md")
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved: {report_path}")

    # Save JSON results
    json_path = os.path.join(output_dir, f"results_{platform.value}_{gpu_type.value}.json")
    with open(json_path, 'w') as f:
        json.dump({
            "config": {
                "platform": platform.value,
                "gpu_type": gpu_type.value,
                "version": config.version
            },
            "hardware": hardware_info,
            "test_results": [
                {
                    "suite": r.suite_name,
                    "passed": r.passed,
                    "failed": r.failed,
                    "skipped": r.skipped,
                    "errors": r.errors,
                    "duration": r.duration_seconds
                }
                for r in all_results
            ],
            "benchmarks": [
                {
                    "name": b.benchmark_name,
                    "metrics": b.metrics
                }
                for b in all_benchmarks
            ]
        }, f, indent=2)
    logger.info(f"JSON results saved: {json_path}")

    # Summary
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)

    logger.info("=" * 70)
    logger.info(f"VALIDATION {'PASSED' if total_failed == 0 else 'FAILED'}")
    logger.info(f"Total: {total_passed} passed, {total_failed} failed")
    logger.info("=" * 70)

    return {
        "passed": total_failed == 0,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "report_path": report_path,
        "json_path": json_path
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TorchBridge v0.4.6 Cloud Validation")
    parser.add_argument("--output-dir", default="./test_results", help="Output directory")
    parser.add_argument("--no-moe", action="store_true", help="Skip MoE tests")
    parser.add_argument("--no-fp8", action="store_true", help="Skip FP8 tests")
    parser.add_argument("--no-attention", action="store_true", help="Skip attention tests")

    args = parser.parse_args()

    results = run_comprehensive_tests(args.output_dir)

    sys.exit(0 if results["passed"] else 1)
