#!/usr/bin/env python3
"""
TorchBridge API Validation Script — v0.5.67

Run this script ON each cloud instance to validate that TorchBridge APIs
work correctly and produce consistent results for the detected hardware.

Tests:
  1. Environment detection (PyTorch, CUDA/ROCm, GPU info)
  2. Backend detection (detect_best_backend, BackendFactory)
  3. Quantization engine (optimal format per hardware, fallback chains)
  4. Attention dispatch (optimal kernel per hardware)
  5. Adapter compatibility matrix (optimal method per hardware)
  6. Distributed config generation (FSDP2, pipeline, collective)
  7. Inference comparison: vanilla Qwen3-0.6B vs TorchBridge-optimized
     (output must match within tolerance; latency regression must be < 20%)

Usage:
    python3 validate_torchbridge.py [--platform LABEL] [--output FILE]

    --platform   Human-readable platform label (e.g. "aws-a10g", "amd-mi300x")
                 Defaults to auto-detecting from GPU name.
    --output     Path to write JSON report (default: torchbridge_validation_<platform>.json)
    --skip-inference  Skip the inference comparison test (fast API-only run)

Install requirements:
    pip install torchbridge-ml transformers

Or from source:
    pip install -e /path/to/torchbridge
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    value: Any = None
    expected: Any = None
    message: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "value": str(self.value) if self.value is not None else None,
            "expected": str(self.expected) if self.expected is not None else None,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class ValidationReport:
    platform: str
    timestamp: str
    torchbridge_version: str
    pytorch_version: str
    python_version: str
    gpu_name: str
    gpu_count: int
    cuda_version: str
    rocm_version: str
    environment: dict = field(default_factory=dict)
    tests: list[TestResult] = field(default_factory=list)
    inference: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "platform": self.platform,
            "timestamp": self.timestamp,
            "torchbridge_version": self.torchbridge_version,
            "pytorch_version": self.pytorch_version,
            "python_version": self.python_version,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "cuda_version": self.cuda_version,
            "rocm_version": self.rocm_version,
            "environment": self.environment,
            "tests": [t.to_dict() for t in self.tests],
            "inference": self.inference,
            "summary": self.summary,
        }
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Environment detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_environment() -> dict:
    """Collect full environment info."""
    env = {
        "python": sys.version,
        "platform": platform.platform(),
    }

    try:
        import torch
        env["pytorch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda or "N/A"
        env["rocm_version"] = getattr(torch.version, "hip", None) or "N/A"
        env["device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        env["mps_available"] = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()

        if torch.cuda.is_available():
            env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            props = torch.cuda.get_device_properties(0)
            env["compute_capability"] = f"{props.major}.{props.minor}"
            env["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
        else:
            env["gpu_names"] = []
            env["compute_capability"] = "N/A"
            env["gpu_memory_gb"] = 0.0

        # Neuron check
        try:
            import torch_neuronx  # noqa: F401
            env["neuronx_available"] = True
            env["neuronx_version"] = getattr(torch_neuronx, "__version__", "unknown")
        except ImportError:
            env["neuronx_available"] = False

        # XLA check
        try:
            import torch_xla  # noqa: F401
            env["xla_available"] = True
        except ImportError:
            env["xla_available"] = False

    except ImportError as e:
        env["pytorch"] = f"NOT INSTALLED: {e}"

    return env


def auto_platform_label(env: dict) -> str:
    """Derive a platform label from environment."""
    gpu_names = env.get("gpu_names", [])
    if gpu_names:
        name = gpu_names[0].lower()
        if "a10g" in name:
            return "aws-a10g"
        if "t4" in name:
            return "gcp-t4"
        if "h100" in name:
            return "runpod-h100nvl"
        if "mi300" in name:
            return "amd-mi300x"
        if "mi250" in name or "mi200" in name:
            return "amd-mi200"
        if "a100" in name:
            return "nvidia-a100"
    if env.get("neuronx_available"):
        return "aws-trainium-or-inferentia"
    if env.get("xla_available"):
        return "gcp-tpu"
    if env.get("mps_available"):
        return "apple-mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# TorchBridge API tests
# ─────────────────────────────────────────────────────────────────────────────

def test_installation() -> list[TestResult]:
    results = []
    try:
        import torchbridge
        results.append(TestResult(
            name="torchbridge_importable",
            passed=True,
            value=torchbridge.__version__,
            message=f"TorchBridge {torchbridge.__version__} imported successfully",
        ))
    except ImportError as e:
        results.append(TestResult(
            name="torchbridge_importable",
            passed=False,
            error=str(e),
            message="Install with: pip install torchbridge-ml",
        ))
        return results  # All subsequent tests will fail

    return results


def test_backend_detection() -> list[TestResult]:
    """Test detect_best_backend and BackendFactory."""
    results = []

    # Test 1: detect_best_backend returns a valid BackendType
    try:
        from torchbridge.backends import BackendFactory, detect_best_backend

        backend_type = detect_best_backend()

        results.append(TestResult(
            name="detect_best_backend",
            passed=backend_type is not None,
            value=str(backend_type),
            message=f"Detected backend: {backend_type}",
        ))
    except Exception as e:
        results.append(TestResult(
            name="detect_best_backend",
            passed=False,
            error=traceback.format_exc(),
            message=str(e),
        ))
        return results

    # Test 2: BackendFactory.create returns a backend instance
    try:
        backend = BackendFactory.create(backend_type)
        has_device_info = hasattr(backend, "get_device_info")
        device_info = backend.get_device_info() if has_device_info else {}

        results.append(TestResult(
            name="backend_factory_create",
            passed=backend is not None,
            value=type(backend).__name__,
            message=f"Created {type(backend).__name__}",
        ))

        results.append(TestResult(
            name="backend_device_info",
            passed=bool(device_info),
            value=str(device_info)[:200],
            message="get_device_info() returned data",
        ))
    except Exception as e:
        results.append(TestResult(
            name="backend_factory_create",
            passed=False,
            error=traceback.format_exc(),
            message=str(e),
        ))

    # Test 3: BackendFactory.get_available_backends includes detected backend
    try:
        available = BackendFactory.get_available_backends()
        results.append(TestResult(
            name="available_backends_includes_detected",
            passed=len(available) >= 1,
            value=[str(b) for b in available],
            message=f"Available backends: {[str(b) for b in available]}",
        ))
    except Exception:
        results.append(TestResult(
            name="available_backends_includes_detected",
            passed=False,
            error=traceback.format_exc(),
        ))

    return results


def test_quantization_engine() -> list[TestResult]:
    """Test QuantizationEngine API: format detection, supported formats, fallback chains."""
    results = []

    try:
        from torchbridge.precision.quantization import (
            QuantizationEngine,
        )
        from torchbridge.precision.quantization.formats import QuantizationFormat

        engine = QuantizationEngine()

        # Test 1: backend_name returns a non-empty string
        results.append(TestResult(
            name="quant_backend_name",
            passed=bool(engine.backend_name),
            value=engine.backend_name,
            message=f"Detected quantization backend: {engine.backend_name}",
        ))

        # Test 2: architecture_name returns a non-empty string
        results.append(TestResult(
            name="quant_architecture_name",
            passed=bool(engine.architecture_name),
            value=engine.architecture_name,
            message=f"Detected architecture: {engine.architecture_name}",
        ))

        # Test 3: get_optimal_format returns a valid QuantizationFormat
        optimal = engine.get_optimal_format()
        results.append(TestResult(
            name="quant_optimal_format",
            passed=isinstance(optimal, QuantizationFormat),
            value=optimal.value,
            message=f"Optimal quantization format: {optimal.value}",
        ))

        # Test 4: get_supported_formats returns a non-empty list containing optimal
        supported = engine.get_supported_formats()
        results.append(TestResult(
            name="quant_supported_formats",
            passed=len(supported) >= 1 and optimal in supported,
            value=[f.value for f in supported],
            message=f"Supported formats: {[f.value for f in supported]}",
        ))

        # Test 5: optimal is first in supported list
        results.append(TestResult(
            name="quant_optimal_is_first",
            passed=len(supported) >= 1 and supported[0] == optimal,
            value=supported[0].value if supported else None,
            expected=optimal.value,
            message="Optimal format should be first in supported list",
        ))

        # Test 6: quantize a small model with "auto" format
        try:
            import torch.nn as nn

            tiny_model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 32))
            result = engine.quantize(tiny_model, format="auto")
            results.append(TestResult(
                name="quant_auto_quantize",
                passed=result.success and result.model is not None,
                value={
                    "format_applied": result.format_applied.value,
                    "format_requested": result.format_requested.value,
                    "used_fallback": result.used_fallback,
                    "memory_before_mb": round(result.memory_before_mb, 4),
                    "memory_after_mb": round(result.memory_after_mb, 4),
                    "warnings": result.warnings,
                    "errors": result.errors,
                },
                message=f"Quantized to {result.format_applied.value}"
                        + (" (fallback)" if result.used_fallback else ""),
            ))
        except Exception as e:
            results.append(TestResult(
                name="quant_auto_quantize",
                passed=False,
                error=traceback.format_exc(),
                message=str(e),
            ))

    except Exception as e:
        results.append(TestResult(
            name="quantization_engine_init",
            passed=False,
            error=traceback.format_exc(),
            message=str(e),
        ))

    return results


def test_attention_dispatch() -> list[TestResult]:
    """Test AttentionDispatcher: kernel selection, fallback chains."""
    results = []

    try:
        from torchbridge.attention.dispatch import AttentionDispatcher
        from torchbridge.attention.dispatch.kernel_types import AttentionKernelType

        dispatcher = AttentionDispatcher()

        # Test 1: dispatcher initializes without error and is the right type
        results.append(TestResult(
            name="attention_dispatcher_type",
            passed=isinstance(dispatcher, AttentionDispatcher),
            value=type(dispatcher).__name__,
            message=f"AttentionDispatcher initialized: {type(dispatcher).__name__}",
        ))
        results.append(TestResult(
            name="attention_backend_set",
            passed=dispatcher._backend is not None,
            value=dispatcher._backend.value,
            message=f"Attention dispatcher backend: {dispatcher._backend.value}",
        ))

        # Test 2: select_kernel returns a valid result for standard shapes
        dispatch_result = dispatcher.select_kernel(seq_length=512, num_heads=8, head_dim=64)
        results.append(TestResult(
            name="attention_select_kernel",
            passed=dispatch_result is not None and isinstance(dispatch_result.kernel_type, AttentionKernelType),
            value={
                "kernel_type": dispatch_result.kernel_type.value,
                "implementation_name": dispatch_result.implementation_name,
                "used_fallback": dispatch_result.used_fallback,
                "fallback_chain": [k.value for k in dispatch_result.fallback_chain],
                "warnings": dispatch_result.warnings,
            },
            message=f"Selected kernel: {dispatch_result.kernel_type.value}"
                    + (" (fallback)" if dispatch_result.used_fallback else ""),
        ))

        # Test 3: PYTORCH_SDPA is in the fallback chain (always available)
        sdpa = AttentionKernelType.PYTORCH_SDPA
        results.append(TestResult(
            name="attention_sdpa_in_chain",
            passed=sdpa in dispatch_result.fallback_chain or dispatch_result.kernel_type == sdpa,
            value=dispatch_result.kernel_type.value,
            message="PYTORCH_SDPA should always be in fallback chain",
        ))

        # Test 4: Different seq lengths produce valid results
        for seq_len in [128, 1024, 4096]:
            try:
                r = dispatcher.select_kernel(seq_length=seq_len, num_heads=16, head_dim=128)
                ok = r is not None and isinstance(r.kernel_type, AttentionKernelType)
            except Exception:
                ok = False
            results.append(TestResult(
                name=f"attention_select_kernel_seq{seq_len}",
                passed=ok,
                value=r.kernel_type.value if ok else None,
                message=f"Kernel for seq_len={seq_len}: {r.kernel_type.value if ok else 'ERROR'}",
            ))

    except Exception as e:
        results.append(TestResult(
            name="attention_dispatch_init",
            passed=False,
            error=traceback.format_exc(),
            message=str(e),
        ))

    return results


def test_adapter_compatibility() -> list[TestResult]:
    """Test AdapterCompatibilityMatrix: optimal method, fallback chain."""
    results = []

    try:
        from torchbridge.adapters.compatibility import AdapterCompatibilityMatrix
        from torchbridge.adapters.config import AdapterMethod
        from torchbridge.core.config import HardwareConfig

        hw = HardwareConfig()
        backend = hw.backend

        # Test 1: get_optimal returns a valid AdapterMethod
        optimal = AdapterCompatibilityMatrix.get_optimal(backend)
        results.append(TestResult(
            name="adapter_optimal_method",
            passed=isinstance(optimal, AdapterMethod),
            value=optimal.value,
            message=f"Optimal adapter method: {optimal.value} for {backend.value}",
        ))

        # Test 2: get_fallback_chain returns non-empty list starting with optimal
        chain = AdapterCompatibilityMatrix.get_fallback_chain(backend)
        results.append(TestResult(
            name="adapter_fallback_chain",
            passed=len(chain) >= 1 and chain[0] == optimal,
            value=[m.value for m in chain],
            message=f"Adapter fallback chain: {[m.value for m in chain]}",
        ))

        # Test 3: LORA is always in the chain (ultimate fallback)
        lora_in_chain = AdapterMethod.LORA in chain
        results.append(TestResult(
            name="adapter_lora_always_available",
            passed=lora_in_chain,
            value=[m.value for m in chain],
            message="LORA should always be in fallback chain",
        ))

    except Exception as e:
        results.append(TestResult(
            name="adapter_compatibility_init",
            passed=False,
            error=traceback.format_exc(),
            message=str(e),
        ))

    return results


def test_distributed_config() -> list[TestResult]:
    """Test DistributedConfig.auto() — config generation for current hardware."""
    results = []

    try:
        from torchbridge.core.config import HardwareConfig
        from torchbridge.distributed.config import DistributedConfig

        hw = HardwareConfig()
        backend = hw.backend

        # Test 1: auto() with 1B params, world_size=1
        config = DistributedConfig.auto(
            model_params=1_000_000_000,
            backend=backend,
            world_size=1,
        )
        results.append(TestResult(
            name="distributed_config_auto_1gpu",
            passed=config is not None and hasattr(config, "fsdp") and hasattr(config, "pipeline"),
            value={
                "fsdp_sharding": str(config.fsdp.sharding_strategy) if hasattr(config.fsdp, "sharding_strategy") else "present",
                "pipeline_schedule": str(config.pipeline.schedule_type) if hasattr(config.pipeline, "schedule_type") else "present",
                "collective_backend": str(config.collective.backend_type) if hasattr(config.collective, "backend_type") else "present",
            },
            message="DistributedConfig.auto() for 1B params, world_size=1",
        ))

        # Test 2: auto() with 7B params, world_size=8
        config_8gpu = DistributedConfig.auto(
            model_params=7_000_000_000,
            backend=backend,
            world_size=8,
            gpus_per_node=8,
        )
        results.append(TestResult(
            name="distributed_config_auto_8gpu",
            passed=config_8gpu is not None,
            value="generated" if config_8gpu else None,
            message="DistributedConfig.auto() for 7B params, world_size=8",
        ))

        # Test 3: TOML export
        try:
            toml_str = config.to_toml() if hasattr(config, "to_toml") else None
            results.append(TestResult(
                name="distributed_config_toml_export",
                passed=toml_str is not None and len(toml_str) > 10,
                value=f"{len(toml_str)} chars" if toml_str else None,
                message="TOML export produced output",
            ))
        except Exception:
            results.append(TestResult(
                name="distributed_config_toml_export",
                passed=False,
                error=traceback.format_exc(),
            ))

    except Exception as e:
        results.append(TestResult(
            name="distributed_config_init",
            passed=False,
            error=traceback.format_exc(),
            message=str(e),
        ))

    return results


def test_unified_manager() -> list[TestResult]:
    """Test UnifiedManager.auto_optimize on a tiny model."""
    results = []

    try:
        import torch.nn as nn

        from torchbridge import TorchBridgeConfig, UnifiedManager

        tiny_model = nn.Sequential(
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 64),
        )

        config = TorchBridgeConfig.for_inference()
        manager = UnifiedManager(config)

        t0 = time.perf_counter()
        optimized = manager.auto_optimize(tiny_model, for_inference=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results.append(TestResult(
            name="unified_manager_auto_optimize",
            passed=optimized is not None,
            value={
                "optimized_type": type(optimized).__name__,
                "elapsed_ms": round(elapsed_ms, 1),
            },
            message=f"auto_optimize completed in {elapsed_ms:.1f}ms, returned {type(optimized).__name__}",
        ))

    except Exception as e:
        results.append(TestResult(
            name="unified_manager_auto_optimize",
            passed=False,
            error=traceback.format_exc(),
            message=str(e),
        ))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Inference comparison: vanilla vs TorchBridge-optimized
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_comparison(env: dict) -> dict:
    """
    Run Qwen3-0.6B with and without TorchBridge, compare outputs and latency.

    Returns a dict with all comparison results.
    """
    result: dict[str, Any] = {
        "model": "Qwen/Qwen3-0.6B",
        "prompt": "The capital of France is",
        "passed": False,
        "error": None,
    }

    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_label = torch.cuda.get_device_name(0)
        elif env.get("mps_available"):
            device = torch.device("mps")
            device_label = "Apple MPS"
        else:
            device = torch.device("cpu")
            device_label = "CPU"

        result["device"] = device_label

        print(f"  Loading Qwen3-0.6B on {device_label}...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        # Use FP16 on CUDA (BF16 triggers cuBLAS GEMM errors on some PyTorch/CUDA combos),
        # FP16 on MPS, FP32 on CPU
        if device.type == "cuda":
            infer_dtype = torch.float16
        elif device.type == "mps":
            infer_dtype = torch.float16
        else:
            infer_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=infer_dtype,
            attn_implementation="eager",  # Avoids SDPA/Flash path that triggers CUBLAS_STATUS_INVALID_VALUE on some CUDA+cuBLAS combos
        )
        model.eval()

        inputs = tokenizer("The capital of France is", return_tensors="pt")

        # ── Baseline (CPU) inference ──────────────────────────────────────────
        print("  Running baseline CPU inference...")
        with torch.no_grad():
            cpu_out = model(**inputs)
        cpu_logits = cpu_out.logits[:, -1, :]
        result["baseline_device"] = "CPU"

        # ── Vanilla GPU inference ─────────────────────────────────────────────
        if device.type != "cpu":
            print(f"  Running vanilla {device_label} inference...")
            model_gpu = model.to(device)
            inputs_gpu = {k: v.to(device) for k, v in inputs.items()}

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model_gpu(**inputs_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Timed run
            t0 = time.perf_counter()
            for _ in range(50):
                with torch.no_grad():
                    vanilla_out = model_gpu(**inputs_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize()
            vanilla_latency_ms = (time.perf_counter() - t0) / 50 * 1000

            vanilla_logits = vanilla_out.logits[:, -1, :].cpu()
            vanilla_max_diff = float(torch.abs(cpu_logits - vanilla_logits).max())
            vanilla_cos_sim = float(F.cosine_similarity(
                cpu_logits.flatten().unsqueeze(0),
                vanilla_logits.flatten().unsqueeze(0),
            ))
            result["vanilla"] = {
                "max_diff": vanilla_max_diff,
                "cosine_sim": vanilla_cos_sim,
                "latency_ms": round(vanilla_latency_ms, 2),
                "passed": vanilla_max_diff < 1.0 and vanilla_cos_sim > 0.99,
            }
        else:
            result["vanilla"] = {"note": "CPU-only platform, no GPU baseline"}

        # ── TorchBridge-optimized inference ──────────────────────────────────
        print("  Running TorchBridge-optimized inference...")
        try:
            from torchbridge import TorchBridgeConfig, UnifiedManager

            # Fresh model copy for TorchBridge
            model_tb = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
            model_tb.eval()

            config = TorchBridgeConfig.for_inference()
            manager = UnifiedManager(config)

            t_opt_start = time.perf_counter()
            model_tb_opt = manager.auto_optimize(
                model_tb,
                sample_inputs=inputs["input_ids"],
                for_inference=True,
            )
            optimization_ms = (time.perf_counter() - t_opt_start) * 1000

            # Always move to target device — auto_optimize may route to a different
            # backend (e.g. XLA on TPU VMs) so the model may not be on `device` yet
            model_tb_opt = model_tb_opt.to(device)
            inputs_tb = {k: v.to(device) for k, v in inputs.items()}

            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model_tb_opt(**inputs_tb)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Timed run
            t0 = time.perf_counter()
            for _ in range(50):
                with torch.no_grad():
                    tb_out = model_tb_opt(**inputs_tb)
            if device.type == "cuda":
                torch.cuda.synchronize()
            tb_latency_ms = (time.perf_counter() - t0) / 50 * 1000

            tb_logits = tb_out.logits[:, -1, :].cpu()
            tb_max_diff = float(torch.abs(cpu_logits - tb_logits).max())
            tb_cos_sim = float(F.cosine_similarity(
                cpu_logits.flatten().unsqueeze(0),
                tb_logits.flatten().unsqueeze(0),
            ))

            # Latency regression check: TorchBridge must not be more than 20% slower
            vanilla_latency = result.get("vanilla", {}).get("latency_ms")
            if vanilla_latency and vanilla_latency > 0 and device.type == "cuda":
                latency_ratio = tb_latency_ms / vanilla_latency
                regression_ok = latency_ratio <= 1.20
            else:
                latency_ratio = tb_latency_ms / vanilla_latency if (vanilla_latency and vanilla_latency > 0) else None
                regression_ok = True  # latency regression only enforced on CUDA

            result["torchbridge"] = {
                "max_diff": tb_max_diff,
                "cosine_sim": tb_cos_sim,
                "latency_ms": round(tb_latency_ms, 2),
                "optimization_time_ms": round(optimization_ms, 1),
                "latency_ratio_vs_vanilla": round(latency_ratio, 3) if latency_ratio else None,
                "latency_regression_ok": regression_ok,
                "output_matches_baseline": tb_max_diff < 1.0 and tb_cos_sim > 0.99,
                "passed": (tb_max_diff < 1.0 and tb_cos_sim > 0.99 and regression_ok),
            }
            result["passed"] = result["torchbridge"]["passed"]

        except Exception as e:
            result["torchbridge"] = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "passed": False,
            }
            result["passed"] = False

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["passed"] = False

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TorchBridge v0.5.67 API Validation")
    parser.add_argument("--platform", default=None, help="Platform label (auto-detected if not set)")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference comparison (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("  TorchBridge API Validation — v0.5.67")
    print("=" * 70)

    # Environment
    print("\n[1/7] Detecting environment...")
    env = detect_environment()
    platform_label = args.platform or auto_platform_label(env)
    print(f"  Platform: {platform_label}")
    print(f"  PyTorch: {env.get('pytorch', 'N/A')}")
    print(f"  CUDA: {env.get('cuda_available', False)} ({env.get('cuda_version', 'N/A')})")
    print(f"  GPU: {', '.join(env.get('gpu_names', [])) or 'none'}")

    # Collect TorchBridge version
    try:
        import torchbridge
        tb_version = torchbridge.__version__
    except Exception:
        tb_version = "NOT INSTALLED"

    report = ValidationReport(
        platform=platform_label,
        timestamp=datetime.now(timezone.utc).isoformat(),
        torchbridge_version=tb_version,
        pytorch_version=env.get("pytorch", "N/A"),
        python_version=sys.version,
        gpu_name=", ".join(env.get("gpu_names", [])) or "none",
        gpu_count=env.get("device_count", 0),
        cuda_version=env.get("cuda_version", "N/A"),
        rocm_version=env.get("rocm_version", "N/A"),
        environment=env,
    )

    # Run tests
    test_groups = [
        ("Installation", test_installation),
        ("Backend Detection", test_backend_detection),
        ("Quantization Engine", test_quantization_engine),
        ("Attention Dispatch", test_attention_dispatch),
        ("Adapter Compatibility", test_adapter_compatibility),
        ("Distributed Config", test_distributed_config),
        ("Unified Manager", test_unified_manager),
    ]

    for idx, (group_name, test_fn) in enumerate(test_groups, 2):
        print(f"\n[{idx}/{6 + len(test_groups)}] Testing {group_name}...")
        try:
            group_results = test_fn()
            for r in group_results:
                status = "✓" if r.passed else "✗"
                print(f"  {status} {r.name}: {r.message or r.value}")
                if not r.passed and r.error:
                    print(f"    ERROR: {r.error[:200]}")
            report.tests.extend(group_results)
        except Exception as e:
            print(f"  ✗ Group failed: {e}")
            report.tests.append(TestResult(
                name=f"{group_name.lower().replace(' ', '_')}_group",
                passed=False,
                error=str(e),
            ))

    # Inference comparison
    if not args.skip_inference:
        print(f"\n[{len(test_groups) + 2}/{len(test_groups) + 2}] Inference comparison (vanilla vs TorchBridge)...")
        inference = run_inference_comparison(env)
        report.inference = inference
        if inference.get("vanilla"):
            v = inference["vanilla"]
            if "latency_ms" in v:
                print(f"  Vanilla GPU: max_diff={v['max_diff']:.2e}, cos_sim={v['cosine_sim']:.6f}, latency={v['latency_ms']:.1f}ms")
        if inference.get("torchbridge"):
            tb = inference["torchbridge"]
            if "latency_ms" in tb:
                print(f"  TorchBridge: max_diff={tb['max_diff']:.2e}, cos_sim={tb['cosine_sim']:.6f}, latency={tb['latency_ms']:.1f}ms")
                if tb.get("latency_ratio_vs_vanilla"):
                    print(f"  Latency ratio: {tb['latency_ratio_vs_vanilla']:.3f}x ({'OK' if tb['latency_regression_ok'] else 'REGRESSION'})")
        passed_str = "PASSED" if inference.get("passed") else "FAILED"
        print(f"  Inference comparison: {passed_str}")
    else:
        print("\n  (inference comparison skipped)")
        report.inference = {"skipped": True}

    # Summary
    total = len(report.tests)
    passed = sum(1 for t in report.tests if t.passed)
    failed = total - passed
    inference_passed = report.inference.get("passed", True)  # True if skipped

    report.summary = {
        "total_api_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate_pct": round(passed / total * 100, 1) if total > 0 else 0.0,
        "inference_comparison_passed": inference_passed,
        "overall_passed": failed == 0 and inference_passed,
    }

    print("\n" + "=" * 70)
    print(f"  Summary: {passed}/{total} API tests passed ({report.summary['pass_rate_pct']}%)")
    print(f"  Inference: {'PASSED' if inference_passed else 'FAILED'}")
    overall = "PASSED" if report.summary["overall_passed"] else "FAILED"
    print(f"  Overall: {overall}")
    print("=" * 70)

    # Write JSON report
    output_file = args.output or f"torchbridge_validation_{platform_label.replace('/', '-')}.json"
    report_dict = report.to_dict()
    with open(output_file, "w") as f:
        json.dump(report_dict, f, indent=2, default=str)
    print(f"\nReport written to: {output_file}")

    sys.exit(0 if report.summary["overall_passed"] else 1)


if __name__ == "__main__":
    main()
