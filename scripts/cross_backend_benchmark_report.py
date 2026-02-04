#!/usr/bin/env python3
"""
TorchBridge Cross-Backend Benchmark Report Generator

Detects available backends and runs forward-pass benchmarks on small/medium
models, generating a markdown or JSON report with latency, throughput,
memory usage, and relative speedup tables.

Usage:
    python scripts/cross_backend_benchmark_report.py
    python scripts/cross_backend_benchmark_report.py --quick
    python scripts/cross_backend_benchmark_report.py --output report.md
    python scripts/cross_backend_benchmark_report.py --json results.json
"""

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Stores the result of a single model benchmark on a single backend."""

    backend: str
    model_name: str
    batch_size: int
    latency_ms: float        # mean latency in milliseconds
    latency_std_ms: float    # standard deviation of latency in milliseconds
    throughput: float         # samples per second
    memory_mb: float          # peak memory usage in megabytes

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "latency_ms": round(self.latency_ms, 4),
            "latency_std_ms": round(self.latency_std_ms, 4),
            "throughput": round(self.throughput, 2),
            "memory_mb": round(self.memory_mb, 2),
        }


@dataclass
class BenchmarkSuite:
    """Aggregates all benchmark results together with run metadata."""

    results: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""
    torch_version: str = ""
    python_version: str = ""
    backends_detected: list[str] = field(default_factory=list)
    n_iterations: int = 0

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "torch_version": self.torch_version,
                "python_version": self.python_version,
                "backends_detected": self.backends_detected,
                "n_iterations": self.n_iterations,
            },
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def detect_backends() -> list[tuple[str, torch.device]]:
    """Return a list of (name, device) tuples for every available backend.

    CPU is always included.  CUDA, ROCm, and XPU are added when the
    corresponding hardware / software stack is detected.
    """
    backends: list[tuple[str, torch.device]] = [("cpu", torch.device("cpu"))]

    # CUDA (NVIDIA)
    if torch.cuda.is_available():
        hip_version = getattr(torch.version, "hip", None)
        if hip_version:
            # ROCm exposes itself through the CUDA API in PyTorch but reports
            # a non-None torch.version.hip value.
            backends.append(("rocm", torch.device("cuda", 0)))
        else:
            backends.append(("cuda", torch.device("cuda", 0)))

    # Intel XPU
    if hasattr(torch, "xpu") and hasattr(torch.xpu, "is_available") and torch.xpu.is_available():
        backends.append(("xpu", torch.device("xpu", 0)))

    return backends


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_linear_small() -> tuple[nn.Module, torch.Tensor, str, int]:
    """Small linear model: Linear(768,768) -> GELU -> Linear(768,768)."""
    model = nn.Sequential(
        nn.Linear(768, 768),
        nn.GELU(),
        nn.Linear(768, 768),
    )
    batch_size = 32
    sample_input = torch.randn(batch_size, 768)
    return model, sample_input, "linear_small", batch_size


def _build_linear_medium() -> tuple[nn.Module, torch.Tensor, str, int]:
    """Medium linear model: 6 layers of Linear(1024,1024) + GELU."""
    layers: list = []
    for _ in range(6):
        layers.append(nn.Linear(1024, 1024))
        layers.append(nn.GELU())
    model = nn.Sequential(*layers)
    batch_size = 16
    sample_input = torch.randn(batch_size, 1024)
    return model, sample_input, "linear_medium", batch_size


def _build_conv_small() -> tuple[nn.Module, torch.Tensor, str, int]:
    """Small conv model: Conv2d -> ReLU -> AdaptiveAvgPool2d -> Flatten -> Linear."""
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )
    batch_size = 8
    sample_input = torch.randn(batch_size, 3, 32, 32)
    return model, sample_input, "conv_small", batch_size


MODEL_BUILDERS = [
    _build_linear_small,
    _build_linear_medium,
    _build_conv_small,
]


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def _estimate_model_memory_mb(model: nn.Module) -> float:
    """Estimate model memory footprint from parameter sizes (CPU fallback)."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


def _is_cuda_device(device: torch.device) -> bool:
    """Return True if *device* maps to a CUDA (or ROCm-via-HIP) device."""
    return device.type == "cuda"


def warmup(model: nn.Module, sample_input: torch.Tensor, device: torch.device, n: int = 5) -> None:
    """Run *n* warmup forward passes so that kernels are compiled / cached."""
    model_dev = model.to(device)
    input_dev = sample_input.to(device)
    with torch.no_grad():
        for _ in range(n):
            model_dev(input_dev)
    if _is_cuda_device(device):
        torch.cuda.synchronize(device)


def benchmark_model(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    backend_name: str,
    model_name: str,
    batch_size: int,
    n_iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark a single model on a single device and return a result.

    Steps:
        1. Move model and input to *device*.
        2. Run warmup passes.
        3. Time *n_iterations* forward passes under ``torch.no_grad()``.
        4. Compute mean / std latency, throughput, and peak memory.
    """
    model_dev = model.to(device)
    model_dev.eval()
    input_dev = sample_input.to(device)

    # Reset peak memory stats when possible.
    if _is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats(device)

    warmup(model_dev, input_dev, device, n=5)

    # Collect per-iteration timings.
    timings: list = []
    with torch.no_grad():
        for _ in range(n_iterations):
            if _is_cuda_device(device):
                torch.cuda.synchronize(device)

            t_start = time.perf_counter()
            model_dev(input_dev)

            if _is_cuda_device(device):
                torch.cuda.synchronize(device)

            t_end = time.perf_counter()
            timings.append(t_end - t_start)

    # Statistics.
    latencies_ms = [t * 1000.0 for t in timings]
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    variance = sum((x - mean_ms) ** 2 for x in latencies_ms) / len(latencies_ms)
    std_ms = variance ** 0.5
    throughput = batch_size / (mean_ms / 1000.0) if mean_ms > 0 else 0.0

    # Memory.
    if _is_cuda_device(device):
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        memory_mb = _estimate_model_memory_mb(model_dev)

    # Cleanup.
    del model_dev, input_dev
    gc.collect()
    if _is_cuda_device(device):
        torch.cuda.empty_cache()

    return BenchmarkResult(
        backend=backend_name,
        model_name=model_name,
        batch_size=batch_size,
        latency_ms=mean_ms,
        latency_std_ms=std_ms,
        throughput=throughput,
        memory_mb=memory_mb,
    )


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_benchmarks(
    backends: list[tuple[str, torch.device]],
    quick: bool = False,
) -> BenchmarkSuite:
    """Run all model benchmarks across all detected backends.

    Args:
        backends: Output of :func:`detect_backends`.
        quick: If ``True``, use 20 iterations instead of 100.

    Returns:
        A populated :class:`BenchmarkSuite`.
    """
    n_iterations = 20 if quick else 100

    suite = BenchmarkSuite(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        torch_version=torch.__version__,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        backends_detected=[name for name, _ in backends],
        n_iterations=n_iterations,
    )

    total_tasks = len(MODEL_BUILDERS) * len(backends)
    task_idx = 0

    for builder in MODEL_BUILDERS:
        model, sample_input, model_name, batch_size = builder()
        for backend_name, device in backends:
            task_idx += 1
            print(
                f"  [{task_idx}/{total_tasks}] {model_name} on {backend_name} "
                f"({n_iterations} iters) ...",
                end=" ",
                flush=True,
            )
            result = benchmark_model(
                model=model,
                sample_input=sample_input,
                device=device,
                backend_name=backend_name,
                model_name=model_name,
                batch_size=batch_size,
                n_iterations=n_iterations,
            )
            print(f"done  [{result.latency_ms:.2f} ms]")
            suite.results.append(result)

    return suite


# ---------------------------------------------------------------------------
# Report formatters
# ---------------------------------------------------------------------------

def format_markdown(suite: BenchmarkSuite) -> str:
    """Generate a Markdown report from benchmark results."""
    lines: list = []

    lines.append("# TorchBridge Cross-Backend Benchmark Report")
    lines.append("")
    lines.append(f"- **Timestamp:** {suite.timestamp}")
    lines.append(f"- **PyTorch version:** {suite.torch_version}")
    lines.append(f"- **Python version:** {suite.python_version}")
    lines.append(f"- **Backends detected:** {', '.join(suite.backends_detected)}")
    lines.append(f"- **Iterations per benchmark:** {suite.n_iterations}")
    lines.append("")

    # Collect unique model names (preserving insertion order).
    model_names: list = []
    for r in suite.results:
        if r.model_name not in model_names:
            model_names.append(r.model_name)

    # --- Per-model performance tables ---
    lines.append("## Per-Model Performance")
    lines.append("")

    for model_name in model_names:
        model_results = [r for r in suite.results if r.model_name == model_name]
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append(
            "| Backend | Latency (ms) | Std (ms) | Throughput (samples/s) | Memory (MB) |"
        )
        lines.append(
            "|---------|-------------:|---------:|-----------------------:|------------:|"
        )
        for r in model_results:
            lines.append(
                f"| {r.backend} | {r.latency_ms:.4f} | {r.latency_std_ms:.4f} "
                f"| {r.throughput:.2f} | {r.memory_mb:.2f} |"
            )
        lines.append("")

    # --- Speedup table relative to CPU ---
    lines.append("## Relative Speedup (vs CPU)")
    lines.append("")
    lines.append("| Model | Backend | Speedup |")
    lines.append("|-------|---------|--------:|")

    for model_name in model_names:
        model_results = [r for r in suite.results if r.model_name == model_name]
        cpu_result = next((r for r in model_results if r.backend == "cpu"), None)
        if cpu_result is None or cpu_result.latency_ms == 0:
            continue
        for r in model_results:
            speedup = cpu_result.latency_ms / r.latency_ms if r.latency_ms > 0 else 0.0
            lines.append(f"| {model_name} | {r.backend} | {speedup:.2f}x |")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by cross_backend_benchmark_report.py*")
    lines.append("")

    return "\n".join(lines)


def format_json(suite: BenchmarkSuite) -> str:
    """Serialize benchmark results to a JSON string."""
    return json.dumps(suite.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TorchBridge Cross-Backend Benchmark Report Generator",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: run 20 iterations instead of 100.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the Markdown report file (default: print to stdout).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        dest="json_path",
        help="Path for the JSON results file.",
    )
    args = parser.parse_args()

    print("TorchBridge Cross-Backend Benchmark Report Generator")
    print("=" * 52)

    backends = detect_backends()
    print(f"Detected backends: {', '.join(name for name, _ in backends)}")
    print()

    suite = run_benchmarks(backends, quick=args.quick)
    print()

    # Markdown output.
    md_report = format_markdown(suite)
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(md_report, encoding="utf-8")
        print(f"Markdown report written to {out_path}")
    else:
        print(md_report)

    # JSON output (optional).
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.write_text(format_json(suite), encoding="utf-8")
        print(f"JSON results written to {json_path}")


if __name__ == "__main__":
    main()
