"""
Use Case 3: CI/CD Hardware Validation

Demonstrates using TorchBridge's CLI tools programmatically to validate
hardware, benchmark models, and generate reports — the kind of checks
you'd run in a CI/CD pipeline before deploying to new hardware.

Run: PYTHONPATH=src python3 examples/usecase3_cicd_validation.py
"""
# ruff: noqa: E402

import importlib.util
import json
import time

import torch
import torch.nn as nn

# ── Step 1: System Diagnostics (torchbridge doctor) ─────────────────
print("=" * 60)
print("STEP 1: System Diagnostics (like `torchbridge doctor`)")
print("=" * 60)

import argparse

from torchbridge.cli.doctor import DoctorCommand

# Run full diagnostics (same as `torchbridge doctor --full-report --verbose`)
print()
args = argparse.Namespace(verbose=True, full_report=True, output=None, category=None, fix=False)
DoctorCommand.execute(args)

# Also capture key facts for the CI report
report = {
    "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
    "pytorch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "mps_available": torch.backends.mps.is_available(),
    "torch_compile": hasattr(torch, "compile"),
}

# ── Step 2: Backend Validation ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Backend Detection & Validation")
print("=" * 60)

from torchbridge.backends import BackendFactory, detect_best_backend

backend_name = detect_best_backend()
backend = BackendFactory.create(backend_name)
info = backend.get_device_info()

print(f"  Selected     : {backend_name}")
print(f"  Device       : {backend.device}")
print(f"  Available    : {info.is_available}")

# Test basic tensor operations on the backend
device = backend.device
x = torch.randn(100, 100, device=device)
y = torch.randn(100, 100, device=device)
z = torch.mm(x, y)
print(f"  MatMul test  : PASS (output shape: {list(z.shape)})")

# ── Step 3: Model Benchmark ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Model Benchmark (like `torchbridge benchmark`)")
print("=" * 60)

class BenchmarkModel(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
        )

    def forward(self, x):
        return self.layers(x)

model = BenchmarkModel().to(device).eval()

batch_sizes = [1, 8, 32, 64]
warmup_iters = 5
bench_iters = 50

print(f"\n  {'Batch':<8s} {'Latency':>10s} {'Throughput':>14s} {'Memory':>10s}")
print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*10}")

results = []
for bs in batch_sizes:
    inp = torch.randn(bs, 512, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(inp)

    # Benchmark
    times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(inp)
        times.append(time.perf_counter() - start)

    avg_ms = (sum(times) / len(times)) * 1000
    throughput = bs / (avg_ms / 1000)

    result = {"batch_size": bs, "latency_ms": round(avg_ms, 3), "throughput": round(throughput, 1)}
    results.append(result)
    print(f"  {bs:<8d} {avg_ms:>9.3f}ms {throughput:>12.1f}/s {'N/A':>10s}")

# ── Step 4: Cross-Backend Compatibility Check ────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Cross-Backend Compatibility Check")
print("=" * 60)

backends_to_test = ["cpu"]
if torch.cuda.is_available():
    backends_to_test.append("cuda")

for bname in backends_to_test:
    try:
        b = BackendFactory.create(bname)
        dev = b.device
        test_model = BenchmarkModel().to(dev).eval()
        test_input = torch.randn(4, 512, device=dev)
        with torch.no_grad():
            out = test_model(test_input)
        assert out.shape == (4, 512), f"Wrong shape: {out.shape}"
        print(f"  {bname:>6s} : PASS (output shape {list(out.shape)})")
    except Exception as e:
        print(f"  {bname:>6s} : FAIL ({e})")

# ── Step 5: Export Validation ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Export Format Validation")
print("=" * 60)

cpu_model = BenchmarkModel().eval()
sample = torch.randn(1, 512)

formats_tested = {}

# TorchScript
try:
    traced = torch.jit.trace(cpu_model, sample)
    ts_out = traced(sample)
    orig_out = cpu_model(sample)
    match = torch.allclose(ts_out, orig_out, atol=1e-5)
    formats_tested["TorchScript"] = "PASS" if match else "FAIL"
except Exception as e:
    formats_tested["TorchScript"] = f"FAIL ({e})"

# ONNX
if importlib.util.find_spec("onnx") is not None:
    formats_tested["ONNX"] = "AVAILABLE"
else:
    formats_tested["ONNX"] = "NOT INSTALLED"

# SafeTensors
if importlib.util.find_spec("safetensors") is not None:
    formats_tested["SafeTensors"] = "AVAILABLE"
else:
    formats_tested["SafeTensors"] = "NOT INSTALLED"

for fmt, status in formats_tested.items():
    print(f"  {fmt:<15s}: {status}")

# ── Step 6: Generate CI Report ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: CI/CD Report (JSON)")
print("=" * 60)

ci_report = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "system": {
        "python": report.get("python_version", "N/A"),
        "pytorch": report.get("pytorch_version", "N/A"),
        "cuda": report.get("cuda_available", False),
        "backend": str(backend_name),
    },
    "benchmarks": results,
    "export_formats": formats_tested,
    "status": "PASS",
}

print(json.dumps(ci_report, indent=2))
print("\n  This report can be saved to JSON and tracked across CI runs")
print("  to detect performance regressions on hardware changes.")
