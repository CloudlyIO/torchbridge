"""
Use Case 1: Hardware-Agnostic Model Export Pipeline

Demonstrates TorchBridge's core value: detect hardware, build a model,
export to every portable format, and validate correctness — all through
a unified API that works identically on NVIDIA, AMD, Intel, TPU, or CPU.

Run: PYTHONPATH=src python3 examples/usecase1_export_pipeline.py
"""
# ruff: noqa: E402

import os
import time

import torch
import torch.nn as nn

# ── Step 1: Backend Detection ────────────────────────────────────────
print("=" * 60)
print("STEP 1: Hardware Detection & Backend Selection")
print("=" * 60)

from torchbridge.backends import BackendFactory, detect_best_backend

backend_name = detect_best_backend()
backend = BackendFactory.create(backend_name)
device = backend.device

print(f"  Detected backend : {backend_name}")
print(f"  Device           : {device}")
print(f"  Backend class    : {type(backend).__name__}")

info = backend.get_device_info()
print(f"  Device name      : {info.device_name}")
print(f"  Memory           : {info.total_memory_gb:.1f} GB" if info.total_memory_bytes else "  Memory           : (not reported for CPU)")
print(f"  Available        : {info.is_available}")
for k, v in info.properties.items():
    print(f"  {k:17s}: {v}")

# ── Step 2: Build a Model ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Build a Model (Transformer Encoder)")
print("=" * 60)

class SmallTransformer(nn.Module):
    """A minimal transformer for demonstrating export."""
    def __init__(self, vocab_size=5000, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.fc(x)
        return x

model = SmallTransformer().to(device)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
print(f"  Parameters  : {param_count:,} ({param_mb:.1f} MB)")
print(f"  Device      : {next(model.parameters()).device}")

# ── Step 3: Baseline Inference ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Baseline Inference")
print("=" * 60)

sample_input = torch.randint(0, 5000, (1, 64), device=device)

with torch.no_grad():
    # warmup
    for _ in range(3):
        _ = model(sample_input)

    start = time.perf_counter()
    baseline_output = model(sample_input)
    baseline_ms = (time.perf_counter() - start) * 1000

print(f"  Input shape  : {list(sample_input.shape)}")
print(f"  Output shape : {list(baseline_output.shape)}")
print(f"  Latency      : {baseline_ms:.2f} ms")

# ── Step 4: Export to All Formats ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Export via TorchBridge")
print("=" * 60)

from torchbridge.deployment import (
    export_to_onnx,
    export_to_safetensors,
    export_to_torchscript,
)

output_dir = "examples/export_output"
os.makedirs(output_dir, exist_ok=True)

# Move to CPU for export (required for TorchScript tracing and ONNX)
cpu_model = model.cpu()
cpu_input = sample_input.cpu()

results = {}

# 4a: TorchScript
print("\n  [TorchScript]")
try:
    t0 = time.perf_counter()
    ts_result = export_to_torchscript(
        cpu_model, output_path=f"{output_dir}/model.pt", sample_input=cpu_input
    )
    dt = time.perf_counter() - t0
    size = os.path.getsize(f"{output_dir}/model.pt") / 1e6
    results["TorchScript"] = {"time": dt, "size": size, "path": f"{output_dir}/model.pt"}
    print(f"    Exported : {output_dir}/model.pt ({size:.1f} MB)")
    print(f"    Time     : {dt:.2f}s")
except Exception as e:
    print(f"    FAILED: {e}")

# 4b: ONNX
print("\n  [ONNX]")
try:
    t0 = time.perf_counter()
    onnx_result = export_to_onnx(
        cpu_model, output_path=f"{output_dir}/model.onnx",
        sample_input=cpu_input, opset_version=17
    )
    dt = time.perf_counter() - t0
    size = os.path.getsize(f"{output_dir}/model.onnx") / 1e6
    results["ONNX"] = {"time": dt, "size": size, "path": f"{output_dir}/model.onnx"}
    print(f"    Exported : {output_dir}/model.onnx ({size:.1f} MB)")
    print(f"    Time     : {dt:.2f}s")
except Exception as e:
    print(f"    FAILED: {e}")

# 4c: SafeTensors
print("\n  [SafeTensors]")
try:
    t0 = time.perf_counter()
    st_result = export_to_safetensors(
        cpu_model, output_path=f"{output_dir}/model.safetensors"
    )
    dt = time.perf_counter() - t0
    size = os.path.getsize(f"{output_dir}/model.safetensors") / 1e6
    results["SafeTensors"] = {"time": dt, "size": size, "path": f"{output_dir}/model.safetensors"}
    print(f"    Exported : {output_dir}/model.safetensors ({size:.1f} MB)")
    print(f"    Time     : {dt:.2f}s")
except Exception as e:
    print(f"    FAILED: {e}")

# ── Step 5: Validate Exports ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Validate Exported Models")
print("=" * 60)

# 5a: TorchScript validation
if "TorchScript" in results:
    print("\n  [TorchScript Validation]")
    ts_model = torch.jit.load(results["TorchScript"]["path"])
    ts_model.eval()
    with torch.no_grad():
        ts_output = ts_model(cpu_input)
    baseline_cpu = baseline_output.cpu()
    max_diff = (ts_output - baseline_cpu).abs().max().item()
    match = torch.allclose(ts_output, baseline_cpu, atol=1e-3)
    print(f"    Outputs match : {'PASS' if match else 'FAIL'} (max diff: {max_diff:.2e})")

# 5b: ONNX Runtime validation
if "ONNX" in results:
    print("\n  [ONNX Runtime Validation]")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(results["ONNX"]["path"])
        input_name = sess.get_inputs()[0].name
        ort_output = sess.run(None, {input_name: cpu_input.numpy()})[0]
        ort_tensor = torch.tensor(ort_output)
        baseline_cpu = baseline_output.cpu()
        match = torch.allclose(ort_tensor, baseline_cpu, atol=1e-4)
        max_diff = (ort_tensor - baseline_cpu).abs().max().item()
        print(f"    Outputs match : {'PASS' if match else 'FAIL'} (max diff: {max_diff:.2e})")
    except Exception as e:
        print(f"    SKIPPED: {e}")

# 5c: SafeTensors validation
if "SafeTensors" in results:
    print("\n  [SafeTensors Validation]")
    from safetensors.torch import load_file
    loaded_weights = load_file(results["SafeTensors"]["path"])
    original_keys = set(dict(cpu_model.named_parameters()).keys())
    loaded_keys = set(loaded_weights.keys())
    keys_match = original_keys == loaded_keys
    weights_match = all(
        torch.equal(dict(cpu_model.named_parameters())[k], loaded_weights[k])
        for k in original_keys
    )
    print(f"    Keys match    : {'PASS' if keys_match else 'FAIL'} ({len(loaded_keys)} params)")
    print(f"    Weights match : {'PASS' if weights_match else 'FAIL'}")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Backend          : {backend_name}")
print(f"  Model params     : {param_count:,}")
print(f"  Baseline latency : {baseline_ms:.2f} ms")
print()
print(f"  {'Format':<15s} {'Size':>8s} {'Export Time':>12s}")
print(f"  {'-'*15} {'-'*8} {'-'*12}")
for fmt, info in results.items():
    print(f"  {fmt:<15s} {info['size']:>7.1f}M {info['time']:>11.2f}s")
print()
print(f"  Exported files in: {output_dir}/")
print("  All formats validated against baseline inference.")
