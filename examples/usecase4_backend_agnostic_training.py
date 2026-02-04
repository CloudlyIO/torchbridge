"""
Use Case 4: Backend-Agnostic Training Loop

Demonstrates the core HAL value proposition: write a training loop ONCE,
run it on any backend. The same code works on NVIDIA, AMD, Intel, TPU,
or CPU — TorchBridge handles device placement and backend-specific
optimizations automatically.

Run: PYTHONPATH=src python3 examples/usecase4_backend_agnostic_training.py
"""
# ruff: noqa: E402

import time

import torch
import torch.nn as nn

# ── Step 1: Backend-Agnostic Setup ───────────────────────────────────
print("=" * 60)
print("STEP 1: Backend-Agnostic Setup")
print("=" * 60)

from torchbridge import TorchBridgeConfig
from torchbridge.backends import BackendFactory, detect_best_backend

# Auto-detect hardware — same code on any machine
backend_name = detect_best_backend()
backend = BackendFactory.create(backend_name)
device = backend.device

print(f"  Backend  : {backend_name}")
print(f"  Device   : {device}")
print(f"  Class    : {type(backend).__name__}")

# Backend-aware config
config = TorchBridgeConfig.for_training()
print("  Config   : training preset loaded")

# ── Step 2: Model & Data (backend-agnostic) ──────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Model & Data Setup")
print("=" * 60)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model = SimpleClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Synthetic MNIST-like data
n_samples = 2048
train_x = torch.randn(n_samples, 784, device=device)
train_y = torch.randint(0, 10, (n_samples,), device=device)

param_count = sum(p.numel() for p in model.parameters())
print(f"  Model       : SimpleClassifier ({param_count:,} params)")
print(f"  Data        : {n_samples} samples on {device}")
print("  Optimizer   : AdamW (lr=1e-3)")

# ── Step 3: Train WITHOUT TorchBridge (baseline) ────────────────────
print("\n" + "=" * 60)
print("STEP 3: Baseline Training (plain PyTorch)")
print("=" * 60)

baseline_model = SimpleClassifier().to(device)
baseline_opt = torch.optim.AdamW(baseline_model.parameters(), lr=1e-3)

n_epochs = 10
batch_size = 64

baseline_model.train()
start = time.perf_counter()
baseline_losses = []

for _epoch in range(n_epochs):
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, n_samples, batch_size):
        batch_x = train_x[i:i+batch_size]
        batch_y = train_y[i:i+batch_size]

        baseline_opt.zero_grad()
        output = baseline_model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        baseline_opt.step()

        epoch_loss += loss.item()
        n_batches += 1

    baseline_losses.append(epoch_loss / n_batches)

baseline_time = time.perf_counter() - start

print(f"  Epochs      : {n_epochs}")
print(f"  Final loss  : {baseline_losses[-1]:.4f}")
print(f"  Total time  : {baseline_time*1000:.1f} ms")
print(f"  Per epoch   : {baseline_time/n_epochs*1000:.1f} ms")

# ── Step 4: Train WITH TorchBridge backend optimization ──────────────
print("\n" + "=" * 60)
print("STEP 4: TorchBridge-Optimized Training")
print("=" * 60)

tb_model = SimpleClassifier().to(device)
tb_opt = torch.optim.AdamW(tb_model.parameters(), lr=1e-3)

# Apply backend-specific model preparation
tb_model = backend.prepare_model(tb_model)
print(f"  Backend prep: {type(backend).__name__}.prepare_model() applied")

# Use torch.amp autocast if available (backend-aware)
use_amp = device != "cpu" and torch.cuda.is_available()
scaler = torch.amp.GradScaler(device) if use_amp else None
amp_dtype = torch.float16 if use_amp else None

tb_model.train()
start = time.perf_counter()
tb_losses = []

for _epoch in range(n_epochs):
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, n_samples, batch_size):
        batch_x = train_x[i:i+batch_size]
        batch_y = train_y[i:i+batch_size]

        tb_opt.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                output = tb_model(batch_x)
                loss = criterion(output, batch_y)
            scaler.scale(loss).backward()
            scaler.step(tb_opt)
            scaler.update()
        else:
            output = tb_model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            tb_opt.step()

        epoch_loss += loss.item()
        n_batches += 1

    tb_losses.append(epoch_loss / n_batches)

tb_time = time.perf_counter() - start

print(f"  AMP enabled : {use_amp}")
print(f"  Epochs      : {n_epochs}")
print(f"  Final loss  : {tb_losses[-1]:.4f}")
print(f"  Total time  : {tb_time*1000:.1f} ms")
print(f"  Per epoch   : {tb_time/n_epochs*1000:.1f} ms")

# ── Step 5: Evaluate Both ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Evaluation")
print("=" * 60)

# Quick eval on same data (just to check convergence)
for name, m in [("Baseline", baseline_model), ("TorchBridge", tb_model)]:
    m.eval()
    with torch.no_grad():
        logits = m(train_x[:256])
        preds = logits.argmax(dim=1)
        acc = (preds == train_y[:256]).float().mean().item()
    print(f"  {name:12s}: train accuracy = {acc:.1%}")

# ── Step 6: Export Optimized Model ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Export Trained Model")
print("=" * 60)

import os

from torchbridge.deployment import export_to_safetensors, export_to_torchscript

output_dir = "examples/training_output"
os.makedirs(output_dir, exist_ok=True)

cpu_model = tb_model.cpu()
sample = torch.randn(1, 784)

ts_result = export_to_torchscript(cpu_model, f"{output_dir}/classifier.pt", sample)

ts_path = f"{output_dir}/classifier.pt"
if os.path.exists(ts_path):
    ts_size = os.path.getsize(ts_path) / 1e3
    print(f"  TorchScript  : {ts_path} ({ts_size:.0f} KB)")
else:
    print("  TorchScript  : FAILED")

try:
    st_result = export_to_safetensors(cpu_model, f"{output_dir}/classifier.safetensors")
    st_path = f"{output_dir}/classifier.safetensors"
    if os.path.exists(st_path):
        st_size = os.path.getsize(st_path) / 1e3
        print(f"  SafeTensors  : {st_path} ({st_size:.0f} KB)")
    else:
        print("  SafeTensors  : FAILED")
except Exception as e:
    print(f"  SafeTensors  : SKIPPED ({e})")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
speedup = baseline_time / tb_time
print(f"\n  Backend: {backend_name}")
print(f"  {'':15s} {'Baseline':>12s} {'TorchBridge':>12s}")
print(f"  {'-'*15} {'-'*12} {'-'*12}")
print(f"  {'Final loss':<15s} {baseline_losses[-1]:>12.4f} {tb_losses[-1]:>12.4f}")
print(f"  {'Total time':<15s} {baseline_time*1000:>11.1f}ms {tb_time*1000:>11.1f}ms")
print(f"  {'Speedup':<15s} {'1.00x':>12s} {f'{speedup:.2f}x':>12s}")
print()
print("  The same training code runs on NVIDIA, AMD, Intel, TPU, or CPU.")
print("  On GPU, AMP (mixed precision) and backend.prepare_model() apply")
print("  vendor-specific optimizations automatically.")
