# Quick Start

Get running with TorchBridge in three commands.

## 1. Install

```bash
pip install torchbridge-ml
```

## 2. Check your hardware

```bash
tb-doctor
```

```
 TorchBridge System Diagnostics
==================================================
 Python Version:        Python 3.11.0  ( Compatible)
 PyTorch Version:       PyTorch 2.11.0 ( Compatible)
 TorchBridge Version:   0.5.100        ( Available)
 CUDA GPU:              NVIDIA A10G    ( Available — 24 GB)
 CPU Cores:             8 cores        ( Available)

 Summary: 6/6 checks passed
 System is ready for optimal TorchBridge performance!
```

## 3. Validate your model

This is the core TorchBridge command — it compares your model's numerical outputs across two backends and reports whether they're within tolerance:

```bash
# Compare CUDA vs CPU (works without a second GPU)
tb-validate --compare cuda cpu --model ./model.pt

# Compare CUDA vs ROCm
tb-validate --compare cuda rocm --model ./model.pt

# Per-layer divergence report — find exactly where outputs diverge
tb-validate --compare cuda rocm --model ./model.pt --per-layer

# Multi-step agentic trace — track divergence amplification across 50 autoregressive steps
tb-validate --compare cuda rocm --model ./model.pt --trace --steps 50 --autoregressive

# CI mode — exits non-zero if max_diff exceeds tolerance
tb-validate --compare cuda rocm --model ./model.pt --ci
```

Example output:

```
TorchBridge Validation Results
================================
Backends:   cuda vs rocm
Model:      ./model.pt
Dtype:      float16

max_diff:   2.10e-05
cosine_sim: 1.000001
Tolerance:  PASS (threshold: 1e-04)

All 1/1 validation checks passed.
```

## Hardware Configuration Advisor

```bash
# What's the optimal config for a 7B model on this hardware?
tb-advisor --model-params 7e9

# Disaggregated prefill/decode fleet config
tb-advisor --mode disaggregated --model-params 7e9 --prefill nvidia:hopper --decode amd:cdna3

# Heterogeneous cluster training config (NVIDIA + AMD mixed)
tb-advisor --mode heterogeneous --model-params 7e9 --nvidia hopper:8 --amd cdna3:4
```

## Python API

```python
from torchbridge.backends import BackendFactory, detect_best_backend

# Auto-detect hardware
backend = BackendFactory.create(detect_best_backend())
device = backend.device
print(f"Backend: {backend}, Device: {device}")
```

```python
# Cross-backend validation
from torchbridge import UnifiedValidator

validator = UnifiedValidator()
results = validator.validate_model(model, input_shape=(1, 768))
print(f"Passed: {results.passed}/{results.total_tests} tests")
print(f"max_diff: {results.max_diff:.2e}, cosine_sim: {results.cosine_sim:.6f}")
```

```python
# Hardware-aware configuration
from torchbridge import TorchBridgeConfig, UnifiedManager

config = TorchBridgeConfig.for_inference()   # or for_training()
manager = UnifiedManager(config)
optimized_model = manager.optimize(model)
```

## Runnable example

A self-contained example that works on any hardware (no GPU required):

```bash
python examples/validate_quickstart.py
```

See [`examples/validate_quickstart.py`](../../examples/validate_quickstart.py) for the full source.

## Common pitfalls

### GPU not detected (silent CPU fallback)

TorchBridge falls back to CPU without error if no GPU is found. Run `tb-doctor` to verify.
If the doctor reports CPU-only, check that your GPU drivers and the correct PyTorch build are installed.

### Device placement mismatch

Always use the detected backend's device for both model and data:

```python
backend = BackendFactory.create(detect_best_backend())
device = backend.device

model = model.to(device)
inputs = inputs.to(device)  # Must match model device
```

### Precision differences after optimization

Optimized models may produce slightly different outputs due to mixed precision or kernel fusion.
This is expected. Use `tb-validate --compare` to confirm outputs are within tolerance.

### Missing optional dependencies

Some features require extras:

```bash
pip install torchbridge-ml[all]
```

## Next steps

- [Installation](installation.md) — backend-specific setup (CUDA, ROCm, Trainium, TPU)
- [Troubleshooting](troubleshooting.md) — common issues and fixes
- [CONTRIBUTING.md](../../CONTRIBUTING.md) — how to submit a hardware tolerance measurement or matrix correction
