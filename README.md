# TorchBridge

TorchBridge **validates that your model produces correct outputs across PyTorch backends** and recommends optimal hardware configurations. It answers two questions no other tool answers in a single command:

1. **"Does my model produce correct outputs across backends?"** — Run it on CUDA and ROCm and get max_diff, cosine_sim, per-layer divergence, pass/fail against empirical tolerances.
2. **"What's the optimal configuration for my model on this hardware?"** — Compatibility matrices that translate `(backend, architecture) → format/kernel/method` with fallback chains.

[![Version](https://img.shields.io/pypi/v/torchbridge-ml?label=version&color=green)](./CHANGELOG.md) [![License](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE) [![Tests](https://img.shields.io/badge/tests-2%2C224%20passed-blue)](./docs/reference/hardware-matrix.md) [![Cloud GPU](https://img.shields.io/badge/platforms-5%2F8%20validated-brightgreen)](./docs/reference/cloud-validation.md) [![AWS A10G](https://img.shields.io/badge/AWS%20A10G-PASS-brightgreen)](./docs/reference/cloud-validation.md) [![GCP L4](https://img.shields.io/badge/GCP%20L4-PASS-brightgreen)](./docs/reference/cloud-validation.md) [![H100 NVL](https://img.shields.io/badge/H100%20NVL-PASS-brightgreen)](./docs/reference/cloud-validation.md) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)](https://pytorch.org)

## Quick Start

```bash
pip install torchbridge-ml
tb-doctor
```

```
 TorchBridge System Diagnostics
==================================================
 Python Version:        Python 3.11.0  ( Compatible)
 PyTorch Version:       PyTorch 2.11.0 ( Compatible)
 TorchBridge Version:   0.5.100        ( Available)
 CUDA GPU:              Not found      ( Apple Silicon MPS provides GPU acceleration)
 Apple Silicon GPU:     Available      ( GPU acceleration enabled)
 CPU Cores:             11 cores       ( Available)

 Summary: 7/7 checks passed
 System is ready for optimal TorchBridge performance!
```

### Cross-Backend Validation (the hero command)

```bash
# Compare CUDA vs ROCm outputs — max_diff, cosine_sim, pass/fail
tb-validate --compare cuda rocm --model ./model.pt

# Per-layer divergence report
tb-validate --compare cuda rocm --model ./model.pt --per-layer

# Multi-step agentic trace — track divergence amplification across 50 steps
tb-validate --compare cuda rocm --model ./model.pt --trace --steps 50 --autoregressive

# Compliance certificate + OTel span export (Langfuse, W&B, etc.)
tb-validate --compare cuda rocm --model ./model.pt --cert --otel

# CI mode — exits non-zero if max_diff exceeds tolerance
tb-validate --compare cuda rocm --model ./model.pt --ci
```

### Hardware Configuration Advisor

```bash
# What's the optimal config for a 7B model on this hardware?
tb-advisor --model-params 7e9

# Disaggregated prefill/decode fleet config
tb-advisor --mode disaggregated --model-params 7e9 --prefill nvidia:hopper --decode amd:cdna3

# Heterogeneous cluster training config (NVIDIA + AMD mixed)
tb-advisor --mode heterogeneous --model-params 7e9 --nvidia hopper:8 --amd cdna3:4

# Doctor — diagnose your hardware setup
tb-doctor
```

### Python API

```python
from torchbridge.backends import BackendFactory, detect_best_backend

backend_type = detect_best_backend()  # NVIDIA, AMD, Trainium, TPU, or CPU
backend = BackendFactory.create(backend_type)
print(backend.get_device_info())
```

```python
# Cross-backend validation
from torchbridge import UnifiedValidator

validator = UnifiedValidator()
results = validator.validate_model(model, input_shape=(1, 768))
print(f"Validation: {results.passed}/{results.total_tests} tests passed")
```

## What TorchBridge Does

| Capability | What TorchBridge adds |
|------------|----------------------|
| **Cross-backend validation** | `tb-validate --compare cuda rocm` — per-layer divergence, empirical tolerance DB (442 entries: 13 model families × 13 backends), CI-ready JSON |
| **Multi-step agentic trace** | `tb-validate --trace --steps 50 --autoregressive` — tracks how max_diff amplifies across N autoregressive steps; reports first-divergence-step and amplification factor |
| **Compliance certificates** | `tb-validate --cert` — SHA256-signed pass/fail certificate for KV handoff physical spec (page size, alignment, layout) |
| **Observability integration** | `tb-validate --otel` — emits validation spans (max_diff, cosine_sim, per-layer child spans) to any OTLP endpoint (Langfuse, W&B Weave, Honeycomb) |
| **Compatibility matrices** | 13 empirically-sourced matrices: `(backend, architecture) → optimal quant format / attention kernel / adapter method / FSDP strategy / torch.compile mode` |
| **Config advisory** | `tb-advisor` — FSDP, quantization, KV cache, speculative decoding, disaggregated fleet (`--mode disaggregated`), heterogeneous clusters (`--mode heterogeneous`) |
| **Backend detection** | Hardware identification, capability queries, priority chain across NVIDIA/AMD/Trainium/TPU/CPU |
| **Tolerance DB** | 442 entries, 3-level fallback, `--model-family` flag — 13 model families × 13 backends (fp16 omitted for XLA/Trainium); tolerances sourced from real Qwen3-0.6B runs across 5 validated platforms |
| **CLI diagnostics** | `tb-doctor`, `tb-validate`, `tb-advisor`, `tb-speculate`, `tb-cache`, `tb-adapter`, `tb-quantize`, `tb-migrate`, `tb-benchmark`, `tb-checkpoint` |

## What TorchBridge Is NOT

- **Not a quantization library** — dispatches format selection to torchao; TorchBridge adds the compatibility matrix
- **Not a serving runtime** — use vLLM, TGI, or similar for production inference serving; TorchBridge validates correctness and advises configuration, it does not serve requests
- **Not a training framework** — adapter math (LoRA/QLoRA) is correct and kept; use PEFT for full training workflows
- **Not a PyTorch wrapper** — if a method body is `return torch.something(...)` with no selection logic, it doesn't belong here

## Supported Backends

| Backend | Hardware | Precision | Status |
|---------|----------|-----------|--------|
| **NVIDIA** | B200, H100, H200, A100, L4, T4 | FP4, FP8, BF16, FP16, FP32 | Production |
| **AMD** | MI350X, MI325X, MI300X, MI200 | FP8, BF16, FP16, FP32 | Production |
| **Trainium** | Trn1, Trn2, Trn3 (AWS NeuronX) | BF16, FP16, FP32 | Supported |
| **TPU** | v4, v5e, v5p, v6e, v7 | BF16, FP32 | Production |
| **CPU** | x86, ARM (Apple Silicon) | FP32, BF16 | Fallback |

See [Hardware Matrix](./docs/reference/hardware-matrix.md) for full details.

## Cloud Hardware Validation

Cross-backend numerical consistency validated on 5 platforms using Qwen3-0.6B (v0.5.100, 2026-07-21):

| Platform | Hardware | Max Diff | Cosine Sim | Latency | Status |
|----------|----------|----------|------------|---------|--------|
| Local | Apple Silicon MPS | 3.72e-05 | 1.000002 | 30.3 ms | PASS |
| AWS | NVIDIA A10G sm_86 (24GB) | 2.62e-05 | 1.000001 | 35.8 ms | PASS |
| GCP | NVIDIA L4 sm_89 (24GB) | 2.77e-05 | 1.000001 | 48.6 ms | PASS |
| RunPod | NVIDIA H100 NVL sm_90 (100GB) | 2.29e-05 | 1.000001 | 17.5 ms | PASS |
| AWS Trainium | Trn1.2xlarge (NeuronX 2.9) | 2.77e-05 | 1.000001 | 31.5 ms | PASS |
| AMD DevCloud | AMD MI300X (192GB) | — | — | — | PENDING† |
| GCP | TPU v5e | — | — | — | PENDING† |
| AWS Inferentia2 | inf2.xlarge | — | — | — | PENDING† |

† **Pending:** AMD MI300X requires user portal access to prevent runaway billing; TPU v5e quota exhausted; Inferentia2 deferred.

All tested GPU backends produce semantically identical outputs (cosine similarity > 0.999).

See [full validation report](./docs/reference/cloud-validation.md) for detailed benchmarks and results.

## Project Structure

```
src/torchbridge/
├── backends/          # Vendor-specific backend implementations
│   ├── nvidia/        #   NVIDIA CUDA backend
│   ├── amd/           #   AMD ROCm backend
│   ├── trainium/      #   AWS Trainium/NeuronX backend
│   └── tpu/           #   Google TPU/XLA backend
├── core/              # Hardware detection, config, architecture enums
├── precision/         # Quantization compatibility matrix + torchao dispatch
├── attention/         # Attention kernel compatibility matrix + dispatcher
├── distributed/       # FSDP/pipeline config advisor
├── adapters/          # LoRA/QLoRA adapter injection (correct math)
├── inference/         # Speculative decoding compatibility matrix
├── checkpoint/        # DCP wrapper with cross-backend metadata
├── testing/           # DivergenceTracer, ToleranceDB, MultiStepTracer, @cross_backend
├── validation/        # UnifiedValidator — model structure, hardware, numerical stability
├── cli/               # Command-line tools (11 CLI commands)
├── models/            # LLM KV cache advisor
└── utils/             # Utilities
```

## Quality

- **2,224 tests passing** (hardware-gated skips on non-GPU environments)
- **0 ruff violations** -- clean linting
- **0 mypy errors** -- full type coverage
- **Cloud validated** on 5 platforms: MPS, A10G, L4, H100 NVL (GPU), Trainium (NeuronX)

```bash
python3 -m pytest tests/ -q
ruff check src/ tests/
```

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](./docs/getting_started/installation.md) | Setup and requirements |
| [Quick Start](./docs/getting_started/quickstart.md) | First steps with TorchBridge |
| [Troubleshooting](./docs/getting_started/troubleshooting.md) | Common issues and fixes |
| [Backends Overview](./docs/backends/overview.md) | How the backend system works |
| [Backend Selection](./docs/guides/backend-selection.md) | Choosing backends + driver setup |
| [Distributed Training](./docs/guides/distributed-training.md) | Multi-GPU and multi-node |
| [Testing Guide](./docs/guides/testing.md) | DivergenceTracer, @cross_backend, ToleranceDB |
| [Deployment](./docs/guides/deployment.md) | Serving and containerization |
| [CLI Reference](./docs/guides/cli.md) | Command-line tools |
| [Hardware Matrix](./docs/reference/hardware-matrix.md) | Full hardware support table |
| [Changelog](./CHANGELOG.md) | Version history |

## Community

The empirical tolerance database (`testing/tolerance_db.py`) is only as strong as the hardware it has been measured on. Contributions that add or correct tolerance entries for hardware you have access to — AMD MI350X, Trainium2, TPU v7 Ironwood, new PyTorch versions — directly expand the validation coverage for everyone. See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to add entries and the source-label conventions (`"measured"`, `"derived"`, `"fallback"`).

**Good first contributions:**
- [Submit a hardware tolerance measurement](https://github.com/CloudlyIO/torchbridge/issues/new?template=tolerance_measurement.yml) — run `tb-validate --compare` and paste the output
- [Fix a compatibility matrix entry](https://github.com/CloudlyIO/torchbridge/issues/new?template=matrix_correction.yml) — if a recommendation doesn't match your hardware

## Versioning

v0.5.100 is the first public release. The v0.5.x series represents an extended private development and validation phase: building the backend abstraction layer, validating numerical consistency on real GPU hardware across 5 platforms, and reaching a quality bar suitable for open source. The version number reflects the maturity of the implementation, not the release count.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE) for the full text.
