# TorchBridge v0.4.40 — Cloud GPU Validation Results

**Date**: 2026-01-31
**Status**: ALL PASS (5/5 on both platforms)

## Platform Summary

| Platform | GPU | PyTorch | CUDA | Memory | Compute Cap |
|----------|-----|---------|------|--------|-------------|
| AWS g5.xlarge | NVIDIA A10G | 2.9.1+cu130 | 13.0 | 23.7 GB | 8.6 (Ampere) |
| GCP g2-standard-4 | NVIDIA L4 | 2.7.1+cu128 | 12.8 | 23.7 GB | 8.9 (Ada Lovelace) |

## Use Case Results

| Use Case | AWS A10G | GCP L4 |
|----------|----------|--------|
| UC1: Export Pipeline | PASS | PASS |
| UC2: LLM Optimization | PASS | PASS |
| UC3: CI/CD Validation | PASS | PASS |
| UC4: Backend Training | PASS | PASS |
| UC5: Cross-Backend Validation | PASS | PASS |
| **Total** | **5/5** | **5/5** |

## Detailed Results

### UC1: Hardware-Agnostic Export Pipeline

| Metric | AWS A10G | GCP L4 |
|--------|----------|--------|
| Backend detected | NVIDIA (NVIDIABackend) | NVIDIA (NVIDIABackend) |
| TorchScript export | 12.4 MB, 0.66s | 12.4 MB |
| ONNX export | 14.5 MB, 5.56s | 14.5 MB |
| SafeTensors export | 14.5 MB | 14.5 MB |
| Baseline inference | 0.86 ms | -- |

### UC2: LLM Optimization (GPT-2 124M)

| Metric | AWS A10G | GCP L4 |
|--------|----------|--------|
| Baseline latency | 2045.8 ms | -- |
| TorchBridge optimized | 2016.6 ms (1.01x) | -- |
| BetterTransformer | applied | applied |
| Outputs match | True | True |

### UC3: CI/CD Hardware Validation

**AWS A10G benchmarks (batch size sweep):**
| Batch | Latency | Throughput |
|-------|---------|------------|
| 1 | 0.118 ms | 8,470/s |
| 8 | 0.110 ms | 72,514/s |
| 32 | 0.109 ms | 293,357/s |
| 64 | 0.110 ms | 581,235/s |

- Diagnostics: 12/14 passed (2 optional warnings)
- Cross-backend: CPU PASS, CUDA PASS
- Export formats: TorchScript PASS, ONNX available, SafeTensors available

### UC4: Backend-Agnostic Training

| Metric | AWS A10G | GCP L4 |
|--------|----------|--------|
| AMP (mixed precision) | enabled | enabled |
| Baseline loss | 0.0201 | 0.0195 |
| TorchBridge loss | 0.0204 | 0.0175 |
| Baseline time | 689.0 ms | 897.3 ms |
| TorchBridge time | 812.8 ms | 1080.1 ms |
| Train accuracy | 100% | 100% |
| TorchScript export | PASS | PASS |
| SafeTensors export | PASS | PASS |

### UC5: Cross-Backend Validation

| Check | AWS A10G | GCP L4 |
|-------|----------|--------|
| Model validation | 5/5 | 5/5 |
| Hardware compatibility | 3/3 | 3/3 |
| Config presets | 3/3 | 3/3 |
| CPU consistency | PASS (diff=0.00) | PASS (diff=0.00) |
| CUDA consistency | PASS (max_diff=3.55e-04) | PASS (max_diff=1.08e-03) |

## Cost Summary

| Instance | Type | Duration | Est. Cost |
|----------|------|----------|-----------|
| AWS g5.xlarge (on-demand) | A10G | ~5 min | ~$0.08 |
| GCP g2-standard-4 (on-demand) | L4 | ~5 min | ~$0.08 |
| **Total** | | | **~$0.16** |

## Validation Script

See [`v0440_gpu_validation.sh`](./v0440_gpu_validation.sh) for the full validation script.
Handles both AWS Deep Learning AMIs and GCP DL VMs automatically.
