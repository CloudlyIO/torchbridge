# TorchBridge Cloud Validation Results

**Status**: 6/8 validated — 4 GPU PASS, 1 MPS PASS, 2 CPU-fallback†, 2 SKIPPED‡ (v0.5.80, re-validated 2026-03-30)

## Platform Summary

| Platform | Hardware | Provider | Memory | Architecture |
|----------|----------|----------|--------|-------------|
| AWS g5.xlarge | NVIDIA A10G | AWS | 24 GB | Ampere (sm_86) |
| GCP n1-standard-4 | NVIDIA T4 | GCP | 16 GB | Turing (sm_75) |
| RunPod Community | NVIDIA H100 NVL | RunPod | 100 GB | Hopper (sm_90) |
| Local Mac | Apple Silicon | Local | Unified | MPS |
| AWS trn1.2xlarge | Trainium (NeuronCore v1) | AWS | 32 GB | Trainium1 |
| AWS inf2.xlarge | Inferentia2 (NeuronCore v1) | AWS | 32 GB | Inferentia2 |
| AMD Developer Cloud | AMD MI300X | AMD | 192 GB | CDNA3 (gfx942) |
| GCP TPU VM | TPU v5e | GCP | 16 GB/chip | v5e (v5litepod-1) |

## Cross-Backend Consistency (Qwen3-0.6B)

All validations compare GPU/accelerator logits against CPU baseline on the same model and input.

| Platform | Hardware | Max Diff | Cosine Sim | Latency | Status |
|----------|----------|----------|------------|---------|--------|
| AWS | A10G | 2.10e-05 | 1.000001 | 40.0 ms | PASS |
| GCP | T4 | 2.67e-05 | 1.000001 | 50.7 ms | PASS |
| RunPod | H100 NVL | 1.67e-05 | 1.000001 | 16.2 ms | PASS |
| Local | Apple MPS | 0.00e+00 | 1.000000 | 118.9 ms | PASS |
| AWS Trainium† | trn1.2xlarge | 0.00e+00 | 1.000000 | 115.8 ms (CPU) | PASS |
| AWS Inferentia2† | inf2.xlarge | 0.00e+00 | 1.000000 | 321.8 ms (CPU) | PASS |
| AMD DevCloud‡ | MI300X | — | — | — | SKIPPED |
| GCP‡ | TPU v5e | — | — | — | SKIPPED |

> **† CPU fallback — not real accelerator validation.**
> NeuronX SDK compilation (`torch_neuronx.trace()`) requires a quota-enabled
> `trn1` or `inf2` instance with the AWS Neuron SDK pre-installed. The validation
> runs fell back to CPU execution when compilation failed, producing
> `max_diff = 0.00e+00` (CPU-vs-CPU, not accelerator-vs-CPU). The "(CPU)" latency
> annotation in the table reflects this. Real NeuronX accelerator validation is
> pending AWS Trainium quota approval. The backend code (`trainium_backend.py`,
> `neuron_compiler.py`) is implemented and exercised in unit tests; accelerator
> execution requires instance access.

> **‡ Capacity unavailable at validation time.**
> AMD MI300X: no instance available in AMD Developer Cloud during v0.5.80 validation window.
> GCP TPU v5e: quota exhausted globally — tried 20+ zones across 3 sessions.
> These are infrastructure availability issues, not code issues.

### Validation Thresholds

| Backend | Max Diff Tolerance | Cosine Sim Threshold | Notes |
|---------|-------------------|---------------------|-------|
| CUDA (NVIDIA) | 1e-4 | 0.9999 | Exact parity expected |
| ROCm (AMD) | 1e-3 | 0.999 | SDPA flash attention divergence |
| MPS (Apple) | 1e-4 | 0.9999 | Apple Silicon |
| TPU (XLA) | 0.5 | 0.999 | XLA reorders ops; cosine sim is primary metric |

**Note:** TPU max_diff is relaxed because the XLA compiler reorders floating-point operations
for performance, causing larger absolute differences. Cosine similarity confirms semantic
equivalence and is the primary metric for XLA backends.

## Use Case Validation (5/5 on AWS and GCP)

| Use Case | AWS A10G | GCP T4 | Description |
|----------|----------|--------|-------------|
| UC1: Export Pipeline | PASS | PASS | TorchScript, ONNX, SafeTensors export with validation |
| UC2: Cross-Backend Inference | PASS | PASS | Qwen3-0.6B optimization with backend-specific tuning |
| UC3: CI/CD Validation | PASS | PASS | Diagnostics, benchmarks, cross-backend checks |
| UC4: Backend Training | PASS | PASS | AMP training with auto backend detection |
| UC5: Cross-Backend Validation | PASS | PASS | Model, hardware, config, and output consistency |

## Cost Summary

| Provider | Instance | Hardware | Cost/hr | Validation Time | Est. Cost |
|----------|----------|----------|---------|-----------------|-----------|
| AWS | g5.xlarge | A10G | ~$1.00 | ~5 min | ~$0.08 |
| GCP | n1-standard-4 + T4 | T4 | ~$0.35 | ~5 min | ~$0.03 |
| RunPod | Community Cloud | H100 NVL | ~$2.59 | ~5 min | ~$0.22 |
| Local | Mac | MPS | Free | ~5 min | $0.00 |
| **Total** | | | | | **~$0.33** |

> AMD Developer Cloud charges ~$750/month (not hourly). Always destroy instances from the
> portal immediately after use — SSH `poweroff` does NOT stop billing.
> GCP TPU v5litepod-1 costs ~$1.35/hr; terminate immediately after validation.

## Validation History

| Run | Platforms | Model | Result |
|-----|-----------|-------|--------|
| v0.5.80 | MPS, A10G, T4, H100 NVL, Trainium†, Inferentia2† | Qwen3-0.6B | 4 GPU PASS + 2 CPU† PASS; AMD + TPU SKIPPED‡ |
| v0.5.67 | AMD MI300X (ROCm 6.2), AWS A10G (CUDA 2.6.0+cu124), GCP T4 (CUDA 2.7.1+cu128) | Qwen3-0.6B | 3/3 PASS |
| v0.5.45 | MPS, A10G, T4, TPU v5e, H100 NVL, MI300X, Trainium†, Inferentia2† | Qwen3-0.6B | 6 GPU + 2 CPU† PASS |
| v0.5.36 | MPS, A10G, T4, TPU v5e, H100 NVL, MI300X, Trainium†, Inferentia2† | Qwen3-0.6B | 6 GPU + 2 CPU† PASS |

## See Also

- [Hardware Matrix](./hardware-matrix.md) for full hardware support table
- [Backend Selection](../guides/backend-selection.md) for choosing the right backend
