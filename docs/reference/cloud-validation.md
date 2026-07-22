# TorchBridge Cloud Validation Results

**Status**: 5/8 validated — 4 GPU PASS, 1 Trainium PASS, 3 PENDING† (v0.5.100, 2026-07-21)

## Platform Summary

| Platform | Hardware | Provider | Memory | Architecture |
|----------|----------|----------|--------|-------------|
| Local Mac | Apple Silicon | Local | Unified | MPS |
| AWS g5.xlarge | NVIDIA A10G | AWS | 24 GB | Ampere (sm_86) |
| GCP g2-standard-4 | NVIDIA L4 | GCP | 24 GB | Ada Lovelace (sm_89) |
| RunPod SECURE | NVIDIA H100 NVL | RunPod | 100 GB | Hopper (sm_90) |
| AWS trn1.2xlarge | Trainium (NeuronCore v2) | AWS | 32 GB | Trainium1/NeuronX 2.9 |
| AWS inf2.xlarge | Inferentia2 | AWS | 32 GB | Inferentia2 |
| AMD Developer Cloud | AMD MI300X | AMD | 192 GB | CDNA3 (gfx942) |
| GCP TPU VM | TPU v5e | GCP | 16 GB/chip | v5e (v5litepod-1) |

## Cross-Backend Consistency (Qwen3-0.6B)

All validations compare GPU/accelerator logits against CPU baseline on the same model and input.

| Platform | Hardware | Max Diff | Cosine Sim | Latency | Status |
|----------|----------|----------|------------|---------|--------|
| Local | Apple MPS | 3.72e-05 | 1.000002 | 30.3 ms | PASS |
| AWS | A10G sm_86 | 2.62e-05 | 1.000001 | 35.8 ms | PASS |
| GCP | L4 sm_89 | 2.77e-05 | 1.000001 | 48.6 ms | PASS |
| RunPod | H100 NVL sm_90 | 2.29e-05 | 1.000001 | 17.5 ms | PASS |
| AWS Trainium | trn1.2xlarge (NeuronX 2.9) | 2.77e-05 | 1.000001 | 31.5 ms | PASS |
| AMD DevCloud† | MI300X | — | — | — | PENDING |
| GCP† | TPU v5e | — | — | — | PENDING |
| AWS Inferentia2† | inf2.xlarge | — | — | — | PENDING |

> **† Pending — infrastructure access constraints.**
> AMD MI300X: validation requires user at portal (no CLI stop — ~$750/mo if forgotten). Deferred.
> GCP TPU v5e: quota exhausted. Deferred.
> AWS Inferentia2: deferred to next validation cycle.
> These are infrastructure availability constraints, not code issues. The backend implementations
> are exercised in the unit test suite.

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

| Use Case | AWS A10G | GCP L4 | Description |
|----------|----------|--------|-------------|
| UC1: tb-doctor | PASS | PASS | System health check (8/8 on both) |
| UC2: Cross-Backend Validation | PASS | PASS | Qwen3-0.6B CPU↔GPU numerical consistency |
| UC3: Tolerance thresholds | PASS | PASS | max_diff ≤ 1e-4, cos_sim ≥ 0.9999 |

## Cost Summary

| Provider | Instance | Hardware | Cost/hr | Validation Time | Est. Cost |
|----------|----------|----------|---------|-----------------|-----------|
| AWS | g5.xlarge | A10G | ~$1.00 | ~5 min | ~$0.08 |
| GCP | g2-standard-4 + L4 | L4 | ~$0.70 | ~5 min | ~$0.06 |
| RunPod | SECURE | H100 NVL | ~$3.00 | ~5 min | ~$0.25 |
| AWS | trn1.2xlarge | Trainium | ~$1.34 | ~5 min | ~$0.11 |
| Local | Mac | MPS | Free | ~5 min | $0.00 |
| **Total** | | | | | **~$0.50** |

> AMD Developer Cloud charges ~$750/month (not hourly). Always destroy instances from the
> portal immediately after use — SSH `poweroff` does NOT stop billing.
> GCP TPU v5litepod-1 costs ~$1.35/hr; terminate immediately after validation.

## Validation History

| Run | Platforms | Model | Result |
|-----|-----------|-------|--------|
| v0.5.100 (2026-07-22) | MPS, A10G, L4, H100 NVL, Trainium (NeuronX 2.9) | Qwen3-0.6B | 5/5 PASS — max_diff ≤ 3.72e-05 on all platforms (NeuronCore real chip) |
| v0.5.80 (2026-03-30) | MPS, A10G, T4, H100 NVL, Trainium†, Inferentia2† | Qwen3-0.6B | 4 GPU PASS + 2 CPU† PASS; AMD + TPU SKIPPED |
| v0.5.67 (2026-03-10) | AMD MI300X, AWS A10G, GCP T4 | Qwen3-0.6B | 3/3 PASS |

## See Also

- [Hardware Matrix](./hardware-matrix.md) for full hardware support table
- [Backend Selection](../guides/backend-selection.md) for choosing the right backend
