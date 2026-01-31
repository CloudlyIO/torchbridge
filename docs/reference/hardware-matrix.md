# Hardware Support Matrix

Complete hardware support table for TorchBridge backends.

## GPU Support

### NVIDIA

| Architecture | GPUs | Compute Cap | FP32 | FP16 | BF16 | FP8 | FlashAttention | Tensor Cores |
|-------------|------|-------------|------|------|------|-----|---------------|--------------|
| **Hopper** | H100, H200 | sm_90 | Yes | Yes | Yes | Yes | v3 | Gen4 |
| **Ada Lovelace** | L4, L40, RTX 4090 | sm_89 | Yes | Yes | Yes | Inference | v2 | Gen4 |
| **Ampere** | A100, A10G, RTX 3090 | sm_80/86 | Yes | Yes | Yes | No | v2 | Gen3 |
| **Turing** | T4, RTX 2080 | sm_75 | Yes | Yes | No | No | No | Gen2 |
| **Volta** | V100 | sm_70 | Yes | Yes | No | No | No | Gen1 |

### AMD

| Architecture | GPUs | FP32 | FP16 | BF16 | Matrix Cores |
|-------------|------|------|------|------|-------------|
| **CDNA3** | MI300X, MI300A | Yes | Yes | Yes | Yes |
| **CDNA2** | MI200, MI250X | Yes | Yes | Yes | Yes |
| **RDNA3** | RX 7900 XTX | Yes | Yes | Yes | No |
| **RDNA2** | RX 6900 XT | Yes | Yes | No | No |

### Intel

| Architecture | Hardware | FP32 | FP16 | BF16 | XMX Cores |
|-------------|----------|------|------|------|-----------|
| **Xe-HPC (PVC)** | Data Center GPU Max | Yes | Yes | Yes | Yes |
| **Xe-HPG (Arc)** | Arc A770, A750 | Yes | Yes | Yes | Yes |
| **Xe-LP (Flex)** | Flex 170, 140 | Yes | Yes | Partial | No |

## TPU Support

| Generation | Chip | BF16 | FP32 | HBM | Pods |
|-----------|------|------|------|-----|------|
| **v7** | Latest | Yes | Yes | 128GB | Yes |
| **v6e** | Trillium | Yes | Yes | 32GB | Yes |
| **v5p** | Viperlight | Yes | Yes | 95GB | Yes |
| **v5e** | Pufferfish | Yes | Yes | 16GB | Yes |
| **v4** | Jf | Yes | Yes | 32GB | Yes |

## Feature Support by Backend

| Feature | NVIDIA | AMD | Intel | TPU | CPU |
|---------|--------|-----|-------|-----|-----|
| Mixed precision | Yes | Yes | Yes | Yes | Partial |
| torch.compile | Yes | Yes | Yes | Partial | Yes |
| Distributed (DDP) | Yes | Yes | Yes | Yes | No |
| FSDP | Yes | Yes | Partial | Partial | No |
| Tensor Parallel | Yes | Yes | Partial | Partial | No |
| Gradient checkpointing | Yes | Yes | Yes | Yes | Yes |
| Activation offloading | Yes | Yes | Yes | Partial | N/A |
| Model export (TorchScript) | Yes | Yes | Yes | Partial | Yes |
| Model export (ONNX) | Yes | Yes | Yes | Partial | Yes |

## Cloud Platform Availability

| GPU | AWS | GCP | Azure |
|-----|-----|-----|-------|
| NVIDIA H100 | p5.48xlarge | a3-highgpu | ND H100 v5 |
| NVIDIA A100 | p4d.24xlarge | a2-highgpu | ND A100 v4 |
| NVIDIA L4 | g6.xlarge | g2-standard | -- |
| NVIDIA T4 | g4dn.xlarge | n1 + T4 | NC T4 v3 |
| AMD MI300X | -- | -- | ND MI300X v5 |
| AMD MI200 | -- | -- | -- |
| Intel Max | -- | -- | -- |
| TPU v5e | -- | ct5lp-hightpu | -- |
| TPU v4 | -- | ct4p-hightpu | -- |

## Validated Configurations

These configurations have been tested and validated:

| Configuration | Platform | Tests | Status |
|--------------|----------|-------|--------|
| NVIDIA L4 (Ada) | GCP | Passed | Validated |
| NVIDIA A10G (Ampere) | AWS | Passed | Validated |
| TPU v5e | GCP | Passed | Validated |
| CPU (x86) | Local | Passed | Validated |
| CPU (ARM/Apple Silicon) | Local | Passed | Validated |

## See Also

- [Backends Overview](../backends/overview.md)
- [Backend Selection](../guides/backend-selection.md)
- [Hardware Setup](../guides/hardware-setup.md)
