# Hardware Support Matrix

Complete hardware support table for TorchBridge backends.

## GPU Support

### NVIDIA

| Architecture | GPUs | Compute Cap | FP32 | FP16 | BF16 | FP8 | FP4 | FlashAttention | Tensor Cores |
|-------------|------|-------------|------|------|------|-----|-----|---------------|--------------|
| **Blackwell DC** | B100, B200, GB200 | sm_100 | Yes | Yes | Yes | Yes | **Yes** | v3 | Gen5 |
| **Blackwell Consumer** | RTX 5090, RTX 5080 | sm_120 | Yes | Yes | Yes | Yes | No | v3 | Gen5 |
| **Hopper** | H100, H200 | sm_90 | Yes | Yes | Yes | Yes | No | v3 | Gen4 |
| **Ada Lovelace** | L4, L40, RTX 4090 | sm_89 | Yes | Yes | Yes | Inference | No | v2 | Gen4 |
| **Ampere** | A100, A10G, RTX 3090 | sm_80/86 | Yes | Yes | Yes | No | No | v2 | Gen3 |
| **Turing** | T4, RTX 2080 | sm_75 | Yes | Yes | No | No | No | No | Gen2 |
| **Volta** | V100 | sm_70 | Yes | Yes | No | No | No | No | Gen1 |

#### Blackwell Data Center Specifications

| GPU | Memory | Bandwidth | TDP | NVLink | Key Feature |
|-----|--------|-----------|-----|--------|-------------|
| B100 | 192 GB HBM3e | 8 TB/s | 700W | NVLink 5 (1.8 TB/s) | NVFP4, Confidential Computing |
| B200 | 192 GB HBM3e | 8 TB/s | 1000W | NVLink 5 (1.8 TB/s) | NVFP4, Confidential Computing |
| GB200 | 384 GB (2x die) | 16 TB/s combined | 1200W | NVLink-C2C | Dual-die, NVLink 5 |

#### Blackwell Consumer Specifications

| GPU | Memory | Bandwidth | TDP | Key Feature |
|-----|--------|-----------|-----|-------------|
| RTX 5090 | 32 GB GDDR7 | 1792 GB/s | 575W | DLSS 4, RT Cores v4 |
| RTX 5080 | 16 GB GDDR7 | 960 GB/s | 360W | DLSS 4, RT Cores v4 |

### AMD

| Architecture | GPUs | Arch ID | Memory | FP32 | FP16 | BF16 | FP8 | FP4 | Matrix Cores |
|-------------|------|---------|--------|------|------|------|-----|-----|-------------|
| **CDNA 4** | MI350X, MI355X | gfx950 | 288 GB HBM3e | Yes | Yes | Yes | Yes | **Yes** | Yes |
| **CDNA 3** | MI300X, MI300A | gfx942 | 192 GB HBM3 | Yes | Yes | Yes | Yes | No | Yes |
| **CDNA 3** | MI325X | gfx942 | 256 GB HBM3e | Yes | Yes | Yes | Yes | No | Yes |
| **CDNA 2** | MI200, MI250X | gfx90a | 128 GB HBM2e | Yes | Yes | Yes | No | No | Yes |
| **RDNA 3** | RX 7900 XTX | gfx1100 | 24 GB GDDR6 | Yes | Yes | Yes | No | No | No |
| **RDNA 2** | RX 6900 XT | gfx1030 | 16 GB GDDR6 | Yes | Yes | No | No | No | No |

## TPU Support

| Generation | Chip | BF16 | FP32 | FP8 | HBM | Pods |
|-----------|------|------|------|-----|-----|------|
| **v7 (Ironwood)** | Latest | Yes | Yes | Yes | 192 GB HBM3e | Yes |
| **v6e (Trillium)** | Trillium | Yes | Yes | Partial | 32 GB | Yes |
| **v5p** | Viperlight | Yes | Yes | No | 95 GB | Yes |
| **v5e** | Pufferfish | Yes | Yes | No | 16 GB | Yes |
| **v4** | Jf | Yes | Yes | No | 32 GB | Yes |

## Feature Support by Backend

| Feature | NVIDIA | AMD | Trainium | TPU | CPU |
|---------|--------|-----|----------|-----|-----|
| Mixed precision | Yes | Yes | Yes | Yes | Partial |
| FP8 training | H100+ | MI300X+ | Trn2+ | No | No |
| FP4 inference | B100+ | MI350X+ | No | No | No |
| torch.compile | Yes | Yes | Partial | Partial | Yes |
| Distributed (DDP) | Yes | Yes | Yes | Yes | No |
| FSDP | Yes | Yes | Yes | Partial | No |
| Tensor Parallel | Yes | Yes | Yes | Partial | No |
| Gradient checkpointing | Yes | Yes | Yes | Yes | Yes |
| Activation offloading | Yes | Yes | Yes | Partial | N/A |
| Model export (TorchScript) | Yes | Yes | No | Partial | Yes |
| Model export (ONNX) | Yes | Yes | No | Partial | Yes |

## Cloud Platform Availability

| GPU | AWS | GCP | Azure | RunPod | AMD DevCloud |
|-----|-----|-----|-------|--------|-------------|
| NVIDIA B200 | Not yet | Not yet | Not yet | -- | -- |
| NVIDIA H100 | p5.48xlarge | a3-highgpu | ND H100 v5 | **H100 NVL/SXM/PCIe** | -- |
| NVIDIA A100 | p4d.24xlarge | a2-highgpu | ND A100 v4 | A100 SXM/PCIe | -- |
| NVIDIA L4 | g6.xlarge | g2-standard | -- | L4 | -- |
| NVIDIA A10G | g5.xlarge | -- | -- | -- | -- |
| NVIDIA T4 | g4dn.xlarge | n1 + T4 | NC T4 v3 | -- | -- |
| AMD MI300X | -- | -- | ND MI300X v5 | MI300X | **MI300X** |
| AMD MI325X | -- | -- | -- | -- | -- |
| TPU v6e | -- | ct6e-standard | -- | -- | -- |
| TPU v5e | -- | **ct5lp/v5litepod** | -- | -- | -- |
| TPU v4 | -- | ct4p-hightpu | -- | -- | -- |
| Trainium | **trn1.2xlarge** | -- | -- | -- | -- |

## Validated Configurations

These configurations have been tested and validated with Qwen3-0.6B cross-backend consistency:

| Configuration | Platform | Max Diff | Cosine Sim | Latency |
|--------------|----------|----------|------------|---------|
| NVIDIA H100 NVL (Hopper) | RunPod | 2.29e-05 | 1.000001 | 18.8 ms |
| Apple Silicon (MPS) | Local Mac | 4.58e-05 | 1.000002 | 27.8 ms |
| AMD MI300X (CDNA3) | AMD Developer Cloud | 4.82e-05 | 1.000001 | 30.0 ms |
| NVIDIA A10G (Ampere) | AWS g5.xlarge | 1.96e-05 | 1.000001 | 41.8 ms |
| TPU v5e | GCP TPU VM | 1.08e-01 | 0.999980 | 47.5 ms |
| NVIDIA T4 (Turing) | GCP n1 + T4 | 2.67e-05 | 1.000001 | 50.8 ms |
| CPU (x86) | Local | Full test suite | — | — |
| CPU (ARM/Apple Silicon) | Local | Full test suite | — | — |

## See Also

- [Backends Overview](../backends/overview.md)
- [Backend Selection](../guides/backend-selection.md)
- [Hardware Setup](../guides/hardware-setup.md)
