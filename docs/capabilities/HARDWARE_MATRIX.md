# Hardware Support Matrix

> **Version**: v0.4.15 | **Status**: ✅ Current | **Last Updated**: Jan 22, 2026

Comprehensive support across NVIDIA, AMD, Intel, and Google TPU hardware.

## GPU Support

| Vendor | Architecture | Model Examples | Backend | Status |
|--------|--------------|----------------|---------|--------|
| NVIDIA | Hopper | H100, H200 | CUDA 12+ | ✅ Full |
| NVIDIA | Ampere | A100, A10G, RTX 4090 | CUDA 11.8+ | ✅ Full |
| NVIDIA | Turing | RTX 2080 Ti, T4 | CUDA 11+ | ✅ Full |
| AMD | CDNA2 | MI250X, MI300 | ROCm 6.0+ | ✅ Full |
| AMD | RDNA3 | RX 7900 XTX | ROCm 6.0+ | ✅ Full |
| Intel | Xe-HPG | Arc A770 | IPEX 2.0+ | ✅ Full |
| Intel | Xe-HPC | Ponte Vecchio | IPEX 2.0+ | ✅ Full |

## TPU Support

| Version | Status | Notes |
|---------|--------|-------|
| TPU v2 | ✅ Full | Via torch_xla |
| TPU v3 | ✅ Full | Via torch_xla |
| TPU v4 | ✅ Full | Via torch_xla |

## Feature Support by Backend

| Feature | NVIDIA | AMD | Intel | TPU |
|---------|--------|-----|-------|-----|
| FP32 | ✅ | ✅ | ✅ | ✅ |
| FP16 | ✅ | ✅ | ✅ | ✅ |
| BF16 | ✅ | ✅ | ✅ | ✅ |
| FP8 | ✅ H100+ | ⚠️ MI300+ | ❌ | ❌ |
| Flash Attention | ✅ | ✅ | ✅ | ⚠️ Custom |
| Triton Kernels | ✅ | ✅ | ⚠️ Limited | ❌ |
| Mixed Precision | ✅ | ✅ | ✅ | ✅ |

## Cloud Platform Availability

| Platform | NVIDIA | AMD | Intel | TPU |
|----------|--------|-----|-------|-----|
| AWS | A100, A10G, H100 | MI250X | - | - |
| GCP | L4, A100, H100 | - | - | v2/v3/v4 |
| Azure | A100, V100 | MI250X | - | - |

## Validated Configurations

The following platforms have been tested and validated:
- GCP L4 GPU (v0.4.6)
- AWS A10G GPU (v0.4.6)
- GCP TPU (v0.4.6)
- AWS AMD GPU (v0.4.6)
- Intel DevCloud XPU (v0.4.7)

**Note**: Test reports are stored locally per [Reports Policy](../REPORTS_POLICY.md).

## Backend Selection

For detailed configuration guides, see:
- [NVIDIA Backend](../backends/nvidia.md)
- [AMD Backend](../backends/amd.md)
- [Intel Backend](../backends/intel.md)
- [TPU Backend](../backends/tpu.md)
- [Backend Selection Guide](../guides/backend_selection.md)
