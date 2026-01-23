# Hardware Backend Selection

KernelPyTorch supports 4 hardware backends with unified interfaces.

## Quick Selection Guide

| Hardware | Backend | Status | Guide |
|----------|---------|--------|-------|
| NVIDIA GPUs (CUDA) | [nvidia.md](nvidia.md) | ✅ Production | A100, H100, RTX 4090 |
| AMD GPUs (ROCm) | [amd.md](amd.md) | ✅ Production | MI250X, MI300, RX 7900 |
| Intel GPUs (XPU) | [intel.md](intel.md) | ✅ Production | Arc, Flex, Ponte Vecchio |
| Google TPUs | [tpu.md](tpu.md) | ✅ Production | TPU v2/v3/v4 |

## Architecture
- [Backend Unification](unification.md) - Unified interface design

## Selection Criteria
Choose based on your hardware availability and workload requirements. All backends support the full feature set with hardware-specific optimizations.
