# Hardware Backend Selection

> **Version**: v0.4.18 | TorchBridge supports 4 hardware backends with unified interfaces

All backends provide the same API with hardware-specific optimizations under the hood.

## Quick Selection Guide

| Hardware | Backend | Status | Best For |
|----------|---------|--------|----------|
| **NVIDIA GPUs** | [nvidia.md](nvidia.md) | ✅ Production | Training & inference, FP8 on H100+ |
| **AMD GPUs** | [amd.md](amd.md) | ✅ Production | HPC workloads, MI250X/MI300 clusters |
| **Intel GPUs** | [intel.md](intel.md) | ✅ Production | Arc for inference, Ponte Vecchio for training |
| **Google TPUs** | [tpu.md](tpu.md) | ✅ Production | Large-scale training on GCP |

## Unified Architecture

- [Backend Unification](unification.md) - How the unified interface works across vendors

## Which Backend Should I Use?

| Scenario | Recommended Backend |
|----------|---------------------|
| General training/inference | NVIDIA (widest support) |
| Cloud training at scale | TPU (best price/performance on GCP) |
| AMD hardware available | AMD ROCm (full feature parity) |
| Intel hardware available | Intel XPU (IPEX integration) |

## Common Operations

```python
from torchbridge.backends.nvidia import NVIDIABackend
from torchbridge.backends.amd import AMDBackend
from torchbridge.backends.intel import IntelBackend
from torchbridge.backends.tpu import TPUBackend

# All backends share the same API
backend = NVIDIABackend()  # or AMDBackend(), IntelBackend(), TPUBackend()
optimized_model = backend.prepare_model(model)
```

---

**See Also**: [Getting Started](../getting-started/README.md) | [Capabilities](../capabilities/README.md)
