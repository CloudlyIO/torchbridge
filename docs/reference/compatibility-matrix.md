# Compatibility Matrix

Tested and supported versions for TorchBridge v0.5.x.

## Python and PyTorch

| Python | PyTorch 2.5 | PyTorch 2.6 | PyTorch 2.7 | Notes |
|--------|-------------|-------------|-------------|-------|
| 3.10 | Yes | Yes | Yes | |
| 3.11 | Yes | Yes | Yes | |
| 3.12 | Yes | Yes | **Recommended** | Best performance and compatibility |
| 3.13 | Untested | Untested | Untested | Not yet validated |

## CUDA (NVIDIA)

| CUDA Version | Driver | Status | Notes |
|-------------|--------|--------|-------|
| 12.6 | 560+ | **Recommended** | Stable across all tested GPUs |
| 12.4 | 550+ | Supported | |
| 12.8 | 580+ | Use with caution | cu128 + driver 580 causes FP16/BF16 GEMM failures on some configs; prefer cu126 |

## ROCm (AMD)

| ROCm Version | Status | Notes |
|-------------|--------|-------|
| 6.2 | **Stable** | Validated on MI300X |
| 7.0 | Supported | Validated on AMD Developer Cloud |
| 7.2 | Current | Latest release, supported |
| 5.x | Not supported | EOL; upgrade to 6.2+ |

## Supported Backends

| Backend | Runtime | Priority | Status |
|---------|---------|----------|--------|
| NVIDIA (CUDA) | CUDA 12.x | 100 | Fully supported |
| AMD (ROCm/HIP) | ROCm 6.2+ | 90 | Fully supported |
| AWS Trainium (NeuronX) | TorchNeuron | 88 | Supported |
| TPU (XLA) | torch_xla | 85 | Supported |
| CPU | Native | 0 | Fallback (always available) |
| Intel | -- | -- | **Removed in v0.5.11** |

Intel support was removed in v0.5.11 due to Falcon Shores cancellation, Gaudi EOL, and IPEX sunset.

## Known Issues

### CUBLAS Version Mismatch (CUDA 12.8)

With CUDA 12.8 (`cu128`) and driver version 580, FP16 and BF16 GEMM operations can produce incorrect results or segfaults on certain GPU architectures. Workaround: use CUDA 12.6 (`cu126`) or pin CUBLAS to a compatible version.

### macOS Stale PCH with torch.compile

On macOS, `torch.compile` may fail with stale precompiled header errors when switching between PyTorch versions. Workaround: clear the Inductor cache:

```bash
rm -rf ~/Library/Caches/torch/inductor/
```

### ROCm SDPA Flash Attention Divergence

ROCm flash attention through SDPA may produce slightly larger numerical differences (up to 1e-3) compared to CUDA (1e-4). Functional correctness is unaffected; cosine similarity remains above 0.999.

## See Also

- [Hardware Support Matrix](hardware-matrix.md) -- detailed GPU specs and feature tables
- [Installation](../getting_started/installation.md) -- setup instructions per backend
- [Backends Overview](../backends/overview.md) -- architecture and backend selection
