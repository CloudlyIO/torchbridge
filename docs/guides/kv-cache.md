# KV-Cache Optimization Guide

TorchBridge provides backend-aware KV-cache quantization and prefix caching for LLM serving. The system auto-selects the optimal KV-cache data type per detected hardware, with ordered fallback chains.

## Overview

KV-cache memory is the dominant bottleneck in LLM serving. Quantizing KV-cache from FP16 to FP8 or FP4 reduces memory by 2-4x, enabling longer context lengths and higher batch sizes without quality loss.

TorchBridge extends its backend compatibility matrix pattern (used for model quantization and attention dispatch) to KV-cache dtypes.

## Backend Compatibility Matrix

| Backend | Architecture | Optimal KV Dtype | Memory Factor | Supported |
|---------|-------------|-----------------|---------------|-----------|
| CUDA | Blackwell DC | NVFP4 | 0.25x | NVFP4, FP8, BF16 |
| CUDA | Blackwell Consumer | FP8 E4M3 | 0.5x | FP8, BF16, FP16 |
| CUDA | Hopper (H100) | FP8 E4M3 | 0.5x | FP8, BF16, FP16 |
| CUDA | Ada (RTX 4090) | FP8 E4M3 | 0.5x | FP8, BF16, FP16 |
| CUDA | Ampere (A100) | BF16 | 1.0x | BF16, FP16 |
| AMD | CDNA3/4 (MI300X) | FP8 E4M3 | 0.5x | FP8, BF16 |
| AMD | CDNA2 | BF16 | 1.0x | BF16 |
| Trainium | TRN1-3 | BF16 | 1.0x | BF16 |
| TPU | v4-v7 | BF16 | 1.0x | BF16 |
| CPU | — | Passthrough | 1.0x | Passthrough |

## KV-Cache Quantization API

### Auto-select dtype based on hardware

```python
from torchbridge.models.llm.kv import (
    KVCacheCompatibilityMatrix,
    QuantizedCacheConfig,
    QuantizedKVCache,
)

# Auto-select optimal KV dtype
qconfig = QuantizedCacheConfig()
qcache = QuantizedKVCache(qconfig, backend_name="cuda")
print(f"Using: {qcache.kv_dtype.value}")  # e.g. "fp8_e4m3" on H100
```

### Query the compatibility matrix

```python
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture
from torchbridge.models.llm.kv import KVCacheCompatibilityMatrix

optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# KVCacheDtype.FP8_E4M3

supported = KVCacheCompatibilityMatrix.get_supported_dtypes(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# [FP8_E4M3, BF16, FP16]
```

## Prefix Caching

Prefix caching deduplicates KV computation for shared prompt prefixes (system prompts, few-shot examples). Uses SHA-256 hashing with LRU eviction.

```python
qconfig = QuantizedCacheConfig(
    cache_config=cache_config,
    enable_prefix_caching=True,
    prefix_cache_max_entries=1024,
    prefix_cache_max_tokens=65536,
)
qcache = QuantizedKVCache(qconfig, backend_name="cuda")

# Store prefix KV tensors
system_prompt_ids = tuple(tokenizer.encode("You are a helpful assistant."))
qcache.store_prefix(system_prompt_ids, kv_tensors)

# Lookup on subsequent requests
entry = qcache.lookup_prefix(system_prompt_ids)
if entry:
    # Skip prefill, use cached KV tensors
    pass

# Check hit rate
stats = qcache.get_prefix_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

## CLI Usage

```bash
# Show optimal KV dtype for current hardware
torchbridge cache

# Show for specific backend
torchbridge cache --backend nvidia

# Full compatibility matrix
torchbridge cache --show-matrix

# JSON output for CI
torchbridge cache --ci
```

## Troubleshooting

### FP8 KV-cache not available
FP8 E4M3 requires Hopper (H100), Ada (RTX 4090), or CDNA3+ (MI300X) hardware. On Ampere (A100), KV-cache will use BF16 instead.

### Prefix cache misses
Ensure prefix token IDs are identical (including BOS/EOS tokens). Even a single token difference produces a different hash.

### Memory not reducing with quantization
PASSTHROUGH mode (CPU fallback) does not quantize. Check `qcache.kv_dtype` to verify the resolved dtype. NVFP4 requires Blackwell DC hardware.
