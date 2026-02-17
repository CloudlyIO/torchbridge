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
from torchbridge.models.llm.kv_cache import CacheConfig

# Create base cache config
cache_config = CacheConfig(
    max_length=4096, num_layers=32, num_heads=32, head_dim=128,
)

# Auto-select optimal KV dtype
qconfig = QuantizedCacheConfig(cache_config=cache_config)
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

## LLM Serving Metrics

TorchBridge provides specialized LLM serving metrics:

- **TTFT** (Time-To-First-Token): Prefill latency
- **TPOT** (Time-Per-Output-Token): Average decode latency
- **ITL** (Inter-Token Latency): Per-token decode latency series

### GenerationTimer

```python
from torchbridge.monitoring import GenerationTimer, LLMMetricsCollector

timer = GenerationTimer(prompt_tokens=128)
with timer:
    first_token = generate_first()
    timer.record_first_token()
    for token in generate_rest():
        timer.record_token()

metrics = timer.finalize(generated_tokens=50)
print(f"TTFT: {metrics.ttft_ms:.1f} ms")
print(f"TPOT: {metrics.tpot_ms:.1f} ms")
print(f"TPS:  {metrics.tokens_per_second:.0f}")
```

### LLMMetricsCollector

```python
collector = LLMMetricsCollector(window_size=1000)

# Record each request
collector.record_request(
    metrics,
    model_name="Qwen/Qwen3-0.6B",
    batch_size=4,
    itl_series=timer.itl_series,
)

# Get aggregated snapshot
snap = collector.get_snapshot()
print(f"TTFT p50/p95/p99: {snap.ttft_p50_ms:.1f}/{snap.ttft_p95_ms:.1f}/{snap.ttft_p99_ms:.1f}")
print(f"TPOT p50/p95/p99: {snap.tpot_p50_ms:.1f}/{snap.tpot_p95_ms:.1f}/{snap.tpot_p99_ms:.1f}")
print(f"Cache hit rate:   {snap.cache_hit_rate:.1%}")
```

## Prometheus Integration

The `LLMMetricsCollector` can optionally export metrics to Prometheus:

```python
collector = LLMMetricsCollector(
    enable_prometheus=True,
    namespace="torchbridge_llm",
)
```

This creates the following Prometheus metrics:
- `torchbridge_llm_ttft_milliseconds` (Histogram)
- `torchbridge_llm_tpot_milliseconds` (Histogram)
- `torchbridge_llm_itl_milliseconds` (Histogram)
- `torchbridge_llm_requests_total` (Counter)
- `torchbridge_llm_tokens_total` (Counter)
- `torchbridge_llm_cache_hits_total` (Counter)
- `torchbridge_llm_tokens_per_second` (Gauge)

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
