# Inference Optimization Guide

TorchBridge provides three backend-aware inference optimization systems, each built on
the same compatibility matrix pattern: auto-select the optimal configuration for your
hardware and fall back gracefully when the preferred option is unavailable.

## Attention Dispatch

TorchBridge automatically selects the optimal attention kernel for your hardware,
with ordered fallback chains when the preferred kernel is unavailable at runtime.

### Quick Start

```python
from torchbridge.attention import AttentionDispatcher

dispatcher = AttentionDispatcher()
result = dispatcher.select_kernel(seq_length=2048, num_heads=32, head_dim=128)
print(f"Selected: {result.kernel_type.value}")
print(f"Fallbacks: {[k.value for k in result.fallback_chain]}")
```

### Kernel Compatibility Matrix

| Backend | Architecture | Optimal Kernel | Fallback Chain |
|---------|-------------|---------------|----------------|
| NVIDIA | Blackwell DC/Consumer | FlexAttention | FA-3 > FA-2 > PyTorch SDPA |
| NVIDIA | Hopper | FlexAttention | FA-3 > FA-2 > PyTorch SDPA |
| NVIDIA | Ampere/Ada | FlashAttention-2 | FlexAttention > PyTorch SDPA |
| NVIDIA | Turing/Volta/Pascal | PyTorch SDPA | — |
| AMD | CDNA3/CDNA4 | FA-2 (Composable Kernel) | PyTorch SDPA |
| AMD | CDNA2/RDNA | PyTorch SDPA | — |
| Trainium | TRN2/TRN3 | NeuronX SDPA | PyTorch SDPA |
| Trainium | TRN1/INF2 | PyTorch SDPA | — |
| TPU | v5+/v7 | Pallas Attention | PyTorch SDPA |
| TPU | v4 | PyTorch SDPA | — |
| CPU | any | PyTorch SDPA | — |

### Benchmark Cache

The dispatcher caches kernel latency measurements at `~/.torchbridge/kernel_benchmarks.json`.
The cache is automatically invalidated when the hardware fingerprint changes.

```python
from torchbridge.attention import AttentionDispatcher

dispatcher = AttentionDispatcher(use_benchmark_cache=True)
result = dispatcher.select_kernel(seq_length=2048, num_heads=32, head_dim=128)

if result.benchmark_latency_ms is not None:
    print(f"Cached latency: {result.benchmark_latency_ms:.2f} ms")
```

### Query the Matrix Directly

```python
from torchbridge.attention import AttentionDispatchMatrix, AttentionKernelType
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture, AMDArchitecture

# What kernels are supported on Hopper?
kernels = AttentionDispatchMatrix.get_supported_kernels(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)

# Is FlexAttention supported on AMD CDNA3?
cdna3_kernels = AttentionDispatchMatrix.get_supported_kernels(
    HardwareBackend.AMD, AMDArchitecture.CDNA3
)
supported = AttentionKernelType.FLEX_ATTENTION in cdna3_kernels
# → False
```

---

## KV-Cache Optimization

KV-cache memory is the dominant bottleneck in LLM serving. Quantizing KV-cache from
FP16 to FP8 or FP4 reduces memory by 2–4×, enabling longer context lengths and higher
batch sizes without quality loss.

### Backend Compatibility Matrix

| Backend | Architecture | Optimal KV Dtype | Memory Factor |
|---------|-------------|-----------------|---------------|
| CUDA | Blackwell DC | NVFP4 | 0.25× |
| CUDA | Blackwell Consumer | FP8 E4M3 | 0.5× |
| CUDA | Hopper (H100) | FP8 E4M3 | 0.5× |
| CUDA | Ada (RTX 4090) | FP8 E4M3 | 0.5× |
| CUDA | Ampere (A100) | BF16 | 1.0× |
| AMD | CDNA3/4 (MI300X) | FP8 E4M3 | 0.5× |
| AMD | CDNA2 | BF16 | 1.0× |
| Trainium | TRN1–3 | BF16 | 1.0× |
| TPU | v4–v7 | BF16 | 1.0× |
| CPU | — | Passthrough | 1.0× |

### Python API

```python
from torchbridge.models.llm.kv import (
    KVCacheCompatibilityMatrix,
    QuantizedCacheConfig,
    QuantizedKVCache,
)
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture

# Auto-select optimal KV dtype
qconfig = QuantizedCacheConfig()
qcache = QuantizedKVCache(qconfig, backend_name="cuda")
print(f"Using: {qcache.kv_dtype.value}")  # e.g. "fp8_e4m3" on H100

# Query compatibility matrix
optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# → KVCacheDtype.FP8_E4M3
```

### Prefix Caching

Prefix caching deduplicates KV computation for shared prompt prefixes (system prompts,
few-shot examples). Uses SHA-256 hashing with LRU eviction.

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
    pass  # Skip prefill, use cached KV tensors

# Check hit rate
stats = qcache.get_prefix_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

### CLI

```bash
# Show optimal KV dtype for detected hardware
tb-cache

# Show for specific backend
tb-cache --backend nvidia

# Full compatibility matrix
tb-cache --show-matrix

# JSON output for CI
tb-cache --ci
```

---

## Speculative Decoding

TorchBridge provides backend-aware speculative decoding method selection via a compatibility
matrix, and an `OutputFormat` enum for structured output configuration. For the generation
loop itself, use `model.generate()` APIs directly.

### Methods

| Method | Description | Requirements |
|--------|-------------|-------------|
| `draft_model` | Standard draft-verify with smaller assistant model | Draft model |
| `eagle` | EAGLE with custom CUDA kernels | NVIDIA Hopper+ |
| `layer_skip` | Self-speculative by skipping later layers | None |
| `medusa` | Multi-head speculative with tree attention | NVIDIA Ampere+ |
| `prompt_lookup` | N-gram matching from prompt tokens | None (universal) |

### Backend Compatibility Matrix

| Backend | Architecture | Optimal | Supported |
|---------|-------------|---------|-----------|
| NVIDIA | Blackwell/Hopper | Draft Model | draft_model, prompt_lookup |
| NVIDIA | Ampere/Ada | Draft Model | draft_model, prompt_lookup |
| AMD | CDNA3/CDNA4 | Draft Model | draft_model, prompt_lookup |
| Trainium | TRN2/TRN3 | Prompt Lookup | prompt_lookup |
| TPU | v5e/v7 | Prompt Lookup | prompt_lookup |
| CPU | — | Prompt Lookup | prompt_lookup |

### Query the Matrix

```python
from torchbridge.inference import SpeculationCompatibilityMatrix, SpeculativeMethod
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture, AMDArchitecture

# Get optimal method for hardware
optimal = SpeculationCompatibilityMatrix.get_optimal_method(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# → SpeculativeMethod.DRAFT_MODEL

# Get fallback chain
chain = SpeculationCompatibilityMatrix.get_fallback_chain(
    SpeculativeMethod.EAGLE, HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
)

# Check if a method is supported on AMD CDNA3
is_supported = SpeculationCompatibilityMatrix.is_method_supported(
    SpeculativeMethod.EAGLE, HardwareBackend.AMD, AMDArchitecture.CDNA3
)
# → False
```

### Structured Output Formats

```python
from torchbridge.inference import OutputFormat

fmt = OutputFormat.JSON_SCHEMA   # JSON schema-constrained output
fmt = OutputFormat.REGEX          # Regex-constrained output
fmt = OutputFormat.JSON           # Unconstrained JSON
fmt = OutputFormat.TEXT           # Plain text (default)
```

> **Note:** TorchBridge defines `OutputFormat` for naming consistency. Use `xgrammar` or
> `outlines` for the actual logits-processor and grammar engine — TorchBridge does not
> provide a generation loop.

### CLI

```bash
# Show optimal speculative method for detected hardware
tb-speculate

# Show full compatibility matrix
tb-speculate --show-matrix

# Check specific method on specific backend
tb-speculate --backend nvidia --method eagle

# JSON output for CI
tb-speculate --ci
```

## See Also

- [Performance Tuning](performance-tuning.md)
- [Backend Selection](backend-selection.md)
- [Model Optimization](model-optimization.md)
