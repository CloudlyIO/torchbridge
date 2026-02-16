# Backend-Aware Attention Dispatch

TorchBridge automatically selects the optimal attention kernel for your hardware,
with ordered fallback chains when the preferred kernel is unavailable at runtime.

## Quick Start

```python
from torchbridge.attention import AttentionDispatcher, AttentionModuleConfig

# Auto-detect hardware and select optimal kernel
dispatcher = AttentionDispatcher()
result = dispatcher.select_kernel(seq_length=2048, num_heads=32, head_dim=128)
print(f"Selected: {result.kernel_type.value}")
print(f"Fallbacks: {[k.value for k in result.fallback_chain]}")

# Create an attention layer with the best kernel
config = AttentionModuleConfig(embed_dim=4096, num_heads=32)
attention = dispatcher.create_attention(config)
```

## Kernel Compatibility Matrix

| Backend | Architecture | Optimal Kernel | Fallback Chain |
|---------|-------------|---------------|----------------|
| NVIDIA | Blackwell DC/Consumer | FlexAttention | FA-3 > FA-2 > PyTorch SDPA |
| NVIDIA | Hopper | FlexAttention | FA-3 > FA-2 > PyTorch SDPA |
| NVIDIA | Ampere/Ada | FlashAttention-2 | FlexAttention > PyTorch SDPA |
| NVIDIA | Turing/Volta/Pascal | PyTorch SDPA | — |
| AMD | CDNA3/CDNA4 | FA-2 (Composable Kernel) | Triton > PyTorch SDPA |
| AMD | CDNA2/RDNA | PyTorch SDPA | — |
| Trainium | TRN2/TRN3 | NeuronX SDPA | PyTorch SDPA |
| Trainium | TRN1/INF2 | PyTorch SDPA | — |
| TPU | v5+/v7 | Pallas Attention | PyTorch SDPA |
| TPU | v4 | PyTorch SDPA | — |
| CPU | any | PyTorch SDPA | — |

## GQA and MQA Support

Grouped-Query Attention (GQA) reduces KV cache memory by using fewer KV heads
than query heads. Multi-Query Attention (MQA) is the extreme case with a single
KV head.

```python
from torchbridge.attention import AttentionModuleConfig

# Llama 3 style: 32 query heads, 8 KV heads
gqa_config = AttentionModuleConfig(
    embed_dim=4096, num_heads=32, num_kv_heads=8
)
print(f"KV repeat factor: {gqa_config.kv_head_repeat_factor}")  # 4

# Multi-query attention: 1 KV head
mqa_config = AttentionModuleConfig(
    embed_dim=4096, num_heads=32, num_kv_heads=1
)
print(f"KV repeat factor: {mqa_config.kv_head_repeat_factor}")  # 32

# Standard MHA (default): num_kv_heads=None
mha_config = AttentionModuleConfig(embed_dim=4096, num_heads=32)
```

**Validation rules:**
- `num_heads` must be evenly divisible by `num_kv_heads`
- `num_kv_heads=None` (default) means standard multi-head attention

## FlexAttention Score Modifiers

On Hopper+ GPUs, FlexAttention supports custom score modifiers compiled via
`torch.compile` for causal masking, sliding windows, and ALiBi:

```python
from torchbridge.attention import create_flex_attention, FlexAttentionScoreMods

# Causal FlexAttention
attn = create_flex_attention(
    embed_dim=512, num_heads=8, score_mod=FlexAttentionScoreMods.causal
)
```

## Benchmark Cache

The dispatcher caches kernel latency measurements to avoid repeated benchmarking.
The cache is stored at `~/.torchbridge/kernel_benchmarks.json` and is automatically
invalidated when the hardware fingerprint changes (new GPU, PyTorch version, etc.).

```python
from torchbridge.attention import AttentionDispatcher

dispatcher = AttentionDispatcher(use_benchmark_cache=True)
result = dispatcher.select_kernel(seq_length=2048, num_heads=32, head_dim=128)

if result.benchmark_latency_ms is not None:
    print(f"Cached latency: {result.benchmark_latency_ms:.2f} ms")
```

## Querying the Compatibility Matrix Directly

```python
from torchbridge.attention import AttentionDispatchMatrix, AttentionKernelType
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture

# What kernels are supported on Hopper?
kernels = AttentionDispatchMatrix.get_supported_kernels(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
print([k.value for k in kernels])

# Is FlexAttention supported on AMD CDNA3?
from torchbridge.core.config import AMDArchitecture
supported = AttentionDispatchMatrix.is_kernel_supported(
    AttentionKernelType.FLEX_ATTENTION, HardwareBackend.AMD, AMDArchitecture.CDNA3
)
print(f"FlexAttention on CDNA3: {supported}")  # False
```

## Architecture

The dispatch system lives in `torchbridge.attention.dispatch` and consists of:

- **`kernel_types.py`** — `AttentionKernelType` enum (8 kernel types)
- **`compatibility.py`** — `AttentionDispatchMatrix` with static lookup tables
- **`dispatcher.py`** — `AttentionDispatcher` that walks fallback chains
- **`benchmark_cache.py`** — `KernelBenchmarkCache` with fingerprint invalidation

The dispatcher integrates with the existing attention registry via
`_select_best_implementation()` in `attention/core/registry.py`, which tries
the dispatcher first and falls back to the original heuristic logic.
