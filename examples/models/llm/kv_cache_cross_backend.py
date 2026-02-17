#!/usr/bin/env python3
"""
KV-Cache Cross-Backend Optimization Example

Demonstrates TorchBridge's backend-aware KV-cache optimization:
1. Querying the KV-cache compatibility matrix
2. Creating a quantized KV-cache with auto dtype selection
3. Prefix caching with hit rate tracking
4. LLM serving metrics (TTFT, TPOT, ITL) with GenerationTimer
"""

import time

import torch

from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture
from torchbridge.models.llm.kv import (
    KVCacheCompatibilityMatrix,
    QuantizedCacheConfig,
    QuantizedKVCache,
)
from torchbridge.models.llm.kv.cache_dtype import KV_DTYPE_SPECS
from torchbridge.models.llm.kv_cache import CacheConfig
from torchbridge.monitoring.llm_metrics import (
    GenerationTimer,
    LLMMetricsCollector,
)


def main():
    print("=" * 60)
    print("TorchBridge -- KV-Cache Cross-Backend Optimization")
    print("=" * 60)

    # -- 1. Compatibility Matrix Query ------------------------------------
    print("\n1. KV-Cache Dtype Compatibility Matrix")
    print("-" * 40)

    combos = [
        ("CUDA/Blackwell DC", HardwareBackend.CUDA, NVIDIAArchitecture.BLACKWELL_DC),
        ("CUDA/Hopper", HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER),
        ("CUDA/Ampere", HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE),
        ("CPU", HardwareBackend.CPU, None),
    ]
    for label, backend, arch in combos:
        optimal = KVCacheCompatibilityMatrix.get_optimal_dtype(backend, arch)
        supported = KVCacheCompatibilityMatrix.get_supported_dtypes(backend, arch)
        spec = KV_DTYPE_SPECS[optimal]
        print(f"  {label:25s} -> {optimal.value:12s} (mem {spec.memory_factor}x)  "
              f"supported: {[d.value for d in supported]}")

    # -- 2. Quantized KV-Cache -------------------------------------------
    print("\n2. Quantized KV-Cache (auto dtype on CPU)")
    print("-" * 40)

    cache_config = CacheConfig(
        max_length=2048, num_layers=32, num_heads=32, head_dim=128,
        dtype=torch.float16, device="cpu",
    )
    qconfig = QuantizedCacheConfig(
        cache_config=cache_config,
        enable_prefix_caching=True,
        prefix_cache_max_entries=256,
        prefix_cache_max_tokens=16384,
    )
    qcache = QuantizedKVCache(qconfig, backend_name="cpu")
    print(f"  Resolved dtype: {qcache.kv_dtype.value}")
    print("  Prefix caching: enabled")

    kv = qcache.create_cache(batch_size=1)
    for layer in range(4):
        keys = torch.randn(1, 32, 1, 128)
        values = torch.randn(1, 32, 1, 128)
        kv = qcache.update_cache(kv, keys, values, layer_idx=layer)
    usage = qcache.get_memory_usage(kv)
    print(f"  Memory usage: {usage}")

    # -- 3. Prefix Caching Demo ------------------------------------------
    print("\n3. Prefix Caching Demo")
    print("-" * 40)

    shared_prefix = tuple(range(128))  # System prompt tokens
    kv_tensors = [(torch.randn(32, 128, 128), torch.randn(32, 128, 128)) for _ in range(32)]
    qcache.store_prefix(shared_prefix, kv_tensors)

    # Simulate repeated requests with same system prompt
    for _ in range(10):
        entry = qcache.lookup_prefix(shared_prefix)
        if entry:
            pass  # Use cached KV instead of recomputing

    # Some requests with different prefixes
    for i in range(5):
        qcache.lookup_prefix(tuple(range(i * 100, i * 100 + 50)))

    stats = qcache.get_prefix_cache_stats()
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"  Cached entries: {stats['cached_entries']}")

    # -- 4. LLM Metrics Demo ---------------------------------------------
    print("\n4. LLM Serving Metrics (GenerationTimer)")
    print("-" * 40)

    collector = LLMMetricsCollector()

    for req_idx in range(20):
        timer = GenerationTimer(prompt_tokens=64, cache_hit=(req_idx % 3 == 0))
        with timer:
            # Simulate prefill
            time.sleep(0.002)
            timer.record_first_token()
            # Simulate decode
            for _ in range(10):
                time.sleep(0.001)
                timer.record_token()

        metrics = timer.finalize(generated_tokens=10)
        collector.record_request(
            metrics,
            model_name="Qwen/Qwen3-0.6B",
            itl_series=timer.itl_series,
        )

    snap = collector.get_snapshot()
    print(f"  Total requests:   {snap.total_requests}")
    print(f"  TTFT p50/p95/p99: {snap.ttft_p50_ms:.1f} / {snap.ttft_p95_ms:.1f} / {snap.ttft_p99_ms:.1f} ms")
    print(f"  TPOT p50/p95/p99: {snap.tpot_p50_ms:.1f} / {snap.tpot_p95_ms:.1f} / {snap.tpot_p99_ms:.1f} ms")
    print(f"  ITL  p50/p95/p99: {snap.itl_p50_ms:.1f} / {snap.itl_p95_ms:.1f} / {snap.itl_p99_ms:.1f} ms")
    print(f"  Cache hit rate:   {snap.cache_hit_rate:.1%}")
    print(f"  Tokens/sec:       {snap.tokens_per_second:.1f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
