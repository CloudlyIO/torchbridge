#!/usr/bin/env python3
"""
Cross-Backend Adapter Training Example

Demonstrates TorchBridge's unified adapter API: inject LoRA/DoRA adapters
into a small model, train for a few steps, merge for deployment, and
manage multiple adapters on a shared base model.

Usage:
    python examples/models/llm/adapter_cross_backend.py
"""

import torch
import torch.nn as nn

from torchbridge.adapters import (
    AdapterCompatibilityMatrix,
    AdapterConfig,
    AdapterEngine,
    AdapterMethod,
    MultiAdapterManager,
)
from torchbridge.core.config import HardwareBackend

# ── Minimal model ────────────────────────────────────────────────────────────


class TinyLM(nn.Module):
    """Tiny language model for demonstration."""

    def __init__(self, vocab_size=256, dim=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(dim, dim),
                        "k_proj": nn.Linear(dim, dim),
                        "v_proj": nn.Linear(dim, dim),
                        "o_proj": nn.Linear(dim, dim),
                    }
                )
            )
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        for layer in self.layers:
            q = layer["q_proj"](h)
            k = layer["k_proj"](h)
            v = layer["v_proj"](h)
            h = h + layer["o_proj"](q + k + v)
        return self.head(h)


# ── 1. Backend-aware adapter recommendation ──────────────────────────────────


def show_recommendations():
    """Show adapter recommendations for all backends."""
    print("=" * 60)
    print("Adapter Method Recommendations by Backend")
    print("=" * 60)

    backends = [
        HardwareBackend.CUDA,
        HardwareBackend.AMD,
        HardwareBackend.TRAINIUM,
        HardwareBackend.TPU,
        HardwareBackend.CPU,
    ]

    for backend in backends:
        optimal = AdapterCompatibilityMatrix.get_optimal(backend)
        chain = AdapterCompatibilityMatrix.get_fallback_chain(backend)
        fmt = AdapterCompatibilityMatrix.get_base_quant_format(backend)
        print(
            f"  {backend.value:<12} optimal={optimal.value:<6} "
            f"chain={[m.value for m in chain]!s:<35} "
            f"qlora_fmt={fmt.value if fmt else 'N/A'}"
        )
    print()


# ── 2. LoRA injection and training ──────────────────────────────────────────


def demo_lora_training():
    """Inject LoRA adapters and simulate a training step."""
    print("=" * 60)
    print("LoRA Adapter Training Demo")
    print("=" * 60)

    torch.manual_seed(42)
    model = TinyLM()

    config = AdapterConfig(
        method=AdapterMethod.LORA,
        rank=8,
        alpha=16.0,
        dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    engine = AdapterEngine(config, HardwareBackend.CPU)
    result = engine.inject(model)

    print(f"  Method:       {result.method_applied.value}")
    print(f"  Modules:      {result.modules_adapted}")
    print(f"  Trainable:    {result.trainable_params:,} / {result.total_params:,}")
    print(f"  Ratio:        {result.trainable_ratio:.4%}")
    print(f"  Memory:       {result.memory_before_mb:.1f} MB -> {result.memory_after_mb:.1f} MB")
    print()

    # Simulate training
    x = torch.randint(0, 256, (4, 16))
    targets = torch.randint(0, 256, (4, 16))

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-3
    )

    model.train()
    for step in range(3):
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, 256), targets.view(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step {step + 1}: loss={loss.item():.4f}")

    # Save adapter weights
    params = engine.get_adapter_params(model)
    print(f"\n  Saved {len(params)} adapter parameter tensors")

    return model, engine, params


# ── 3. Merge for deployment ──────────────────────────────────────────────────


def demo_merge(model, engine):
    """Merge adapters into base weights for zero-overhead inference."""
    print()
    print("=" * 60)
    print("Merge for Deployment")
    print("=" * 60)

    x = torch.randint(0, 256, (1, 8))
    model.eval()
    with torch.no_grad():
        pre_merge = model(x)

    merged_count = engine.merge(model)
    with torch.no_grad():
        post_merge = model(x)

    max_diff = (pre_merge - post_merge).abs().max().item()
    print(f"  Merged {merged_count} modules")
    print(f"  Max output diff: {max_diff:.2e} (should be ~0)")
    print()


# ── 4. Multi-adapter serving ────────────────────────────────────────────────


def demo_multi_adapter():
    """Demonstrate hot-swapping between multiple adapters."""
    print("=" * 60)
    print("Multi-Adapter Serving Demo")
    print("=" * 60)

    torch.manual_seed(42)
    model = TinyLM(dim=32, n_layers=1)

    config = AdapterConfig(
        method=AdapterMethod.LORA,
        rank=4,
        target_modules=["q_proj", "v_proj"],
    )
    engine = AdapterEngine(config, HardwareBackend.CPU)
    engine.inject(model)

    # Create two different "trained" adapters
    torch.manual_seed(100)
    for p in model.parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.1)
    params_a = engine.get_adapter_params(model)

    torch.manual_seed(200)
    for p in model.parameters():
        if p.requires_grad:
            p.data.normal_(0, 0.2)
    params_b = engine.get_adapter_params(model)

    # Multi-adapter manager
    mgr = MultiAdapterManager(model, max_loaded=4)
    mgr.load_adapter("sentiment", params_a, config)
    mgr.load_adapter("summarize", params_b, config)

    x = torch.randint(0, 256, (1, 8))
    model.eval()

    mgr.activate("sentiment")
    with torch.no_grad():
        out_a = model(x)

    mgr.activate("summarize")
    with torch.no_grad():
        out_b = model(x)

    diff = (out_a - out_b).abs().mean().item()
    print(f"  Loaded adapters: {mgr.loaded_count}")
    print(f"  Active: {mgr.get_active()}")
    print(f"  Mean output diff between adapters: {diff:.4f}")

    for info in mgr.list_adapters():
        print(
            f"  - {info['name']}: method={info['method']} "
            f"rank={info['rank']} active={info['active']}"
        )
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    print()
    print("TorchBridge Adapter Training — Cross-Backend Example")
    print()

    show_recommendations()
    model, engine, params = demo_lora_training()
    demo_merge(model, engine)
    demo_multi_adapter()

    print("Done!")
    print()


if __name__ == "__main__":
    main()
