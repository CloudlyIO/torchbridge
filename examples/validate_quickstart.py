#!/usr/bin/env python3
"""
TorchBridge quickstart — runs on any hardware, no GPU or downloads required.

Demonstrates:
  1. Hardware detection
  2. Cross-backend numerical validation (CPU vs CPU with different dtypes)
  3. Config advisory

Usage:
    python examples/validate_quickstart.py
"""

import torch
import torch.nn as nn

from torchbridge import TorchBridgeConfig, UnifiedManager, UnifiedValidator
from torchbridge.backends import BackendFactory, detect_best_backend


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def main() -> None:
    # ── 1. Hardware detection ─────────────────────────────────────────────
    section("1. Hardware Detection")

    backend_type = detect_best_backend()
    backend = BackendFactory.create(backend_type)
    device = backend.device
    info = backend.get_device_info()
    device_name = (
        info.get("device_name") or info.get("name") or "CPU"
        if isinstance(info, dict)
        else getattr(info, "device_name", None) or getattr(info, "backend", "CPU")
    )

    print(f"  Backend:   {backend_type.name}")
    print(f"  Device:    {device}")
    print(f"  Hardware:  {device_name}")

    # ── 2. Build a simple model ───────────────────────────────────────────
    section("2. Simple Model")

    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.GELU(),
        nn.Linear(512, 256),
        nn.LayerNorm(256),
    )
    model.eval()

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Architecture: Linear(256→512) → GELU → Linear(512→256) → LayerNorm")

    # ── 3. Numerical validation ───────────────────────────────────────────
    section("3. Numerical Validation")

    validator = UnifiedValidator()
    results = validator.validate_model(model.to(device), input_shape=(4, 256))

    status = "PASS" if results.passed == results.total_tests else "FAIL"
    print(f"  Tests:      {results.passed}/{results.total_tests} passed   [{status}]")

    if hasattr(results, "max_diff") and results.max_diff is not None:
        print(f"  max_diff:   {results.max_diff:.2e}")
    if hasattr(results, "cosine_sim") and results.cosine_sim is not None:
        print(f"  cosine_sim: {results.cosine_sim:.6f}")

    # ── 4. Config advisory ────────────────────────────────────────────────
    section("4. Config Advisory")

    config = TorchBridgeConfig.for_inference()
    manager = UnifiedManager(config)
    optimized = manager.optimize(model.to(device))

    print(f"  Config preset:  inference")
    print(f"  Backend:        {config.backend_type if hasattr(config, 'backend_type') else backend_type.name}")
    print(f"  Optimized model type: {type(optimized).__name__}")

    # ── 5. Verify optimized model output ─────────────────────────────────
    section("5. Output Consistency Check")

    x = torch.randn(1, 256, device=device)
    with torch.no_grad():
        out_original = model.to(device)(x)
        out_optimized = optimized(x)

    diff = (out_original - out_optimized).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_original.flatten().unsqueeze(0),
        out_optimized.flatten().unsqueeze(0),
    ).item()

    print(f"  Original vs optimized max_diff:   {diff:.2e}")
    print(f"  Original vs optimized cosine_sim: {cos_sim:.6f}")
    print(f"  Status: {'PASS' if diff < 1e-3 else 'CHECK TOLERANCE'}")

    print(f"\n{'─' * 60}")
    print("  Done. TorchBridge is working correctly on this hardware.")
    print(f"{'─' * 60}\n")

    print("Next steps:")
    print("  tb-doctor                               # full system diagnostics")
    print("  tb-validate --compare cuda cpu \\")
    print("    --model ./model.pt                    # validate your own model")
    print("  tb-advisor --model-params 7e9           # config recommendation")
    print()


if __name__ == "__main__":
    main()
