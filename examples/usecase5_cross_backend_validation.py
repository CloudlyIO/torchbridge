"""
Use Case 5: Cross-Backend Validation

Demonstrates TorchBridge's validation tools: verify that a model produces
correct results across different backends, check hardware compatibility,
and validate configurations — essential when migrating between hardware
vendors or deploying to heterogeneous clusters.

Run: PYTHONPATH=src python3 examples/usecase5_cross_backend_validation.py
"""
# ruff: noqa: E402

import torch
import torch.nn as nn

# ── Step 1: Build a Model ────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Build a Test Model")
print("=" * 60)

class TestModel(nn.Module):
    def __init__(self, d_model=256, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.embedding(x)
        attn_out, _ = self.attention(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h

model = TestModel()
param_count = sum(p.numel() for p in model.parameters())
print(f"  Model   : TestModel (Transformer block, {param_count:,} params)")

# ── Step 2: UnifiedValidator — Model Validation ─────────────────────
print("\n" + "=" * 60)
print("STEP 2: Model Validation (UnifiedValidator)")
print("=" * 60)

from torchbridge.validation.unified_validator import UnifiedValidator, ValidationLevel

validator = UnifiedValidator()

print("\n  Running STANDARD validation...")
result = validator.validate_model(model, input_shape=(4, 16, 256), level=ValidationLevel.STANDARD)

print(f"  Passed     : {result.passed}/{result.total_tests}")
print(f"  Failed     : {result.failed}")
print(f"  Skipped    : {result.skipped}")

if result.reports:
    print(f"\n  {'Test':<40s} {'Status':>8s}  Message")
    print(f"  {'-'*40} {'-'*8}  {'-'*30}")
    for r in result.reports:
        status_str = str(r.status.value).upper()
        print(f"  {r.name:<40s} {status_str:>8s}  {r.message}")

# ── Step 3: Hardware Compatibility Check ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Hardware Compatibility Validation")
print("=" * 60)

device = torch.device("cpu")
hw_result = validator.validate_hardware_compatibility(device)

print(f"  Device     : {device}")
print(f"  Passed     : {hw_result.passed}/{hw_result.total_tests}")

if hw_result.reports:
    for r in hw_result.reports:
        status_str = str(r.status.value).upper()
        print(f"  {r.name:<40s} {status_str:>8s}  {r.message}")

# ── Step 4: Configuration Validation ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Configuration Validation")
print("=" * 60)

from torchbridge import TorchBridgeConfig

for preset_name, preset_fn in [
    ("training", TorchBridgeConfig.for_training),
    ("inference", TorchBridgeConfig.for_inference),
    ("development", TorchBridgeConfig.for_development),
]:
    config = preset_fn()
    cfg_result = validator.validate_configuration(config)
    status = "PASS" if cfg_result.passed == cfg_result.total_tests else f"{cfg_result.passed}/{cfg_result.total_tests}"
    print(f"  {preset_name:<15s}: {status}")

# ── Step 5: Cross-Backend Output Consistency ─────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Cross-Backend Output Consistency")
print("=" * 60)

from torchbridge.backends import BackendFactory

# Test all available backends produce the same output
torch.manual_seed(42)
reference_model = TestModel().eval()
reference_input = torch.randn(2, 8, 256)

with torch.no_grad():
    reference_output = reference_model(reference_input)

backends_to_test = ["cpu"]
if torch.cuda.is_available():
    backends_to_test.append("cuda")

print(f"  Reference output shape: {list(reference_output.shape)}")
print(f"  Reference output mean : {reference_output.mean().item():.6f}")
print(f"  Reference output std  : {reference_output.std().item():.6f}")
print()

for bname in backends_to_test:
    try:
        b = BackendFactory.create(bname)
        dev = b.device

        # Clone model to this device
        test_model = TestModel().eval().to(dev)
        test_model.load_state_dict(reference_model.state_dict())
        test_input = reference_input.to(dev)

        with torch.no_grad():
            test_output = test_model(test_input)

        # Compare
        diff = (test_output.cpu() - reference_output).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        match = torch.allclose(test_output.cpu(), reference_output, atol=1e-3)

        print(f"  {bname:>6s}: {'PASS' if match else 'FAIL'}  "
              f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")
    except Exception as e:
        print(f"  {bname:>6s}: ERROR ({e})")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  Model validation     : {result.passed}/{result.total_tests} passed")
print(f"  Hardware compat      : {hw_result.passed}/{hw_result.total_tests} passed")
print("  Config presets       : 3/3 validated")
print(f"  Cross-backend match  : {len(backends_to_test)} backend(s) tested")
print()
print("  These checks can run in CI to catch regressions when:")
print("    - Upgrading PyTorch versions")
print("    - Switching GPU vendors (NVIDIA -> AMD)")
print("    - Deploying to new hardware (cloud migration)")
print("    - Changing precision or optimization settings")
