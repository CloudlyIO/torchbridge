# Testing and Validation Guide

TorchBridge provides a Python testing toolkit for embedding cross-backend validation
into your own test suites. This is useful when you want to:

- Validate that a model produces consistent outputs across hardware in CI
- Identify which layers diverge when moving from NVIDIA to AMD
- Set per-model-family numerical tolerances for pass/fail thresholds

The same machinery powers `tb-validate` under the hood.

## DivergenceTracer

`DivergenceTracer` registers forward hooks to capture per-layer outputs, then diffs
them between two runs (typically two different hardware backends).

```python
import torch
import torch.nn as nn
from torchbridge.testing import DivergenceTracer

model = nn.Sequential(
    nn.Linear(128, 256),
    nn.GELU(),
    nn.Linear(256, 64),
)
x = torch.randn(4, 128)

# Run on backend A (CPU reference)
ref = DivergenceTracer(model)
with ref:
    with torch.no_grad():
        model(x)

# Run on backend B (same or different device)
test = DivergenceTracer(model)
with test:
    with torch.no_grad():
        model(x)

# Compare
divergences = test.compare_with(ref)
for d in divergences:
    print(f"{d.layer_name}: max_diff={d.max_diff:.2e}  cos={d.cosine_sim:.6f}")
```

**`compare_with` takes a second `DivergenceTracer`**, not a model or tensor. Both
tracers must have been run before comparison.

### LayerDivergence fields

Each item returned by `compare_with()` is a `LayerDivergence` with:

| Field | Type | Description |
|-------|------|-------------|
| `layer_name` | `str` | Module name (e.g., `"fc1"`, `"transformer.h.0.attn"`) |
| `max_diff` | `float` | Max absolute difference across all elements |
| `mean_diff` | `float` | Mean absolute difference |
| `cosine_sim` | `float` | Cosine similarity between layer outputs |
| `output_shape` | `tuple` | Shape of the captured tensor |
| `exceeds_threshold` | `bool` | `True` if `max_diff > atol` |

### Large models — limit layers

On very large models (ResNets, large transformers), hooking every layer uses
significant memory. Use `max_layers` to limit:

```python
tracer = DivergenceTracer(model, max_layers=32)
```

## @cross_backend

The `@cross_backend` decorator runs a test function on every available backend,
collects all failures, and reports them together.

```python
import torch
import torch.nn as nn
from torchbridge.testing import cross_backend

@cross_backend(min_backends=1)
def test_linear_output_shape(backend):
    model = nn.Linear(8, 4).to(backend.device)
    out = model(torch.randn(2, 8).to(backend.device))
    assert out.shape == (2, 4)

test_linear_output_shape()
```

The decorated function receives a `backend` argument — a `BaseBackend` instance with
a `.device` property.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_backends` | `1` | Skip the test if fewer backends are available |
| `skip_if_unavailable` | `True` | Use `pytest.skip` when `min_backends` not met |

Works as a plain function call or inside pytest — the decorator is pytest-aware.

## ToleranceDB

`ToleranceDB` stores empirically measured numerical tolerances keyed by
`(backend, dtype)` with an optional `model_family` override. These are the same
tolerances used by `tb-validate` to determine pass/fail.

```python
from torchbridge.testing import ToleranceDB

db = ToleranceDB()

# Look up tolerance for a backend/dtype pair
tol = db.get("cuda", "float32")
print(f"atol={tol.atol}  rtol={tol.rtol}  source={tol.source}")

# With model family (more specific — overrides generic entry if available)
tol = db.get("cuda", "float16", model_family="decoder-small")

# Check if an entry was measured on real hardware vs. derived/fallback
db.is_measured("cuda", "float32")  # True → real measured data
db.is_measured("rocm", "float16")  # may be False → derived or fallback

# Inspect available entries
print(db.all_backends())   # ['cpu', 'cuda', 'mps', 'rocm', 'trainium', 'xla']
print(db.families())       # ['decoder-large', 'decoder-medium', ...]
```

**`ToleranceEntry` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `atol` | `float` | Absolute tolerance |
| `rtol` | `float` | Relative tolerance |
| `source` | `str` | `"measured"`, `"derived"`, or `"fallback"` |

### Adding entries

If you have hardware not yet in the DB, add measured tolerances and contribute them
back (see [CONTRIBUTING.md](../../CONTRIBUTING.md)):

```python
from torchbridge.testing import ToleranceDB

db = ToleranceDB()

# Register a base (backend, dtype) entry:
db.register("mi350x", "float16", atol=1e-3, rtol=1e-3)

# Register a family-specific entry (supports source + notes metadata):
db.register_family(
    "decoder-small", "mi350x", "float16",
    atol=1e-3, rtol=1e-3,
    source="measured",
    notes="measured on MI350X; Qwen3-0.6B; 2026-04-07",
)
```

## Writing validation tests

A complete example — validate model consistency in a pytest test:

```python
import torch
import torch.nn as nn
import pytest
from torchbridge.testing import DivergenceTracer, ToleranceDB

def make_model():
    return nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 64))

def test_cpu_consistency():
    model = make_model().eval()
    x = torch.randn(4, 128)

    ref = DivergenceTracer(model)
    test = DivergenceTracer(model)

    with ref:
        with torch.no_grad():
            model(x)
    with test:
        with torch.no_grad():
            model(x)

    db = ToleranceDB()
    tol = db.get("cpu", "float32")

    divergences = test.compare_with(ref)
    failures = [d for d in divergences if d.max_diff > tol.atol]
    assert not failures, f"Divergent layers: {[(d.layer_name, d.max_diff) for d in failures]}"
```

## See Also

- [CLI Reference](cli.md) — `tb-validate` runs this same machinery from the command line
- [Quickstart](../getting_started/quickstart.md) — `UnifiedValidator` for single-backend smoke tests
- [Compatibility Matrix](../reference/compatibility-matrix.md) — tolerance DB source data
