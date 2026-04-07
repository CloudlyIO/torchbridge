# Speculative Decoding & Structured Output

TorchBridge provides backend-aware speculative decoding method selection via a compatibility matrix, and structured output format definitions. For generation loop execution use the native `model.generate()` APIs directly.

## Speculative Decoding Methods

| Method | Description | Requirements |
|--------|-------------|-------------|
| `draft_model` | Standard draft-verify with smaller assistant model | Draft model |
| `eagle` | EAGLE with custom CUDA kernels | NVIDIA Hopper+ |
| `layer_skip` | Self-speculative by skipping later layers | None |
| `medusa` | Multi-head speculative with tree attention | NVIDIA Ampere+ |
| `prompt_lookup` | N-gram matching from prompt tokens | None (universal) |

## Compatibility Matrix

| Backend | Architecture | Optimal | All Supported |
|---------|-------------|---------|---------------|
| NVIDIA | Blackwell/Hopper | Draft Model | draft_model, prompt_lookup |
| NVIDIA | Ampere/Ada | Draft Model | draft_model, prompt_lookup |
| AMD | CDNA3/CDNA4 | Draft Model | draft_model, prompt_lookup |
| Trainium | TRN2/TRN3 | Prompt Lookup | prompt_lookup |
| TPU | v5e/v7 | Prompt Lookup | prompt_lookup |
| CPU | — | Prompt Lookup | prompt_lookup |

## Query the Compatibility Matrix

```python
from torchbridge.inference import SpeculationCompatibilityMatrix, SpeculativeMethod
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture, AMDArchitecture

# Get optimal method for hardware
optimal = SpeculationCompatibilityMatrix.get_optimal_method(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# → SpeculativeMethod.DRAFT_MODEL

# Get all supported methods
supported = SpeculationCompatibilityMatrix.get_supported_methods(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)

# Check if a method is supported
is_supported = SpeculationCompatibilityMatrix.is_method_supported(
    SpeculativeMethod.EAGLE, HardwareBackend.AMD, AMDArchitecture.CDNA3
)
# → False

# Get methods compatible with model.generate() kwargs
generate_methods = SpeculationCompatibilityMatrix.get_generate_compatible_methods(
    HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
)

# Get fallback chain
chain = SpeculationCompatibilityMatrix.get_fallback_chain(
    SpeculativeMethod.EAGLE, HardwareBackend.CUDA, NVIDIAArchitecture.AMPERE
)
```

## Structured Output Formats

TorchBridge defines an `OutputFormat` enum for consistent cross-framework configuration.
Pass the selected format to your grammar engine (xgrammar, outlines, etc.):

```python
from torchbridge.inference import OutputFormat

# Select format
fmt = OutputFormat.JSON_SCHEMA   # JSON schema-constrained output
fmt = OutputFormat.REGEX          # Regex-constrained output
fmt = OutputFormat.JSON           # Unconstrained JSON
fmt = OutputFormat.TEXT           # Plain text (default)
```

> **Note:** TorchBridge provides the `OutputFormat` enum for naming consistency. Use
> `xgrammar` or `outlines` for the actual logits-processor and grammar engine
> implementation — TorchBridge does not provide a generation loop.

## CLI

```bash
# Show optimal method for detected hardware
torchbridge speculate

# Show full compatibility matrix
torchbridge speculate --show-matrix

# Check specific method on specific backend
torchbridge speculate --backend nvidia --method eagle

# JSON output for CI
torchbridge speculate --ci
```
