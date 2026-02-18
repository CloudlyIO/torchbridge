# Speculative Decoding & Structured Output

TorchBridge v0.5.26 adds backend-aware speculative decoding, grammar-guided structured output, and disaggregated serving phase detection.

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
| NVIDIA | Blackwell/Hopper | EAGLE | eagle, draft_model, layer_skip, medusa, prompt_lookup |
| NVIDIA | Ampere/Ada | Draft Model | draft_model, layer_skip, medusa, prompt_lookup |
| AMD | CDNA3/CDNA4 | Draft Model | draft_model, layer_skip, prompt_lookup |
| Trainium | TRN2/TRN3 | Layer Skip | layer_skip, prompt_lookup |
| TPU | v6e/v7 | Layer Skip | layer_skip, prompt_lookup |
| CPU | — | Prompt Lookup | prompt_lookup |

## Quick Start

### Auto-Select Method

```python
from torchbridge.inference import SpeculationEngine, SpeculationConfig

# Auto-selects optimal method for detected hardware
engine = SpeculationEngine()
kwargs = engine.get_generation_kwargs()

# Pass to model.generate()
outputs = model.generate(input_ids, **kwargs)
```

### Explicit Method

```python
from torchbridge.inference import SpeculationEngine, SpeculationConfig, SpeculativeMethod

config = SpeculationConfig(
    method=SpeculativeMethod.DRAFT_MODEL,
    draft_model_name="Qwen/Qwen3-0.6B",
    num_speculative_tokens=5,
)
engine = SpeculationEngine(
    config=config,
    backend=HardwareBackend.CUDA,
    architecture=NVIDIAArchitecture.AMPERE,
)
```

### Query the Compatibility Matrix

```python
from torchbridge.inference import SpeculationCompatibilityMatrix, SpeculativeMethod
from torchbridge.core.config import HardwareBackend, NVIDIAArchitecture

# Get optimal method
optimal = SpeculationCompatibilityMatrix.get_optimal_method(
    HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
# → SpeculativeMethod.EAGLE

# Check if a method is supported
supported = SpeculationCompatibilityMatrix.is_method_supported(
    SpeculativeMethod.EAGLE, HardwareBackend.AMD, AMDArchitecture.CDNA3
)
# → False
```

## Structured Output

### JSON Constrained Generation

```python
from torchbridge.inference import StructuredOutputProcessor, OutputFormat

# Constrain output to valid JSON matching a schema
schema = {"type": "object", "required": ["name", "age"]}
processor = StructuredOutputProcessor(
    format=OutputFormat.JSON_SCHEMA,
    schema=schema,
)

# Get logits processors for model.generate()
logits_processors = processor.get_logits_processor(tokenizer)

# Validate output
valid = processor.validate_output('{"name": "Alice", "age": 30}')
```

### Regex Constrained Generation

```python
processor = StructuredOutputProcessor(
    format=OutputFormat.REGEX,
    pattern=r"\d{4}-\d{2}-\d{2}",  # Date format
)
```

> **Note:** Structured output requires `xgrammar` (`pip install xgrammar`). Without it, constraints are disabled and a warning is logged.

## Phase Detection

```python
from torchbridge.inference import PhaseDetector, PhaseType

# Detect current inference phase
phase = PhaseDetector.detect_phase(
    prompt_tokens=512,
    generated_tokens=10,
)
# → PhaseType.PREFILL (ratio < 0.1)

# Get hardware recommendations
profile = PhaseDetector.get_hardware_profile(
    phase, HardwareBackend.CUDA, NVIDIAArchitecture.HOPPER
)
print(profile.is_compute_bound)  # True for prefill
print(profile.recommended_hardware)
```

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

## Batch Size Gating

Speculative decoding loses efficiency at high batch sizes. The engine auto-disables speculation when `batch_size > max_batch_size_for_speculation` (default: 8).

```python
engine.should_speculate(batch_size=1)   # True
engine.should_speculate(batch_size=16)  # False
```
