# CLI Reference

TorchBridge provides command-line tools for validation, benchmarking, and diagnostics.

## Commands

All commands are available as standalone entry points (`tb-benchmark`, `tb-doctor`, etc.)
and as sub-commands of the main `torchbridge` dispatcher (`torchbridge benchmark`, etc.).
Both forms are equivalent — use whichever fits your workflow.

### `tb-benchmark`

Run performance benchmarks on a model.

```bash
tb-benchmark --model model.pt --batch-sizes 1,8,32 --output results.json
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--batch-sizes` | Comma-separated batch sizes | `1,8,32` |
| `--input-shape` | Input tensor shape | inferred |
| `--warmup` | Warmup iterations | `5` |
| `--iterations` | Benchmark iterations | `100` |
| `--output` | Save results to JSON | stdout |
| `--predefined` | Use predefined suite: `optimization`, `memory`, `throughput` | -- |
| `--quick` | Reduced iterations for fast check | false |
| `--format` | Output format: `json`, `csv` | `json` |
| `--compare-baseline` | Compare results against a baseline JSON file | -- |
| `--regression-threshold` | Regression threshold as fraction (e.g., 0.15 = 15%) | `0.15` |

**Examples:**

```bash
# Quick benchmark
tb-benchmark --model model.pt --quick

# Detailed with specific input shape
tb-benchmark --model model.pt --input-shape 1,128 --iterations 500

# Predefined benchmark suite
tb-benchmark --predefined optimization --quick

# CSV output
tb-benchmark --predefined optimization --quick --format csv --output results.csv

# Compare against baseline (fails CI if regressions exceed 15%)
tb-benchmark --predefined optimization --output current.json
tb-benchmark --predefined optimization --compare-baseline current.json --regression-threshold 0.10
```

### `tb-doctor`

System diagnostics and compatibility checking.

```bash
tb-doctor
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--verbose` | Detailed output | false |
| `--full-report` | Generate comprehensive report | false |
| `--output` | Save report to file | stdout |
| `--category` | Check specific category: `hardware`, `software`, `backends` | all |
| `--ci` | CI mode: JSON to stdout, structured exit codes (0=pass, 1=fail, 2=warn) | false |

**Examples:**

```bash
# Quick check
tb-doctor

# Full diagnostic report
tb-doctor --full-report --output system_report.json

# Hardware-specific check
tb-doctor --category hardware --verbose

# CI/CD pipeline (JSON output, structured exit codes)
tb-doctor --ci
```

**Output includes:**
- Python and PyTorch versions
- Available backends (CUDA, ROCm, XLA)
- GPU information (model, memory, compute capability)
- Driver versions
- TorchBridge version and configuration

### `tb-validate`

Cross-backend output validation — compares model outputs across two backends and reports
numerical divergence, per-layer analysis, and multi-step agentic trace drift.

```bash
tb-validate --compare cuda rocm --model ./model.pt
```

**Core flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--compare A B` | Compare backend A vs backend B (cuda, rocm, cpu, tpu, mps) | required |
| `--model` | Path to model file or HuggingFace model ID | -- |
| `--input-shape` | Input tensor shape | inferred |
| `--per-layer` | Report per-layer divergence breakdown | false |
| `--dtype` | Torch dtype: `float32`, `float16`, `bfloat16` | `float32` |
| `--model-family` | Model family for tolerance lookup: `transformer`, `cnn`, `rnn`, `diffusion`, `custom` | auto |
| `--ci` | CI mode: JSON to stdout, exits non-zero on failure | false |
| `--output` | Save report to JSON file | stdout |

**Agentic trace flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--trace` | Enable multi-step trace mode | false |
| `--steps N` | Number of autoregressive steps to trace | 10 |
| `--autoregressive` | Feed greedy token from backend A as input at each step | false |
| `--trace-output` | Save per-step trace report to JSON file | stdout |

**Observability flags:**

| Flag | Description | Default |
|------|-------------|---------|
| `--otel` | Emit validation results as OpenTelemetry spans | false |
| `--otel-endpoint` | OTLP endpoint URL (e.g. Langfuse, W&B, Honeycomb) | `OTEL_EXPORTER_OTLP_ENDPOINT` env |
| `--cert` | Generate compliance certificate (SHA256 signed pass/fail) | false |

**Examples:**

```bash
# Basic cross-backend comparison
tb-validate --compare cuda rocm --model ./model.pt

# Per-layer divergence
tb-validate --compare cuda rocm --model ./model.pt --per-layer

# 50-step agentic trace — reports first-divergence-step and amplification factor
tb-validate --compare cuda rocm --model ./model.pt --trace --steps 50 --autoregressive

# CI mode with JSON output
tb-validate --compare cuda cpu --model ./model.pt --ci --output report.json

# Compliance certificate + OTel export to Langfuse
tb-validate --compare cuda rocm --model ./model.pt --cert --otel --otel-endpoint https://cloud.langfuse.com/api/public/otel
```

## All Entry Points

TorchBridge installs these standalone commands:

| Command | Purpose |
|---------|---------|
| `torchbridge` | Main dispatcher (sub-command interface) |
| `tb-benchmark` | Performance benchmarking |
| `tb-doctor` | System diagnostics |
| `tb-validate` | Cross-backend validation |
| `tb-advisor` | Hardware configuration advisor |
| `tb-migrate` | Config migration between versions |
| `tb-quantize` | Backend-aware quantization |
| `tb-cache` | KV-cache configuration |
| `tb-speculate` | Speculative decoding configuration |
| `tb-checkpoint` | Checkpoint management |
| `tb-adapter` | Adapter training configuration |

## Configuration

TorchBridge is configured programmatically via `TorchBridgeConfig`:

```python
from torchbridge import TorchBridgeConfig, configure

config = TorchBridgeConfig.for_training()
configure(config)
```

See the [quickstart guide](../getting_started/quickstart.md) for configuration presets.

## Use Cases

### CI/CD Pipeline

```bash
# Validate system (CI mode — JSON output, structured exit codes)
tb-doctor --ci
tb-validate --ci --level quick

# Run benchmarks with regression detection
tb-benchmark --predefined optimization --quick --output results.json
tb-benchmark --predefined optimization --compare-baseline results.json
```

### Cross-Backend Validation Workflow

```bash
# 1. Validate outputs match across backends
tb-validate --compare cuda rocm --model model.pt --per-layer

# 2. Multi-step agentic trace
tb-validate --compare cuda rocm --model model.pt --trace --steps 50 --autoregressive

# 3. Get hardware configuration recommendation
tb-advisor

# 4. Benchmark
tb-benchmark --model model.pt --output results.json
```

## See Also

- [Deployment](deployment.md)
- [Backend Selection](backend-selection.md)
- [Installation](../getting_started/installation.md)
