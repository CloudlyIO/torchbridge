# CLI Reference

TorchBridge provides command-line tools for optimization, benchmarking, and diagnostics.

## Commands

### `torchbridge optimize`

Optimize a saved PyTorch model.

```bash
torchbridge optimize --model model.pt --output optimized.pt --level production
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--output`, `-o` | Output path for optimized model | auto-generated |
| `--level` | Optimization level: `basic`, `jit`, `compile`, `triton`, `production` | `compile` |
| `--hardware` | Target hardware: `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `--input-shape` | Input tensor shape (e.g., `1,3,224,224`) | inferred |
| `--benchmark` | Run performance benchmark after optimization | false |
| `--validate` | Validate optimization correctness | false |
| `--trust-source` | Allow loading untrusted model files (pickle deserialization) | false |
| `--verbose`, `-v` | Enable verbose output | false |

**Examples:**

```bash
# Compile-level optimization (default)
torchbridge optimize --model model.pt --level compile

# Production optimization with benchmarking
torchbridge optimize --model model.pt --level production --benchmark

# JIT optimization with custom input shape
torchbridge optimize --model model.pt --level jit --input-shape 1,3,224,224
```

### `torchbridge benchmark`

Run performance benchmarks on a model.

```bash
torchbridge benchmark --model model.pt --batch-sizes 1,8,32 --output results.json
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
torchbridge benchmark --model model.pt --quick

# Detailed with specific input shape
torchbridge benchmark --model model.pt --input-shape 1,128 --iterations 500

# Predefined benchmark suite
torchbridge benchmark --predefined optimization --quick

# CSV output
tb-benchmark --predefined optimization --quick --format csv --output results.csv

# Compare against baseline (fails CI if regressions exceed 15%)
tb-benchmark --predefined optimization --output current.json
tb-benchmark --predefined optimization --compare-baseline current.json --regression-threshold 0.10
```

### `torchbridge doctor`

System diagnostics and compatibility checking.

```bash
torchbridge doctor
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
torchbridge doctor

# Full diagnostic report
torchbridge doctor --full-report --output system_report.json

# Hardware-specific check
torchbridge doctor --category hardware --verbose

# CI/CD pipeline (JSON output, structured exit codes)
tb-doctor --ci
```

**Output includes:**
- Python and PyTorch versions
- Available backends (CUDA, ROCm, XLA)
- GPU information (model, memory, compute capability)
- Driver versions
- TorchBridge version and configuration

### `torchbridge export`

Export a model to portable formats.

```bash
torchbridge export --model model.pt --format onnx --output model.onnx
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--format` | Export format: `torchscript`, `onnx`, `safetensors` | `torchscript` |
| `--output` | Output path | auto |
| `--input-shape` | Sample input shape for tracing | required |
| `--opset` | ONNX opset version | `17` |

### `torchbridge profile`

Profile model performance.

```bash
torchbridge profile --model model.pt --input-shape 1,128 --output profile.json
```

### `torchbridge validate`

Run a structured validation pipeline with multiple levels.

```bash
torchbridge validate --level standard
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--level` | Validation level: `quick`, `standard`, `full`, `cloud` | `standard` |
| `--model` | Path to a specific model to validate | -- |
| `--output` | Save report to file | stdout |
| `--format` | Output format: `json`, `yaml`, `text` | `text` |
| `--ci` | CI mode: JSON to stdout, structured exit codes (0=pass, 1=fail, 2=warn) | false |
| `--verbose` | Detailed output | false |

**Levels:**

| Level | What it checks |
|-------|----------------|
| `quick` | Hardware detection + import checks |
| `standard` | Quick + model validation + export format checks |
| `full` | Standard + benchmark suite + cross-backend consistency |
| `cloud` | Run cloud validation scripts via subprocess |

**Examples:**

```bash
# Quick hardware check
torchbridge validate --level quick

# Full validation with JSON report
torchbridge validate --level full --output report.json --format json

# CI/CD pipeline
tb-validate --ci --level quick
```

### `torchbridge init`

Scaffold a new backend-agnostic PyTorch project from templates.

```bash
torchbridge init --name my_project --template training
```

**Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--name` | Project name (required) | -- |
| `--template` | Template: `training`, `inference`, `distributed`, `serving` | `training` |
| `--backend` | Backend hint: `auto`, `nvidia`, `amd`, `tpu`, `cpu` | `auto` |
| `--output-dir` | Parent directory for generated project | `.` |
| `--force` | Overwrite existing directory | false |
| `--verbose` | Detailed output | false |

**Generated files per template:**

| File | training | inference | distributed | serving |
|------|----------|-----------|-------------|---------|
| `train.py` | yes | -- | yes | -- |
| `serve.py` | -- | yes | -- | yes |
| `config.yaml` | yes | yes | yes | yes |
| `requirements.txt` | base | base | base + NCCL | base + FastAPI |
| `Dockerfile` | CPU | CPU | multi-GPU note | port 8000 |
| `README.md` | yes | yes | yes | yes |
| `.gitignore` | yes | yes | yes | yes |

**Examples:**

```bash
# Training project
tb-init --name my_model --template training

# Serving project with NVIDIA hint
tb-init --name api_server --template serving --backend nvidia

# Distributed training project (overwrites existing)
tb-init --name big_model --template distributed --force
```

## Standalone Entry Points

For CI/CD pipelines, standalone commands are available:

```bash
tb-benchmark --model model.pt --quick
tb-doctor --full-report
tb-validate --ci --level quick
tb-init --name my_project --template training
```

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

### Model Optimization Workflow

```bash
# 1. Benchmark baseline
torchbridge benchmark --model model.pt --output baseline.json

# 2. Optimize
torchbridge optimize --model model.pt --level production --output optimized.pt

# 3. Benchmark optimized
torchbridge benchmark --model optimized.pt --output optimized.json

# 4. Export
torchbridge export --model optimized.pt --format onnx --input-shape 1,128
```

## See Also

- [Deployment](deployment.md)
- [Backend Selection](backend-selection.md)
- [Installation](../getting_started/installation.md)
