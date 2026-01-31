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
| `--output` | Output path | `optimized_<input>` |
| `--level` | Optimization level: `basic`, `jit`, `compile`, `triton`, `production` | `basic` |
| `--backend` | Force backend: `cuda`, `rocm`, `xpu`, `tpu`, `cpu` | auto |
| `--dtype` | Target dtype: `fp32`, `fp16`, `bf16` | auto |
| `--verbose` | Enable verbose output | false |

**Examples:**

```bash
# Basic JIT optimization
torchbridge optimize --model model.pt --level jit

# Production with specific backend
torchbridge optimize --model model.pt --level production --backend cuda

# BF16 conversion
torchbridge optimize --model model.pt --level compile --dtype bf16
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

**Examples:**

```bash
# Quick benchmark
torchbridge benchmark --model model.pt --quick

# Detailed with specific input shape
torchbridge benchmark --model model.pt --input-shape 1,128 --iterations 500

# Predefined benchmark suite
torchbridge benchmark --predefined optimization --quick
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

**Examples:**

```bash
# Quick check
torchbridge doctor

# Full diagnostic report
torchbridge doctor --full-report --output system_report.json

# Hardware-specific check
torchbridge doctor --category hardware --verbose
```

**Output includes:**
- Python and PyTorch versions
- Available backends (CUDA, ROCm, IPEX, XLA)
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

## Standalone Entry Points

For CI/CD pipelines, standalone commands are available:

```bash
tb-optimize --model model.pt --level production
tb-benchmark --model model.pt --quick
tb-doctor --full-report
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TORCHBRIDGE_BACKEND` | Force backend selection | auto |
| `TORCHBRIDGE_LOG_LEVEL` | Logging level | `INFO` |
| `TORCHBRIDGE_LOG_FORMAT` | Log format: `text`, `json` | `text` |
| `TORCHBRIDGE_DEFAULT_DTYPE` | Default dtype | auto |

## Configuration File

Create `torchbridge.yaml` in your project root:

```yaml
backend: auto
optimization_level: O2
precision:
  enabled: true
  dtype: bfloat16
logging:
  level: INFO
  format: json
```

## Use Cases

### CI/CD Pipeline

```bash
# Validate system
torchbridge doctor --category backends
# Run benchmarks
torchbridge benchmark --predefined optimization --quick
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
- [Installation](../getting-started/installation.md)
