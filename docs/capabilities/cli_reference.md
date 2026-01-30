# 🛠️ CLI Reference (v0.4.18)

**TorchBridge** provides professional command-line tools for PyTorch optimization, benchmarking, and system diagnostics using the unified architecture with TPU support.

## Overview

```bash
torchbridge --help     # Main CLI help
torchbridge --version  # Show version information
```

## Commands

### 🔧 optimize - Model Optimization

Optimize PyTorch models for production deployment with various optimization levels.

```bash
torchbridge optimize [OPTIONS]
```

**Basic Usage:**
```bash
# Optimize a saved model
torchbridge optimize --model model.pt --level production

# Optimize with specific hardware target
torchbridge optimize --model resnet50 --level compile --hardware cuda

# Optimize with validation and benchmarking
torchbridge optimize --model model.pt --level triton --validate --benchmark
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--model` | `str` | **Required.** Path to model file (.pt/.pth) or model name (resnet50, bert) |
| `--level` | `choice` | Optimization level: `basic`, `jit`, `compile`, `triton`, `production` (default: `compile`) |
| `--output`, `-o` | `str` | Output path for optimized model (auto-generated if not specified) |
| `--input-shape` | `str` | Input tensor shape, e.g., "1,3,224,224" for batch,channels,height,width |
| `--hardware` | `choice` | Target hardware: `auto`, `cpu`, `cuda`, `mps` (default: `auto`) |
| `--benchmark` | `flag` | Run performance benchmark after optimization |
| `--validate` | `flag` | Validate optimization correctness |
| `--verbose`, `-v` | `flag` | Enable verbose output |

**Optimization Levels:**

- **`basic`**: PyTorch native optimizations (cuDNN, cuBLAS)
- **`jit`**: TorchScript JIT compilation
- **`compile`**: torch.compile with Inductor backend
- **`triton`**: Triton kernel optimizations (GPU required)
- **`production`**: Full optimization stack for deployment

**Examples:**
```bash
# Quick optimization for inference
torchbridge optimize --model model.pt --level compile

# Production optimization with validation
torchbridge optimize \
    --model transformer_model.pt \
    --level production \
    --input-shape "8,512,768" \
    --validate \
    --benchmark \
    --verbose

# Optimize for specific GPU architecture
torchbridge optimize \
    --model resnet50 \
    --level triton \
    --hardware cuda \
    --output optimized_resnet50.pt
```

### 📊 benchmark - Performance Benchmarking

Run comprehensive performance benchmarks for models and optimizations.

```bash
torchbridge benchmark [OPTIONS]
```

**Basic Usage:**
```bash
# Quick benchmark suite
torchbridge benchmark --predefined optimization --quick

# Benchmark a specific model
torchbridge benchmark --model resnet50 --type model

# Compare optimization levels
torchbridge benchmark --model bert --type compare --levels basic,compile,triton
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--model` | `str` | Model to benchmark (file path or predefined name) |
| `--type` | `choice` | Benchmark type: `model`, `compare`, `regression`, `stress` (default: `model`) |
| `--levels` | `str` | Optimization levels to compare (comma-separated, default: `basic,compile`) |
| `--batch-sizes` | `str` | Batch sizes for stress testing (comma-separated, default: `1,8,16`) |
| `--input-shape` | `str` | Input tensor shape (e.g., "1,3,224,224") |
| `--predefined` | `choice` | Run predefined benchmark suite: `transformers`, `vision`, `optimization` |
| `--quick` | `flag` | Quick benchmark (fewer runs for faster results) |
| `--warmup` | `int` | Number of warmup runs (default: 10) |
| `--runs` | `int` | Number of benchmark runs (default: 100, 20 if --quick) |
| `--output`, `-o` | `str` | Output file for results (JSON format) |
| `--verbose`, `-v` | `flag` | Enable verbose output |

**Benchmark Types:**

- **`model`**: Single model performance benchmark
- **`compare`**: Compare multiple optimization levels
- **`regression`**: Performance regression testing
- **`stress`**: Stress test with various batch sizes

**Predefined Suites:**

- **`optimization`**: Core TorchBridge optimization benchmarks
- **`transformers`**: Transformer model benchmarks (BERT-like)
- **`vision`**: Computer vision model benchmarks (ResNet-like)

**Examples:**
```bash
# Comprehensive optimization benchmarks
torchbridge benchmark --predefined optimization --output results.json

# Compare optimization levels for a model
torchbridge benchmark \
    --model transformer.pt \
    --type compare \
    --levels basic,jit,compile,triton \
    --verbose

# Stress test with multiple batch sizes
torchbridge benchmark \
    --model resnet50 \
    --type stress \
    --batch-sizes "1,4,8,16,32" \
    --quick

# Vision model benchmarks
torchbridge benchmark --predefined vision --output vision_results.json
```

### 🩺 doctor - System Diagnostics

Diagnose system compatibility and optimization readiness.

```bash
torchbridge doctor [OPTIONS]
```

**Basic Usage:**
```bash
# Quick system check
torchbridge doctor

# Comprehensive diagnostics
torchbridge doctor --full-report

# Check specific category
torchbridge doctor --category hardware
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--category` | `choice` | Focus on specific category: `basic`, `hardware`, `optimization`, `advanced` |
| `--full-report` | `flag` | Run comprehensive diagnostics (all categories) |
| `--fix` | `flag` | Attempt to fix detected issues (where possible) |
| `--output`, `-o` | `str` | Save diagnostic report to file (JSON format) |
| `--verbose`, `-v` | `flag` | Enable verbose output |

**Diagnostic Categories:**

- **`basic`**: Python, PyTorch, and basic dependencies
- **`hardware`**: GPU detection and capabilities
- **`optimization`**: Optimization framework availability
- **`advanced`**: Advanced features (Triton, CUDA kernels)

**Examples:**
```bash
# Quick health check
torchbridge doctor

# Detailed hardware analysis
torchbridge doctor --category hardware --verbose

# Full system report with fix attempts
torchbridge doctor --full-report --fix --output system_report.json

# Check optimization capabilities
torchbridge doctor --category optimization
```

## Standalone Commands

Individual commands are also available as separate entry points:

```bash
tb-optimize --model model.pt --level production
tb-benchmark --predefined optimization --quick
tb-doctor --full-report
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `KERNELPYTORCH_DEVICE` | Default device for operations | `auto` |
| `KERNELPYTORCH_CACHE_DIR` | Cache directory for models | `~/.cache/torchbridge` |
| `KERNELPYTORCH_LOG_LEVEL` | Logging level | `INFO` |

## Configuration File

Create `~/.torchbridge/config.yaml` for default settings:

```yaml
# Default optimization settings
optimization:
  default_level: compile
  default_hardware: auto
  validate_by_default: true

# Benchmark settings
benchmark:
  default_runs: 100
  default_warmup: 10
  save_results: true

# Output settings
output:
  verbose: false
  save_logs: true
  log_directory: "~/.torchbridge/logs"
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments / help displayed |
| 130 | Interrupted by user (Ctrl+C) |

## Examples by Use Case

### Research & Development
```bash
# System setup check
torchbridge doctor --full-report

# Quick model optimization
torchbridge optimize --model research_model.pt --level compile --validate

# Performance analysis
torchbridge benchmark --model research_model.pt --type compare --verbose
```

### Production Deployment
```bash
# Production optimization with validation
torchbridge optimize \
    --model production_model.pt \
    --level production \
    --validate \
    --benchmark \
    --output optimized_production_model.pt

# Deployment readiness check
torchbridge doctor --category optimization --verbose
```

### CI/CD Integration
```bash
# Automated testing in CI
torchbridge benchmark --type regression --quick --output ci_results.json

# System validation
torchbridge doctor --output ci_system_report.json
if [ $? -ne 0 ]; then echo "System check failed"; exit 1; fi
```

### Performance Monitoring
```bash
# Regular performance monitoring
torchbridge benchmark \
    --predefined optimization \
    --output "performance_$(date +%Y%m%d).json"

# Compare with baseline
torchbridge benchmark \
    --type regression \
    --output current_performance.json
```

## Tips and Best Practices

1. **Always run `doctor`** before deployment to ensure optimal configuration
2. **Use `--validate`** when optimizing critical models to ensure correctness
3. **Save benchmark results** with `--output` for performance tracking
4. **Use `--verbose`** during development for detailed diagnostics
5. **Test with `--quick`** during development, full runs for production validation

---

**Need help with a specific command?** Use `torchbridge <command> --help` for detailed usage information.