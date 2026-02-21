# Trainium Backend

TorchBridge Trainium backend provides unified access to AWS Trainium and Inferentia2 chips via the Neuron SDK (torch_neuronx) through the `BaseBackend` interface.

## Supported Hardware

| Generation | Chip | HBM | Key Features |
|-----------|------|-----|--------------|
| **Trn3** | NeuronCore v4 | 144GB HBM3e | FP8/FP4 hardware support, highest throughput |
| **Trn2** | NeuronCore v3 | 96GB HBM | FP8 hardware support, large model training |
| **Trn1** | NeuronCore v1 | 32GB HBM | cFP8, cost-efficient training |
| **Inf2** | NeuronCore v1 | 32GB HBM | Inference-optimized |

## Quick Start

```python
from torchbridge.backends.trainium import TrainiumBackend

backend = TrainiumBackend()
print(backend.get_device_info())

model = backend.prepare_model(your_model)
```

## Core Components

- **`TrainiumBackend`** -- main backend implementing `BaseBackend`
- **`TrainiumAdapter`** -- Neuron-specific optimization strategies
- **`NeuronCompiler`** -- Neuron graph compilation with caching
- **`TrainiumMemoryManager`** -- HBM monitoring and management

## Configuration

```python
from torchbridge.core.config import TrainiumConfig

config = TrainiumConfig(
    cache_max_size=100,
    compilation_timeout_seconds=600,
    memory_fraction=0.9,
    precision="bfloat16",
    enable_graph_caching=True,
)
```

### Configuration Modes

```python
# Development: fast iteration
config = TrainiumConfig(precision="float32", mixed_precision=False)

# Training: balanced
config = TrainiumConfig(precision="bfloat16", gradient_checkpointing=True)

# Inference: maximum throughput
config = TrainiumConfig(precision="bfloat16", cache_max_size=200)
```

## Precision Support

| Dtype | Chip Support | Notes |
|-------|-------------|-------|
| BF16 | All | Default and recommended |
| FP32 | All | Supported but slower |
| cFP8 | Trn1+ | Configurable FP8 |
| FP8 | Trn2+ | FP8 E4M3 via Neuron SDK |
| MXFP4 | Trn3 only | Microscaling FP4 |

Trainium is optimized for BF16. TorchBridge auto-converts models to BF16 by default.

```python
# Disable auto-conversion if needed
config = TrainiumConfig(precision="float32", mixed_precision=False)
```

## Instance Types

| Instance | Chip | NeuronCores | HBM | Use Case |
|----------|------|------------|-----|----------|
| trn1.2xlarge | Trn1 | 2 | 32GB | Dev/test |
| trn1.32xlarge | Trn1 | 32 | 512GB | Large training |
| trn2.48xlarge | Trn2 | 16 | 1.5TB | Production training |

## Neuron SDK Requirements

The Trainium backend requires the AWS Neuron SDK:

```bash
# Install on Trainium instance (Ubuntu 24.04)
pip install torch-neuronx neuronx-cc torch-xla

# Verify
python3 -c "import torch_neuronx; print(torch_neuronx.__version__)"
```

### Docker

Use the provided Trainium Dockerfile:

```bash
docker build -f docker/Dockerfile.trainium -t torchbridge:trainium .
```

## Neuron Compilation

Trainium operations require Neuron graph compilation (neuronx-cc). First iterations are slower while graphs are compiled and cached:

```python
from torchbridge.backends.trainium import NeuronCompiler

compiler = NeuronCompiler(config)
stats = compiler.get_compilation_stats()
```

### Graph Caching

Enable graph caching to avoid recompilation:

```python
config = TrainiumConfig(enable_graph_caching=True)
# Cache is stored at $NEURON_COMPILE_CACHE_URL (default: /tmp/neuron_cache)
```

## Memory Management

```python
backend = TrainiumBackend(config)

stats = backend.get_memory_stats()
print(f"Device: {stats['device']}")

# Reduce memory usage
config = TrainiumConfig(memory_fraction=0.8)
```

## Distributed Training

For multi-chip Trainium training (tensor parallelism, pipeline parallelism):

```python
config = TrainiumConfig(
    tensor_parallel_size=2,
    pipeline_parallel_size=4,
)
```

## Error Handling

- `DeviceNotAvailableError` -- Neuron SDK not installed or no Trainium detected (from base exceptions)
- `OutOfMemoryError` -- HBM exhausted (from base exceptions)
- `CompilationError` -- Neuron graph compilation failure or timeout (from base exceptions)

## Troubleshooting

### torch_neuronx not found

```bash
pip install torch-neuronx --extra-index-url https://pip.repos.neuron.amazonaws.com
```

### Compilation timeout

Increase the timeout in configuration:

```python
config = TrainiumConfig(compilation_timeout_seconds=1200)
```

### NEURON_RT_VISIBLE_CORES not set

Ensure the Neuron runtime is configured:

```bash
export NEURON_RT_VISIBLE_CORES=0-1  # Use first 2 NeuronCores
```

## See Also

- [Backends Overview](overview.md)
- [Hardware Setup](../guides/hardware-setup.md)
- [Hardware Matrix](../reference/hardware-matrix.md)
