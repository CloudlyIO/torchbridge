# Efficient Attention Guide

> **Version**: v0.4.23 | **Status**: Production Ready | **Last Updated**: January 2026

This guide covers the efficient attention implementations in TorchBridge, including memory-efficient, sparse, and sliced attention mechanisms for handling long sequences and reducing memory usage.

## Overview

TorchBridge provides several attention variants optimized for different use cases:

| Attention Type | Memory Complexity | Best For |
|----------------|-------------------|----------|
| Standard | O(N²) | Short sequences (<512) |
| Sliced | O(N × S) | ViT models, inference |
| Block Sparse | O(N × B) | Long sequences, training |
| Strided Sparse | O(N × (W + N/S)) | Very long sequences |
| Chunked | O(Q × K) | Memory-constrained GPUs |
| Long Sequence | O(N × W) + O(N/S) | Documents, code, DNA |

Where:
- N = sequence length
- S = slice size
- B = block size
- W = window size
- Q, K = chunk sizes

## ViT Attention Slicing

### When to Use

- Vision Transformer inference
- Large image inputs (high resolution)
- Memory-constrained environments
- Batch processing multiple images

### Usage

```python
from torchbridge.models.vision.vit import (
    SlicedMultiheadAttention,
    SlicedAttentionWrapper,
)

# Create sliced attention layer
sliced_attn = SlicedMultiheadAttention(
    embed_dim=768,
    num_heads=12,
    slice_size=64,  # Process 64 queries at a time
)

# Or wrap existing attention
import torch.nn as nn
standard_attn = nn.MultiheadAttention(embed_dim=768, num_heads=12)
sliced_attn = SlicedMultiheadAttention.from_pretrained(standard_attn, slice_size=64)

# Forward pass
output, _ = sliced_attn(query, key, value)
```

### Memory Savings

For a ViT-Large (seq_len=577 patches):
- Standard: ~1.3 GB for attention matrix
- Sliced (s=64): ~145 MB (9x reduction)

### Apply to Existing Model

```python
from torchbridge.models.vision.vit import ViTOptimizer, VisionOptimizationConfig

config = VisionOptimizationConfig(
    model_type=VisionModelType.VIT,
    optimization_level=OptimizationLevel.MAXIMUM,
)
optimizer = ViTOptimizer(config)

# Apply attention slicing to model
optimizer.apply_attention_slicing(model, slice_size=64)
```

## Sparse Attention

### Block Sparse Attention (BigBird-style)

Combines local, global, and random attention patterns.

```python
from torchbridge.attention.implementations.sparse import BlockSparseAttention
from torchbridge.attention.core.config import AttentionConfig

config = AttentionConfig(
    embed_dim=768,
    num_heads=12,
    max_sequence_length=4096,
)

# Create block sparse attention
attn = BlockSparseAttention(
    config,
    block_size=64,           # Attention computed in 64x64 blocks
    num_random_blocks=3,     # Random blocks for long-range
    num_global_blocks=1,     # First block attends everywhere
)

# Use in model
output = attn(x)  # x: [batch, seq, embed]
```

### Strided Sparse Attention (Sparse Transformer-style)

Local window + strided global pattern.

```python
from torchbridge.attention.implementations.sparse import StridedSparseAttention

attn = StridedSparseAttention(
    config,
    local_window=256,  # Attend to 256 neighbors
    stride=256,        # Also attend every 256th position
)
```

### Dynamic Sparse Attention

Learns which positions to attend to.

```python
from torchbridge.attention.implementations.sparse import DynamicSparseAttention

attn = DynamicSparseAttention(config)
# Automatically learns sparsity patterns during training
```

## Memory-Efficient Attention

### Chunked Attention

Processes queries in chunks to reduce peak memory.

```python
from torchbridge.attention.implementations.memory_efficient import (
    MemoryEfficientAttention,
)

attn = MemoryEfficientAttention(
    config,
    chunk_size=256,  # Process 256 queries at a time
)

# Enable gradient checkpointing for training
attn.enable_gradient_checkpointing()
```

### Double-Chunked Attention

For extremely long sequences, chunk both Q and KV.

```python
from torchbridge.attention.implementations.memory_efficient import ChunkedAttention

attn = ChunkedAttention(
    config,
    query_chunk_size=256,
    kv_chunk_size=256,
)
```

### Long Sequence Attention

Combines local window + global strided for very long sequences.

```python
from torchbridge.attention.implementations.memory_efficient import (
    LongSequenceAttention,
)

attn = LongSequenceAttention(
    config,
    window_size=512,     # Local attention window
    global_stride=256,   # Global attention every 256 positions
    chunk_size=256,      # Process in chunks
)
```

## Choosing the Right Attention

### Decision Tree

```
Is sequence length > 4096?
├─ Yes: Use LongSequenceAttention or ChunkedAttention
└─ No: Is this a ViT model?
       ├─ Yes: Use SlicedMultiheadAttention
       └─ No: Is training with limited GPU memory?
              ├─ Yes: Use MemoryEfficientAttention with checkpointing
              └─ No: Is sparse pattern acceptable?
                     ├─ Yes: Use BlockSparseAttention
                     └─ No: Use standard attention
```

### Performance Comparison

Typical results on NVIDIA A100 (batch=4, embed=768, heads=12):

| Attention Type | Seq=512 | Seq=1024 | Seq=2048 |
|----------------|---------|----------|----------|
| Standard | 850 s/s | 420 s/s | OOM |
| Sliced (s=64) | 720 s/s | 380 s/s | 180 s/s |
| Block Sparse | 680 s/s | 450 s/s | 320 s/s |
| Chunked | 650 s/s | 400 s/s | 250 s/s |

(s/s = samples per second)

## Pipeline Parallel Scheduling

### InterleavedScheduler

For distributed training with pipeline parallelism.

```python
from torchbridge.models.distributed.pipeline_parallel import (
    InterleavedScheduler,
    PipelineStage,
    PipelineParallelConfig,
)

config = PipelineParallelConfig(
    num_stages=4,
    num_micro_batches=8,
    stage_id=0,
)

# Create scheduler
scheduler = InterleavedScheduler(stages, config)

# Forward pass
outputs = scheduler.run_forward(micro_batches)

# Compute loss, then backward
scheduler.run_backward(gradients)
```

### 1F1B Schedule Benefits

The 1F1B (one-forward-one-backward) interleaved schedule:
- Reduces memory by ~4x vs GPipe (all-forward-then-backward)
- Keeps GPU utilization high during steady state
- Enables training larger models on limited hardware

## Benchmarking

Run the attention efficiency benchmarks:

```bash
python benchmarks/attention_efficiency.py \
    --seq-lengths 256 512 1024 2048 \
    --batch-size 4 \
    --output results.json
```

## Best Practices

1. **Start with standard attention** for short sequences (<512)
2. **Use sliced attention** for ViT inference - minimal accuracy impact
3. **Enable gradient checkpointing** during training for memory savings
4. **Profile your specific workload** - results vary by hardware and model
5. **Consider accuracy tradeoffs** - sparse patterns may affect model quality

## Troubleshooting

### Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing
3. Use chunked attention with smaller chunk size
4. Use sparse attention patterns

### Slow Performance

1. Increase chunk/slice size (trades memory for speed)
2. Use larger blocks for sparse attention
3. Profile to identify bottleneck
4. Consider flash attention if available

### Numerical Issues

1. Check for NaN values in output
2. Verify attention mask format (True = masked)
3. Use `torch.nan_to_num()` for edge cases
4. Reduce learning rate if training is unstable

## API Reference

### AttentionConfig

```python
@dataclass
class AttentionConfig:
    embed_dim: int           # Model dimension
    num_heads: int           # Number of attention heads
    max_sequence_length: int # Maximum sequence length
    head_dim: int = None     # Head dimension (default: embed_dim / num_heads)
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    causal: bool = False     # Use causal masking
```

### Common Methods

All attention classes share:
- `forward(query, key=None, value=None, attention_mask=None)` - Compute attention
- `get_attention_stats()` - Get layer statistics
- `reset_parameters()` - Reset to initial weights

Memory-efficient classes also provide:
- `enable_gradient_checkpointing()` / `disable_gradient_checkpointing()`

## See Also

- [Performance Benchmarks](../capabilities/performance.md)
- [Hardware Backend Selection](../backends/README.md)
- [Vision Model Guide](vision_model_guide.md)
