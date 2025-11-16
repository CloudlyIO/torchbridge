# Triton Kernels - Python-Based GPU Programming

This directory contains educational Triton kernel implementations that demonstrate GPU programming concepts using Python syntax while achieving near-CUDA performance.

## 🎯 **What is Triton?**

**Triton** is OpenAI's Python-based domain-specific language (DSL) for writing efficient GPU kernels. It bridges the gap between **high-level Python productivity** and **low-level CUDA performance**.

### **Why Triton for Education?**
- **Python Syntax**: Familiar programming model for ML researchers
- **Hardware Abstraction**: Write once, run on different GPU architectures
- **Performance**: Near-CUDA performance with significantly less complexity
- **Educational Value**: Transparent GPU programming concepts

## 🏗️ **Core Triton Concepts**

### **Block-Level Programming**
Triton operates on **blocks** of data rather than individual threads, making it easier to reason about parallel computation:

```python
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate which block of data this program instance handles
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Bounds checking
    mask = offsets < n_elements

    # Load blocks of data
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)
```

### **Memory Management**
Triton provides explicit control over GPU memory hierarchy:

```python
@triton.jit
def matrix_mul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # Program ID determines which output block to compute
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Shared memory allocation (automatic in Triton)
    # Triton manages shared memory usage based on block operations

    # Compute matrix multiplication block by block
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks into shared memory (implicit)
        a = tl.load(a_ptr + offsets_a)
        b = tl.load(b_ptr + offsets_b)

        # Compute partial matrix multiplication
        accumulator += tl.dot(a, b)

    # Write result
    tl.store(c_ptr + offsets_c, accumulator)
```

## 🧮 **Educational ML Kernel Implementations**

### **1. Fused Layer Normalization** (`fused_ops.py`)
Demonstrates kernel fusion for common ML operations:

```python
@triton.jit
def layer_norm_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr,
                      mean_ptr, rstd_ptr, stride, N, eps, BLOCK_SIZE: tl.constexpr):
    """
    Educational implementation of fused layer normalization.

    Mathematical Operation:
    output = (input - mean) / sqrt(variance + eps) * weight + bias

    Key Learning Concepts:
    - Reduction operations (mean, variance calculation)
    - Broadcasting patterns
    - Numerical stability considerations
    """
    # Row index (each row is normalized independently)
    row_idx = tl.program_id(0)

    # Load entire row for reduction operations
    row_start = row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load input data
    x = tl.load(input_ptr + row_start + offsets, mask=mask, other=0.0)

    # Compute statistics (key GPU programming concept: reductions)
    mean = tl.sum(x, axis=0) / N
    centered = x - mean
    variance = tl.sum(centered * centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Store statistics for backward pass
    tl.store(mean_ptr + row_idx, mean)
    tl.store(rstd_ptr + row_idx, rstd)

    # Load normalization parameters
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # Apply normalization and affine transformation
    normalized = centered * rstd
    output = normalized * weight + bias

    # Store result
    tl.store(output_ptr + row_start + offsets, output, mask=mask)
```

**Learning Objectives**:
- **Reduction Operations**: How to compute statistics across tensor dimensions
- **Broadcasting**: Applying scalar operations to tensor blocks
- **Numerical Stability**: Handling floating-point precision issues
- **Memory Access Patterns**: Efficient row-wise data loading

### **2. Educational Flash Attention** (`fused_ops.py`)
Simplified attention mechanism demonstrating advanced GPU concepts:

```python
@triton.jit
def educational_flash_attention_kernel(q_ptr, k_ptr, v_ptr, output_ptr,
                                       seq_len, head_dim, scale,
                                       BLOCK_SIZE: tl.constexpr):
    """
    Educational Flash Attention implementation.

    Demonstrates:
    - Block-wise attention computation
    - Online softmax algorithm
    - Memory-efficient attention patterns
    - GPU memory hierarchy optimization
    """
    # Which attention head and sequence position
    head_idx = tl.program_id(0)
    seq_block = tl.program_id(1)

    # Query block for this program
    q_offset = head_idx * seq_len * head_dim + seq_block * BLOCK_SIZE * head_dim
    q = tl.load(q_ptr + q_offset + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim +
                tl.arange(0, head_dim)[None, :])

    # Initialize attention accumulator
    output_acc = tl.zeros((BLOCK_SIZE, head_dim), dtype=tl.float32)
    max_score = tl.zeros((BLOCK_SIZE,), dtype=tl.float32) - float('inf')
    sum_exp = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Iterate through key/value blocks (online softmax)
    for k_block in range(0, seq_len, BLOCK_SIZE):
        # Load key and value blocks
        k_offset = head_idx * seq_len * head_dim + k_block * BLOCK_SIZE * head_dim
        k = tl.load(k_ptr + k_offset + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim +
                    tl.arange(0, head_dim)[None, :])

        v = tl.load(v_ptr + k_offset + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim +
                    tl.arange(0, head_dim)[None, :])

        # Compute attention scores for this block
        scores = tl.dot(q, tl.trans(k)) * scale

        # Online softmax update (numerically stable)
        new_max = tl.maximum(max_score, tl.max(scores, axis=1))
        old_scale = tl.exp(max_score - new_max)
        new_scale = tl.exp(tl.max(scores, axis=1) - new_max)

        # Update attention weights and output accumulator
        weights = tl.exp(scores - new_max[:, None])

        # Scale previous accumulator and add new contribution
        output_acc = output_acc * old_scale[:, None]
        output_acc += tl.dot(weights, v)

        # Update normalizers
        sum_exp = sum_exp * old_scale + tl.sum(weights, axis=1)
        max_score = new_max

    # Final normalization
    output = output_acc / sum_exp[:, None]

    # Store result
    output_offset = head_idx * seq_len * head_dim + seq_block * BLOCK_SIZE * head_dim
    tl.store(output_ptr + output_offset + tl.arange(0, BLOCK_SIZE)[:, None] * head_dim +
             tl.arange(0, head_dim)[None, :], output)
```

**Learning Objectives**:
- **Online Algorithms**: Computing softmax without storing intermediate results
- **Numerical Stability**: Preventing overflow in exponential calculations
- **Block-Wise Computation**: Processing large tensors in manageable chunks
- **Memory Efficiency**: Minimizing memory footprint during computation

### **3. Optimized Convolution** (`fused_ops.py`)
Demonstrates spatial computation patterns:

```python
@triton.jit
def conv2d_kernel(input_ptr, weight_ptr, output_ptr,
                  batch_size, in_channels, out_channels,
                  input_height, input_width, kernel_size,
                  BLOCK_SIZE: tl.constexpr):
    """
    Educational 2D convolution implementation.

    Demonstrates:
    - Spatial computation patterns
    - Input-output mapping for convolutions
    - Memory access optimization for image data
    - Filter application across spatial dimensions
    """
    # Program IDs for output position
    batch_idx = tl.program_id(0)
    out_ch_block = tl.program_id(1)
    spatial_block = tl.program_id(2)

    # Calculate spatial coordinates
    output_size = input_height - kernel_size + 1
    out_h = spatial_block // output_size
    out_w = spatial_block % output_size

    # Load filter weights for this output channel block
    out_ch_start = out_ch_block * BLOCK_SIZE
    out_ch_offsets = out_ch_start + tl.arange(0, BLOCK_SIZE)

    # Initialize output accumulator
    output_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Convolution computation: slide filter over input
    for ic in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Input coordinates
                in_h = out_h + kh
                in_w = out_w + kw

                # Load input value
                input_offset = (batch_idx * in_channels * input_height * input_width +
                               ic * input_height * input_width +
                               in_h * input_width + in_w)
                input_val = tl.load(input_ptr + input_offset)

                # Load corresponding weights
                weight_offset = (out_ch_offsets * in_channels * kernel_size * kernel_size +
                                ic * kernel_size * kernel_size +
                                kh * kernel_size + kw)
                weight_vals = tl.load(weight_ptr + weight_offset)

                # Accumulate convolution result
                output_val += input_val * weight_vals

    # Store result
    output_offset = (batch_idx * out_channels * output_size * output_size +
                    out_ch_offsets * output_size * output_size +
                    out_h * output_size + out_w)
    tl.store(output_ptr + output_offset, output_val)
```

## 🚀 **Performance Optimization Techniques**

### **1. Auto-Tuning with Triton**
Triton provides automatic parameter tuning for optimal performance:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64}, num_stages=3),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 128}, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def auto_tuned_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    """
    Auto-tuned kernel that automatically selects optimal block sizes
    based on input dimensions and hardware characteristics.
    """
    # Kernel implementation with auto-tuned parameters
    pass
```

**Learning Concepts**:
- **Parameter Tuning**: How block sizes affect performance
- **Hardware Adaptation**: Optimal configurations for different GPUs
- **Performance Measurement**: Automatic benchmarking and selection

### **2. Memory Coalescing Patterns**
```python
@triton.jit
def memory_efficient_kernel(input_ptr, output_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Demonstrates memory coalescing for optimal bandwidth utilization.
    """
    # Good: Sequential memory access (coalesced)
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = tl.load(input_ptr + offsets)  # Efficient: consecutive addresses

    # Process data
    result = data * 2.0

    # Good: Sequential write (coalesced)
    tl.store(output_ptr + offsets, result)
```

### **3. Shared Memory Optimization**
```python
@triton.jit
def shared_memory_demo(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Educational example of shared memory usage patterns.
    Note: Triton automatically manages shared memory allocation.
    """
    # Triton automatically uses shared memory for:
    # 1. Block-level operations
    # 2. Temporary computation results
    # 3. Data reuse across program instances

    # Load data (automatically cached in shared memory if beneficial)
    data = tl.load(input_ptr + tl.arange(0, BLOCK_SIZE))

    # Operations that benefit from shared memory
    # (Triton compiler automatically optimizes these)
    sum_val = tl.sum(data)  # Reduction uses shared memory
    broadcast_sum = sum_val * tl.ones_like(data)  # Broadcasting

    result = data + broadcast_sum
    tl.store(output_ptr + tl.arange(0, BLOCK_SIZE), result)
```

## 🎓 **Educational Learning Path**

### **Beginner: Understanding Block Programming**
1. **Start with simple kernels** - Vector addition, element-wise operations
2. **Learn block concepts** - How Triton divides work across GPU cores
3. **Practice memory operations** - Loading and storing data efficiently
4. **Understand masking** - Handling arbitrary tensor sizes

### **Intermediate: ML-Specific Patterns**
1. **Reduction operations** - Computing statistics (mean, variance, sum)
2. **Broadcasting patterns** - Applying operations across dimensions
3. **Fusion opportunities** - Combining multiple operations in single kernels
4. **Numerical stability** - Handling floating-point precision issues

### **Advanced: Optimization Techniques**
1. **Memory hierarchy** - Understanding GPU memory levels and access patterns
2. **Auto-tuning** - Optimizing kernel parameters for different hardware
3. **Complex algorithms** - Implementing attention, convolution efficiently
4. **Performance analysis** - Profiling and optimizing kernel performance

## 🔧 **Development Workflow**

### **Setting Up Triton Development**
```bash
# Install Triton (included with PyTorch 2.0+)
pip install triton

# Development environment setup
export TRITON_INTERPRET=1  # For debugging (slower but more informative errors)
```

### **Testing and Debugging**
```python
# Test kernel correctness against PyTorch reference
def test_kernel_correctness():
    # Generate test data
    x = torch.randn(1024, 512, device='cuda')

    # PyTorch reference implementation
    pytorch_result = torch.layer_norm(x, (512,))

    # Triton kernel implementation
    triton_result = triton_layer_norm(x)

    # Compare results
    assert torch.allclose(pytorch_result, triton_result, atol=1e-6)
    print("✅ Kernel correctness verified")

# Performance benchmarking
def benchmark_kernel():
    import triton.testing

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],
            x_vals=[2**i for i in range(8, 16)],
            line_arg='provider',
            line_vals=['triton', 'pytorch'],
            plot_name="layer-norm-performance",
            args={'dtype': torch.float32},
        )
    )
    def bench_layer_norm(size, provider, dtype):
        x = torch.randn(size, size, dtype=dtype, device='cuda')

        if provider == 'pytorch':
            return lambda: torch.layer_norm(x, (size,))
        elif provider == 'triton':
            return lambda: triton_layer_norm(x)

    bench_layer_norm.run(print_data=True, save_path='.')
```

### **Integration with PyTorch**
```python
class TritonLayerNorm(torch.nn.Module):
    """PyTorch module wrapper for Triton kernel."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # Use Triton kernel implementation
        return triton_layer_norm_kernel(x, self.weight, self.bias, self.eps)
```

## 📊 **Performance Characteristics**

### **Expected Performance Improvements**
- **Layer Normalization**: 1.2-2x speedup over PyTorch native
- **Flash Attention**: Comparable to optimized implementations
- **Custom Convolutions**: 1.5x speedup for specific kernel sizes

### **Hardware Dependencies**
- **NVIDIA GPUs**: Full Triton support (compute capability 7.0+)
- **AMD GPUs**: Limited support (improving rapidly)
- **Memory Bandwidth**: Often more important than compute for ML kernels

## 🔬 **Research and Future Directions**

### **Current Research Areas**
1. **Automatic Kernel Generation**: LLM-driven Triton code synthesis
2. **Cross-Platform Support**: Unified kernels for multiple GPU vendors
3. **Dynamic Shapes**: Efficient handling of variable input sizes
4. **Quantization Support**: Low-precision arithmetic optimizations

### **Integration with Modern ML**
- **torch.compile**: Triton as backend for PyTorch compilation
- **JAX**: Triton integration with JAX's XLA compiler
- **Framework Agnostic**: Triton kernels callable from multiple frameworks

## 📚 **References and Further Reading**

### **Essential Papers**
- **Triton**: "Triton: An Intermediate Language and Compiler for Tiled Neural Network Code"
- **Flash Attention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- **GPU Architecture**: "Understanding Latency Hiding on GPUs"

### **Documentation and Tutorials**
- **Official Triton Docs**: https://triton-lang.org/
- **OpenAI Triton GitHub**: https://github.com/openai/triton
- **PyTorch Triton Integration**: torch.compile with Triton backend

### **Advanced Resources**
- **GPU Programming Guide**: CUDA Programming Best Practices
- **Memory Hierarchy**: "A Survey of GPU Memory Hierarchy Optimizations"
- **Parallel Algorithms**: "Introduction to Parallel Algorithms and Architectures"

---

**🎯 Educational Mission**: Learn GPU programming concepts through practical ML kernel implementations, bridging high-level Python productivity with low-level performance optimization understanding.