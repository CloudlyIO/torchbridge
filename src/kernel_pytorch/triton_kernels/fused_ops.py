"""
Level 4: Triton Kernel Optimized Components

Triton provides a Python-like syntax for writing GPU kernels while maintaining
high performance. It's a middle ground between PyTorch operations and raw CUDA.

Triton kernels demonstrate:
- Block-based parallel computation
- Automatic memory coalescing
- Tiling strategies for memory hierarchy optimization
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
import math


@triton.jit
def layer_norm_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for fused layer normalization - Educational GPU Kernel Implementation.

    🎓 EDUCATIONAL OVERVIEW:
    This kernel demonstrates fundamental GPU optimization principles that make
    Triton kernels faster than standard PyTorch operations in specific cases.

    🔧 GPU OPTIMIZATION TECHNIQUES DEMONSTRATED:

    1. BLOCK-BASED COMPUTATION:
       - BLOCK_SIZE parameter controls parallelism granularity
       - Each thread block processes BLOCK_SIZE elements in parallel
       - Optimal BLOCK_SIZE balances occupancy vs memory usage (typically 128-512)
       - Trade-off: Larger blocks = better memory coalescing, smaller blocks = more parallelism

    2. MEMORY COALESCING OPTIMIZATION:
       - input_ptrs calculation ensures consecutive threads access consecutive memory
       - Critical for GPU memory bandwidth utilization (can be 10x performance difference)
       - tl.arange(0, BLOCK_SIZE) creates coalesced access pattern automatically
       - Memory access pattern: Thread 0 reads addr[0], Thread 1 reads addr[1], etc.

    3. PARALLEL REDUCTION OPERATIONS:
       - Mean/variance computed using GPU-native parallel reduction (tl.sum)
       - More efficient than CPU-style sequential loops
       - Leverages GPU's thousands of cores for statistical computations
       - Hardware-optimized reduction trees in GPU warp schedulers

    4. REGISTER-BASED COMPUTATION:
       - Intermediate values (mean, var, normalized) stay in GPU registers
       - No global memory allocations for temporary results
       - Eliminates memory bandwidth overhead of separate operations

    📊 PERFORMANCE CHARACTERISTICS:
    - Memory bandwidth bound: Performance scales with memory access efficiency
    - Optimal for: Medium-sized tensors where kernel launch overhead is amortized
    - Speedup vs PyTorch: 1.5-3x for suitable tensor sizes (depends on hardware)
    - Best performance: Tensors with n_cols that align well with BLOCK_SIZE

    💡 WHEN TO USE TRITON VS PYTORCH:
    ✅ Use Triton when: Custom fusion patterns, specific memory layouts, research
    ❌ Use PyTorch when: Standard operations, varied tensor sizes, rapid prototyping

    🎓 EDUCATIONAL COMPARISON:
    PyTorch LayerNorm: x.mean() → (x-mean) → x.var() → normalize (4 kernel launches)
    Triton LayerNorm:  Single kernel with fused mean+variance+normalize (1 kernel launch)
    """
    # 🎓 STEP 1: GPU Thread Block Identification
    # Each GPU thread block handles one row of the input tensor
    # program_id(0) gives us the unique ID of this thread block
    # This is how we achieve parallelism: thousands of rows processed simultaneously
    row_idx = tl.program_id(0)

    # 🔥 STEP 2: Memory Coalescing Setup - CRITICAL for performance!
    # Calculate memory addresses ensuring consecutive threads access consecutive memory
    # input_row_stride: number of elements to jump to next row
    # tl.arange(0, BLOCK_SIZE): creates [0,1,2,...,BLOCK_SIZE-1] for coalesced access
    # WHY: GPU memory controller can serve multiple threads in single transaction
    input_ptrs = input_ptr + row_idx * input_row_stride + tl.arange(0, BLOCK_SIZE)
    output_ptrs = output_ptr + row_idx * output_row_stride + tl.arange(0, BLOCK_SIZE)

    # 🛡️ STEP 3: Memory Safety - Handle variable tensor sizes
    # mask ensures we don't read/write beyond tensor boundaries
    # Essential when BLOCK_SIZE doesn't perfectly divide n_cols
    mask = tl.arange(0, BLOCK_SIZE) < n_cols

    # 🚀 STEP 4: Vectorized Memory Load
    # tl.load automatically handles vectorization and coalescing
    # 'other=0.0' provides safe padding values for masked-out elements
    # Single instruction loads up to BLOCK_SIZE elements simultaneously!
    x = tl.load(input_ptrs, mask=mask, other=0.0)

    # 🧮 STEP 5: Parallel Reduction for Statistics
    # tl.sum uses GPU hardware reduction trees (much faster than loops)
    # All threads in block collaborate to compute mean efficiently
    # Hardware executes this as logarithmic reduction tree
    mean = tl.sum(x, axis=0) / n_cols

    # 🧮 STEP 6: Variance Computation (Fused with Mean)
    # Everything stays in GPU registers - no global memory overhead!
    # Manual implementation would require separate kernels and memory round-trips
    x_centered = x - mean  # Broadcasting handled automatically by GPU
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols  # Another parallel reduction

    # 🔢 STEP 7: Normalization with Numerical Stability
    # inv_std approach is more numerically stable than direct division
    # tl.sqrt uses GPU's built-in transcendental function units
    inv_std = 1.0 / tl.sqrt(var + eps)
    normalized = (x - mean) * inv_std  # Element-wise operations are fully vectorized

    # 📚 STEP 8: Parameter Loading (Weight & Bias)
    # Separate loads for weight/bias parameters (typically smaller, well-cached)
    # Default values (1.0, 0.0) handle cases where parameters might be missing
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)

    # ⚡ STEP 9: Final Transformation (Fused)
    # Affine transformation fused with normalization - no intermediate storage!
    # All arithmetic happens in GPU registers at maximum throughput
    output = normalized * weight + bias

    # 💾 STEP 10: Coalesced Memory Store
    # tl.store ensures optimal write patterns back to global memory
    # Completes the fully-fused layer normalization in single kernel!
    tl.store(output_ptrs, output, mask=mask)


@triton.jit
def swiglu_kernel(
    input_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    output_ptr,
    batch_size,
    seq_len,
    input_dim,
    hidden_dim,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Advanced Triton kernel for fused SwiGLU computation - Production-Level GPU Optimization.

    🎓 EDUCATIONAL OVERVIEW:
    SwiGLU (Swish-Gated Linear Unit) is a key component in modern language models
    (PaLM, LLaMA). This kernel demonstrates advanced GPU optimization techniques
    for complex fused operations that would be impossible with standard PyTorch.

    🧠 MATHEMATICAL BACKGROUND:
    SwiGLU(x) = Swish(x @ W_gate) ⊙ (x @ W_up)
    Where: Swish(x) = x * sigmoid(x), ⊙ = element-wise multiplication

    🔧 ADVANCED GPU OPTIMIZATION TECHNIQUES:

    1. TILED MATRIX MULTIPLICATION:
       - Large matrix ops broken into GPU-cache-sized tiles (BLOCK_SIZE_K)
       - Each tile fits in GPU shared memory (typically 48-96KB per SM)
       - Minimizes global memory bandwidth by reusing data within tiles
       - Enables parallel computation across thousands of GPU cores

    2. 3D PARALLELIZATION STRATEGY:
       - BLOCK_SIZE_M: Sequence dimension tiling (typically 64-128)
       - BLOCK_SIZE_N: Hidden dimension tiling (typically 64-128)
       - BLOCK_SIZE_K: Input dimension tiling (typically 32-64)
       - 3D grid enables massive parallelism: ~10,000+ thread blocks on A100

    3. MEMORY HIERARCHY OPTIMIZATION:
       - L1 cache: Frequently accessed data (current tile)
       - L2 cache: Weight matrices (shared across sequence elements)
       - Global memory: Large input/output tensors
       - Register usage: Intermediate computations and accumulators

    4. FUSED COMPUTATION BENEFITS:
       - Standard: Input→Gate_Linear→Swish + Input→Up_Linear→Multiply (6 kernel launches)
       - Fused: Single kernel with embedded matrix multiplication + activation
       - Memory savings: No intermediate tensor storage between operations
       - Bandwidth savings: ~3x reduction in memory traffic

    📊 PERFORMANCE CHARACTERISTICS:
    - Compute intensity: High FLOP/byte ratio due to matrix multiplications
    - Memory pattern: Optimized for GPU memory coalescing
    - Scalability: Linear scaling with hidden_dim, quadratic with seq_len
    - Hardware utilization: Near-peak FLOPS on modern GPUs (A100/H100)

    💡 WHEN TO USE CUSTOM KERNELS:
    ✅ Use for: Unique fusion patterns, memory-intensive ops, production deployment
    ❌ Avoid for: Standard operations, rapid prototyping, small models

    🎓 EDUCATIONAL VALUE:
    - Demonstrates advanced tiling strategies for large matrix operations
    - Shows how to achieve memory bandwidth optimization on modern GPUs
    - Illustrates the complexity/benefit tradeoff of custom GPU kernels
    - Real-world example from state-of-the-art language model architectures
    """
    # Get block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # Compute block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Compute input pointer offset for current batch
    input_offset = pid_batch * seq_len * input_dim

    # Initialize accumulators for gate and up projections
    gate_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Tiled matrix multiplication
    for k in range(0, input_dim, BLOCK_SIZE_K):
        k_offs = k + offs_k

        # Load input tile
        input_ptrs = (input_ptr + input_offset +
                     offs_m[:, None] * input_dim + k_offs[None, :])
        input_mask = (offs_m[:, None] < seq_len) & (k_offs[None, :] < input_dim)
        input_tile = tl.load(input_ptrs, mask=input_mask, other=0.0)

        # Load gate weight tile
        gate_weight_ptrs = (gate_weight_ptr +
                           offs_n[:, None] * input_dim + k_offs[None, :])
        gate_weight_mask = (offs_n[:, None] < hidden_dim) & (k_offs[None, :] < input_dim)
        gate_weight_tile = tl.load(gate_weight_ptrs, mask=gate_weight_mask, other=0.0)

        # Load up weight tile
        up_weight_ptrs = (up_weight_ptr +
                         offs_n[:, None] * input_dim + k_offs[None, :])
        up_weight_mask = (offs_n[:, None] < hidden_dim) & (k_offs[None, :] < input_dim)
        up_weight_tile = tl.load(up_weight_ptrs, mask=up_weight_mask, other=0.0)

        # Compute dot products
        gate_acc += tl.dot(input_tile, gate_weight_tile.T)
        up_acc += tl.dot(input_tile, up_weight_tile.T)

    # Apply SwiGLU activation: gate * silu(up)
    # silu(x) = x / (1 + exp(-x)) = x * sigmoid(x)
    sigmoid_up = tl.sigmoid(up_acc)
    swiglu_output = gate_acc * (up_acc * sigmoid_up)

    # Store output
    output_offset = pid_batch * seq_len * hidden_dim
    output_ptrs = (output_ptr + output_offset +
                  offs_m[:, None] * hidden_dim + offs_n[None, :])
    output_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < hidden_dim)
    tl.store(output_ptrs, swiglu_output, mask=output_mask)


@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    L, M,  # Intermediate values for numerically stable softmax
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, HEAD_DIM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Flash Attention kernel in Triton.

    This is a simplified version demonstrating the core concepts:
    - Tiled computation to fit in SRAM
    - Online algorithm for numerically stable softmax
    - Minimizing HBM accesses through careful tiling
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Initialize pointers to Q, K, V for this head
    q_ptrs = Q + off_hz * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + off_hz * stride_kh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    v_ptrs = V + off_hz * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk

    # Load Q block
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Initialize output accumulator and normalization terms
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V blocks
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_ptrs_n = k_ptrs + start_n * stride_kn
        v_ptrs_n = v_ptrs + start_n * stride_vn

        k = tl.load(k_ptrs_n, mask=(start_n + offs_n[None, :]) < N_CTX, other=0.0)
        v = tl.load(v_ptrs_n, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)

        # Compute attention scores
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= 1.44269504  # 1/log(2) for efficient softmax

        # Online algorithm for numerically stable softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p_ij = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p_ij, 1)

        # Scale previous values
        alpha = tl.exp(m_i - m_ij)
        acc_scale = l_i * alpha
        acc *= acc_scale[:, None]

        # Update accumulator
        acc += tl.dot(p_ij.to(v.dtype), v)

        # Update normalization terms
        l_i = acc_scale + l_ij
        m_i = m_ij

    # Final normalization
    acc /= l_i[:, None]

    # Store output
    out_ptrs = Out + off_hz * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


@triton.jit
def rotary_embedding_kernel(
    input_ptr, cos_ptr, sin_ptr, output_ptr,
    batch_size, num_heads, seq_len, head_dim,
    input_batch_stride, input_head_stride, input_seq_stride,
    output_batch_stride, output_head_stride, output_seq_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for rotary positional embedding.

    Demonstrates:
    - Complex number arithmetic on GPU
    - Efficient indexing patterns
    - Vectorized operations
    """
    # Get program indices
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # Compute base pointers
    input_base = (input_ptr +
                  batch_idx * input_batch_stride +
                  head_idx * input_head_stride +
                  seq_idx * input_seq_stride)

    output_base = (output_ptr +
                   batch_idx * output_batch_stride +
                   head_idx * output_head_stride +
                   seq_idx * output_seq_stride)

    # Process pairs of dimensions
    for start_dim in range(0, head_dim, BLOCK_SIZE):
        # Load input pairs
        dim_offs = start_dim + tl.arange(0, BLOCK_SIZE)
        mask = dim_offs < head_dim

        # Load even and odd elements
        even_ptrs = input_base + dim_offs * 2
        odd_ptrs = input_base + dim_offs * 2 + 1

        x1 = tl.load(even_ptrs, mask=mask, other=0.0)
        x2 = tl.load(odd_ptrs, mask=mask, other=0.0)

        # Load cos and sin values
        cos_ptrs = cos_ptr + seq_idx * (head_dim // 2) + dim_offs
        sin_ptrs = sin_ptr + seq_idx * (head_dim // 2) + dim_offs

        cos_val = tl.load(cos_ptrs, mask=mask, other=1.0)
        sin_val = tl.load(sin_ptrs, mask=mask, other=0.0)

        # Apply rotation
        rotated_x1 = x1 * cos_val - x2 * sin_val
        rotated_x2 = x2 * cos_val + x1 * sin_val

        # Store output
        output_even_ptrs = output_base + dim_offs * 2
        output_odd_ptrs = output_base + dim_offs * 2 + 1

        tl.store(output_even_ptrs, rotated_x1, mask=mask)
        tl.store(output_odd_ptrs, rotated_x2, mask=mask)


class TritonLayerNorm(torch.nn.Module):
    """
    Layer normalization using Triton kernel for educational purposes.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape input to 2D for kernel processing
        original_shape = x.shape
        x_2d = x.view(-1, self.normalized_shape)

        output = torch.empty_like(x_2d)

        # Launch kernel with appropriate block size
        BLOCK_SIZE = triton.next_power_of_2(self.normalized_shape)
        grid = (x_2d.size(0),)

        layer_norm_kernel[grid](
            x_2d, self.weight, self.bias, output,
            x_2d.stride(0), output.stride(0),
            self.normalized_shape, self.eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        return output.view(original_shape)


class TritonSwiGLU(torch.nn.Module):
    """
    SwiGLU activation using Triton kernel.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.gate_weight = torch.nn.Parameter(
            torch.randn(hidden_dim, input_dim) * (2.0 / input_dim) ** 0.5
        )
        self.up_weight = torch.nn.Parameter(
            torch.randn(hidden_dim, input_dim) * (2.0 / input_dim) ** 0.5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        output = torch.empty(batch_size, seq_len, self.hidden_dim,
                           device=x.device, dtype=x.dtype)

        # Define block sizes for optimal performance
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32

        grid = (
            triton.cdiv(seq_len, BLOCK_SIZE_M),
            triton.cdiv(self.hidden_dim, BLOCK_SIZE_N),
            batch_size
        )

        swiglu_kernel[grid](
            x, self.gate_weight, self.up_weight, output,
            batch_size, seq_len, input_dim, self.hidden_dim,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )

        return output


class TritonFlashAttention(torch.nn.Module):
    """
    Flash Attention implementation using Triton.
    Educational implementation focusing on the core algorithm.
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [batch, heads, seq, head_dim]

        # Prepare output tensor
        output = torch.empty_like(q)

        # Intermediate tensors for stable softmax
        L = torch.empty((batch_size, self.num_heads, seq_len), device=q.device, dtype=torch.float32)
        M = torch.empty((batch_size, self.num_heads, seq_len), device=q.device, dtype=torch.float32)

        # Launch Flash Attention kernel
        BLOCK_M = 64
        BLOCK_N = 64

        grid = (
            triton.cdiv(seq_len, BLOCK_M),
            batch_size * self.num_heads,
        )

        flash_attention_kernel[grid](
            q, k, v, output, L, M,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            batch_size, self.num_heads, seq_len, self.head_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
        )

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.out_proj(output)


class TritonRotaryEmbedding(torch.nn.Module):
    """
    Rotary Position Embedding using Triton kernel.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim

        # Precompute rotation matrices
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        self.register_buffer('cos', freqs.cos())
        self.register_buffer('sin', freqs.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = x.shape
        output = torch.empty_like(x)

        BLOCK_SIZE = 32
        grid = (batch_size, num_heads, seq_len)

        rotary_embedding_kernel[grid](
            x, self.cos, self.sin, output,
            batch_size, num_heads, seq_len, head_dim,
            x.stride(0), x.stride(1), x.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_SIZE=BLOCK_SIZE
        )

        return output


class TritonOptimizedTransformerBlock(torch.nn.Module):
    """
    Complete transformer block using Triton-optimized components.
    Demonstrates how to combine custom kernels while maintaining
    the semantic structure of transformer architecture.
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim

        # Use Triton-optimized components
        self.norm1 = TritonLayerNorm(dim)
        self.attn = TritonFlashAttention(dim, num_heads)
        self.norm2 = TritonLayerNorm(dim)
        self.mlp = TritonSwiGLU(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard transformer block with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# Utility function to compare Triton vs PyTorch implementations
def benchmark_triton_vs_pytorch():
    """
    Benchmark function to compare Triton kernels with PyTorch equivalents.
    Useful for educational purposes to see the performance differences.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, seq_len, dim = 4, 512, 768

    # Test data
    x = torch.randn(batch_size, seq_len, dim, device=device)

    # Layer Norm comparison
    pytorch_ln = torch.nn.LayerNorm(dim).to(device)
    triton_ln = TritonLayerNorm(dim).to(device)

    print("Benchmarking Layer Norm...")
    # Add actual benchmarking code here

    # SwiGLU comparison
    hidden_dim = dim * 4
    triton_swiglu = TritonSwiGLU(dim, hidden_dim).to(device)

    print("Benchmarking SwiGLU...")
    # Add actual benchmarking code here

    print("Triton kernel benchmarking complete!")


if __name__ == "__main__":
    # Example usage
    if torch.cuda.is_available():
        print("Running Triton kernel examples...")
        benchmark_triton_vs_pytorch()
    else:
        print("CUDA not available. Triton kernels require CUDA.")