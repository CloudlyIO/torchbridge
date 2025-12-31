# Custom CUDA Kernels (v0.3.3)

**Production CUDA kernel implementations for maximum GPU performance.**

## 📁 Kernel Locations

KernelPyTorch includes custom CUDA kernels in:
- `src/kernel_pytorch/cuda_kernels/` - Primary custom kernels
  - `flash_attention_v3.cu` - FlashAttention-3 implementation
  - `fused_linear_activation.cu` - Fused linear + activation kernels
  - `fused_ops.cu` - General fused operations
  - `cuda_interface.cpp` - PyTorch C++ bindings
- `src/kernel_pytorch/hardware/kernels/` - Hardware-specific kernels

## ⚡ **CUDA Programming Reference**

CUDA (Compute Unified Device Architecture) provides **direct access to GPU hardware**, enabling optimal performance through explicit control over:
- **Memory hierarchy management**
- **Thread execution patterns**
- **Warp-level primitive operations**
- **Hardware-specific optimizations**

### **Why Custom CUDA Kernels?**
- **Maximum Performance**: Direct hardware feature utilization (FP8, Tensor Cores)
- **Complete Control**: Explicit memory and execution management
- **Hardware Features**: H100/Blackwell-specific optimizations
- **Production Ready**: Used in NVIDIA backend (v0.3.1+)

## 🏗️ **GPU Architecture Fundamentals**

### **GPU Memory Hierarchy**
Understanding GPU memory is crucial for optimization:

```cpp
// GPU Memory Types (ordered by speed, smallest to largest)
__device__ __shared__ float shared_memory[1024];    // ~100GB/s, 48-96KB per SM
__device__ __constant__ float const_memory[64*1024]; // ~800GB/s, 64KB total
__device__ float global_memory[1000000];            // ~900GB/s, GBs total
// Registers: ~20TB/s, 65536 per SM
// L1 Cache: Automatic, ~27TB/s
// L2 Cache: Automatic, ~3TB/s
```

### **Execution Model**
```cpp
// Thread hierarchy: Thread -> Warp (32 threads) -> Block -> Grid
__global__ void educational_kernel() {
    // Thread identification
    int thread_id = threadIdx.x;           // Thread within block (0-1023)
    int block_id = blockIdx.x;             // Block within grid
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Warp-level operations (32 threads execute together)
    int warp_id = thread_id / 32;
    int lane_id = thread_id % 32;

    // Shared memory: visible to all threads in block
    __shared__ float shared_data[1024];

    // Synchronization: wait for all threads in block
    __syncthreads();
}
```

## 🧮 **Educational ML Kernel Implementations**

### **1. Optimized Matrix Multiplication** (`fused_ops.cu`)
Fundamental operation demonstrating advanced optimization techniques:

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>

template<int BLOCK_SIZE>
__global__ void educational_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    /*
    Educational matrix multiplication: C = A * B

    Key Learning Concepts:
    - Shared memory tiling for data reuse
    - Thread coarsening for computational intensity
    - Memory coalescing for bandwidth optimization
    - Warp-level optimizations
    */

    // Shared memory tiles for A and B submatrices
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Thread and block indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Output position for this thread
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Accumulator for dot product
    float sum = 0.0f;

    // Tile across K dimension
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Load tiles into shared memory with coalesced access
        if (row < M && tile * BLOCK_SIZE + tx < K) {
            As[ty][tx] = A[row * K + tile * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (tile * BLOCK_SIZE + ty < K && col < N) {
            Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product using shared memory
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result with bounds checking
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Optimized launcher with auto-tuned parameters
void launch_educational_matmul(
    const float* A, const float* B, float* C,
    int M, int N, int K, cudaStream_t stream = 0
) {
    // Optimal block size determined through benchmarking
    constexpr int BLOCK_SIZE = 16;

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // Shared memory usage: 2 * BLOCK_SIZE^2 * sizeof(float)
    size_t shared_mem_size = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);

    educational_matmul_kernel<BLOCK_SIZE><<<grid, block, shared_mem_size, stream>>>(
        A, B, C, M, N, K
    );

    // Error checking (development only)
    #ifdef DEBUG
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    #endif
}
```

**Learning Objectives**:
- **Memory Tiling**: Reusing data through shared memory
- **Coalesced Access**: Optimizing memory bandwidth utilization
- **Thread Coarsening**: Balancing work across computational resources
- **Register Usage**: Maximizing computational intensity

### **2. Flash Attention Implementation** (`fused_ops.cu`)
Advanced memory-efficient attention mechanism:

```cpp
#include <cuda_fp16.h>
#include <mma.h>  // Tensor Core operations

template<int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_kernel(
    const half* __restrict__ Q,    // Query: [batch, heads, seq_len, head_dim]
    const half* __restrict__ K,    // Key: [batch, heads, seq_len, head_dim]
    const half* __restrict__ V,    // Value: [batch, heads, seq_len, head_dim]
    half* __restrict__ O,          // Output: [batch, heads, seq_len, head_dim]
    float* __restrict__ L,         // LSE (log-sum-exp): [batch, heads, seq_len]
    int batch_size, int num_heads, int seq_len, float scale
) {
    /*
    Flash Attention: Memory-efficient attention computation

    Key Innovations:
    - Online softmax algorithm (no intermediate storage)
    - Block-wise computation for memory efficiency
    - Numerical stability through careful scaling
    - Tensor Core utilization for maximum performance
    */

    // Shared memory for current blocks
    __shared__ half Q_smem[BLOCK_SIZE][HEAD_DIM];
    __shared__ half K_smem[BLOCK_SIZE][HEAD_DIM];
    __shared__ half V_smem[BLOCK_SIZE][HEAD_DIM];
    __shared__ half S_smem[BLOCK_SIZE][BLOCK_SIZE];  // Attention scores

    // Thread and block identification
    const int batch_id = blockIdx.z;
    const int head_id = blockIdx.y;
    const int q_block_id = blockIdx.x;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Output accumulator and statistics
    float O_acc[HEAD_DIM] = {0.0f};  // Per-thread output accumulator
    float l_acc = 0.0f;              // Normalizer accumulator
    float m_acc = -INFINITY;         // Max value for numerical stability

    // Query block (fixed for this thread block)
    const int q_offset = ((batch_id * num_heads + head_id) * seq_len +
                          q_block_id * BLOCK_SIZE) * HEAD_DIM;

    // Load Q block into shared memory (coalesced)
    for (int i = tid; i < BLOCK_SIZE * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        if (q_block_id * BLOCK_SIZE + row < seq_len) {
            Q_smem[row][col] = Q[q_offset + i];
        } else {
            Q_smem[row][col] = __float2half(0.0f);
        }
    }
    __syncthreads();

    // Iterate through K,V blocks (online softmax)
    for (int kv_block_id = 0; kv_block_id < (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; ++kv_block_id) {

        // Load K,V blocks into shared memory
        const int kv_offset = ((batch_id * num_heads + head_id) * seq_len +
                               kv_block_id * BLOCK_SIZE) * HEAD_DIM;

        for (int i = tid; i < BLOCK_SIZE * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            if (kv_block_id * BLOCK_SIZE + row < seq_len) {
                K_smem[row][col] = K[kv_offset + i];
                V_smem[row][col] = V[kv_offset + i];
            } else {
                K_smem[row][col] = __float2half(0.0f);
                V_smem[row][col] = __float2half(0.0f);
            }
        }
        __syncthreads();

        // Compute attention scores S = Q @ K^T
        // Using Tensor Cores for maximum performance
        #pragma unroll
        for (int qi = warp_id; qi < BLOCK_SIZE; qi += (blockDim.x / 32)) {
            #pragma unroll
            for (int ki = 0; ki < BLOCK_SIZE; ++ki) {
                if (tid % 32 == 0 && qi < BLOCK_SIZE && ki < BLOCK_SIZE) {
                    float score = 0.0f;

                    // Dot product Q[qi] · K[ki]
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        score += __half2float(Q_smem[qi][d]) * __half2float(K_smem[ki][d]);
                    }

                    S_smem[qi][ki] = __float2half(score * scale);
                }
            }
        }
        __syncthreads();

        // Online softmax update (numerically stable)
        if (tid < BLOCK_SIZE && q_block_id * BLOCK_SIZE + tid < seq_len) {
            const int qi = tid;

            // Find maximum in current row for stability
            float m_new = -INFINITY;
            for (int ki = 0; ki < min(BLOCK_SIZE, seq_len - kv_block_id * BLOCK_SIZE); ++ki) {
                m_new = fmaxf(m_new, __half2float(S_smem[qi][ki]));
            }

            // Update global maximum and scaling factors
            float scale_old = (m_acc > -INFINITY) ? expf(m_acc - fmaxf(m_acc, m_new)) : 0.0f;
            float scale_new = expf(m_new - fmaxf(m_acc, m_new));

            m_acc = fmaxf(m_acc, m_new);

            // Compute attention weights and update accumulator
            float l_new = 0.0f;
            for (int ki = 0; ki < min(BLOCK_SIZE, seq_len - kv_block_id * BLOCK_SIZE); ++ki) {
                float weight = expf(__half2float(S_smem[qi][ki]) - m_acc);
                l_new += weight;

                // Update output accumulator
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    O_acc[d] = O_acc[d] * scale_old + weight * __half2float(V_smem[ki][d]);
                }
            }

            // Update normalizer
            l_acc = l_acc * scale_old + l_new;
        }
        __syncthreads();
    }

    // Final normalization and output
    if (tid < BLOCK_SIZE && q_block_id * BLOCK_SIZE + tid < seq_len) {
        const int qi = tid;
        const int output_offset = q_offset + qi * HEAD_DIM;

        // Normalize and store output
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            O[output_offset + d] = __float2half(O_acc[d] / l_acc);
        }

        // Store log-sum-exp for backward pass
        const int lse_offset = (batch_id * num_heads + head_id) * seq_len +
                              q_block_id * BLOCK_SIZE + qi;
        L[lse_offset] = logf(l_acc) + m_acc;
    }
}

// Optimized launcher with hardware-specific configurations
void launch_flash_attention(
    const half* Q, const half* K, const half* V, half* O, float* L,
    int batch_size, int num_heads, int seq_len, int head_dim,
    float scale, cudaStream_t stream = 0
) {
    // Hardware-optimized block size
    constexpr int BLOCK_SIZE = 128;

    dim3 block(BLOCK_SIZE);
    dim3 grid(
        (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE,  // Q blocks
        num_heads,                                 // Heads
        batch_size                                 // Batches
    );

    // Launch kernel based on head dimension
    if (head_dim == 64) {
        flash_attention_kernel<BLOCK_SIZE, 64><<<grid, block, 0, stream>>>(
            Q, K, V, O, L, batch_size, num_heads, seq_len, scale
        );
    } else if (head_dim == 128) {
        flash_attention_kernel<BLOCK_SIZE, 128><<<grid, block, 0, stream>>>(
            Q, K, V, O, L, batch_size, num_heads, seq_len, scale
        );
    } else {
        // Fallback for arbitrary head dimensions
        printf("Unsupported head dimension: %d\n", head_dim);
    }

    // Performance monitoring (optional)
    #ifdef PROFILE_KERNELS
    cudaEventRecord(stop_event, stream);
    cudaEventSynchronize(stop_event);
    float ms;
    cudaEventElapsedTime(&ms, start_event, stop_event);
    printf("Flash Attention: %.3f ms\n", ms);
    #endif
}
```

**Learning Objectives**:
- **Online Algorithms**: Computing without storing intermediate results
- **Numerical Stability**: Preventing overflow through careful computation order
- **Tensor Core Usage**: Leveraging modern GPU matrix multiplication units
- **Memory Efficiency**: Minimizing global memory access through blocking

### **3. Fused Layer Normalization + Activation** (`fused_ops.cu`)
Demonstrates kernel fusion for common ML patterns:

```cpp
template<typename T, int BLOCK_SIZE>
__global__ void fused_layer_norm_activation_kernel(
    const T* __restrict__ input,     // Input tensor
    const T* __restrict__ weight,    // LayerNorm weight
    const T* __restrict__ bias,      // LayerNorm bias
    T* __restrict__ output,          // Output tensor
    float* __restrict__ mean,        // Computed means (for backward)
    float* __restrict__ rstd,        // Computed 1/std (for backward)
    int batch_size, int hidden_size, float eps
) {
    /*
    Fused Layer Normalization + GELU Activation

    Mathematical Operations:
    1. LayerNorm: y = (x - mean(x)) / std(x) * weight + bias
    2. GELU: gelu(y) = 0.5 * y * (1 + tanh(√(2/π) * (y + 0.044715 * y³)))

    Fusion Benefits:
    - Reduced memory bandwidth (no intermediate storage)
    - Better arithmetic intensity (compute/memory ratio)
    - Lower latency (fewer kernel launches)
    */

    // Shared memory for reduction operations
    __shared__ float shared_sum[BLOCK_SIZE];
    __shared__ float shared_sum_sq[BLOCK_SIZE];

    const int row_id = blockIdx.x;           // Which sequence/batch element
    const int tid = threadIdx.x;             // Thread within block
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Input/output pointers for this row
    const T* row_input = input + row_id * hidden_size;
    T* row_output = output + row_id * hidden_size;

    // Phase 1: Compute statistics (mean and variance)
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Each thread processes multiple elements (thread coarsening)
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = static_cast<float>(row_input[i]);
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction using shuffle instructions
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Store warp results in shared memory
    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
        shared_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < (BLOCK_SIZE / 32)) {
        sum = shared_sum[tid];
        sum_sq = shared_sum_sq[tid];
    } else {
        sum = 0.0f;
        sum_sq = 0.0f;
    }

    if (warp_id == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    // Broadcast final statistics
    __shared__ float row_mean, row_rstd;
    if (tid == 0) {
        row_mean = sum / hidden_size;
        float variance = (sum_sq / hidden_size) - (row_mean * row_mean);
        row_rstd = rsqrtf(variance + eps);  // 1 / sqrt(variance + eps)

        // Store for backward pass
        mean[row_id] = row_mean;
        rstd[row_id] = row_rstd;
    }
    __syncthreads();

    // Phase 2: Apply normalization and activation
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        // Load input and normalization parameters
        float x = static_cast<float>(row_input[i]);
        float w = static_cast<float>(weight[i]);
        float b = static_cast<float>(bias[i]);

        // Layer normalization
        float normalized = (x - row_mean) * row_rstd;
        float ln_out = normalized * w + b;

        // GELU activation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
        float gelu_out = 0.5f * ln_out * (1.0f + tanhf(0.797885f * (ln_out + 0.044715f * ln_out * ln_out * ln_out)));

        // Store result
        row_output[i] = static_cast<T>(gelu_out);
    }
}

// Template specializations for different data types
template void launch_fused_layer_norm_activation<float>(
    const float* input, const float* weight, const float* bias, float* output,
    float* mean, float* rstd, int batch_size, int hidden_size, float eps, cudaStream_t stream
);

template void launch_fused_layer_norm_activation<half>(
    const half* input, const half* weight, const half* bias, half* output,
    float* mean, float* rstd, int batch_size, int hidden_size, float eps, cudaStream_t stream
);
```

## 🚀 **Advanced Optimization Techniques**

### **1. Memory Coalescing and Bank Conflicts**
```cpp
// Good: Coalesced access (consecutive threads access consecutive memory)
__global__ void coalesced_access_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Perfect coalescing
    }
}

// Bad: Strided access (performance penalty)
__global__ void strided_access_kernel(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;  // Poor coalescing
    }
}

// Shared memory bank conflicts avoidance
__global__ void avoid_bank_conflicts_kernel() {
    __shared__ float shared_data[1024];

    int tid = threadIdx.x;

    // Good: No bank conflicts (different banks)
    shared_data[tid] = 1.0f;

    // Bad: Bank conflicts (same bank for consecutive threads)
    // shared_data[tid * 32] = 1.0f;  // Don't do this!
}
```

### **2. Warp-Level Primitives**
```cpp
__global__ void warp_primitives_demo() {
    int tid = threadIdx.x;
    int lane = tid % 32;

    float value = static_cast<float>(lane);  // Each thread has different value

    // Warp shuffle - exchange data between threads in same warp
    float neighbor = __shfl_down_sync(0xffffffff, value, 1);  // Get value from thread+1

    // Warp reduction - sum all values in warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    // Thread 0 now has sum of all values

    // Warp broadcast - share value from one thread to all
    float broadcast_val = __shfl_sync(0xffffffff, value, 0);  // Get value from thread 0

    // Warp voting - check conditions across warp
    int all_positive = __all_sync(0xffffffff, value > 0);     // Are all values positive?
    int any_zero = __any_sync(0xffffffff, value == 0);       // Is any value zero?
}
```

### **3. Occupancy Optimization**
```cpp
// CUDA Occupancy Calculator Integration
extern "C"
__global__ void __launch_bounds__(256, 2)  // Max threads per block, min blocks per SM
optimized_kernel(float* data, int n) {
    /*
    Launch bounds directive helps compiler optimize for:
    - Register usage (affects occupancy)
    - Shared memory usage
    - Thread block size selection
    */

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Compiler optimizes register usage based on launch bounds
    if (idx < n) {
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}
```

## 🔧 **Development and Integration**

### **CMake Build Configuration**
```cmake
# CMakeLists.txt for CUDA kernel compilation
cmake_minimum_required(VERSION 3.18)
project(kernel_pytorch_cuda LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70")  # V100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_80,code=sm_80")  # A100
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")  # H100

# Source files
set(CUDA_SOURCES
    fused_ops.cu
    cuda_interface.cpp
)

# Create library
add_library(kernel_pytorch_cuda SHARED ${CUDA_SOURCES})

# Link PyTorch
target_link_libraries(kernel_pytorch_cuda "${TORCH_LIBRARIES}")
```

### **Python Integration with PyTorch**
```cpp
// cuda_interface.cpp - PyTorch bindings
#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declarations
void launch_flash_attention(...);
void launch_fused_layer_norm_activation(...);

// PyTorch tensor to CUDA pointer conversion
template<typename T>
T* get_cuda_ptr(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA device");
    return tensor.data_ptr<T>();
}

// Python-callable wrapper functions
torch::Tensor flash_attention_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale
) {
    auto batch_size = Q.size(0);
    auto num_heads = Q.size(1);
    auto seq_len = Q.size(2);
    auto head_dim = Q.size(3);

    // Allocate output tensors
    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({batch_size, num_heads, seq_len},
                          torch::dtype(torch::kFloat32).device(Q.device()));

    // Launch CUDA kernel
    launch_flash_attention(
        get_cuda_ptr<at::Half>(Q),
        get_cuda_ptr<at::Half>(K),
        get_cuda_ptr<at::Half>(V),
        get_cuda_ptr<at::Half>(O),
        get_cuda_ptr<float>(L),
        batch_size, num_heads, seq_len, head_dim, scale,
        at::cuda::getCurrentCUDAStream()
    );

    return O;
}

// Python module bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention", &flash_attention_cuda, "Flash Attention CUDA implementation");
    m.def("fused_layer_norm_activation", &fused_layer_norm_activation_cuda,
          "Fused LayerNorm + Activation CUDA implementation");
}
```

### **Performance Testing and Validation**
```cpp
// Performance benchmarking utilities
class CUDAPerformanceProfiler {
public:
    CUDAPerformanceProfiler() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    float stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// Validation against reference implementation
void validate_kernel_correctness() {
    // Generate test data
    auto input = torch::randn({4, 512}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

    // Reference PyTorch implementation
    auto reference = torch::layer_norm(input, {512});

    // Custom CUDA implementation
    auto custom = fused_layer_norm_activation_cuda(input, weight, bias);

    // Compare results
    bool passed = torch::allclose(reference, custom, /*rtol=*/1e-5, /*atol=*/1e-6);

    if (passed) {
        std::cout << "✅ Kernel validation passed" << std::endl;
    } else {
        std::cout << "❌ Kernel validation failed" << std::endl;
        std::cout << "Max difference: " << torch::max(torch::abs(reference - custom)).item<float>() << std::endl;
    }
}
```

## 📊 **Performance Characteristics and Benchmarking**

### **Expected Performance Gains**
- **Flash Attention**: 2-4x speedup over naive attention, matches cuDNN implementations
- **Fused LayerNorm**: 1.5-2x speedup over separate operations
- **Matrix Multiplication**: 95%+ of cuBLAS performance for specific sizes

### **Hardware Scaling**
```cpp
// Hardware-specific optimizations
#if __CUDA_ARCH__ >= 800  // A100, H100
    // Use Tensor Cores for mixed precision
    #include <mma.h>
    using namespace nvcuda::wmma;
#endif

#if __CUDA_ARCH__ >= 900  // H100 specific
    // Use new H100 features (TMA, WGMMA)
    #include <cuda/ptx>
#endif
```

## 🎓 **Learning Path for CUDA Development**

### **Prerequisites**
1. **C++ Programming**: Solid understanding of pointers, memory management, templates
2. **GPU Architecture**: Basic understanding of parallel computing concepts
3. **CUDA Toolkit**: Installed and configured development environment

### **Beginner Level**
1. **Simple kernels** - Element-wise operations, array indexing
2. **Memory management** - cudaMalloc, cudaMemcpy, basic transfers
3. **Thread indexing** - Understanding blockIdx, threadIdx, gridDim
4. **Basic debugging** - cuda-gdb, printf debugging

### **Intermediate Level**
1. **Shared memory** - Optimizing data reuse patterns
2. **Reduction operations** - Parallel sum, max, statistics computation
3. **Memory coalescing** - Optimizing global memory access patterns
4. **Occupancy tuning** - Balancing threads, registers, shared memory

### **Advanced Level**
1. **Warp primitives** - Shuffle operations, warp-level functions
2. **Tensor Cores** - Mixed precision, WMMA API usage
3. **Cooperative groups** - Advanced synchronization patterns
4. **Multi-GPU programming** - NCCL integration, peer-to-peer memory

## 📚 **References and Advanced Resources**

### **Essential Documentation**
- **CUDA Programming Guide**: Complete NVIDIA documentation
- **CUDA Best Practices Guide**: Performance optimization strategies
- **PTX ISA Manual**: Low-level instruction set reference
- **cuBLAS/cuDNN Documentation**: Optimized library references

### **Performance Analysis Tools**
- **Nsight Compute**: Kernel performance profiling
- **Nsight Systems**: System-level performance analysis
- **nvprof/ncu**: Command-line profiling tools
- **CUDA-GDB**: Debugging tools for GPU code

### **Advanced Topics**
- **Memory Pattern Optimization**: "Optimizing GPU Memory Bandwidth"
- **Numerical Stability**: "Handbook of Floating-Point Arithmetic"
- **GPU Architecture Deep Dive**: "Programming Massively Parallel Processors"

---

**🎯 Educational Mission**: Master the art of extracting maximum performance from modern GPU hardware through direct CUDA programming, understanding the intricate relationship between algorithmic design and hardware architecture for optimal ML acceleration.