/*
Custom CUDA Kernels for Optimized Operations

These kernels demonstrate maximum control over GPU execution,
showing how to implement fused operations for specific neural network patterns.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>
#include <cub/cub.cuh>

// Helper for error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        } \
    } while(0)

// Constants for optimizations
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;

/*
=== FUSED LAYER NORM KERNEL ===
Demonstrates:
- Warp-level reductions for parallel statistics computation
- Shared memory for efficient data reuse
- Vectorized memory access patterns
*/

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    static __shared__ T shared[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum<T>(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
    if (wid == 0) val = warpReduceSum<T>(val);

    return val;
}

template<typename T>
__global__ void fused_layer_norm_kernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int batch_size,
    int hidden_size,
    float eps
) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (batch_idx >= batch_size) return;

    const T* x = input + batch_idx * hidden_size;
    T* y = output + batch_idx * hidden_size;

    // Compute mean
    T sum = T(0);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += x[i];
    }
    sum = blockReduceSum<T>(sum);
    __shared__ T s_mean;
    if (tid == 0) s_mean = sum / T(hidden_size);
    __syncthreads();
    T mean = s_mean;

    // Compute variance
    T var_sum = T(0);
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        T diff = x[i] - mean;
        var_sum += diff * diff;
    }
    var_sum = blockReduceSum<T>(var_sum);
    __shared__ T s_var;
    if (tid == 0) s_var = var_sum / T(hidden_size);
    __syncthreads();
    T variance = s_var;

    // Normalize and scale
    T inv_std = T(1) / sqrt(variance + T(eps));
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        y[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
    }
}

/*
=== FUSED SWIGLU KERNEL ===
Demonstrates:
- Fused activation computation
- Memory bandwidth optimization
- Efficient elementwise operations
*/

template<typename T>
__device__ __forceinline__ T silu(T x) {
    return x / (T(1) + exp(-x));
}

template<typename T>
__global__ void fused_swiglu_kernel(
    const T* __restrict__ x,
    const T* __restrict__ gate_proj,
    const T* __restrict__ up_proj,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int input_dim,
    int hidden_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int hidden_idx = threadIdx.x + blockIdx.z * blockDim.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len || hidden_idx >= hidden_dim) {
        return;
    }

    const T* input_ptr = x + batch_idx * seq_len * input_dim + seq_idx * input_dim;
    T* output_ptr = output + batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim;

    // Compute gate and up projections
    T gate_val = T(0);
    T up_val = T(0);

    for (int i = 0; i < input_dim; i++) {
        T input_val = input_ptr[i];
        gate_val += input_val * gate_proj[hidden_idx * input_dim + i];
        up_val += input_val * up_proj[hidden_idx * input_dim + i];
    }

    // Apply SwiGLU: gate * silu(up)
    output_ptr[hidden_idx] = gate_val * silu(up_val);
}

/*
=== FLASH ATTENTION KERNEL (Simplified) ===
Demonstrates:
- Tiled computation for memory efficiency
- Online algorithm for softmax
- Shared memory management
*/

template<typename T, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void flash_attention_kernel(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Shared memory for tiles
    __shared__ T s_q[BLOCK_M * BLOCK_K];
    __shared__ T s_k[BLOCK_N * BLOCK_K];
    __shared__ T s_v[BLOCK_N * BLOCK_K];
    __shared__ T s_qk[BLOCK_M * BLOCK_N];

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int m_block = blockIdx.x;

    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;

    // Initialize output accumulator
    T acc[BLOCK_M] = {0};
    T max_val[BLOCK_M];
    T sum_exp[BLOCK_M];

    for (int i = 0; i < BLOCK_M; i++) {
        max_val[i] = -INFINITY;
        sum_exp[i] = T(0);
    }

    // Process K,V blocks
    for (int n_block = 0; n_block < (seq_len + BLOCK_N - 1) / BLOCK_N; n_block++) {
        // Load Q tile (only once per block)
        if (n_block == 0) {
            for (int i = tid; i < BLOCK_M * BLOCK_K; i += blockDim.x) {
                int m_local = i / BLOCK_K;
                int k_local = i % BLOCK_K;
                int m_global = m_block * BLOCK_M + m_local;

                if (m_global < seq_len && k_local < head_dim) {
                    int q_idx = batch_idx * num_heads * seq_len * head_dim +
                               head_idx * seq_len * head_dim +
                               m_global * head_dim + k_local;
                    s_q[i] = Q[q_idx];
                } else {
                    s_q[i] = T(0);
                }
            }
        }

        // Load K,V tiles
        for (int i = tid; i < BLOCK_N * BLOCK_K; i += blockDim.x) {
            int n_local = i / BLOCK_K;
            int k_local = i % BLOCK_K;
            int n_global = n_block * BLOCK_N + n_local;

            if (n_global < seq_len && k_local < head_dim) {
                int k_idx = batch_idx * num_heads * seq_len * head_dim +
                           head_idx * seq_len * head_dim +
                           n_global * head_dim + k_local;
                s_k[i] = K[k_idx];
                s_v[i] = V[k_idx];
            } else {
                s_k[i] = T(0);
                s_v[i] = T(0);
            }
        }
        __syncthreads();

        // Compute Q*K^T
        for (int m_local = 0; m_local < BLOCK_M; m_local++) {
            for (int n_local = tid; n_local < BLOCK_N; n_local += blockDim.x) {
                T sum = T(0);
                for (int k_local = 0; k_local < BLOCK_K; k_local++) {
                    sum += s_q[m_local * BLOCK_K + k_local] *
                           s_k[n_local * BLOCK_K + k_local];
                }
                s_qk[m_local * BLOCK_N + n_local] = sum * T(scale);
            }
        }
        __syncthreads();

        // Online softmax and accumulation
        for (int m_local = 0; m_local < BLOCK_M; m_local++) {
            T new_max = max_val[m_local];
            for (int n_local = 0; n_local < BLOCK_N; n_local++) {
                new_max = max(new_max, s_qk[m_local * BLOCK_N + n_local]);
            }

            T exp_sum = T(0);
            for (int n_local = 0; n_local < BLOCK_N; n_local++) {
                T exp_val = exp(s_qk[m_local * BLOCK_N + n_local] - new_max);
                s_qk[m_local * BLOCK_N + n_local] = exp_val;
                exp_sum += exp_val;
            }

            T old_scale = exp(max_val[m_local] - new_max);
            T new_scale = old_scale * sum_exp[m_local] + exp_sum;

            // Update accumulator
            acc[m_local] = acc[m_local] * old_scale;
            for (int k_local = 0; k_local < head_dim; k_local++) {
                T v_sum = T(0);
                for (int n_local = 0; n_local < BLOCK_N; n_local++) {
                    v_sum += s_qk[m_local * BLOCK_N + n_local] *
                             s_v[n_local * BLOCK_K + k_local];
                }
                if (k_local < head_dim) {
                    acc[m_local] += v_sum;
                }
            }
            acc[m_local] /= new_scale;

            max_val[m_local] = new_max;
            sum_exp[m_local] = new_scale;
        }
        __syncthreads();
    }

    // Write output
    for (int m_local = 0; m_local < BLOCK_M; m_local++) {
        int m_global = m_block * BLOCK_M + m_local;
        if (m_global < seq_len) {
            int o_idx = batch_idx * num_heads * seq_len * head_dim +
                       head_idx * seq_len * head_dim +
                       m_global * head_dim + tid;
            if (tid < head_dim) {
                O[o_idx] = acc[m_local];
            }
        }
    }
}

/*
=== ROTARY POSITION EMBEDDING KERNEL ===
Demonstrates:
- Complex number operations on GPU
- Efficient trigonometric computations
*/

template<typename T>
__global__ void rotary_embedding_kernel(
    const T* __restrict__ input,
    const T* __restrict__ cos,
    const T* __restrict__ sin,
    T* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int dim_idx = threadIdx.x * 2; // Process pairs

    if (batch_idx >= batch_size || seq_idx >= seq_len ||
        head_idx >= num_heads || dim_idx >= head_dim) {
        return;
    }

    int base_idx = batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim +
                   seq_idx * head_dim;

    T x1 = input[base_idx + dim_idx];
    T x2 = input[base_idx + dim_idx + 1];
    T cos_val = cos[seq_idx * head_dim / 2 + dim_idx / 2];
    T sin_val = sin[seq_idx * head_dim / 2 + dim_idx / 2];

    output[base_idx + dim_idx] = x1 * cos_val - x2 * sin_val;
    output[base_idx + dim_idx + 1] = x2 * cos_val + x1 * sin_val;
}

// C++ interface functions
torch::Tensor fused_layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps
) {
    auto output = torch::zeros_like(input);

    int batch_size = input.size(0);
    int hidden_size = input.size(-1);

    dim3 grid(batch_size);
    dim3 block(min(hidden_size, BLOCK_SIZE));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_layer_norm_cuda", ([&] {
        fused_layer_norm_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            hidden_size,
            eps
        );
    }));

    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor fused_swiglu_cuda(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight
) {
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int input_dim = input.size(2);
    int hidden_dim = gate_weight.size(0);

    auto output = torch::zeros({batch_size, seq_len, hidden_dim}, input.options());

    dim3 grid(batch_size, seq_len, (hidden_dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(min(hidden_dim, BLOCK_SIZE));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "fused_swiglu_cuda", ([&] {
        fused_swiglu_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            gate_weight.data_ptr<scalar_t>(),
            up_weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            input_dim,
            hidden_dim
        );
    }));

    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor flash_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale
) {
    int batch_size = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    auto output = torch::zeros_like(Q);

    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;
    constexpr int BLOCK_K = 64;

    dim3 grid((seq_len + BLOCK_M - 1) / BLOCK_M, num_heads, batch_size);
    dim3 block(128); // Threads per block

    AT_DISPATCH_FLOATING_TYPES(Q.type(), "flash_attention_cuda", ([&] {
        flash_attention_kernel<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K><<<grid, block>>>(
            Q.data_ptr<scalar_t>(),
            K.data_ptr<scalar_t>(),
            V.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            scale
        );
    }));

    CUDA_CHECK(cudaGetLastError());
    return output;
}

torch::Tensor rotary_embedding_cuda(
    torch::Tensor input,
    torch::Tensor cos,
    torch::Tensor sin
) {
    auto output = torch::zeros_like(input);

    int batch_size = input.size(0);
    int num_heads = input.size(1);
    int seq_len = input.size(2);
    int head_dim = input.size(3);

    dim3 grid(batch_size, seq_len, num_heads);
    dim3 block(head_dim / 2);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "rotary_embedding_cuda", ([&] {
        rotary_embedding_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            cos.data_ptr<scalar_t>(),
            sin.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            seq_len,
            num_heads,
            head_dim
        );
    }));

    CUDA_CHECK(cudaGetLastError());
    return output;
}