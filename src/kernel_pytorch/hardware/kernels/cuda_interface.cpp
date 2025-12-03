/*
C++ Interface for Custom CUDA Kernels

This file provides the Python binding interface for our custom CUDA kernels,
demonstrating how to bridge low-level GPU programming with PyTorch's autograd system.
*/

#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor fused_layer_norm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps);

torch::Tensor fused_swiglu_cuda(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight);

torch::Tensor flash_attention_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale);

torch::Tensor rotary_embedding_cuda(
    torch::Tensor input,
    torch::Tensor cos,
    torch::Tensor sin);

// CPU fallback implementations for debugging/testing
torch::Tensor fused_layer_norm_cpu(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps) {

    // Simple CPU implementation for comparison
    auto mean = input.mean(-1, true);
    auto var = input.var(-1, false, true);
    auto normalized = (input - mean) / torch::sqrt(var + eps);
    return normalized * weight + bias;
}

torch::Tensor fused_swiglu_cpu(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight) {

    auto gate = torch::matmul(input, gate_weight.t());
    auto up = torch::matmul(input, up_weight.t());
    return gate * torch::sigmoid(gate) * up;  // SwiGLU approximation
}

// Dispatch functions that choose between CPU and GPU implementations
torch::Tensor fused_layer_norm(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    float eps = 1e-5) {

    // Input validation
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1-dimensional");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1-dimensional");
    TORCH_CHECK(weight.size(0) == input.size(-1), "Weight size must match last input dimension");
    TORCH_CHECK(bias.size(0) == input.size(-1), "Bias size must match last input dimension");

    // Ensure contiguous memory layout
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    if (input.is_cuda()) {
        TORCH_CHECK(weight.is_cuda() && bias.is_cuda(), "All tensors must be on the same device");
        return fused_layer_norm_cuda(input, weight, bias, eps);
    } else {
        return fused_layer_norm_cpu(input, weight, bias, eps);
    }
}

torch::Tensor fused_swiglu(
    torch::Tensor input,
    torch::Tensor gate_weight,
    torch::Tensor up_weight) {

    // Input validation
    TORCH_CHECK(input.dim() == 3, "Input must be 3-dimensional [batch, seq, dim]");
    TORCH_CHECK(gate_weight.dim() == 2, "Gate weight must be 2-dimensional");
    TORCH_CHECK(up_weight.dim() == 2, "Up weight must be 2-dimensional");
    TORCH_CHECK(gate_weight.size(1) == input.size(-1), "Gate weight input dimension mismatch");
    TORCH_CHECK(up_weight.size(1) == input.size(-1), "Up weight input dimension mismatch");
    TORCH_CHECK(gate_weight.size(0) == up_weight.size(0), "Gate and up weights must have same output dimension");

    // Ensure contiguous memory layout
    input = input.contiguous();
    gate_weight = gate_weight.contiguous();
    up_weight = up_weight.contiguous();

    if (input.is_cuda()) {
        TORCH_CHECK(gate_weight.is_cuda() && up_weight.is_cuda(),
                   "All tensors must be on the same device");
        return fused_swiglu_cuda(input, gate_weight, up_weight);
    } else {
        return fused_swiglu_cpu(input, gate_weight, up_weight);
    }
}

torch::Tensor flash_attention(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    float scale) {

    // Input validation
    TORCH_CHECK(Q.dim() == 4, "Q must be 4-dimensional [batch, heads, seq, head_dim]");
    TORCH_CHECK(K.dim() == 4, "K must be 4-dimensional [batch, heads, seq, head_dim]");
    TORCH_CHECK(V.dim() == 4, "V must be 4-dimensional [batch, heads, seq, head_dim]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K must have the same shape");
    TORCH_CHECK(Q.sizes() == V.sizes(), "Q and V must have the same shape");

    // Ensure contiguous memory layout
    Q = Q.contiguous();
    K = K.contiguous();
    V = V.contiguous();

    if (Q.is_cuda()) {
        TORCH_CHECK(K.is_cuda() && V.is_cuda(), "All tensors must be on the same device");
        return flash_attention_cuda(Q, K, V, scale);
    } else {
        // CPU fallback using PyTorch's native implementation
        auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale;
        auto weights = torch::softmax(scores, -1);
        return torch::matmul(weights, V);
    }
}

torch::Tensor rotary_embedding(
    torch::Tensor input,
    torch::Tensor cos,
    torch::Tensor sin) {

    // Input validation
    TORCH_CHECK(input.dim() == 4, "Input must be 4-dimensional [batch, heads, seq, head_dim]");
    TORCH_CHECK(cos.dim() == 2, "Cos must be 2-dimensional [seq, head_dim/2]");
    TORCH_CHECK(sin.dim() == 2, "Sin must be 2-dimensional [seq, head_dim/2]");
    TORCH_CHECK(input.size(-1) % 2 == 0, "Head dimension must be even");

    // Ensure contiguous memory layout
    input = input.contiguous();
    cos = cos.contiguous();
    sin = sin.contiguous();

    if (input.is_cuda()) {
        TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "All tensors must be on the same device");
        return rotary_embedding_cuda(input, cos, sin);
    } else {
        // CPU fallback
        auto x1 = input.slice(-1, 0, input.size(-1), 2);  // Even indices
        auto x2 = input.slice(-1, 1, input.size(-1), 2);  // Odd indices

        cos = cos.unsqueeze(0).unsqueeze(0);  // Broadcast to [1, 1, seq, head_dim/2]
        sin = sin.unsqueeze(0).unsqueeze(0);

        auto rotated = torch::zeros_like(input);
        rotated.slice(-1, 0, input.size(-1), 2) = x1 * cos - x2 * sin;
        rotated.slice(-1, 1, input.size(-1), 2) = x2 * cos + x1 * sin;
        return rotated;
    }
}

// Utility functions for performance profiling
std::vector<double> benchmark_kernel(
    const std::string& kernel_name,
    std::function<torch::Tensor()> kernel_func,
    int num_iterations = 100,
    int warmup_iterations = 10) {

    std::vector<double> times;
    times.reserve(num_iterations);

    // Warmup
    for (int i = 0; i < warmup_iterations; i++) {
        auto result = kernel_func();
        if (result.is_cuda()) {
            torch::cuda::synchronize();
        }
    }

    // Actual timing
    for (int i = 0; i < num_iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto result = kernel_func();
        if (result.is_cuda()) {
            torch::cuda::synchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0);  // Convert to milliseconds
    }

    return times;
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom CUDA kernels for optimized PyTorch operations";

    // Core kernel functions
    m.def("fused_layer_norm", &fused_layer_norm,
          "Fused layer normalization",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);

    m.def("fused_swiglu", &fused_swiglu,
          "Fused SwiGLU activation",
          py::arg("input"), py::arg("gate_weight"), py::arg("up_weight"));

    m.def("flash_attention", &flash_attention,
          "Flash attention implementation",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("scale"));

    m.def("rotary_embedding", &rotary_embedding,
          "Rotary positional embedding",
          py::arg("input"), py::arg("cos"), py::arg("sin"));

    // Utility functions
    m.def("benchmark_kernel", &benchmark_kernel,
          "Benchmark a kernel function",
          py::arg("kernel_name"), py::arg("kernel_func"),
          py::arg("num_iterations") = 100, py::arg("warmup_iterations") = 10);

    // Version info
    m.attr("__version__") = "0.1.0";

    // Device info functions
    m.def("cuda_is_available", []() {
        return torch::cuda::is_available();
    });

    m.def("cuda_device_count", []() {
        return torch::cuda::device_count();
    });

    m.def("cuda_get_device_name", [](int device_id) {
        return torch::cuda::get_device_name(device_id);
    });
}