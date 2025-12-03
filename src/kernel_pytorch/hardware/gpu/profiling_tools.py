"""
Advanced GPU Profiling Tools
===========================

Comprehensive GPU profiling, benchmarking, and performance analysis tools
for PyTorch neural networks and kernel optimization.

This module provides:
1. Advanced GPU kernel profiling
2. Memory bandwidth analysis
3. Compute utilization measurement
4. Performance bottleneck detection
5. Optimization recommendation system
6. Educational profiling examples

Author: Advanced GPU Optimization Framework
"""

import torch
import torch.nn as nn
import torch.profiler
import torch.utils.benchmark as benchmark
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import time
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import psutil
import gc
from collections import defaultdict, OrderedDict


@dataclass
class KernelProfile:
    """Detailed kernel execution profile."""
    name: str
    duration_ms: float
    gpu_utilization: float
    memory_bandwidth_utilization: float
    arithmetic_intensity: float
    occupancy: float
    registers_per_thread: int
    shared_memory_usage: int
    grid_size: Tuple[int, int, int]
    block_size: Tuple[int, int, int]


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_time_ms: float
    gpu_time_ms: float
    cpu_time_ms: float
    memory_transfer_time_ms: float
    kernel_time_ms: float
    memory_usage_mb: float
    peak_memory_mb: float
    compute_efficiency: float
    memory_efficiency: float
    throughput_ops_per_sec: float


class GPUProfiler:
    """
    Advanced GPU profiler with detailed kernel and memory analysis.

    Provides comprehensive profiling capabilities for PyTorch operations
    with focus on GPU kernel performance and optimization opportunities.
    """

    def __init__(self,
                 device: Optional[torch.device] = None,
                 warmup_runs: int = 5,
                 benchmark_runs: int = 100):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.profiling_results = []

    def profile_function(self,
                        func: Callable,
                        *args,
                        **kwargs) -> PerformanceMetrics:
        """
        Profile a function with comprehensive GPU metrics.

        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            PerformanceMetrics with detailed performance data
        """
        # Move arguments to device
        args = tuple(arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args)
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kwargs[key] = value.to(self.device)

        # Warmup
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                func(*args, **kwargs)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Memory measurements
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        torch.cuda.reset_peak_memory_stats()

        # Timing measurements
        start_time = time.perf_counter()
        torch.cuda.synchronize()

        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)

        gpu_start.record()

        # Execute function
        results = []
        for _ in range(self.benchmark_runs):
            result = func(*args, **kwargs)
            results.append(result)

        gpu_end.record()
        torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000
        gpu_time_ms = gpu_start.elapsed_time(gpu_end)
        current_memory = torch.cuda.memory_allocated() / 1024**2
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2

        # Estimate memory bandwidth utilization
        memory_accessed = peak_memory - initial_memory
        theoretical_bandwidth = self._get_theoretical_memory_bandwidth()
        memory_bandwidth_utilization = (memory_accessed * self.benchmark_runs / (gpu_time_ms / 1000)) / theoretical_bandwidth

        # Estimate compute efficiency
        ops_per_run = self._estimate_operations(func, args, kwargs)
        total_ops = ops_per_run * self.benchmark_runs
        compute_efficiency = self._calculate_compute_efficiency(total_ops, gpu_time_ms)

        return PerformanceMetrics(
            total_time_ms=total_time_ms,
            gpu_time_ms=gpu_time_ms,
            cpu_time_ms=total_time_ms - gpu_time_ms,
            memory_transfer_time_ms=0.0,  # TODO: Implement H2D/D2H transfer timing
            kernel_time_ms=gpu_time_ms,
            memory_usage_mb=current_memory,
            peak_memory_mb=peak_memory,
            compute_efficiency=compute_efficiency,
            memory_efficiency=min(memory_bandwidth_utilization, 1.0),
            throughput_ops_per_sec=total_ops / (gpu_time_ms / 1000) if gpu_time_ms > 0 else 0.0
        )

    def profile_model(self,
                     model: nn.Module,
                     sample_inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
                     include_backward: bool = True,
                     detailed_kernels: bool = False) -> Dict[str, Any]:
        """
        Profile a complete model with detailed layer-by-layer analysis.

        Args:
            model: PyTorch model to profile
            sample_inputs: Sample input tensors
            include_backward: Include backward pass profiling
            detailed_kernels: Include detailed kernel profiling

        Returns:
            Comprehensive profiling results
        """
        model = model.to(self.device)
        if isinstance(sample_inputs, torch.Tensor):
            sample_inputs = sample_inputs.to(self.device)
        else:
            sample_inputs = tuple(inp.to(self.device) for inp in sample_inputs)

        profiling_results = {
            'model_info': self._get_model_info(model),
            'forward_pass': {},
            'backward_pass': {},
            'layer_profiles': {},
            'memory_analysis': {},
            'optimization_recommendations': []
        }

        # Profile forward pass
        if isinstance(sample_inputs, tuple):
            forward_metrics = self.profile_function(model, *sample_inputs)
        else:
            forward_metrics = self.profile_function(model, sample_inputs)

        profiling_results['forward_pass'] = asdict(forward_metrics)

        # Profile backward pass
        if include_backward:
            def backward_pass():
                if isinstance(sample_inputs, tuple):
                    outputs = model(*sample_inputs)
                else:
                    outputs = model(sample_inputs)

                if isinstance(outputs, torch.Tensor):
                    loss = outputs.sum()
                else:
                    loss = sum(out.sum() for out in outputs if isinstance(out, torch.Tensor))

                loss.backward()
                model.zero_grad()

            backward_metrics = self.profile_function(backward_pass)
            profiling_results['backward_pass'] = asdict(backward_metrics)

        # Detailed layer profiling
        if detailed_kernels:
            profiling_results['layer_profiles'] = self._profile_layers(model, sample_inputs)

        # Memory analysis
        profiling_results['memory_analysis'] = self._analyze_memory_patterns(model, sample_inputs)

        # Generate optimization recommendations
        profiling_results['optimization_recommendations'] = self._generate_recommendations(
            forward_metrics,
            backward_metrics if include_backward else None
        )

        return profiling_results

    def benchmark_implementations(self,
                                implementations: Dict[str, Callable],
                                *args,
                                **kwargs) -> Dict[str, PerformanceMetrics]:
        """
        Benchmark multiple implementations of the same operation.

        Args:
            implementations: Dictionary mapping names to implementation functions
            *args: Arguments to pass to implementations
            **kwargs: Keyword arguments to pass to implementations

        Returns:
            Dictionary mapping implementation names to performance metrics
        """
        results = {}

        for name, impl in implementations.items():
            print(f"Benchmarking {name}...")
            try:
                metrics = self.profile_function(impl, *args, **kwargs)
                results[name] = metrics
            except Exception as e:
                print(f"Error benchmarking {name}: {e}")
                continue

        return results

    def _get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get basic model information."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024**2,  # Assuming float32
            'num_layers': len(list(model.modules())) - 1  # Exclude root module
        }

    def _profile_layers(self, model: nn.Module, sample_inputs) -> Dict[str, Dict[str, Any]]:
        """Profile individual layers of the model."""
        layer_profiles = {}

        # Register hooks for layer profiling
        layer_times = {}

        def make_hook(name):
            def hook(module, input, output):
                torch.cuda.synchronize()
                if name in layer_times:
                    layer_times[name].append(time.time())
                else:
                    layer_times[name] = [time.time()]
            return hook

        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(make_hook(name))
                hooks.append(hook)

        # Run model with timing
        model.eval()
        with torch.no_grad():
            if isinstance(sample_inputs, tuple):
                model(*sample_inputs)
            else:
                model(sample_inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Process timing data
        for name, times in layer_times.items():
            if len(times) >= 2:
                duration = times[-1] - times[0]
                layer_profiles[name] = {
                    'duration_ms': duration * 1000,
                    'relative_time': 0.0  # Will be calculated after all layers
                }

        # Calculate relative times
        total_time = sum(prof['duration_ms'] for prof in layer_profiles.values())
        if total_time > 0:
            for prof in layer_profiles.values():
                prof['relative_time'] = prof['duration_ms'] / total_time

        return layer_profiles

    def _analyze_memory_patterns(self, model: nn.Module, sample_inputs) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated()

        # Forward pass memory analysis
        if isinstance(sample_inputs, tuple):
            outputs = model(*sample_inputs)
        else:
            outputs = model(sample_inputs)

        forward_peak = torch.cuda.max_memory_allocated()

        # Backward pass memory analysis
        if isinstance(outputs, torch.Tensor):
            loss = outputs.sum()
        else:
            loss = sum(out.sum() for out in outputs if isinstance(out, torch.Tensor))

        loss.backward()
        backward_peak = torch.cuda.max_memory_allocated()

        return {
            'initial_memory_mb': initial_memory / 1024**2,
            'forward_peak_mb': forward_peak / 1024**2,
            'backward_peak_mb': backward_peak / 1024**2,
            'total_memory_increase_mb': (backward_peak - initial_memory) / 1024**2,
            'memory_efficiency': initial_memory / forward_peak if forward_peak > 0 else 0.0
        }

    def _generate_recommendations(self,
                                forward_metrics: PerformanceMetrics,
                                backward_metrics: Optional[PerformanceMetrics] = None) -> List[str]:
        """Generate optimization recommendations based on profiling results."""
        recommendations = []

        # Memory efficiency recommendations
        if forward_metrics.memory_efficiency < 0.5:
            recommendations.append("Consider using gradient checkpointing to reduce memory usage")

        if forward_metrics.peak_memory_mb > 8000:  # 8GB threshold
            recommendations.append("Large memory usage detected - consider reducing batch size or using memory-efficient attention")

        # Compute efficiency recommendations
        if forward_metrics.compute_efficiency < 0.3:
            recommendations.append("Low compute efficiency - consider kernel fusion or using torch.compile")

        if forward_metrics.memory_efficiency > 0.8 and forward_metrics.compute_efficiency < 0.5:
            recommendations.append("Memory-bound operation detected - focus on reducing memory access overhead")

        # Performance recommendations
        if backward_metrics and backward_metrics.total_time_ms > forward_metrics.total_time_ms * 3:
            recommendations.append("Backward pass is significantly slower - consider mixed precision training")

        if forward_metrics.throughput_ops_per_sec < 1e6:  # Arbitrary threshold
            recommendations.append("Low throughput detected - consider batch size optimization or model parallelism")

        return recommendations

    def _get_theoretical_memory_bandwidth(self) -> float:
        """Get theoretical memory bandwidth for the current GPU (GB/s)."""
        # This is a simplified estimation - real implementation would query GPU properties
        gpu_name = torch.cuda.get_device_name(self.device)

        # Rough estimates for common GPUs (in GB/s)
        bandwidth_estimates = {
            'V100': 900,
            'A100': 1555,
            'H100': 3000,
            'RTX 3080': 760,
            'RTX 3090': 936,
            'RTX 4090': 1008
        }

        for gpu_type, bandwidth in bandwidth_estimates.items():
            if gpu_type in gpu_name:
                return bandwidth

        return 500  # Default estimate

    def _estimate_operations(self, func: Callable, args: tuple, kwargs: dict) -> int:
        """Estimate the number of operations in a function call."""
        # This is a simplified estimation
        # Real implementation would analyze the computational graph
        if hasattr(func, '__self__') and isinstance(func.__self__, nn.Module):
            return self._estimate_module_operations(func.__self__, args)
        return 1000000  # Default estimate

    def _estimate_module_operations(self, module: nn.Module, inputs: tuple) -> int:
        """Estimate operations for a PyTorch module."""
        total_ops = 0

        for name, layer in module.named_modules():
            if isinstance(layer, nn.Linear):
                input_size = layer.in_features
                output_size = layer.out_features
                batch_size = inputs[0].shape[0] if inputs and len(inputs[0].shape) > 0 else 1
                total_ops += batch_size * input_size * output_size * 2  # FMA operations
            elif isinstance(layer, nn.Conv2d):
                # Simplified convolution operation estimation
                if inputs and len(inputs[0].shape) >= 4:
                    batch, channels, height, width = inputs[0].shape[:4]
                    kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
                    output_elements = batch * layer.out_channels * height * width
                    total_ops += kernel_ops * output_elements

        return max(total_ops, 1000000)  # Minimum estimate

    def _calculate_compute_efficiency(self, operations: int, gpu_time_ms: float) -> float:
        """Calculate compute efficiency based on operations and time."""
        if gpu_time_ms <= 0:
            return 0.0

        ops_per_second = operations / (gpu_time_ms / 1000)

        # Get theoretical peak FLOPS for current GPU (simplified)
        gpu_name = torch.cuda.get_device_name(self.device)

        # Rough estimates for peak FLOPS (in GFLOPS)
        peak_flops_estimates = {
            'V100': 15700,
            'A100': 19500,
            'H100': 67000,
            'RTX 3080': 29700,
            'RTX 3090': 35600,
            'RTX 4090': 83000
        }

        theoretical_peak = 10000 * 1e9  # Default 10 TFLOPS
        for gpu_type, flops in peak_flops_estimates.items():
            if gpu_type in gpu_name:
                theoretical_peak = flops * 1e9
                break

        return min(ops_per_second / theoretical_peak, 1.0)


class PerformanceBenchmark:
    """
    Comprehensive benchmarking suite for PyTorch operations.

    Provides standardized benchmarks for common operations and optimization techniques.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.profiler = GPUProfiler(device)

    def benchmark_attention_implementations(self,
                                          batch_size: int = 8,
                                          seq_len: int = 512,
                                          embed_dim: int = 512,
                                          num_heads: int = 8) -> Dict[str, PerformanceMetrics]:
        """Benchmark different attention implementations."""

        # Create test data
        q = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
        k = torch.randn(batch_size, seq_len, embed_dim, device=self.device)
        v = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

        implementations = {}

        # Standard attention
        def standard_attention(q, k, v):
            scale = (embed_dim // num_heads) ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(attn, dim=-1)
            return torch.matmul(attn, v)

        implementations['standard'] = standard_attention

        # Flash attention if available
        if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            def flash_attention(q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
            implementations['flash'] = flash_attention

        # Chunked attention
        def chunked_attention(q, k, v, chunk_size=128):
            batch_size, seq_len, embed_dim = q.shape
            outputs = []

            for i in range(0, seq_len, chunk_size):
                q_chunk = q[:, i:i+chunk_size]
                attn_chunk = torch.matmul(q_chunk, k.transpose(-2, -1))
                attn_chunk = torch.softmax(attn_chunk, dim=-1)
                out_chunk = torch.matmul(attn_chunk, v)
                outputs.append(out_chunk)

            return torch.cat(outputs, dim=1)

        implementations['chunked'] = chunked_attention

        return self.profiler.benchmark_implementations(implementations, q, k, v)

    def benchmark_normalization_implementations(self,
                                              batch_size: int = 32,
                                              seq_len: int = 512,
                                              embed_dim: int = 512) -> Dict[str, PerformanceMetrics]:
        """Benchmark different normalization implementations."""

        x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

        implementations = {}

        # Layer normalization
        def layer_norm(x):
            return torch.layer_norm(x, (embed_dim,))

        implementations['layer_norm'] = layer_norm

        # RMS normalization
        def rms_norm(x):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + 1e-5)

        implementations['rms_norm'] = rms_norm

        # Manual layer norm
        def manual_layer_norm(x):
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True, unbiased=False)
            return (x - mean) / torch.sqrt(var + 1e-5)

        implementations['manual_layer_norm'] = manual_layer_norm

        return self.profiler.benchmark_implementations(implementations, x)

    def benchmark_activation_functions(self,
                                     batch_size: int = 32,
                                     seq_len: int = 512,
                                     embed_dim: int = 512) -> Dict[str, PerformanceMetrics]:
        """Benchmark different activation function implementations."""

        x = torch.randn(batch_size, seq_len, embed_dim, device=self.device)

        implementations = {
            'relu': torch.relu,
            'gelu': torch.nn.functional.gelu,
            'gelu_tanh': lambda x: torch.nn.functional.gelu(x, approximate='tanh'),
            'swish': lambda x: x * torch.sigmoid(x),
            'mish': lambda x: x * torch.tanh(torch.nn.functional.softplus(x))
        }

        return self.profiler.benchmark_implementations(implementations, x)


def demonstrate_profiling_tools():
    """
    Comprehensive demonstration of GPU profiling tools.
    """
    print("📊 Advanced GPU Profiling Tools Demonstration")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    # Initialize profiler
    profiler = GPUProfiler(device)
    benchmark = PerformanceBenchmark(device)

    # Create sample model for profiling
    class ProfilingTestModel(nn.Module):
        def __init__(self, embed_dim=512, num_heads=8):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)

        def forward(self, x):
            # Self-attention block
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # Feed-forward block
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)

            return x

    model = ProfilingTestModel().to(device)
    sample_input = torch.randn(8, 128, 512, device=device)

    print(f"\n🏗️ Model Architecture:")
    print(f"  Embed dimension: 512")
    print(f"  Number of heads: 8")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Model profiling
    print(f"\n🔍 Comprehensive Model Profiling:")
    profile_results = profiler.profile_model(model, sample_input, include_backward=True, detailed_kernels=True)

    # Display results
    forward_metrics = profile_results['forward_pass']
    print(f"  Forward pass:")
    print(f"    Time: {forward_metrics['gpu_time_ms']:.2f} ms")
    print(f"    Memory: {forward_metrics['peak_memory_mb']:.1f} MB peak")
    print(f"    Compute efficiency: {forward_metrics['compute_efficiency']:.1%}")
    print(f"    Memory efficiency: {forward_metrics['memory_efficiency']:.1%}")

    if 'backward_pass' in profile_results:
        backward_metrics = profile_results['backward_pass']
        print(f"  Backward pass:")
        print(f"    Time: {backward_metrics['gpu_time_ms']:.2f} ms")
        print(f"    Memory: {backward_metrics['peak_memory_mb']:.1f} MB peak")

    # Display layer profiling
    if profile_results['layer_profiles']:
        print(f"\n⚙️ Layer-by-Layer Profiling:")
        sorted_layers = sorted(
            profile_results['layer_profiles'].items(),
            key=lambda x: x[1]['duration_ms'],
            reverse=True
        )

        for layer_name, layer_prof in sorted_layers[:5]:  # Top 5 layers
            print(f"    {layer_name}: {layer_prof['duration_ms']:.2f} ms "
                  f"({layer_prof['relative_time']:.1%})")

    # Optimization recommendations
    print(f"\n💡 Optimization Recommendations:")
    for rec in profile_results['optimization_recommendations']:
        print(f"    • {rec}")

    # Attention benchmarks
    print(f"\n⚡ Attention Implementation Benchmark:")
    attention_results = benchmark.benchmark_attention_implementations()

    for impl_name, metrics in attention_results.items():
        print(f"    {impl_name.capitalize():>12}: {metrics.gpu_time_ms:.2f} ms "
              f"({metrics.compute_efficiency:.1%} compute, {metrics.memory_efficiency:.1%} memory)")

    # Find best attention implementation
    best_attention = min(attention_results.items(), key=lambda x: x[1].gpu_time_ms)
    baseline = attention_results.get('standard')
    if baseline and best_attention[0] != 'standard':
        speedup = baseline.gpu_time_ms / best_attention[1].gpu_time_ms
        print(f"    Best: {best_attention[0]} ({speedup:.2f}x speedup)")

    # Normalization benchmarks
    print(f"\n🔢 Normalization Implementation Benchmark:")
    norm_results = benchmark.benchmark_normalization_implementations()

    for impl_name, metrics in norm_results.items():
        print(f"    {impl_name.replace('_', ' ').title():>15}: {metrics.gpu_time_ms:.2f} ms")

    # Activation function benchmarks
    print(f"\n🎯 Activation Function Benchmark:")
    activation_results = benchmark.benchmark_activation_functions()

    for impl_name, metrics in activation_results.items():
        print(f"    {impl_name.upper():>8}: {metrics.gpu_time_ms:.2f} ms")

    # Memory analysis
    print(f"\n💾 Memory Analysis:")
    memory_analysis = profile_results['memory_analysis']
    print(f"    Forward peak: {memory_analysis['forward_peak_mb']:.1f} MB")
    print(f"    Backward peak: {memory_analysis['backward_peak_mb']:.1f} MB")
    print(f"    Total increase: {memory_analysis['total_memory_increase_mb']:.1f} MB")
    print(f"    Memory efficiency: {memory_analysis['memory_efficiency']:.1%}")

    # Performance comparison with torch.compile
    print(f"\n🚀 torch.compile Performance Comparison:")

    # Original model
    original_metrics = profiler.profile_function(model, sample_input)

    # Compiled model
    compiled_model = torch.compile(model)
    compiled_metrics = profiler.profile_function(compiled_model, sample_input)

    speedup = original_metrics.gpu_time_ms / compiled_metrics.gpu_time_ms
    memory_reduction = (original_metrics.peak_memory_mb - compiled_metrics.peak_memory_mb) / original_metrics.peak_memory_mb * 100

    print(f"    Original: {original_metrics.gpu_time_ms:.2f} ms")
    print(f"    Compiled: {compiled_metrics.gpu_time_ms:.2f} ms")
    print(f"    Speedup: {speedup:.2f}x")
    print(f"    Memory change: {memory_reduction:+.1f}%")

    print(f"\n✅ GPU profiling demonstration complete!")
    print(f"Key insights:")
    print(f"  • Comprehensive performance analysis across multiple dimensions")
    print(f"  • Layer-by-layer profiling for bottleneck identification")
    print(f"  • Implementation comparison for optimization selection")
    print(f"  • Automated optimization recommendations")
    print(f"  • torch.compile impact measurement")


if __name__ == "__main__":
    demonstrate_profiling_tools()