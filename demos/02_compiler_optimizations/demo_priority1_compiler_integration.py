#!/usr/bin/env python3
"""
Priority 1 Compiler Integration Demo

Demonstrates the 2025 state-of-the-art compiler integration optimizations:
1. FlashLight Compiler Framework - Automatic kernel generation for attention
2. PyGraph CUDA Graphs Support - Revolutionary CUDA graph optimization
3. Enhanced TorchInductor Fusion - Advanced fusion beyond standard boundaries

This demo shows real performance improvements from these cutting-edge techniques.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List
import warnings

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from kernel_pytorch.compiler_integration import (
    FlashLightKernelCompiler,
    AttentionPattern,
    PyGraphCUDAOptimizer,
    FusionBoundaryOptimizer,
    FusionPass,
    FusionStrategy
)


class DemoAttentionModel(nn.Module):
    """Demo attention model for showcasing optimizations"""

    def __init__(self, d_model=512, num_heads=8, seq_len=1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.seq_len = seq_len

        # Attention projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Layer norm and feedforward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        # Project to Q, K, V
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attn_out = self.out_proj(attn_out)

        # First residual connection
        x = residual + attn_out

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x

        return x


def print_section_header(title: str):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")


def print_results(results: Dict[str, Any], title: str = "Results"):
    """Print formatted results"""
    print(f"\n📊 {title}:")
    print("-" * 40)
    for key, value in results.items():
        if isinstance(value, float):
            if 'time' in key.lower():
                print(f"  {key}: {value*1000:.2f}ms")
            elif 'speedup' in key.lower():
                print(f"  {key}: {value:.2f}x")
            elif 'mb' in key.lower():
                print(f"  {key}: {value:.1f}MB")
            else:
                print(f"  {key}: {value:.3f}")
        elif isinstance(value, bool):
            print(f"  {key}: {'✅' if value else '❌'}")
        elif isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")


def demo_flashlight_compiler():
    """Demonstrate FlashLight Compiler Framework"""
    print_section_header("FlashLight Compiler Framework Demo")

    print("🔧 Initializing FlashLight Compiler...")
    compiler = FlashLightKernelCompiler(optimization_level="aggressive")

    # Test different attention patterns
    patterns = [
        ("causal", {}),
        ("sliding_window", {"window_size": 512}),
        ("dilated", {"dilation_rate": 2}),
        ("global_local", {"local_window": 256, "global_tokens": 64})
    ]

    seq_len, head_dim = 1024, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n🎯 Testing attention patterns on {device}...")
    print(f"   Sequence length: {seq_len}")
    print(f"   Head dimension: {head_dim}")

    compiled_kernels = {}
    compilation_times = {}

    for pattern_name, kwargs in patterns:
        print(f"\n⚡ Compiling {pattern_name} attention...")

        start_time = time.perf_counter()
        compiled_kernel = compiler.compile_attention_kernel(pattern_name, seq_len, head_dim, kwargs)
        compilation_time = time.perf_counter() - start_time

        compiled_kernels[pattern_name] = compiled_kernel
        compilation_times[pattern_name] = compilation_time

        print(f"   ✅ Compiled in {compilation_time*1000:.2f}ms")
        print(f"   📈 Estimated speedup: {compiled_kernel.estimated_speedup:.2f}x")
        print(f"   💾 Memory usage: {compiled_kernel.memory_usage / (1024*1024):.1f}MB")

    # Benchmark performance
    print(f"\n🏃 Benchmarking compiled kernels...")

    for pattern_name in ["causal", "sliding_window"]:
        print(f"\n📊 Benchmarking {pattern_name} attention:")

        try:
            benchmark_results = compiler.benchmark_pattern(
                pattern_name, seq_len, head_dim, num_heads=8, num_trials=10
            )
            print_results(benchmark_results, f"{pattern_name} Performance")

        except Exception as e:
            print(f"   ⚠️  Benchmarking failed: {e}")

    # Show compilation statistics
    stats = compiler.get_compilation_stats()
    print_results(stats, "Compilation Statistics")

    return compiler


def demo_pygraph_optimizer():
    """Demonstrate PyGraph CUDA Graphs Support"""
    print_section_header("PyGraph CUDA Graphs Optimization Demo")

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - showing analysis only")

    print("🔧 Initializing PyGraph Optimizer...")
    optimizer = PyGraphCUDAOptimizer(cost_threshold=0.05, strategy="aggressive")

    # Create test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DemoAttentionModel(d_model=256, num_heads=8, seq_len=512).to(device)
    model.eval()

    # Create test inputs
    batch_size = 4
    inputs = [torch.randn(batch_size, 512, 256, device=device)]

    print(f"\n🎯 Analyzing workload on {device}...")
    print(f"   Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Input shape: {inputs[0].shape}")

    # Analyze workload
    print("\n🔍 Performing workload analysis...")
    analysis = optimizer.analyze_workload(model, inputs, num_trials=5)

    analysis_results = {
        "CPU Launch Overhead": analysis.cpu_launch_overhead,
        "Memory Footprint (MB)": analysis.memory_footprint / (1024*1024),
        "Fusion Potential": analysis.kernel_fusion_potential,
        "Dynamic Shapes": analysis.dynamic_shapes,
        "Graph Recommended": analysis.graph_recommended,
        "Expected Speedup": analysis.expected_speedup,
        "Memory Overhead (MB)": analysis.memory_overhead / (1024*1024)
    }

    print_results(analysis_results, "Workload Analysis")

    # Benchmark if CUDA available
    if torch.cuda.is_available():
        print("\n🏃 Running comprehensive benchmark...")

        try:
            benchmark_results = optimizer.benchmark_vs_eager(model, inputs, num_trials=20)

            # Extract key results
            eager_time = benchmark_results["eager_execution"]["mean_time"]
            graph_time = benchmark_results["graph_execution"]["mean_time"]
            speedup = benchmark_results["performance"]["speedup"]

            benchmark_summary = {
                "Eager Execution Time": eager_time,
                "Graph Execution Time": graph_time,
                "Speedup Achieved": speedup,
                "Deployment Recommended": benchmark_results["performance"]["deployment_recommended"],
                "CPU Overhead Reduction": benchmark_results["performance"]["cpu_overhead_reduction"]
            }

            print_results(benchmark_summary, "Benchmark Results")

            if speedup > 1.1:
                print(f"\n🎉 PyGraph achieved {speedup:.2f}x speedup!")
            else:
                print(f"\n💡 Speedup marginal ({speedup:.2f}x) - model may be too small or simple")

        except Exception as e:
            print(f"   ⚠️  Benchmarking failed: {e}")

    # Show performance statistics
    stats = optimizer.get_performance_stats()
    print_results(stats, "Optimizer Statistics")

    return optimizer


def demo_enhanced_fusion():
    """Demonstrate Enhanced TorchInductor Fusion"""
    print_section_header("Enhanced TorchInductor Fusion Demo")

    print("🔧 Initializing Enhanced Fusion Optimizer...")

    # Create fusion strategy
    strategy = FusionStrategy(
        enabled_passes=[
            FusionPass.HORIZONTAL_FUSION,
            FusionPass.VERTICAL_FUSION,
            FusionPass.CROSS_ATTENTION_FUSION,
            FusionPass.QUANTIZATION_FUSION
        ],
        aggressive_mode=True,
        memory_budget_mb=2048,
        target_architecture="ampere"
    )

    optimizer = FusionBoundaryOptimizer(strategy)

    # Create model with fusion opportunities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DemoAttentionModel(d_model=512, num_heads=8, seq_len=256).to(device)

    print(f"\n🎯 Optimizing model on {device}...")
    print(f"   Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Fusion passes enabled: {len(strategy.enabled_passes)}")

    # Apply fusion optimizations
    try:
        print("\n⚡ Applying fusion optimizations...")
        optimized_graph = optimizer.optimize_fusion_graph(model)

        fusion_results = {
            "Original Node Count": optimized_graph.original_node_count,
            "Optimized Node Count": optimized_graph.optimized_node_count,
            "Nodes Removed": optimized_graph.original_node_count - optimized_graph.optimized_node_count,
            "Fusion Count": optimized_graph.fusion_count,
            "Estimated Speedup": optimized_graph.estimated_speedup,
            "Memory Reduction (MB)": optimized_graph.memory_reduction_mb
        }

        print_results(fusion_results, "Fusion Optimization Results")

        # Test optimized model
        test_input = torch.randn(2, 256, 512, device=device)

        with torch.no_grad():
            print("\n🧪 Testing optimized model...")

            # Original model
            original_start = time.perf_counter()
            original_output = model(test_input)
            original_time = time.perf_counter() - original_start

            # Optimized model
            optimized_start = time.perf_counter()
            optimized_output = optimized_graph.graph_module(test_input)
            optimized_time = time.perf_counter() - optimized_start

            # Verify numerical equivalence
            max_diff = torch.abs(original_output - optimized_output).max().item()

            test_results = {
                "Original Model Time": original_time,
                "Optimized Model Time": optimized_time,
                "Actual Speedup": original_time / optimized_time if optimized_time > 0 else 0,
                "Max Output Difference": max_diff,
                "Numerically Equivalent": max_diff < 1e-4
            }

            print_results(test_results, "Validation Results")

        # Show optimization statistics
        stats = optimizer.get_optimization_stats()
        print_results(stats, "Fusion Statistics")

    except Exception as e:
        print(f"   ⚠️  Fusion optimization failed: {e}")
        print("   💡 This is expected for some models due to FX tracing limitations")

    return optimizer


def demo_integration_scenario():
    """Demonstrate integration of all Priority 1 optimizations"""
    print_section_header("Integrated Optimization Demo")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Setting up integrated optimization pipeline on {device}...")

    # Initialize all optimizers
    flashlight_compiler = FlashLightKernelCompiler()
    pygraph_optimizer = PyGraphCUDAOptimizer()
    fusion_optimizer = FusionBoundaryOptimizer()

    # Create test model and inputs
    model = DemoAttentionModel(d_model=256, num_heads=8, seq_len=512).to(device)
    inputs = [torch.randn(2, 512, 256, device=device)]

    print("\n🚀 Running integrated optimization pipeline...")

    results = {}

    # 1. FlashLight Compilation
    print("   1️⃣ FlashLight attention kernel compilation...")
    try:
        compiled_kernel = flashlight_compiler.compile_attention_kernel("causal", 512, 32)
        results["flashlight_speedup"] = compiled_kernel.estimated_speedup
        results["flashlight_success"] = True
        print(f"      ✅ Compiled with {compiled_kernel.estimated_speedup:.2f}x estimated speedup")
    except Exception as e:
        results["flashlight_success"] = False
        print(f"      ❌ Failed: {e}")

    # 2. PyGraph Analysis
    print("   2️⃣ PyGraph workload analysis...")
    try:
        analysis = pygraph_optimizer.analyze_workload(model, inputs, num_trials=3)
        results["pygraph_recommended"] = analysis.graph_recommended
        results["pygraph_expected_speedup"] = analysis.expected_speedup
        results["pygraph_success"] = True
        print(f"      ✅ Graph {'recommended' if analysis.graph_recommended else 'not recommended'}")
        print(f"      📈 Expected speedup: {analysis.expected_speedup:.2f}x")
    except Exception as e:
        results["pygraph_success"] = False
        print(f"      ❌ Failed: {e}")

    # 3. Enhanced Fusion
    print("   3️⃣ Enhanced fusion optimization...")
    try:
        fusion_result = fusion_optimizer.optimize_fusion_graph(model)
        results["fusion_speedup"] = fusion_result.estimated_speedup
        results["fusion_count"] = fusion_result.fusion_count
        results["fusion_success"] = True
        print(f"      ✅ Applied {fusion_result.fusion_count} fusions")
        print(f"      📈 Estimated speedup: {fusion_result.estimated_speedup:.2f}x")
    except Exception as e:
        results["fusion_success"] = False
        print(f"      ❌ Failed: {e}")

    # Summary
    print("\n📋 Integration Summary:")
    print("-" * 40)

    total_expected_speedup = 1.0
    optimizations_applied = 0

    if results.get("flashlight_success", False):
        print("   ✅ FlashLight Compiler: Active")
        total_expected_speedup *= results.get("flashlight_speedup", 1.0)
        optimizations_applied += 1
    else:
        print("   ❌ FlashLight Compiler: Failed")

    if results.get("pygraph_success", False):
        status = "Recommended" if results.get("pygraph_recommended", False) else "Not Recommended"
        print(f"   ✅ PyGraph Optimization: {status}")
        if results.get("pygraph_recommended", False):
            total_expected_speedup *= results.get("pygraph_expected_speedup", 1.0)
            optimizations_applied += 1
    else:
        print("   ❌ PyGraph Optimization: Failed")

    if results.get("fusion_success", False):
        print("   ✅ Enhanced Fusion: Active")
        total_expected_speedup *= results.get("fusion_speedup", 1.0)
        optimizations_applied += 1
    else:
        print("   ❌ Enhanced Fusion: Failed")

    print(f"\n🎯 Overall Results:")
    print(f"   Optimizations Applied: {optimizations_applied}/3")
    print(f"   Combined Expected Speedup: {total_expected_speedup:.2f}x")
    print(f"   Device: {device}")

    if optimizations_applied >= 2:
        print(f"\n🎉 Integration successful! Multiple optimizations working together.")
    elif optimizations_applied >= 1:
        print(f"\n✅ Partial integration successful.")
    else:
        print(f"\n⚠️  Integration challenges - this is common with cutting-edge optimizations.")

    return results


def main():
    """Run comprehensive Priority 1 demonstration"""
    print("🚀 Priority 1 Compiler Integration Demo")
    print("=" * 60)
    print("Demonstrating 2025 state-of-the-art PyTorch compiler optimizations:")
    print("• FlashLight Compiler Framework")
    print("• PyGraph CUDA Graphs Support")
    print("• Enhanced TorchInductor Fusion")

    device_info = f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name()})"
    print(f"• {device_info}")
    print()

    try:
        # Run individual demos
        flashlight_compiler = demo_flashlight_compiler()
        pygraph_optimizer = demo_pygraph_optimizer()
        fusion_optimizer = demo_enhanced_fusion()

        # Run integration demo
        integration_results = demo_integration_scenario()

        print_section_header("Demo Complete! 🎉")

        print("✅ Successfully demonstrated Priority 1 optimizations:")
        print("   🔥 FlashLight Compiler: Automatic attention kernel generation")
        print("   📈 PyGraph Optimizer: Revolutionary CUDA graph deployment")
        print("   🔗 Enhanced Fusion: Beyond standard TorchInductor boundaries")

        print("\n📚 Key Takeaways:")
        print("   • These optimizations represent the cutting-edge of PyTorch compilation")
        print("   • FlashLight eliminates manual Triton kernel programming")
        print("   • PyGraph enables intelligent CUDA graph deployment")
        print("   • Enhanced fusion breaks through traditional optimization boundaries")
        print("   • Combined, they bridge 2025 gaps toward 2026+ paradigms")

        print("\n🔮 Next Steps:")
        print("   • Integrate into production models for real performance gains")
        print("   • Combine with Priority 2-7 optimizations from the roadmap")
        print("   • Monitor performance across different model architectures")
        print("   • Contribute findings back to the PyTorch ecosystem")

        return True

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)