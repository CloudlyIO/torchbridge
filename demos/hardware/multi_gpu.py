#!/usr/bin/env python3
"""
🔧 Hardware Abstraction Demo

Demonstrates multi-vendor GPU concepts and hardware optimization strategies:
- Hardware detection and capability analysis
- Vendor-specific optimization patterns
- Cross-platform performance comparison
- Production deployment considerations

Expected learning: Understanding hardware abstraction for cross-platform optimization
Hardware: Works on all platforms with automatic adaptation
Runtime: 2-3 minutes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import time
import argparse
import platform
from typing import Dict, List, Tuple, Optional


class TestModel(nn.Module):
    """Test model for hardware optimization demonstration."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        ])
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


def detect_hardware_info():
    """Detect and display comprehensive hardware information."""
    print(f"\n🔍 Hardware Detection and Capabilities")
    print("-" * 50)

    # System information
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {platform.python_version()}")

    # CPU information
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        print(f"CPU: {cpu_count} cores")
        if cpu_freq:
            print(f"     {cpu_freq.current:.0f}MHz (max: {cpu_freq.max:.0f}MHz)")
    except ImportError:
        print(f"CPU: {os.cpu_count()} cores")

    # CUDA/GPU information
    print(f"\nGPU Information:")
    if torch.cuda.is_available():
        print(f"  CUDA Available: Yes")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Device Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vendor = detect_gpu_vendor(props.name)
            print(f"  GPU {i}: {props.name} ({vendor})")
            print(f"    Memory: {props.total_memory // 1024**3:.1f}GB")
            print(f"    Compute: {props.major}.{props.minor}")
            print(f"    Multiprocessors: {props.multi_processor_count}")
    else:
        print(f"  CUDA Available: No")

    # PyTorch backend information
    print(f"\nPyTorch Backend Information:")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  MPS Available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
    print(f"  MKLDNN: {torch.backends.mkldnn.is_available()}")


def detect_gpu_vendor(gpu_name: str) -> str:
    """Detect GPU vendor from device name."""
    gpu_name_lower = gpu_name.lower()

    if any(keyword in gpu_name_lower for keyword in ['nvidia', 'geforce', 'tesla', 'quadro', 'rtx', 'gtx']):
        return 'NVIDIA'
    elif any(keyword in gpu_name_lower for keyword in ['amd', 'radeon', 'rx', 'vega']):
        return 'AMD'
    elif any(keyword in gpu_name_lower for keyword in ['intel', 'arc', 'xe']):
        return 'Intel'
    else:
        return 'Unknown'


def benchmark_on_device(device: torch.device, config: Dict, optimization_level: str = "baseline") -> Dict:
    """Benchmark model performance on specific device with different optimization levels."""

    input_size = config['input_size']
    hidden_size = config['hidden_size']
    output_size = config['output_size']
    batch_size = config['batch_size']
    seq_len = config['seq_len']

    # Create model
    model = TestModel(input_size, hidden_size, output_size).to(device)

    # Apply optimization based on level
    if optimization_level == "torch_compile" and device.type == 'cuda':
        try:
            model = torch.compile(model, mode='default')
            print(f"    ✅ torch.compile optimization applied")
        except Exception as e:
            print(f"    ⚠️ torch.compile failed: {e}")
            optimization_level = "baseline"

    # Create test data
    inputs = torch.randn(batch_size, seq_len, input_size, device=device)

    model.eval()

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(inputs)

    # Synchronize for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    memory_usage = []

    trials = 20
    for _ in range(trials):
        # Memory measurement
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()
        with torch.no_grad():
            output = model(inputs)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        times.append((end_time - start_time) * 1000)  # Convert to ms

        if device.type == 'cuda':
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB

    # Calculate statistics
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    mean_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0

    return {
        'device': str(device),
        'optimization': optimization_level,
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'mean_memory_mb': mean_memory,
        'output_shape': output.shape
    }


def compare_hardware_performance(config: Dict):
    """Compare performance across different hardware and optimization levels."""
    print(f"\n🏁 Multi-Hardware Performance Comparison")
    print("-" * 50)

    results = []

    # Test CPU
    cpu_device = torch.device('cpu')
    print(f"\nTesting CPU...")
    try:
        cpu_result = benchmark_on_device(cpu_device, config, "baseline")
        results.append(cpu_result)
        print(f"  Baseline: {cpu_result['mean_time_ms']:.1f}ms")

    except Exception as e:
        print(f"  ❌ CPU test failed: {e}")

    # Test GPU(s) with different optimization levels
    if torch.cuda.is_available():
        for gpu_id in range(min(torch.cuda.device_count(), 2)):  # Test up to 2 GPUs
            gpu_device = torch.device(f'cuda:{gpu_id}')
            gpu_name = torch.cuda.get_device_name(gpu_id)
            vendor = detect_gpu_vendor(gpu_name)
            print(f"\nTesting GPU {gpu_id}: {gpu_name} ({vendor})")

            try:
                # Baseline performance
                gpu_result = benchmark_on_device(gpu_device, config, "baseline")
                results.append(gpu_result)
                print(f"  Baseline: {gpu_result['mean_time_ms']:.1f}ms, {gpu_result['mean_memory_mb']:.1f}MB")

                # Optimized performance
                gpu_opt_result = benchmark_on_device(gpu_device, config, "torch_compile")
                results.append(gpu_opt_result)
                print(f"  Optimized: {gpu_opt_result['mean_time_ms']:.1f}ms")

            except Exception as e:
                print(f"  ❌ GPU {gpu_id} test failed: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Print comparison table
    if results:
        print(f"\n📊 Performance Summary:")
        print(f"{'Device':<20} {'Optimization':<12} {'Time (ms)':<12} {'Speedup':<10} {'Memory (MB)'}")
        print("-" * 75)

        # Find CPU baseline for speedup calculation
        cpu_baseline_time = None
        for result in results:
            if 'cpu' in result['device'] and result['optimization'] == 'baseline':
                cpu_baseline_time = result['mean_time_ms']
                break

        if cpu_baseline_time is None and results:
            cpu_baseline_time = results[0]['mean_time_ms']

        for result in results:
            device_short = result['device'].replace('cuda:', 'GPU ')
            speedup = cpu_baseline_time / result['mean_time_ms'] if cpu_baseline_time else 1.0

            print(f"{device_short:<20} {result['optimization']:<12} {result['mean_time_ms']:.2f} ± {result['std_time_ms']:.1f}   {speedup:.2f}x      {result['mean_memory_mb']:.1f}")

    return results


def explain_vendor_optimizations():
    """Explain vendor-specific optimization strategies."""
    print(f"\n🏭 Vendor-Specific Optimization Strategies")
    print("-" * 50)

    vendor_optimizations = {
        "NVIDIA": {
            "hardware_features": [
                "Tensor Cores (Mixed Precision)",
                "NVLink for Multi-GPU",
                "CUDA Graphs for Inference"
            ],
            "software_stack": [
                "CUDA Toolkit",
                "cuDNN for Neural Networks",
                "TensorRT for Inference"
            ],
            "pytorch_optimizations": [
                "torch.compile with inductor backend",
                "Flash Attention support",
                "FP8 training on H100+"
            ]
        },
        "AMD": {
            "hardware_features": [
                "Matrix Core Units",
                "Infinity Fabric",
                "Large VRAM capacity"
            ],
            "software_stack": [
                "ROCm Platform",
                "MIOpen library",
                "ROCm PyTorch"
            ],
            "pytorch_optimizations": [
                "torch.compile with ROCm",
                "Mixed precision training",
                "Memory optimization patterns"
            ]
        },
        "Intel": {
            "hardware_features": [
                "XMX Matrix Engines",
                "Xe-Link for Scale-out",
                "Unified Memory Architecture"
            ],
            "software_stack": [
                "Intel oneAPI",
                "Intel Extension for PyTorch",
                "Intel Neural Compressor"
            ],
            "pytorch_optimizations": [
                "XPU device optimization",
                "Auto-mixed precision",
                "Graph optimization"
            ]
        }
    }

    for vendor, optimizations in vendor_optimizations.items():
        print(f"\n{vendor}:")
        print(f"  Hardware Features:")
        for feature in optimizations["hardware_features"]:
            print(f"    • {feature}")
        print(f"  Software Stack:")
        for software in optimizations["software_stack"]:
            print(f"    • {software}")
        print(f"  PyTorch Optimizations:")
        for opt in optimizations["pytorch_optimizations"]:
            print(f"    • {opt}")


def explain_cross_platform_deployment():
    """Explain cross-platform deployment strategies."""
    print(f"\n🌍 Cross-Platform Deployment Strategies")
    print("-" * 45)

    deployment_strategies = {
        "Cloud Deployment": [
            "AWS: EC2 with NVIDIA/AMD instances",
            "Azure: GPU-optimized virtual machines",
            "GCP: A100/H100 compute instances",
            "Auto-scaling based on demand"
        ],
        "Edge Deployment": [
            "CPU-optimized models with quantization",
            "Intel Neural Compressor integration",
            "ARM processor optimization",
            "Power-efficient inference patterns"
        ],
        "Multi-Vendor Strategy": [
            "Hardware abstraction layer (HAL)",
            "Automatic vendor detection",
            "Optimization selection based on capabilities",
            "Fallback mechanisms for unsupported features"
        ],
        "Performance Monitoring": [
            "Hardware utilization tracking",
            "Performance regression detection",
            "Cost optimization metrics",
            "Multi-vendor benchmarking"
        ]
    }

    for strategy, details in deployment_strategies.items():
        print(f"\n{strategy}:")
        for detail in details:
            print(f"  • {detail}")

    print(f"\n💡 Best Practices:")
    print(f"  • Test on target hardware during development")
    print(f"  • Use hardware-agnostic optimization where possible")
    print(f"  • Implement graceful degradation for unsupported features")
    print(f"  • Monitor performance across different hardware configurations")


def main():
    parser = argparse.ArgumentParser(description='Hardware Abstraction Demo')
    parser.add_argument('--quick', action='store_true', help='Run quick test with small config')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Force device selection')
    args = parser.parse_args()

    print("🔧 Hardware Abstraction Demo")
    print("=" * 60)
    print("Understanding multi-vendor GPU support and optimization strategies\n")

    # Configuration
    if args.quick:
        config = {
            'input_size': 128,
            'hidden_size': 256,
            'output_size': 64,
            'batch_size': 4,
            'seq_len': 64
        }
        print("🏃‍♂️ Quick test mode")
    else:
        config = {
            'input_size': 512,
            'hidden_size': 1024,
            'output_size': 256,
            'batch_size': 8,
            'seq_len': 128
        }
        print("🏋️‍♂️ Full analysis mode")

    # Run demonstrations
    detect_hardware_info()
    results = compare_hardware_performance(config)
    explain_vendor_optimizations()
    explain_cross_platform_deployment()

    print(f"\n🎉 Hardware Abstraction Demo Completed!")
    print(f"\n💡 Key Takeaways:")
    print(f"   • Hardware abstraction enables cross-vendor optimization")
    print(f"   • Each GPU vendor has unique optimization opportunities")
    print(f"   • torch.compile provides significant speedups across vendors")
    print(f"   • Production deployment requires vendor-specific strategies")
    print(f"   • Monitoring and fallback mechanisms ensure reliability")
    print(f"   • Cost optimization through multi-vendor deployment")

    print(f"\n✅ Demo completed! Try --quick for faster testing.")


if __name__ == "__main__":
    main()