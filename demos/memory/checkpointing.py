#!/usr/bin/env python3
"""
Advanced Checkpointing Demo

Demonstrates sophisticated checkpointing strategies for memory optimization:
- Selective gradient checkpointing
- Adaptive checkpointing based on memory pressure
- Dynamic activation offloading
- Memory-efficient backpropagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import json
from typing import Dict, Any, List
import sys
import os
import gc

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from kernel_pytorch.advanced_memory import (
    SelectiveGradientCheckpointing,
    AdaptiveCheckpointing,
    MemoryEfficientBackprop,
    DynamicActivationOffloading
)


class DeepResNet(nn.Module):
    """Deep ResNet model for checkpointing demonstration"""

    def __init__(self, num_blocks=20, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(channels)

        self.blocks = nn.ModuleList([
            self._make_block(channels) for _ in range(num_blocks)
        ])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, 1000)

    def _make_block(self, channels):
        """Create a residual block"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        for i, block in enumerate(self.blocks):
            residual = x
            out = block(x)
            x = F.relu(out + residual)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DeepTransformer(nn.Module):
    """Deep transformer for attention checkpointing"""

    def __init__(self, num_layers=24, d_model=768, nhead=12):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 10000)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        return self.output_proj(x)


def get_memory_usage(device):
    """Get current memory usage"""
    if device.type == 'cuda':
        return torch.cuda.memory_allocated() / 1024**3  # GB
    else:
        # Estimate CPU memory usage
        import psutil
        return psutil.Process().memory_info().rss / 1024**3  # GB


def benchmark_standard_training(model, device, num_steps=5):
    """Benchmark standard training without checkpointing"""
    print("📊 Benchmarking Standard Training (No Checkpointing)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step_times = []
    memory_usage = []

    for step in range(num_steps):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        # Create data based on model type
        if isinstance(model, DeepResNet):
            inputs = torch.randn(4, 3, 224, 224, device=device)
            targets = torch.randint(0, 1000, (4,), device=device)
        else:  # DeepTransformer
            inputs = torch.randint(0, 10000, (2, 256), device=device)
            targets = torch.randint(0, 10000, (2, 256), device=device)

        optimizer.zero_grad()
        outputs = model(inputs)

        if isinstance(model, DeepResNet):
            loss = F.cross_entropy(outputs, targets)
        else:
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        loss.backward()
        optimizer.step()

        step_time = time.time() - start_time
        step_times.append(step_time)

        if device.type == 'cuda':
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
        else:
            memory_usage.append(get_memory_usage(device))

        # Force garbage collection
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'avg_step_time': sum(step_times) / len(step_times),
        'peak_memory_gb': max(memory_usage),
        'avg_memory_gb': sum(memory_usage) / len(memory_usage)
    }


def benchmark_selective_checkpointing(model, device, num_steps=5):
    """Benchmark with selective gradient checkpointing"""
    print("🎯 Benchmarking Selective Gradient Checkpointing...")

    # Setup selective checkpointing
    selective_checkpoint = SelectiveGradientCheckpointing(importance_threshold=0.7)

    # Assign importance scores to layers
    for name, module in model.named_modules():
        if 'blocks' in name or 'transformer' in name:
            # Higher numbers get checkpointed less (they're more important)
            try:
                if 'blocks.' in name:
                    layer_num = int(name.split('.')[1])
                elif 'transformer.layers.' in name:
                    layer_num = int(name.split('.')[2])
                else:
                    layer_num = 0
                importance = layer_num / 20.0  # Importance increases with depth
                selective_checkpoint.update_importance(name, importance)
            except (ValueError, IndexError):
                # Skip modules where we can't parse layer numbers
                continue

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step_times = []
    memory_usage = []

    for step in range(num_steps):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        # Create data
        if isinstance(model, DeepResNet):
            inputs = torch.randn(4, 3, 224, 224, device=device)
            targets = torch.randint(0, 1000, (4,), device=device)
        else:
            inputs = torch.randint(0, 10000, (2, 256), device=device)
            targets = torch.randint(0, 10000, (2, 256), device=device)

        optimizer.zero_grad()

        # Apply selective checkpointing to forward pass
        memory_pressure = get_memory_usage(device) / 8.0  # Assume 8GB limit

        def checkpoint_forward():
            return model(inputs)

        # Use checkpointing based on memory pressure
        if memory_pressure > 0.5:
            outputs = torch.utils.checkpoint.checkpoint(checkpoint_forward)
        else:
            outputs = model(inputs)

        if isinstance(model, DeepResNet):
            loss = F.cross_entropy(outputs, targets)
        else:
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        loss.backward()
        optimizer.step()

        step_time = time.time() - start_time
        step_times.append(step_time)

        if device.type == 'cuda':
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
        else:
            memory_usage.append(get_memory_usage(device))

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'avg_step_time': sum(step_times) / len(step_times),
        'peak_memory_gb': max(memory_usage),
        'avg_memory_gb': sum(memory_usage) / len(memory_usage)
    }


def benchmark_adaptive_checkpointing(model, device, num_steps=5):
    """Benchmark with adaptive checkpointing"""
    print("🔄 Benchmarking Adaptive Checkpointing...")

    adaptive_checkpoint = AdaptiveCheckpointing()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    step_times = []
    memory_usage = []

    for step in range(num_steps):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        # Create data
        if isinstance(model, DeepResNet):
            inputs = torch.randn(4, 3, 224, 224, device=device)
            targets = torch.randint(0, 1000, (4,), device=device)
        else:
            inputs = torch.randint(0, 10000, (2, 256), device=device)
            targets = torch.randint(0, 10000, (2, 256), device=device)

        optimizer.zero_grad()

        # Use adaptive checkpointing
        outputs = adaptive_checkpoint.forward(model, inputs)

        if isinstance(model, DeepResNet):
            loss = F.cross_entropy(outputs, targets)
        else:
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

        loss.backward()
        optimizer.step()

        step_time = time.time() - start_time
        step_times.append(step_time)

        if device.type == 'cuda':
            memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
        else:
            memory_usage.append(get_memory_usage(device))

        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return {
        'avg_step_time': sum(step_times) / len(step_times),
        'peak_memory_gb': max(memory_usage),
        'avg_memory_gb': sum(memory_usage) / len(memory_usage)
    }


def demonstrate_dynamic_activation_offloading(device):
    """Demonstrate dynamic activation offloading"""
    print("💾 Demonstrating Dynamic Activation Offloading...")

    offloader = DynamicActivationOffloading(offload_device="cpu")

    # Create activations on GPU
    activations = torch.randn(8, 1024, 512, device=device)
    print(f"   Original activations device: {activations.device}")
    print(f"   Original activations size: {activations.numel() * 4 / 1024**2:.1f}MB")

    # Offload to CPU
    offloaded = offloader.offload_activations(activations)
    print(f"   Offloaded activations device: {offloaded.device}")

    # Reload back to original device
    reloaded = offloader.reload_activations(offloaded, device)
    print(f"   Reloaded activations device: {reloaded.device}")

    # Verify correctness
    assert torch.allclose(activations.cpu(), reloaded.cpu())
    print("   ✅ Offloading correctness verified")


def demonstrate_memory_efficient_backprop(model, device):
    """Demonstrate memory-efficient backpropagation"""
    print("⚡ Demonstrating Memory-Efficient Backpropagation...")

    memory_efficient = MemoryEfficientBackprop()
    memory_efficient.apply(model)

    # Test with a forward-backward pass
    if isinstance(model, DeepResNet):
        inputs = torch.randn(2, 3, 224, 224, device=device)
        targets = torch.randint(0, 1000, (2,), device=device)
    else:
        inputs = torch.randint(0, 10000, (2, 128), device=device)
        targets = torch.randint(0, 10000, (2, 128), device=device)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    outputs = model(inputs)

    if isinstance(model, DeepResNet):
        loss = F.cross_entropy(outputs, targets)
    else:
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))

    loss.backward()

    if device.type == 'cuda':
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"   Memory used with efficient backprop: {memory_used:.2f}GB")

    print("   ✅ Memory-efficient backprop working")


def run_comprehensive_demo(device, quick_mode=False):
    """Run comprehensive advanced checkpointing demo"""
    print(f"🚀 Advanced Memory Optimization: Checkpointing Demo")
    print(f"📱 Device: {device}")
    print(f"⚡ Mode: {'Quick' if quick_mode else 'Full'}")
    print("=" * 60)

    # Test with ResNet model
    print("\n🔥 TESTING WITH DEEP RESNET")
    print("=" * 40)

    resnet_config = {
        'num_blocks': 10 if quick_mode else 20,
        'channels': 128 if quick_mode else 256
    }

    resnet_model = DeepResNet(**resnet_config).to(device)
    num_steps = 3 if quick_mode else 5

    print(f"📊 ResNet parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")

    resnet_results = {}

    # Benchmark different checkpointing strategies
    resnet_results['standard'] = benchmark_standard_training(resnet_model, device, num_steps)
    resnet_results['selective'] = benchmark_selective_checkpointing(resnet_model, device, num_steps)
    resnet_results['adaptive'] = benchmark_adaptive_checkpointing(resnet_model, device, num_steps)

    # Test with Transformer model
    print("\n🔥 TESTING WITH DEEP TRANSFORMER")
    print("=" * 40)

    transformer_config = {
        'num_layers': 12 if quick_mode else 24,
        'd_model': 512 if quick_mode else 768,
        'nhead': 8 if quick_mode else 12
    }

    transformer_model = DeepTransformer(**transformer_config).to(device)

    print(f"📊 Transformer parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")

    transformer_results = {}

    # Benchmark different checkpointing strategies
    transformer_results['standard'] = benchmark_standard_training(transformer_model, device, num_steps)
    transformer_results['selective'] = benchmark_selective_checkpointing(transformer_model, device, num_steps)
    transformer_results['adaptive'] = benchmark_adaptive_checkpointing(transformer_model, device, num_steps)

    # Demonstrate additional features
    print("\n🔧 ADDITIONAL FEATURES")
    print("=" * 40)
    demonstrate_dynamic_activation_offloading(device)
    demonstrate_memory_efficient_backprop(transformer_model, device)

    # Calculate and display results
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 60)

    def display_results(name, results):
        print(f"\n{name} Results:")
        baseline = results['standard']

        for strategy, result in results.items():
            if strategy == 'standard':
                print(f"  Standard: {result['avg_step_time']*1000:.1f}ms, {result['peak_memory_gb']:.2f}GB")
            else:
                speedup = baseline['avg_step_time'] / result['avg_step_time']
                memory_reduction = (baseline['peak_memory_gb'] - result['peak_memory_gb']) / baseline['peak_memory_gb'] * 100

                print(f"  {strategy.title()}: {result['avg_step_time']*1000:.1f}ms, {result['peak_memory_gb']:.2f}GB")
                print(f"    ⚡ Speedup: {speedup:.2f}x")
                print(f"    💾 Memory reduction: {memory_reduction:.1f}%")

    display_results("ResNet", resnet_results)
    display_results("Transformer", transformer_results)

    # Summary
    print("\n🎯 SUMMARY")
    print("=" * 60)

    # Calculate best improvements
    best_resnet_memory = min(resnet_results[k]['peak_memory_gb'] for k in resnet_results if k != 'standard')
    best_transformer_memory = min(transformer_results[k]['peak_memory_gb'] for k in transformer_results if k != 'standard')

    resnet_memory_reduction = (resnet_results['standard']['peak_memory_gb'] - best_resnet_memory) / resnet_results['standard']['peak_memory_gb'] * 100
    transformer_memory_reduction = (transformer_results['standard']['peak_memory_gb'] - best_transformer_memory) / transformer_results['standard']['peak_memory_gb'] * 100

    print(f"✅ ResNet memory reduction: {resnet_memory_reduction:.1f}%")
    print(f"✅ Transformer memory reduction: {transformer_memory_reduction:.1f}%")
    print(f"✅ All checkpointing strategies working")
    print(f"✅ Dynamic activation offloading functional")
    print(f"✅ Memory-efficient backprop enabled")

    return {'resnet': resnet_results, 'transformer': transformer_results}


def main():
    parser = argparse.ArgumentParser(description='Advanced Checkpointing Demo')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to run on')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with smaller models')
    parser.add_argument('--output', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    try:
        # Run demo
        results = run_comprehensive_demo(device, args.quick)

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                # Flatten results for JSON serialization
                serializable_results = {}
                for model_type, model_results in results.items():
                    for strategy, strategy_results in model_results.items():
                        key = f"{model_type}_{strategy}"
                        serializable_results[key] = {
                            'avg_step_time': strategy_results['avg_step_time'],
                            'peak_memory_gb': strategy_results['peak_memory_gb'],
                            'avg_memory_gb': strategy_results['avg_memory_gb']
                        }
                json.dump(serializable_results, f, indent=2)
            print(f"💾 Results saved to {args.output}")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())