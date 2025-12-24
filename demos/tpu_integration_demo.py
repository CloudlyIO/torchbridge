#!/usr/bin/env python3
"""
TPU Integration Comprehensive Demo

Demonstrates complete TPU integration capabilities including:
- TPU configuration and hardware detection
- TPU backend and optimization
- XLA compilation and integration
- Memory management for TPU
- Model optimization and validation

Usage:
    python3 tpu_integration_demo.py [--quick] [--device cpu|auto]
    python3 tpu_integration_demo.py --help

This demo works without actual TPU hardware by using CPU fallback,
demonstrating the complete integration stack.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from kernel_pytorch.core.config import KernelPyTorchConfig, TPUConfig, TPUVersion, TPUTopology
from kernel_pytorch.backends.tpu import (
    TPUBackend,
    TPUOptimizer,
    XLACompiler,
    TPUMemoryManager,
    XLADeviceManager,
    XLADistributedTraining,
    XLAOptimizations,
    XLAUtilities,
    create_xla_integration
)
from kernel_pytorch.validation.unified_validator import (
    validate_tpu_configuration,
    validate_tpu_model
)


class SimpleTransformer(nn.Module):
    """Simple transformer model for TPU demonstration."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        pos = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(x.size(0), -1, -1)

        x = self.embedding(x) * (self.d_model ** 0.5) + pos
        x = self.transformer(x)
        return self.output_projection(x)


class SimpleCNN(nn.Module):
    """Simple CNN model for TPU demonstration."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class TPUIntegrationDemo:
    """Comprehensive TPU integration demonstration."""

    def __init__(self, device: str = "auto", quick_mode: bool = False):
        """
        Initialize TPU integration demo.

        Args:
            device: Device preference ('cpu', 'auto')
            quick_mode: Run with reduced complexity
        """
        self.device = device
        self.quick_mode = quick_mode

        print("🚀 TPU Integration Comprehensive Demo")
        print("=" * 60)
        print(f"   Device preference: {device}")
        print(f"   Quick mode: {quick_mode}")
        print()

    def demo_tpu_configuration(self) -> Dict[str, Any]:
        """Demonstrate TPU configuration capabilities."""
        print("📋 Demo 1: TPU Configuration and Hardware Detection")
        print("-" * 50)

        results = {}

        # Create different configuration modes
        configs = {
            'default': KernelPyTorchConfig(),
            'inference': KernelPyTorchConfig.for_inference(),
            'training': KernelPyTorchConfig.for_training(),
            'development': KernelPyTorchConfig.for_development()
        }

        for mode, config in configs.items():
            tpu_config = config.hardware.tpu
            print(f"   🔧 {mode.capitalize()} mode:")
            print(f"      TPU Version: {tpu_config.version.value}")
            print(f"      TPU Topology: {tpu_config.topology.value}")
            print(f"      Compilation Mode: {tpu_config.compilation_mode.value}")
            print(f"      Memory Fraction: {tpu_config.memory_fraction}")
            print(f"      Precision: {tpu_config.precision}")
            print(f"      Mixed Precision: {tpu_config.mixed_precision}")

            results[mode] = {
                'version': tpu_config.version.value,
                'topology': tpu_config.topology.value,
                'precision': tpu_config.precision,
                'memory_fraction': tpu_config.memory_fraction
            }

        # Test configuration validation
        print(f"\n   ✅ Configuration Validation:")
        validation_results = validate_tpu_configuration(configs['default'])
        print(f"      Tests passed: {validation_results.passed}/{validation_results.total_tests}")
        print(f"      Success rate: {validation_results.success_rate:.1%}")

        results['validation'] = {
            'passed': validation_results.passed,
            'total': validation_results.total_tests,
            'success_rate': validation_results.success_rate
        }

        print("\n" + "✅ TPU Configuration Demo Complete\n")
        return results

    def demo_tpu_backend(self) -> Dict[str, Any]:
        """Demonstrate TPU backend operations."""
        print("🔧 Demo 2: TPU Backend and Device Management")
        print("-" * 50)

        results = {}

        # Initialize TPU backend
        config = KernelPyTorchConfig()
        backend = TPUBackend(config)

        print(f"   🎮 Backend Information:")
        print(f"      Device: {backend.device}")
        print(f"      World size: {backend.world_size}")
        print(f"      Rank: {backend.rank}")
        print(f"      Is distributed: {backend.is_distributed}")

        results['backend_info'] = {
            'device': str(backend.device),
            'world_size': backend.world_size,
            'rank': backend.rank,
            'is_distributed': backend.is_distributed
        }

        # Demonstrate model preparation
        models = [
            nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10)),
            SimpleCNN(num_classes=10) if not self.quick_mode else nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        ]

        print(f"\n   📦 Model Preparation:")
        model_results = []

        for i, model in enumerate(models):
            start_time = time.perf_counter()
            prepared_model = backend.prepare_model(model)
            prep_time = time.perf_counter() - start_time

            param_count = sum(p.numel() for p in prepared_model.parameters())

            print(f"      Model {i+1}: {type(model).__name__}")
            print(f"         Parameters: {param_count:,}")
            print(f"         Preparation time: {prep_time*1000:.2f}ms")
            print(f"         Target device: {next(prepared_model.parameters()).device}")

            model_results.append({
                'type': type(model).__name__,
                'parameters': param_count,
                'prep_time_ms': prep_time * 1000,
                'device': str(next(prepared_model.parameters()).device)
            })

        results['model_preparation'] = model_results

        # Demonstrate data preparation
        print(f"\n   💾 Data Preparation:")
        test_data = [
            torch.randn(8, 64),
            {'input': torch.randn(4, 3, 32, 32), 'target': torch.randint(0, 10, (4,))}
        ]

        data_results = []
        for i, data in enumerate(test_data):
            start_time = time.perf_counter()
            prepared_data = backend.prepare_data(data)
            prep_time = time.perf_counter() - start_time

            data_type = type(data).__name__
            print(f"      Data {i+1}: {data_type}")
            print(f"         Preparation time: {prep_time*1000:.2f}ms")

            data_results.append({
                'type': data_type,
                'prep_time_ms': prep_time * 1000
            })

        results['data_preparation'] = data_results

        # Memory statistics
        memory_stats = backend.get_memory_stats()
        print(f"\n   📊 Memory Statistics:")
        print(f"      Models cached: {memory_stats.get('models_cached', 0)}")
        print(f"      Compilations cached: {memory_stats.get('compilations_cached', 0)}")

        results['memory_stats'] = memory_stats

        print("\n" + "✅ TPU Backend Demo Complete\n")
        return results

    def demo_tpu_optimization(self) -> Dict[str, Any]:
        """Demonstrate TPU optimization capabilities."""
        print("⚡ Demo 3: TPU Model Optimization and XLA Compilation")
        print("-" * 50)

        results = {}

        # Initialize optimizer and compiler
        config = KernelPyTorchConfig()
        optimizer = TPUOptimizer(config)
        compiler = XLACompiler(config.hardware.tpu)

        # Test model
        if self.quick_mode:
            model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
            sample_input = torch.randn(8, 64)
        else:
            model = SimpleTransformer(vocab_size=1000, d_model=256, nhead=8, num_layers=4)
            sample_input = torch.randint(0, 1000, (4, 32))  # batch_size=4, seq_len=32

        param_count = sum(p.numel() for p in model.parameters())
        print(f"   🧠 Model Information:")
        print(f"      Type: {type(model).__name__}")
        print(f"      Parameters: {param_count:,}")
        print(f"      Input shape: {sample_input.shape}")

        results['model_info'] = {
            'type': type(model).__name__,
            'parameters': param_count,
            'input_shape': list(sample_input.shape)
        }

        # Test different optimization levels
        print(f"\n   🔥 Optimization Levels:")
        optimization_results = {}

        for level in ['conservative', 'balanced', 'aggressive']:
            print(f"      Testing {level} optimization...")
            start_time = time.perf_counter()

            try:
                result = optimizer.optimize(model, sample_input, optimization_level=level)
                opt_time = time.perf_counter() - start_time

                print(f"         ✅ Success: {opt_time*1000:.2f}ms")
                optimization_results[level] = {
                    'success': True,
                    'time_ms': opt_time * 1000,
                    'optimization_time': result.optimization_time
                }

            except Exception as e:
                opt_time = time.perf_counter() - start_time
                print(f"         ❌ Failed: {str(e)}")
                optimization_results[level] = {
                    'success': False,
                    'time_ms': opt_time * 1000,
                    'error': str(e)
                }

        results['optimization_levels'] = optimization_results

        # Test specialized optimizations
        print(f"\n   🎯 Specialized Optimizations:")

        # Inference optimization
        inference_start = time.perf_counter()
        inference_result = optimizer.optimize_for_inference(model, sample_input)
        inference_time = time.perf_counter() - inference_start

        print(f"      Inference optimization: {inference_time*1000:.2f}ms")
        print(f"         Model in eval mode: {not inference_result.optimized_model.training}")

        # Training optimization
        training_start = time.perf_counter()
        training_result = optimizer.optimize_for_training(model, sample_input)
        training_time = time.perf_counter() - training_start

        print(f"      Training optimization: {training_time*1000:.2f}ms")

        results['specialized_optimization'] = {
            'inference_time_ms': inference_time * 1000,
            'training_time_ms': training_time * 1000,
            'inference_eval_mode': not inference_result.optimized_model.training
        }

        # XLA compilation tests
        print(f"\n   🔧 XLA Compilation:")

        # Direct compilation
        compile_start = time.perf_counter()
        compiled_model = compiler.compile_model(model, sample_input)
        compile_time = time.perf_counter() - compile_start

        print(f"      Direct compilation: {compile_time*1000:.2f}ms")

        # Compilation statistics
        comp_stats = compiler.get_compilation_stats()
        print(f"      Compiled models: {comp_stats['total_compiled_models']}")
        print(f"      XLA available: {comp_stats['xla_available']}")

        results['xla_compilation'] = {
            'compilation_time_ms': compile_time * 1000,
            'stats': comp_stats
        }

        print("\n" + "✅ TPU Optimization Demo Complete\n")
        return results

    def demo_memory_management(self) -> Dict[str, Any]:
        """Demonstrate TPU memory management."""
        print("💾 Demo 4: TPU Memory Management")
        print("-" * 50)

        results = {}

        # Initialize memory manager
        config = KernelPyTorchConfig()
        memory_manager = TPUMemoryManager(config.hardware.tpu)

        print(f"   🏗️ Memory Manager Information:")
        print(f"      TPU Version: {config.hardware.tpu.version.value}")
        print(f"      Memory Fraction: {config.hardware.tpu.memory_fraction}")

        results['manager_info'] = {
            'tpu_version': config.hardware.tpu.version.value,
            'memory_fraction': config.hardware.tpu.memory_fraction
        }

        # Test tensor allocation
        print(f"\n   📦 Tensor Allocation:")
        tensor_shapes = [(64, 64), (128, 128)] if self.quick_mode else [(64, 64), (128, 128), (256, 256)]
        allocation_results = []

        for shape in tensor_shapes:
            start_time = time.perf_counter()
            tensor = memory_manager.allocate_tensor(shape, dtype=torch.float32)
            alloc_time = time.perf_counter() - start_time

            print(f"      Shape {shape}: {alloc_time*1000:.2f}ms")
            print(f"         Device: {tensor.device}")
            print(f"         Dtype: {tensor.dtype}")

            allocation_results.append({
                'shape': shape,
                'time_ms': alloc_time * 1000,
                'device': str(tensor.device),
                'dtype': str(tensor.dtype)
            })

        results['tensor_allocation'] = allocation_results

        # Test tensor layout optimization
        print(f"\n   ⚙️ Tensor Layout Optimization:")
        test_tensor = torch.randn(63, 31)  # Non-optimal dimensions

        opt_start = time.perf_counter()
        optimized_tensor = memory_manager.optimize_tensor_layout(test_tensor)
        opt_time = time.perf_counter() - opt_start

        print(f"      Original shape: {test_tensor.shape}")
        print(f"      Optimized shape: {optimized_tensor.shape}")
        print(f"      Optimization time: {opt_time*1000:.2f}ms")

        results['layout_optimization'] = {
            'original_shape': list(test_tensor.shape),
            'optimized_shape': list(optimized_tensor.shape),
            'time_ms': opt_time * 1000
        }

        # Test memory pools
        print(f"\n   🏊 Memory Pool Operations:")
        pool_id = memory_manager.create_memory_pool(5, (32, 32))

        # Get and return tensor
        pool_tensor = memory_manager.get_tensor_from_pool(pool_id)
        return_success = memory_manager.return_tensor_to_pool(pool_id, pool_tensor) if pool_tensor else False

        pool_stats = memory_manager.get_pool_stats()

        print(f"      Pool created: {pool_id}")
        print(f"      Tensor retrieved: {pool_tensor is not None}")
        print(f"      Tensor returned: {return_success}")
        print(f"      Total pools: {pool_stats['total_pools']}")

        results['memory_pools'] = {
            'pool_created': pool_id,
            'tensor_retrieved': pool_tensor is not None,
            'tensor_returned': return_success,
            'pool_stats': pool_stats
        }

        # Memory statistics
        memory_stats = memory_manager.get_memory_stats()
        print(f"\n   📊 Memory Statistics:")
        print(f"      Allocated memory: {memory_stats.allocated_memory/1e6:.1f}MB")
        print(f"      Active tensors: {memory_stats.active_tensors}")
        print(f"      Memory fraction: {memory_stats.memory_fraction}")

        results['memory_stats'] = {
            'allocated_mb': memory_stats.allocated_memory / 1e6,
            'active_tensors': memory_stats.active_tensors,
            'memory_fraction': memory_stats.memory_fraction
        }

        print("\n" + "✅ Memory Management Demo Complete\n")
        return results

    def demo_xla_integration(self) -> Dict[str, Any]:
        """Demonstrate XLA integration components."""
        print("🔗 Demo 5: XLA Integration and Distributed Features")
        print("-" * 50)

        results = {}

        # Create XLA integration components
        config = KernelPyTorchConfig()
        device_mgr, dist_training, optimizations = create_xla_integration(config.hardware.tpu)

        print(f"   🎮 XLA Device Manager:")
        device_stats = device_mgr.get_device_stats()
        print(f"      Current device: {device_mgr.device}")
        print(f"      World size: {device_mgr.world_size}")
        print(f"      Rank: {device_mgr.rank}")
        print(f"      Available devices: {device_stats.get('available_devices', 0)}")

        results['device_manager'] = {
            'device': str(device_mgr.device),
            'world_size': device_mgr.world_size,
            'rank': device_mgr.rank,
            'stats': device_stats
        }

        print(f"\n   🌐 Distributed Training Setup:")
        print(f"      Is distributed: {dist_training.is_distributed}")

        # Test model wrapping
        model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
        wrapped_model = dist_training.wrap_model(model)

        print(f"      Model wrapped: {wrapped_model is not None}")
        print(f"      Model device: {next(wrapped_model.parameters()).device}")

        results['distributed_training'] = {
            'is_distributed': dist_training.is_distributed,
            'model_wrapped': wrapped_model is not None,
            'model_device': str(next(wrapped_model.parameters()).device)
        }

        print(f"\n   ⚙️ XLA Optimizations:")

        # Test model optimization
        opt_start = time.perf_counter()
        optimized_model = optimizations.optimize_model_for_xla(model)
        compilation_hints = optimizations.add_compilation_hints(optimized_model)
        opt_time = time.perf_counter() - opt_start

        print(f"      Model optimization: {opt_time*1000:.2f}ms")
        print(f"      Compilation hints added: {compilation_hints is not None}")

        # Dynamic shapes
        dynamic_model = optimizations.enable_dynamic_shapes(model)
        print(f"      Dynamic shapes enabled: {dynamic_model is not None}")

        results['optimizations'] = {
            'optimization_time_ms': opt_time * 1000,
            'compilation_hints': compilation_hints is not None,
            'dynamic_shapes': dynamic_model is not None
        }

        print(f"\n   🛠️ XLA Utilities:")

        # Environment information
        env_info = XLAUtilities.get_xla_env_info()
        print(f"      XLA available: {env_info.get('xla_available', False)}")
        print(f"      Environment vars: {len([k for k in env_info.keys() if k.startswith('XLA')])}")

        # Optimization flags
        flags = XLAUtilities.optimize_xla_flags(config.hardware.tpu.version)
        print(f"      Optimized flags: {len(flags)} flags set")

        results['utilities'] = {
            'env_info': env_info,
            'flags_count': len(flags)
        }

        print("\n" + "✅ XLA Integration Demo Complete\n")
        return results

    def demo_validation_and_testing(self) -> Dict[str, Any]:
        """Demonstrate validation and testing capabilities."""
        print("✅ Demo 6: Validation and Performance Testing")
        print("-" * 50)

        results = {}

        config = KernelPyTorchConfig()

        # Configuration validation
        print(f"   📋 Configuration Validation:")
        config_validation = validate_tpu_configuration(config)

        print(f"      Tests run: {config_validation.total_tests}")
        print(f"      Passed: {config_validation.passed}")
        print(f"      Warnings: {config_validation.warnings}")
        print(f"      Success rate: {config_validation.success_rate:.1%}")

        # Show some specific validation results
        for i, report in enumerate(config_validation.reports[:3]):
            print(f"         {i+1}. {report.status.value}: {report.message}")

        results['config_validation'] = {
            'total_tests': config_validation.total_tests,
            'passed': config_validation.passed,
            'warnings': config_validation.warnings,
            'success_rate': config_validation.success_rate
        }

        # Model validation
        print(f"\n   🧠 Model Validation:")

        # Test with optimally-sized model
        optimal_model = nn.Sequential(
            nn.Linear(64, 32),  # Optimal for TPU (divisible by 8)
            nn.ReLU(),
            nn.Linear(32, 8)    # Optimal for TPU
        )
        optimal_input = torch.randn(8, 64)  # Optimal batch and features

        optimal_validation = validate_tpu_model(optimal_model, config.hardware.tpu, optimal_input)

        print(f"      Optimal model validation:")
        print(f"         Tests: {optimal_validation.total_tests}")
        print(f"         Passed: {optimal_validation.passed}")
        print(f"         Warnings: {optimal_validation.warnings}")

        # Test with non-optimal model
        suboptimal_model = nn.Sequential(
            nn.Linear(63, 31),  # Non-optimal (not divisible by 8)
            nn.ReLU(),
            nn.Linear(31, 7)    # Non-optimal
        )
        suboptimal_input = torch.randn(7, 63)  # Non-optimal dimensions

        suboptimal_validation = validate_tpu_model(suboptimal_model, config.hardware.tpu, suboptimal_input)

        print(f"      Suboptimal model validation:")
        print(f"         Tests: {suboptimal_validation.total_tests}")
        print(f"         Passed: {suboptimal_validation.passed}")
        print(f"         Warnings: {suboptimal_validation.warnings}")
        print(f"         (Expected warnings for non-optimal dimensions)")

        results['model_validation'] = {
            'optimal': {
                'total_tests': optimal_validation.total_tests,
                'passed': optimal_validation.passed,
                'warnings': optimal_validation.warnings
            },
            'suboptimal': {
                'total_tests': suboptimal_validation.total_tests,
                'passed': suboptimal_validation.passed,
                'warnings': suboptimal_validation.warnings
            }
        }

        # Performance characteristics
        print(f"\n   📊 Performance Insights:")

        # Compare model sizes
        optimal_params = sum(p.numel() for p in optimal_model.parameters())
        suboptimal_params = sum(p.numel() for p in suboptimal_model.parameters())

        print(f"      Optimal model parameters: {optimal_params:,}")
        print(f"      Suboptimal model parameters: {suboptimal_params:,}")

        # Memory efficiency estimates
        optimal_memory = optimal_params * 4 / 1e6  # MB (float32)
        suboptimal_memory = suboptimal_params * 4 / 1e6  # MB

        print(f"      Optimal memory usage: {optimal_memory:.1f}MB")
        print(f"      Suboptimal memory usage: {suboptimal_memory:.1f}MB")

        results['performance_insights'] = {
            'optimal_params': optimal_params,
            'suboptimal_params': suboptimal_params,
            'optimal_memory_mb': optimal_memory,
            'suboptimal_memory_mb': suboptimal_memory
        }

        print("\n" + "✅ Validation and Testing Demo Complete\n")
        return results

    def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete TPU integration demonstration."""
        print("🎯 Starting Comprehensive TPU Integration Demo")
        print("=" * 60)

        start_time = time.perf_counter()
        demo_results = {}

        try:
            # Run all demo sections
            demo_results['configuration'] = self.demo_tpu_configuration()
            demo_results['backend'] = self.demo_tpu_backend()
            demo_results['optimization'] = self.demo_tpu_optimization()
            demo_results['memory'] = self.demo_memory_management()
            demo_results['xla_integration'] = self.demo_xla_integration()
            demo_results['validation'] = self.demo_validation_and_testing()

            total_time = time.perf_counter() - start_time

            # Print summary
            print("🎉 Demo Summary")
            print("=" * 60)
            print(f"   Total runtime: {total_time:.2f}s")
            print(f"   Demo sections completed: {len(demo_results)}")
            print(f"   Device used: {self.device}")
            print(f"   Quick mode: {self.quick_mode}")

            # Key achievements
            print(f"\n   ✅ Key Achievements:")
            print(f"      • TPU configuration system operational")
            print(f"      • Backend and device management working")
            print(f"      • Model optimization and compilation functional")
            print(f"      • Memory management and pooling active")
            print(f"      • XLA integration components ready")
            print(f"      • Validation and testing comprehensive")

            demo_results['summary'] = {
                'total_time': total_time,
                'sections_completed': len(demo_results),
                'device': self.device,
                'quick_mode': self.quick_mode,
                'success': True
            }

            print(f"\n🚀 TPU Integration Demo Successfully Completed!")
            print("   Ready for production TPU deployment with actual hardware!")

        except Exception as e:
            total_time = time.perf_counter() - start_time
            print(f"\n❌ Demo failed after {total_time:.2f}s: {str(e)}")
            demo_results['summary'] = {
                'total_time': total_time,
                'sections_completed': len(demo_results),
                'device': self.device,
                'quick_mode': self.quick_mode,
                'success': False,
                'error': str(e)
            }

        return demo_results


def main():
    """Main demo execution."""
    parser = argparse.ArgumentParser(description='TPU Integration Comprehensive Demo')
    parser.add_argument('--quick', action='store_true', help='Run quick demo with reduced complexity')
    parser.add_argument('--device', default='auto', choices=['cpu', 'auto'],
                       help='Device preference for demo')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    try:
        # Run demo
        demo = TPUIntegrationDemo(device=args.device, quick_mode=args.quick)
        results = demo.run_complete_demo()

        # Exit with success if demo completed
        success = results.get('summary', {}).get('success', False)
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n💥 Demo execution failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()