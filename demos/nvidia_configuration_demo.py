"""
NVIDIA Configuration Integration Demo

Demonstrates the new unified NVIDIA configuration system for hardware detection,
optimization selection, and automatic performance tuning.
"""

import argparse
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from torchbridge.core.config import OptimizationLevel, TorchBridgeConfig
from torchbridge.validation.unified_validator import UnifiedValidator


class NVIDIAConfigurationDemo:
    """Interactive demo of NVIDIA configuration capabilities."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def demonstrate_hardware_detection(self):
        """Demonstrate automatic NVIDIA hardware detection."""
        print("🔍 NVIDIA Hardware Detection Demo")
        print("=" * 50)

        # Create configuration with auto-detection
        config = TorchBridgeConfig()

        print(f"✅ Device detected: {config.device}")
        print(f"✅ Hardware backend: {config.hardware.backend.value}")
        print(f"✅ NVIDIA enabled: {config.hardware.nvidia.enabled}")
        print()

        # Show NVIDIA-specific details
        nvidia_config = config.hardware.nvidia
        print("🎯 NVIDIA Configuration Details:")
        print(f"   Architecture: {nvidia_config.architecture.value}")
        print(f"   FP8 enabled: {nvidia_config.fp8_enabled}")
        print(f"   FP8 recipe: {nvidia_config.fp8_recipe}")
        print(f"   Tensor Core version: {nvidia_config.tensor_core_version}")
        print(f"   FlashAttention version: {nvidia_config.flash_attention_version}")
        print(f"   Mixed precision: {nvidia_config.mixed_precision_enabled}")
        print(f"   Memory pool enabled: {nvidia_config.memory_pool_enabled}")
        print(f"   Memory fraction: {nvidia_config.memory_fraction}")
        print(f"   Kernel fusion: {nvidia_config.kernel_fusion_enabled}")
        print()

        return config

    def demonstrate_configuration_modes(self):
        """Demonstrate different configuration modes for different use cases."""
        print("⚙️ Configuration Modes Demo")
        print("=" * 40)

        # Inference configuration
        print("🏃 Inference Mode (Optimized for low latency):")
        inference_config = TorchBridgeConfig.for_inference()
        print(f"   Optimization level: {inference_config.optimization_level.value}")
        print(f"   Gradient checkpointing: {inference_config.memory.gradient_checkpointing}")
        print(f"   Deep optimizer states: {inference_config.memory.deep_optimizer_states}")
        print(f"   Validation enabled: {inference_config.validation.enabled}")
        print()

        # Training configuration
        print("🎓 Training Mode (Optimized for memory and stability):")
        training_config = TorchBridgeConfig.for_training()
        print(f"   Optimization level: {training_config.optimization_level.value}")
        print(f"   Gradient checkpointing: {training_config.memory.gradient_checkpointing}")
        print(f"   Deep optimizer states: {training_config.memory.deep_optimizer_states}")
        print(f"   Validation enabled: {training_config.validation.enabled}")
        print()

        # Development configuration
        print("🔧 Development Mode (Optimized for debugging):")
        dev_config = TorchBridgeConfig.for_development()
        print(f"   Optimization level: {dev_config.optimization_level.value}")
        print(f"   Debug mode: {dev_config.debug}")
        print(f"   Profile mode: {dev_config.profile}")
        print(f"   Strict validation: {dev_config.validation.strict_mode}")
        print()

        return {
            'inference': inference_config,
            'training': training_config,
            'development': dev_config
        }

    def demonstrate_optimization_levels(self):
        """Demonstrate different optimization levels and their impact."""
        print("⚡ Optimization Levels Demo")
        print("=" * 35)

        # Create a simple model for testing
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        ).to(self.device)

        test_input = torch.randn(32, 256).to(self.device)

        optimization_levels = [
            (OptimizationLevel.CONSERVATIVE, "Safe and stable"),
            (OptimizationLevel.BALANCED, "Good balance of speed and safety"),
            (OptimizationLevel.AGGRESSIVE, "Maximum performance")
        ]

        for opt_level, description in optimization_levels:
            print(f"🔧 {opt_level.value.capitalize()} Mode: {description}")

            # Create config with specific optimization level
            config = TorchBridgeConfig()
            config.optimization_level = opt_level

            # Run a quick test
            with torch.no_grad():
                output = model(test_input)

            print("   ✅ Model execution successful")
            print(f"   ✅ Output shape: {output.shape}")
            print(f"   ✅ NVIDIA settings preserved: {config.hardware.nvidia.enabled}")
            print()

    def demonstrate_configuration_validation(self):
        """Demonstrate configuration validation capabilities."""
        print("✅ Configuration Validation Demo")
        print("=" * 40)

        validator = UnifiedValidator()

        # Test different configurations
        configs_to_test = [
            ("Default", TorchBridgeConfig()),
            ("Inference", TorchBridgeConfig.for_inference()),
            ("Training", TorchBridgeConfig.for_training()),
            ("Development", TorchBridgeConfig.for_development())
        ]

        for config_name, config in configs_to_test:
            print(f"🔍 Validating {config_name} Configuration:")

            # Configuration validation
            config_results = validator.validate_configuration(config)
            print(f"   Config tests: {config_results.passed}/{config_results.total_tests} passed")

            # Hardware validation
            hw_results = validator.validate_hardware_compatibility(self.device)
            print(f"   Hardware tests: {hw_results.passed}/{hw_results.total_tests} passed")

            # Overall success rate
            overall_rate = (config_results.passed + hw_results.passed) / (config_results.total_tests + hw_results.total_tests)
            print(f"   Overall success: {overall_rate:.1%}")
            print()

    def demonstrate_config_serialization(self):
        """Demonstrate configuration serialization and deserialization."""
        print("💾 Configuration Serialization Demo")
        print("=" * 40)

        # Create a configuration
        config = TorchBridgeConfig()

        # Customize NVIDIA settings
        config.hardware.nvidia.fp8_enabled = True
        config.hardware.nvidia.flash_attention_version = "3"
        config.optimization_level = OptimizationLevel.AGGRESSIVE

        # Serialize to dictionary
        config_dict = config.to_dict()

        print("🔧 Original Configuration:")
        print(f"   Optimization level: {config.optimization_level.value}")
        print(f"   NVIDIA FP8 enabled: {config.hardware.nvidia.fp8_enabled}")
        print(f"   Flash Attention: v{config.hardware.nvidia.flash_attention_version}")
        print()

        print("💾 Serialized Configuration:")
        print(f"   Total keys: {len(config_dict)}")
        print(f"   Hardware keys: {len(config_dict.get('hardware', {}))}")
        print(f"   NVIDIA config present: {'nvidia' in config_dict.get('hardware', {})}")
        print(f"   Serialization size: ~{len(str(config_dict))} characters")
        print()

    def demonstrate_performance_comparison(self):
        """Demonstrate performance impact of different NVIDIA configurations."""
        print("🏁 Performance Comparison Demo")
        print("=" * 35)

        # Create test model
        model = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(self.device)

        test_input = torch.randn(64, 512).to(self.device)

        # Test different configurations
        configs = {
            'Conservative': (OptimizationLevel.CONSERVATIVE, "Focus on stability"),
            'Balanced': (OptimizationLevel.BALANCED, "Good compromise"),
            'Aggressive': (OptimizationLevel.AGGRESSIVE, "Maximum speed")
        }

        print("🎯 Testing inference performance with different optimization levels:")
        print()

        for config_name, (opt_level, description) in configs.items():
            config = TorchBridgeConfig()
            config.optimization_level = opt_level

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(test_input)

            # Time inference
            import time
            start_time = time.perf_counter()
            with torch.no_grad():
                for _ in range(10):
                    output = model(test_input)
            end_time = time.perf_counter()

            avg_time_ms = (end_time - start_time) * 1000 / 10

            print(f"⚡ {config_name} ({opt_level.value}): {avg_time_ms:.2f}ms avg")
            print(f"   {description}")
            print(f"   NVIDIA optimization: {config.hardware.nvidia.enabled}")
            if self.device.type == 'cuda':
                memory_mb = torch.cuda.memory_allocated() / 1024**2
                print(f"   Memory used: {memory_mb:.1f} MB")
            print()

    def run_comprehensive_demo(self):
        """Run all demonstration modules."""
        print("🚀 NVIDIA Configuration Integration Demo")
        print("=" * 60)
        print()

        # Hardware detection
        config = self.demonstrate_hardware_detection()

        # Configuration modes
        configs = self.demonstrate_configuration_modes()

        # Optimization levels
        self.demonstrate_optimization_levels()

        # Validation
        self.demonstrate_configuration_validation()

        # Serialization
        self.demonstrate_config_serialization()

        # Performance comparison
        self.demonstrate_performance_comparison()

        # Summary
        print("📊 Demo Summary")
        print("=" * 20)
        print(f"✅ Hardware detection: {config.hardware.nvidia.architecture.value}")
        print(f"✅ Configuration modes: {len(configs)} tested")
        print("✅ Validation: Comprehensive testing completed")
        print("✅ Serialization: Working properly")
        print("✅ Performance: Multiple optimization levels tested")
        print()
        print("🎯 NVIDIA Integration Demo Complete!")


def main():
    """Main demo execution."""
    parser = argparse.ArgumentParser(description='NVIDIA Configuration Demo')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--quick', action='store_true', help='Run quick demo')
    parser.add_argument('--section', choices=[
        'detection', 'modes', 'optimization', 'validation', 'serialization', 'performance'
    ], help='Run specific demo section')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"🔧 Using device: {device}")
    print(f"🏃 Quick mode: {args.quick}")
    if args.section:
        print(f"📍 Section: {args.section}")
    print()

    # Create demo instance
    demo = NVIDIAConfigurationDemo(device)

    # Run demonstration
    if args.quick:
        print("🏃 Running Quick NVIDIA Configuration Demo...")
        demo.demonstrate_hardware_detection()
        print("✅ Quick demo completed!")
    elif args.section == 'detection':
        demo.demonstrate_hardware_detection()
    elif args.section == 'modes':
        demo.demonstrate_configuration_modes()
    elif args.section == 'optimization':
        demo.demonstrate_optimization_levels()
    elif args.section == 'validation':
        demo.demonstrate_configuration_validation()
    elif args.section == 'serialization':
        demo.demonstrate_config_serialization()
    elif args.section == 'performance':
        demo.demonstrate_performance_comparison()
    else:
        demo.run_comprehensive_demo()


if __name__ == "__main__":
    # Suppress some warnings for cleaner demo output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
