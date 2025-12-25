"""
Production Pipeline Demo - Stage 3C: Production Examples

Demonstrates a complete production-ready workflow integrating all Phase 3 features:
- Stage 3A: Automatic hardware detection and optimization selection
- Stage 3B: Performance tracking and regression detection
- Complete end-to-end pipeline from model creation to production deployment

This demo shows:
1. Production model development workflow
2. Automatic optimization and validation
3. Performance regression detection in CI/CD
4. Best practices for production deployment
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional, Dict, Any
from pathlib import Path

from kernel_pytorch.core.management import get_manager
from kernel_pytorch.core.hardware_detector import detect_hardware, get_optimal_backend
from kernel_pytorch.core.performance_tracker import get_performance_tracker
from kernel_pytorch.validation.unified_validator import UnifiedValidator
from kernel_pytorch.core.config import KernelPyTorchConfig


# Production Model Example
class ProductionTransformer(nn.Module):
    """
    Production-ready transformer model.

    This represents a real-world model that would be deployed in production.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)

        # Embedding
        x = self.embedding(x) * (self.d_model ** 0.5)

        # Positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)

        # Output projection
        return self.fc_out(x)


class ProductionPipeline:
    """
    Complete production pipeline integrating all optimization features.

    This class demonstrates best practices for using KernelPyTorch in production.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_dir: Optional[str] = None,
        enable_regression_detection: bool = True
    ):
        """
        Initialize production pipeline.

        Args:
            model_name: Name of the model
            checkpoint_dir: Directory to save checkpoints
            enable_regression_detection: Whether to enable regression detection
        """
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.enable_regression_detection = enable_regression_detection

        # Initialize components
        self.manager = get_manager()
        self.tracker = get_performance_tracker()
        self.validator = UnifiedValidator()

        # Detect hardware
        self.hardware_profile = detect_hardware()
        self.optimal_backend = get_optimal_backend()

        print(f"🚀 Production Pipeline Initialized")
        print(f"   Model: {model_name}")
        print(f"   Hardware: {self.hardware_profile.hardware_type.value}")
        print(f"   Backend: {self.optimal_backend}")
        print(f"   Checkpoint Dir: {self.checkpoint_dir}")

    def optimize_for_training(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        optimization_level: str = "auto"
    ) -> nn.Module:
        """
        Optimize model for training with automatic configuration.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs for the model
            optimization_level: Optimization level (auto/conservative/balanced/aggressive)

        Returns:
            Optimized model
        """
        print(f"\n📊 Optimizing for Training")
        print(f"   Optimization Level: {optimization_level}")

        # Auto-optimize
        if optimization_level == "auto":
            optimized_model = self.manager.auto_optimize(
                model,
                sample_inputs=sample_inputs,
                for_inference=False
            )
        else:
            optimized_model = self.manager.auto_optimize(
                model,
                sample_inputs=sample_inputs,
                optimization_level=optimization_level,
                for_inference=False
            )

        # Record baseline performance
        if self.enable_regression_detection:
            print("   Recording baseline performance...")
            self.tracker.record_performance(
                model=optimized_model,
                sample_inputs=sample_inputs,
                model_name=f"{self.model_name}_training",
                backend=self.optimal_backend,
                optimization_level=optimization_level
            )

        print("   ✅ Training optimization complete")
        return optimized_model

    def optimize_for_inference(
        self,
        model: nn.Module,
        sample_inputs: torch.Tensor,
        check_regression: bool = True
    ) -> nn.Module:
        """
        Optimize model for inference with regression detection.

        Args:
            model: Model to optimize
            sample_inputs: Sample inputs for the model
            check_regression: Whether to check for regressions

        Returns:
            Optimized model
        """
        print(f"\n🎯 Optimizing for Inference")

        # Get baseline if exists
        baseline = None
        if check_regression and self.enable_regression_detection:
            baseline = self.tracker.get_baseline(model, backend=self.optimal_backend)
            if baseline:
                print(f"   Found baseline: {baseline.latency_ms:.3f} ms")

        # Auto-optimize
        optimized_model = self.manager.auto_optimize(
            model,
            sample_inputs=sample_inputs,
            for_inference=True
        )

        # Record and check performance
        if self.enable_regression_detection:
            print("   Measuring performance...")
            current_metrics = self.tracker.record_performance(
                model=optimized_model,
                sample_inputs=sample_inputs,
                model_name=f"{self.model_name}_inference",
                backend=self.optimal_backend,
                optimization_level="aggressive"
            )

            print(f"   Current: {current_metrics.latency_ms:.3f} ms")

            # Check for regressions
            if check_regression and baseline:
                print("   Checking for regressions...")
                regressions = self.tracker.detect_regression(
                    model,
                    current_metrics,
                    baseline
                )

                if regressions:
                    print(f"   ⚠️  {len(regressions)} regression(s) detected!")
                    for reg in regressions:
                        print(f"      {reg.message}")
                else:
                    print("   ✅ No regressions detected")

        print("   ✅ Inference optimization complete")
        return optimized_model

    def validate_optimized_model(
        self,
        model: nn.Module,
        original_model: nn.Module,
        sample_inputs: torch.Tensor,
        tolerance: float = 1e-4
    ) -> bool:
        """
        Validate that optimized model produces correct results.

        Args:
            model: Optimized model
            original_model: Original model
            sample_inputs: Sample inputs
            tolerance: Numerical tolerance

        Returns:
            True if validation passes
        """
        print(f"\n✓ Validating Optimized Model")

        model.eval()
        original_model.eval()

        with torch.no_grad():
            optimized_output = model(sample_inputs)
            original_output = original_model(sample_inputs)

        # Check outputs match
        max_diff = torch.max(torch.abs(optimized_output - original_output)).item()
        print(f"   Max output difference: {max_diff:.2e}")

        if max_diff < tolerance:
            print(f"   ✅ Validation passed (within tolerance {tolerance})")
            return True
        else:
            print(f"   ❌ Validation failed (exceeds tolerance {tolerance})")
            return False

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save model checkpoint with metadata.

        Args:
            model: Model to save
            optimizer: Optional optimizer state
            epoch: Current epoch
            metadata: Additional metadata
        """
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pt"

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'hardware_type': self.hardware_profile.hardware_type.value,
            'backend': self.optimal_backend,
            'metadata': metadata or {}
        }

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"\n💾 Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> int:
        """
        Load model checkpoint.

        Args:
            model: Model to load into
            checkpoint_path: Path to checkpoint
            optimizer: Optional optimizer to load state into

        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint.get('epoch', 0)

        print(f"\n📂 Checkpoint loaded: {checkpoint_path}")
        print(f"   Epoch: {epoch}")
        print(f"   Hardware: {checkpoint.get('hardware_type', 'unknown')}")
        print(f"   Backend: {checkpoint.get('backend', 'unknown')}")

        return epoch

    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get optimization recommendations for the current hardware.

        Returns:
            Dictionary with recommendations
        """
        return self.manager.get_optimization_recommendations()


def print_section(title: str):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_1_training_workflow():
    """Demo 1: Complete Training Workflow."""
    print_section("Demo 1: Production Training Workflow")

    # Initialize pipeline
    pipeline = ProductionPipeline(
        model_name="ProductionTransformer",
        checkpoint_dir="./checkpoints",
        enable_regression_detection=True
    )

    # Create model
    print("\n📦 Creating production model...")
    model = ProductionTransformer(
        vocab_size=5000,
        d_model=256,
        nhead=4,
        num_layers=3
    )

    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Sample inputs
    sample_inputs = torch.randint(0, 5000, (8, 32))  # batch=8, seq_len=32

    # Optimize for training
    optimized_model = pipeline.optimize_for_training(
        model=model,
        sample_inputs=sample_inputs,
        optimization_level="auto"
    )

    # Validate
    is_valid = pipeline.validate_optimized_model(
        model=optimized_model,
        original_model=model,
        sample_inputs=sample_inputs
    )

    if is_valid:
        # Save checkpoint
        pipeline.save_checkpoint(
            model=optimized_model,
            epoch=0,
            metadata={'validation': 'passed'}
        )

    print("\n✅ Training workflow complete!")


def demo_2_inference_deployment():
    """Demo 2: Inference Deployment Workflow."""
    print_section("Demo 2: Production Inference Deployment")

    # Initialize pipeline
    pipeline = ProductionPipeline(
        model_name="InferenceModel",
        enable_regression_detection=True
    )

    # Create model
    print("\n📦 Creating model for inference...")
    model = ProductionTransformer(
        vocab_size=5000,
        d_model=256,
        nhead=4,
        num_layers=2
    )

    sample_inputs = torch.randint(0, 5000, (1, 16))  # Small batch for inference

    # Record baseline
    print("\n📊 Recording baseline performance...")
    tracker = get_performance_tracker()
    tracker.clear_history(model)  # Clear for demo

    baseline_metrics = tracker.record_performance(
        model=model,
        sample_inputs=sample_inputs,
        model_name="InferenceModel_baseline",
        backend="cpu",
        optimization_level="none"
    )

    print(f"   Baseline latency: {baseline_metrics.latency_ms:.3f} ms")

    # Optimize for inference
    optimized_model = pipeline.optimize_for_inference(
        model=model,
        sample_inputs=sample_inputs,
        check_regression=True
    )

    # Get recommendations
    print("\n💡 Getting optimization recommendations...")
    recommendations = pipeline.get_recommendations()

    print(f"   Hardware: {recommendations['hardware_type']}")
    print(f"   Backend: {recommendations['backend']}")
    print(f"   Optimization Level: {recommendations['optimization_level']}")

    print("\n✅ Inference deployment ready!")


def demo_3_ci_cd_integration():
    """Demo 3: CI/CD Pipeline Integration."""
    print_section("Demo 3: CI/CD Pipeline Integration")

    print("Simulating CI/CD pipeline for model deployment...\n")

    # Initialize pipeline
    pipeline = ProductionPipeline(
        model_name="CI_CD_Model",
        enable_regression_detection=True
    )

    # Create model
    model = ProductionTransformer(
        vocab_size=2000,
        d_model=128,
        nhead=4,
        num_layers=2
    )

    sample_inputs = torch.randint(0, 2000, (4, 16))

    print("📋 CI/CD Steps:")
    print("   1. Build model")
    print("   2. Optimize model")
    print("   3. Run performance tests")
    print("   4. Check for regressions")
    print("   5. Deploy if all checks pass\n")

    # Step 1: Build (already done)
    print("✓ Step 1: Model built")

    # Step 2: Optimize
    print("✓ Step 2: Optimizing...")
    optimized_model = pipeline.optimize_for_inference(
        model=model,
        sample_inputs=sample_inputs,
        check_regression=False  # Will check manually
    )

    # Step 3: Performance tests
    print("✓ Step 3: Running performance tests...")
    tracker = get_performance_tracker()
    current_metrics = tracker.record_performance(
        model=optimized_model,
        sample_inputs=sample_inputs,
        model_name="CI_CD_Model",
        backend="cpu",
        optimization_level="aggressive"
    )

    print(f"   Latency: {current_metrics.latency_ms:.3f} ms")
    print(f"   Throughput: {current_metrics.throughput:.1f} samples/sec")

    # Step 4: Check regressions
    print("✓ Step 4: Checking for regressions...")
    baseline = tracker.get_baseline(model)

    if baseline:
        regressions = tracker.detect_regression(model, current_metrics, baseline)

        if regressions:
            print(f"   ❌ {len(regressions)} regression(s) detected - BLOCKING DEPLOYMENT")
            for reg in regressions:
                print(f"      {reg.message}")
            deployment_allowed = False
        else:
            print("   ✅ No regressions - deployment allowed")
            deployment_allowed = True
    else:
        print("   ℹ️  No baseline - skipping regression check")
        deployment_allowed = True

    # Step 5: Deploy
    if deployment_allowed:
        print("✓ Step 5: Deploying to production...")
        pipeline.save_checkpoint(
            model=optimized_model,
            metadata={'ci_cd': 'passed', 'deployed': True}
        )
        print("   ✅ Deployed successfully!")
    else:
        print("✗ Step 5: Deployment blocked due to regressions")

    print("\n✅ CI/CD pipeline complete!")


def demo_4_multi_backend_deployment():
    """Demo 4: Multi-Backend Deployment."""
    print_section("Demo 4: Multi-Backend Deployment Strategy")

    print("Demonstrating deployment across multiple backends...\n")

    model = ProductionTransformer(
        vocab_size=1000,
        d_model=128,
        nhead=4,
        num_layers=2
    )

    sample_inputs = torch.randint(0, 1000, (2, 8))

    # Simulate different backends
    backends = ["cpu", "nvidia", "tpu"]

    print("📊 Testing model on different backends:\n")

    for backend in backends:
        print(f"Backend: {backend}")

        # Simulate backend-specific optimization
        pipeline = ProductionPipeline(
            model_name=f"Model_{backend}",
            enable_regression_detection=False
        )

        try:
            # Get recommendations for this backend
            recommendations = pipeline.get_recommendations()

            print(f"   Recommended level: {recommendations['optimization_level']}")
            print(f"   Capabilities: {len(recommendations['capabilities'])}")

            # Note: In production, you'd actually deploy to the specific backend
            print(f"   ✅ Ready for {backend} deployment")

        except Exception as e:
            print(f"   ⚠️  Skipping {backend}: {e}")

        print()

    print("✅ Multi-backend deployment strategy complete!")


def demo_5_monitoring_and_alerts():
    """Demo 5: Production Monitoring and Alerts."""
    print_section("Demo 5: Production Monitoring and Alerts")

    print("Demonstrating production monitoring...\n")

    pipeline = ProductionPipeline(
        model_name="MonitoredModel",
        enable_regression_detection=True
    )

    model = ProductionTransformer(
        vocab_size=1000,
        d_model=64,
        nhead=2,
        num_layers=2
    )

    sample_inputs = torch.randint(0, 1000, (2, 8))

    # Simulate monitoring over time
    print("📈 Simulating monitoring over multiple deployments:\n")

    tracker = get_performance_tracker()
    tracker.clear_history(model)

    versions = ["v1.0", "v1.1", "v1.2", "v2.0"]

    for version in versions:
        print(f"Deploying {version}...")

        # Record performance
        metrics = tracker.record_performance(
            model=model,
            sample_inputs=sample_inputs,
            model_name=f"MonitoredModel_{version}",
            backend="cpu",
            optimization_level="balanced"
        )

        print(f"   Latency: {metrics.latency_ms:.3f} ms")

        # Check against baseline
        baseline = tracker.get_baseline(model)
        if baseline and baseline != metrics:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                tracker.warn_if_regression(model, metrics, baseline)

                if w:
                    print(f"   ⚠️  Alert triggered: {len(w)} warning(s)")
                else:
                    print("   ✅ Performance within acceptable range")
        print()

    # Show history
    print("📊 Performance History:")
    history = tracker.get_performance_history(model, limit=5)

    for i, metrics in enumerate(history, 1):
        print(f"   {i}. {metrics.model_name}: {metrics.latency_ms:.3f} ms")

    print("\n✅ Monitoring complete!")


def main():
    """Run all production demos."""
    print("\n" + "="*70)
    print("  Production Pipeline Demo - Stage 3C")
    print("  Complete End-to-End Production Examples")
    print("="*70)

    try:
        # Run all demos
        demo_1_training_workflow()
        demo_2_inference_deployment()
        demo_3_ci_cd_integration()
        demo_4_multi_backend_deployment()
        demo_5_monitoring_and_alerts()

        # Final summary
        print_section("Summary")
        print("✅ All production demos completed successfully!")

        print("\nProduction Features Demonstrated:")
        print("  1. Complete training workflow with auto-optimization")
        print("  2. Inference deployment with regression detection")
        print("  3. CI/CD pipeline integration")
        print("  4. Multi-backend deployment strategy")
        print("  5. Production monitoring and alerting")

        print("\nBest Practices:")
        print("  • Always validate optimized models against originals")
        print("  • Record baseline performance before optimization")
        print("  • Check for regressions in CI/CD pipeline")
        print("  • Monitor performance over time")
        print("  • Use automatic hardware detection for portability")

        print("\nProduction Usage:")
        print("  pipeline = ProductionPipeline('my_model')")
        print("  optimized = pipeline.optimize_for_inference(model, inputs)")
        print("  pipeline.validate_optimized_model(optimized, model, inputs)")
        print("  pipeline.save_checkpoint(optimized)")

        print("\n" + "="*70)
        print("  Demo Complete!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
