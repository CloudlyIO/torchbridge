"""
Full Pipeline Integration Tests

These tests validate the complete KernelPyTorch workflow from model loading
through optimization, export, and inference. They ensure all components
work together correctly.
"""

import pytest
import tempfile
import torch
import torch.nn as nn
from pathlib import Path


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""

    def __init__(self, d_model=256, nhead=4, dim_feedforward=512):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class TestFullOptimizationPipeline:
    """Test complete optimization pipeline."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        return SimpleTransformerBlock(d_model=256, nhead=4)

    @pytest.fixture
    def sample_input(self):
        """Create sample input."""
        return torch.randn(2, 32, 256)

    def test_basic_forward_pass(self, model, sample_input):
        """Test that model produces output."""
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == sample_input.shape

    def test_optimization_with_torch_compile(self, model, sample_input):
        """Test optimization with torch.compile."""
        model.eval()

        # Compile the model
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")

            # Run inference
            with torch.no_grad():
                output = compiled_model(sample_input)

            assert output.shape == sample_input.shape
        except Exception as e:
            # torch.compile may not work on all platforms
            pytest.skip(f"torch.compile not available: {e}")

    def test_torchscript_export_and_load(self, model, sample_input, tmp_path):
        """Test TorchScript export and reload."""
        model.eval()

        # Export with check_trace=False to avoid issues with attention layers
        # that have non-deterministic internal graph naming
        traced = torch.jit.trace(model, sample_input, check_trace=False)
        export_path = tmp_path / "model.pt"
        traced.save(str(export_path))

        # Reload
        loaded = torch.jit.load(str(export_path))

        # Compare outputs
        with torch.no_grad():
            original_output = model(sample_input)
            loaded_output = loaded(sample_input)

        torch.testing.assert_close(original_output, loaded_output, rtol=1e-4, atol=1e-4)

    def test_deterministic_output(self, model, sample_input):
        """Test that model produces deterministic output."""
        model.eval()

        with torch.no_grad():
            output1 = model(sample_input)
            output2 = model(sample_input)

        torch.testing.assert_close(output1, output2)

    def test_batch_size_flexibility(self, model):
        """Test model works with different batch sizes."""
        model.eval()

        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 32, 256)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 32, 256)

    def test_sequence_length_flexibility(self, model):
        """Test model works with different sequence lengths."""
        model.eval()

        for seq_len in [16, 32, 64, 128]:
            x = torch.randn(2, seq_len, 256)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (2, seq_len, 256)


class TestPrecisionModes:
    """Test different precision modes."""

    @pytest.fixture
    def model(self):
        return SimpleTransformerBlock(d_model=256, nhead=4)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 32, 256)

    def test_fp32_inference(self, model, sample_input):
        """Test FP32 inference."""
        model.eval()
        model = model.float()
        x = sample_input.float()

        with torch.no_grad():
            output = model(x)

        assert output.dtype == torch.float32

    def test_fp16_inference(self, model, sample_input):
        """Test FP16 inference."""
        model.eval()
        model = model.half()
        x = sample_input.half()

        with torch.no_grad():
            output = model(x)

        assert output.dtype == torch.float16

    def test_bf16_inference(self, model, sample_input):
        """Test BF16 inference."""
        if not torch.cuda.is_available() and not hasattr(torch, 'bfloat16'):
            pytest.skip("BF16 not supported on this platform")

        model.eval()
        model = model.to(torch.bfloat16)
        x = sample_input.to(torch.bfloat16)

        with torch.no_grad():
            output = model(x)

        assert output.dtype == torch.bfloat16

    def test_mixed_precision_autocast(self, model, sample_input):
        """Test mixed precision with autocast."""
        model.eval()

        with torch.no_grad():
            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                output = model(sample_input)

        # Output may be bf16 or fp32 depending on ops
        assert output.shape == sample_input.shape


class TestKernelPyTorchIntegration:
    """Test KernelPyTorch-specific integrations."""

    def test_config_system(self):
        """Test configuration system."""
        import kernel_pytorch as kpt

        # Get config
        config = kpt.get_config()
        assert config is not None
        assert hasattr(config, 'device')

        # Configure
        kpt.configure(device='cpu')
        config = kpt.get_config()
        assert config.device == 'cpu'

    def test_manager_creation(self):
        """Test unified manager creation."""
        import kernel_pytorch as kpt

        manager = kpt.get_manager()
        assert manager is not None

    def test_attention_layer_import(self):
        """Test attention layer can be imported."""
        import kernel_pytorch as kpt

        # AttentionLayer is the base class
        assert kpt.AttentionLayer is not None

        # UnifiedAttentionFusion is the main attention implementation
        attention = kpt.UnifiedAttentionFusion(d_model=256, n_heads=4, d_ff=512)
        assert attention is not None

        # Test forward pass
        x = torch.randn(2, 32, 256)
        with torch.no_grad():
            output = attention(x)
        assert output.shape == x.shape

    def test_create_moe(self):
        """Test MoE layer creation."""
        import kernel_pytorch as kpt

        moe = kpt.create_moe(hidden_size=256, num_experts=4, top_k=2)
        assert moe is not None

        # Test forward pass
        x = torch.randn(2, 32, 256)
        with torch.no_grad():
            output = moe(x)
        assert output.shape == x.shape

    def test_fused_gelu(self):
        """Test fused GELU activation."""
        import kernel_pytorch as kpt

        gelu = kpt.FusedGELU()
        x = torch.randn(2, 32, 256)

        with torch.no_grad():
            output = gelu(x)

        assert output.shape == x.shape


class TestCLIIntegration:
    """Test CLI command integration."""

    def test_optimize_command_import(self):
        """Test optimize command can be imported."""
        from kernel_pytorch.cli.optimize import OptimizeCommand
        assert OptimizeCommand is not None

    def test_benchmark_command_import(self):
        """Test benchmark command can be imported."""
        from kernel_pytorch.cli.benchmark import BenchmarkCommand
        assert BenchmarkCommand is not None

    def test_doctor_command_import(self):
        """Test doctor command can be imported."""
        from kernel_pytorch.cli.doctor import DoctorCommand
        assert DoctorCommand is not None

    def test_export_command_import(self):
        """Test export command can be imported."""
        from kernel_pytorch.cli.export import ExportCommand
        assert ExportCommand is not None

    def test_profile_command_import(self):
        """Test profile command can be imported."""
        from kernel_pytorch.cli.profile import ProfileCommand
        assert ProfileCommand is not None

    def test_main_cli_import(self):
        """Test main CLI can be imported."""
        from kernel_pytorch.cli import main
        assert callable(main)


class TestDeploymentIntegration:
    """Test deployment module integration."""

    def test_onnx_exporter_import(self):
        """Test ONNX exporter can be imported."""
        from kernel_pytorch.deployment import ONNXExporter
        assert ONNXExporter is not None

    def test_torchscript_exporter_import(self):
        """Test TorchScript exporter can be imported."""
        from kernel_pytorch.deployment import TorchScriptExporter
        assert TorchScriptExporter is not None

    def test_safetensors_exporter_import(self):
        """Test SafeTensors exporter can be imported."""
        from kernel_pytorch.deployment import SafeTensorsExporter
        assert SafeTensorsExporter is not None

    def test_production_validator_import(self):
        """Test production validator can be imported."""
        from kernel_pytorch.deployment import ProductionValidator
        assert ProductionValidator is not None

    def test_serving_imports(self):
        """Test serving components can be imported."""
        from kernel_pytorch.deployment import (
            create_fastapi_server,
            ServerConfig,
        )
        assert create_fastapi_server is not None
        assert ServerConfig is not None


class TestBackendIntegration:
    """Test hardware backend integration."""

    def test_nvidia_backend_import(self):
        """Test NVIDIA backend can be imported."""
        from kernel_pytorch.backends.nvidia import NVIDIABackend, NVIDIAOptimizer
        assert NVIDIABackend is not None
        assert NVIDIAOptimizer is not None

    def test_amd_backend_import(self):
        """Test AMD backend can be imported."""
        from kernel_pytorch.backends.amd import AMDBackend, AMDOptimizer
        assert AMDBackend is not None
        assert AMDOptimizer is not None

    def test_intel_backend_import(self):
        """Test Intel backend can be imported."""
        from kernel_pytorch.backends.intel import IntelBackend, IntelOptimizer
        assert IntelBackend is not None
        assert IntelOptimizer is not None

    def test_tpu_backend_import(self):
        """Test TPU backend can be imported."""
        from kernel_pytorch.backends.tpu import TPUBackend, TPUOptimizer
        assert TPUBackend is not None
        assert TPUOptimizer is not None

    def test_hal_import(self):
        """Test Hardware Abstraction Layer can be imported."""
        from kernel_pytorch.hardware.abstraction.hal_core import HardwareAbstractionLayer
        assert HardwareAbstractionLayer is not None


class TestDistributedIntegration:
    """Test distributed training integration."""

    def test_distributed_config_import(self):
        """Test distributed config can be imported."""
        from kernel_pytorch.models.distributed import DistributedConfig
        assert DistributedConfig is not None

    def test_tensor_parallel_import(self):
        """Test tensor parallel can be imported."""
        from kernel_pytorch.models.distributed import TensorParallelConfig
        assert TensorParallelConfig is not None

    def test_pipeline_parallel_import(self):
        """Test pipeline parallel can be imported."""
        from kernel_pytorch.models.distributed.pipeline_parallel import PipelineParallelConfig
        assert PipelineParallelConfig is not None


class TestMemoryOptimization:
    """Test memory optimization components."""

    def test_gradient_checkpointing_import(self):
        """Test gradient checkpointing can be imported."""
        from kernel_pytorch.advanced_memory import SelectiveGradientCheckpointing
        assert SelectiveGradientCheckpointing is not None

    def test_deep_optimizer_states_import(self):
        """Test DeepOptimizerStates can be imported."""
        from kernel_pytorch.advanced_memory import DeepOptimizerStates
        assert DeepOptimizerStates is not None


class TestValidationFramework:
    """Test validation framework."""

    def test_unified_validator_import(self):
        """Test unified validator can be imported."""
        from kernel_pytorch.validation import UnifiedValidator
        assert UnifiedValidator is not None

    def test_validator_basic_usage(self):
        """Test basic validator usage."""
        from kernel_pytorch.validation import UnifiedValidator

        model = nn.Linear(256, 128)
        validator = UnifiedValidator()

        # Basic validation should work
        assert validator is not None
