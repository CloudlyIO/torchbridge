"""
Export Pipeline E2E Tests (v0.4.25)

Tests for the complete model export and deployment pipeline including:
- ONNX export
- TorchScript export
- SafeTensors export
- Production validation
- Export CLI

These tests validate the export pipeline works correctly
without requiring external services.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


def _onnx_available() -> bool:
    """Check if ONNX is available."""
    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
        return True
    except ImportError:
        return False


def _safetensors_available() -> bool:
    """Check if SafeTensors is available."""
    try:
        import safetensors  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# Test Models
# =============================================================================

class SimpleLinear(nn.Module):
    """Simple linear model for testing."""

    def __init__(self, in_features: int = 512, out_features: int = 256):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""

    def __init__(self, in_features: int = 512, hidden: int = 256, out_features: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TransformerLike(nn.Module):
    """Transformer-like model for testing."""

    def __init__(self, hidden_size: int = 256, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm(x + self.ffn(x))
        return x


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    model = SimpleLinear()
    model.eval()
    return model


@pytest.fixture
def mlp_model():
    """Create MLP model for testing."""
    model = SimpleMLP()
    model.eval()
    return model


@pytest.fixture
def transformer_model():
    """Create transformer-like model for testing."""
    model = TransformerLike()
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(1, 512)


@pytest.fixture
def batch_input():
    """Create batched input tensor."""
    return torch.randn(4, 512)


@pytest.fixture
def sequence_input():
    """Create sequence input tensor."""
    return torch.randn(2, 32, 256)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# SafeTensors Export Tests
# =============================================================================

class TestSafeTensorsExport:
    """Test SafeTensors export functionality."""

    def test_safetensors_import(self):
        """Test SafeTensors module can be imported."""
        from torchbridge.deployment import (
            SafeTensorsExportConfig,
            SafeTensorsExporter,
        )
        assert SafeTensorsExporter is not None
        assert SafeTensorsExportConfig is not None

    def test_export_simple_model(self, simple_model, temp_dir):
        """Test exporting simple model to SafeTensors."""
        pytest.importorskip("safetensors")

        from torchbridge.deployment import export_to_safetensors

        output_path = Path(temp_dir) / "model.safetensors"
        result = export_to_safetensors(simple_model, output_path)

        assert result.success
        assert output_path.exists()
        assert result.file_size_mb > 0
        assert result.num_tensors > 0

    def test_export_with_metadata(self, simple_model, temp_dir):
        """Test export with custom metadata."""
        pytest.importorskip("safetensors")

        from torchbridge.deployment import export_to_safetensors

        output_path = Path(temp_dir) / "model.safetensors"
        metadata = {"description": "Test model", "version": "1.0"}
        result = export_to_safetensors(simple_model, output_path, metadata=metadata)

        assert result.success
        assert "description" in result.metadata
        assert result.metadata["description"] == "Test model"

    def test_export_half_precision(self, simple_model, temp_dir):
        """Test FP16 export."""
        pytest.importorskip("safetensors")

        from torchbridge.deployment import export_to_safetensors

        output_path = Path(temp_dir) / "model_fp16.safetensors"
        result = export_to_safetensors(simple_model, output_path, half_precision=True)

        assert result.success
        # FP16 should be smaller
        assert result.file_size_mb > 0

    def test_load_safetensors(self, simple_model, temp_dir):
        """Test loading SafeTensors file."""
        pytest.importorskip("safetensors")

        from torchbridge.deployment import export_to_safetensors, load_safetensors

        output_path = Path(temp_dir) / "model.safetensors"
        export_to_safetensors(simple_model, output_path)

        # Load tensors
        state_dict = load_safetensors(output_path)
        assert len(state_dict) > 0
        assert "linear.weight" in state_dict

    def test_load_into_model(self, temp_dir):
        """Test loading SafeTensors into model."""
        pytest.importorskip("safetensors")

        from torchbridge.deployment import (
            export_to_safetensors,
            load_model_safetensors,
        )

        # Create and export model
        model1 = SimpleLinear()
        output_path = Path(temp_dir) / "model.safetensors"
        export_to_safetensors(model1, output_path)

        # Load into new model
        model2 = SimpleLinear()
        model2 = load_model_safetensors(model2, output_path)

        # Check weights are same
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):  # noqa: B007
            assert torch.allclose(p1, p2)


# =============================================================================
# Production Validator Tests
# =============================================================================

class TestProductionValidator:
    """Test production validation functionality."""

    def test_validator_import(self):
        """Test production validator can be imported."""
        from torchbridge.deployment import (
            ProductionRequirements,
            ProductionValidator,
        )
        assert ProductionValidator is not None
        assert ProductionRequirements is not None

    def test_validate_simple_model(self, simple_model, sample_input):
        """Test validating simple model."""
        from torchbridge.deployment import validate_production_readiness

        result = validate_production_readiness(simple_model, sample_input)

        assert result is not None
        assert hasattr(result, "passed")
        assert hasattr(result, "checks")
        assert len(result.checks) > 0

    def test_validate_forward_pass(self, simple_model, sample_input):
        """Test forward pass validation."""
        from torchbridge.deployment import (
            ProductionRequirements,
            ProductionValidator,
        )

        validator = ProductionValidator()
        requirements = ProductionRequirements()
        result = validator.validate(simple_model, sample_input, requirements)

        # Find forward pass check
        forward_check = next((c for c in result.checks if c.name == "forward_pass"), None)
        assert forward_check is not None
        assert forward_check.passed

    def test_validate_determinism(self, simple_model, sample_input):
        """Test determinism validation."""
        from torchbridge.deployment import (
            ProductionRequirements,
            ProductionValidator,
        )

        validator = ProductionValidator()
        requirements = ProductionRequirements()
        result = validator.validate(simple_model, sample_input, requirements)

        # Find determinism check
        determinism_check = next((c for c in result.checks if c.name == "determinism"), None)
        assert determinism_check is not None
        assert determinism_check.passed

    def test_validate_eval_mode(self, sample_input):
        """Test eval mode validation."""
        from torchbridge.deployment import (
            ProductionRequirements,
            ProductionValidator,
        )

        # Model in training mode
        model = SimpleLinear()
        model.train()  # Keep in training mode

        validator = ProductionValidator()
        requirements = ProductionRequirements()
        result = validator.validate(model, sample_input, requirements)

        # Find eval mode check - model is put in eval mode during validation
        eval_check = next((c for c in result.checks if c.name == "eval_mode"), None)
        assert eval_check is not None
        # The validator puts model in eval mode, so it should pass
        assert eval_check.passed

    def test_validate_torchscript_export(self, simple_model, sample_input):
        """Test TorchScript export validation."""
        from torchbridge.deployment import (
            ProductionRequirements,
            ProductionValidator,
        )

        validator = ProductionValidator()
        requirements = ProductionRequirements(require_torchscript_export=True)
        result = validator.validate(simple_model, sample_input, requirements)

        # Find TorchScript check
        ts_check = next((c for c in result.checks if c.name == "torchscript_export"), None)
        assert ts_check is not None
        assert ts_check.passed

    def test_validate_with_requirements(self, simple_model, sample_input):
        """Test validation with custom requirements."""
        from torchbridge.deployment import validate_production_readiness

        result = validate_production_readiness(
            simple_model,
            sample_input,
            requirements={
                "max_latency_ms": 1000,  # Very permissive
            }
        )

        assert result is not None
        # Should pass with permissive latency requirement
        latency_check = next((c for c in result.checks if c.name == "latency"), None)
        if latency_check:
            assert latency_check.passed

    def test_validate_latency_stats(self, simple_model, sample_input):
        """Test that latency stats are collected."""
        from torchbridge.deployment import (
            ProductionRequirements,
            ProductionValidator,
        )

        validator = ProductionValidator()
        requirements = ProductionRequirements()
        result = validator.validate(simple_model, sample_input, requirements)

        assert result.latency_stats is not None
        assert "avg_ms" in result.latency_stats
        assert "p95_ms" in result.latency_stats
        assert "throughput" in result.latency_stats

    def test_validation_result_properties(self, simple_model, sample_input):
        """Test validation result properties."""
        from torchbridge.deployment import validate_production_readiness

        result = validate_production_readiness(simple_model, sample_input)

        assert hasattr(result, "passed_checks")
        assert hasattr(result, "failed_checks")
        assert hasattr(result, "critical_failures")
        assert hasattr(result, "summary")


# =============================================================================
# ONNX Export Tests
# =============================================================================

class TestONNXExport:
    """Test ONNX export functionality."""

    @pytest.mark.skipif(
        not _onnx_available(),
        reason="ONNX not available"
    )
    def test_onnx_export_simple(self, simple_model, sample_input, temp_dir):
        """Test basic ONNX export."""
        from torchbridge.deployment import export_to_onnx

        output_path = Path(temp_dir) / "model.onnx"
        result = export_to_onnx(simple_model, output_path, sample_input)

        assert result.success
        assert output_path.exists()

    @pytest.mark.skipif(
        not _onnx_available(),
        reason="ONNX not available"
    )
    def test_onnx_export_mlp(self, mlp_model, sample_input, temp_dir):
        """Test ONNX export of MLP."""
        from torchbridge.deployment import export_to_onnx

        output_path = Path(temp_dir) / "mlp.onnx"
        result = export_to_onnx(mlp_model, output_path, sample_input)

        assert result.success


# =============================================================================
# TorchScript Export Tests
# =============================================================================

class TestTorchScriptExport:
    """Test TorchScript export functionality."""

    def test_torchscript_trace(self, simple_model, sample_input, temp_dir):
        """Test TorchScript trace export."""
        from torchbridge.deployment import export_to_torchscript

        output_path = Path(temp_dir) / "model.pt"
        result = export_to_torchscript(
            simple_model, output_path, sample_input, method="trace"
        )

        assert result.success
        assert output_path.exists()

    def test_torchscript_load(self, simple_model, sample_input, temp_dir):
        """Test loading TorchScript model."""
        from torchbridge.deployment import export_to_torchscript, load_torchscript

        output_path = Path(temp_dir) / "model.pt"
        export_to_torchscript(simple_model, output_path, sample_input)

        result = load_torchscript(output_path)
        # load_torchscript returns (model, metadata) tuple
        if isinstance(result, tuple):
            loaded, metadata = result
        else:
            loaded = result

        assert loaded is not None

        # Test inference
        with torch.no_grad():
            output = loaded(sample_input)
        assert output.shape[0] == sample_input.shape[0]


# =============================================================================
# Export CLI Tests
# =============================================================================

class TestExportCLI:
    """Test export CLI functionality."""

    def test_cli_parser(self):
        """Test CLI parser creation."""
        from torchbridge.deployment.export_cli import create_parser

        parser = create_parser()
        assert parser is not None

    def test_parse_shape(self):
        """Test shape parsing."""
        from torchbridge.deployment.export_cli import parse_shape

        assert parse_shape("(1, 512)") == (1, 512)
        assert parse_shape("1, 512") == (1, 512)
        assert parse_shape("[2, 32, 256]") == (2, 32, 256)

    def test_create_sample_input(self):
        """Test sample input creation."""
        from torchbridge.deployment.export_cli import create_sample_input

        tensor = create_sample_input((1, 512))
        assert tensor.shape == (1, 512)
        assert tensor.dtype == torch.float32

        tensor_fp16 = create_sample_input((2, 256), dtype="float16")
        assert tensor_fp16.dtype == torch.float16


# =============================================================================
# Integration Tests
# =============================================================================

class TestExportIntegration:
    """Integration tests for export pipeline."""

    def test_full_export_pipeline(self, mlp_model, sample_input, temp_dir):
        """Test complete export pipeline."""
        from torchbridge.deployment import (
            ProductionRequirements,
            ProductionValidator,
            export_to_torchscript,
        )

        # Validate with ONNX optional
        validator = ProductionValidator()
        requirements = ProductionRequirements(
            require_onnx_export=_onnx_available(),
            require_torchscript_export=True,
        )
        validation = validator.validate(mlp_model, sample_input, requirements)

        # Should pass if TorchScript works
        ts_check = next((c for c in validation.checks if c.name == "torchscript_export"), None)
        assert ts_check is not None and ts_check.passed

        # Export to TorchScript
        ts_path = Path(temp_dir) / "model.pt"
        ts_result = export_to_torchscript(mlp_model, ts_path, sample_input)
        assert ts_result.success

    @pytest.mark.skipif(
        not _onnx_available(),
        reason="ONNX not available"
    )
    def test_export_with_validation(self, simple_model, sample_input, temp_dir):
        """Test export with validation enabled."""
        from torchbridge.deployment import export_to_onnx

        output_path = Path(temp_dir) / "validated.onnx"
        result = export_to_onnx(simple_model, output_path, sample_input)

        assert result.success

    def test_transformer_export(self, transformer_model, sequence_input, temp_dir):
        """Test exporting transformer-like model."""
        from torchbridge.deployment import (
            export_to_torchscript,
            validate_production_readiness,
        )

        # Validate
        validation = validate_production_readiness(transformer_model, sequence_input)
        # Should at least have forward pass working
        forward_check = next((c for c in validation.checks if c.name == "forward_pass"), None)
        assert forward_check is not None
        assert forward_check.passed

        # Export to TorchScript
        ts_path = Path(temp_dir) / "transformer.pt"
        ts_result = export_to_torchscript(transformer_model, ts_path, sequence_input)
        assert ts_result.success


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases for export pipeline."""

    def test_empty_model(self, temp_dir):
        """Test exporting model with no parameters."""
        pytest.importorskip("safetensors")

        from torchbridge.deployment import export_to_safetensors

        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        model = EmptyModel()
        output_path = Path(temp_dir) / "empty.safetensors"

        # Should handle empty model
        result = export_to_safetensors(model, output_path)
        assert result.success
        assert result.num_tensors == 0

    def test_large_batch_validation(self, simple_model):
        """Test validation with large batch."""
        from torchbridge.deployment import validate_production_readiness

        large_input = torch.randn(64, 512)
        result = validate_production_readiness(simple_model, large_input)
        assert result is not None

    def test_multiple_exports_same_dir(self, simple_model, sample_input, temp_dir):
        """Test multiple exports to same directory."""
        from torchbridge.deployment import export_to_torchscript

        # Export to TorchScript (always available)
        ts_result = export_to_torchscript(
            simple_model, Path(temp_dir) / "model.pt", sample_input
        )

        assert ts_result.success
        assert (Path(temp_dir) / "model.pt").exists()

        # Export to ONNX if available
        if _onnx_available():
            from torchbridge.deployment import export_to_onnx
            onnx_result = export_to_onnx(
                simple_model, Path(temp_dir) / "model.onnx", sample_input
            )
            assert onnx_result.success
            assert (Path(temp_dir) / "model.onnx").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
