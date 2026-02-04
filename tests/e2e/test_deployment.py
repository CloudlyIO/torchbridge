"""
Tests for the Deployment Module

Tests cover:
- Optimization metadata creation and serialization
- ONNX export functionality
- TorchScript export functionality
- Export validation
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from torchbridge.deployment import (
    FusionMetadata,
    HardwareMetadata,
    ONNXExportConfig,
    # ONNX
    ONNXExporter,
    # Metadata
    OptimizationMetadata,
    TorchScriptExportConfig,
    # TorchScript
    TorchScriptExporter,
    create_metadata,
    export_to_onnx,
    export_to_torchscript,
    load_torchscript,
)


# Test Models
class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size=512, hidden_size=256, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TransformerLikeModel(nn.Module):
    """Transformer-like model for testing."""

    def __init__(self, hidden_size=256, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(hidden_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.fc(x)
        return x


# ============================================================================
# Optimization Metadata Tests
# ============================================================================

class TestOptimizationMetadata:
    """Tests for OptimizationMetadata class."""

    def test_create_default_metadata(self):
        """Test creating metadata with defaults."""
        metadata = OptimizationMetadata()
        assert metadata.schema_version == "1.0.0"
        assert metadata.optimization_level == "balanced"
        assert metadata.export_format == "onnx"

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = OptimizationMetadata(
            optimization_level="aggressive",
            export_format="torchscript"
        )
        data = metadata.to_dict()

        assert data['optimization_level'] == "aggressive"
        assert data['export_format'] == "torchscript"
        assert 'hardware' in data
        assert 'precision' in data
        assert 'fusion' in data

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            'schema_version': '1.0.0',
            'optimization_level': 'conservative',
            'export_format': 'onnx',
            'hardware': {'backend': 'cuda', 'architecture': 'hopper'},
            'precision': {'default_dtype': 'float16'},
            'fusion': {'fusion_count': 5},
            'performance': {'p50_latency_ms': 10.5},
            'model': {'num_parameters': 1000000}
        }

        metadata = OptimizationMetadata.from_dict(data)
        assert metadata.optimization_level == 'conservative'
        assert metadata.hardware.backend == 'cuda'
        assert metadata.precision.default_dtype == 'float16'
        assert metadata.fusion.fusion_count == 5

    def test_metadata_save_load(self):
        """Test saving and loading metadata."""
        metadata = OptimizationMetadata(
            optimization_level="aggressive",
            export_format="onnx"
        )
        metadata.hardware.backend = "cuda"
        metadata.precision.fp8_enabled = True

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "metadata.json")
            metadata.save(path)

            loaded = OptimizationMetadata.load(path)
            assert loaded.optimization_level == "aggressive"
            assert loaded.hardware.backend == "cuda"
            assert loaded.precision.fp8_enabled is True

    def test_metadata_summary(self):
        """Test metadata summary generation."""
        metadata = OptimizationMetadata()
        summary = metadata.summary()

        assert "Optimization Metadata Summary" in summary
        assert "Export Format" in summary
        assert "Hardware" in summary

    def test_create_metadata_from_model(self):
        """Test creating metadata from a model."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        metadata = create_metadata(
            model=model,
            backend="cpu",
            optimization_level="balanced",
            export_format="onnx",
            sample_input=sample_input,
            benchmark=False
        )

        assert metadata.model.num_parameters > 0
        assert metadata.optimization_level == "balanced"


class TestHardwareMetadata:
    """Tests for HardwareMetadata class."""

    def test_hardware_metadata_defaults(self):
        """Test hardware metadata defaults."""
        hw = HardwareMetadata()
        assert hw.backend == "auto"
        assert hw.tensor_cores is False
        assert hw.custom_kernels_used == []

    def test_hardware_metadata_to_dict(self):
        """Test hardware metadata serialization."""
        hw = HardwareMetadata(
            backend="cuda",
            compute_capability=(8, 0),
            tensor_cores=True
        )
        data = hw.to_dict()

        assert data['backend'] == "cuda"
        assert data['compute_capability'] == [8, 0]
        assert data['tensor_cores'] is True


class TestFusionMetadata:
    """Tests for FusionMetadata class."""

    def test_add_fusion(self):
        """Test adding fusion records."""
        fusion = FusionMetadata()
        fusion.add_fusion(
            fusion_type="linear_gelu",
            layers=["layer1", "layer2"],
            estimated_speedup=1.5
        )

        assert fusion.fusion_count == 1
        assert len(fusion.fused_operations) == 1
        assert fusion.fused_operations[0]['type'] == "linear_gelu"


# ============================================================================
# TorchScript Export Tests
# ============================================================================

class TestTorchScriptExporter:
    """Tests for TorchScriptExporter class."""

    def test_export_trace_simple_model(self):
        """Test tracing a simple model."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            exporter = TorchScriptExporter()
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                method="trace",
                benchmark=False
            )

            assert result.success
            assert os.path.exists(result.output_path)
            assert result.export_method == "trace"
            assert result.model_size_mb > 0

    def test_export_script_simple_model(self):
        """Test scripting a simple model."""
        model = SimpleModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            exporter = TorchScriptExporter()
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=torch.randn(1, 512),
                method="script",
                benchmark=False
            )

            assert result.success
            assert result.export_method == "script"

    def test_export_with_validation(self):
        """Test export with validation."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        config = TorchScriptExportConfig(
            validate_export=True,
            validation_tolerance=1e-5
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            exporter = TorchScriptExporter(config)
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                benchmark=False
            )

            assert result.success
            assert result.validated

    def test_export_with_metadata(self):
        """Test export with metadata preservation."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        config = TorchScriptExportConfig(include_metadata=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            exporter = TorchScriptExporter(config)
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                optimization_level="aggressive",
                benchmark=False
            )

            assert result.success
            assert result.metadata is not None
            assert result.metadata.optimization_level == "aggressive"
            assert result.metadata_path is not None

    def test_load_torchscript_model(self):
        """Test loading a TorchScript model with metadata."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            # Export
            exporter = TorchScriptExporter()
            exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                benchmark=False
            )

            # Load
            loaded_model, metadata = load_torchscript(output_path)

            # Verify loaded model works
            with torch.no_grad():
                output = loaded_model(sample_input)
            assert output.shape == (1, 10)

    def test_export_frozen_model(self):
        """Test exporting with model freezing."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        config = TorchScriptExportConfig(freeze_model=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            exporter = TorchScriptExporter(config)
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                benchmark=False
            )

            assert result.success
            assert result.frozen

    def test_convenience_function(self):
        """Test export_to_torchscript convenience function."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")

            result = export_to_torchscript(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                method="trace"
            )

            assert result.success


# ============================================================================
# ONNX Export Tests
# ============================================================================

class TestONNXExporter:
    """Tests for ONNXExporter class."""

    @pytest.fixture
    def check_onnx_available(self):
        """Check if ONNX is available."""
        try:
            import onnx  # noqa: F401
            return True
        except ImportError:
            pytest.skip("ONNX not available")

    def test_export_simple_model(self, check_onnx_available):
        """Test exporting a simple model to ONNX."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            exporter = ONNXExporter()
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                benchmark=False
            )

            assert result.success
            assert os.path.exists(result.output_path)
            assert result.model_size_mb > 0

    def test_export_with_dynamic_axes(self, check_onnx_available):
        """Test export with dynamic axes."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        config = ONNXExportConfig(
            dynamic_batch=True,
            dynamic_sequence=False
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            exporter = ONNXExporter(config)
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                benchmark=False
            )

            assert result.success
            assert "batch_size" in str(result.dynamic_axes)

    def test_export_with_metadata(self, check_onnx_available):
        """Test ONNX export with metadata."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        config = ONNXExportConfig(include_metadata=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            exporter = ONNXExporter(config)
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                optimization_level="balanced",
                benchmark=False
            )

            assert result.success
            assert result.metadata is not None
            assert result.metadata_path is not None

            # Check metadata file exists
            assert os.path.exists(result.metadata_path)

    def test_export_validation(self, check_onnx_available):
        """Test ONNX export validation."""
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pytest.skip("ONNX Runtime not available")

        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        config = ONNXExportConfig(
            validate_export=True,
            validation_tolerance=1e-4
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            exporter = ONNXExporter(config)
            result = exporter.export(
                model=model,
                output_path=output_path,
                sample_input=sample_input,
                benchmark=False
            )

            assert result.success
            assert result.validated

    def test_convenience_function(self, check_onnx_available):
        """Test export_to_onnx convenience function."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")

            result = export_to_onnx(
                model=model,
                output_path=output_path,
                sample_input=sample_input
            )

            assert result.success


# ============================================================================
# Integration Tests
# ============================================================================

class TestDeploymentIntegration:
    """Integration tests for deployment module."""

    def test_export_both_formats(self):
        """Test exporting to both ONNX and TorchScript."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        with tempfile.TemporaryDirectory() as tmpdir:
            # TorchScript export
            ts_path = os.path.join(tmpdir, "model.pt")
            ts_result = export_to_torchscript(
                model=model,
                output_path=ts_path,
                sample_input=sample_input
            )
            assert ts_result.success

            # ONNX export (if available)
            try:
                import onnx  # noqa: F401
                onnx_path = os.path.join(tmpdir, "model.onnx")
                onnx_result = export_to_onnx(
                    model=model,
                    output_path=onnx_path,
                    sample_input=sample_input
                )
                assert onnx_result.success
            except ImportError:
                pass  # ONNX not available

    def test_metadata_consistency(self):
        """Test that metadata is consistent across exports."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        with tempfile.TemporaryDirectory() as tmpdir:
            ts_path = os.path.join(tmpdir, "model.pt")

            result = export_to_torchscript(
                model=model,
                output_path=ts_path,
                sample_input=sample_input,
                optimization_level="aggressive"
            )

            assert result.metadata.optimization_level == "aggressive"
            assert result.metadata.model.num_parameters > 0

    def test_model_output_consistency(self):
        """Test that exported model produces same output."""
        model = SimpleModel()
        sample_input = torch.randn(1, 512)

        # Get original output
        model.eval()
        with torch.no_grad():
            original_output = model(sample_input)

        with tempfile.TemporaryDirectory() as tmpdir:
            ts_path = os.path.join(tmpdir, "model.pt")

            export_to_torchscript(
                model=model,
                output_path=ts_path,
                sample_input=sample_input
            )

            # Load and run
            loaded_model, _ = load_torchscript(ts_path)
            with torch.no_grad():
                loaded_output = loaded_model(sample_input)

            # Compare outputs
            assert torch.allclose(original_output, loaded_output, atol=1e-5)
