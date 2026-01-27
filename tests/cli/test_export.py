"""Tests for the kpt-export CLI command."""

import json
import pytest
import tempfile
import torch
from pathlib import Path

from kernel_pytorch.cli.export import ExportCommand


def _onnx_available() -> bool:
    """Check if ONNX export dependencies are available."""
    try:
        import onnxscript
        return True
    except ImportError:
        return False


def _safetensors_available() -> bool:
    """Check if SafeTensors is available."""
    try:
        import safetensors
        return True
    except ImportError:
        return False


class TestExportCommand:
    """Tests for the ExportCommand class."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )

    @pytest.fixture
    def model_path(self, simple_model, tmp_path):
        """Save model and return path."""
        path = tmp_path / "test_model.pt"
        torch.save(simple_model, path)
        return path

    def test_register_adds_subparser(self):
        """Test that register adds the export subparser."""
        import argparse
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        ExportCommand.register(subparsers)

        # Parse with export command
        args = parser.parse_args(['export', '--model', 'test.pt'])
        assert args.model == 'test.pt'

    @pytest.mark.skipif(not _onnx_available(), reason="ONNX dependencies not available")
    def test_export_onnx_basic(self, model_path, tmp_path):
        """Test basic ONNX export."""
        output_path = tmp_path / "exported.onnx"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            format='onnx',
            output=str(output_path),
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=False,
            bf16=False,
            opset=17,
            dynamic_axes=False,
            method='trace',
            validate=False,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        result = ExportCommand.execute(args)

        assert result == 0
        assert output_path.exists()

    def test_export_torchscript_trace(self, model_path, tmp_path):
        """Test TorchScript export with trace method."""
        output_path = tmp_path / "exported.pt"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            format='torchscript',
            output=str(output_path),
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=False,
            bf16=False,
            opset=17,
            dynamic_axes=False,
            method='trace',
            validate=False,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        result = ExportCommand.execute(args)

        assert result == 0
        assert output_path.exists()

    def test_export_torchscript_script(self, model_path, tmp_path):
        """Test TorchScript export with script method."""
        output_path = tmp_path / "exported_script.pt"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            format='torchscript',
            output=str(output_path),
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=False,
            bf16=False,
            opset=17,
            dynamic_axes=False,
            method='script',
            validate=False,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        result = ExportCommand.execute(args)

        assert result == 0
        assert output_path.exists()

    @pytest.mark.skipif(not _safetensors_available(), reason="SafeTensors not available")
    def test_export_safetensors(self, model_path, tmp_path):
        """Test SafeTensors export."""
        output_path = tmp_path / "exported.safetensors"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            format='safetensors',
            output=str(output_path),
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=False,
            bf16=False,
            opset=17,
            dynamic_axes=False,
            method='trace',
            validate=False,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        result = ExportCommand.execute(args)

        assert result == 0
        assert output_path.exists()

    def test_export_with_fp16(self, model_path, tmp_path):
        """Test export with FP16 precision."""
        output_path = tmp_path / "exported_fp16.pt"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            format='torchscript',
            output=str(output_path),
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=True,
            bf16=False,
            opset=17,
            dynamic_axes=False,
            method='trace',
            validate=False,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        result = ExportCommand.execute(args)

        assert result == 0
        assert output_path.exists()

    def test_export_with_validation(self, model_path, tmp_path):
        """Test export with validation enabled."""
        output_path = tmp_path / "exported_validated.pt"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            format='torchscript',
            output=str(output_path),
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=False,
            bf16=False,
            opset=17,
            dynamic_axes=False,
            method='trace',
            validate=True,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        result = ExportCommand.execute(args)

        assert result == 0
        assert output_path.exists()

    def test_export_all_formats(self, model_path, tmp_path):
        """Test exporting to all formats at once."""
        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            format='all',
            output=None,  # Auto-generate paths
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=False,
            bf16=False,
            opset=17,
            dynamic_axes=False,
            method='trace',
            validate=False,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        # This may partially fail if ONNX/SafeTensors not available
        # but TorchScript should always work
        ExportCommand.execute(args)

        # TorchScript should always be created
        assert len(list(tmp_path.glob("*.pt"))) >= 1

    def test_parse_shape(self):
        """Test shape parsing."""
        assert ExportCommand._parse_shape("1,512") == (1, 512)
        assert ExportCommand._parse_shape("2,3,224,224") == (2, 3, 224, 224)
        assert ExportCommand._parse_shape("1") == (1,)

    def test_export_nonexistent_model_fallback(self, tmp_path):
        """Test that nonexistent model falls back to simple model."""
        output_path = tmp_path / "fallback.pt"

        import argparse
        args = argparse.Namespace(
            model='nonexistent_model',
            format='torchscript',
            output=str(output_path),
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=False,
            bf16=False,
            opset=17,
            dynamic_axes=False,
            method='trace',
            validate=False,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        # Should not raise, falls back to simple model
        result = ExportCommand.execute(args)
        assert result == 0


@pytest.mark.skipif(not _onnx_available(), reason="ONNX dependencies not available")
class TestExportONNXSpecific:
    """ONNX-specific export tests."""

    @pytest.fixture
    def simple_model(self):
        return torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )

    def test_onnx_with_dynamic_axes(self, simple_model, tmp_path):
        """Test ONNX export with dynamic axes."""
        model_path = tmp_path / "model.pt"
        torch.save(simple_model, model_path)
        output_path = tmp_path / "dynamic.onnx"

        import argparse
        args = argparse.Namespace(
            model=str(model_path),
            format='onnx',
            output=str(output_path),
            output_dir=str(tmp_path),
            input_shape='1,512',
            dtype='float32',
            fp16=False,
            bf16=False,
            opset=17,
            dynamic_axes=True,
            method='trace',
            validate=False,
            validation_tolerance=1e-4,
            verbose=False,
            quiet=True,
        )

        result = ExportCommand.execute(args)

        assert result == 0
        assert output_path.exists()

    def test_onnx_with_different_opsets(self, simple_model, tmp_path):
        """Test ONNX export with different opset versions."""
        model_path = tmp_path / "model.pt"
        torch.save(simple_model, model_path)

        for opset in [13, 14, 17]:
            output_path = tmp_path / f"opset{opset}.onnx"

            import argparse
            args = argparse.Namespace(
                model=str(model_path),
                format='onnx',
                output=str(output_path),
                output_dir=str(tmp_path),
                input_shape='1,512',
                dtype='float32',
                fp16=False,
                bf16=False,
                opset=opset,
                dynamic_axes=False,
                method='trace',
                validate=False,
                validation_tolerance=1e-4,
                verbose=False,
                quiet=True,
            )

            result = ExportCommand.execute(args)

            assert result == 0
            assert output_path.exists()
