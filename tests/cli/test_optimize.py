"""
Tests for the optimize CLI command.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from kernel_pytorch.cli.optimize import OptimizeCommand


class TestOptimizeCommand:
    """Test the optimize CLI command."""

    def test_detect_hardware_auto_cuda(self):
        """Test hardware detection with CUDA available."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value='Tesla V100'):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value.total_memory = 16 * 1024**3  # 16 GB
                    device = OptimizeCommand._detect_hardware('auto', verbose=True)
                    assert device.type == 'cuda'

    def test_detect_hardware_auto_mps(self):
        """Test hardware detection with MPS available."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = OptimizeCommand._detect_hardware('auto', verbose=False)
                assert device.type == 'mps'

    def test_detect_hardware_auto_cpu(self):
        """Test hardware detection falling back to CPU."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = OptimizeCommand._detect_hardware('auto', verbose=False)
                assert device.type == 'cpu'

    def test_detect_hardware_explicit(self):
        """Test explicit hardware selection."""
        device = OptimizeCommand._detect_hardware('cpu', verbose=False)
        assert device.type == 'cpu'

    def test_load_model_resnet50(self):
        """Test loading ResNet50 model."""
        # Create a mock module with resnet50
        mock_models = MagicMock()
        mock_model = MagicMock()
        # Make eval() return the model itself (as PyTorch does)
        mock_model.eval.return_value = mock_model
        mock_models.resnet50 = MagicMock(return_value=mock_model)

        # Mock torchvision at sys.modules level
        mock_torchvision = MagicMock()
        mock_torchvision.models = mock_models

        with patch.dict('sys.modules', {'torchvision': mock_torchvision, 'torchvision.models': mock_models}):
            model = OptimizeCommand._load_model('resnet50', verbose=False)
            mock_model.eval.assert_called_once()
            # The function returns the model after calling eval()
            assert model == mock_model

    def test_load_model_bert(self):
        """Test loading BERT-like model."""
        model = OptimizeCommand._load_model('bert', verbose=False)
        assert isinstance(model, torch.nn.Sequential)
        assert len(model) == 4  # Linear, GELU, Linear, LayerNorm

    def test_load_model_from_file(self):
        """Test loading model from file."""
        # Create a temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_model = torch.nn.Linear(10, 1)
            torch.save(temp_model, f.name)

            try:
                model = OptimizeCommand._load_model(f.name, verbose=False)
                assert isinstance(model, torch.nn.Linear)
            finally:
                os.unlink(f.name)

    def test_load_model_invalid(self):
        """Test loading invalid model."""
        with pytest.raises(ValueError):
            OptimizeCommand._load_model('nonexistent_model', verbose=False)

    def test_parse_input_shape_explicit(self):
        """Test parsing explicit input shape."""
        shape = OptimizeCommand._parse_input_shape('1,3,224,224', None)
        assert shape == (1, 3, 224, 224)

    def test_parse_input_shape_default(self):
        """Test default input shape."""
        model = torch.nn.Linear(10, 1)
        shape = OptimizeCommand._parse_input_shape(None, model)
        assert shape == (1, 3, 224, 224)  # Default shape

    def test_apply_optimizations_basic(self):
        """Test basic optimization level."""
        model = torch.nn.Linear(10, 1)
        sample_input = torch.randn(1, 10)

        optimized = OptimizeCommand._apply_optimizations(model, 'basic', sample_input, verbose=False)
        assert optimized is model  # Should return same model

    def test_apply_optimizations_jit(self):
        """Test JIT optimization level."""
        model = torch.nn.Linear(10, 1).eval()
        sample_input = torch.randn(1, 10)

        optimized = OptimizeCommand._apply_optimizations(model, 'jit', sample_input, verbose=False)
        assert hasattr(optimized, 'graph')  # JIT traced model has graph

    def test_apply_optimizations_compile(self):
        """Test torch.compile optimization level."""
        model = torch.nn.Linear(10, 1)
        sample_input = torch.randn(1, 10)

        with patch('torch.compile') as mock_compile:
            mock_compile.return_value = model
            OptimizeCommand._apply_optimizations(model, 'compile', sample_input, verbose=False)
            mock_compile.assert_called_with(model, mode='max-autotune')

    def test_apply_optimizations_production(self):
        """Test production optimization level."""
        model = torch.nn.Linear(10, 1)
        sample_input = torch.randn(1, 10)

        with patch('kernel_pytorch.utils.compiler_assistant.CompilerOptimizationAssistant') as mock_assistant:
            mock_result = MagicMock()
            mock_result.optimization_opportunities = []
            mock_assistant.return_value.optimize_model.return_value = mock_result

            with patch('torch.compile') as mock_compile:
                mock_compile.return_value = model
                OptimizeCommand._apply_optimizations(model, 'production', sample_input, verbose=False)
                mock_compile.assert_called_with(model, mode='max-autotune')

    def test_validate_optimization_success(self):
        """Test successful optimization validation."""
        model = torch.nn.Linear(10, 1)
        optimized_model = torch.nn.Linear(10, 1)

        # Set same weights for identical outputs
        optimized_model.load_state_dict(model.state_dict())

        sample_input = torch.randn(1, 10)

        # Should not raise an exception
        OptimizeCommand._validate_optimization(model, optimized_model, sample_input, verbose=False)

    def test_validate_optimization_shape_mismatch(self):
        """Test validation with shape mismatch."""
        original_model = torch.nn.Linear(10, 1)
        optimized_model = torch.nn.Linear(10, 2)  # Different output size
        sample_input = torch.randn(1, 10)

        with pytest.raises(ValueError, match="Output shape mismatch"):
            OptimizeCommand._validate_optimization(original_model, optimized_model, sample_input, verbose=False)

    def test_benchmark_models(self):
        """Test model benchmarking."""
        model1 = torch.nn.Linear(10, 1)
        model2 = torch.nn.Linear(10, 1)
        sample_input = torch.randn(1, 10)

        # This should run without errors
        OptimizeCommand._benchmark_models(model1, model2, sample_input, verbose=False)

    def test_save_model(self):
        """Test model saving."""
        model = torch.nn.Linear(10, 1)

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            try:
                OptimizeCommand._save_model(model, f.name, verbose=False)
                assert os.path.exists(f.name)

                # Verify model can be loaded
                loaded_model = torch.load(f.name, weights_only=False)
                assert isinstance(loaded_model, torch.nn.Linear)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_execute_full_workflow(self):
        """Test full optimization workflow."""
        # Mock arguments
        args = MagicMock()
        args.model = 'bert'
        args.level = 'basic'
        args.output = None
        args.input_shape = '1,512,768'
        args.hardware = 'cpu'
        args.benchmark = False
        args.validate = True
        args.verbose = False

        result = OptimizeCommand.execute(args)
        assert result == 0

    def test_execute_with_file_output(self):
        """Test optimization with file output."""
        args = MagicMock()
        args.model = 'bert'
        args.level = 'basic'
        args.input_shape = '1,512,768'
        args.hardware = 'cpu'
        args.benchmark = False
        args.validate = False
        args.verbose = False

        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            args.output = f.name
            try:
                result = OptimizeCommand.execute(args)
                assert result == 0
                assert os.path.exists(f.name)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_execute_with_benchmark(self):
        """Test optimization with benchmarking."""
        args = MagicMock()
        args.model = 'bert'
        args.level = 'basic'
        args.input_shape = '1,512,768'
        args.hardware = 'cpu'
        args.benchmark = True
        args.validate = False
        args.verbose = True
        args.output = None

        result = OptimizeCommand.execute(args)
        assert result == 0

    def test_execute_error_handling(self):
        """Test error handling in execute."""
        args = MagicMock()
        args.model = 'nonexistent_model'
        args.level = 'basic'
        args.hardware = 'cpu'
        args.verbose = False

        result = OptimizeCommand.execute(args)
        assert result == 1
