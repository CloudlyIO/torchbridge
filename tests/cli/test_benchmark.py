"""
Tests for the benchmark CLI command.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import torch

from torchbridge.cli.benchmark import BenchmarkCommand, BenchmarkResult


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a BenchmarkResult."""
        result = BenchmarkResult(
            name="test_model",
            mean_time_ms=10.5,
            std_time_ms=1.2,
            throughput_ops_per_sec=95.2,
            memory_usage_mb=128.0
        )

        assert result.name == "test_model"
        assert result.mean_time_ms == 10.5
        assert result.std_time_ms == 1.2
        assert result.throughput_ops_per_sec == 95.2
        assert result.memory_usage_mb == 128.0
        assert result.gpu_utilization_percent == 0.0  # Default value


class TestBenchmarkCommand:
    """Test the benchmark CLI command."""

    def test_detect_hardware_cuda(self):
        """Test hardware detection with CUDA."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_name', return_value='Tesla V100'):
                with patch('torch.cuda.get_device_properties') as mock_props:
                    mock_props.return_value.total_memory = 16 * 1024**3
                    device = BenchmarkCommand._detect_hardware(verbose=True)
                    assert device.type == 'cuda'

    def test_detect_hardware_mps(self):
        """Test hardware detection with MPS."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                device = BenchmarkCommand._detect_hardware(verbose=False)
                assert device.type == 'mps'

    def test_detect_hardware_cpu(self):
        """Test hardware detection with CPU only."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = BenchmarkCommand._detect_hardware(verbose=False)
                assert device.type == 'cpu'

    def test_load_model_linear_stress_test(self):
        """Test loading linear stress test model."""
        device = torch.device('cpu')
        model = BenchmarkCommand._load_model('linear_stress_test', device)
        assert isinstance(model, torch.nn.Linear)
        assert model.in_features == 1024
        assert model.out_features == 1024

    def test_load_model_resnet50(self):
        """Test loading ResNet50 model."""
        device = torch.device('cpu')

        # Create a mock module with resnet50
        mock_models = MagicMock()
        mock_model = MagicMock()
        mock_models.resnet50 = MagicMock(return_value=mock_model)

        # Mock torchvision at sys.modules level
        mock_torchvision = MagicMock()
        mock_torchvision.models = mock_models

        with patch.dict('sys.modules', {'torchvision': mock_torchvision, 'torchvision.models': mock_models}):
            BenchmarkCommand._load_model('resnet50', device)

            # Verify resnet50 was called with pretrained=False
            mock_models.resnet50.assert_called_once_with(pretrained=False)
            # Verify model was moved to device
            mock_model.to.assert_called_once_with(device)
            # Verify eval was called
            mock_model.to.return_value.eval.assert_called_once()

    def test_load_model_fallback(self):
        """Test loading model fallback."""
        device = torch.device('cpu')
        model = BenchmarkCommand._load_model('unknown_model', device)
        assert isinstance(model, torch.nn.Linear)

    def test_parse_input_shape_explicit(self):
        """Test parsing explicit input shape."""
        shape = BenchmarkCommand._parse_input_shape('1,3,224,224', 'resnet50')
        assert shape == (1, 3, 224, 224)

    def test_parse_input_shape_vision_model(self):
        """Test parsing shape for vision models."""
        shape = BenchmarkCommand._parse_input_shape(None, 'resnet50')
        assert shape == (1, 3, 224, 224)

    def test_parse_input_shape_transformer_model(self):
        """Test parsing shape for transformer models."""
        shape = BenchmarkCommand._parse_input_shape(None, 'transformer_model')
        assert shape == (1, 512, 768)

    def test_parse_input_shape_default(self):
        """Test parsing shape with default."""
        shape = BenchmarkCommand._parse_input_shape(None, 'unknown_model')
        assert shape == (16, 512)

    def test_apply_optimization_basic(self):
        """Test applying basic optimization."""
        model = torch.nn.Linear(10, 1)
        input_shape = (1, 10)
        device = torch.device('cpu')

        optimized = BenchmarkCommand._apply_optimization(model, 'basic', input_shape, device)
        assert optimized is model  # Should be same model

    def test_apply_optimization_jit(self):
        """Test applying JIT optimization."""
        model = torch.nn.Linear(10, 1).eval()
        input_shape = (1, 10)
        device = torch.device('cpu')

        optimized = BenchmarkCommand._apply_optimization(model, 'jit', input_shape, device)
        assert hasattr(optimized, 'graph')  # JIT traced model

    def test_apply_optimization_compile(self):
        """Test applying torch.compile optimization."""
        model = torch.nn.Linear(10, 1)
        input_shape = (1, 10)
        device = torch.device('cpu')

        with patch('torch.compile') as mock_compile:
            mock_compile.return_value = model
            BenchmarkCommand._apply_optimization(model, 'compile', input_shape, device)
            mock_compile.assert_called_with(model, mode='default')

    def test_benchmark_model(self):
        """Test benchmarking a single model."""
        model = torch.nn.Linear(10, 1)
        input_shape = (1, 10)
        device = torch.device('cpu')

        args = MagicMock()
        args.warmup = 2
        args.runs = 5

        result = BenchmarkCommand._benchmark_model(model, 'test_model', input_shape, device, args)

        assert isinstance(result, BenchmarkResult)
        assert result.name == 'test_model'
        assert result.mean_time_ms > 0
        assert result.throughput_ops_per_sec > 0

    def test_run_predefined_benchmarks_optimization(self):
        """Test running predefined optimization benchmarks."""
        args = MagicMock()
        args.predefined = 'optimization'
        args.verbose = False
        args.warmup = 2
        args.runs = 5
        device = torch.device('cpu')

        results = BenchmarkCommand._run_predefined_benchmarks(args, device)

        assert len(results) >= 1  # Should have at least one benchmark result
        assert all(hasattr(r, 'name') for r in results)
        assert all(hasattr(r, 'mean_time_ms') for r in results)
        assert all(r.mean_time_ms > 0 for r in results)  # Should have valid timing
        # Check that we have FusedGELU benchmark
        assert any('FusedGELU' in r.name for r in results)

    def test_run_predefined_benchmarks_transformers(self):
        """Test running predefined transformer benchmarks."""
        args = MagicMock()
        args.predefined = 'transformers'
        args.verbose = False
        args.warmup = 2
        args.runs = 5
        device = torch.device('cpu')

        results = BenchmarkCommand._run_predefined_benchmarks(args, device)

        assert len(results) >= 1  # Should have at least one transformer benchmark
        for result in results:
            assert 'Transformer' in result.name

    def test_benchmark_single_model(self):
        """Test benchmarking a single model."""
        args = MagicMock()
        args.model = 'linear_stress_test'
        args.input_shape = '16,1024'
        args.verbose = False
        args.warmup = 2
        args.runs = 5
        device = torch.device('cpu')

        results = BenchmarkCommand._benchmark_single_model(args, device)

        assert len(results) == 1
        assert results[0].name == args.model

    def test_compare_optimization_levels(self):
        """Test comparing optimization levels."""
        args = MagicMock()
        args.model = 'linear_stress_test'
        args.levels = 'basic,compile'
        args.input_shape = '16,1024'  # Match Linear(1024, 1024) input size
        args.verbose = False
        args.warmup = 2
        args.runs = 5
        device = torch.device('cpu')

        with patch('torch.compile') as mock_compile:
            mock_compile.return_value = torch.nn.Linear(1024, 1024)
            results = BenchmarkCommand._compare_optimization_levels(args, device)

            assert len(results) == 2  # basic and compile
            assert any('basic' in r.name for r in results)
            assert any('compile' in r.name for r in results)

    def test_stress_test(self):
        """Test stress testing with multiple batch sizes."""
        args = MagicMock()
        args.model = 'linear_stress_test'
        args.batch_sizes = '1,8,16'
        args.input_shape = None
        args.verbose = False
        args.warmup = 2
        args.runs = 5
        device = torch.device('cpu')

        results = BenchmarkCommand._stress_test(args, device)

        assert len(results) == 3  # Three batch sizes
        assert any('batch_1' in r.name for r in results)
        assert any('batch_8' in r.name for r in results)
        assert any('batch_16' in r.name for r in results)

    def test_display_results_verbose(self):
        """Test displaying results in verbose mode."""
        results = [
            BenchmarkResult(
                name="test_model",
                mean_time_ms=10.5,
                std_time_ms=1.2,
                throughput_ops_per_sec=95.2,
                memory_usage_mb=128.0
            )
        ]

        # Should not raise any exceptions
        BenchmarkCommand._display_results(results, verbose=True)

    def test_display_results_non_verbose(self):
        """Test displaying results in non-verbose mode."""
        results = [
            BenchmarkResult(
                name="test_model",
                mean_time_ms=10.5,
                std_time_ms=1.2,
                throughput_ops_per_sec=95.2,
                memory_usage_mb=128.0
            )
        ]

        # Should not raise any exceptions
        BenchmarkCommand._display_results(results, verbose=False)

    def test_save_results(self):
        """Test saving results to JSON file."""
        results = [
            BenchmarkResult(
                name="test_model",
                mean_time_ms=10.5,
                std_time_ms=1.2,
                throughput_ops_per_sec=95.2,
                memory_usage_mb=128.0
            )
        ]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            try:
                BenchmarkCommand._save_results(results, f.name, verbose=False)
                assert os.path.exists(f.name)

                # Verify JSON content
                with open(f.name) as json_file:
                    data = json.load(json_file)

                assert 'benchmark_results' in data
                assert len(data['benchmark_results']) == 1
                assert data['benchmark_results'][0]['name'] == 'test_model'
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_execute_single_model(self):
        """Test executing single model benchmark."""
        args = MagicMock()
        args.type = 'model'
        args.model = 'linear_stress_test'
        args.input_shape = '16,1024'
        args.predefined = None
        args.verbose = False
        args.warmup = 2
        args.runs = 5
        args.output = None
        args.quick = True

        result = BenchmarkCommand.execute(args)
        assert result == 0

    def test_execute_predefined_optimization(self):
        """Test executing predefined optimization benchmarks."""
        args = MagicMock()
        args.type = 'model'
        args.predefined = 'optimization'
        args.verbose = False
        args.warmup = 2
        args.runs = 5
        args.output = None
        args.quick = True

        result = BenchmarkCommand.execute(args)
        assert result == 0

    def test_execute_error_handling(self):
        """Test error handling in execute."""
        args = MagicMock()
        args.type = 'model'
        args.model = None  # Will cause error
        args.predefined = None
        args.verbose = False

        result = BenchmarkCommand.execute(args)
        assert result == 1
