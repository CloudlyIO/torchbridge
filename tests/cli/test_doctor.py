"""
Tests for the doctor CLI command.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

from kernel_pytorch.cli.doctor import DiagnosticResult, DoctorCommand


class TestDiagnosticResult:
    """Test DiagnosticResult dataclass."""

    def test_diagnostic_result_creation(self):
        """Test creating a DiagnosticResult."""
        result = DiagnosticResult(
            name="Test Check",
            status="pass",
            message="Everything is working",
            details="Additional details",
            recommendation="No action needed"
        )

        assert result.name == "Test Check"
        assert result.status == "pass"
        assert result.message == "Everything is working"
        assert result.details == "Additional details"
        assert result.recommendation == "No action needed"

    def test_diagnostic_result_minimal(self):
        """Test creating DiagnosticResult with minimal arguments."""
        result = DiagnosticResult(
            name="Simple Check",
            status="warning",
            message="Something might be wrong"
        )

        assert result.name == "Simple Check"
        assert result.status == "warning"
        assert result.message == "Something might be wrong"
        assert result.details is None
        assert result.recommendation is None


class TestDoctorCommand:
    """Test the doctor CLI command."""

    @patch('platform.python_version', return_value='3.10.0')
    def test_check_basic_requirements_python_good(self, mock_version):
        """Test basic requirements check with good Python version."""
        results = DoctorCommand._check_basic_requirements(verbose=False)

        python_result = next((r for r in results if r.name == "Python Version"), None)
        assert python_result is not None
        assert python_result.status == "pass"

    @patch('platform.python_version', return_value='3.7.0')
    def test_check_basic_requirements_python_old(self, mock_version):
        """Test basic requirements check with old Python version."""
        results = DoctorCommand._check_basic_requirements(verbose=False)

        python_result = next((r for r in results if r.name == "Python Version"), None)
        assert python_result is not None
        assert python_result.status == "fail"

    @patch('torch.__version__', '2.1.0')
    def test_check_basic_requirements_pytorch_good(self):
        """Test PyTorch version check with good version."""
        results = DoctorCommand._check_basic_requirements(verbose=False)

        pytorch_result = next((r for r in results if r.name == "PyTorch Version"), None)
        assert pytorch_result is not None
        assert pytorch_result.status == "pass"

    @patch('torch.__version__', '1.10.0')
    def test_check_basic_requirements_pytorch_old(self):
        """Test PyTorch version check with old version."""
        results = DoctorCommand._check_basic_requirements(verbose=False)

        pytorch_result = next((r for r in results if r.name == "PyTorch Version"), None)
        assert pytorch_result is not None
        assert pytorch_result.status == "fail"

    def test_check_basic_requirements_numpy(self):
        """Test NumPy availability check."""
        results = DoctorCommand._check_basic_requirements(verbose=False)

        numpy_result = next((r for r in results if r.name == "NumPy Version"), None)
        assert numpy_result is not None
        assert numpy_result.status == "pass"  # Should be available in test environment

    def test_check_basic_requirements_kernelpytorch(self):
        """Test KernelPyTorch installation check."""
        results = DoctorCommand._check_basic_requirements(verbose=False)

        kpt_result = next((r for r in results if r.name == "KernelPyTorch Version"), None)
        assert kpt_result is not None
        # Should pass since we're running tests from the package

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value='Tesla V100')
    @patch('torch.cuda.get_device_properties')
    def test_check_hardware_cuda_available(self, mock_props, mock_name, mock_available):
        """Test hardware check with CUDA available."""
        mock_props.return_value.total_memory = 16 * 1024**3
        mock_props.return_value.major = 7
        mock_props.return_value.minor = 0

        results = DoctorCommand._check_hardware(verbose=False)

        cuda_result = next((r for r in results if r.name == "CUDA GPU"), None)
        assert cuda_result is not None
        assert cuda_result.status == "pass"

    @patch('torch.cuda.is_available', return_value=False)
    def test_check_hardware_no_cuda(self, mock_available):
        """Test hardware check without CUDA."""
        results = DoctorCommand._check_hardware(verbose=False)

        cuda_result = next((r for r in results if r.name == "CUDA GPU"), None)
        assert cuda_result is not None
        assert cuda_result.status == "warning"

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_check_hardware_compute_capability(self, mock_props, mock_cuda_available):
        """Test GPU compute capability checking."""
        # Test modern GPU
        mock_props.return_value.major = 8
        mock_props.return_value.minor = 0
        mock_props.return_value.total_memory = 8 * 1024**3

        with patch('torch.cuda.get_device_name', return_value='RTX 3080'):
            results = DoctorCommand._check_hardware(verbose=False)

            compute_result = next((r for r in results if r.name == "GPU Compute Capability"), None)
            assert compute_result is not None
            assert compute_result.status == "pass"

        # Test older GPU
        mock_props.return_value.major = 5
        mock_props.return_value.minor = 0

        with patch('torch.cuda.get_device_name', return_value='GTX 960'):
            results = DoctorCommand._check_hardware(verbose=False)

            compute_result = next((r for r in results if r.name == "GPU Compute Capability"), None)
            assert compute_result is not None
            assert compute_result.status == "warning"

    @patch('torch.backends.mps.is_available', return_value=True)
    @patch('torch.cuda.is_available', return_value=False)
    def test_check_hardware_mps(self, mock_cuda, mock_mps):
        """Test Apple Silicon MPS detection."""
        results = DoctorCommand._check_hardware(verbose=False)

        mps_result = next((r for r in results if r.name == "Apple Silicon GPU"), None)
        assert mps_result is not None
        assert mps_result.status == "pass"

    @patch('multiprocessing.cpu_count', return_value=8)
    def test_check_hardware_cpu(self, mock_cpu_count):
        """Test CPU cores detection."""
        results = DoctorCommand._check_hardware(verbose=False)

        cpu_result = next((r for r in results if r.name == "CPU Cores"), None)
        assert cpu_result is not None
        assert cpu_result.status == "pass"

    @patch('torch.compile')
    def test_check_optimization_frameworks_torch_compile(self, mock_compile):
        """Test torch.compile availability."""
        # Mock torch.compile to be available
        mock_compile.return_value = MagicMock()

        results = DoctorCommand._check_optimization_frameworks(verbose=False)

        compile_result = next((r for r in results if r.name == "torch.compile"), None)
        assert compile_result is not None
        assert compile_result.status == "pass"

    def test_check_optimization_frameworks_torchscript(self):
        """Test TorchScript availability."""
        results = DoctorCommand._check_optimization_frameworks(verbose=False)

        jit_result = next((r for r in results if r.name == "TorchScript"), None)
        assert jit_result is not None
        assert jit_result.status == "pass"  # Should always be available

    def test_check_optimization_frameworks_kernelpytorch(self):
        """Test KernelPyTorch optimization framework availability."""
        results = DoctorCommand._check_optimization_frameworks(verbose=False)

        kpt_result = next((r for r in results if r.name == "KernelPyTorch Optimization"), None)
        assert kpt_result is not None
        # Should pass since we're running tests from the package

    @patch('subprocess.run')
    def test_check_advanced_features_cuda_toolkit(self, mock_run):
        """Test CUDA toolkit detection."""
        # Mock successful nvcc command
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Cuda compilation tools, release 11.8"

        results = DoctorCommand._check_advanced_features(verbose=False)

        cuda_result = next((r for r in results if r.name == "CUDA Toolkit"), None)
        assert cuda_result is not None
        assert cuda_result.status == "pass"

    @patch('subprocess.run')
    def test_check_advanced_features_no_cuda_toolkit(self, mock_run):
        """Test CUDA toolkit not found."""
        mock_run.side_effect = FileNotFoundError()

        results = DoctorCommand._check_advanced_features(verbose=False)

        cuda_result = next((r for r in results if r.name == "CUDA Toolkit"), None)
        assert cuda_result is not None
        assert cuda_result.status == "warning"

    def test_display_results(self):
        """Test displaying diagnostic results."""
        results = [
            DiagnosticResult("Test Pass", "pass", "Everything works"),
            DiagnosticResult("Test Warning", "warning", "Something might be wrong",
                           recommendation="Fix this issue"),
            DiagnosticResult("Test Fail", "fail", "Something is broken")
        ]

        # Should not raise any exceptions
        DoctorCommand._display_results(results, verbose=False)
        DoctorCommand._display_results(results, verbose=True)

    def test_generate_summary_all_pass(self):
        """Test summary generation with all passing tests."""
        results = [
            DiagnosticResult("Test 1", "pass", "OK"),
            DiagnosticResult("Test 2", "pass", "OK")
        ]

        summary = DoctorCommand._generate_summary(results)
        assert "2/2 checks passed" in summary
        assert "optimal" in summary.lower()

    def test_generate_summary_with_warnings(self):
        """Test summary generation with warnings."""
        results = [
            DiagnosticResult("Test 1", "pass", "OK"),
            DiagnosticResult("Test 2", "warning", "Warning")
        ]

        summary = DoctorCommand._generate_summary(results)
        assert "1/2 checks passed" in summary
        assert "1 warnings" in summary

    def test_generate_summary_with_failures(self):
        """Test summary generation with failures."""
        results = [
            DiagnosticResult("Test 1", "pass", "OK"),
            DiagnosticResult("Test 2", "fail", "Failed")
        ]

        summary = DoctorCommand._generate_summary(results)
        assert "1/2 checks passed" in summary
        assert "1 failures" in summary
        assert "Critical issues" in summary

    def test_attempt_fixes(self):
        """Test attempting to fix issues."""
        results = [
            DiagnosticResult("Test", "warning", "Issue",
                           recommendation="pip install package")
        ]

        # Should not raise exceptions
        DoctorCommand._attempt_fixes(results, verbose=False)

    def test_save_report(self):
        """Test saving diagnostic report."""
        results = [
            DiagnosticResult("Test", "pass", "OK", details="Some details")
        ]

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            try:
                DoctorCommand._save_report(results, f.name, verbose=False)
                assert os.path.exists(f.name)

                # Verify JSON content
                with open(f.name) as json_file:
                    data = json.load(json_file)

                assert 'timestamp' in data
                assert 'system_info' in data
                assert 'diagnostics' in data
                assert len(data['diagnostics']) == 1
                assert data['diagnostics'][0]['name'] == 'Test'
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_execute_basic_category(self):
        """Test executing basic diagnostics."""
        args = MagicMock()
        args.category = 'basic'
        args.full_report = False
        args.fix = False
        args.output = None
        args.verbose = False

        result = DoctorCommand.execute(args)
        assert result in [0, 1]  # Should complete, may have warnings

    def test_execute_hardware_category(self):
        """Test executing hardware diagnostics."""
        args = MagicMock()
        args.category = 'hardware'
        args.full_report = False
        args.fix = False
        args.output = None
        args.verbose = False

        result = DoctorCommand.execute(args)
        assert result in [0, 1]

    def test_execute_optimization_category(self):
        """Test executing optimization diagnostics."""
        args = MagicMock()
        args.category = 'optimization'
        args.full_report = False
        args.fix = False
        args.output = None
        args.verbose = False

        result = DoctorCommand.execute(args)
        assert result in [0, 1]

    def test_execute_advanced_category(self):
        """Test executing advanced diagnostics."""
        args = MagicMock()
        args.category = 'advanced'
        args.full_report = False
        args.fix = False
        args.output = None
        args.verbose = False

        result = DoctorCommand.execute(args)
        assert result in [0, 1]

    def test_execute_full_report(self):
        """Test executing full diagnostic report."""
        args = MagicMock()
        args.category = None
        args.full_report = True
        args.fix = False
        args.output = None
        args.verbose = False

        result = DoctorCommand.execute(args)
        assert result in [0, 1]

    def test_execute_quick_check(self):
        """Test executing quick check (default)."""
        args = MagicMock()
        args.category = None
        args.full_report = False
        args.fix = False
        args.output = None
        args.verbose = False

        result = DoctorCommand.execute(args)
        assert result in [0, 1]

    def test_execute_with_output(self):
        """Test executing diagnostics with output file."""
        args = MagicMock()
        args.category = 'basic'
        args.full_report = False
        args.fix = False
        args.verbose = False

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            args.output = f.name
            try:
                result = DoctorCommand.execute(args)
                assert result in [0, 1]
                assert os.path.exists(f.name)
            finally:
                if os.path.exists(f.name):
                    os.unlink(f.name)

    def test_execute_error_handling(self):
        """Test error handling in execute."""
        args = MagicMock()
        args.category = 'invalid_category'  # This will cause an attribute error
        args.full_report = False
        args.fix = False
        args.output = None
        args.verbose = False

        # Mock the category check to raise an exception
        with patch.object(DoctorCommand, '_check_basic_requirements', side_effect=Exception("Test error")):
            result = DoctorCommand.execute(args)
            assert result == 1
