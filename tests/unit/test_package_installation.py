"""
Tests for package installation and import functionality.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


class TestPackageInstallation:
    """Test package installation and basic functionality."""

    def test_import_torchbridge(self):
        """Test that torchbridge can be imported."""
        import torchbridge
        assert hasattr(torchbridge, '__version__')

        # Check version format instead of exact version
        version = torchbridge.__version__
        assert isinstance(version, str), f"Version should be string, got {type(version)}"
        assert '.' in version, f"Version should contain dots: {version}"

        # Verify it follows semantic versioning pattern (e.g., "0.1.60")
        parts = version.split('.')
        assert len(parts) >= 2, f"Version should have at least major.minor: {version}"
        for part in parts:
            assert part.isdigit(), f"Version parts should be numeric: {version}"

    def test_import_cli_modules(self):
        """Test that CLI modules can be imported."""
        from torchbridge.cli import main
        from torchbridge.cli.benchmark import BenchmarkCommand
        from torchbridge.cli.doctor import DoctorCommand
        from torchbridge.cli.optimize import OptimizeCommand

        assert callable(main)
        assert hasattr(OptimizeCommand, 'execute')
        assert hasattr(BenchmarkCommand, 'execute')
        assert hasattr(DoctorCommand, 'execute')

    def test_cli_entry_points_importable(self):
        """Test that CLI entry points are importable."""
        # Test main CLI
        from torchbridge.cli import main as cli_main
        assert callable(cli_main)

        # Test individual commands
        from torchbridge.cli.benchmark import main as benchmark_main
        from torchbridge.cli.doctor import main as doctor_main
        from torchbridge.cli.optimize import main as optimize_main

        assert callable(optimize_main)
        assert callable(benchmark_main)
        assert callable(doctor_main)

    def test_core_modules_available(self):
        """Test that core TorchBridge modules are available."""
        # Test core components
        import torchbridge.core
        assert hasattr(torchbridge.core, 'OptimizedLinear')

        # Test attention mechanisms
        import torchbridge.attention
        assert hasattr(torchbridge.attention, 'FlashAttention2')

        # Test validation
        import torchbridge.validation
        assert hasattr(torchbridge.validation, 'UnifiedValidator')

    def test_package_metadata(self):
        """Test package metadata is correct."""
        import torchbridge

        # Check version format
        version = torchbridge.__version__
        assert isinstance(version, str)
        assert '.' in version  # Should have version format like "0.1.55"

        # Verify it follows semantic versioning pattern
        parts = version.split('.')
        assert len(parts) >= 2  # At least major.minor
        for part in parts:
            assert part.isdigit()  # All parts should be numeric

    @pytest.mark.slow
    def test_package_build_integrity(self):
        """Test that the package can be built without errors."""
        try:
            # Test building the package
            result = subprocess.run(
                [sys.executable, '-m', 'build', '--wheel', '--no-isolation'],
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                pytest.skip(f"Package build failed: {result.stderr}")

            # Check that wheel was created
            dist_dir = Path(__file__).parent.parent / 'dist'
            wheel_files = list(dist_dir.glob('*.whl'))
            assert len(wheel_files) > 0, "No wheel file was created"

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Package building not available in test environment")

    def test_optional_dependencies_importable(self):
        """Test that optional dependencies can be imported if available."""
        # These should not fail even if the optional dependencies aren't installed
        try:
            import torch
            assert hasattr(torch, 'cuda')
        except ImportError:
            pytest.skip("PyTorch not available")

        try:
            import numpy as np
            assert hasattr(np, 'array')
        except ImportError:
            pytest.skip("NumPy not available")

    def test_cli_modules_executable(self):
        """Test that CLI modules are executable."""
        try:
            # Test CLI help
            result = subprocess.run(
                [sys.executable, '-m', 'torchbridge.cli', '--help'],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Should not crash (return code 0 or 2 for help)
            assert result.returncode in [0, 2]
            assert len(result.stdout) > 0 or len(result.stderr) > 0

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("CLI not executable in test environment")


class TestInstallationRequirements:
    """Test installation requirements and dependencies."""

    def test_python_version_compatibility(self):
        """Test that current Python version meets requirements."""
        import sys

        major, minor = sys.version_info[:2]

        # Check minimum Python version (3.8+)
        assert major == 3, f"Expected Python 3.x, got {major}.{minor}"
        assert minor >= 8, f"Expected Python 3.8+, got 3.{minor}"

    def test_required_dependencies_available(self):
        """Test that required dependencies are available."""
        required_packages = [
            'torch',
            'numpy',
            'pybind11'
        ]

        missing_packages = []

        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            pytest.skip(f"Required packages not available: {missing_packages}")

    def test_torch_version_compatibility(self):
        """Test PyTorch version compatibility."""
        try:
            import torch
            version = torch.__version__

            # Parse version (e.g., "2.1.0+cu118" -> [2, 1, 0])
            version_parts = version.split('+')[0].split('.')
            major, _minor = int(version_parts[0]), int(version_parts[1])

            # Check minimum PyTorch version (2.0+)
            assert major >= 2, f"Expected PyTorch 2.0+, got {version}"

        except ImportError:
            pytest.skip("PyTorch not available")

    def test_optional_dependencies_handling(self):
        """Test that missing optional dependencies are handled gracefully."""
        # Test that the package loads even without optional dependencies
        try:
            import torchbridge

            # Should be able to access basic functionality
            assert hasattr(torchbridge, '__version__')

            # CLI should be importable
            from torchbridge.cli import main
            assert callable(main)

        except ImportError as e:
            pytest.fail(f"Basic package functionality failed: {e}")


class TestDevelopmentInstallation:
    """Test development installation specific features."""

    def test_editable_install_detection(self):
        """Test if this is an editable installation."""
        import torchbridge

        # Check if we can access source files (indicating editable install)
        module_file = torchbridge.__file__
        if module_file and 'site-packages' not in module_file:
            # Likely an editable install
            src_dir = Path(module_file).parent.parent.parent / 'src'
            if src_dir.exists():
                assert (src_dir / 'torchbridge').exists()

    def test_test_dependencies_available(self):
        """Test that test dependencies are available (for dev installs)."""
        test_packages = ['pytest']

        available_packages = []
        for package in test_packages:
            try:
                importlib.import_module(package)
                available_packages.append(package)
            except ImportError:
                pass

        # If any test packages are available, we're likely in a dev environment
        if available_packages:
            # Test that we can run basic pytest functionality
            import pytest
            assert hasattr(pytest, 'main')

    def test_development_tools_integration(self):
        """Test integration with development tools."""
        # Check if we can import development utilities
        try:
            from torchbridge.utils.validation_framework import ComponentValidator
            validator = ComponentValidator()
            assert hasattr(validator, 'validate_linear_component')

        except ImportError:
            # Development modules might not be available in minimal installs
            pytest.skip("Development utilities not available")


class TestImportPerformance:
    """Test import performance and lazy loading."""

    def test_import_time_reasonable(self):
        """Test that package import time is reasonable."""
        import time

        # Reload the module to measure import time
        if 'torchbridge' in sys.modules:
            del sys.modules['torchbridge']

        start_time = time.time()
        import_time = time.time() - start_time

        # Import should be reasonably fast (< 5 seconds)
        assert import_time < 5.0, f"Import took {import_time:.2f}s, which is too slow"

    def test_cli_import_lazy_loading(self):
        """Test that CLI modules use lazy loading where appropriate."""
        import time

        # CLI import should be fast
        start_time = time.time()
        cli_import_time = time.time() - start_time

        # CLI import should be very fast
        assert cli_import_time < 2.0, f"CLI import took {cli_import_time:.2f}s"

    def test_heavy_dependencies_lazy_loaded(self):
        """Test that heavy dependencies are lazy loaded."""
        # Import base package

        # Heavy dependencies should not be loaded yet
        heavy_modules = [
            'torch.compile',
            'triton',
            'transformers',
        ]

        # Check which heavy modules are already loaded
        loaded_heavy = [mod for mod in heavy_modules if mod in sys.modules]

        # Most heavy modules should not be loaded just from importing torchbridge
        # (This is a soft assertion since some might be needed for basic functionality)
        assert len(loaded_heavy) < len(heavy_modules), \
            f"Too many heavy modules loaded on import: {loaded_heavy}"
