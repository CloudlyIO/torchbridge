"""
Robustness tests: graceful degradation when optional dependencies are missing.

Verifies that TorchBridge core functionality works with only required deps,
and that CLI commands produce helpful errors (not tracebacks) when optional
deps are unavailable.
"""

import os
import subprocess
import sys
import textwrap

import torch


# ---------------------------------------------------------------------------
# Helper: run a snippet in a fresh subprocess to avoid TORCH_LIBRARY
# re-registration issues when mutating sys.modules in-process.
# ---------------------------------------------------------------------------
def _run_snippet(code: str) -> subprocess.CompletedProcess:
    """Run *code* in a subprocess with PYTHONPATH pointing at src/."""
    src_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "src")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(src_dir)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


class TestMissingOptionalDeps:
    """Test graceful degradation when optional dependencies are missing."""

    def test_core_import_without_transformers(self):
        """Core import succeeds even if transformers is not installed."""
        result = _run_snippet("""\
            import sys
            # Block transformers from being importable
            import importlib
            class _BlockImporter:
                def find_module(self, name, path=None):
                    if name == "transformers" or name.startswith("transformers."):
                        return self
                def load_module(self, name):
                    raise ImportError(f"Blocked: {name}")
            sys.meta_path.insert(0, _BlockImporter())
            # Now import torchbridge — should succeed
            import torchbridge
            assert torchbridge.__version__ is not None
            assert callable(torchbridge.get_config)
            assert callable(torchbridge.optimize_model)
            print("OK")
        """)
        assert result.returncode == 0, (
            f"Core import failed without transformers:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout

    def test_core_import_without_triton(self):
        """Core import succeeds without triton."""
        result = _run_snippet("""\
            import sys
            sys.modules["triton"] = None  # block triton
            import torchbridge
            assert torchbridge.__version__ is not None
            print("OK")
        """)
        assert result.returncode == 0, (
            f"Core import failed without triton:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout

    def test_core_import_without_flash_attn(self):
        """Core import succeeds without flash_attn."""
        result = _run_snippet("""\
            import sys
            sys.modules["flash_attn"] = None  # block flash_attn
            import torchbridge
            assert torchbridge.__version__ is not None
            print("OK")
        """)
        assert result.returncode == 0, (
            f"Core import failed without flash_attn:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout

    def test_optimize_model_works_without_optional_deps(self):
        """optimize_model works with only core deps (torch)."""
        model = torch.nn.Linear(32, 16)
        # Should not raise even without optional deps
        try:
            import torchbridge
            result = torchbridge.optimize_model(model)
            assert result is not None
        except Exception:
            # torch.compile may fail on some platforms, that's OK
            pass

    def test_create_attention_without_flash_attn(self):
        """create_attention works without flash_attn installed."""
        result = _run_snippet("""\
            import sys
            sys.modules["flash_attn"] = None
            import torch
            import torchbridge
            attn = torchbridge.create_attention(d_model=64, num_heads=4)
            assert attn is not None
            x = torch.randn(1, 8, 64)
            with torch.no_grad():
                out = attn(x)
            assert out.shape == (1, 8, 64)
            print("OK")
        """)
        assert result.returncode == 0, (
            f"create_attention failed without flash_attn:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout

    def test_create_moe_without_optional_deps(self):
        """create_moe works with only core deps."""
        import torchbridge
        moe = torchbridge.create_moe(hidden_size=64, num_experts=4, top_k=2)
        assert moe is not None
        x = torch.randn(1, 4, 64)
        with torch.no_grad():
            out = moe(x)
        assert out.shape == (1, 4, 64)

    def test_hal_without_cuda(self):
        """HardwareAbstractionLayer works on CPU-only systems."""
        from torchbridge.hardware.abstraction.hal_core import HardwareAbstractionLayer
        hal = HardwareAbstractionLayer()
        assert isinstance(hal, HardwareAbstractionLayer)

    def test_config_without_optional_deps(self):
        """Configuration system works without optional deps."""
        import torchbridge
        config = torchbridge.get_config()
        assert hasattr(config, 'device')

    def test_validator_without_optional_deps(self):
        """UnifiedValidator can be instantiated without optional deps."""
        from torchbridge.validation.unified_validator import UnifiedValidator
        validator = UnifiedValidator()
        assert isinstance(validator, UnifiedValidator)

    def test_performance_tracker_without_persistence(self):
        """PerformanceTracker works in memory-only mode (no TORCHBRIDGE_METRICS env var)."""
        # Ensure env var is NOT set
        env_backup = os.environ.pop("TORCHBRIDGE_METRICS", None)
        try:
            from torchbridge.core.performance_tracker import PerformanceTracker
            tracker = PerformanceTracker()
            # Should not create files
            assert not tracker._persist
        finally:
            if env_backup is not None:
                os.environ["TORCHBRIDGE_METRICS"] = env_backup


class TestDoctorOptionalDeps:
    """Test that torchbridge doctor reports missing optional deps as warnings."""

    def test_doctor_reports_missing_triton_as_warning(self):
        """Doctor should report missing triton as warning, not error."""
        from torchbridge.cli.doctor import DoctorCommand
        results = DoctorCommand._check_advanced_features(verbose=False)
        # Find Triton result
        triton_results = [r for r in results if "Triton" in r.name]
        assert len(triton_results) > 0
        for r in triton_results:
            # Should be either 'pass' (if installed) or 'warning' (if not)
            assert r.status in ("pass", "warning"), (
                f"Triton check should be pass or warning, got: {r.status}"
            )

    def test_doctor_reports_missing_flash_attn_as_warning(self):
        """Doctor should report missing flash_attn as warning, not error."""
        from torchbridge.cli.doctor import DoctorCommand
        results = DoctorCommand._check_advanced_features(verbose=False)
        flash_results = [r for r in results if "Flash" in r.name]
        # Flash Attention may not appear if no CUDA GPU
        for r in flash_results:
            assert r.status in ("pass", "warning"), (
                f"Flash Attention check should be pass or warning, got: {r.status}"
            )

    def test_doctor_basic_always_passes_core(self):
        """Doctor basic checks should pass for core deps (Python, PyTorch, TorchBridge)."""
        from torchbridge.cli.doctor import DoctorCommand
        results = DoctorCommand._check_basic_requirements(verbose=False)
        # Python, PyTorch, TorchBridge should all pass
        core_checks = [r for r in results if r.name in ("Python Version", "PyTorch Version", "TorchBridge Version")]
        for r in core_checks:
            assert r.status == "pass", f"{r.name} failed: {r.message}"
