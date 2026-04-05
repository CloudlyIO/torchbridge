"""
Tests for CrossBackendTestSuite and QualificationReport

Verifies that the suite framework:
- Can be inherited and overridden
- Runs on all available backends
- Generates a QualificationReport
- Saves/loads reports correctly
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torchbridge.testing import (
    BackendTolerance,
    CrossBackendTestSuite,
    QualificationReport,
)


class SimpleLinearSuite(CrossBackendTestSuite):
    """Minimal test suite for a single Linear layer."""

    def build_model(self) -> nn.Module:
        torch.manual_seed(42)
        return nn.Linear(8, 4)

    def build_inputs(self) -> dict:
        torch.manual_seed(0)
        return {"x": torch.randn(4, 8)}

    def forward(self, model, inputs, device):
        return model(inputs["x"].to(device))


class TestCrossBackendSuiteInheritance:
    """Tests for CrossBackendTestSuite base class."""

    def test_suite_can_be_instantiated(self):
        """CrossBackendTestSuite subclass instantiates without error."""
        suite = SimpleLinearSuite()
        assert suite is not None

    def test_base_class_build_model_raises(self):
        """Base class build_model() raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            CrossBackendTestSuite().build_model()

    def test_base_class_build_inputs_raises(self):
        """Base class build_inputs() raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            CrossBackendTestSuite().build_inputs()

    def test_base_class_forward_raises(self):
        """Base class forward() raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            CrossBackendTestSuite().forward(None, {}, torch.device("cpu"))

    def test_suite_run_returns_results_list(self):
        """run() returns a non-empty list of BackendResult."""
        suite = SimpleLinearSuite()
        results = suite.run()
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_cpu_backend_always_passes(self):
        """CPU backend always passes (reference vs itself)."""
        suite = SimpleLinearSuite()
        results = suite.run()
        cpu_results = [r for r in results if r.backend_name == "cpu"]
        assert len(cpu_results) == 1
        # CPU compared to CPU reference: max_diff should be ~0
        assert cpu_results[0].max_diff is None or cpu_results[0].max_diff < 1e-6


class TestBackendTolerance:
    """Tests for BackendTolerance dataclass."""

    def test_default_values(self):
        tol = BackendTolerance()
        assert tol.atol == 1e-3
        assert tol.rtol == 1e-4
        assert tol.cosine_threshold == 0.999

    def test_for_backend_cuda(self):
        tol = BackendTolerance.for_backend("cuda", "float32")
        assert isinstance(tol, BackendTolerance)
        assert tol.atol > 0

    def test_for_backend_cpu(self):
        tol = BackendTolerance.for_backend("cpu", "float32")
        # CPU tolerance should be very tight
        assert tol.atol <= 1e-4


class TestQualificationReport:
    """Tests for QualificationReport."""

    def _make_results(self, backend_names: list[str], passed: list[bool]):
        from torchbridge.testing.suite import BackendResult

        return [
            BackendResult(
                backend_name=name,
                passed=p,
                max_diff=0.001 if p else 0.5,
                cosine_sim=1.0 if p else 0.9,
                latency_ms=10.0,
            )
            for name, p in zip(backend_names, passed)
        ]

    def test_from_results(self):
        """from_results() creates a QualificationReport."""
        results = self._make_results(["cuda", "cpu"], [True, True])
        report = QualificationReport.from_results("TestSuite", results)
        assert report.suite_name == "TestSuite"
        assert report.passed_count == 2
        assert report.total_count == 2
        assert report.all_passed is True

    def test_summary_pass(self):
        results = self._make_results(["cpu"], [True])
        report = QualificationReport.from_results("MySuite", results)
        assert "[PASS]" in report.summary()

    def test_summary_fail(self):
        results = self._make_results(["cuda", "cpu"], [False, True])
        report = QualificationReport.from_results("MySuite", results)
        assert "[FAIL]" in report.summary()
        assert "1/2" in report.summary()

    def test_save_and_load(self):
        """Reports can be saved to JSON and reloaded."""
        results = self._make_results(["cpu"], [True])
        report = QualificationReport.from_results("SaveTest", results)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.json"
            report.save(path)
            assert path.exists()

            loaded = QualificationReport.load(path)
            assert loaded.suite_name == "SaveTest"
            assert loaded.passed_count == 1

    def test_to_dict_keys(self):
        results = self._make_results(["cpu"], [True])
        report = QualificationReport.from_results("DictTest", results)
        d = report.to_dict()
        for key in (
            "suite_name",
            "timestamp",
            "passed",
            "total",
            "all_passed",
            "results",
        ):
            assert key in d

    def test_failed_backends(self):
        results = self._make_results(["cuda", "cpu"], [False, True])
        report = QualificationReport.from_results("FailTest", results)
        failed = report.failed_backends()
        assert "cuda" in failed
        assert "cpu" not in failed
