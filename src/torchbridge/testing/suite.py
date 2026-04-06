"""
CrossBackendTestSuite Base Class

Inherit from CrossBackendTestSuite and override forward() to run your model
on every available backend, comparing outputs to a CPU baseline.

Usage::

    from torchbridge.testing import CrossBackendTestSuite, BackendTolerance
    import torch
    import torch.nn as nn

    class TestMyModel(CrossBackendTestSuite):
        def build_model(self):
            return nn.Linear(64, 32)

        def build_inputs(self):
            return {"input": torch.randn(4, 64)}

        def forward(self, model, inputs, device):
            return model(inputs["input"].to(device))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BackendTolerance:
    """Numerical tolerance thresholds for cross-backend output comparisons."""

    atol: float = 1e-3
    """Absolute tolerance for ``torch.allclose``."""
    rtol: float = 1e-4
    """Relative tolerance for ``torch.allclose``."""
    cosine_threshold: float = 0.999
    """Minimum cosine similarity for flattened output vectors."""

    @classmethod
    def for_backend(
        cls, backend_name: str, dtype_name: str = "float32"
    ) -> BackendTolerance:
        """Look up empirical tolerances from the built-in tolerance DB."""
        from torchbridge.testing.tolerance_db import ToleranceDB

        db = ToleranceDB()
        pair = db.get(backend_name, dtype_name)
        return cls(atol=pair.atol, rtol=pair.rtol)


@dataclass
class BackendResult:
    """Result of running forward() on a single backend."""

    backend_name: str
    passed: bool
    max_diff: float | None = None
    cosine_sim: float | None = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CrossBackendTestSuite:
    """Base class for running a model on every available backend.

    Subclasses must implement :meth:`build_model`, :meth:`build_inputs`,
    and :meth:`forward`.  Call :meth:`run` to execute the suite and collect
    :class:`BackendResult` objects.

    Example::

        class TestLinear(CrossBackendTestSuite):
            def build_model(self):
                return torch.nn.Linear(32, 16)

            def build_inputs(self):
                return {"x": torch.randn(4, 32)}

            def forward(self, model, inputs, device):
                return model(inputs["x"].to(device))

        results = TestLinear().run()
        assert all(r.passed for r in results)
    """

    tolerance: BackendTolerance = BackendTolerance()

    def build_model(self) -> Any:
        """Return a fresh PyTorch model (called once per run)."""
        raise NotImplementedError("Subclasses must implement build_model()")

    def build_inputs(self) -> dict[str, Any]:
        """Return input tensors as a dict (called once per run)."""
        raise NotImplementedError("Subclasses must implement build_inputs()")

    def forward(self, model: Any, inputs: dict[str, Any], device: Any) -> Any:
        """Run model forward pass on the given device.

        Args:
            model: The model as returned by :meth:`build_model`.
            inputs: The inputs dict as returned by :meth:`build_inputs`.
            device: ``torch.device`` for the current backend.

        Returns:
            Output tensor (or dict of tensors).
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def run(self, reference_backend: str = "cpu") -> list[BackendResult]:
        """Run the suite on every available backend and compare to CPU baseline.

        Args:
            reference_backend: Backend to use as numerical reference (default "cpu").

        Returns:
            List of :class:`BackendResult`, one per available backend.
        """
        import time

        import torch
        import torch.nn.functional as F

        from torchbridge.backends.backend_factory import BackendFactory

        model = self.build_model()
        inputs = self.build_inputs()

        # Compute CPU reference output
        cpu_device = torch.device("cpu")
        model.eval()
        with torch.no_grad():
            cpu_output = self.forward(model, inputs, cpu_device)
        cpu_flat = self._flatten_output(cpu_output)

        results: list[BackendResult] = []

        available = BackendFactory.get_available_backends()
        for bt in available:
            try:
                backend = BackendFactory.create(bt)
                device = backend.device
                backend_name = bt.value
            except Exception as e:
                results.append(
                    BackendResult(
                        backend_name=str(bt),
                        passed=False,
                        error=f"Backend init failed: {e}",
                    )
                )
                continue

            try:
                test_model = self.build_model().to(device)
                test_model.eval()

                # Warmup
                with torch.no_grad():
                    self.forward(test_model, inputs, device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    output = self.forward(test_model, inputs, device)
                latency_ms = (time.perf_counter() - t0) * 1000

                out_flat = self._flatten_output(output).cpu()

                max_diff = float(torch.abs(cpu_flat - out_flat).max())
                cos_sim = float(
                    F.cosine_similarity(
                        cpu_flat.flatten().unsqueeze(0),
                        out_flat.flatten().unsqueeze(0),
                    )
                )

                tol = BackendTolerance.for_backend(backend_name)
                passed = max_diff <= tol.atol and cos_sim >= tol.cosine_threshold

                results.append(
                    BackendResult(
                        backend_name=backend_name,
                        passed=passed,
                        max_diff=max_diff,
                        cosine_sim=cos_sim,
                        latency_ms=latency_ms,
                    )
                )

            except Exception as e:
                results.append(
                    BackendResult(
                        backend_name=backend_name,
                        passed=False,
                        error=str(e),
                    )
                )
                logger.warning("Suite failed on %s: %s", backend_name, e)

        return results

    def _flatten_output(self, output: Any) -> Any:
        """Flatten model output to a 1-D or 2-D tensor for comparison."""
        import torch

        if isinstance(output, torch.Tensor):
            return output.detach().cpu().float()
        if isinstance(output, dict):
            # Use the first tensor value
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v.detach().cpu().float()
        if hasattr(output, "logits"):
            return output.logits.detach().cpu().float()
        raise ValueError(f"Cannot flatten output of type {type(output)}")
