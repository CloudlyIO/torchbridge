# SPDX-License-Identifier: Apache-2.0
"""
Hardware Qualification Report Generator

Converts :class:`~torchbridge.testing.suite.BackendResult` lists into
structured pass/fail JSON reports — same format used by the cloud validation
scripts in ``scripts/validation/``.

Usage::

    from torchbridge.testing import CrossBackendTestSuite, QualificationReport

    results = MySuite().run()
    report = QualificationReport.from_results("MyModel", results)
    report.save("reports/my_model_qualification.json")
    print(report.summary())
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QualificationReport:
    """Hardware qualification report for a model or test suite.

    Attributes:
        suite_name: Name of the test suite or model.
        timestamp: ISO-8601 timestamp when the report was generated.
        results: Per-backend result summaries.
        metadata: Arbitrary extra metadata (model name, version, etc.).
    """

    suite_name: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        suite_name: str,
        results: list[Any],  # list[BackendResult]
        metadata: dict[str, Any] | None = None,
    ) -> QualificationReport:
        """Build a report from a list of :class:`~torchbridge.testing.suite.BackendResult`.

        Args:
            suite_name: Human-readable name for the suite.
            results: List of ``BackendResult`` objects.
            metadata: Optional extra metadata dict.

        Returns:
            A new :class:`QualificationReport`.
        """
        serialized = []
        for r in results:
            if hasattr(r, "__dataclass_fields__"):
                serialized.append(asdict(r))
            else:
                serialized.append(dict(r))

        return cls(
            suite_name=suite_name,
            results=serialized,
            metadata=metadata or {},
        )

    @property
    def passed_count(self) -> int:
        """Number of backends that passed."""
        return sum(1 for r in self.results if r.get("passed", False))

    @property
    def total_count(self) -> int:
        """Total number of backends tested."""
        return len(self.results)

    @property
    def all_passed(self) -> bool:
        """True if every backend passed."""
        return self.passed_count == self.total_count and self.total_count > 0

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        status = "PASS" if self.all_passed else "FAIL"
        return (
            f"[{status}] {self.suite_name}: "
            f"{self.passed_count}/{self.total_count} backends passed"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dict."""
        return {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "passed": self.passed_count,
            "total": self.total_count,
            "all_passed": self.all_passed,
            "results": self.results,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        """Save the report as JSON to *path*.

        Args:
            path: File path (will create parent directories as needed).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("QualificationReport saved → %s", path)

    @classmethod
    def load(cls, path: str | Path) -> QualificationReport:
        """Load a report from a JSON file.

        Args:
            path: Path to a JSON file previously saved with :meth:`save`.

        Returns:
            :class:`QualificationReport` with loaded data.
        """
        path = Path(path)
        with path.open() as fh:
            data = json.load(fh)
        return cls(
            suite_name=data["suite_name"],
            timestamp=data.get("timestamp", ""),
            results=data.get("results", []),
            metadata=data.get("metadata", {}),
        )

    def failed_backends(self) -> list[str]:
        """Return names of all backends that failed."""
        return [
            r.get("backend_name", "unknown")
            for r in self.results
            if not r.get("passed", False)
        ]
