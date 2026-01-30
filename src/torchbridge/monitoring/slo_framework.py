"""
SLO/SLI Framework for TorchBridge

Provides Service Level Objectives (SLOs) and Service Level Indicators (SLIs)
for production monitoring and reliability tracking.

Key Concepts:
- SLI (Service Level Indicator): A quantitative measure of service behavior
- SLO (Service Level Objective): A target value or range for an SLI
- Error Budget: The acceptable amount of unreliability (100% - SLO target)
- Compliance: Whether the SLO is being met over a time window

Version: 0.4.33

Example:
    ```python
    from torchbridge.monitoring import (
        SLOManager, SLOConfig, SLIType
    )

    # Create SLO manager
    manager = SLOManager()

    # Define SLOs
    manager.add_slo(SLOConfig(
        name="inference_latency_p99",
        sli_type=SLIType.LATENCY_P99,
        target=100.0,  # 100ms p99 latency
        window_minutes=60,
    ))

    manager.add_slo(SLOConfig(
        name="availability",
        sli_type=SLIType.AVAILABILITY,
        target=99.9,  # 99.9% availability
        window_minutes=1440,  # 24 hours
    ))

    # Record measurements
    manager.record_latency(latency_ms=50.0)
    manager.record_request(success=True)

    # Check compliance
    report = manager.get_compliance_report()
    print(f"Error budget remaining: {report.error_budget_remaining_percent}%")
    ```
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .structured_logging import correlation_context, get_logger

logger = get_logger(__name__)


class SLIType(Enum):
    """Types of Service Level Indicators."""
    LATENCY_P50 = "latency_p50"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    LATENCY_MEAN = "latency_mean"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    THROUGHPUT = "throughput"
    SATURATION = "saturation"
    CUSTOM = "custom"


class ComplianceStatus(Enum):
    """SLO compliance status."""
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    VIOLATED = "violated"
    INSUFFICIENT_DATA = "insufficient_data"


class BudgetStatus(Enum):
    """Error budget status."""
    HEALTHY = "healthy"       # > 50% budget remaining
    WARNING = "warning"       # 10-50% budget remaining
    CRITICAL = "critical"     # < 10% budget remaining
    EXHAUSTED = "exhausted"   # 0% budget remaining


@dataclass
class SLOConfig:
    """Configuration for a Service Level Objective."""
    name: str
    sli_type: SLIType
    target: float  # Target value (e.g., 99.9 for availability, 100 for latency)
    window_minutes: int = 60  # Time window for measurement
    description: str = ""
    unit: str = ""
    comparison: str = "lte"  # "lte" (less than or equal), "gte" (greater than or equal)

    def __post_init__(self):
        """Set defaults based on SLI type."""
        if not self.description:
            self.description = f"{self.sli_type.value} SLO"

        if not self.unit:
            if self.sli_type in (SLIType.LATENCY_P50, SLIType.LATENCY_P95,
                                 SLIType.LATENCY_P99, SLIType.LATENCY_MEAN):
                self.unit = "ms"
            elif self.sli_type in (SLIType.AVAILABILITY, SLIType.ERROR_RATE):
                self.unit = "%"
            elif self.sli_type == SLIType.THROUGHPUT:
                self.unit = "req/s"

        # Set comparison direction based on SLI type
        if self.sli_type in (SLIType.AVAILABILITY, SLIType.THROUGHPUT):
            self.comparison = "gte"  # Want availability >= target
        else:
            self.comparison = "lte"  # Want latency/error_rate <= target


@dataclass
class SLIMeasurement:
    """A single SLI measurement."""
    timestamp: datetime
    value: float
    sli_type: SLIType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SLOStatus:
    """Current status of an SLO."""
    slo_name: str
    sli_type: SLIType
    target: float
    current_value: float | None
    compliance_status: ComplianceStatus
    error_budget_total: float
    error_budget_consumed: float
    error_budget_remaining: float
    error_budget_remaining_percent: float
    budget_status: BudgetStatus
    window_start: datetime
    window_end: datetime
    measurement_count: int
    is_compliant: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "slo_name": self.slo_name,
            "sli_type": self.sli_type.value,
            "target": self.target,
            "current_value": self.current_value,
            "compliance_status": self.compliance_status.value,
            "error_budget": {
                "total": self.error_budget_total,
                "consumed": self.error_budget_consumed,
                "remaining": self.error_budget_remaining,
                "remaining_percent": self.error_budget_remaining_percent,
            },
            "budget_status": self.budget_status.value,
            "window": {
                "start": self.window_start.isoformat(),
                "end": self.window_end.isoformat(),
            },
            "measurement_count": self.measurement_count,
            "is_compliant": self.is_compliant,
        }


@dataclass
class ComplianceReport:
    """Full SLO compliance report."""
    timestamp: datetime
    overall_status: ComplianceStatus
    slo_statuses: list[SLOStatus]
    compliant_count: int
    violated_count: int
    at_risk_count: int
    error_budget_burn_rate: float  # How fast budget is being consumed
    time_to_budget_exhaustion: timedelta | None
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "overall_status": self.overall_status.value,
            "summary": {
                "compliant": self.compliant_count,
                "violated": self.violated_count,
                "at_risk": self.at_risk_count,
            },
            "error_budget_burn_rate": self.error_budget_burn_rate,
            "time_to_budget_exhaustion": str(self.time_to_budget_exhaustion) if self.time_to_budget_exhaustion else None,
            "recommendations": self.recommendations,
            "slos": [s.to_dict() for s in self.slo_statuses],
        }


class SLICollector:
    """Collects and aggregates SLI measurements."""

    def __init__(self, max_measurements: int = 100000):
        self._latencies: deque[SLIMeasurement] = deque(maxlen=max_measurements)
        self._requests: deque[SLIMeasurement] = deque(maxlen=max_measurements)
        self._custom: dict[str, deque[SLIMeasurement]] = {}
        self._lock = threading.Lock()
        self._total_requests = 0
        self._successful_requests = 0

    def record_latency(
        self,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a latency measurement."""
        measurement = SLIMeasurement(
            timestamp=datetime.now(),
            value=latency_ms,
            sli_type=SLIType.LATENCY_MEAN,
            metadata=metadata or {},
        )
        with self._lock:
            self._latencies.append(measurement)

    def record_request(
        self,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a request (for availability/error rate)."""
        measurement = SLIMeasurement(
            timestamp=datetime.now(),
            value=1.0 if success else 0.0,
            sli_type=SLIType.AVAILABILITY,
            metadata=metadata or {},
        )
        with self._lock:
            self._requests.append(measurement)
            self._total_requests += 1
            if success:
                self._successful_requests += 1

    def record_custom(
        self,
        name: str,
        value: float,
        sli_type: SLIType = SLIType.CUSTOM,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a custom metric."""
        measurement = SLIMeasurement(
            timestamp=datetime.now(),
            value=value,
            sli_type=sli_type,
            metadata=metadata or {},
        )
        with self._lock:
            if name not in self._custom:
                self._custom[name] = deque(maxlen=10000)
            self._custom[name].append(measurement)

    def get_latencies(
        self,
        window_minutes: int | None = None,
    ) -> list[float]:
        """Get latency values within window."""
        with self._lock:
            measurements = list(self._latencies)

        if window_minutes is not None:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            measurements = [m for m in measurements if m.timestamp >= cutoff]

        return [m.value for m in measurements]

    def get_requests(
        self,
        window_minutes: int | None = None,
    ) -> list[SLIMeasurement]:
        """Get request measurements within window."""
        with self._lock:
            measurements = list(self._requests)

        if window_minutes is not None:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            measurements = [m for m in measurements if m.timestamp >= cutoff]

        return measurements

    def get_custom(
        self,
        name: str,
        window_minutes: int | None = None,
    ) -> list[float]:
        """Get custom metric values within window."""
        with self._lock:
            if name not in self._custom:
                return []
            measurements = list(self._custom[name])

        if window_minutes is not None:
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            measurements = [m for m in measurements if m.timestamp >= cutoff]

        return [m.value for m in measurements]

    def calculate_sli(
        self,
        sli_type: SLIType,
        window_minutes: int,
    ) -> float | None:
        """Calculate an SLI value."""
        if sli_type == SLIType.LATENCY_P50:
            latencies = self.get_latencies(window_minutes)
            if not latencies:
                return None
            return self._percentile(latencies, 50)

        elif sli_type == SLIType.LATENCY_P95:
            latencies = self.get_latencies(window_minutes)
            if not latencies:
                return None
            return self._percentile(latencies, 95)

        elif sli_type == SLIType.LATENCY_P99:
            latencies = self.get_latencies(window_minutes)
            if not latencies:
                return None
            return self._percentile(latencies, 99)

        elif sli_type == SLIType.LATENCY_MEAN:
            latencies = self.get_latencies(window_minutes)
            if not latencies:
                return None
            return statistics.mean(latencies)

        elif sli_type == SLIType.AVAILABILITY:
            requests = self.get_requests(window_minutes)
            if not requests:
                return None
            successful = sum(1 for r in requests if r.value == 1.0)
            return (successful / len(requests)) * 100

        elif sli_type == SLIType.ERROR_RATE:
            requests = self.get_requests(window_minutes)
            if not requests:
                return None
            errors = sum(1 for r in requests if r.value == 0.0)
            return (errors / len(requests)) * 100

        elif sli_type == SLIType.THROUGHPUT:
            requests = self.get_requests(window_minutes)
            if not requests:
                return None
            # Calculate requests per second
            return len(requests) / (window_minutes * 60)

        return None

    @staticmethod
    def _percentile(data: list[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1
        if upper >= len(sorted_data):
            return sorted_data[-1]
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


class SLOManager:
    """
    Manages Service Level Objectives and compliance tracking.

    Provides:
    - SLO definition and configuration
    - SLI collection and aggregation
    - Error budget calculation
    - Compliance reporting
    - Burn rate analysis
    """

    def __init__(self, max_measurements: int = 100000):
        """
        Initialize SLO manager.

        Args:
            max_measurements: Maximum measurements to retain per SLI
        """
        self._slos: dict[str, SLOConfig] = {}
        self._collector = SLICollector(max_measurements)
        self._lock = threading.Lock()
        self._compliance_history: deque[ComplianceReport] = deque(maxlen=10000)

    def add_slo(self, config: SLOConfig) -> None:
        """Add an SLO configuration."""
        with self._lock:
            self._slos[config.name] = config
            logger.info(
                "SLO added",
                slo_name=config.name,
                sli_type=config.sli_type.value,
                target=config.target,
            )

    def remove_slo(self, name: str) -> bool:
        """Remove an SLO configuration."""
        with self._lock:
            if name in self._slos:
                del self._slos[name]
                logger.info("SLO removed", slo_name=name)
                return True
            return False

    def get_slo(self, name: str) -> SLOConfig | None:
        """Get an SLO configuration."""
        return self._slos.get(name)

    def list_slos(self) -> list[SLOConfig]:
        """List all configured SLOs."""
        return list(self._slos.values())

    # Convenience methods for recording measurements
    def record_latency(
        self,
        latency_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a latency measurement."""
        self._collector.record_latency(latency_ms, metadata)

    def record_request(
        self,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a request (for availability/error rate)."""
        self._collector.record_request(success, metadata)

    def record_custom(
        self,
        name: str,
        value: float,
        sli_type: SLIType = SLIType.CUSTOM,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a custom metric."""
        self._collector.record_custom(name, value, sli_type, metadata)

    def get_slo_status(self, slo_name: str) -> SLOStatus | None:
        """Get current status of a specific SLO."""
        slo = self._slos.get(slo_name)
        if not slo:
            return None

        return self._calculate_slo_status(slo)

    def _calculate_slo_status(self, slo: SLOConfig) -> SLOStatus:
        """Calculate the status of an SLO."""
        now = datetime.now()
        window_start = now - timedelta(minutes=slo.window_minutes)

        # Get current SLI value
        current_value = self._collector.calculate_sli(
            slo.sli_type,
            slo.window_minutes,
        )

        # Get measurement count
        if slo.sli_type in (SLIType.LATENCY_P50, SLIType.LATENCY_P95,
                           SLIType.LATENCY_P99, SLIType.LATENCY_MEAN):
            measurements = self._collector.get_latencies(slo.window_minutes)
            measurement_count = len(measurements)
        else:
            measurements = self._collector.get_requests(slo.window_minutes)
            measurement_count = len(measurements)

        # Determine compliance
        if current_value is None or measurement_count < 10:
            compliance_status = ComplianceStatus.INSUFFICIENT_DATA
            is_compliant = True  # Assume compliant if no data
        else:
            if slo.comparison == "lte":
                is_compliant = current_value <= slo.target
            else:  # gte
                is_compliant = current_value >= slo.target

            if is_compliant:
                compliance_status = ComplianceStatus.COMPLIANT
            else:
                compliance_status = ComplianceStatus.VIOLATED

        # Calculate error budget
        if slo.sli_type == SLIType.AVAILABILITY:
            # For availability, error budget is (100 - target)%
            error_budget_total = 100 - slo.target
            if current_value is not None:
                error_budget_consumed = max(0, slo.target - current_value)
            else:
                error_budget_consumed = 0
        elif slo.sli_type == SLIType.ERROR_RATE:
            # For error rate, target IS the budget
            error_budget_total = slo.target
            if current_value is not None:
                error_budget_consumed = max(0, current_value)
            else:
                error_budget_consumed = 0
        else:
            # For latency, budget is percentage over target
            error_budget_total = slo.target * 0.1  # 10% margin
            if current_value is not None:
                error_budget_consumed = max(0, current_value - slo.target)
            else:
                error_budget_consumed = 0

        error_budget_remaining = max(0, error_budget_total - error_budget_consumed)
        error_budget_remaining_percent = (
            (error_budget_remaining / error_budget_total * 100)
            if error_budget_total > 0 else 100
        )

        # Determine budget status
        if error_budget_remaining_percent <= 0:
            budget_status = BudgetStatus.EXHAUSTED
            if compliance_status == ComplianceStatus.COMPLIANT:
                compliance_status = ComplianceStatus.AT_RISK
        elif error_budget_remaining_percent < 10:
            budget_status = BudgetStatus.CRITICAL
            if compliance_status == ComplianceStatus.COMPLIANT:
                compliance_status = ComplianceStatus.AT_RISK
        elif error_budget_remaining_percent < 50:
            budget_status = BudgetStatus.WARNING
        else:
            budget_status = BudgetStatus.HEALTHY

        return SLOStatus(
            slo_name=slo.name,
            sli_type=slo.sli_type,
            target=slo.target,
            current_value=current_value,
            compliance_status=compliance_status,
            error_budget_total=error_budget_total,
            error_budget_consumed=error_budget_consumed,
            error_budget_remaining=error_budget_remaining,
            error_budget_remaining_percent=error_budget_remaining_percent,
            budget_status=budget_status,
            window_start=window_start,
            window_end=now,
            measurement_count=measurement_count,
            is_compliant=is_compliant,
        )

    def get_compliance_report(self) -> ComplianceReport:
        """Generate a full compliance report for all SLOs."""
        with correlation_context():
            logger.debug("Generating compliance report")

            slo_statuses = []
            for slo in self._slos.values():
                status = self._calculate_slo_status(slo)
                slo_statuses.append(status)

            # Count statuses
            compliant_count = sum(
                1 for s in slo_statuses
                if s.compliance_status == ComplianceStatus.COMPLIANT
            )
            violated_count = sum(
                1 for s in slo_statuses
                if s.compliance_status == ComplianceStatus.VIOLATED
            )
            at_risk_count = sum(
                1 for s in slo_statuses
                if s.compliance_status == ComplianceStatus.AT_RISK
            )

            # Determine overall status
            if violated_count > 0:
                overall_status = ComplianceStatus.VIOLATED
            elif at_risk_count > 0:
                overall_status = ComplianceStatus.AT_RISK
            elif compliant_count == 0:
                overall_status = ComplianceStatus.INSUFFICIENT_DATA
            else:
                overall_status = ComplianceStatus.COMPLIANT

            # Calculate burn rate (how fast budget is being consumed)
            burn_rate = self._calculate_burn_rate(slo_statuses)

            # Estimate time to budget exhaustion
            time_to_exhaustion = self._estimate_time_to_exhaustion(
                slo_statuses, burn_rate
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(slo_statuses)

            report = ComplianceReport(
                timestamp=datetime.now(),
                overall_status=overall_status,
                slo_statuses=slo_statuses,
                compliant_count=compliant_count,
                violated_count=violated_count,
                at_risk_count=at_risk_count,
                error_budget_burn_rate=burn_rate,
                time_to_budget_exhaustion=time_to_exhaustion,
                recommendations=recommendations,
            )

            # Record in history
            self._compliance_history.append(report)

            logger.info(
                "Compliance report generated",
                overall_status=overall_status.value,
                compliant=compliant_count,
                violated=violated_count,
                at_risk=at_risk_count,
            )

            return report

    def _calculate_burn_rate(self, slo_statuses: list[SLOStatus]) -> float:
        """Calculate overall error budget burn rate."""
        if not slo_statuses:
            return 0.0

        # Average burn rate across all SLOs
        burn_rates = []
        for status in slo_statuses:
            if status.error_budget_total > 0:
                consumed_percent = (
                    status.error_budget_consumed / status.error_budget_total * 100
                )
                # Normalize by window (burn per hour)
                window_hours = (
                    status.window_end - status.window_start
                ).total_seconds() / 3600
                if window_hours > 0:
                    hourly_burn = consumed_percent / window_hours
                    burn_rates.append(hourly_burn)

        return statistics.mean(burn_rates) if burn_rates else 0.0

    def _estimate_time_to_exhaustion(
        self,
        slo_statuses: list[SLOStatus],
        burn_rate: float,
    ) -> timedelta | None:
        """Estimate time until error budget is exhausted."""
        if burn_rate <= 0:
            return None

        # Find minimum remaining budget percentage
        min_remaining = min(
            (s.error_budget_remaining_percent for s in slo_statuses),
            default=100
        )

        if min_remaining <= 0:
            return timedelta(0)

        # Estimate hours until exhaustion
        hours_to_exhaustion = min_remaining / burn_rate

        return timedelta(hours=hours_to_exhaustion)

    def _generate_recommendations(
        self,
        slo_statuses: list[SLOStatus],
    ) -> list[str]:
        """Generate recommendations based on SLO statuses."""
        recommendations = []

        for status in slo_statuses:
            if status.compliance_status == ComplianceStatus.VIOLATED:
                if status.sli_type in (SLIType.LATENCY_P50, SLIType.LATENCY_P95,
                                       SLIType.LATENCY_P99, SLIType.LATENCY_MEAN):
                    recommendations.append(
                        f"SLO '{status.slo_name}' violated: Latency is {status.current_value:.1f}ms "
                        f"(target: {status.target}ms). Consider optimizing model or scaling up."
                    )
                elif status.sli_type == SLIType.AVAILABILITY:
                    recommendations.append(
                        f"SLO '{status.slo_name}' violated: Availability is {status.current_value:.2f}% "
                        f"(target: {status.target}%). Investigate error causes."
                    )
                elif status.sli_type == SLIType.ERROR_RATE:
                    recommendations.append(
                        f"SLO '{status.slo_name}' violated: Error rate is {status.current_value:.2f}% "
                        f"(target: {status.target}%). Review error logs."
                    )

            elif status.budget_status in (BudgetStatus.CRITICAL, BudgetStatus.EXHAUSTED):
                recommendations.append(
                    f"SLO '{status.slo_name}' error budget is "
                    f"{'exhausted' if status.budget_status == BudgetStatus.EXHAUSTED else 'critical'} "
                    f"({status.error_budget_remaining_percent:.1f}% remaining). "
                    "Consider pausing deployments."
                )

        if not recommendations:
            recommendations.append("All SLOs are healthy. No immediate action required.")

        return recommendations

    def get_compliance_history(
        self,
        hours: int | None = None,
        limit: int | None = None,
    ) -> list[ComplianceReport]:
        """Get compliance report history."""
        reports = list(self._compliance_history)

        if hours is not None:
            cutoff = datetime.now() - timedelta(hours=hours)
            reports = [r for r in reports if r.timestamp >= cutoff]

        if limit is not None:
            reports = reports[-limit:]

        return reports


def create_slo_manager(
    max_measurements: int = 100000,
    default_slos: bool = True,
) -> SLOManager:
    """
    Create an SLO manager with optional default SLOs.

    Args:
        max_measurements: Maximum measurements to retain
        default_slos: Add default SLO configurations

    Returns:
        Configured SLOManager
    """
    manager = SLOManager(max_measurements)

    if default_slos:
        # Add common default SLOs
        manager.add_slo(SLOConfig(
            name="latency_p99",
            sli_type=SLIType.LATENCY_P99,
            target=200.0,  # 200ms p99 latency
            window_minutes=60,
            description="99th percentile inference latency",
        ))

        manager.add_slo(SLOConfig(
            name="availability",
            sli_type=SLIType.AVAILABILITY,
            target=99.9,  # 99.9% availability
            window_minutes=1440,  # 24 hours
            description="Service availability",
        ))

        manager.add_slo(SLOConfig(
            name="error_rate",
            sli_type=SLIType.ERROR_RATE,
            target=0.1,  # 0.1% error rate
            window_minutes=60,
            description="Request error rate",
        ))

    return manager


# Export all public APIs
__all__ = [
    "SLIType",
    "ComplianceStatus",
    "BudgetStatus",
    "SLOConfig",
    "SLIMeasurement",
    "SLOStatus",
    "ComplianceReport",
    "SLICollector",
    "SLOManager",
    "create_slo_manager",
]
