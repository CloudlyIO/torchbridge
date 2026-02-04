#!/usr/bin/env python3
"""
Regression Detection Engine for Performance Testing

Statistical detection of performance regressions with configurable thresholds,
significance testing, and severity classification.
"""

import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings

# Import from existing benchmark framework
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ..framework.benchmark_runner import PerformanceMetrics
from .baseline_manager import BaselineMetrics

class RegressionSeverity(Enum):
    """Severity levels for performance regressions"""
    NONE = "none"                    # No significant regression detected
    MINOR = "minor"                  # 2-5% degradation
    MAJOR = "major"                  # 5-10% degradation
    CRITICAL = "critical"            # >10% degradation

@dataclass
class RegressionResult:
    """Result of regression detection analysis"""
    model_name: str
    current_performance: PerformanceMetrics
    baseline_performance: BaselineMetrics
    performance_delta_percent: float
    throughput_delta_percent: float
    memory_delta_percent: float
    statistical_significance: bool
    severity: RegressionSeverity
    confidence_level: float
    recommendation: str
    timestamp: datetime
    raw_analysis: dict = None

    def is_regression(self) -> bool:
        """Check if this represents a performance regression"""
        return self.severity != RegressionSeverity.NONE

    def is_blocking(self) -> bool:
        """Check if this regression should block CI/deployment"""
        return self.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'model_name': self.model_name,
            'performance_delta_percent': self.performance_delta_percent,
            'throughput_delta_percent': self.throughput_delta_percent,
            'memory_delta_percent': self.memory_delta_percent,
            'statistical_significance': self.statistical_significance,
            'severity': self.severity.value,
            'confidence_level': self.confidence_level,
            'recommendation': self.recommendation,
            'timestamp': self.timestamp.isoformat(),
            'current_latency_ms': self.current_performance.latency_ms,
            'baseline_latency_ms': self.baseline_performance.mean_latency_ms,
            'current_throughput': self.current_performance.throughput_samples_per_sec,
            'baseline_throughput': self.baseline_performance.mean_throughput,
            'raw_analysis': self.raw_analysis or {}
        }

class RegressionDetector:
    """
    Detects performance regressions using statistical analysis.

    Provides configurable thresholds, statistical significance testing,
    and comprehensive regression analysis with severity classification.
    """

    def __init__(
        self,
        minor_threshold_percent: float = 2.0,
        major_threshold_percent: float = 5.0,
        critical_threshold_percent: float = 10.0,
        confidence_level: float = 0.95,
        min_sample_size: int = 3
    ):
        """
        Initialize regression detector with thresholds.

        Args:
            minor_threshold_percent: Threshold for minor regressions (2-5%)
            major_threshold_percent: Threshold for major regressions (5-10%)
            critical_threshold_percent: Threshold for critical regressions (>10%)
            confidence_level: Statistical confidence level (0.95 = 95%)
            min_sample_size: Minimum samples for statistical significance
        """
        self.minor_threshold = minor_threshold_percent
        self.major_threshold = major_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size

    def detect_regression(
        self,
        current: PerformanceMetrics,
        baseline: BaselineMetrics
    ) -> RegressionResult:
        """
        Detect performance regression by comparing current metrics to baseline.

        Args:
            current: Current performance metrics
            baseline: Baseline performance metrics

        Returns:
            RegressionResult with analysis and severity classification
        """
        # Calculate performance deltas
        latency_delta = self._calculate_performance_delta(
            current.latency_ms, baseline.mean_latency_ms
        )

        throughput_delta = self._calculate_performance_delta(
            current.throughput_samples_per_sec, baseline.mean_throughput, higher_is_better=True
        )

        memory_delta = self._calculate_performance_delta(
            current.peak_memory_mb, baseline.mean_memory_mb
        )

        # Determine statistical significance
        statistical_significance = self._analyze_statistical_significance(
            current, baseline
        )

        # Determine severity based on primary metric (latency)
        severity = self._determine_severity(latency_delta)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            latency_delta, throughput_delta, memory_delta, severity, statistical_significance
        )

        # Create detailed raw analysis
        raw_analysis = {
            'latency_analysis': {
                'current_ms': current.latency_ms,
                'baseline_mean_ms': baseline.mean_latency_ms,
                'baseline_std_ms': baseline.std_latency_ms,
                'delta_percent': latency_delta,
                'z_score': self._calculate_z_score(current.latency_ms, baseline.mean_latency_ms, baseline.std_latency_ms)
            },
            'throughput_analysis': {
                'current_ops_sec': current.throughput_samples_per_sec,
                'baseline_mean_ops_sec': baseline.mean_throughput,
                'baseline_std_ops_sec': baseline.std_throughput,
                'delta_percent': throughput_delta
            },
            'memory_analysis': {
                'current_mb': current.peak_memory_mb,
                'baseline_mean_mb': baseline.mean_memory_mb,
                'delta_percent': memory_delta
            },
            'baseline_quality': {
                'sample_count': baseline.sample_count,
                'confidence_interval_95': baseline.confidence_interval_95,
                'coefficient_of_variation': baseline.std_latency_ms / baseline.mean_latency_ms if baseline.mean_latency_ms > 0 else 0
            }
        }

        return RegressionResult(
            model_name=baseline.model_name,
            current_performance=current,
            baseline_performance=baseline,
            performance_delta_percent=latency_delta,
            throughput_delta_percent=throughput_delta,
            memory_delta_percent=memory_delta,
            statistical_significance=statistical_significance,
            severity=severity,
            confidence_level=self.confidence_level,
            recommendation=recommendation,
            timestamp=datetime.now(),
            raw_analysis=raw_analysis
        )

    def _calculate_performance_delta(
        self,
        current_value: float,
        baseline_value: float,
        higher_is_better: bool = False
    ) -> float:
        """
        Calculate percentage change in performance.

        Args:
            current_value: Current metric value
            baseline_value: Baseline metric value
            higher_is_better: True if higher values are better (e.g., throughput)

        Returns:
            Percentage change (positive = worse performance for lower_is_better metrics)
        """
        if baseline_value == 0:
            return 0.0

        delta_percent = ((current_value - baseline_value) / baseline_value) * 100

        # For "higher is better" metrics, invert the sign so that
        # positive delta still means "worse performance"
        if higher_is_better:
            delta_percent = -delta_percent

        return delta_percent

    def _calculate_z_score(self, current_value: float, mean: float, std: float) -> float:
        """Calculate z-score for statistical significance testing"""
        if std == 0:
            return 0.0
        return abs(current_value - mean) / std

    def _analyze_statistical_significance(
        self,
        current: PerformanceMetrics,
        baseline: BaselineMetrics
    ) -> bool:
        """
        Analyze if the performance difference is statistically significant.

        Args:
            current: Current performance metrics
            baseline: Baseline performance metrics

        Returns:
            True if the difference is statistically significant
        """
        # Check if we have sufficient baseline data
        if baseline.sample_count < self.min_sample_size:
            return False

        # Check if current measurement falls outside confidence interval
        if baseline.confidence_interval_95[0] <= current.latency_ms <= baseline.confidence_interval_95[1]:
            return False

        # Calculate z-score for latency (primary metric)
        if baseline.std_latency_ms > 0:
            z_score = self._calculate_z_score(
                current.latency_ms, baseline.mean_latency_ms, baseline.std_latency_ms
            )

            # For 95% confidence, z-score threshold is approximately 1.96
            confidence_threshold = 1.96 if self.confidence_level >= 0.95 else 1.64
            return z_score > confidence_threshold

        return True  # If no variance in baseline, any change is significant

    def _determine_severity(self, performance_delta_percent: float) -> RegressionSeverity:
        """
        Determine regression severity based on performance delta.

        Args:
            performance_delta_percent: Percentage change in performance

        Returns:
            RegressionSeverity classification
        """
        abs_delta = abs(performance_delta_percent)

        if abs_delta >= self.critical_threshold:
            return RegressionSeverity.CRITICAL
        elif abs_delta >= self.major_threshold:
            return RegressionSeverity.MAJOR
        elif abs_delta >= self.minor_threshold:
            return RegressionSeverity.MINOR
        else:
            return RegressionSeverity.NONE

    def _generate_recommendation(
        self,
        latency_delta: float,
        throughput_delta: float,
        memory_delta: float,
        severity: RegressionSeverity,
        statistical_significance: bool
    ) -> str:
        """Generate human-readable recommendation based on analysis"""

        if severity == RegressionSeverity.NONE:
            return "✅ No significant performance regression detected. Performance is within acceptable variance."

        # Build recommendation based on severity and metrics
        recommendations = []

        if severity == RegressionSeverity.CRITICAL:
            recommendations.append("🚨 CRITICAL regression detected - immediate investigation required.")
        elif severity == RegressionSeverity.MAJOR:
            recommendations.append("⚠️  MAJOR regression detected - review changes before merging.")
        else:
            recommendations.append("⚠️  Minor regression detected - monitor closely.")

        # Add specific metric analysis
        if abs(latency_delta) > self.minor_threshold:
            direction = "increased" if latency_delta > 0 else "decreased"
            recommendations.append(f"Latency {direction} by {abs(latency_delta):.1f}%.")

        if abs(throughput_delta) > self.minor_threshold:
            direction = "decreased" if throughput_delta > 0 else "increased"
            recommendations.append(f"Throughput {direction} by {abs(throughput_delta):.1f}%.")

        if abs(memory_delta) > self.minor_threshold:
            direction = "increased" if memory_delta > 0 else "decreased"
            recommendations.append(f"Memory usage {direction} by {abs(memory_delta):.1f}%.")

        # Add statistical significance note
        if not statistical_significance:
            recommendations.append("Note: Change may not be statistically significant - consider retesting.")

        # Add action recommendations
        if severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]:
            recommendations.append("Action: Review recent code changes, run additional benchmarks, consider reverting changes.")
        else:
            recommendations.append("Action: Monitor performance in subsequent benchmarks.")

        return " ".join(recommendations)

    def analyze_trend(self, historical_metrics: List[PerformanceMetrics]) -> dict:
        """
        Analyze performance trends over time.

        Args:
            historical_metrics: List of historical performance measurements

        Returns:
            Dictionary with trend analysis
        """
        if len(historical_metrics) < 3:
            return {"error": "Insufficient data for trend analysis"}

        latencies = [m.latency_ms for m in historical_metrics if m.latency_ms > 0]
        throughputs = [m.throughput_samples_per_sec for m in historical_metrics if m.throughput_samples_per_sec > 0]

        if not latencies:
            return {"error": "No valid latency data for trend analysis"}

        # Simple linear regression for trend detection
        x = np.arange(len(latencies))
        latency_slope = np.polyfit(x, latencies, 1)[0] if len(latencies) > 1 else 0
        throughput_slope = np.polyfit(x, throughputs, 1)[0] if len(throughputs) > 1 else 0

        # Calculate trend strength
        latency_trend_strength = abs(latency_slope) / np.mean(latencies) * 100 if latencies else 0
        throughput_trend_strength = abs(throughput_slope) / np.mean(throughputs) * 100 if throughputs else 0

        return {
            "latency_trend": {
                "slope_ms_per_measurement": latency_slope,
                "trend_strength_percent": latency_trend_strength,
                "direction": "degrading" if latency_slope > 0 else "improving"
            },
            "throughput_trend": {
                "slope_ops_per_measurement": throughput_slope,
                "trend_strength_percent": throughput_trend_strength,
                "direction": "degrading" if throughput_slope < 0 else "improving"
            },
            "measurements_analyzed": len(historical_metrics),
            "time_span": "recent measurements",
            "recommendation": self._generate_trend_recommendation(latency_trend_strength, throughput_trend_strength)
        }

    def _generate_trend_recommendation(self, latency_trend_strength: float, throughput_trend_strength: float) -> str:
        """Generate recommendation based on trend analysis"""
        if max(latency_trend_strength, throughput_trend_strength) > 5:
            return "⚠️  Significant performance trend detected - investigate systematic performance changes."
        elif max(latency_trend_strength, throughput_trend_strength) > 2:
            return "📈 Moderate performance trend detected - monitor closely."
        else:
            return "✅ No significant performance trend detected - performance is stable."

    def batch_analyze(
        self,
        measurements: List[Tuple[str, PerformanceMetrics]],
        baselines: dict
    ) -> List[RegressionResult]:
        """
        Analyze multiple measurements for regressions.

        Args:
            measurements: List of (model_name, metrics) tuples
            baselines: Dictionary mapping model names to BaselineMetrics

        Returns:
            List of RegressionResult objects
        """
        results = []

        for model_name, metrics in measurements:
            if model_name in baselines:
                result = self.detect_regression(metrics, baselines[model_name])
                results.append(result)
            else:
                warnings.warn(f"No baseline found for model: {model_name}")

        return results