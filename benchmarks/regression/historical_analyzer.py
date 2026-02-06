#!/usr/bin/env python3
"""
Historical Performance Analysis Engine

Provides long-term performance trend analysis, anomaly detection,
and performance drift identification for regression testing.
"""

import glob
import json

# Import from existing components
import os
import statistics
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ..framework.benchmark_runner import PerformanceMetrics
from .baseline_manager import BaselineManager


class TrendDirection(Enum):
    """Performance trend directions"""
    IMPROVING = "improving"
    DEGRADING = "degrading"
    STABLE = "stable"
    VOLATILE = "volatile"


class AnomalyType(Enum):
    """Types of performance anomalies"""
    SPIKE = "spike"           # Sudden performance degradation
    DROP = "drop"             # Sudden performance improvement
    PLATEAU = "plateau"       # Extended period of consistent performance
    OSCILLATION = "oscillation"  # Regular performance fluctuation


@dataclass
class TrendAnalysis:
    """Results of performance trend analysis"""
    model_name: str
    metric_name: str
    trend_direction: TrendDirection
    trend_strength: float  # Percentage change over time period
    confidence_score: float  # Statistical confidence in trend
    slope_per_day: float
    r_squared: float  # Goodness of fit
    data_points: int
    time_period_days: int
    analysis_date: datetime
    raw_data: list[tuple[datetime, float]] = field(default_factory=list)

    def is_significant_trend(self, threshold: float = 2.0) -> bool:
        """Check if trend is statistically significant"""
        return abs(self.trend_strength) >= threshold and self.confidence_score >= 0.8


@dataclass
class AnomalyReport:
    """Performance anomaly detection result"""
    model_name: str
    metric_name: str
    anomaly_type: AnomalyType
    detection_date: datetime
    anomaly_value: float
    baseline_value: float
    deviation_percent: float
    severity_score: float  # 0-1 scale
    time_window: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSummary:
    """Comprehensive performance summary over time period"""
    model_name: str
    time_period: str
    start_date: datetime
    end_date: datetime

    # Performance statistics
    mean_latency_ms: float
    latency_trend: TrendAnalysis
    latency_volatility: float

    mean_throughput: float
    throughput_trend: TrendAnalysis
    throughput_volatility: float

    mean_memory_mb: float
    memory_trend: TrendAnalysis
    memory_volatility: float

    # Quality metrics
    total_measurements: int
    anomaly_count: int
    stability_score: float  # 0-1, higher = more stable
    reliability_score: float  # 0-1, higher = more reliable

    # Identified issues
    anomalies: list[AnomalyReport] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class HistoricalAnalyzer:
    """
    Analyzes long-term performance trends and identifies anomalies.

    Provides functionality to:
    - Detect gradual performance degradation over time
    - Identify performance anomalies and spikes
    - Analyze seasonal patterns in performance
    - Generate comprehensive performance summaries
    - Correlate performance changes with code changes
    """

    def __init__(self, baseline_manager: BaselineManager = None):
        self.baseline_manager = baseline_manager or BaselineManager()
        self.version = "0.1.59"

    def analyze_performance_trends(
        self,
        model_name: str,
        days: int = 90,
        results_dir: str = "benchmarks/results"
    ) -> dict[str, TrendAnalysis]:
        """
        Analyze performance trends over specified time period.

        Args:
            model_name: Name of the model to analyze
            days: Number of days to look back
            results_dir: Directory containing benchmark results

        Returns:
            Dictionary mapping metric names to trend analysis
        """
        print(f"📈 Analyzing {days}-day performance trends for {model_name}...")

        # Load historical data
        historical_data = self._load_historical_data(model_name, days, results_dir)

        if not historical_data:
            warnings.warn(f"No historical data found for {model_name}")
            return {}

        trends = {}

        # Analyze latency trends
        latency_data = [(date, metrics.latency_ms) for date, metrics in historical_data
                       if metrics.latency_ms > 0]
        if latency_data:
            trends["latency"] = self._calculate_trend_analysis(
                model_name, "latency_ms", latency_data, days
            )

        # Analyze throughput trends
        throughput_data = [(date, metrics.throughput_samples_per_sec) for date, metrics in historical_data
                          if metrics.throughput_samples_per_sec > 0]
        if throughput_data:
            trends["throughput"] = self._calculate_trend_analysis(
                model_name, "throughput_samples_per_sec", throughput_data, days
            )

        # Analyze memory trends
        memory_data = [(date, metrics.peak_memory_mb) for date, metrics in historical_data
                      if metrics.peak_memory_mb > 0]
        if memory_data:
            trends["memory"] = self._calculate_trend_analysis(
                model_name, "peak_memory_mb", memory_data, days
            )

        return trends

    def detect_performance_drift(
        self,
        model_name: str,
        window_days: int = 30,
        drift_threshold: float = 5.0,
        results_dir: str = "benchmarks/results"
    ) -> dict[str, Any]:
        """
        Detect gradual performance drift over time.

        Args:
            model_name: Name of the model
            window_days: Size of rolling window for drift detection
            drift_threshold: Percentage threshold for significant drift
            results_dir: Directory containing results

        Returns:
            Drift analysis report
        """
        print(f"🔍 Detecting performance drift for {model_name} (window: {window_days} days)...")

        # Load extended historical data
        historical_data = self._load_historical_data(model_name, window_days * 3, results_dir)

        if len(historical_data) < window_days:
            return {"error": f"Insufficient data for drift analysis: {len(historical_data)} < {window_days}"}

        # Sort by date
        historical_data.sort(key=lambda x: x[0])

        # Calculate rolling window statistics
        drift_analysis = {
            "model_name": model_name,
            "analysis_date": datetime.now(),
            "window_days": window_days,
            "drift_threshold_percent": drift_threshold,
            "metrics": {}
        }

        # Analyze latency drift
        latency_values = [metrics.latency_ms for _, metrics in historical_data if metrics.latency_ms > 0]
        if len(latency_values) >= window_days:
            drift_analysis["metrics"]["latency"] = self._calculate_metric_drift(
                latency_values, window_days, drift_threshold, "latency_ms"
            )

        # Analyze throughput drift
        throughput_values = [metrics.throughput_samples_per_sec for _, metrics in historical_data
                            if metrics.throughput_samples_per_sec > 0]
        if len(throughput_values) >= window_days:
            drift_analysis["metrics"]["throughput"] = self._calculate_metric_drift(
                throughput_values, window_days, drift_threshold, "throughput_samples_per_sec"
            )

        return drift_analysis

    def identify_performance_anomalies(
        self,
        model_name: str,
        days: int = 30,
        sensitivity: float = 2.0,
        results_dir: str = "benchmarks/results"
    ) -> list[AnomalyReport]:
        """
        Identify performance anomalies using statistical analysis.

        Args:
            model_name: Name of the model
            days: Number of days to analyze
            sensitivity: Standard deviation multiplier for anomaly detection
            results_dir: Directory containing results

        Returns:
            List of identified anomalies
        """
        print(f"🔍 Identifying performance anomalies for {model_name} (sensitivity: {sensitivity}σ)...")

        historical_data = self._load_historical_data(model_name, days, results_dir)

        if len(historical_data) < 10:
            warnings.warn(f"Insufficient data for anomaly detection: {len(historical_data)} < 10")
            return []

        anomalies = []

        # Detect latency anomalies
        latency_data = [(date, metrics.latency_ms) for date, metrics in historical_data
                       if metrics.latency_ms > 0]
        if latency_data:
            latency_anomalies = self._detect_metric_anomalies(
                model_name, "latency_ms", latency_data, sensitivity
            )
            anomalies.extend(latency_anomalies)

        # Detect throughput anomalies
        throughput_data = [(date, metrics.throughput_samples_per_sec) for date, metrics in historical_data
                          if metrics.throughput_samples_per_sec > 0]
        if throughput_data:
            throughput_anomalies = self._detect_metric_anomalies(
                model_name, "throughput_samples_per_sec", throughput_data, sensitivity, higher_is_better=True
            )
            anomalies.extend(throughput_anomalies)

        return sorted(anomalies, key=lambda x: x.severity_score, reverse=True)

    def generate_performance_summary(
        self,
        model_name: str,
        time_range: str = "30d",
        results_dir: str = "benchmarks/results"
    ) -> PerformanceSummary:
        """
        Generate comprehensive performance summary.

        Args:
            model_name: Name of the model
            time_range: Time range (e.g., "30d", "7d", "90d")
            results_dir: Directory containing results

        Returns:
            PerformanceSummary with comprehensive analysis
        """
        # Parse time range
        days = self._parse_time_range(time_range)

        print(f"📊 Generating performance summary for {model_name} ({time_range})...")

        # Load historical data
        historical_data = self._load_historical_data(model_name, days, results_dir)

        if not historical_data:
            raise ValueError(f"No historical data found for {model_name}")

        # Calculate date range
        dates = [date for date, _ in historical_data]
        start_date = min(dates)
        end_date = max(dates)

        # Extract metrics
        latency_values = [metrics.latency_ms for _, metrics in historical_data if metrics.latency_ms > 0]
        throughput_values = [metrics.throughput_samples_per_sec for _, metrics in historical_data
                            if metrics.throughput_samples_per_sec > 0]
        memory_values = [metrics.peak_memory_mb for _, metrics in historical_data if metrics.peak_memory_mb > 0]

        # Get trend analyses
        trends = self.analyze_performance_trends(model_name, days, results_dir)

        # Detect anomalies
        anomalies = self.identify_performance_anomalies(model_name, days, 2.0, results_dir)

        # Calculate stability and reliability scores
        stability_score = self._calculate_stability_score(latency_values, throughput_values, memory_values)
        reliability_score = self._calculate_reliability_score(historical_data, anomalies)

        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            trends, anomalies, stability_score, reliability_score
        )

        return PerformanceSummary(
            model_name=model_name,
            time_period=time_range,
            start_date=start_date,
            end_date=end_date,
            mean_latency_ms=float(np.mean(latency_values)) if latency_values else 0.0,
            latency_trend=trends.get("latency"),
            latency_volatility=float(np.std(latency_values) / np.mean(latency_values)) if latency_values else 0.0,
            mean_throughput=float(np.mean(throughput_values)) if throughput_values else 0.0,
            throughput_trend=trends.get("throughput"),
            throughput_volatility=float(np.std(throughput_values) / np.mean(throughput_values)) if throughput_values else 0.0,
            mean_memory_mb=float(np.mean(memory_values)) if memory_values else 0.0,
            memory_trend=trends.get("memory"),
            memory_volatility=float(np.std(memory_values) / np.mean(memory_values)) if memory_values else 0.0,
            total_measurements=len(historical_data),
            anomaly_count=len(anomalies),
            stability_score=stability_score,
            reliability_score=reliability_score,
            anomalies=anomalies,
            recommendations=recommendations
        )

    def _load_historical_data(
        self,
        model_name: str,
        days: int,
        results_dir: str
    ) -> list[tuple[datetime, PerformanceMetrics]]:
        """Load historical benchmark data for a model"""
        pattern = f"{results_dir}/{model_name}_*_*.json"
        result_files = glob.glob(pattern)

        cutoff_date = datetime.now() - timedelta(days=days)
        historical_data = []

        for file_path in result_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                # Extract timestamp and metrics
                timestamp = self.baseline_manager._extract_timestamp(file_path, data)
                if timestamp and timestamp >= cutoff_date:
                    metrics = self.baseline_manager._extract_metrics_from_result(data, model_name)
                    if metrics:
                        historical_data.append((timestamp, metrics))

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                warnings.warn(f"Failed to process {file_path}: {e}")
                continue

        return historical_data

    def _calculate_trend_analysis(
        self,
        model_name: str,
        metric_name: str,
        data_points: list[tuple[datetime, float]],
        time_period_days: int
    ) -> TrendAnalysis:
        """Calculate trend analysis for a specific metric"""
        if len(data_points) < 3:
            return TrendAnalysis(
                model_name=model_name,
                metric_name=metric_name,
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                confidence_score=0.0,
                slope_per_day=0.0,
                r_squared=0.0,
                data_points=len(data_points),
                time_period_days=time_period_days,
                analysis_date=datetime.now(),
                raw_data=data_points
            )

        # Sort by date and convert to numerical arrays
        sorted_data = sorted(data_points, key=lambda x: x[0])

        # Convert dates to days from start
        start_date = sorted_data[0][0]
        x_values = np.array([(date - start_date).days for date, _ in sorted_data])
        y_values = np.array([value for _, value in sorted_data])

        # Perform linear regression
        try:
            slope, intercept = np.polyfit(x_values, y_values, 1)

            # Calculate R-squared
            y_pred = slope * x_values + intercept
            ss_res = np.sum((y_values - y_pred) ** 2)
            ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Calculate trend strength (percentage change over period)
            initial_value = intercept
            final_value = slope * time_period_days + intercept
            trend_strength = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0.0

            # Determine trend direction
            if abs(trend_strength) < 1.0:
                direction = TrendDirection.STABLE
            elif trend_strength > 0:
                # For latency/memory, positive slope is degrading
                # For throughput, positive slope is improving
                direction = TrendDirection.DEGRADING if metric_name != "throughput_samples_per_sec" else TrendDirection.IMPROVING
            else:
                direction = TrendDirection.IMPROVING if metric_name != "throughput_samples_per_sec" else TrendDirection.DEGRADING

            # Calculate confidence based on R-squared and data points
            confidence_score = min(r_squared * (len(data_points) / 30.0), 1.0)

            return TrendAnalysis(
                model_name=model_name,
                metric_name=metric_name,
                trend_direction=direction,
                trend_strength=trend_strength,
                confidence_score=confidence_score,
                slope_per_day=float(slope),
                r_squared=float(r_squared),
                data_points=len(data_points),
                time_period_days=time_period_days,
                analysis_date=datetime.now(),
                raw_data=data_points
            )

        except (np.linalg.LinAlgError, ValueError) as e:
            warnings.warn(f"Failed to calculate trend for {metric_name}: {e}")
            return TrendAnalysis(
                model_name=model_name,
                metric_name=metric_name,
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                confidence_score=0.0,
                slope_per_day=0.0,
                r_squared=0.0,
                data_points=len(data_points),
                time_period_days=time_period_days,
                analysis_date=datetime.now(),
                raw_data=data_points
            )

    def _calculate_metric_drift(
        self,
        values: list[float],
        window_days: int,
        drift_threshold: float,
        metric_name: str
    ) -> dict[str, Any]:
        """Calculate drift analysis for a specific metric"""
        if len(values) < window_days * 2:
            return {"error": "Insufficient data for drift calculation"}

        # Calculate rolling window means
        early_window = values[:window_days]
        late_window = values[-window_days:]

        early_mean = statistics.mean(early_window)
        late_mean = statistics.mean(late_window)

        # Calculate drift percentage
        if early_mean > 0:
            drift_percent = ((late_mean - early_mean) / early_mean) * 100
        else:
            drift_percent = 0.0

        # Determine drift significance
        is_significant = abs(drift_percent) >= drift_threshold

        # Determine drift direction
        if abs(drift_percent) < 1.0:
            direction = "stable"
        elif drift_percent > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        return {
            "metric_name": metric_name,
            "early_period_mean": early_mean,
            "late_period_mean": late_mean,
            "drift_percent": drift_percent,
            "direction": direction,
            "is_significant": is_significant,
            "threshold_percent": drift_threshold
        }

    def _detect_metric_anomalies(
        self,
        model_name: str,
        metric_name: str,
        data_points: list[tuple[datetime, float]],
        sensitivity: float,
        higher_is_better: bool = False
    ) -> list[AnomalyReport]:
        """Detect anomalies in metric data using statistical methods"""
        if len(data_points) < 10:
            return []

        # Extract values and calculate statistics
        values = [value for _, value in data_points]
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0

        if std_value == 0:
            return []  # No variance, no anomalies

        anomalies = []
        threshold = sensitivity * std_value

        for date, value in data_points:
            deviation = abs(value - mean_value)

            if deviation > threshold:
                # Determine anomaly type
                if value > mean_value + threshold:
                    anomaly_type = AnomalyType.DROP if higher_is_better else AnomalyType.SPIKE
                else:
                    anomaly_type = AnomalyType.SPIKE if higher_is_better else AnomalyType.DROP

                # Calculate severity (0-1 scale)
                severity_score = min(deviation / (sensitivity * std_value * 2), 1.0)

                # Calculate deviation percentage
                deviation_percent = (deviation / mean_value) * 100 if mean_value > 0 else 0

                anomaly = AnomalyReport(
                    model_name=model_name,
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    detection_date=date,
                    anomaly_value=value,
                    baseline_value=mean_value,
                    deviation_percent=deviation_percent,
                    severity_score=severity_score,
                    time_window=f"{len(data_points)} measurements",
                    context={
                        "mean": mean_value,
                        "std": std_value,
                        "sensitivity": sensitivity,
                        "threshold": threshold
                    }
                )
                anomalies.append(anomaly)

        return anomalies

    def _calculate_stability_score(
        self,
        latency_values: list[float],
        throughput_values: list[float],
        memory_values: list[float]
    ) -> float:
        """Calculate overall performance stability score (0-1)"""
        stability_scores = []

        # Calculate coefficient of variation for each metric
        for values in [latency_values, throughput_values, memory_values]:
            if values and len(values) > 1:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values)
                cv = std_val / mean_val if mean_val > 0 else 0
                # Convert CV to stability (lower CV = higher stability)
                stability = max(0, 1 - (cv / 0.5))  # Normalize assuming 50% CV = 0 stability
                stability_scores.append(stability)

        return statistics.mean(stability_scores) if stability_scores else 0.5

    def _calculate_reliability_score(
        self,
        historical_data: list[tuple[datetime, PerformanceMetrics]],
        anomalies: list[AnomalyReport]
    ) -> float:
        """Calculate performance reliability score (0-1)"""
        if not historical_data:
            return 0.0

        # Base reliability on anomaly frequency
        anomaly_rate = len(anomalies) / len(historical_data)

        # Calculate reliability (fewer anomalies = higher reliability)
        reliability = max(0, 1 - (anomaly_rate * 2))  # Normalize assuming 50% anomaly rate = 0 reliability

        return min(reliability, 1.0)

    def _generate_performance_recommendations(
        self,
        trends: dict[str, TrendAnalysis],
        anomalies: list[AnomalyReport],
        stability_score: float,
        reliability_score: float
    ) -> list[str]:
        """Generate actionable performance recommendations"""
        recommendations = []

        # Trend-based recommendations
        for metric_name, trend in trends.items():
            if trend.is_significant_trend():
                if trend.trend_direction == TrendDirection.DEGRADING:
                    recommendations.append(f"📈 {metric_name.replace('_', ' ').title()} shows degrading trend ({trend.trend_strength:+.1f}% over {trend.time_period_days} days) - investigate recent changes")
                elif trend.trend_direction == TrendDirection.IMPROVING:
                    recommendations.append(f"✅ {metric_name.replace('_', ' ').title()} shows improving trend ({trend.trend_strength:+.1f}% over {trend.time_period_days} days)")

        # Stability recommendations
        if stability_score < 0.7:
            recommendations.append(f"⚠️ Performance stability is low ({stability_score:.1%}) - consider investigating environmental factors")
        elif stability_score > 0.9:
            recommendations.append(f"✅ Excellent performance stability ({stability_score:.1%})")

        # Reliability recommendations
        if reliability_score < 0.8:
            recommendations.append(f"🔍 Performance reliability needs attention ({reliability_score:.1%}) - {len(anomalies)} anomalies detected")

        # Anomaly-specific recommendations
        critical_anomalies = [a for a in anomalies if a.severity_score > 0.7]
        if critical_anomalies:
            recommendations.append(f"🚨 {len(critical_anomalies)} critical performance anomalies require investigation")

        if not recommendations:
            recommendations.append("✅ Performance appears stable with no significant issues detected")

        return recommendations

    def _parse_time_range(self, time_range: str) -> int:
        """Parse time range string to days"""
        time_range = time_range.lower()

        if time_range.endswith('d'):
            return int(time_range[:-1])
        elif time_range.endswith('w'):
            return int(time_range[:-1]) * 7
        elif time_range.endswith('m'):
            return int(time_range[:-1]) * 30
        else:
            # Default to 30 days
            return 30
