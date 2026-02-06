#!/usr/bin/env python3
"""
Baseline Management System for Performance Regression Testing

Manages performance baselines with versioning, validation, and automatic
baseline establishment from historical benchmark data.
"""

import glob
import json
import os

# Import from existing benchmark framework
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ..framework.benchmark_runner import PerformanceMetrics


@dataclass
class BaselineMetrics:
    """Baseline performance metrics with statistical properties"""
    model_name: str
    mean_latency_ms: float
    std_latency_ms: float
    mean_throughput: float
    std_throughput: float
    mean_memory_mb: float
    std_memory_mb: float
    sample_count: int
    confidence_interval_95: tuple[float, float]
    established_date: datetime
    last_validated_date: datetime
    environment: dict[str, str] = field(default_factory=dict)
    version: str = "0.1.59"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert datetime to ISO format
        data['established_date'] = self.established_date.isoformat()
        data['last_validated_date'] = self.last_validated_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'BaselineMetrics':
        """Create from dictionary (JSON deserialization)"""
        # Convert ISO format back to datetime
        data['established_date'] = datetime.fromisoformat(data['established_date'])
        data['last_validated_date'] = datetime.fromisoformat(data['last_validated_date'])
        return cls(**data)

class BaselineManager:
    """
    Manages performance baselines with versioning and validation.

    Provides functionality to:
    - Establish baselines from historical benchmark data
    - Validate baseline quality and statistical significance
    - Store and retrieve baselines with versioning
    - Update baselines when performance legitimately improves
    """

    def __init__(self, baselines_dir: str = "benchmarks/baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.baselines_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.registry_file = self.baselines_dir / "baseline_registry.json"
        self._load_registry()

    def _load_registry(self):
        """Load the baseline registry"""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "version": "0.1.59",
                "created": datetime.now().isoformat(),
                "models": {},
                "last_updated": datetime.now().isoformat()
            }

    def _save_registry(self):
        """Save the baseline registry"""
        self.registry["last_updated"] = datetime.now().isoformat()
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def establish_baseline_from_historical_data(
        self,
        model_name: str,
        results_dir: str = "benchmarks/results",
        window_days: int = 30,
        min_samples: int = 10
    ) -> BaselineMetrics | None:
        """
        Establish baseline from historical benchmark data.

        Args:
            model_name: Name of the model to establish baseline for
            results_dir: Directory containing historical benchmark results
            window_days: Number of days to look back for baseline data
            min_samples: Minimum number of samples required for baseline

        Returns:
            BaselineMetrics if successful, None if insufficient data
        """
        # Find all relevant benchmark files
        pattern = f"{results_dir}/{model_name}_*_*.json"
        result_files = glob.glob(pattern)

        if not result_files:
            warnings.warn(f"No historical data found for model: {model_name}")
            return None

        # Load and filter results by time window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        valid_results = []

        for file_path in result_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)

                # Extract timestamp from filename or data
                timestamp = self._extract_timestamp(file_path, data)
                if timestamp and timestamp >= cutoff_date:
                    # Extract performance metrics
                    metrics = self._extract_metrics_from_result(data, model_name)
                    if metrics:
                        valid_results.append(metrics)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                warnings.warn(f"Failed to process {file_path}: {e}")
                continue

        if len(valid_results) < min_samples:
            warnings.warn(f"Insufficient samples for {model_name}: {len(valid_results)} < {min_samples}")
            return None

        # Calculate baseline statistics
        return self._calculate_baseline_statistics(model_name, valid_results)

    def _extract_timestamp(self, file_path: str, data: dict[str, Any]) -> datetime | None:
        """Extract timestamp from filename or benchmark data"""
        try:
            # Try to extract from filename (format: *_YYYYMMDD_HHMMSS.json)
            filename = Path(file_path).name
            parts = filename.split('_')
            if len(parts) >= 3:
                date_part = parts[-2]
                time_part = parts[-1].replace('.json', '')
                datetime_str = f"{date_part}_{time_part}"
                return datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            pass

        try:
            # Try to extract from data timestamp
            if 'timestamp' in data:
                return datetime.fromisoformat(data['timestamp'])
        except (ValueError, KeyError):
            pass

        return None

    def _extract_metrics_from_result(self, data: dict[str, Any], model_name: str) -> PerformanceMetrics | None:
        """Extract performance metrics from benchmark result data"""
        try:
            # Look for results in the data structure
            if 'results' in data:
                results = data['results']

                # Try different naming patterns for our optimizations
                for key in ['Our Optimizations', 'PyTorch Native', 'PyTorch Optimized']:
                    if key in results:
                        result_data = results[key]
                        return PerformanceMetrics(
                            latency_ms=float(result_data.get('latency_ms', 0.0)),
                            throughput_samples_per_sec=float(result_data.get('throughput_samples_per_sec', 0.0)),
                            peak_memory_mb=float(result_data.get('peak_memory_mb', 0.0)),
                            memory_efficiency=float(result_data.get('memory_efficiency', 1.0)),
                            accuracy_loss=float(result_data.get('accuracy_loss', 0.0)),
                            statistical_significance=bool(result_data.get('statistical_significance', True)),
                            confidence_interval_95=tuple(result_data.get('confidence_interval_95', (0.0, 0.0)))
                        )
        except (KeyError, ValueError, TypeError) as e:
            warnings.warn(f"Failed to extract metrics from result: {e}")
            return None

        return None

    def _calculate_baseline_statistics(
        self,
        model_name: str,
        metrics_list: list[PerformanceMetrics]
    ) -> BaselineMetrics:
        """Calculate baseline statistics from metrics list"""
        # Extract latency values
        latencies = [m.latency_ms for m in metrics_list if m.latency_ms > 0]
        throughputs = [m.throughput_samples_per_sec for m in metrics_list if m.throughput_samples_per_sec > 0]
        memories = [m.peak_memory_mb for m in metrics_list]

        # Calculate statistics
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        std_latency = float(np.std(latencies)) if len(latencies) > 1 else 0.0

        mean_throughput = float(np.mean(throughputs)) if throughputs else 0.0
        std_throughput = float(np.std(throughputs)) if len(throughputs) > 1 else 0.0

        mean_memory = float(np.mean(memories)) if memories else 0.0
        std_memory = float(np.std(memories)) if len(memories) > 1 else 0.0

        # Calculate 95% confidence interval for latency
        if len(latencies) > 1:
            confidence_interval = (
                mean_latency - 1.96 * std_latency / np.sqrt(len(latencies)),
                mean_latency + 1.96 * std_latency / np.sqrt(len(latencies))
            )
        else:
            confidence_interval = (mean_latency, mean_latency)

        return BaselineMetrics(
            model_name=model_name,
            mean_latency_ms=mean_latency,
            std_latency_ms=std_latency,
            mean_throughput=mean_throughput,
            std_throughput=std_throughput,
            mean_memory_mb=mean_memory,
            std_memory_mb=std_memory,
            sample_count=len(metrics_list),
            confidence_interval_95=confidence_interval,
            established_date=datetime.now(),
            last_validated_date=datetime.now(),
            environment={
                'python_version': sys.version.split()[0],
                'platform': sys.platform,
                'sample_period_days': 30
            }
        )

    def establish_baseline(self, model_name: str, metrics: PerformanceMetrics) -> bool:
        """
        Establish a new baseline from single metrics.

        Args:
            model_name: Name of the model
            metrics: Performance metrics to use as baseline

        Returns:
            True if baseline was established successfully
        """
        baseline = BaselineMetrics(
            model_name=model_name,
            mean_latency_ms=metrics.latency_ms,
            std_latency_ms=0.0,  # Single measurement, no variance
            mean_throughput=metrics.throughput_samples_per_sec,
            std_throughput=0.0,
            mean_memory_mb=metrics.peak_memory_mb,
            std_memory_mb=0.0,
            sample_count=1,
            confidence_interval_95=(metrics.latency_ms, metrics.latency_ms),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        return self._store_baseline(baseline)

    def get_baseline(self, model_name: str) -> BaselineMetrics | None:
        """
        Get the current baseline for a model.

        Args:
            model_name: Name of the model

        Returns:
            BaselineMetrics if found, None otherwise
        """
        baseline_file = self.models_dir / f"{model_name}_baseline.json"

        if not baseline_file.exists():
            return None

        try:
            with open(baseline_file) as f:
                data = json.load(f)
            return BaselineMetrics.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            warnings.warn(f"Failed to load baseline for {model_name}: {e}")
            return None

    def update_baseline(self, model_name: str, new_metrics: PerformanceMetrics) -> bool:
        """
        Update baseline with new performance metrics.

        Args:
            model_name: Name of the model
            new_metrics: New performance metrics

        Returns:
            True if baseline was updated successfully
        """
        current_baseline = self.get_baseline(model_name)

        if current_baseline is None:
            # No existing baseline, establish new one
            return self.establish_baseline(model_name, new_metrics)

        # Create updated baseline (for now, replace entirely)
        # In production, we might want to incorporate new data into running statistics
        updated_baseline = BaselineMetrics(
            model_name=model_name,
            mean_latency_ms=new_metrics.latency_ms,
            std_latency_ms=0.0,  # Reset variance for single update
            mean_throughput=new_metrics.throughput_samples_per_sec,
            std_throughput=0.0,
            mean_memory_mb=new_metrics.peak_memory_mb,
            std_memory_mb=0.0,
            sample_count=1,
            confidence_interval_95=(new_metrics.latency_ms, new_metrics.latency_ms),
            established_date=current_baseline.established_date,  # Keep original
            last_validated_date=datetime.now(),  # Update validation time
            environment=current_baseline.environment
        )

        return self._store_baseline(updated_baseline)

    def _store_baseline(self, baseline: BaselineMetrics) -> bool:
        """Store baseline to disk and update registry"""
        try:
            baseline_file = self.models_dir / f"{baseline.model_name}_baseline.json"

            with open(baseline_file, 'w') as f:
                json.dump(baseline.to_dict(), f, indent=2, default=str)

            # Update registry
            self.registry["models"][baseline.model_name] = {
                "baseline_file": str(baseline_file.relative_to(self.baselines_dir)),
                "established_date": baseline.established_date.isoformat(),
                "last_validated": baseline.last_validated_date.isoformat(),
                "sample_count": baseline.sample_count
            }
            self._save_registry()

            return True
        except (OSError, json.JSONEncodeError) as e:
            warnings.warn(f"Failed to store baseline for {baseline.model_name}: {e}")
            return False

    def validate_baseline_quality(self, metrics: BaselineMetrics) -> bool:
        """
        Validate baseline quality and statistical significance.

        Args:
            metrics: Baseline metrics to validate

        Returns:
            True if baseline meets quality requirements
        """
        # Check minimum sample size
        if metrics.sample_count < 5:
            return False

        # Check that we have valid performance data
        if metrics.mean_latency_ms <= 0 and metrics.mean_throughput <= 0:
            return False

        # Check confidence interval width (should be reasonable)
        if metrics.confidence_interval_95[1] - metrics.confidence_interval_95[0] > metrics.mean_latency_ms:
            return False

        # Check that standard deviation is reasonable (not too high)
        if metrics.mean_latency_ms > 0 and metrics.std_latency_ms / metrics.mean_latency_ms > 0.5:
            return False  # Coefficient of variation > 50%

        return True

    def get_historical_baselines(self, model_name: str, days: int = 30) -> list[BaselineMetrics]:
        """
        Get historical baselines for a model.

        Args:
            model_name: Name of the model
            days: Number of days to look back

        Returns:
            List of historical baselines
        """
        # For now, return current baseline
        # In production, we might maintain baseline history
        current = self.get_baseline(model_name)
        return [current] if current else []

    def list_available_models(self) -> list[str]:
        """Get list of models with established baselines"""
        return list(self.registry["models"].keys())

    def get_baseline_summary(self) -> dict[str, Any]:
        """Get summary of all baselines"""
        summary = {
            "total_models": len(self.registry["models"]),
            "registry_version": self.registry["version"],
            "last_updated": self.registry["last_updated"],
            "models": {}
        }

        for model_name in self.registry["models"]:
            baseline = self.get_baseline(model_name)
            if baseline:
                summary["models"][model_name] = {
                    "mean_latency_ms": baseline.mean_latency_ms,
                    "sample_count": baseline.sample_count,
                    "established_date": baseline.established_date.isoformat(),
                    "quality_validated": self.validate_baseline_quality(baseline)
                }

        return summary
