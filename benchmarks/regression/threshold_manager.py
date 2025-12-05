#!/usr/bin/env python3
"""
Threshold Management for Performance Regression Testing

Dynamic threshold management with model-specific configurations,
automatic threshold tuning based on historical variance, and
environment-aware threshold adjustments.
"""

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

from .baseline_manager import BaselineMetrics

@dataclass
class ThresholdConfig:
    """Configuration for performance regression thresholds"""
    model_name: str
    latency_threshold_percent: float = 5.0
    memory_threshold_percent: float = 10.0
    throughput_threshold_percent: float = 5.0
    confidence_level: float = 0.95
    min_sample_size: int = 10
    minor_threshold_percent: float = 2.0
    major_threshold_percent: float = 5.0
    critical_threshold_percent: float = 10.0
    environment: str = "default"
    auto_tuned: bool = False
    last_updated: datetime = None
    version: str = "0.1.59"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdConfig':
        """Create from dictionary (JSON deserialization)"""
        if 'last_updated' in data and data['last_updated']:
            data['last_updated'] = datetime.fromisoformat(data['last_updated'])
        return cls(**data)

    def get_severity_thresholds(self) -> Tuple[float, float, float]:
        """Get (minor, major, critical) threshold tuple"""
        return (
            self.minor_threshold_percent,
            self.major_threshold_percent,
            self.critical_threshold_percent
        )

class ThresholdManager:
    """
    Manages dynamic performance thresholds for regression testing.

    Provides functionality to:
    - Maintain model-specific threshold configurations
    - Automatically tune thresholds based on historical variance
    - Adjust thresholds for different environments (CPU/GPU/Cloud)
    - Validate threshold sensitivity and effectiveness
    """

    def __init__(self, thresholds_dir: str = "benchmarks/baselines/thresholds"):
        self.thresholds_dir = Path(thresholds_dir)
        self.thresholds_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.thresholds_dir / "threshold_config.json"
        self.model_configs = {}
        self._load_configurations()

    def _load_configurations(self):
        """Load threshold configurations from disk"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)

                for model_name, config_data in data.get('models', {}).items():
                    self.model_configs[model_name] = ThresholdConfig.from_dict(config_data)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                warnings.warn(f"Failed to load threshold configurations: {e}")

    def _save_configurations(self):
        """Save threshold configurations to disk"""
        try:
            config_data = {
                "version": "0.1.59",
                "last_updated": datetime.now().isoformat(),
                "models": {
                    model_name: config.to_dict()
                    for model_name, config in self.model_configs.items()
                }
            }

            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except (OSError, json.JSONEncodeError) as e:
            warnings.warn(f"Failed to save threshold configurations: {e}")

    def get_thresholds(self, model_name: str, environment: str = "default") -> ThresholdConfig:
        """
        Get threshold configuration for a model and environment.

        Args:
            model_name: Name of the model
            environment: Environment identifier (e.g., "cpu", "gpu", "cloud")

        Returns:
            ThresholdConfig for the model, or default config if not found
        """
        config_key = f"{model_name}_{environment}"

        if config_key in self.model_configs:
            return self.model_configs[config_key]
        elif model_name in self.model_configs:
            return self.model_configs[model_name]
        else:
            # Return default configuration
            default_config = ThresholdConfig(
                model_name=model_name,
                environment=environment,
                last_updated=datetime.now()
            )
            self.model_configs[config_key] = default_config
            self._save_configurations()
            return default_config

    def update_thresholds_from_baseline(
        self,
        baseline: BaselineMetrics,
        environment: str = "default",
        sensitivity_factor: float = 2.0
    ) -> ThresholdConfig:
        """
        Update thresholds based on baseline performance variance.

        Args:
            baseline: Baseline metrics to derive thresholds from
            environment: Environment identifier
            sensitivity_factor: Multiplier for standard deviation (higher = less sensitive)

        Returns:
            Updated ThresholdConfig
        """
        config_key = f"{baseline.model_name}_{environment}"

        # Calculate adaptive thresholds based on baseline variance
        if baseline.std_latency_ms > 0 and baseline.mean_latency_ms > 0:
            # Use coefficient of variation to set thresholds
            cv = baseline.std_latency_ms / baseline.mean_latency_ms
            base_threshold = max(cv * 100 * sensitivity_factor, 2.0)  # Minimum 2%

            minor_threshold = base_threshold
            major_threshold = base_threshold * 2.5
            critical_threshold = base_threshold * 5.0
        else:
            # Use default thresholds if no variance data
            minor_threshold = 2.0
            major_threshold = 5.0
            critical_threshold = 10.0

        # Cap thresholds at reasonable maximums
        minor_threshold = min(minor_threshold, 5.0)
        major_threshold = min(major_threshold, 10.0)
        critical_threshold = min(critical_threshold, 20.0)

        # Create updated configuration
        updated_config = ThresholdConfig(
            model_name=baseline.model_name,
            latency_threshold_percent=major_threshold,
            memory_threshold_percent=major_threshold * 2,  # Memory is typically more variable
            throughput_threshold_percent=major_threshold,
            confidence_level=0.95,
            min_sample_size=max(baseline.sample_count // 2, 5),
            minor_threshold_percent=minor_threshold,
            major_threshold_percent=major_threshold,
            critical_threshold_percent=critical_threshold,
            environment=environment,
            auto_tuned=True,
            last_updated=datetime.now()
        )

        # Store updated configuration
        self.model_configs[config_key] = updated_config
        self._save_configurations()

        return updated_config

    def update_thresholds_from_history(
        self,
        model_name: str,
        historical_variances: List[float],
        environment: str = "default"
    ) -> ThresholdConfig:
        """
        Update thresholds based on historical performance variance.

        Args:
            model_name: Name of the model
            historical_variances: List of historical variance percentages
            environment: Environment identifier

        Returns:
            Updated ThresholdConfig
        """
        if not historical_variances or len(historical_variances) < 3:
            warnings.warn(f"Insufficient variance data for {model_name}")
            return self.get_thresholds(model_name, environment)

        # Calculate percentile-based thresholds
        variance_90th = np.percentile(historical_variances, 90)
        variance_95th = np.percentile(historical_variances, 95)
        variance_99th = np.percentile(historical_variances, 99)

        # Set thresholds based on historical variance
        minor_threshold = max(variance_90th, 2.0)
        major_threshold = max(variance_95th, 5.0)
        critical_threshold = max(variance_99th, 10.0)

        config_key = f"{model_name}_{environment}"
        updated_config = ThresholdConfig(
            model_name=model_name,
            latency_threshold_percent=major_threshold,
            memory_threshold_percent=major_threshold * 2,
            throughput_threshold_percent=major_threshold,
            minor_threshold_percent=minor_threshold,
            major_threshold_percent=major_threshold,
            critical_threshold_percent=critical_threshold,
            environment=environment,
            auto_tuned=True,
            last_updated=datetime.now()
        )

        self.model_configs[config_key] = updated_config
        self._save_configurations()

        return updated_config

    def validate_threshold_sensitivity(self, model_name: str, environment: str = "default") -> Dict[str, Any]:
        """
        Validate threshold sensitivity for a model.

        Args:
            model_name: Name of the model
            environment: Environment identifier

        Returns:
            Validation report with sensitivity analysis
        """
        config = self.get_thresholds(model_name, environment)

        # Analyze threshold spacing
        thresholds = config.get_severity_thresholds()
        threshold_gaps = [
            thresholds[1] - thresholds[0],  # major - minor
            thresholds[2] - thresholds[1]   # critical - major
        ]

        # Check for reasonable threshold progression
        reasonable_spacing = all(gap >= 1.0 for gap in threshold_gaps)
        reasonable_values = all(0.5 <= t <= 20.0 for t in thresholds)

        validation = {
            "model_name": model_name,
            "environment": environment,
            "thresholds": {
                "minor": thresholds[0],
                "major": thresholds[1],
                "critical": thresholds[2]
            },
            "threshold_gaps": threshold_gaps,
            "validation": {
                "reasonable_spacing": reasonable_spacing,
                "reasonable_values": reasonable_values,
                "auto_tuned": config.auto_tuned,
                "last_updated": config.last_updated.isoformat() if config.last_updated else None
            },
            "recommendations": []
        }

        # Add recommendations
        if not reasonable_spacing:
            validation["recommendations"].append("Consider increasing threshold gaps for better sensitivity")

        if not reasonable_values:
            validation["recommendations"].append("Some thresholds are outside reasonable ranges (0.5%-20%)")

        if not config.auto_tuned:
            validation["recommendations"].append("Consider auto-tuning thresholds based on baseline variance")

        if not validation["recommendations"]:
            validation["recommendations"].append("Thresholds appear well-configured")

        return validation

    def get_environment_multipliers(self, environment: str) -> Dict[str, float]:
        """
        Get environment-specific threshold multipliers.

        Args:
            environment: Environment identifier

        Returns:
            Dictionary of multipliers for different metrics
        """
        multipliers = {
            "default": {"latency": 1.0, "memory": 1.0, "throughput": 1.0},
            "cpu": {"latency": 1.2, "memory": 0.8, "throughput": 1.2},  # CPU more variable latency
            "gpu": {"latency": 0.9, "memory": 1.5, "throughput": 0.9},  # GPU more variable memory
            "cloud": {"latency": 1.5, "memory": 1.3, "throughput": 1.4}, # Cloud has more variance
            "ci": {"latency": 2.0, "memory": 1.5, "throughput": 2.0}   # CI environments very variable
        }

        return multipliers.get(environment.lower(), multipliers["default"])

    def apply_environment_adjustments(
        self,
        base_config: ThresholdConfig,
        environment: str
    ) -> ThresholdConfig:
        """
        Apply environment-specific adjustments to threshold configuration.

        Args:
            base_config: Base threshold configuration
            environment: Target environment

        Returns:
            Adjusted ThresholdConfig
        """
        multipliers = self.get_environment_multipliers(environment)

        adjusted_config = ThresholdConfig(
            model_name=base_config.model_name,
            latency_threshold_percent=base_config.latency_threshold_percent * multipliers["latency"],
            memory_threshold_percent=base_config.memory_threshold_percent * multipliers["memory"],
            throughput_threshold_percent=base_config.throughput_threshold_percent * multipliers["throughput"],
            confidence_level=base_config.confidence_level,
            min_sample_size=base_config.min_sample_size,
            minor_threshold_percent=base_config.minor_threshold_percent * multipliers["latency"],
            major_threshold_percent=base_config.major_threshold_percent * multipliers["latency"],
            critical_threshold_percent=base_config.critical_threshold_percent * multipliers["latency"],
            environment=environment,
            auto_tuned=base_config.auto_tuned,
            last_updated=datetime.now()
        )

        return adjusted_config

    def export_threshold_config(self) -> Dict[str, Any]:
        """
        Export all threshold configurations.

        Returns:
            Dictionary with all threshold configurations
        """
        return {
            "version": "0.1.59",
            "exported_at": datetime.now().isoformat(),
            "total_configurations": len(self.model_configs),
            "configurations": {
                model_name: config.to_dict()
                for model_name, config in self.model_configs.items()
            }
        }

    def import_threshold_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Import threshold configurations from external source.

        Args:
            config_data: Configuration data to import

        Returns:
            True if import was successful
        """
        try:
            for model_name, config_dict in config_data.get('configurations', {}).items():
                self.model_configs[model_name] = ThresholdConfig.from_dict(config_dict)

            self._save_configurations()
            return True
        except (KeyError, ValueError, TypeError) as e:
            warnings.warn(f"Failed to import threshold configurations: {e}")
            return False

    def cleanup_old_configurations(self, days_old: int = 90):
        """
        Clean up threshold configurations that haven't been used recently.

        Args:
            days_old: Remove configurations older than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        to_remove = []

        for model_name, config in self.model_configs.items():
            if config.last_updated and config.last_updated < cutoff_date:
                to_remove.append(model_name)

        for model_name in to_remove:
            del self.model_configs[model_name]
            warnings.warn(f"Removed old threshold configuration for {model_name}")

        if to_remove:
            self._save_configurations()

    def get_threshold_summary(self) -> Dict[str, Any]:
        """Get summary of all threshold configurations"""
        summary = {
            "total_models": len(self.model_configs),
            "auto_tuned_count": sum(1 for config in self.model_configs.values() if config.auto_tuned),
            "environments": list(set(config.environment for config in self.model_configs.values())),
            "average_thresholds": {},
            "models": {}
        }

        if self.model_configs:
            # Calculate average thresholds
            minor_thresholds = [config.minor_threshold_percent for config in self.model_configs.values()]
            major_thresholds = [config.major_threshold_percent for config in self.model_configs.values()]
            critical_thresholds = [config.critical_threshold_percent for config in self.model_configs.values()]

            summary["average_thresholds"] = {
                "minor": np.mean(minor_thresholds),
                "major": np.mean(major_thresholds),
                "critical": np.mean(critical_thresholds)
            }

            # Add per-model summary
            for model_name, config in self.model_configs.items():
                summary["models"][model_name] = {
                    "thresholds": config.get_severity_thresholds(),
                    "environment": config.environment,
                    "auto_tuned": config.auto_tuned,
                    "last_updated": config.last_updated.isoformat() if config.last_updated else None
                }

        return summary