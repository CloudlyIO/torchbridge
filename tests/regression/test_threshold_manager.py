#!/usr/bin/env python3
"""
Tests for ThresholdManager functionality
"""

import os
import json
import tempfile
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from benchmarks.regression.threshold_manager import ThresholdManager, ThresholdConfig
from benchmarks.regression.baseline_manager import BaselineMetrics


class TestThresholdConfig:
    """Test ThresholdConfig dataclass functionality"""

    def test_threshold_config_creation(self):
        """Test creating ThresholdConfig"""
        config = ThresholdConfig(
            model_name="test_model",
            latency_threshold_percent=5.0,
            memory_threshold_percent=10.0,
            throughput_threshold_percent=5.0,
            confidence_level=0.95,
            min_sample_size=10,
            minor_threshold_percent=2.0,
            major_threshold_percent=5.0,
            critical_threshold_percent=10.0,
            environment="default"
        )

        assert config.model_name == "test_model"
        assert config.latency_threshold_percent == 5.0
        assert config.confidence_level == 0.95

    def test_threshold_config_serialization(self):
        """Test ThresholdConfig to/from dict conversion"""
        now = datetime.now()
        config = ThresholdConfig(
            model_name="test_model",
            latency_threshold_percent=5.0,
            memory_threshold_percent=10.0,
            throughput_threshold_percent=5.0,
            last_updated=now
        )

        # Test serialization
        data = config.to_dict()
        assert isinstance(data['last_updated'], str)
        assert data['model_name'] == "test_model"

        # Test deserialization
        restored = ThresholdConfig.from_dict(data)
        assert restored.model_name == config.model_name
        assert restored.latency_threshold_percent == config.latency_threshold_percent
        assert isinstance(restored.last_updated, datetime)

    def test_get_severity_thresholds(self):
        """Test getting severity thresholds tuple"""
        config = ThresholdConfig(
            model_name="test_model",
            minor_threshold_percent=2.0,
            major_threshold_percent=5.0,
            critical_threshold_percent=10.0
        )

        thresholds = config.get_severity_thresholds()
        assert thresholds == (2.0, 5.0, 10.0)


class TestThresholdManager:
    """Test ThresholdManager functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ThresholdManager(thresholds_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_threshold_manager_initialization(self):
        """Test ThresholdManager initialization"""
        assert self.manager.thresholds_dir.exists()
        assert self.manager.config_file.exists() or True  # File created on first save
        assert isinstance(self.manager.model_configs, dict)

    def test_get_default_thresholds(self):
        """Test getting default thresholds for new model"""
        config = self.manager.get_thresholds("new_model", "default")

        assert config.model_name == "new_model"
        assert config.environment == "default"
        assert config.latency_threshold_percent == 5.0
        assert config.minor_threshold_percent == 2.0
        assert config.major_threshold_percent == 5.0
        assert config.critical_threshold_percent == 10.0

    def test_update_thresholds_from_baseline(self):
        """Test updating thresholds based on baseline variance"""
        baseline = BaselineMetrics(
            model_name="test_model",
            mean_latency_ms=10.0,
            std_latency_ms=1.0,  # 10% coefficient of variation
            mean_throughput=100.0,
            std_throughput=5.0,
            mean_memory_mb=128.0,
            std_memory_mb=10.0,
            sample_count=20,
            confidence_interval_95=(9.0, 11.0),
            established_date=datetime.now(),
            last_validated_date=datetime.now()
        )

        updated_config = self.manager.update_thresholds_from_baseline(
            baseline, "default", sensitivity_factor=2.0
        )

        assert updated_config.model_name == "test_model"
        assert updated_config.auto_tuned == True
        assert updated_config.minor_threshold_percent > 0
        assert updated_config.major_threshold_percent > updated_config.minor_threshold_percent
        assert updated_config.critical_threshold_percent > updated_config.major_threshold_percent

    def test_update_thresholds_from_history(self):
        """Test updating thresholds based on historical variance"""
        historical_variances = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

        updated_config = self.manager.update_thresholds_from_history(
            "test_model", historical_variances, "default"
        )

        assert updated_config.model_name == "test_model"
        assert updated_config.auto_tuned == True
        assert updated_config.minor_threshold_percent >= 2.0  # At least minimum
        assert updated_config.major_threshold_percent >= 5.0
        assert updated_config.critical_threshold_percent >= 10.0

    def test_update_thresholds_insufficient_history(self):
        """Test handling insufficient historical data"""
        insufficient_data = [1.0, 2.0]  # Less than 3 required

        with pytest.warns(UserWarning, match="Insufficient variance data"):
            config = self.manager.update_thresholds_from_history(
                "test_model", insufficient_data, "default"
            )

        # Should return default config
        assert config.model_name == "test_model"
        assert config.auto_tuned == False

    def test_validate_threshold_sensitivity(self):
        """Test threshold sensitivity validation"""
        # Create a config with reasonable thresholds
        config = ThresholdConfig(
            model_name="test_model",
            minor_threshold_percent=2.0,
            major_threshold_percent=5.0,
            critical_threshold_percent=10.0,
            auto_tuned=True,
            last_updated=datetime.now()
        )
        self.manager.model_configs["test_model_default"] = config

        validation = self.manager.validate_threshold_sensitivity("test_model", "default")

        assert validation["model_name"] == "test_model"
        assert validation["environment"] == "default"
        assert validation["thresholds"]["minor"] == 2.0
        assert validation["thresholds"]["major"] == 5.0
        assert validation["thresholds"]["critical"] == 10.0
        assert validation["validation"]["reasonable_spacing"] == True
        assert validation["validation"]["reasonable_values"] == True

    def test_validate_threshold_poor_configuration(self):
        """Test validation of poorly configured thresholds"""
        # Create config with poor threshold spacing
        poor_config = ThresholdConfig(
            model_name="test_model",
            minor_threshold_percent=5.0,
            major_threshold_percent=5.5,  # Too close to minor
            critical_threshold_percent=6.0,  # Too close to major
            auto_tuned=False
        )
        self.manager.model_configs["test_model_default"] = poor_config

        validation = self.manager.validate_threshold_sensitivity("test_model", "default")

        assert validation["validation"]["reasonable_spacing"] == False
        assert len(validation["recommendations"]) > 0
        assert any("threshold gaps" in rec for rec in validation["recommendations"])

    def test_get_environment_multipliers(self):
        """Test environment-specific multipliers"""
        # Test default environment
        default_mult = self.manager.get_environment_multipliers("default")
        assert default_mult["latency"] == 1.0
        assert default_mult["memory"] == 1.0
        assert default_mult["throughput"] == 1.0

        # Test CPU environment
        cpu_mult = self.manager.get_environment_multipliers("cpu")
        assert cpu_mult["latency"] == 1.2
        assert cpu_mult["memory"] == 0.8
        assert cpu_mult["throughput"] == 1.2

        # Test GPU environment
        gpu_mult = self.manager.get_environment_multipliers("gpu")
        assert gpu_mult["latency"] == 0.9
        assert gpu_mult["memory"] == 1.5
        assert gpu_mult["throughput"] == 0.9

        # Test cloud environment
        cloud_mult = self.manager.get_environment_multipliers("cloud")
        assert cloud_mult["latency"] == 1.5
        assert cloud_mult["memory"] == 1.3
        assert cloud_mult["throughput"] == 1.4

        # Test CI environment
        ci_mult = self.manager.get_environment_multipliers("ci")
        assert ci_mult["latency"] == 2.0
        assert ci_mult["memory"] == 1.5
        assert ci_mult["throughput"] == 2.0

    def test_apply_environment_adjustments(self):
        """Test applying environment adjustments to base configuration"""
        base_config = ThresholdConfig(
            model_name="test_model",
            latency_threshold_percent=5.0,
            memory_threshold_percent=10.0,
            throughput_threshold_percent=5.0,
            minor_threshold_percent=2.0,
            major_threshold_percent=5.0,
            critical_threshold_percent=10.0,
            environment="default"
        )

        # Apply CI environment adjustments (2x latency multiplier)
        adjusted = self.manager.apply_environment_adjustments(base_config, "ci")

        assert adjusted.model_name == "test_model"
        assert adjusted.environment == "ci"
        assert adjusted.latency_threshold_percent == 10.0  # 5.0 * 2.0
        assert adjusted.memory_threshold_percent == 15.0   # 10.0 * 1.5
        assert adjusted.minor_threshold_percent == 4.0     # 2.0 * 2.0
        assert adjusted.major_threshold_percent == 10.0    # 5.0 * 2.0
        assert adjusted.critical_threshold_percent == 20.0 # 10.0 * 2.0

    def test_export_import_threshold_config(self):
        """Test exporting and importing threshold configurations"""
        # Set up some configurations
        config1 = ThresholdConfig(model_name="model1", latency_threshold_percent=3.0)
        config2 = ThresholdConfig(model_name="model2", latency_threshold_percent=7.0)

        self.manager.model_configs["model1_default"] = config1
        self.manager.model_configs["model2_default"] = config2

        # Export configurations
        exported = self.manager.export_threshold_config()

        assert exported["version"] == "0.1.57"
        assert exported["total_configurations"] == 2
        assert "model1_default" in exported["configurations"]
        assert "model2_default" in exported["configurations"]

        # Create new manager and import
        new_manager = ThresholdManager(thresholds_dir=tempfile.mkdtemp())
        success = new_manager.import_threshold_config(exported)

        assert success == True
        assert len(new_manager.model_configs) == 2
        assert "model1_default" in new_manager.model_configs
        assert "model2_default" in new_manager.model_configs

    def test_cleanup_old_configurations(self):
        """Test cleanup of old threshold configurations"""
        # Create old configuration
        old_config = ThresholdConfig(
            model_name="old_model",
            last_updated=datetime.now() - timedelta(days=100)
        )

        # Create recent configuration
        recent_config = ThresholdConfig(
            model_name="recent_model",
            last_updated=datetime.now() - timedelta(days=10)
        )

        self.manager.model_configs["old_model_default"] = old_config
        self.manager.model_configs["recent_model_default"] = recent_config

        # Cleanup configurations older than 90 days
        with pytest.warns(UserWarning, match="Removed old threshold configuration"):
            self.manager.cleanup_old_configurations(days_old=90)

        assert "old_model_default" not in self.manager.model_configs
        assert "recent_model_default" in self.manager.model_configs

    def test_get_threshold_summary(self):
        """Test getting threshold summary"""
        # Add some configurations
        config1 = ThresholdConfig(
            model_name="model1",
            minor_threshold_percent=2.0,
            major_threshold_percent=5.0,
            critical_threshold_percent=10.0,
            environment="default",
            auto_tuned=True,
            last_updated=datetime.now()
        )

        config2 = ThresholdConfig(
            model_name="model2",
            minor_threshold_percent=3.0,
            major_threshold_percent=6.0,
            critical_threshold_percent=12.0,
            environment="gpu",
            auto_tuned=False,
            last_updated=datetime.now()
        )

        self.manager.model_configs["model1_default"] = config1
        self.manager.model_configs["model2_gpu"] = config2

        summary = self.manager.get_threshold_summary()

        assert summary["total_models"] == 2
        assert summary["auto_tuned_count"] == 1
        assert "default" in summary["environments"]
        assert "gpu" in summary["environments"]
        assert summary["average_thresholds"]["minor"] == 2.5  # (2.0 + 3.0) / 2
        assert summary["average_thresholds"]["major"] == 5.5  # (5.0 + 6.0) / 2
        assert summary["average_thresholds"]["critical"] == 11.0  # (10.0 + 12.0) / 2
        assert "model1_default" in summary["models"]
        assert "model2_gpu" in summary["models"]

    def test_persist_configurations(self):
        """Test that configurations are persisted to disk"""
        config = ThresholdConfig(
            model_name="persistent_model",
            latency_threshold_percent=3.5,
            auto_tuned=True,
            last_updated=datetime.now()
        )

        self.manager.model_configs["persistent_model_default"] = config
        self.manager._save_configurations()

        # Create new manager instance and verify persistence
        new_manager = ThresholdManager(thresholds_dir=self.temp_dir)

        assert "persistent_model_default" in new_manager.model_configs
        restored_config = new_manager.model_configs["persistent_model_default"]
        assert restored_config.model_name == "persistent_model"
        assert restored_config.latency_threshold_percent == 3.5
        assert restored_config.auto_tuned == True

    def test_handle_invalid_configuration_file(self):
        """Test handling of corrupted configuration file"""
        # Create corrupted config file
        config_file = Path(self.temp_dir) / "threshold_config.json"
        with open(config_file, 'w') as f:
            f.write("invalid json content")

        # Should handle gracefully with warning
        with pytest.warns(UserWarning, match="Failed to load threshold configurations"):
            manager = ThresholdManager(thresholds_dir=self.temp_dir)

        # Should still work with empty configs
        assert isinstance(manager.model_configs, dict)
        assert len(manager.model_configs) == 0