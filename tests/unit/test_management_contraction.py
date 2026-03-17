"""
Regression tests for v0.5.76 Contraction X — management/ cleanup.

These tests verify:
- HardwareManager and OptimizationManager are deleted (no longer importable)
- UnifiedManager no longer carries stub sub-managers as attributes
- UnifiedManager.optimize() delegates to auto_optimize() for nn.Module
- UnifiedManager.get_status() returns infrastructure key only
- Public API surface is preserved (auto_optimize, get_hardware_profile, etc.)
"""

import importlib

import pytest
import torch.nn as nn

from torchbridge.core.management import UnifiedManager, get_manager

# ---------------------------------------------------------------------------
# Deleted classes must not be importable
# ---------------------------------------------------------------------------

class TestDeletedClasses:
    def test_hardware_manager_not_in_management_init(self):
        """HardwareManager must be removed from management __init__ exports."""
        import torchbridge.core.management as mgmt
        assert not hasattr(mgmt, "HardwareManager"), (
            "HardwareManager is a stub — it was deleted in v0.5.76"
        )

    def test_optimization_manager_not_in_management_init(self):
        """OptimizationManager must be removed from management __init__ exports."""
        import torchbridge.core.management as mgmt
        assert not hasattr(mgmt, "OptimizationManager"), (
            "OptimizationManager is a stub — it was deleted in v0.5.76"
        )

    def test_hardware_manager_module_deleted(self):
        """hardware_manager.py must not exist as a loadable module."""
        with pytest.raises(ImportError):
            importlib.import_module("torchbridge.core.management.hardware_manager")

    def test_optimization_manager_module_deleted(self):
        """optimization_manager.py must not exist as a loadable module."""
        with pytest.raises(ImportError):
            importlib.import_module("torchbridge.core.management.optimization_manager")


# ---------------------------------------------------------------------------
# UnifiedManager no longer carries stub sub-managers
# ---------------------------------------------------------------------------

class TestUnifiedManagerAttributes:
    def test_no_hardware_manager_attribute(self):
        """UnifiedManager must not instantiate HardwareManager."""
        manager = UnifiedManager()
        assert not hasattr(manager, "hardware_manager"), (
            "hardware_manager sub-manager was removed in v0.5.76"
        )

    def test_no_optimization_manager_attribute(self):
        """UnifiedManager must not instantiate OptimizationManager."""
        manager = UnifiedManager()
        assert not hasattr(manager, "optimization_manager"), (
            "optimization_manager sub-manager was removed in v0.5.76"
        )

    def test_infrastructure_manager_present(self):
        """UnifiedManager must still have infrastructure_manager."""
        manager = UnifiedManager()
        assert hasattr(manager, "infrastructure_manager")

    def test_hardware_detector_present(self):
        """UnifiedManager must still carry a HardwareDetector."""
        manager = UnifiedManager()
        assert hasattr(manager, "hardware_detector")


# ---------------------------------------------------------------------------
# UnifiedManager.optimize() delegates to auto_optimize for nn.Module
# ---------------------------------------------------------------------------

class TestOptimizeDelegation:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))

    def test_optimize_nn_module_returns_module(self, simple_model):
        """optimize(model) must return an nn.Module."""
        manager = get_manager()
        result = manager.optimize(simple_model)
        assert isinstance(result, nn.Module)

    def test_optimize_non_module_returns_target(self):
        """optimize(non-Module) must return target unchanged."""
        manager = get_manager()
        target = {"key": "value"}
        result = manager.optimize(target)
        assert result is target

    def test_optimize_and_auto_optimize_consistent(self, simple_model):
        """optimize(model) and auto_optimize(model) must both return nn.Module."""
        manager = UnifiedManager()
        r1 = manager.optimize(simple_model)
        r2 = manager.auto_optimize(simple_model)
        assert isinstance(r1, nn.Module)
        assert isinstance(r2, nn.Module)


# ---------------------------------------------------------------------------
# UnifiedManager.get_status() returns simplified status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_get_status_has_infrastructure_key(self):
        """get_status() must include 'infrastructure' key."""
        manager = get_manager()
        status = manager.get_status()
        assert "infrastructure" in status

    def test_get_status_no_hardware_key(self):
        """get_status() must not include 'hardware' key (HardwareManager deleted)."""
        manager = get_manager()
        status = manager.get_status()
        assert "hardware" not in status

    def test_get_status_no_optimization_key(self):
        """get_status() must not include 'optimization' key (OptimizationManager deleted)."""
        manager = get_manager()
        status = manager.get_status()
        assert "optimization" not in status


# ---------------------------------------------------------------------------
# Preserved public API (regression — must still work)
# ---------------------------------------------------------------------------

class TestPreservedPublicAPI:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))

    def test_auto_optimize_still_works(self, simple_model):
        manager = get_manager()
        result = manager.auto_optimize(simple_model)
        assert isinstance(result, nn.Module)

    def test_get_hardware_profile_still_works(self):
        manager = get_manager()
        profile = manager.get_hardware_profile()
        assert profile is not None

    def test_get_optimization_recommendations_still_works(self):
        manager = get_manager()
        recs = manager.get_optimization_recommendations()
        assert isinstance(recs, dict)
        assert "hardware_type" in recs
        assert "backend" in recs

    def test_get_manager_convenience_function(self):
        manager = get_manager()
        assert isinstance(manager, UnifiedManager)

    def test_infrastructure_manager_still_exportable(self):
        from torchbridge.core.management import InfrastructureManager
        assert InfrastructureManager is not None

    def test_base_manager_still_exportable(self):
        from torchbridge.core.management import BaseManager
        assert BaseManager is not None

    def test_manager_type_has_no_hardware_value(self):
        """ManagerType.HARDWARE must be removed — HardwareManager is deleted."""
        from torchbridge.core.management import ManagerType
        assert not hasattr(ManagerType, "HARDWARE"), (
            "ManagerType.HARDWARE is an orphaned value — HardwareManager was deleted in v0.5.76"
        )

    def test_manager_type_has_no_optimization_value(self):
        """ManagerType.OPTIMIZATION must be removed — OptimizationManager is deleted."""
        from torchbridge.core.management import ManagerType
        assert not hasattr(ManagerType, "OPTIMIZATION"), (
            "ManagerType.OPTIMIZATION is an orphaned value — OptimizationManager was deleted in v0.5.76"
        )

    def test_manager_type_infrastructure_still_valid(self):
        """ManagerType.INFRASTRUCTURE must remain — InfrastructureManager uses it."""
        from torchbridge.core.management import ManagerType
        assert ManagerType.INFRASTRUCTURE.value == "infrastructure"
