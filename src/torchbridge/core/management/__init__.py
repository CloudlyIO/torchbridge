"""
Unified Management System for TorchBridge

This module provides a unified management system that consolidates 38+ scattered
Manager and Optimizer classes into a single, comprehensive framework.

Key Components:
- UnifiedManager: Master coordinator for all management domains
- HardwareManager: GPU, memory, and distributed hardware management
- OptimizationManager: Compilation, precision, and performance optimization
- InfrastructureManager: Testing, validation, and lifecycle management
- BaseManager: Abstract base class for all managers

Usage:
    from torchbridge.core.management import UnifiedManager, get_manager

    # Direct usage
    manager = UnifiedManager()
    optimized_model = manager.optimize(model)

    # Auto-optimization based on hardware
    optimized_model = manager.auto_optimize(model)

    # Convenience function
    optimized_model = get_manager().optimize(model)

Version: 0.3.11
"""

# Base classes and types
from .base import (
    BaseManager,
    ManagerContext,
    ManagerState,
    ManagerType,
)

# Individual managers
from .hardware_manager import HardwareManager
from .infrastructure_manager import InfrastructureManager
from .optimization_manager import OptimizationManager

# Main coordinator and helpers
from .unified_manager import (
    UnifiedManager,
    get_manager,
    optimize_with_unified_manager,
    reset_manager,
)

__version__ = "0.5.1"

__all__ = [
    # Base classes
    "BaseManager",
    "ManagerType",
    "ManagerState",
    "ManagerContext",
    # Managers
    "HardwareManager",
    "OptimizationManager",
    "InfrastructureManager",
    "UnifiedManager",
    # Helper functions
    "get_manager",
    "optimize_with_unified_manager",
    "reset_manager",
]
