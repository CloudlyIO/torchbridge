"""
Unified Management System for KernelPyTorch

This module provides a unified management system that consolidates 38+ scattered
Manager and Optimizer classes into a single, comprehensive framework.

Key Components:
- UnifiedManager: Master coordinator for all management domains
- HardwareManager: GPU, memory, and distributed hardware management
- OptimizationManager: Compilation, precision, and performance optimization
- InfrastructureManager: Testing, validation, and lifecycle management

Usage:
    from kernel_pytorch.core.management import UnifiedManager, get_manager

    # Direct usage
    manager = UnifiedManager()
    optimized_model = manager.optimize(model)

    # Convenience function
    optimized_model = get_manager().optimize(model)
"""

from .unified_manager import (
    UnifiedManager,
    HardwareManager,
    OptimizationManager,
    InfrastructureManager,
    ManagerType,
    ManagerState,
    ManagerContext,
    get_manager,
    optimize_with_unified_manager,
    reset_manager
)

__all__ = [
    "UnifiedManager",
    "HardwareManager",
    "OptimizationManager",
    "InfrastructureManager",
    "ManagerType",
    "ManagerState",
    "ManagerContext",
    "get_manager",
    "optimize_with_unified_manager",
    "reset_manager"
]