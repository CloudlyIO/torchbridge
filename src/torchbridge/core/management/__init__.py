# SPDX-License-Identifier: Apache-2.0
"""
Unified Management System for TorchBridge

Key Components:
- UnifiedManager: Hardware-aware model optimization with backend dispatch
- InfrastructureManager: Deprecation tracking and lifecycle management
- BaseManager: Abstract base class for managers

Usage:
    from torchbridge.core.management import UnifiedManager, get_manager

    manager = UnifiedManager()
    optimized_model = manager.auto_optimize(model)

    # Convenience function
    optimized_model = get_manager().optimize(model)

"""

# Base classes and types
from .base import (
    BaseManager,
    ManagerContext,
    ManagerState,
    ManagerType,
)

# Managers
from .infrastructure_manager import InfrastructureManager

# Main coordinator and helpers
from .unified_manager import (
    UnifiedManager,
    get_manager,
    optimize_with_unified_manager,
    reset_manager,
)

__all__ = [
    # Base classes
    "BaseManager",
    "ManagerType",
    "ManagerState",
    "ManagerContext",
    # Managers
    "InfrastructureManager",
    "UnifiedManager",
    # Helper functions
    "get_manager",
    "optimize_with_unified_manager",
    "reset_manager",
]
