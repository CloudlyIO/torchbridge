"""
Hardware Adaptation and Optimization for Large-Scale Training (2025) - Refactored

This module now serves as a compatibility layer for the refactored hardware adaptation system.
The functionality has been split into focused modules:

- hardware_discovery.py: Hardware topology discovery and device detection
- thermal_power_management.py: Thermal-aware scheduling and power management
- fault_tolerance.py: Hardware health monitoring and fault detection
- hardware_adapter.py: Main orchestration and unified interface

This maintains backward compatibility while providing better code organization.
"""

import warnings

# Import DeviceMesh for test compatibility
try:
    from torch.distributed._tensor import DeviceMesh
except ImportError:
    # Fallback for older PyTorch versions
    DeviceMesh = None

# Import all functionality from split modules
from .hardware_discovery import (
    HardwareVendor,
    DeviceCapability,
    ThermalState,
    DeviceInfo,
    NodeTopology,
    ClusterTopology,
    HardwareTopologyManager
)

from .thermal_power_management import (
    ThermalAwareScheduler,
    PowerEfficiencyOptimizer
)

from .fault_tolerance import (
    HardwareHealthMonitor
)

from .hardware_adapter import (
    HardwareAdapter,
    DeviceMeshOptimizer
)

# Backward compatibility warning
warnings.warn(
    "hardware_adaptation.py has been refactored into multiple focused modules. "
    "Consider importing from the specific modules directly: "
    "hardware_discovery, thermal_power_management, fault_tolerance, hardware_adapter",
    FutureWarning,
    stacklevel=2
)

# Re-export everything for backward compatibility
__all__ = [
    # Core enums and data classes
    'HardwareVendor',
    'DeviceCapability',
    'ThermalState',
    'DeviceInfo',
    'NodeTopology',
    'ClusterTopology',

    # Main classes
    'HardwareTopologyManager',
    'ThermalAwareScheduler',
    'PowerEfficiencyOptimizer',
    'HardwareHealthMonitor',
    'HardwareAdapter',
    'DeviceMeshOptimizer',

    # PyTorch distributed components
    'DeviceMesh'
]