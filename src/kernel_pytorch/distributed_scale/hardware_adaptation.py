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


# Import DeviceMesh for test compatibility
try:
    from torch.distributed._tensor import DeviceMesh
except ImportError:
    # Fallback for older PyTorch versions
    DeviceMesh = None

# Import all functionality from split modules
from .fault_tolerance import HardwareHealthMonitor
from .hardware_adapter import DeviceMeshOptimizer, HardwareAdapter
from .hardware_discovery import (
    ClusterTopology,
    DeviceCapability,
    DeviceInfo,
    HardwareTopologyManager,
    HardwareVendor,
    NodeTopology,
    ThermalState,
)
from .thermal_power_management import PowerEfficiencyOptimizer, ThermalAwareScheduler

# NOTE: This module provides backward compatibility for the refactored hardware adaptation system.
# For new code, consider importing from: hardware_discovery, thermal_power_management, fault_tolerance, hardware_adapter

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
