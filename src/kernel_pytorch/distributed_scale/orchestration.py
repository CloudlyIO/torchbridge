"""
Container Orchestration and Cluster Management for Large-Scale Training (2025) - Refactored

This module now serves as a compatibility layer for the refactored orchestration system.
The functionality has been split into focused modules:

- job_management.py: Job specifications, status tracking, and state management
- cluster_management.py: Kubernetes and SLURM cluster management
- scaling_fault_tolerance.py: Auto-scaling and fault tolerance management
- orchestration.py: Backward compatibility layer with factory functions

This maintains backward compatibility while providing better code organization.
"""

import warnings
from typing import Optional

# Import all functionality from split modules
from .job_management import (
    JobState,
    ResourceType,
    FailureType,
    ResourceRequirement,
    TrainingJobSpec,
    JobStatus,
    ClusterNode
)

from .cluster_management import (
    KubernetesDistributedOrchestrator,
    SLURMClusterManager
)

from .scaling_fault_tolerance import (
    AutoScalingManager,
    FaultToleranceManager
)

# NOTE: This module provides backward compatibility for the refactored orchestration system.
# For new code, consider importing from: job_management, cluster_management, scaling_fault_tolerance


# Factory functions for easy orchestrator creation
def create_kubernetes_orchestrator(
    namespace: str = "ml-training",
    kubeconfig_path: Optional[str] = None
) -> KubernetesDistributedOrchestrator:
    """Create Kubernetes orchestrator with default configuration"""
    return KubernetesDistributedOrchestrator(namespace, kubeconfig_path)


def create_slurm_manager(partition: str = "gpu") -> SLURMClusterManager:
    """Create SLURM cluster manager with default configuration"""
    return SLURMClusterManager(partition)


def create_auto_scaling_manager(
    min_replicas: int = 1,
    max_replicas: int = 100
) -> AutoScalingManager:
    """Create auto-scaling manager with default configuration"""
    return AutoScalingManager(min_replicas, max_replicas)


def create_fault_tolerance_manager(
    checkpoint_interval_minutes: int = 30
) -> FaultToleranceManager:
    """Create fault tolerance manager with default configuration"""
    return FaultToleranceManager(checkpoint_interval_minutes)


# Re-export everything for backward compatibility
__all__ = [
    # Core enums and data classes
    'JobState',
    'ResourceType',
    'FailureType',
    'ResourceRequirement',
    'TrainingJobSpec',
    'JobStatus',
    'ClusterNode',

    # Main orchestration classes
    'KubernetesDistributedOrchestrator',
    'SLURMClusterManager',
    'AutoScalingManager',
    'FaultToleranceManager',

    # Factory functions
    'create_kubernetes_orchestrator',
    'create_slurm_manager',
    'create_auto_scaling_manager',
    'create_fault_tolerance_manager'
]