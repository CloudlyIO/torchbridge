"""
Advanced Communication Optimization for Large-Scale Distributed Training (2025) - Refactored

This module now serves as a compatibility layer for the refactored communication system.
The functionality has been split into focused modules:

- communication_primitives.py: Core communication patterns and collective operations
- network_optimization.py: Bandwidth scheduling and topology optimization
- communication_profiling.py: Performance profiling and bottleneck identification
- communication_optimization.py: Backward compatibility layer with factory functions

This maintains backward compatibility while providing better code organization.
"""


# Import all functionality from split modules
from .communication_primitives import (
    AdvancedCollectiveOps,
    CollectiveOpConfig,
    CommunicationMetrics,
    CommunicationPattern,
    CompressionMethod,
    NetworkTopology,
)
from .communication_profiling import CommunicationProfiler
from .network_optimization import BandwidthAwareScheduler, NetworkTopologyOptimizer

# NOTE: This module provides backward compatibility for the refactored communication system.
# For new code, consider importing from: communication_primitives, network_optimization, communication_profiling


# Factory functions for easy component creation
def create_collective_ops(
    world_size: int,
    rank: int,
    topology: NetworkTopology,
    config: CollectiveOpConfig | None = None
) -> AdvancedCollectiveOps:
    """Create AdvancedCollectiveOps with default configuration"""
    return AdvancedCollectiveOps(world_size, rank, topology, config)


def create_topology_optimizer(
    world_size: int,
    node_count: int,
    gpus_per_node: int
) -> NetworkTopologyOptimizer:
    """Create NetworkTopologyOptimizer with default configuration"""
    topology = NetworkTopology(node_count, gpus_per_node)
    return NetworkTopologyOptimizer(world_size, topology)


def create_bandwidth_scheduler(
    world_size: int = 8,
    node_count: int = 4,
    gpus_per_node: int = 8
) -> BandwidthAwareScheduler:
    """Create BandwidthAwareScheduler with default configuration"""
    topology_optimizer = create_topology_optimizer(world_size, node_count, gpus_per_node)
    return BandwidthAwareScheduler(topology_optimizer)


def create_adaptive_scheduler() -> BandwidthAwareScheduler:
    """Create BandwidthAwareScheduler with default configuration (renamed from AdaptiveCommunicationScheduler)"""
    topology_optimizer = create_topology_optimizer(8, 4, 8)
    return BandwidthAwareScheduler(topology_optimizer)


def create_communication_profiler() -> CommunicationProfiler:
    """Create CommunicationProfiler with default configuration"""
    return CommunicationProfiler()


def create_default_topology(
    node_count: int = 4,
    gpus_per_node: int = 8
) -> NetworkTopology:
    """Create default NetworkTopology configuration"""
    return NetworkTopology(
        node_count=node_count,
        gpus_per_node=gpus_per_node,
        intra_node_bandwidth_gbps=600.0,
        inter_node_bandwidth_gbps=200.0,
        network_latency_us=2.0,
        topology_type="fat_tree"
    )


# Re-export everything for backward compatibility
__all__ = [
    # Core enums and data classes
    'CommunicationPattern',
    'CompressionMethod',
    'NetworkTopology',
    'CommunicationMetrics',
    'CollectiveOpConfig',

    # Main communication classes
    'AdvancedCollectiveOps',
    'NetworkTopologyOptimizer',
    'BandwidthAwareScheduler',
    'CommunicationProfiler',

    # Factory functions
    'create_collective_ops',
    'create_topology_optimizer',
    'create_bandwidth_scheduler',
    'create_adaptive_scheduler',
    'create_communication_profiler',
    'create_default_topology'
]
