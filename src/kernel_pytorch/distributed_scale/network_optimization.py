"""
Network Topology Optimization and Bandwidth Management

Advanced network optimization for large-scale distributed training:
- Automatic topology discovery and analysis
- Bandwidth profiling and congestion monitoring
- Optimal routing and communication pattern selection
- Bandwidth-aware scheduling and congestion avoidance
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from .communication_primitives import (
    NetworkTopology, CommunicationPattern, CompressionMethod
)

logger = logging.getLogger(__name__)


class NetworkTopologyOptimizer:
    """
    Network topology optimizer for communication efficiency

    Features:
    - Automatic topology discovery
    - Bandwidth profiling
    - Optimal routing calculation
    - Network congestion monitoring
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

        # Topology discovery
        self.discovered_topology = self._discover_topology()
        self.bandwidth_matrix = self._initialize_bandwidth_matrix()
        self.latency_matrix = self._initialize_latency_matrix()

        # Optimization state
        self.optimal_routes: Dict[Tuple[int, int], List[int]] = {}
        self.congestion_levels: Dict[int, float] = {}

    def _discover_topology(self) -> NetworkTopology:
        """Discover network topology automatically"""
        # In practice, this would query the actual network topology
        # For simulation, we make reasonable assumptions

        if self.world_size <= 8:
            # Single node
            return NetworkTopology(
                node_count=1,
                gpus_per_node=self.world_size,
                intra_node_bandwidth_gbps=600.0,  # NVLink
                inter_node_bandwidth_gbps=0.0,
                topology_type="single_node"
            )
        elif self.world_size <= 64:
            # Multi-node, single rack
            gpus_per_node = 8
            node_count = (self.world_size + gpus_per_node - 1) // gpus_per_node

            return NetworkTopology(
                node_count=node_count,
                gpus_per_node=gpus_per_node,
                intra_node_bandwidth_gbps=600.0,
                inter_node_bandwidth_gbps=200.0,  # InfiniBand
                topology_type="single_rack"
            )
        else:
            # Multi-rack topology
            gpus_per_node = 8
            nodes_per_rack = 8
            node_count = (self.world_size + gpus_per_node - 1) // gpus_per_node
            rack_count = (node_count + nodes_per_rack - 1) // nodes_per_rack

            return NetworkTopology(
                node_count=node_count,
                gpus_per_node=gpus_per_node,
                rack_count=rack_count,
                nodes_per_rack=nodes_per_rack,
                intra_node_bandwidth_gbps=600.0,
                inter_node_bandwidth_gbps=200.0,
                rack_bandwidth_gbps=400.0,  # Inter-rack bandwidth
                topology_type="multi_rack"
            )

    def _initialize_bandwidth_matrix(self) -> np.ndarray:
        """Initialize bandwidth matrix between all rank pairs"""
        matrix = np.zeros((self.world_size, self.world_size))

        for i in range(self.world_size):
            for j in range(self.world_size):
                if i == j:
                    matrix[i, j] = float('inf')  # Self-communication
                else:
                    matrix[i, j] = self._estimate_bandwidth(i, j)

        return matrix

    def _initialize_latency_matrix(self) -> np.ndarray:
        """Initialize latency matrix between all rank pairs"""
        matrix = np.zeros((self.world_size, self.world_size))

        for i in range(self.world_size):
            for j in range(self.world_size):
                if i == j:
                    matrix[i, j] = 0.0
                else:
                    matrix[i, j] = self._estimate_latency(i, j)

        return matrix

    def _estimate_bandwidth(self, rank1: int, rank2: int) -> float:
        """Estimate bandwidth between two ranks"""
        node1 = rank1 // self.discovered_topology.gpus_per_node
        node2 = rank2 // self.discovered_topology.gpus_per_node

        if node1 == node2:
            # Intra-node communication (NVLink)
            return self.discovered_topology.intra_node_bandwidth_gbps
        else:
            # Inter-node communication
            if (self.discovered_topology.rack_count and
                self.discovered_topology.nodes_per_rack):
                # Check if same rack
                rack1 = node1 // self.discovered_topology.nodes_per_rack
                rack2 = node2 // self.discovered_topology.nodes_per_rack

                if rack1 == rack2:
                    return self.discovered_topology.inter_node_bandwidth_gbps
                else:
                    # Inter-rack communication (typically slower)
                    return self.discovered_topology.rack_bandwidth_gbps or self.discovered_topology.inter_node_bandwidth_gbps * 0.5
            else:
                return self.discovered_topology.inter_node_bandwidth_gbps

    def _estimate_latency(self, rank1: int, rank2: int) -> float:
        """Estimate latency between two ranks (in microseconds)"""
        node1 = rank1 // self.discovered_topology.gpus_per_node
        node2 = rank2 // self.discovered_topology.gpus_per_node

        if node1 == node2:
            # Intra-node latency (very low for NVLink)
            return 0.5
        else:
            # Inter-node latency
            base_latency = self.discovered_topology.network_latency_us

            if (self.discovered_topology.rack_count and
                self.discovered_topology.nodes_per_rack):
                # Check if same rack
                rack1 = node1 // self.discovered_topology.nodes_per_rack
                rack2 = node2 // self.discovered_topology.nodes_per_rack

                if rack1 != rack2:
                    # Inter-rack adds additional latency
                    base_latency += 1.0

            return base_latency

    def optimize_communication_pattern(
        self,
        operation: str,
        tensor_size: int,
        participants: List[int]
    ) -> Dict[str, Any]:
        """
        Optimize communication pattern for given operation

        Args:
            operation: Type of operation ('allreduce', 'allgather', etc.)
            tensor_size: Size of tensor in bytes
            participants: List of participating ranks

        Returns:
            Optimization recommendations
        """
        recommendations = {
            'optimal_pattern': CommunicationPattern.RING,
            'chunk_size': 64 * 1024 * 1024,  # 64MB default
            'compression_method': CompressionMethod.NONE,
            'expected_time_ms': 0.0
        }

        # Analyze network conditions
        avg_bandwidth = np.mean([self.bandwidth_matrix[i, j]
                                for i in participants for j in participants if i != j])
        avg_latency = np.mean([self.latency_matrix[i, j]
                              for i in participants for j in participants if i != j])

        # Select optimal pattern based on conditions
        if len(participants) <= 8:
            # Small groups: optimize for latency
            if avg_latency < 2.0:  # Low latency network
                recommendations['optimal_pattern'] = CommunicationPattern.TREE
            else:
                recommendations['optimal_pattern'] = CommunicationPattern.RING

        elif tensor_size < 1024 * 1024:  # < 1MB
            # Small tensors: minimize latency overhead
            recommendations['optimal_pattern'] = CommunicationPattern.TREE

        elif avg_bandwidth < 100.0:  # Low bandwidth
            # Use hierarchical to reduce inter-node traffic
            recommendations['optimal_pattern'] = CommunicationPattern.HIERARCHICAL
            recommendations['compression_method'] = CompressionMethod.FP16

        else:
            # Good network conditions: optimize for bandwidth
            recommendations['optimal_pattern'] = CommunicationPattern.RING

        # Calculate optimal chunk size
        if operation == 'allreduce':
            # Optimize chunk size for pipelining
            optimal_chunks = max(4, len(participants))
            recommendations['chunk_size'] = max(tensor_size // optimal_chunks, 1024 * 1024)

        # Estimate completion time
        recommendations['expected_time_ms'] = self._estimate_completion_time(
            operation, tensor_size, participants, recommendations['optimal_pattern']
        )

        return recommendations

    def _estimate_completion_time(
        self,
        operation: str,
        tensor_size: int,
        participants: List[int],
        pattern: CommunicationPattern
    ) -> float:
        """Estimate completion time for given operation"""
        num_participants = len(participants)
        avg_bandwidth_gbps = np.mean([self.bandwidth_matrix[i, j]
                                     for i in participants for j in participants if i != j])
        avg_latency_us = np.mean([self.latency_matrix[i, j]
                                 for i in participants for j in participants if i != j])

        # Convert bandwidth to bytes per second
        bandwidth_bps = avg_bandwidth_gbps * 1024**3 / 8

        if operation == 'allreduce':
            if pattern == CommunicationPattern.RING:
                # Ring AllReduce: 2 * (N-1) / N * data_size / bandwidth + latency
                transfer_time = 2 * (num_participants - 1) / num_participants * tensor_size / bandwidth_bps
                latency_penalty = (num_participants - 1) * avg_latency_us / 1000.0  # Convert to ms
            elif pattern == CommunicationPattern.TREE:
                # Tree reduction: log2(N) * data_size / bandwidth + log2(N) * latency
                tree_depth = np.ceil(np.log2(num_participants))
                transfer_time = tree_depth * tensor_size / bandwidth_bps
                latency_penalty = tree_depth * avg_latency_us / 1000.0
            else:  # Hierarchical
                # Estimate as ring within nodes + tree across nodes
                gpus_per_node = self.discovered_topology.gpus_per_node
                num_nodes = np.ceil(num_participants / gpus_per_node)

                intra_node_time = 2 * (gpus_per_node - 1) / gpus_per_node * tensor_size / (600e9 / 8)  # NVLink
                inter_node_time = np.log2(num_nodes) * tensor_size / (avg_bandwidth_gbps * 1e9 / 8)
                transfer_time = max(intra_node_time, inter_node_time)
                latency_penalty = avg_latency_us / 1000.0

        elif operation == 'allgather':
            if pattern == CommunicationPattern.RING:
                # Ring AllGather: (N-1) / N * total_data / bandwidth
                total_data = tensor_size * num_participants
                transfer_time = (num_participants - 1) / num_participants * total_data / bandwidth_bps
                latency_penalty = (num_participants - 1) * avg_latency_us / 1000.0
            else:
                # Simplified estimation for other patterns
                total_data = tensor_size * num_participants
                transfer_time = total_data / bandwidth_bps
                latency_penalty = np.log2(num_participants) * avg_latency_us / 1000.0

        else:
            # Generic estimation
            transfer_time = tensor_size / bandwidth_bps
            latency_penalty = avg_latency_us / 1000.0

        return (transfer_time + latency_penalty / 1000.0) * 1000.0  # Convert to ms

    def update_congestion_levels(self, link_utilizations: Dict[Tuple[int, int], float]):
        """Update network congestion levels"""
        for (src, dst), utilization in link_utilizations.items():
            if utilization > 0.8:  # High utilization threshold
                self.congestion_levels[src] = max(
                    self.congestion_levels.get(src, 0.0), utilization
                )
                self.congestion_levels[dst] = max(
                    self.congestion_levels.get(dst, 0.0), utilization
                )

    def get_topology_info(self) -> Dict[str, Any]:
        """Get comprehensive topology information"""
        return {
            'discovered_topology': {
                'node_count': self.discovered_topology.node_count,
                'gpus_per_node': self.discovered_topology.gpus_per_node,
                'topology_type': self.discovered_topology.topology_type,
                'intra_node_bandwidth_gbps': self.discovered_topology.intra_node_bandwidth_gbps,
                'inter_node_bandwidth_gbps': self.discovered_topology.inter_node_bandwidth_gbps,
                'network_latency_us': self.discovered_topology.network_latency_us
            },
            'bandwidth_matrix_stats': {
                'min_bandwidth': float(np.min(self.bandwidth_matrix[self.bandwidth_matrix > 0])),
                'max_bandwidth': float(np.max(self.bandwidth_matrix[self.bandwidth_matrix != float('inf')])),
                'avg_bandwidth': float(np.mean(self.bandwidth_matrix[self.bandwidth_matrix != float('inf')]))
            },
            'latency_matrix_stats': {
                'min_latency': float(np.min(self.latency_matrix[self.latency_matrix > 0])),
                'max_latency': float(np.max(self.latency_matrix)),
                'avg_latency': float(np.mean(self.latency_matrix[self.latency_matrix > 0]))
            },
            'congestion_info': {
                'congested_ranks': len(self.congestion_levels),
                'avg_congestion': np.mean(list(self.congestion_levels.values())) if self.congestion_levels else 0.0
            }
        }


class BandwidthAwareScheduler:
    """
    Scheduler that adapts to network bandwidth conditions

    Features:
    - Real-time bandwidth monitoring
    - Communication scheduling optimization
    - Congestion avoidance
    """

    def __init__(self, topology_optimizer: NetworkTopologyOptimizer):
        self.topology_optimizer = topology_optimizer
        self.active_communications: Dict[str, Dict] = {}
        self.bandwidth_utilization: Dict[Tuple[int, int], float] = {}
        self.scheduled_operations: List[Dict] = []

    def schedule_communication(
        self,
        operation_id: str,
        operation_type: str,
        participants: List[int],
        tensor_size: int,
        priority: float = 1.0
    ) -> Dict[str, Any]:
        """
        Schedule communication operation with bandwidth awareness

        Args:
            operation_id: Unique identifier for operation
            operation_type: Type of operation
            participants: Participating ranks
            tensor_size: Size of data to transfer
            priority: Operation priority (higher = more urgent)

        Returns:
            Scheduling decision and timing
        """
        # Analyze current network state
        network_load = self._calculate_network_load(participants)
        congestion_score = self._calculate_congestion_score(participants)

        # Get optimization recommendations
        recommendations = self.topology_optimizer.optimize_communication_pattern(
            operation_type, tensor_size, participants
        )

        # Determine optimal scheduling time
        if congestion_score > 0.7:  # High congestion
            # Delay or use alternative routing
            delay_ms = max(10, recommendations['expected_time_ms'] * 0.2)
            use_compression = True
        else:
            delay_ms = 0
            use_compression = recommendations['compression_method'] != CompressionMethod.NONE

        # Create scheduling decision
        scheduling_decision = {
            'schedule_time_ms': delay_ms,  # Expected by tests
            'optimization': {  # Expected by tests
                'optimal_pattern': recommendations['optimal_pattern'].value,
                'compression_method': recommendations['compression_method'].value if hasattr(recommendations['compression_method'], 'value') else str(recommendations['compression_method']),
                'chunk_size': recommendations.get('chunk_size', 64 * 1024 * 1024),
                'expected_time_ms': recommendations['expected_time_ms']
            },
            # Legacy keys for backward compatibility
            'operation_id': operation_id,
            'immediate_execution': delay_ms == 0,
            'recommended_delay_ms': delay_ms,
            'pattern': recommendations['optimal_pattern'].value,
            'compression': use_compression,
            'estimated_completion_ms': recommendations['expected_time_ms'] + delay_ms,
            'priority_score': priority / (1 + congestion_score),
            'network_conditions': {
                'load': network_load,
                'congestion': congestion_score,
                'affected_links': self._identify_affected_links(participants)
            }
        }

        # Schedule the operation
        scheduled_op = {
            'operation_id': operation_id,
            'operation_type': operation_type,
            'participants': participants,
            'tensor_size': tensor_size,
            'priority': priority,
            'scheduled_time': time.time() + delay_ms / 1000.0,
            'decision': scheduling_decision
        }

        self.scheduled_operations.append(scheduled_op)

        return scheduling_decision

    def _calculate_network_load(self, participants: List[int]) -> float:
        """Calculate current network load for participants"""
        if not participants:
            return 0.0

        # Calculate load based on active communications
        total_load = 0.0
        affected_links = self._identify_affected_links(participants)

        for link in affected_links:
            utilization = self.bandwidth_utilization.get(link, 0.0)
            total_load += utilization

        return total_load / len(affected_links) if affected_links else 0.0

    def _calculate_congestion_score(self, participants: List[int]) -> float:
        """Calculate congestion score for participants"""
        congestion_levels = [
            self.topology_optimizer.congestion_levels.get(rank, 0.0)
            for rank in participants
        ]

        return max(congestion_levels) if congestion_levels else 0.0

    def _identify_affected_links(self, participants: List[int]) -> List[Tuple[int, int]]:
        """Identify network links affected by communication"""
        links = []

        for i, rank1 in enumerate(participants):
            for rank2 in participants[i+1:]:
                links.append((rank1, rank2))
                links.append((rank2, rank1))

        return links

    def update_bandwidth_utilization(self, link: Tuple[int, int], utilization: float):
        """Update bandwidth utilization for a link"""
        self.bandwidth_utilization[link] = utilization

        # Update topology optimizer's congestion info
        if utilization > 0.8:
            self.topology_optimizer.update_congestion_levels({link: utilization})

    def get_active_communications(self) -> List[Dict[str, Any]]:
        """Get list of currently active communications"""
        current_time = time.time()
        active = [
            op for op in self.scheduled_operations
            if op['scheduled_time'] <= current_time <= op['scheduled_time'] + op['decision']['estimated_completion_ms'] / 1000.0
        ]

        return active

    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduling statistics"""
        current_time = time.time()

        # Categorize operations
        pending = [op for op in self.scheduled_operations if op['scheduled_time'] > current_time]
        active = self.get_active_communications()
        completed = [op for op in self.scheduled_operations
                    if op['scheduled_time'] + op['decision']['estimated_completion_ms'] / 1000.0 < current_time]

        # Calculate statistics
        stats = {
            'operations': {
                'total_scheduled': len(self.scheduled_operations),
                'pending': len(pending),
                'active': len(active),
                'completed': len(completed)
            },
            'bandwidth_utilization': {
                'average': np.mean(list(self.bandwidth_utilization.values())) if self.bandwidth_utilization else 0.0,
                'peak': max(self.bandwidth_utilization.values()) if self.bandwidth_utilization else 0.0,
                'utilized_links': len(self.bandwidth_utilization)
            },
            'congestion': {
                'congested_ranks': len(self.topology_optimizer.congestion_levels),
                'max_congestion': max(self.topology_optimizer.congestion_levels.values()) if self.topology_optimizer.congestion_levels else 0.0
            }
        }

        # Calculate delay statistics
        if completed:
            scheduled_delays = [op['decision']['recommended_delay_ms'] for op in completed]
            stats['delays'] = {
                'average_delay_ms': np.mean(scheduled_delays),
                'max_delay_ms': max(scheduled_delays),
                'operations_delayed': sum(1 for delay in scheduled_delays if delay > 0)
            }

        return stats

    def optimize_pending_operations(self):
        """Re-optimize pending operations based on current network state"""
        current_time = time.time()
        pending_ops = [op for op in self.scheduled_operations if op['scheduled_time'] > current_time]

        for op in pending_ops:
            # Recalculate scheduling decision
            new_decision = self.schedule_communication(
                f"{op['operation_id']}_reopt",
                op['operation_type'],
                op['participants'],
                op['tensor_size'],
                op['priority']
            )

            # Update if new decision is better
            if new_decision['estimated_completion_ms'] < op['decision']['estimated_completion_ms']:
                op['decision'] = new_decision
                op['scheduled_time'] = current_time + new_decision['recommended_delay_ms'] / 1000.0

                logger.info(f"Re-optimized operation {op['operation_id']}: "
                           f"new completion time {new_decision['estimated_completion_ms']:.2f}ms")

    def clear_completed_operations(self, retention_hours: float = 1.0):
        """Clear completed operations older than retention period"""
        current_time = time.time()
        retention_seconds = retention_hours * 3600

        self.scheduled_operations = [
            op for op in self.scheduled_operations
            if current_time - (op['scheduled_time'] + op['decision']['estimated_completion_ms'] / 1000.0) < retention_seconds
        ]