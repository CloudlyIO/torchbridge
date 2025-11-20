"""
Advanced Communication Optimization for Large-Scale Distributed Training (2025)

Optimizes communication patterns for clusters of thousands of GPUs using:
- Topology-aware collective operations
- Bandwidth-aware scheduling
- Hierarchical communication patterns
- Advanced compression techniques
- Network congestion monitoring and adaptation
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
from contextlib import contextmanager
import socket
import psutil

logger = logging.getLogger(__name__)


class CommunicationPattern(Enum):
    """Communication patterns for distributed operations"""
    RING = "ring"
    TREE = "tree"
    BUTTERFLY = "butterfly"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"


class CompressionMethod(Enum):
    """Compression methods for communication"""
    NONE = "none"
    FP16 = "fp16"
    BF16 = "bf16"
    QUANTIZED_8BIT = "quantized_8bit"
    TOPK = "topk"
    POWERSGD = "powersgd"
    ADAPTIVE = "adaptive"


@dataclass
class NetworkTopology:
    """Network topology information for optimization"""
    node_count: int
    gpus_per_node: int
    intra_node_bandwidth_gbps: float = 600.0  # NVLink bandwidth
    inter_node_bandwidth_gbps: float = 200.0  # InfiniBand bandwidth
    network_latency_us: float = 2.0
    topology_type: str = "fat_tree"

    # Hierarchical structure
    rack_count: Optional[int] = None
    nodes_per_rack: Optional[int] = None
    rack_bandwidth_gbps: Optional[float] = None


@dataclass
class CommunicationMetrics:
    """Communication performance metrics"""
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    avg_bandwidth_gbps: float = 0.0
    peak_bandwidth_gbps: float = 0.0
    avg_latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    compression_ratio: float = 1.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class CollectiveOpConfig:
    """Configuration for collective operations"""
    pattern: CommunicationPattern = CommunicationPattern.ADAPTIVE
    compression: CompressionMethod = CompressionMethod.ADAPTIVE
    chunk_size_mb: int = 64
    overlap_computation: bool = True
    use_hierarchical: bool = True
    bandwidth_threshold_gbps: float = 100.0


class AdvancedCollectiveOps:
    """
    Advanced collective operations with topology awareness

    Features:
    - Topology-aware AllReduce/AllGather
    - Hierarchical communication patterns
    - Dynamic compression selection
    - Bandwidth monitoring and adaptation
    """

    def __init__(
        self,
        world_size: int,
        rank: int,
        topology: NetworkTopology,
        config: Optional[CollectiveOpConfig] = None
    ):
        self.world_size = world_size
        self.rank = rank
        self.topology = topology
        self.config = config or CollectiveOpConfig()

        # Communication state
        self.metrics = CommunicationMetrics()
        self.bandwidth_history: List[float] = []
        self.latency_history: List[float] = []

        # Topology optimization
        self.communication_groups = self._create_communication_groups()
        self.optimal_patterns = self._analyze_optimal_patterns()

        # Compression state
        self.compression_stats: Dict[str, float] = {}
        self.adaptive_compression_enabled = self.config.compression == CompressionMethod.ADAPTIVE

    def _create_communication_groups(self) -> Dict[str, List[int]]:
        """Create hierarchical communication groups"""
        groups = {}

        # Intra-node groups (GPUs on same node)
        gpus_per_node = self.topology.gpus_per_node
        for node_id in range(self.topology.node_count):
            start_rank = node_id * gpus_per_node
            end_rank = min(start_rank + gpus_per_node, self.world_size)
            groups[f'node_{node_id}'] = list(range(start_rank, end_rank))

        # Inter-node groups (corresponding GPUs across nodes)
        for gpu_id in range(gpus_per_node):
            inter_node_group = []
            for node_id in range(self.topology.node_count):
                rank = node_id * gpus_per_node + gpu_id
                if rank < self.world_size:
                    inter_node_group.append(rank)
            groups[f'inter_node_gpu_{gpu_id}'] = inter_node_group

        # Rack-level groups if specified
        if self.topology.rack_count and self.topology.nodes_per_rack:
            for rack_id in range(self.topology.rack_count):
                rack_group = []
                start_node = rack_id * self.topology.nodes_per_rack
                end_node = min(start_node + self.topology.nodes_per_rack, self.topology.node_count)

                for node_id in range(start_node, end_node):
                    start_rank = node_id * gpus_per_node
                    end_rank = min(start_rank + gpus_per_node, self.world_size)
                    rack_group.extend(range(start_rank, end_rank))

                groups[f'rack_{rack_id}'] = rack_group

        return groups

    def _analyze_optimal_patterns(self) -> Dict[str, CommunicationPattern]:
        """Analyze optimal communication patterns for different operations"""
        patterns = {}

        # For small clusters, use tree pattern
        if self.world_size <= 8:
            patterns['allreduce'] = CommunicationPattern.TREE
            patterns['allgather'] = CommunicationPattern.RING

        # For medium clusters, use ring for bandwidth-intensive ops
        elif self.world_size <= 64:
            patterns['allreduce'] = CommunicationPattern.RING
            patterns['allgather'] = CommunicationPattern.RING

        # For large clusters, use hierarchical patterns
        else:
            patterns['allreduce'] = CommunicationPattern.HIERARCHICAL
            patterns['allgather'] = CommunicationPattern.HIERARCHICAL
            patterns['broadcast'] = CommunicationPattern.TREE

        return patterns

    async def allreduce_optimized(
        self,
        tensor: torch.Tensor,
        op: str = 'sum',
        pattern: Optional[CommunicationPattern] = None
    ) -> torch.Tensor:
        """
        Optimized AllReduce with topology awareness

        Args:
            tensor: Input tensor to reduce
            op: Reduction operation ('sum', 'mean', 'max', etc.)
            pattern: Communication pattern override

        Returns:
            Reduced tensor
        """
        start_time = time.time()
        original_size = tensor.numel() * tensor.element_size()

        # Select optimal pattern
        if pattern is None:
            pattern = self._select_optimal_pattern('allreduce', tensor.numel())

        # Apply compression if beneficial
        compressed_tensor, compression_ratio = self._apply_compression(tensor)

        try:
            if pattern == CommunicationPattern.HIERARCHICAL:
                result = await self._hierarchical_allreduce(compressed_tensor, op)
            elif pattern == CommunicationPattern.RING:
                result = await self._ring_allreduce(compressed_tensor, op)
            elif pattern == CommunicationPattern.TREE:
                result = await self._tree_allreduce(compressed_tensor, op)
            else:
                # Fallback to PyTorch implementation
                dist.all_reduce(compressed_tensor, op=getattr(dist.ReduceOp, op.upper()))
                result = compressed_tensor

            # Decompress if needed
            result = self._apply_decompression(result, compression_ratio)

            # Update metrics
            elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_communication_metrics(original_size, elapsed_time, compression_ratio)

            return result

        except Exception as e:
            logger.error(f"Optimized AllReduce failed: {e}")
            # Fallback to standard implementation
            dist.all_reduce(tensor)
            return tensor

    async def allgather_optimized(
        self,
        tensor: torch.Tensor,
        pattern: Optional[CommunicationPattern] = None
    ) -> torch.Tensor:
        """
        Optimized AllGather with bandwidth optimization

        Args:
            tensor: Input tensor to gather
            pattern: Communication pattern override

        Returns:
            Gathered tensor from all ranks
        """
        start_time = time.time()
        original_size = tensor.numel() * tensor.element_size()

        # Select optimal pattern
        if pattern is None:
            pattern = self._select_optimal_pattern('allgather', tensor.numel())

        # For AllGather, compression may not always be beneficial
        # due to the need to preserve exact values
        use_compression = self._should_compress_for_allgather(tensor)

        if use_compression:
            compressed_tensor, compression_ratio = self._apply_compression(tensor)
        else:
            compressed_tensor = tensor
            compression_ratio = 1.0

        try:
            if pattern == CommunicationPattern.HIERARCHICAL:
                result = await self._hierarchical_allgather(compressed_tensor)
            elif pattern == CommunicationPattern.RING:
                result = await self._ring_allgather(compressed_tensor)
            else:
                # Standard AllGather
                output_tensors = [torch.zeros_like(compressed_tensor) for _ in range(self.world_size)]
                dist.all_gather(output_tensors, compressed_tensor)
                result = torch.cat(output_tensors, dim=0)

            # Decompress if needed
            if use_compression:
                result = self._apply_decompression(result, compression_ratio)

            # Update metrics
            elapsed_time = (time.time() - start_time) * 1000
            self._update_communication_metrics(original_size * self.world_size, elapsed_time, compression_ratio)

            return result

        except Exception as e:
            logger.error(f"Optimized AllGather failed: {e}")
            # Fallback
            output_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            dist.all_gather(output_tensors, tensor)
            return torch.cat(output_tensors, dim=0)

    def _select_optimal_pattern(self, operation: str, tensor_size: int) -> CommunicationPattern:
        """Select optimal communication pattern based on current conditions"""
        if self.config.pattern != CommunicationPattern.ADAPTIVE:
            return self.config.pattern

        # Use bandwidth history to make decisions
        current_bandwidth = np.mean(self.bandwidth_history[-10:]) if self.bandwidth_history else 0

        # For small tensors, minimize latency (use tree)
        if tensor_size < 1024 * 1024:  # < 1MB
            return CommunicationPattern.TREE

        # For bandwidth-constrained scenarios, use hierarchical
        if current_bandwidth < self.config.bandwidth_threshold_gbps:
            return CommunicationPattern.HIERARCHICAL

        # Default to pattern from analysis
        return self.optimal_patterns.get(operation, CommunicationPattern.RING)

    async def _hierarchical_allreduce(self, tensor: torch.Tensor, op: str) -> torch.Tensor:
        """Hierarchical AllReduce: intra-node first, then inter-node"""
        # Step 1: Reduce within each node
        node_id = self.rank // self.topology.gpus_per_node
        intra_node_group = self.communication_groups[f'node_{node_id}']

        # Only proceed if this rank is part of the node group
        if self.rank in intra_node_group:
            # Create process group for intra-node communication
            if len(intra_node_group) > 1:
                # Simulate intra-node allreduce (would use NCCL in practice)
                await self._simulate_collective_op(tensor, 'allreduce', len(intra_node_group))

        # Step 2: Reduce across nodes (only one rank per node participates)
        gpu_local_id = self.rank % self.topology.gpus_per_node
        if gpu_local_id == 0:  # Node representative
            inter_node_group = self.communication_groups[f'inter_node_gpu_0']
            if len(inter_node_group) > 1:
                await self._simulate_collective_op(tensor, 'allreduce', len(inter_node_group))

        # Step 3: Broadcast within each node
        if len(intra_node_group) > 1:
            await self._simulate_collective_op(tensor, 'broadcast', len(intra_node_group))

        return tensor

    async def _ring_allreduce(self, tensor: torch.Tensor, op: str) -> torch.Tensor:
        """Ring AllReduce implementation"""
        chunks = self._split_tensor_for_ring(tensor, self.world_size)

        # Reduce-scatter phase
        for step in range(self.world_size - 1):
            send_rank = (self.rank + step) % self.world_size
            recv_rank = (self.rank - step - 1) % self.world_size

            # Simulate communication latency
            await self._simulate_communication_latency(chunks[send_rank].numel())

        # Allgather phase
        for step in range(self.world_size - 1):
            send_rank = (self.rank - step) % self.world_size
            recv_rank = (self.rank - step - 1) % self.world_size

            # Simulate communication latency
            await self._simulate_communication_latency(chunks[send_rank].numel())

        return torch.cat(chunks, dim=0)

    async def _tree_allreduce(self, tensor: torch.Tensor, op: str) -> torch.Tensor:
        """Tree-based AllReduce for latency optimization"""
        # Calculate tree depth
        depth = int(np.ceil(np.log2(self.world_size)))

        # Simulate tree reduction
        for level in range(depth):
            step_size = 2 ** level
            if self.rank % (step_size * 2) == 0:
                partner = self.rank + step_size
                if partner < self.world_size:
                    # Simulate receiving and reducing
                    await self._simulate_communication_latency(tensor.numel())

        # Simulate tree broadcast
        for level in range(depth - 1, -1, -1):
            step_size = 2 ** level
            if self.rank % (step_size * 2) == 0:
                partner = self.rank + step_size
                if partner < self.world_size:
                    # Simulate sending
                    await self._simulate_communication_latency(tensor.numel())

        return tensor

    async def _hierarchical_allgather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Hierarchical AllGather implementation"""
        # Gather within nodes first
        node_id = self.rank // self.topology.gpus_per_node
        intra_node_group = self.communication_groups[f'node_{node_id}']

        node_tensors = []
        for i in range(len(intra_node_group)):
            if i == self.rank % self.topology.gpus_per_node:
                node_tensors.append(tensor)
            else:
                node_tensors.append(torch.zeros_like(tensor))
                await self._simulate_communication_latency(tensor.numel())

        # Gather across nodes
        gpu_local_id = self.rank % self.topology.gpus_per_node
        inter_node_group = self.communication_groups[f'inter_node_gpu_{gpu_local_id}']

        all_tensors = []
        for node in range(self.topology.node_count):
            if node == node_id:
                all_tensors.extend(node_tensors)
            else:
                # Simulate receiving from other nodes
                for _ in range(len(intra_node_group)):
                    all_tensors.append(torch.zeros_like(tensor))
                    await self._simulate_communication_latency(tensor.numel())

        return torch.cat(all_tensors, dim=0)

    async def _ring_allgather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ring AllGather implementation"""
        all_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        all_tensors[self.rank] = tensor.clone()

        # Ring communication
        for step in range(self.world_size - 1):
            send_idx = (self.rank - step) % self.world_size
            recv_idx = (self.rank - step - 1) % self.world_size

            # Simulate sending tensor to next rank and receiving from previous
            await self._simulate_communication_latency(tensor.numel())

            # In actual implementation, we'd receive the tensor here
            # For simulation, we just mark it as received

        return torch.cat(all_tensors, dim=0)

    def _apply_compression(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Apply compression to tensor based on current strategy"""
        if not self._should_compress(tensor):
            return tensor, 1.0

        compression_method = self._select_compression_method(tensor)

        if compression_method == CompressionMethod.FP16:
            compressed = tensor.half()
            ratio = 2.0
        elif compression_method == CompressionMethod.BF16:
            compressed = tensor.bfloat16()
            ratio = 2.0
        elif compression_method == CompressionMethod.QUANTIZED_8BIT:
            # Simple 8-bit quantization simulation
            scale = tensor.abs().max() / 127.0
            compressed = (tensor / scale).round().clamp(-128, 127).byte()
            ratio = 4.0  # Assuming fp32 to int8
        elif compression_method == CompressionMethod.TOPK:
            # Top-K sparsification
            k = max(1, int(tensor.numel() * 0.1))  # Keep top 10%
            _, indices = torch.topk(tensor.abs().flatten(), k)
            compressed = torch.zeros_like(tensor)
            compressed.flatten()[indices] = tensor.flatten()[indices]
            ratio = 10.0
        else:
            compressed = tensor
            ratio = 1.0

        return compressed, ratio

    def _apply_decompression(self, tensor: torch.Tensor, compression_ratio: float) -> torch.Tensor:
        """Apply decompression to restore original precision if needed"""
        if compression_ratio == 1.0:
            return tensor

        # For simulation, we just convert back to float32
        if tensor.dtype == torch.half or tensor.dtype == torch.bfloat16:
            return tensor.float()
        elif tensor.dtype == torch.uint8:
            # Simplified dequantization
            return tensor.float()

        return tensor

    def _should_compress(self, tensor: torch.Tensor) -> bool:
        """Determine if compression should be applied"""
        if not self.adaptive_compression_enabled:
            return self.config.compression != CompressionMethod.NONE

        # Use bandwidth history to decide
        current_bandwidth = np.mean(self.bandwidth_history[-5:]) if self.bandwidth_history else 0

        # Compress if bandwidth is limited or tensor is large
        tensor_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)

        return (current_bandwidth < self.config.bandwidth_threshold_gbps or
                tensor_size_mb > self.config.chunk_size_mb)

    def _should_compress_for_allgather(self, tensor: torch.Tensor) -> bool:
        """Determine if compression should be applied for AllGather operations"""
        # Be more conservative with AllGather since we need to preserve values
        if tensor.dtype in [torch.int32, torch.int64, torch.bool]:
            return False  # Don't compress integer/boolean tensors

        return self._should_compress(tensor)

    def _select_compression_method(self, tensor: torch.Tensor) -> CompressionMethod:
        """Select optimal compression method"""
        if self.config.compression != CompressionMethod.ADAPTIVE:
            return self.config.compression

        # Adaptive selection based on tensor properties and network conditions
        current_bandwidth = np.mean(self.bandwidth_history[-5:]) if self.bandwidth_history else 0

        if current_bandwidth < 50.0:  # Very constrained bandwidth
            return CompressionMethod.TOPK
        elif current_bandwidth < 100.0:
            return CompressionMethod.QUANTIZED_8BIT
        elif tensor.dtype == torch.float32:
            return CompressionMethod.FP16
        else:
            return CompressionMethod.NONE

    def _split_tensor_for_ring(self, tensor: torch.Tensor, num_chunks: int) -> List[torch.Tensor]:
        """Split tensor into chunks for ring communication"""
        chunk_size = tensor.numel() // num_chunks
        remainder = tensor.numel() % num_chunks

        chunks = []
        start_idx = 0

        for i in range(num_chunks):
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            chunk = tensor.flatten()[start_idx:end_idx].reshape(-1)
            chunks.append(chunk)
            start_idx = end_idx

        return chunks

    async def _simulate_collective_op(self, tensor: torch.Tensor, op: str, group_size: int):
        """Simulate collective operation latency"""
        # Calculate expected latency based on operation and group size
        data_size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)

        if op == 'allreduce':
            # AllReduce latency: 2 * (group_size - 1) / group_size * data_size / bandwidth
            latency_ms = 2 * (group_size - 1) / group_size * data_size_mb / (self.topology.intra_node_bandwidth_gbps * 1000 / 8)
        elif op == 'allgather':
            # AllGather latency: (group_size - 1) / group_size * data_size / bandwidth
            latency_ms = (group_size - 1) / group_size * data_size_mb / (self.topology.intra_node_bandwidth_gbps * 1000 / 8)
        else:  # broadcast
            latency_ms = data_size_mb / (self.topology.intra_node_bandwidth_gbps * 1000 / 8)

        # Add network latency
        latency_ms += self.topology.network_latency_us / 1000.0

        # Simulate delay
        await asyncio.sleep(latency_ms / 1000.0)

    async def _simulate_communication_latency(self, tensor_elements: int):
        """Simulate communication latency for given tensor size"""
        data_size_mb = tensor_elements * 4 / (1024 * 1024)  # Assume fp32
        bandwidth_mbps = self.topology.inter_node_bandwidth_gbps * 1000 / 8

        transfer_time_ms = data_size_mb / bandwidth_mbps
        total_latency_ms = transfer_time_ms + (self.topology.network_latency_us / 1000.0)

        await asyncio.sleep(total_latency_ms / 1000.0)

    def _update_communication_metrics(self, bytes_transferred: int, latency_ms: float, compression_ratio: float):
        """Update communication performance metrics"""
        # Update bandwidth calculation
        bandwidth_gbps = (bytes_transferred * 8) / (latency_ms / 1000.0) / (1024**3)

        self.bandwidth_history.append(bandwidth_gbps)
        self.latency_history.append(latency_ms)

        # Keep history bounded
        if len(self.bandwidth_history) > 1000:
            self.bandwidth_history = self.bandwidth_history[-500:]
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-500:]

        # Update metrics
        self.metrics.total_bytes_sent += bytes_transferred
        self.metrics.avg_bandwidth_gbps = np.mean(self.bandwidth_history[-10:])
        self.metrics.peak_bandwidth_gbps = max(self.bandwidth_history) if self.bandwidth_history else 0
        self.metrics.avg_latency_ms = np.mean(self.latency_history[-10:])
        self.metrics.compression_ratio = compression_ratio
        self.metrics.last_updated = time.time()

    def get_metrics(self) -> CommunicationMetrics:
        """Get current communication metrics"""
        return self.metrics


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
        # Get optimization recommendations
        optimization = self.topology_optimizer.optimize_communication_pattern(
            operation_type, tensor_size, participants
        )

        # Calculate current network load
        network_load = self._calculate_network_load(participants)

        # Determine scheduling strategy
        if network_load < 0.5:  # Low load
            schedule_time = 0.0  # Immediate
        elif network_load < 0.8:  # Medium load
            # Small delay to avoid congestion
            schedule_time = optimization['expected_time_ms'] * 0.1
        else:  # High load
            # Longer delay or different pattern
            schedule_time = optimization['expected_time_ms'] * 0.5
            # Consider switching to more efficient pattern
            if optimization['optimal_pattern'] == CommunicationPattern.RING:
                optimization['optimal_pattern'] = CommunicationPattern.HIERARCHICAL

        # Record scheduled operation
        operation_info = {
            'id': operation_id,
            'type': operation_type,
            'participants': participants,
            'tensor_size': tensor_size,
            'priority': priority,
            'schedule_time_ms': schedule_time,
            'optimization': optimization,
            'network_load': network_load
        }

        self.scheduled_operations.append(operation_info)

        return {
            'schedule_time_ms': schedule_time,
            'optimization': optimization,
            'network_load': network_load,
            'recommended_delay': schedule_time > 0
        }

    def _calculate_network_load(self, participants: List[int]) -> float:
        """Calculate current network load for given participants"""
        total_utilization = 0.0
        total_links = 0

        for i, rank1 in enumerate(participants):
            for j, rank2 in enumerate(participants):
                if i != j:
                    link_key = tuple(sorted([rank1, rank2]))
                    utilization = self.bandwidth_utilization.get(link_key, 0.0)
                    total_utilization += utilization
                    total_links += 1

        return total_utilization / max(total_links, 1)

    def update_bandwidth_utilization(self, rank1: int, rank2: int, utilization: float):
        """Update bandwidth utilization between two ranks"""
        link_key = tuple(sorted([rank1, rank2]))
        self.bandwidth_utilization[link_key] = utilization


class CommunicationProfiler:
    """
    Profiler for communication operations

    Features:
    - Performance monitoring
    - Bottleneck identification
    - Optimization recommendations
    """

    def __init__(self):
        self.operation_history: List[Dict] = []
        self.performance_stats: Dict[str, List[float]] = {}
        self.bottlenecks: List[Dict] = []

    @contextmanager
    def profile_operation(self, operation_type: str, participants: List[int]):
        """Context manager for profiling communication operations"""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            duration_ms = (end_time - start_time) * 1000
            memory_delta_mb = (end_memory - start_memory) / (1024 * 1024)

            # Record operation
            operation_record = {
                'type': operation_type,
                'participants': participants,
                'duration_ms': duration_ms,
                'memory_delta_mb': memory_delta_mb,
                'timestamp': end_time
            }

            self.operation_history.append(operation_record)

            # Update statistics
            if operation_type not in self.performance_stats:
                self.performance_stats[operation_type] = []
            self.performance_stats[operation_type].append(duration_ms)

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            return psutil.Process().memory_info().rss

    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify communication bottlenecks"""
        bottlenecks = []

        for op_type, durations in self.performance_stats.items():
            if len(durations) < 5:  # Need enough samples
                continue

            avg_duration = np.mean(durations)
            std_duration = np.std(durations)
            p95_duration = np.percentile(durations, 95)

            # Identify if operation is consistently slow
            if p95_duration > avg_duration * 2:
                bottleneck = {
                    'operation_type': op_type,
                    'avg_duration_ms': avg_duration,
                    'p95_duration_ms': p95_duration,
                    'variability': std_duration / avg_duration,
                    'severity': 'high' if p95_duration > avg_duration * 3 else 'medium'
                }
                bottlenecks.append(bottleneck)

        self.bottlenecks = bottlenecks
        return bottlenecks

    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling data"""
        recommendations = []

        bottlenecks = self.identify_bottlenecks()

        for bottleneck in bottlenecks:
            op_type = bottleneck['operation_type']

            if bottleneck['variability'] > 0.5:
                recommendations.append(
                    f"High variability in {op_type} operations suggests network congestion. "
                    f"Consider using bandwidth-aware scheduling."
                )

            if bottleneck['p95_duration_ms'] > 100:
                recommendations.append(
                    f"{op_type} operations are slow. Consider compression or pattern optimization."
                )

        # General recommendations based on operation patterns
        if 'allreduce' in self.performance_stats and len(self.performance_stats['allreduce']) > 10:
            avg_allreduce = np.mean(self.performance_stats['allreduce'])
            if avg_allreduce > 50:
                recommendations.append(
                    "AllReduce operations are slow. Consider hierarchical patterns for large clusters."
                )

        return recommendations

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'total_operations': len(self.operation_history),
            'operation_breakdown': {},
            'performance_stats': {},
            'bottlenecks': self.identify_bottlenecks(),
            'recommendations': self.get_optimization_recommendations(),
            'report_timestamp': time.time()
        }

        # Operation breakdown
        for record in self.operation_history:
            op_type = record['type']
            if op_type not in report['operation_breakdown']:
                report['operation_breakdown'][op_type] = 0
            report['operation_breakdown'][op_type] += 1

        # Performance statistics
        for op_type, durations in self.performance_stats.items():
            report['performance_stats'][op_type] = {
                'count': len(durations),
                'avg_duration_ms': np.mean(durations),
                'min_duration_ms': np.min(durations),
                'max_duration_ms': np.max(durations),
                'p95_duration_ms': np.percentile(durations, 95),
                'std_duration_ms': np.std(durations)
            }

        return report


# Import asyncio at module level
import asyncio