"""
Unified Communication Management for Distributed Scale

Consolidates communication functionality from:
- communication_optimization.py
- communication_primitives.py
- communication_profiling.py
- network_optimization.py

Provides unified interface for:
- Communication primitives and optimization
- Network topology optimization
- Communication profiling and analysis
"""

import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import warnings
from abc import ABC, abstractmethod

try:
    import torch.distributed._functional_collectives as funcol
except ImportError:
    funcol = None
    warnings.warn("Functional collectives not available")


class CommunicationPattern(Enum):
    """Communication pattern types."""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    POINT_TO_POINT = "point_to_point"


class NetworkTopology(Enum):
    """Network topology types."""
    TREE = "tree"
    RING = "ring"
    MESH = "mesh"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"


@dataclass
class CommunicationConfig:
    """Unified communication configuration."""
    backend: str = "nccl"
    timeout_seconds: int = 1800
    compression_enabled: bool = False
    profiling_enabled: bool = False
    optimization_level: str = "balanced"
    buffer_size: int = 1024 * 1024  # 1MB default


@dataclass
class CommunicationMetrics:
    """Communication performance metrics."""
    operation: str
    data_size: int
    duration_ms: float
    bandwidth_gbps: float
    latency_ms: float
    efficiency: float
    timestamp: float = field(default_factory=time.time)


class CommunicationProfiler:
    """Unified communication profiling and analysis."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = []
        self.active_operations = {}

    def start_operation(self, operation: str, data_size: int) -> str:
        """Start profiling a communication operation."""
        if not self.enabled:
            return ""

        op_id = f"{operation}_{int(time.time() * 1000000)}"
        self.active_operations[op_id] = {
            "operation": operation,
            "data_size": data_size,
            "start_time": time.time()
        }
        return op_id

    def end_operation(self, op_id: str) -> Optional[CommunicationMetrics]:
        """End profiling and record metrics."""
        if not self.enabled or op_id not in self.active_operations:
            return None

        op_info = self.active_operations.pop(op_id)
        duration_s = time.time() - op_info["start_time"]
        duration_ms = duration_s * 1000

        # Calculate bandwidth and efficiency
        data_gb = op_info["data_size"] / (1024**3)
        bandwidth_gbps = data_gb / duration_s if duration_s > 0 else 0

        metrics = CommunicationMetrics(
            operation=op_info["operation"],
            data_size=op_info["data_size"],
            duration_ms=duration_ms,
            bandwidth_gbps=bandwidth_gbps,
            latency_ms=duration_ms,  # Simplified
            efficiency=min(bandwidth_gbps / 100, 1.0)  # Normalize to theoretical max
        )

        self.metrics.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get profiling summary."""
        if not self.metrics:
            return {"total_operations": 0}

        total_ops = len(self.metrics)
        avg_bandwidth = sum(m.bandwidth_gbps for m in self.metrics) / total_ops
        avg_latency = sum(m.latency_ms for m in self.metrics) / total_ops

        return {
            "total_operations": total_ops,
            "avg_bandwidth_gbps": avg_bandwidth,
            "avg_latency_ms": avg_latency,
            "total_data_gb": sum(m.data_size for m in self.metrics) / (1024**3)
        }


class NetworkTopologyOptimizer:
    """Unified network topology optimization."""

    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.topology = NetworkTopology.RING  # Default
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def optimize_topology(self, operation: CommunicationPattern) -> Dict[str, Any]:
        """Optimize network topology for operation."""
        if operation == CommunicationPattern.ALL_REDUCE:
            return self._optimize_allreduce_topology()
        elif operation == CommunicationPattern.ALL_GATHER:
            return self._optimize_allgather_topology()
        else:
            return {"topology": self.topology.value, "optimized": False}

    def _optimize_allreduce_topology(self) -> Dict[str, Any]:
        """Optimize for all-reduce operations."""
        if self.world_size <= 8:
            optimal_topology = NetworkTopology.TREE
        elif self.world_size <= 32:
            optimal_topology = NetworkTopology.RING
        else:
            optimal_topology = NetworkTopology.HIERARCHICAL

        return {
            "topology": optimal_topology.value,
            "optimized": True,
            "expected_improvement": self._estimate_improvement(optimal_topology)
        }

    def _optimize_allgather_topology(self) -> Dict[str, Any]:
        """Optimize for all-gather operations."""
        # All-gather typically benefits from hierarchical for large scale
        if self.world_size > 16:
            optimal_topology = NetworkTopology.HIERARCHICAL
        else:
            optimal_topology = NetworkTopology.RING

        return {
            "topology": optimal_topology.value,
            "optimized": True,
            "expected_improvement": self._estimate_improvement(optimal_topology)
        }

    def _estimate_improvement(self, topology: NetworkTopology) -> float:
        """Estimate performance improvement."""
        # Simplified improvement estimation
        improvements = {
            NetworkTopology.TREE: 1.2,
            NetworkTopology.RING: 1.1,
            NetworkTopology.HIERARCHICAL: 1.3,
            NetworkTopology.MESH: 1.4
        }
        return improvements.get(topology, 1.0)


class CommunicationPrimitives:
    """Unified communication primitives with optimization."""

    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.profiler = CommunicationProfiler(config.profiling_enabled)
        self.optimizer = NetworkTopologyOptimizer(config)

    def optimized_all_reduce(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Optimized all-reduce with profiling."""
        if not dist.is_initialized():
            return tensor

        # Profile the operation
        op_id = self.profiler.start_operation("all_reduce", tensor.numel() * tensor.element_size())

        # Optimize topology
        topology_info = self.optimizer.optimize_topology(CommunicationPattern.ALL_REDUCE)

        try:
            # Perform all-reduce
            dist.all_reduce(tensor, **kwargs)

            # Record metrics
            self.profiler.end_operation(op_id)

            return tensor
        except Exception as e:
            # Cleanup on error
            self.profiler.end_operation(op_id)
            raise e

    def optimized_all_gather(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Optimized all-gather with profiling."""
        if not dist.is_initialized():
            return tensor

        op_id = self.profiler.start_operation("all_gather", tensor.numel() * tensor.element_size())

        try:
            # Prepare output tensor
            world_size = dist.get_world_size()
            output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]

            # Perform all-gather
            dist.all_gather(output_tensors, tensor, **kwargs)

            result = torch.cat(output_tensors, dim=0)
            self.profiler.end_operation(op_id)

            return result
        except Exception as e:
            self.profiler.end_operation(op_id)
            raise e

    def optimized_reduce_scatter(self, input_list: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """Optimized reduce-scatter with profiling."""
        if not dist.is_initialized():
            return input_list[0] if input_list else torch.tensor([])

        total_size = sum(t.numel() * t.element_size() for t in input_list)
        op_id = self.profiler.start_operation("reduce_scatter", total_size)

        try:
            # Prepare output tensor
            output = torch.empty_like(input_list[0])

            # Perform reduce-scatter
            dist.reduce_scatter(output, input_list, **kwargs)

            self.profiler.end_operation(op_id)
            return output
        except Exception as e:
            self.profiler.end_operation(op_id)
            raise e

    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get communication profiling summary."""
        return self.profiler.get_summary()


class UnifiedCommunicationManager:
    """
    Unified communication management system.

    Consolidates functionality from:
    - CommunicationOptimizer
    - NetworkTopologyOptimizer
    - CommunicationProfiler
    - Various communication primitive implementations
    """

    def __init__(self, config: Optional[CommunicationConfig] = None):
        self.config = config or CommunicationConfig()
        self.primitives = CommunicationPrimitives(self.config)
        self.is_initialized = False

    def initialize(self, backend: Optional[str] = None) -> bool:
        """Initialize distributed communication."""
        if self.is_initialized:
            return True

        try:
            if not dist.is_initialized():
                if backend:
                    self.config.backend = backend

                # Initialize distributed if not already done
                # Note: This requires proper environment setup
                logging.info(f"Initializing distributed with backend: {self.config.backend}")

            self.is_initialized = dist.is_initialized()
            return self.is_initialized
        except Exception as e:
            logging.error(f"Failed to initialize communication: {e}")
            return False

    def all_reduce(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Unified all-reduce operation."""
        return self.primitives.optimized_all_reduce(tensor, **kwargs)

    def all_gather(self, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Unified all-gather operation."""
        return self.primitives.optimized_all_gather(tensor, **kwargs)

    def reduce_scatter(self, input_list: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """Unified reduce-scatter operation."""
        return self.primitives.optimized_reduce_scatter(input_list, **kwargs)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = self.primitives.get_profiling_summary()
        summary.update({
            "backend": self.config.backend,
            "optimization_level": self.config.optimization_level,
            "profiling_enabled": self.config.profiling_enabled,
            "is_initialized": self.is_initialized
        })
        return summary

    def cleanup(self) -> None:
        """Cleanup communication resources."""
        if self.is_initialized and dist.is_initialized():
            try:
                dist.destroy_process_group()
                self.is_initialized = False
            except Exception as e:
                logging.error(f"Error during communication cleanup: {e}")


# Global communication manager instance
default_comm_manager = None


def get_communication_manager(config: Optional[CommunicationConfig] = None) -> UnifiedCommunicationManager:
    """Get global communication manager."""
    global default_comm_manager
    if default_comm_manager is None or config is not None:
        default_comm_manager = UnifiedCommunicationManager(config)
    return default_comm_manager


def cleanup_communication() -> None:
    """Cleanup global communication manager."""
    global default_comm_manager
    if default_comm_manager:
        default_comm_manager.cleanup()
        default_comm_manager = None