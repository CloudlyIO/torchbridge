"""
Hardware Adaptation Orchestrator

Main orchestrator for hardware adaptation and optimization in large-scale training:
- Coordinates hardware discovery, thermal management, and fault tolerance
- Provides unified interface for device placement and optimization
- Handles device mesh creation for distributed training
- Manages cluster-wide statistics and reporting
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from torch.distributed._tensor import DeviceMesh
from contextlib import contextmanager

from .hardware_discovery import (
    HardwareTopologyManager, HardwareVendor, DeviceInfo, ClusterTopology
)
from .thermal_power_management import ThermalAwareScheduler, PowerEfficiencyOptimizer
from .fault_tolerance import HardwareHealthMonitor

logger = logging.getLogger(__name__)


class HardwareAdapter:
    """
    Main hardware adaptation orchestrator for large-scale training

    Integrates hardware discovery, thermal management, and fault tolerance
    to provide optimal device placement and cluster management.
    """

    def __init__(
        self,
        enable_monitoring: bool = True,
        thermal_threshold: float = 85.0,
        power_budget_w: Optional[float] = None
    ):
        # Core components
        self.topology_manager = HardwareTopologyManager()
        self.thermal_scheduler = ThermalAwareScheduler(
            self.topology_manager, thermal_threshold, power_budget_w
        )
        self.power_optimizer = PowerEfficiencyOptimizer(
            self.topology_manager, power_budget_w
        )
        self.health_monitor = HardwareHealthMonitor(
            self.topology_manager, enable_monitoring
        )

        # Device mesh optimizer
        self.mesh_optimizer = DeviceMeshOptimizer(self.topology_manager)

        logger.info("Hardware adapter initialized")

    @contextmanager
    def optimal_device_context(
        self,
        memory_requirement_gb: float,
        compute_requirement_tflops: float,
        prefer_vendor: Optional[HardwareVendor] = None
    ):
        """Context manager for optimal device placement"""
        optimal_devices = self.get_optimal_device_placement(
            memory_requirement_gb, compute_requirement_tflops, prefer_vendor
        )

        if not optimal_devices:
            raise RuntimeError("No suitable devices available")

        # Set optimal device
        original_device = torch.cuda.current_device() if torch.cuda.is_available() else None

        try:
            if torch.cuda.is_available() and optimal_devices:
                torch.cuda.set_device(optimal_devices[0])
            yield optimal_devices
        finally:
            if original_device is not None:
                torch.cuda.set_device(original_device)

    def get_optimal_device_placement(
        self,
        memory_requirement_gb: float,
        compute_requirement_tflops: float,
        prefer_vendor: Optional[HardwareVendor] = None
    ) -> List[int]:
        """
        Get optimal device placement for given requirements

        Args:
            memory_requirement_gb: Memory requirement per device
            compute_requirement_tflops: Compute requirement in TFLOPS
            prefer_vendor: Preferred hardware vendor

        Returns:
            List of optimal device IDs
        """
        if not self.topology_manager.cluster_topology:
            return []

        candidate_devices = []

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                # Check health status
                device_health = self.health_monitor.device_health_status.get(device.device_id, "healthy")
                if device_health == "failed":
                    continue

                # Check memory requirement
                available_memory = device.memory_gb - device.memory_used_gb
                if available_memory < memory_requirement_gb:
                    continue

                # Check compute requirement (rough estimate)
                device_tflops = device.peak_flops_fp32 / 1e12
                if device_tflops < compute_requirement_tflops:
                    continue

                # Check vendor preference
                if prefer_vendor and device.vendor != prefer_vendor:
                    continue

                # Calculate device score
                score = self._calculate_device_score(device, memory_requirement_gb, compute_requirement_tflops)
                candidate_devices.append((device.device_id, score))

        # Sort by score (higher is better)
        candidate_devices.sort(key=lambda x: x[1], reverse=True)

        return [device_id for device_id, _ in candidate_devices]

    def _calculate_device_score(
        self,
        device: DeviceInfo,
        memory_req: float,
        compute_req: float
    ) -> float:
        """Calculate device suitability score"""
        score = 0.0

        # Memory score (prefer devices with more available memory)
        available_memory = device.memory_gb - device.memory_used_gb
        memory_score = available_memory / device.memory_gb
        score += memory_score * 0.3

        # Compute score (prefer more powerful devices)
        device_tflops = device.peak_flops_fp32 / 1e12
        compute_score = min(device_tflops / max(compute_req, 1.0), 1.0)
        score += compute_score * 0.3

        # Thermal score (prefer cooler devices)
        thermal_score = 1.0 - (device.current_temp_c / 100.0)
        score += thermal_score * 0.2

        # Utilization score (prefer less utilized devices)
        util_score = 1.0 - (device.current_utilization / 100.0)
        score += util_score * 0.2

        return score

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics"""
        if not self.topology_manager.cluster_topology:
            return {}

        stats = {
            'topology': self._get_topology_stats(),
            'thermal': self.thermal_scheduler.get_thermal_report(),
            'health': self.health_monitor.get_device_health_status(),
            'power': self._get_power_stats(),
            'performance': self._get_performance_stats()
        }

        return stats

    def _get_topology_stats(self) -> Dict[str, Any]:
        """Get topology statistics"""
        if not self.topology_manager.cluster_topology:
            return {}

        return {
            'total_devices': self.topology_manager.cluster_topology.total_devices,
            'total_memory_gb': self.topology_manager.cluster_topology.total_memory_gb,
            'vendor_distribution': dict(self.topology_manager.cluster_topology.vendor_distribution),
            'capability_distribution': dict(self.topology_manager.cluster_topology.capability_distribution),
        }

    def _get_power_stats(self) -> Dict[str, Any]:
        """Get power consumption statistics"""
        if not self.topology_manager.cluster_topology:
            return {}

        total_power = 0.0
        total_compute = 0.0

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                total_power += device.current_power_w
                utilized_flops = device.peak_flops_fp32 * (device.current_utilization / 100.0)
                total_compute += utilized_flops

        efficiency = total_compute / max(total_power, 1.0) if total_power > 0 else 0.0

        return {
            'total_power_w': total_power,
            'total_compute_flops': total_compute,
            'power_efficiency': efficiency
        }

    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.topology_manager.cluster_topology:
            return {}

        utilizations = []
        temperatures = []
        memory_usage = []

        for node in self.topology_manager.cluster_topology.nodes.values():
            for device in node.devices:
                utilizations.append(device.current_utilization)
                temperatures.append(device.current_temp_c)
                memory_usage.append(device.memory_used_gb / device.memory_gb * 100)

        if utilizations:
            return {
                'avg_utilization': np.mean(utilizations),
                'max_utilization': np.max(utilizations),
                'avg_temperature': np.mean(temperatures),
                'max_temperature': np.max(temperatures),
                'avg_memory_usage_percent': np.mean(memory_usage),
                'max_memory_usage_percent': np.max(memory_usage)
            }

        return {}

    def optimize_for_workload(
        self,
        job_id: str,
        required_devices: int,
        estimated_power_per_device: float = 300.0,
        thermal_sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """Optimize cluster for specific workload"""
        # Use thermal scheduler for job scheduling
        scheduling_result = self.thermal_scheduler.schedule_job(
            job_id, required_devices, estimated_power_per_device, thermal_sensitivity
        )

        if scheduling_result['success']:
            # Monitor thermal state
            self.thermal_scheduler.monitor_thermal_state()

            # Get power optimization recommendations
            power_optimization = self.power_optimizer.optimize_power_distribution(
                {job_id: 1.0}  # Single job with priority 1.0
            )

            scheduling_result['power_optimization'] = power_optimization

        return scheduling_result

    def create_optimal_device_mesh(
        self,
        world_size: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: Optional[int] = None
    ) -> DeviceMesh:
        """Create optimal device mesh for distributed training"""
        return self.mesh_optimizer.create_optimal_mesh(
            world_size, tensor_parallel_size, pipeline_parallel_size, data_parallel_size
        )

    def get_health_report(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive health report"""
        return self.health_monitor.get_performance_report(hours=hours)

    def shutdown(self):
        """Shutdown hardware adapter and stop monitoring"""
        self.health_monitor.stop_monitoring()
        logger.info("Hardware adapter shutdown complete")


class DeviceMeshOptimizer:
    """
    Optimizer for creating optimal device meshes for distributed training

    Features:
    - Topology-aware mesh creation
    - Communication cost minimization
    - Load balancing across heterogeneous hardware
    """

    def __init__(self, topology_manager: HardwareTopologyManager):
        self.topology_manager = topology_manager
        self.mesh_cache: Dict[str, DeviceMesh] = {}

    def create_optimal_mesh(
        self,
        world_size: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: Optional[int] = None
    ) -> DeviceMesh:
        """
        Create optimal device mesh for given parallelism configuration

        Args:
            world_size: Total number of devices
            tensor_parallel_size: Tensor parallelism dimension
            pipeline_parallel_size: Pipeline parallelism dimension
            data_parallel_size: Data parallelism dimension (auto-calculated if None)

        Returns:
            Optimized device mesh
        """
        if data_parallel_size is None:
            data_parallel_size = world_size // (tensor_parallel_size * pipeline_parallel_size)

        # Validate configuration
        total_required = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        if total_required != world_size:
            raise ValueError(f"Invalid parallelism configuration: {total_required} != {world_size}")

        # Cache key for mesh configuration
        cache_key = f"mesh_{world_size}_{tensor_parallel_size}_{pipeline_parallel_size}_{data_parallel_size}"
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key]

        # Get optimal device placement
        if hasattr(self.topology_manager, 'get_optimal_device_placement'):
            optimal_devices = self.topology_manager.get_optimal_device_placement(
                memory_requirement_gb=1.0,  # Minimal requirement for placement
                compute_requirement_tflops=1.0
            )[:world_size]
        else:
            # Fallback: use available devices
            optimal_devices = list(range(min(world_size, torch.cuda.device_count())))

        if len(optimal_devices) < world_size:
            raise RuntimeError(f"Not enough suitable devices: {len(optimal_devices)} < {world_size}")

        # Create mesh with topology awareness
        mesh_shape = []
        if data_parallel_size > 1:
            mesh_shape.append(data_parallel_size)
        if pipeline_parallel_size > 1:
            mesh_shape.append(pipeline_parallel_size)
        if tensor_parallel_size > 1:
            mesh_shape.append(tensor_parallel_size)

        # Optimize device ordering for communication efficiency
        optimized_devices = self._optimize_device_ordering(
            optimal_devices, mesh_shape
        )

        # Create device mesh
        device_mesh = DeviceMesh("cuda", optimized_devices.reshape(mesh_shape))

        self.mesh_cache[cache_key] = device_mesh
        return device_mesh

    def _optimize_device_ordering(
        self,
        device_ids: List[int],
        mesh_shape: List[int]
    ) -> np.ndarray:
        """Optimize device ordering for communication efficiency"""
        if not self.topology_manager.cluster_topology:
            return np.array(device_ids)

        # For now, use simple ordering that prioritizes:
        # 1. NVLink connected devices for tensor parallelism
        # 2. Same node devices for pipeline parallelism
        # 3. Cross-node for data parallelism

        optimized_order = []
        remaining_devices = device_ids.copy()

        # Group devices by node
        node_groups = {}
        for node in self.topology_manager.cluster_topology.nodes.values():
            node_devices = [dev.device_id for dev in node.devices if dev.device_id in remaining_devices]
            if node_devices:
                node_groups[node.node_id] = node_devices

        # Assign devices based on mesh dimensions
        total_devices = len(device_ids)
        devices_assigned = 0

        for node_id, node_devices in node_groups.items():
            available = min(len(node_devices), total_devices - devices_assigned)
            optimized_order.extend(node_devices[:available])
            devices_assigned += available

            if devices_assigned >= total_devices:
                break

        return np.array(optimized_order)