"""
Core Hardware Abstraction Layer (HAL) Implementation

Provides universal hardware abstraction for PyTorch,
enabling seamless integration of proprietary GPUs and AI chips.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

# Import HardwareVendor from the original location for compatibility
from ...distributed_scale.hardware_discovery import HardwareVendor

logger = logging.getLogger(__name__)


@dataclass
class DeviceMesh:
    """Cross-vendor device mesh for distributed computation"""
    mesh_id: str
    devices: list['DeviceSpec']
    topology: str = "ring"  # ring, tree, mesh, custom
    communication_backend: str | None = None
    bandwidth_matrix: list[list[float]] | None = None
    latency_matrix: list[list[float]] | None = None

    def __post_init__(self):
        if self.bandwidth_matrix is None:
            # Initialize with default bandwidth estimates
            n = len(self.devices)
            self.bandwidth_matrix = [[0.0] * n for _ in range(n)]

        if self.latency_matrix is None:
            # Initialize with default latency estimates
            n = len(self.devices)
            self.latency_matrix = [[0.0] * n for _ in range(n)]


@dataclass
class CrossVendorCapabilities:
    """Aggregated capabilities across multiple vendors"""
    total_devices: int
    vendor_distribution: dict[HardwareVendor, int]
    total_memory_gb: float
    peak_compute_tflops: float
    mixed_precision_support: bool
    cross_vendor_communication: bool
    mesh_topologies: list[str]


class ComputeCapability(Enum):
    """Hardware compute capabilities"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    TENSOR_CORES = "tensor_cores"
    MIXED_PRECISION = "mixed_precision"
    SPARSITY = "sparsity"
    VECTOR_UNITS = "vector_units"


@dataclass
class HardwareCapabilities:
    """Hardware capabilities and specifications"""
    vendor: HardwareVendor
    device_name: str
    compute_capability: str
    memory_gb: float
    peak_flops_fp32: float
    peak_flops_fp16: float
    memory_bandwidth_gbps: float
    supported_precisions: list[ComputeCapability]
    tensor_core_support: bool = False
    custom_instruction_sets: list[str] = field(default_factory=list)
    interconnect_type: str = "PCIe"
    max_power_w: float = 300.0
    # Extended vendor-specific information
    generation: str | None = None          # GPU generation (e.g., Ampere, CDNA2)
    architecture: str | None = None        # Architecture code (e.g., GA100, gfx90a)
    features: list[str] = field(default_factory=list)  # Vendor-specific features


@dataclass
class DeviceSpec:
    """Device specification for workload placement"""
    device_id: int
    vendor: HardwareVendor
    capabilities: HardwareCapabilities
    current_utilization: float = 0.0
    current_memory_usage_gb: float = 0.0
    current_temperature_c: float = 25.0
    is_available: bool = True


class VendorAdapter(ABC):
    """
    Abstract adapter for vendor-specific hardware implementations

    Enables pluggable support for different hardware vendors while
    maintaining a consistent interface for the abstraction layer.
    """

    def __init__(self, vendor: HardwareVendor):
        self.vendor = vendor
        self._devices: list[DeviceSpec] = []
        self._capabilities_cache: dict[int, HardwareCapabilities] = {}

    @abstractmethod
    def initialize_device(self, device_id: int) -> DeviceSpec:
        """Initialize and return device specification"""
        pass

    @abstractmethod
    def discover_devices(self) -> list[DeviceSpec]:
        """Discover all available devices"""
        pass

    @abstractmethod
    def compile_kernel(self, kernel_source: str, target_device: DeviceSpec) -> Any:
        """Compile kernel for target device"""
        pass

    @abstractmethod
    def optimize_memory_layout(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Apply vendor-specific memory optimizations"""
        pass

    @abstractmethod
    def create_communication_backend(self, devices: list[DeviceSpec]) -> Any:
        """Create vendor-specific communication backend"""
        pass

    @abstractmethod
    def get_device_metrics(self, device_id: int) -> dict[str, float]:
        """Get real-time device performance metrics"""
        pass

    def get_optimal_kernel_variant(self,
                                  operation: str,
                                  input_shapes: list[tuple[int, ...]],
                                  device: DeviceSpec) -> str:
        """Get optimal kernel variant for given operation and device"""
        # Default implementation - can be overridden by vendors
        if device.capabilities.tensor_core_support and operation in ["matmul", "conv"]:
            return f"{operation}_tensor_core"
        return f"{operation}_standard"


class HardwareAbstractionLayer:
    """
    Main Hardware Abstraction Layer coordinating multiple vendor adapters

    Provides unified interface for:
    - Device discovery and management
    - Workload placement optimization
    - Performance monitoring
    - Hardware-specific optimizations
    """

    def __init__(self):
        self.vendor_adapters: dict[HardwareVendor, VendorAdapter] = {}
        self.devices: dict[int, DeviceSpec] = {}
        self.performance_models: dict[HardwareVendor, Any] = {}
        self._device_counter = 0

    def register_vendor_adapter(self, adapter: VendorAdapter) -> None:
        """Register vendor-specific adapter"""
        self.vendor_adapters[adapter.vendor] = adapter

        # Discover devices from this vendor
        vendor_devices = adapter.discover_devices()
        for device in vendor_devices:
            device.device_id = self._device_counter
            self.devices[self._device_counter] = device
            self._device_counter += 1

        logger.info(f"Registered {adapter.vendor} adapter with {len(vendor_devices)} devices")

    def register_device(self, device: DeviceSpec) -> None:
        """Register a single device with the HAL"""
        device.device_id = self._device_counter
        self.devices[self._device_counter] = device
        self._device_counter += 1
        logger.debug(f"Registered device {device.device_id}: {device.vendor.value} - {device.capabilities.device_name}")

    def discover_all_hardware(self) -> dict[HardwareVendor, list[DeviceSpec]]:
        """Discover all available hardware across vendors"""
        hardware_inventory = {}

        for vendor, adapter in self.vendor_adapters.items():
            try:
                devices = adapter.discover_devices()
                hardware_inventory[vendor] = devices
                logger.info(f"Discovered {len(devices)} {vendor} devices")
            except Exception as e:
                logger.error(f"Failed to discover {vendor} devices: {e}")
                hardware_inventory[vendor] = []

        return hardware_inventory

    def get_optimal_device(self,
                          memory_requirement_gb: float,
                          compute_requirement_tflops: float,
                          preferred_vendors: list[HardwareVendor] | None = None,
                          precision_requirements: list[ComputeCapability] | None = None) -> DeviceSpec | None:
        """
        Find optimal device for given requirements

        Args:
            memory_requirement_gb: Required memory capacity
            compute_requirement_tflops: Required compute performance
            preferred_vendors: Optional list of preferred vendors
            precision_requirements: Required precision support

        Returns:
            Optimal device specification or None if no suitable device found
        """
        candidates = []

        for device in self.devices.values():
            if not device.is_available:
                continue

            # Check vendor preference
            if preferred_vendors and device.vendor not in preferred_vendors:
                continue

            # Check memory requirement
            available_memory = device.capabilities.memory_gb - device.current_memory_usage_gb
            if available_memory < memory_requirement_gb:
                continue

            # Check compute requirement
            device_tflops = device.capabilities.peak_flops_fp32 / 1e12
            if device_tflops < compute_requirement_tflops:
                continue

            # Check precision requirements
            if precision_requirements:
                if not all(prec in device.capabilities.supported_precisions
                          for prec in precision_requirements):
                    continue

            # Calculate device score
            score = self._calculate_device_score(device, memory_requirement_gb, compute_requirement_tflops)
            candidates.append((device, score))

        if not candidates:
            return None

        # Return device with highest score
        return max(candidates, key=lambda x: x[1])[0]

    def _calculate_device_score(self,
                               device: DeviceSpec,
                               memory_req: float,
                               compute_req: float) -> float:
        """Calculate device suitability score"""
        score = 0.0

        # Memory score (prefer devices with more available memory)
        available_memory = device.capabilities.memory_gb - device.current_memory_usage_gb
        memory_score = available_memory / device.capabilities.memory_gb
        score += memory_score * 0.3

        # Compute score (prefer more powerful devices)
        device_tflops = device.capabilities.peak_flops_fp32 / 1e12
        compute_score = min(device_tflops / max(compute_req, 1.0), 1.0)
        score += compute_score * 0.3

        # Utilization score (prefer less utilized devices)
        util_score = 1.0 - device.current_utilization
        score += util_score * 0.2

        # Temperature score (prefer cooler devices)
        temp_score = max(0.0, 1.0 - device.current_temperature_c / 90.0)
        score += temp_score * 0.1

        # Vendor preference bonus (can be customized)
        if device.vendor == HardwareVendor.NVIDIA:  # Example preference
            score += 0.1

        return score

    def create_device_mesh(self,
                          devices: list[DeviceSpec],
                          tensor_parallel_size: int = 1,
                          pipeline_parallel_size: int = 1,
                          data_parallel_size: int | None = None) -> torch.distributed._tensor.DeviceMesh:
        """Create device mesh for distributed operations"""
        if data_parallel_size is None:
            data_parallel_size = len(devices) // (tensor_parallel_size * pipeline_parallel_size)

        # Validate configuration
        total_required = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        if total_required > len(devices):
            raise ValueError(f"Not enough devices: need {total_required}, have {len(devices)}")

        # Group devices by vendor for optimal communication
        vendor_groups = {}
        for device in devices[:total_required]:
            if device.vendor not in vendor_groups:
                vendor_groups[device.vendor] = []
            vendor_groups[device.vendor].append(device.device_id)

        # Create optimized device ordering
        device_ids = []
        for vendor_devices in vendor_groups.values():
            device_ids.extend(vendor_devices)

        # Create mesh shape
        mesh_shape = []
        if data_parallel_size > 1:
            mesh_shape.append(data_parallel_size)
        if pipeline_parallel_size > 1:
            mesh_shape.append(pipeline_parallel_size)
        if tensor_parallel_size > 1:
            mesh_shape.append(tensor_parallel_size)

        import torch.distributed._tensor as dt
        return dt.DeviceMesh("cuda", torch.tensor(device_ids).reshape(mesh_shape))

    def get_performance_estimate(self,
                                operation: str,
                                input_shapes: list[tuple[int, ...]],
                                device: DeviceSpec) -> float:
        """Estimate operation performance on specific device"""
        adapter = self.vendor_adapters.get(device.vendor)
        if not adapter:
            return float('inf')  # Unknown performance

        # Get vendor-specific performance model
        if device.vendor in self.performance_models:
            model = self.performance_models[device.vendor]
            return model.predict(operation, input_shapes, device)

        # Fallback to simple FLOP-based estimation
        total_flops = self._estimate_operation_flops(operation, input_shapes)
        device_flops = device.capabilities.peak_flops_fp32 * (1.0 - device.current_utilization)

        return total_flops / device_flops if device_flops > 0 else float('inf')

    def _estimate_operation_flops(self, operation: str, shapes: list[tuple[int, ...]]) -> float:
        """Estimate FLOPs for operation"""
        if operation == "matmul" and len(shapes) >= 2:
            # Matrix multiplication: 2 * M * N * K
            m, k = shapes[0]
            k2, n = shapes[1]
            if k == k2:
                return 2.0 * m * n * k
        elif operation == "conv2d" and len(shapes) >= 2:
            # Convolution: approximate
            batch, in_ch, h, w = shapes[0]
            out_ch, in_ch2, kh, kw = shapes[1]
            if in_ch == in_ch2:
                return 2.0 * batch * out_ch * h * w * in_ch * kh * kw

        return 1e6  # Default estimate

    def get_cluster_status(self) -> dict[str, Any]:
        """Get comprehensive cluster status"""
        status = {
            'total_devices': len(self.devices),
            'available_devices': sum(1 for d in self.devices.values() if d.is_available),
            'vendor_distribution': {},
            'total_memory_gb': 0.0,
            'available_memory_gb': 0.0,
            'avg_utilization': 0.0,
            'device_details': []
        }

        vendor_counts = {}
        total_util = 0.0

        for device in self.devices.values():
            # Vendor distribution
            if device.vendor not in vendor_counts:
                vendor_counts[device.vendor] = 0
            vendor_counts[device.vendor] += 1

            # Memory stats
            status['total_memory_gb'] += device.capabilities.memory_gb
            available_mem = device.capabilities.memory_gb - device.current_memory_usage_gb
            status['available_memory_gb'] += available_mem

            # Utilization
            total_util += device.current_utilization

            # Device details
            status['device_details'].append({
                'device_id': device.device_id,
                'vendor': device.vendor.value,
                'name': device.capabilities.device_name,
                'utilization': device.current_utilization,
                'memory_used_gb': device.current_memory_usage_gb,
                'memory_total_gb': device.capabilities.memory_gb,
                'temperature_c': device.current_temperature_c,
                'available': device.is_available
            })

        status['vendor_distribution'] = {v.value: c for v, c in vendor_counts.items()}
        status['avg_utilization'] = total_util / len(self.devices) if self.devices else 0.0

        return status

    def optimize_workload_placement(self,
                                  workloads: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        """Optimize placement of multiple workloads across devices"""
        placement = {device_id: [] for device_id in self.devices.keys()}

        # Sort workloads by resource requirements (largest first)
        sorted_workloads = sorted(workloads,
                                key=lambda w: w.get('memory_gb', 0) + w.get('compute_tflops', 0),
                                reverse=True)

        for workload in sorted_workloads:
            optimal_device = self.get_optimal_device(
                memory_requirement_gb=workload.get('memory_gb', 0),
                compute_requirement_tflops=workload.get('compute_tflops', 0),
                preferred_vendors=workload.get('preferred_vendors'),
                precision_requirements=workload.get('precision_requirements')
            )

            if optimal_device:
                placement[optimal_device.device_id].append(workload)
                # Update device utilization for next placement decision
                optimal_device.current_memory_usage_gb += workload.get('memory_gb', 0)
                optimal_device.current_utilization += workload.get('utilization_estimate', 0.1)

        return placement

    def create_cross_vendor_mesh(self,
                                 devices: list[DeviceSpec],
                                 mesh_id: str,
                                 topology: str = "ring") -> DeviceMesh:
        """
        Create cross-vendor device mesh for heterogeneous training

        Args:
            devices: List of devices from potentially different vendors
            mesh_id: Unique identifier for the mesh
            topology: Communication topology (ring, tree, mesh, custom)

        Returns:
            DeviceMesh object with optimized communication paths
        """
        if not devices:
            raise ValueError("Cannot create mesh with empty device list")

        # Validate devices are available
        available_devices = [d for d in devices if d.is_available]
        if len(available_devices) != len(devices):
            logger.warning(f"Some devices unavailable: {len(devices) - len(available_devices)} filtered out")
            devices = available_devices

        # Analyze vendor distribution for optimal topology
        vendor_groups = {}
        for device in devices:
            if device.vendor not in vendor_groups:
                vendor_groups[device.vendor] = []
            vendor_groups[device.vendor].append(device)

        # Select optimal communication backend
        communication_backend = self._select_communication_backend(vendor_groups)

        # Create device mesh with optimized layout
        mesh = DeviceMesh(
            mesh_id=mesh_id,
            devices=devices,
            topology=topology,
            communication_backend=communication_backend
        )

        # Calculate bandwidth and latency matrices
        self._populate_mesh_matrices(mesh)

        logger.info(f"Created cross-vendor mesh {mesh_id} with {len(devices)} devices "
                   f"from {len(vendor_groups)} vendors using {communication_backend}")

        return mesh

    def _select_communication_backend(self, vendor_groups: dict[HardwareVendor, list[DeviceSpec]]) -> str:
        """Select optimal communication backend for vendor mix"""
        # If all NVIDIA, use NCCL
        if len(vendor_groups) == 1 and HardwareVendor.NVIDIA in vendor_groups:
            return "nccl"

        # If all AMD, use RCCL
        if len(vendor_groups) == 1 and HardwareVendor.AMD in vendor_groups:
            return "rccl"

        # If mixed vendors, use Gloo or MPI
        nvidia_count = len(vendor_groups.get(HardwareVendor.NVIDIA, []))
        total_count = sum(len(devices) for devices in vendor_groups.values())

        if total_count == 0:
            return "gloo"  # Default backend for no devices

        if nvidia_count / total_count > 0.7:
            return "mixed_nccl_gloo"  # Mostly NVIDIA with fallback
        else:
            return "gloo"  # Generic cross-vendor backend

    def _populate_mesh_matrices(self, mesh: DeviceMesh) -> None:
        """Populate bandwidth and latency matrices for device mesh"""
        len(mesh.devices)

        for i, device_a in enumerate(mesh.devices):
            for j, device_b in enumerate(mesh.devices):
                if i == j:
                    mesh.bandwidth_matrix[i][j] = float('inf')  # Self-connection
                    mesh.latency_matrix[i][j] = 0.0
                    continue

                # Estimate inter-device bandwidth and latency
                bandwidth_gbps, latency_ms = self._estimate_device_connection(device_a, device_b)
                mesh.bandwidth_matrix[i][j] = bandwidth_gbps
                mesh.latency_matrix[i][j] = latency_ms

    def _estimate_device_connection(self, device_a: DeviceSpec, device_b: DeviceSpec) -> tuple[float, float]:
        """Estimate bandwidth and latency between two devices"""
        # Same vendor, likely better interconnects
        if device_a.vendor == device_b.vendor:
            if device_a.vendor == HardwareVendor.NVIDIA:
                # Check for NVLink
                if device_a.capabilities.interconnect_type == "NVLink":
                    return 600.0, 1.0  # NVLink bandwidth ~600 GB/s, latency ~1ms
                else:
                    return 32.0, 5.0   # PCIe bandwidth ~32 GB/s, latency ~5ms
            elif device_a.vendor == HardwareVendor.AMD:
                return 50.0, 3.0       # Infinity Fabric estimates
            else:
                return 16.0, 10.0      # Generic same-vendor estimate
        else:
            # Cross-vendor likely through PCIe/network
            return 12.8, 20.0          # PCIe 4.0 x16 bandwidth, higher latency

    def get_cross_vendor_capabilities(self) -> CrossVendorCapabilities:
        """Get aggregated capabilities across all vendors"""
        vendor_counts = {}
        total_memory = 0.0
        total_compute = 0.0
        mixed_precision_devices = 0
        cross_vendor_comm = len(set(d.vendor for d in self.devices.values())) > 1  # noqa: C401

        for device in self.devices.values():
            # Count by vendor
            if device.vendor not in vendor_counts:
                vendor_counts[device.vendor] = 0
            vendor_counts[device.vendor] += 1

            # Aggregate resources
            total_memory += device.capabilities.memory_gb
            total_compute += device.capabilities.peak_flops_fp32 / 1e12  # Convert to TFLOPS

            # Check mixed precision support
            if ComputeCapability.MIXED_PRECISION in device.capabilities.supported_precisions:
                mixed_precision_devices += 1

        # Determine supported mesh topologies
        device_count = len(self.devices)
        mesh_topologies = ["ring"]  # Always supported

        if device_count >= 4:
            mesh_topologies.append("tree")
        if device_count >= 8:
            mesh_topologies.append("mesh")
        if cross_vendor_comm:
            mesh_topologies.append("hybrid")

        return CrossVendorCapabilities(
            total_devices=device_count,
            vendor_distribution=vendor_counts,
            total_memory_gb=total_memory,
            peak_compute_tflops=total_compute,
            mixed_precision_support=mixed_precision_devices > 0,
            cross_vendor_communication=cross_vendor_comm,
            mesh_topologies=mesh_topologies
        )
