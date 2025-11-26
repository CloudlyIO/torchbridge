"""
Core Hardware Abstraction Layer (HAL) Implementation

Provides universal hardware abstraction for PyTorch optimization framework,
enabling seamless integration of proprietary GPUs and AI chips.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HardwareVendor(Enum):
    """Supported hardware vendors"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    CUSTOM_ASIC = "custom_asic"
    FPGA = "fpga"
    TPU = "tpu"
    CEREBRAS = "cerebras"
    GRAPHCORE = "graphcore"
    UNKNOWN = "unknown"


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
    supported_precisions: List[ComputeCapability]
    tensor_core_support: bool = False
    custom_instruction_sets: List[str] = field(default_factory=list)
    interconnect_type: str = "PCIe"
    max_power_w: float = 300.0


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
    maintaining a consistent interface for the optimization framework.
    """

    def __init__(self, vendor: HardwareVendor):
        self.vendor = vendor
        self._devices: List[DeviceSpec] = []
        self._capabilities_cache: Dict[int, HardwareCapabilities] = {}

    @abstractmethod
    def initialize_device(self, device_id: int) -> DeviceSpec:
        """Initialize and return device specification"""
        pass

    @abstractmethod
    def discover_devices(self) -> List[DeviceSpec]:
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
    def create_communication_backend(self, devices: List[DeviceSpec]) -> Any:
        """Create vendor-specific communication backend"""
        pass

    @abstractmethod
    def get_device_metrics(self, device_id: int) -> Dict[str, float]:
        """Get real-time device performance metrics"""
        pass

    def get_optimal_kernel_variant(self,
                                  operation: str,
                                  input_shapes: List[Tuple[int, ...]],
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
        self.vendor_adapters: Dict[HardwareVendor, VendorAdapter] = {}
        self.devices: Dict[int, DeviceSpec] = {}
        self.performance_models: Dict[HardwareVendor, Any] = {}
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

    def discover_all_hardware(self) -> Dict[HardwareVendor, List[DeviceSpec]]:
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
                          preferred_vendors: Optional[List[HardwareVendor]] = None,
                          precision_requirements: Optional[List[ComputeCapability]] = None) -> Optional[DeviceSpec]:
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
                          devices: List[DeviceSpec],
                          tensor_parallel_size: int = 1,
                          pipeline_parallel_size: int = 1,
                          data_parallel_size: Optional[int] = None) -> torch.distributed._tensor.DeviceMesh:
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
                                input_shapes: List[Tuple[int, ...]],
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

    def _estimate_operation_flops(self, operation: str, shapes: List[Tuple[int, ...]]) -> float:
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

    def get_cluster_status(self) -> Dict[str, Any]:
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
                                  workloads: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
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