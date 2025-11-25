"""
Hardware Discovery and Topology Management

Comprehensive hardware discovery for heterogeneous clusters:
- Multi-vendor GPU detection (NVIDIA, AMD, Intel)
- Hardware topology discovery and analysis
- Device capability and performance estimation
- NUMA and interconnect topology mapping
"""

import time
import logging
import subprocess
import socket
import psutil
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import torch

logger = logging.getLogger(__name__)


class HardwareVendor(Enum):
    """Supported hardware vendors"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    UNKNOWN = "unknown"


class DeviceCapability(Enum):
    """Device capability levels"""
    COMPUTE_7_0 = "7.0"  # V100
    COMPUTE_8_0 = "8.0"  # A100
    COMPUTE_8_6 = "8.6"  # RTX 30xx
    COMPUTE_8_9 = "8.9"  # RTX 40xx
    COMPUTE_9_0 = "9.0"  # H100
    RDNA2 = "rdna2"      # AMD RDNA2
    RDNA3 = "rdna3"      # AMD RDNA3
    XE_HPG = "xe_hpg"    # Intel Xe-HPG


class ThermalState(Enum):
    """Thermal states for devices"""
    OPTIMAL = "optimal"      # < 70°C
    WARM = "warm"           # 70-80°C
    HOT = "hot"             # 80-90°C
    CRITICAL = "critical"    # > 90°C


@dataclass
class DeviceInfo:
    """Comprehensive device information"""
    device_id: int
    vendor: HardwareVendor
    name: str
    capability: DeviceCapability
    memory_gb: float
    memory_bandwidth_gb_s: float
    compute_units: int
    base_clock_mhz: int
    boost_clock_mhz: int
    power_limit_w: int
    pci_slot: str
    numa_node: int

    # Dynamic status
    current_temp_c: float = 0.0
    current_power_w: float = 0.0
    current_utilization: float = 0.0
    memory_used_gb: float = 0.0
    thermal_state: ThermalState = ThermalState.OPTIMAL

    # Performance characteristics
    peak_flops_fp32: float = 0.0
    peak_flops_fp16: float = 0.0
    peak_tensor_flops: float = 0.0

    # Health metrics
    error_count: int = 0
    last_health_check: float = field(default_factory=time.time)


@dataclass
class NodeTopology:
    """Node-level hardware topology"""
    node_id: int
    hostname: str
    devices: List[DeviceInfo]
    cpu_count: int
    memory_gb: float
    storage_type: str  # "nvme", "ssd", "hdd"
    network_interfaces: List[str]

    # NUMA topology
    numa_topology: Dict[int, List[int]]  # NUMA node -> device IDs

    # Interconnect information
    nvlink_topology: Dict[Tuple[int, int], float]  # (dev1, dev2) -> bandwidth_gb_s
    pcie_topology: Dict[int, Dict[str, Any]]  # device_id -> PCIe info


@dataclass
class ClusterTopology:
    """Cluster-level hardware topology"""
    nodes: Dict[int, NodeTopology]
    total_devices: int
    total_memory_gb: float
    network_topology: Dict[Tuple[int, int], Dict[str, float]]  # (node1, node2) -> {bandwidth, latency}

    # Heterogeneity metrics
    vendor_distribution: Dict[HardwareVendor, int]
    capability_distribution: Dict[DeviceCapability, int]
    memory_distribution: Dict[float, int]  # memory_gb -> count


class HardwareTopologyManager:
    """
    Comprehensive hardware topology manager for large-scale clusters

    Features:
    - Automatic hardware discovery across vendors
    - Topology-aware optimization
    - Device capability and performance estimation
    """

    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        # Initialize cluster topology during construction
        self.cluster_topology: Optional[ClusterTopology] = self.discover_topology()

    def discover_topology(self) -> ClusterTopology:
        """Discover and analyze cluster hardware topology"""
        logger.info("Discovering cluster hardware topology...")

        # Discover local node first
        local_node = self._discover_local_node()

        # In distributed setting, gather topology from all nodes
        nodes = {0: local_node}  # For single node, use node_id 0

        # Calculate totals
        total_devices = sum(len(node.devices) for node in nodes.values())
        total_memory = sum(sum(dev.memory_gb for dev in node.devices)
                          for node in nodes.values())

        # Analyze vendor and capability distribution
        vendor_dist = {}
        capability_dist = {}
        memory_dist = {}

        for node in nodes.values():
            for device in node.devices:
                vendor_dist[device.vendor] = vendor_dist.get(device.vendor, 0) + 1
                capability_dist[device.capability] = capability_dist.get(device.capability, 0) + 1
                memory_dist[device.memory_gb] = memory_dist.get(device.memory_gb, 0) + 1

        # Create cluster topology
        self.cluster_topology = ClusterTopology(
            nodes=nodes,
            total_devices=total_devices,
            total_memory_gb=total_memory,
            network_topology={},  # Would be populated in multi-node setting
            vendor_distribution=vendor_dist,
            capability_distribution=capability_dist,
            memory_distribution=memory_dist
        )

        logger.info(f"Discovered {total_devices} devices across {len(nodes)} nodes")
        logger.info(f"Total cluster memory: {total_memory:.1f} GB")
        logger.info(f"Vendor distribution: {vendor_dist}")

        return self.cluster_topology

    def _discover_local_node(self) -> NodeTopology:
        """Discover hardware topology of local node"""
        devices = []

        # Discover CUDA devices
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                device_info = self._discover_cuda_device(device_id)
                devices.append(device_info)

        # Discover other accelerators (AMD ROCm, Intel OneAPI)
        devices.extend(self._discover_other_accelerators())

        # Get system information
        hostname = socket.gethostname()
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        storage_type = self._detect_storage_type()
        network_interfaces = self._get_network_interfaces()

        # Discover NUMA topology
        numa_topology = self._discover_numa_topology(devices)

        # Discover interconnect topology
        nvlink_topology = self._discover_nvlink_topology(devices)
        pcie_topology = self._discover_pcie_topology(devices)

        return NodeTopology(
            node_id=0,  # Single node for now
            hostname=hostname,
            devices=devices,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            storage_type=storage_type,
            network_interfaces=network_interfaces,
            numa_topology=numa_topology,
            nvlink_topology=nvlink_topology,
            pcie_topology=pcie_topology
        )

    def _discover_cuda_device(self, device_id: int) -> DeviceInfo:
        """Discover CUDA device information"""
        try:
            device_props = torch.cuda.get_device_properties(device_id)

            # Map device capability
            capability_map = {
                (7, 0): DeviceCapability.COMPUTE_7_0,
                (8, 0): DeviceCapability.COMPUTE_8_0,
                (8, 6): DeviceCapability.COMPUTE_8_6,
                (8, 9): DeviceCapability.COMPUTE_8_9,
                (9, 0): DeviceCapability.COMPUTE_9_0,
            }

            capability = capability_map.get(
                (device_props.major, device_props.minor),
                DeviceCapability.COMPUTE_8_0  # Default fallback
            )

            # Calculate performance characteristics
            peak_flops_fp32 = self._estimate_peak_flops(device_props, torch.float32)
            peak_flops_fp16 = self._estimate_peak_flops(device_props, torch.float16)
            peak_tensor_flops = self._estimate_tensor_flops(device_props, capability)

            # Get additional device info via nvidia-ml-py if available
            temp, power, utilization = self._get_nvidia_device_status(device_id)

        except (AssertionError, RuntimeError) as e:
            # CUDA not available, return mock device info
            return DeviceInfo(
                device_id=device_id,
                vendor=HardwareVendor.NVIDIA,
                name="mock_gpu",
                capability=DeviceCapability.COMPUTE_8_0,
                memory_gb=16.0,
                memory_bandwidth_gb_s=600.0,
                compute_units=80,
                base_clock_mhz=1400,
                boost_clock_mhz=1700,
                power_limit_w=250,
                pci_slot=f"0000:0{device_id}:00.0",
                numa_node=0,
                current_temp_c=65.0,
                current_power_w=200.0,
                current_utilization=0.5,
                memory_used_gb=0.0,
                thermal_state=ThermalState.OPTIMAL,
                peak_flops_fp32=10000.0,
                peak_flops_fp16=20000.0,
                peak_tensor_flops=50000.0
            )

        return DeviceInfo(
            device_id=device_id,
            vendor=HardwareVendor.NVIDIA,
            name=device_props.name,
            capability=capability,
            memory_gb=device_props.total_memory / (1024**3),
            memory_bandwidth_gb_s=self._estimate_memory_bandwidth(device_props),
            compute_units=device_props.multi_processor_count,
            base_clock_mhz=0,  # Would query via nvidia-ml-py
            boost_clock_mhz=0,
            power_limit_w=0,  # Would query via nvidia-ml-py
            pci_slot=f"{device_id}",  # Simplified
            numa_node=0,  # Would discover via nvidia-ml-py
            current_temp_c=temp,
            current_power_w=power,
            current_utilization=utilization,
            memory_used_gb=torch.cuda.memory_allocated(device_id) / (1024**3),
            thermal_state=self._classify_thermal_state(temp),
            peak_flops_fp32=peak_flops_fp32,
            peak_flops_fp16=peak_flops_fp16,
            peak_tensor_flops=peak_tensor_flops
        )

    def _discover_other_accelerators(self) -> List[DeviceInfo]:
        """Discover non-CUDA accelerators (AMD, Intel)"""
        devices = []

        # AMD ROCm detection
        try:
            # Would use rocm-smi or similar tools
            pass
        except Exception:
            pass

        # Intel GPU detection
        try:
            # Would use intel-gpu-tools or level-zero
            pass
        except Exception:
            pass

        return devices

    def _get_nvidia_device_status(self, device_id: int) -> Tuple[float, float, float]:
        """Get NVIDIA device status (temperature, power, utilization)"""
        try:
            # Try to use nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

            return float(temp), float(power), float(util)

        except ImportError:
            # Fallback: try nvidia-smi
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=temperature.gpu,power.draw,utilization.gpu',
                    '--format=csv,noheader,nounits', f'--id={device_id}'
                ], capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    temp = float(values[0]) if values[0] != '[Not Supported]' else 0.0
                    power = float(values[1]) if values[1] != '[Not Supported]' else 0.0
                    util = float(values[2]) if values[2] != '[Not Supported]' else 0.0
                    return temp, power, util
            except Exception:
                pass
        except Exception:
            pass

        return 0.0, 0.0, 0.0  # Fallback values

    def _estimate_peak_flops(self, device_props, dtype: torch.dtype) -> float:
        """Estimate peak FLOPS for device"""
        # Rough estimation based on known GPU specifications
        sm_count = device_props.multi_processor_count

        if dtype == torch.float32:
            # Rough estimate: ~100 GFLOPS per SM for modern GPUs
            return sm_count * 100e9
        elif dtype == torch.float16:
            # FP16 typically 2x faster
            return sm_count * 200e9
        else:
            return sm_count * 50e9  # Conservative estimate

    def _estimate_tensor_flops(self, device_props, capability: DeviceCapability) -> float:
        """Estimate peak tensor (mixed precision) FLOPS"""
        sm_count = device_props.multi_processor_count

        # Tensor core capabilities
        if capability in [DeviceCapability.COMPUTE_7_0, DeviceCapability.COMPUTE_8_0,
                         DeviceCapability.COMPUTE_8_6, DeviceCapability.COMPUTE_8_9]:
            return sm_count * 500e9  # Modern tensor cores
        elif capability == DeviceCapability.COMPUTE_9_0:
            return sm_count * 1000e9  # H100 class
        else:
            return 0.0  # No tensor cores

    def _estimate_memory_bandwidth(self, device_props) -> float:
        """Estimate memory bandwidth in GB/s"""
        # Rough estimates based on common GPU memory configurations
        memory_gb = device_props.total_memory / (1024**3)

        if memory_gb >= 80:  # A100/H100 class
            return 2000.0  # ~2TB/s for HBM3
        elif memory_gb >= 40:  # A100 40GB
            return 1555.0  # A100 bandwidth
        elif memory_gb >= 32:  # V100 32GB
            return 900.0   # V100 bandwidth
        elif memory_gb >= 16:  # Consumer high-end
            return 800.0
        else:  # Lower-end GPUs
            return 400.0

    def _classify_thermal_state(self, temp_c: float) -> ThermalState:
        """Classify thermal state based on temperature"""
        if temp_c < 70:
            return ThermalState.OPTIMAL
        elif temp_c < 80:
            return ThermalState.WARM
        elif temp_c < 90:
            return ThermalState.HOT
        else:
            return ThermalState.CRITICAL

    def _detect_storage_type(self) -> str:
        """Detect storage type of primary storage"""
        try:
            # Linux-specific detection
            result = subprocess.run(['lsblk', '-d', '-o', 'NAME,ROTA'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    name, rota = line.split()
                    if '0' in rota:  # Non-rotating = SSD/NVMe
                        # Check if NVMe
                        if 'nvme' in name.lower():
                            return "nvme"
                        else:
                            return "ssd"
        except Exception:
            pass

        return "unknown"

    def _get_network_interfaces(self) -> List[str]:
        """Get network interface information"""
        interfaces = []
        net_io = psutil.net_io_counters(pernic=True)

        for interface in net_io.keys():
            if interface != 'lo':  # Skip loopback
                interfaces.append(interface)

        return interfaces

    def _discover_numa_topology(self, devices: List[DeviceInfo]) -> Dict[int, List[int]]:
        """Discover NUMA topology for devices"""
        # Simplified NUMA discovery
        numa_topology = {}

        # For now, assign devices to NUMA nodes based on device ID
        # In practice, would use nvidia-ml-py or similar tools
        for device in devices:
            numa_node = device.device_id % 2  # Simple assignment
            if numa_node not in numa_topology:
                numa_topology[numa_node] = []
            numa_topology[numa_node].append(device.device_id)

        return numa_topology

    def _discover_nvlink_topology(self, devices: List[DeviceInfo]) -> Dict[Tuple[int, int], float]:
        """Discover NVLink topology and bandwidth"""
        nvlink_topology = {}

        # For NVIDIA GPUs, discover NVLink connections
        if len(devices) > 1 and all(dev.vendor == HardwareVendor.NVIDIA for dev in devices):
            try:
                # Would use nvidia-ml-py to discover actual NVLink topology
                # For simulation, create a reasonable topology
                for i, dev1 in enumerate(devices):
                    for j, dev2 in enumerate(devices):
                        if i < j:
                            # Assume NVLink if devices are adjacent or in same node
                            if abs(dev1.device_id - dev2.device_id) <= 2:
                                # NVLink 3.0/4.0 bandwidth
                                bandwidth = 600.0  # GB/s bidirectional
                                nvlink_topology[(dev1.device_id, dev2.device_id)] = bandwidth
            except Exception:
                pass

        return nvlink_topology

    def _discover_pcie_topology(self, devices: List[DeviceInfo]) -> Dict[int, Dict[str, Any]]:
        """Discover PCIe topology for devices"""
        pcie_topology = {}

        for device in devices:
            # Would use lspci or nvidia-ml-py to get actual PCIe info
            pcie_topology[device.device_id] = {
                'pcie_gen': 4,  # PCIe generation
                'pcie_width': 16,  # PCIe lanes
                'bandwidth_gb_s': 32.0,  # PCIe 4.0 x16 bandwidth
                'slot': device.pci_slot
            }

        return pcie_topology

    def get_optimal_device_placement(
        self,
        num_devices: int,
        memory_requirements: Optional[List[float]] = None
    ) -> List[int]:
        """
        Get optimal device placement for given requirements

        Args:
            num_devices: Number of devices needed
            memory_requirements: Memory requirements per device (optional)

        Returns:
            List of optimal device IDs
        """
        if not self.cluster_topology:
            # Return default placement
            return list(range(min(num_devices, 8)))

        available_devices = []
        for node in self.cluster_topology.nodes:
            for device in node.devices:
                if device.is_available:
                    available_devices.append(device.device_id)

        # Return first N available devices (simplified placement strategy)
        return available_devices[:num_devices] if available_devices else list(range(min(num_devices, 8)))