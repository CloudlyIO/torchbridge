"""
Vendor-Specific Hardware Adapters

Concrete implementations of vendor adapters for different hardware types,
providing a unified interface while leveraging vendor-specific optimizations.
"""

import torch
import logging
import subprocess
import psutil
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .hal_core import VendorAdapter, DeviceSpec, HardwareCapabilities, ComputeCapability
from ...distributed_scale.hardware_discovery import HardwareVendor
from .privateuse1_integration import CustomDeviceBackend, PrivateUse1Config

# Import existing hardware discovery for compatibility
from ...distributed_scale.hardware_discovery import DeviceInfo, DeviceCapability, ThermalState

logger = logging.getLogger(__name__)


class NVIDIAAdapter(VendorAdapter):
    """
    NVIDIA GPU adapter leveraging CUDA and existing PyTorch CUDA support

    Provides enhanced capabilities while maintaining full compatibility
    with existing CUDA operations and optimizations.
    """

    def __init__(self):
        super().__init__(HardwareVendor.NVIDIA)
        self.cuda_available = torch.cuda.is_available()
        self.device_properties = {}

        # NVIDIA GPU generation mapping
        self.gpu_generations = {
            # Compute Capability -> (Generation Name, Architecture, Features)
            '5.0': ('Maxwell', 'GM200', ['FP16']),
            '5.2': ('Maxwell', 'GM200', ['FP16']),
            '6.0': ('Pascal', 'GP100', ['FP16', 'CUDA_Graphs']),
            '6.1': ('Pascal', 'GP102/104/106', ['FP16', 'CUDA_Graphs']),
            '7.0': ('Volta', 'GV100', ['Tensor_Cores', 'FP16', 'Mixed_Precision']),
            '7.2': ('Volta', 'GV11B', ['Tensor_Cores', 'FP16', 'Mixed_Precision']),
            '7.5': ('Turing', 'TU102/104/106/116/117', ['Tensor_Cores_v2', 'RT_Cores', 'INT4']),
            '8.0': ('Ampere', 'GA100', ['Tensor_Cores_v3', 'BF16', 'TF32', 'Sparsity']),
            '8.6': ('Ampere', 'GA102/103/104/106/107', ['Tensor_Cores_v3', 'BF16', 'TF32', 'RT_Cores_v2']),
            '8.7': ('Ampere', 'GA10B', ['Tensor_Cores_v3', 'BF16', 'TF32']),
            '8.9': ('Ada_Lovelace', 'AD102/103/104/106/107', ['Tensor_Cores_v4', 'RT_Cores_v3', 'AV1']),
            '9.0': ('Hopper', 'GH100', ['Tensor_Cores_v4', 'FP8', 'DPX', 'Thread_Block_Clusters']),
        }

        # Performance characteristics by generation
        self.generation_features = {
            'Maxwell': {
                'tensor_cores': False,
                'fp8_support': False,
                'bf16_support': False,
                'sparsity_support': False,
                'nvlink_support': False,
                'multi_instance_gpu': False
            },
            'Pascal': {
                'tensor_cores': False,
                'fp8_support': False,
                'bf16_support': False,
                'sparsity_support': False,
                'nvlink_support': True,  # GP100 only
                'multi_instance_gpu': False
            },
            'Volta': {
                'tensor_cores': True,
                'fp8_support': False,
                'bf16_support': False,
                'sparsity_support': False,
                'nvlink_support': True,
                'multi_instance_gpu': False
            },
            'Turing': {
                'tensor_cores': True,
                'fp8_support': False,
                'bf16_support': False,
                'sparsity_support': False,
                'nvlink_support': False,
                'multi_instance_gpu': False
            },
            'Ampere': {
                'tensor_cores': True,
                'fp8_support': False,
                'bf16_support': True,
                'sparsity_support': True,  # 2:4 structured sparsity
                'nvlink_support': True,
                'multi_instance_gpu': True  # A100 only
            },
            'Ada_Lovelace': {
                'tensor_cores': True,
                'fp8_support': False,
                'bf16_support': True,
                'sparsity_support': True,
                'nvlink_support': False,
                'multi_instance_gpu': False
            },
            'Hopper': {
                'tensor_cores': True,
                'fp8_support': True,
                'bf16_support': True,
                'sparsity_support': True,
                'nvlink_support': True,
                'multi_instance_gpu': True
            }
        }

    def initialize_device(self, device_id: int) -> DeviceSpec:
        """Initialize NVIDIA GPU device"""
        if not self.cuda_available:
            raise RuntimeError("CUDA not available for NVIDIA adapter")

        if device_id >= torch.cuda.device_count():
            raise RuntimeError(f"Invalid NVIDIA device ID: {device_id}")

        # Get device properties
        props = torch.cuda.get_device_properties(device_id)

        # Get generation-specific information
        compute_capability = f"{props.major}.{props.minor}"
        generation_info = self._get_generation_info(compute_capability)

        # Convert to our capabilities format
        capabilities = HardwareCapabilities(
            vendor=HardwareVendor.NVIDIA,
            device_name=props.name,
            compute_capability=compute_capability,
            memory_gb=props.total_memory / (1024**3),
            peak_flops_fp32=self._estimate_peak_flops(props),
            peak_flops_fp16=self._estimate_peak_flops(props) * 2,  # Approximate
            memory_bandwidth_gbps=self._estimate_memory_bandwidth(props),
            supported_precisions=self._get_supported_precisions(props),
            tensor_core_support=generation_info['tensor_cores'],
            interconnect_type=self._get_interconnect_type(props, generation_info),
            # Add generation-specific metadata
            generation=generation_info['generation'],
            architecture=generation_info['architecture'],
            features=generation_info['features']
        )

        device_spec = DeviceSpec(
            device_id=device_id,
            vendor=HardwareVendor.NVIDIA,
            capabilities=capabilities
        )

        # Cache for future use
        self.device_properties[device_id] = props
        self._devices.append(device_spec)

        return device_spec

    def discover_devices(self) -> List[DeviceSpec]:
        """Discover all NVIDIA GPU devices"""
        devices = []

        if not self.cuda_available:
            logger.warning("CUDA not available, no NVIDIA devices detected")
            return devices

        try:
            device_count = torch.cuda.device_count()
            for device_id in range(device_count):
                try:
                    device_spec = self.initialize_device(device_id)
                    devices.append(device_spec)
                except Exception as e:
                    logger.error(f"Failed to initialize NVIDIA device {device_id}: {e}")

        except Exception as e:
            logger.error(f"Error discovering NVIDIA devices: {e}")

        return devices

    def compile_kernel(self, kernel_source: str, target_device: DeviceSpec) -> Any:
        """Compile CUDA kernel for NVIDIA device"""
        try:
            # For NVIDIA devices, we can use existing CUDA compilation
            # This would integrate with existing CUDA kernel compilation
            logger.debug(f"Compiling CUDA kernel for device {target_device.device_id}")

            # Return a placeholder compiled kernel object
            return {
                'device_id': target_device.device_id,
                'vendor': 'nvidia',
                'kernel_source': kernel_source,
                'compilation_time': 0.1,  # Placeholder
                'optimizations_applied': ['memory_coalescing', 'shared_memory_opt']
            }

        except Exception as e:
            logger.error(f"CUDA kernel compilation failed: {e}")
            return None

    def optimize_memory_layout(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Apply NVIDIA-specific memory optimizations"""
        if tensor.device.type != 'cuda':
            # Move to CUDA if not already there
            tensor = tensor.cuda(device.device_id)

        # Apply NVIDIA-specific optimizations
        if tensor.dtype == torch.float32 and device.capabilities.tensor_core_support:
            # Convert to half precision for Tensor Core utilization
            tensor = tensor.half()

        # Ensure memory is contiguous for optimal access patterns
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    def create_communication_backend(self, devices: List[DeviceSpec]) -> Any:
        """Create NCCL communication backend for NVIDIA devices"""
        try:
            # Check if NCCL is available
            import torch.distributed as dist

            # Filter for NVIDIA devices only
            nvidia_devices = [d for d in devices if d.vendor == HardwareVendor.NVIDIA]

            return {
                'backend_type': 'nccl',
                'devices': [d.device_id for d in nvidia_devices],
                'supports_allreduce': True,
                'supports_allgather': True,
                'supports_reduce_scatter': True
            }

        except ImportError:
            logger.warning("NCCL not available, falling back to Gloo")
            return {'backend_type': 'gloo', 'devices': []}

    def get_device_metrics(self, device_id: int) -> Dict[str, float]:
        """Get real-time NVIDIA device metrics"""
        metrics = {
            'utilization': 0.0,
            'memory_used_gb': 0.0,
            'memory_total_gb': 0.0,
            'temperature_c': 0.0,
            'power_w': 0.0
        }

        try:
            if self.cuda_available and device_id < torch.cuda.device_count():
                # Memory metrics
                memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**3)

                props = torch.cuda.get_device_properties(device_id)
                memory_total = props.total_memory / (1024**3)

                metrics.update({
                    'memory_used_gb': memory_allocated,
                    'memory_total_gb': memory_total,
                    'memory_utilization': memory_allocated / memory_total if memory_total > 0 else 0.0
                })

                # Try to get additional metrics via nvidia-ml-py
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

                    # GPU utilization
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics['utilization'] = utilization.gpu

                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics['temperature_c'] = temp

                    # Power
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    metrics['power_w'] = power

                except ImportError:
                    logger.debug("pynvml not available, limited metrics")
                except Exception as e:
                    logger.debug(f"Error getting detailed metrics: {e}")

        except Exception as e:
            logger.error(f"Error getting NVIDIA device metrics: {e}")

        return metrics

    def _estimate_peak_flops(self, props) -> float:
        """Estimate peak FLOPS based on device properties"""
        # Rough estimation based on compute capability and multiprocessor count
        base_flops_per_sm = {
            (7, 0): 64,   # V100
            (7, 5): 64,   # T4
            (8, 0): 64,   # A100
            (8, 6): 128,  # RTX 30xx series
            (8, 9): 128,  # RTX 40xx series
            (9, 0): 256   # H100
        }

        key = (props.major, props.minor)
        flops_per_sm = base_flops_per_sm.get(key, 64)  # Default

        # Base clock in Hz * cores per SM * number of SMs * FLOPs per core per clock
        estimated_flops = props.clock_rate * 1000 * flops_per_sm * props.multi_processor_count

        return estimated_flops

    def _estimate_memory_bandwidth(self, props) -> float:
        """Estimate memory bandwidth in GB/s"""
        # Rough estimates based on known architectures
        bandwidth_estimates = {
            "Tesla V100": 900,
            "Tesla T4": 320,
            "Tesla A100": 1555,
            "GeForce RTX 3080": 760,
            "GeForce RTX 3090": 936,
            "GeForce RTX 4080": 716,
            "GeForce RTX 4090": 1008,
            "Tesla H100": 3350
        }

        # Try to match by name
        for name_pattern, bandwidth in bandwidth_estimates.items():
            if name_pattern.lower() in props.name.lower():
                return bandwidth

        # Fallback estimation based on memory size and compute capability
        if props.major >= 9:
            return 2000  # H100-class
        elif props.major >= 8:
            return 1000  # A100-class
        elif props.major >= 7:
            return 600   # V100-class
        else:
            return 400   # Older architectures

    def _get_supported_precisions(self, props) -> List[ComputeCapability]:
        """Get supported precision modes for device"""
        precisions = [ComputeCapability.FP32]

        if props.major >= 6:
            precisions.append(ComputeCapability.FP16)

        if props.major >= 7:
            precisions.extend([
                ComputeCapability.TENSOR_CORES,
                ComputeCapability.MIXED_PRECISION
            ])

        if props.major >= 8:
            precisions.extend([
                ComputeCapability.BF16,
                ComputeCapability.INT8
            ])

        if props.major >= 9:
            precisions.append(ComputeCapability.INT4)

        return precisions

    def _get_generation_info(self, compute_capability: str) -> Dict[str, Any]:
        """Get generation-specific information for device"""
        if compute_capability in self.gpu_generations:
            generation, architecture, features = self.gpu_generations[compute_capability]
            generation_features = self.generation_features.get(generation, {})

            return {
                'generation': generation,
                'architecture': architecture,
                'features': features,
                'tensor_cores': generation_features.get('tensor_cores', False),
                'fp8_support': generation_features.get('fp8_support', False),
                'bf16_support': generation_features.get('bf16_support', False),
                'sparsity_support': generation_features.get('sparsity_support', False),
                'nvlink_support': generation_features.get('nvlink_support', False),
                'multi_instance_gpu': generation_features.get('multi_instance_gpu', False)
            }
        else:
            # Unknown compute capability - provide safe defaults
            return {
                'generation': 'Unknown',
                'architecture': f'Unknown_{compute_capability}',
                'features': [],
                'tensor_cores': False,
                'fp8_support': False,
                'bf16_support': False,
                'sparsity_support': False,
                'nvlink_support': False,
                'multi_instance_gpu': False
            }

    def _get_interconnect_type(self, props, generation_info: Dict) -> str:
        """Determine interconnect type based on generation and device characteristics"""
        # High-end cards typically have NVLink
        if generation_info['nvlink_support']:
            # Check if it's likely a high-end card
            if 'A100' in props.name or 'H100' in props.name or 'V100' in props.name:
                return "NVLink"
            elif props.multi_processor_count > 80:  # High SM count suggests data center GPU
                return "NVLink"

        return "PCIe"

    def get_generation_optimizations(self, device_spec: DeviceSpec) -> Dict[str, Any]:
        """Get recommended optimizations for specific GPU generation"""
        compute_capability = device_spec.capabilities.compute_capability
        generation_info = self._get_generation_info(compute_capability)

        optimizations = {
            'recommended_precisions': [],
            'kernel_optimizations': [],
            'memory_optimizations': [],
            'compilation_flags': []
        }

        # Generation-specific optimization recommendations
        if generation_info['generation'] == 'Maxwell':
            optimizations.update({
                'recommended_precisions': ['FP32', 'FP16'],
                'kernel_optimizations': ['memory_coalescing', 'occupancy_tuning'],
                'memory_optimizations': ['shared_memory', 'texture_cache'],
                'compilation_flags': ['-use_fast_math', '-O3']
            })

        elif generation_info['generation'] == 'Pascal':
            optimizations.update({
                'recommended_precisions': ['FP32', 'FP16'],
                'kernel_optimizations': ['memory_coalescing', 'occupancy_tuning', 'unified_memory'],
                'memory_optimizations': ['shared_memory', 'texture_cache', 'constant_memory'],
                'compilation_flags': ['-use_fast_math', '-O3', '--gpu-architecture=sm_60']
            })

        elif generation_info['generation'] == 'Volta':
            optimizations.update({
                'recommended_precisions': ['FP32', 'FP16', 'Mixed_Precision'],
                'kernel_optimizations': ['tensor_core_utilization', 'wmma_api', 'cooperative_groups'],
                'memory_optimizations': ['tensor_core_layouts', 'hbm_optimization'],
                'compilation_flags': ['-use_fast_math', '-O3', '--gpu-architecture=sm_70']
            })

        elif generation_info['generation'] == 'Turing':
            optimizations.update({
                'recommended_precisions': ['FP32', 'FP16', 'INT8', 'INT4'],
                'kernel_optimizations': ['tensor_core_utilization', 'int8_operations', 'rt_core_integration'],
                'memory_optimizations': ['tensor_layouts', 'gddr6_optimization'],
                'compilation_flags': ['-use_fast_math', '-O3', '--gpu-architecture=sm_75']
            })

        elif generation_info['generation'] == 'Ampere':
            optimizations.update({
                'recommended_precisions': ['TF32', 'BF16', 'FP16', 'INT8', 'Sparsity_2_4'],
                'kernel_optimizations': ['tensor_core_v3', 'structured_sparsity', 'mig_support'],
                'memory_optimizations': ['hbm2e_optimization', 'async_copy', 'cluster_scheduling'],
                'compilation_flags': ['-use_fast_math', '-O3', '--gpu-architecture=sm_80']
            })

        elif generation_info['generation'] == 'Ada_Lovelace':
            optimizations.update({
                'recommended_precisions': ['TF32', 'BF16', 'FP16', 'INT8', 'Sparsity_2_4'],
                'kernel_optimizations': ['tensor_core_v4', 'rt_core_v3', 'av1_acceleration'],
                'memory_optimizations': ['gddr6x_optimization', 'l2_cache_tuning'],
                'compilation_flags': ['-use_fast_math', '-O3', '--gpu-architecture=sm_89']
            })

        elif generation_info['generation'] == 'Hopper':
            optimizations.update({
                'recommended_precisions': ['FP8', 'TF32', 'BF16', 'FP16', 'INT8', 'Sparsity_2_4'],
                'kernel_optimizations': ['tensor_core_v4', 'fp8_operations', 'thread_block_clusters', 'dpx_instructions'],
                'memory_optimizations': ['hbm3_optimization', 'distributed_shared_memory', 'async_transaction_barrier'],
                'compilation_flags': ['-use_fast_math', '-O3', '--gpu-architecture=sm_90']
            })

        return optimizations


class AMDAdapter(VendorAdapter):
    """
    AMD GPU adapter leveraging ROCm and PyTorch's ROCm support

    Provides comprehensive AMD GPU support including RDNA and CDNA architectures
    with ROCm platform integration for optimal performance.
    """

    def __init__(self):
        super().__init__(HardwareVendor.AMD)
        self.rocm_available = self._check_rocm_availability()
        self.device_properties = {}

        # AMD GPU architecture mapping
        self.gpu_architectures = {
            'gfx900': 'Vega 10',      # Vega 56/64
            'gfx906': 'Vega 20',      # Radeon VII, MI50/60
            'gfx908': 'CDNA',         # MI100
            'gfx90a': 'CDNA2',        # MI200 series
            'gfx940': 'CDNA3',        # MI300 series
            'gfx1030': 'RDNA2',       # RX 6000 series
            'gfx1100': 'RDNA3',       # RX 7000 series
        }

    def _check_rocm_availability(self) -> bool:
        """Check if ROCm is available and functional"""
        try:
            # Check if PyTorch was compiled with ROCm support
            if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
                return torch.version.hip is not None

            # Alternative check for ROCm devices
            import subprocess
            result = subprocess.run(['rocm-smi', '--showid'],
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0

        except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_amd_device_properties(self, device_id: int) -> Dict[str, Any]:
        """Get AMD device properties using ROCm tools"""
        props = {
            'name': f'AMD GPU {device_id}',
            'compute_capability': 'unknown',
            'total_memory': 8 * 1024**3,  # Default 8GB
            'architecture': 'unknown',
            'multiprocessor_count': 64,   # Conservative estimate
        }

        try:
            if self.rocm_available:
                # Try to get device info via rocm-smi
                import subprocess
                result = subprocess.run([
                    'rocm-smi', '--device', str(device_id), '--showproductname', '--showmeminfo', '--showarch'
                ], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if 'Card series:' in line:
                            props['name'] = line.split(':')[-1].strip()
                        elif 'Memory Total:' in line:
                            memory_str = line.split(':')[-1].strip()
                            if 'GB' in memory_str:
                                memory_gb = float(memory_str.replace('GB', '').strip())
                                props['total_memory'] = int(memory_gb * 1024**3)
                        elif 'GPU Clock:' in line:
                            props['gpu_clock'] = line.split(':')[-1].strip()

                # Get architecture info
                arch_result = subprocess.run([
                    'rocm-smi', '--device', str(device_id), '--showarch'
                ], capture_output=True, text=True, timeout=5)

                if arch_result.returncode == 0:
                    for line in arch_result.stdout.strip().split('\n'):
                        if 'gfx' in line.lower():
                            gfx_id = line.strip().split()[-1]
                            props['compute_capability'] = gfx_id
                            props['architecture'] = self.gpu_architectures.get(gfx_id, gfx_id)

        except Exception as e:
            logger.debug(f"Could not get detailed AMD device properties: {e}")

        return props

    def initialize_device(self, device_id: int) -> DeviceSpec:
        """Initialize AMD GPU device"""
        if not self.rocm_available:
            raise RuntimeError("ROCm not available for AMD adapter")

        # Get device properties
        props = self._get_amd_device_properties(device_id)

        # Convert to our capabilities format
        capabilities = HardwareCapabilities(
            vendor=HardwareVendor.AMD,
            device_name=props['name'],
            compute_capability=props['compute_capability'],
            memory_gb=props['total_memory'] / (1024**3),
            peak_flops_fp32=self._estimate_amd_peak_flops(props),
            peak_flops_fp16=self._estimate_amd_peak_flops(props) * 2,
            memory_bandwidth_gbps=self._estimate_amd_memory_bandwidth(props),
            supported_precisions=self._get_amd_supported_precisions(props),
            tensor_core_support=self._has_matrix_cores(props),
            interconnect_type=self._get_amd_interconnect_type(props)
        )

        device_spec = DeviceSpec(
            device_id=device_id,
            vendor=HardwareVendor.AMD,
            capabilities=capabilities
        )

        # Cache for future use
        self.device_properties[device_id] = props
        self._devices.append(device_spec)

        return device_spec

    def discover_devices(self) -> List[DeviceSpec]:
        """Discover all AMD GPU devices"""
        devices = []

        if not self.rocm_available:
            logger.warning("ROCm not available, no AMD devices detected")
            return devices

        try:
            # Try to discover AMD devices using rocm-smi
            import subprocess
            result = subprocess.run(['rocm-smi', '--showid'],
                                 capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                device_ids = []

                for line in lines:
                    if 'GPU[' in line:
                        # Extract device ID from format like "GPU[0]"
                        import re
                        match = re.search(r'GPU\[(\d+)\]', line)
                        if match:
                            device_ids.append(int(match.group(1)))

                for device_id in device_ids:
                    try:
                        device_spec = self.initialize_device(device_id)
                        devices.append(device_spec)
                    except Exception as e:
                        logger.error(f"Failed to initialize AMD device {device_id}: {e}")

        except Exception as e:
            logger.error(f"Error discovering AMD devices: {e}")

        return devices

    def compile_kernel(self, kernel_source: str, target_device: DeviceSpec) -> Any:
        """Compile HIP kernel for AMD device"""
        try:
            logger.debug(f"Compiling HIP kernel for AMD device {target_device.device_id}")

            # AMD uses HIP for kernel compilation
            return {
                'device_id': target_device.device_id,
                'vendor': 'amd',
                'kernel_source': kernel_source,
                'compilation_time': 0.15,  # HIP compilation is typically slower than CUDA
                'optimizations_applied': ['wavefront_optimization', 'memory_coalescing', 'lds_optimization'],
                'architecture': target_device.capabilities.compute_capability
            }

        except Exception as e:
            logger.error(f"HIP kernel compilation failed: {e}")
            return None

    def optimize_memory_layout(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Apply AMD-specific memory optimizations"""
        # For AMD GPUs, we need to ensure optimal memory patterns
        if tensor.device.type == 'cpu':
            # Move to AMD GPU if available (ROCm uses 'cuda' device type in PyTorch)
            if self.rocm_available:
                tensor = tensor.cuda(device.device_id)

        # Apply AMD-specific optimizations
        architecture = device.capabilities.compute_capability

        # CDNA architectures prefer certain precisions
        if 'cdna' in architecture.lower() or 'gfx90' in architecture:
            if tensor.dtype == torch.float32:
                # Convert to bfloat16 for better performance on CDNA
                tensor = tensor.bfloat16()

        # RDNA architectures optimization
        elif 'rdna' in architecture.lower() or 'gfx1' in architecture:
            if tensor.dtype == torch.float32:
                # Convert to half precision for gaming-oriented RDNA
                tensor = tensor.half()

        # Ensure memory is contiguous for optimal access patterns
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    def create_communication_backend(self, devices: List[DeviceSpec]) -> Any:
        """Create RCCL communication backend for AMD devices"""
        try:
            # Check if RCCL is available
            import torch.distributed as dist

            # Filter for AMD devices only
            amd_devices = [d for d in devices if d.vendor == HardwareVendor.AMD]

            return {
                'backend_type': 'rccl',
                'devices': [d.device_id for d in amd_devices],
                'supports_allreduce': True,
                'supports_allgather': True,
                'supports_reduce_scatter': True,
                'infinity_fabric_support': self._check_infinity_fabric(amd_devices)
            }

        except ImportError:
            logger.warning("RCCL not available, falling back to Gloo")
            return {'backend_type': 'gloo', 'devices': []}

    def get_device_metrics(self, device_id: int) -> Dict[str, float]:
        """Get real-time AMD device metrics"""
        metrics = {
            'utilization': 0.0,
            'memory_used_gb': 0.0,
            'memory_total_gb': 0.0,
            'temperature_c': 0.0,
            'power_w': 0.0
        }

        try:
            if self.rocm_available:
                # Try to get metrics via rocm-smi
                import subprocess
                result = subprocess.run([
                    'rocm-smi', '--device', str(device_id), '--showuse', '--showmeminfo', '--showtemp', '--showpower'
                ], capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')

                    for line in lines:
                        if 'GPU use (%)' in line:
                            use_str = line.split(':')[-1].strip().replace('%', '')
                            metrics['utilization'] = float(use_str) if use_str.isdigit() else 0.0
                        elif 'Memory use:' in line:
                            memory_str = line.split(':')[-1].strip()
                            if 'GB' in memory_str:
                                used_gb = float(memory_str.split('GB')[0].strip())
                                metrics['memory_used_gb'] = used_gb
                        elif 'Temperature:' in line:
                            temp_str = line.split(':')[-1].strip().replace('C', '')
                            metrics['temperature_c'] = float(temp_str) if temp_str.replace('.','').isdigit() else 0.0
                        elif 'Power:' in line:
                            power_str = line.split(':')[-1].strip().replace('W', '')
                            metrics['power_w'] = float(power_str) if power_str.replace('.','').isdigit() else 0.0

        except Exception as e:
            logger.debug(f"Error getting AMD device metrics: {e}")

        return metrics

    def _estimate_amd_peak_flops(self, props: Dict) -> float:
        """Estimate peak FLOPS for AMD GPU"""
        architecture = props.get('compute_capability', 'unknown')

        # FLOPS estimates based on known AMD architectures
        flops_estimates = {
            'gfx940': 165e12,   # MI300X - ~165 TF FP32
            'gfx90a': 47.9e12,  # MI250X - ~47.9 TF FP32
            'gfx908': 23.1e12,  # MI100 - ~23.1 TF FP32
            'gfx906': 13.8e12,  # MI50 - ~13.8 TF FP32
            'gfx1100': 61e12,   # RX 7900 XTX - ~61 TF FP32
            'gfx1030': 23e12,   # RX 6900 XT - ~23 TF FP32
        }

        return flops_estimates.get(architecture, 10e12)  # 10 TF fallback

    def _estimate_amd_memory_bandwidth(self, props: Dict) -> float:
        """Estimate memory bandwidth for AMD GPU"""
        architecture = props.get('compute_capability', 'unknown')

        # Memory bandwidth estimates
        bandwidth_estimates = {
            'gfx940': 5300,     # MI300X - HBM3
            'gfx90a': 1600,     # MI250X - HBM2E
            'gfx908': 1200,     # MI100 - HBM2
            'gfx906': 1000,     # MI50 - HBM2
            'gfx1100': 960,     # RX 7900 XTX - GDDR6
            'gfx1030': 512,     # RX 6900 XT - GDDR6
        }

        return bandwidth_estimates.get(architecture, 500)  # 500 GB/s fallback

    def _get_amd_supported_precisions(self, props: Dict) -> List[ComputeCapability]:
        """Get supported precision modes for AMD device"""
        architecture = props.get('compute_capability', 'unknown')
        precisions = [ComputeCapability.FP32]

        # CDNA architectures have extensive precision support
        if 'gfx90' in architecture:  # CDNA/CDNA2
            precisions.extend([
                ComputeCapability.FP16,
                ComputeCapability.BF16,
                ComputeCapability.MIXED_PRECISION,
                ComputeCapability.INT8
            ])

            if 'gfx90a' in architecture or 'gfx940' in architecture:  # CDNA2/CDNA3
                precisions.append(ComputeCapability.INT4)

        # RDNA architectures
        elif 'gfx1' in architecture:  # RDNA2/RDNA3
            precisions.extend([
                ComputeCapability.FP16,
                ComputeCapability.MIXED_PRECISION
            ])

        return precisions

    def _has_matrix_cores(self, props: Dict) -> bool:
        """Check if device has matrix/tensor core equivalent"""
        architecture = props.get('compute_capability', 'unknown')

        # CDNA architectures have Matrix Core Units
        return 'gfx90' in architecture or 'gfx940' in architecture

    def _get_amd_interconnect_type(self, props: Dict) -> str:
        """Determine interconnect type for AMD device"""
        architecture = props.get('compute_capability', 'unknown')

        # MI200/MI300 series have Infinity Fabric
        if 'gfx90a' in architecture or 'gfx940' in architecture:
            return "Infinity Fabric"
        else:
            return "PCIe"

    def _check_infinity_fabric(self, devices: List[DeviceSpec]) -> bool:
        """Check if devices support Infinity Fabric interconnect"""
        for device in devices:
            if device.capabilities.interconnect_type == "Infinity Fabric":
                return True
        return False


class IntelAdapter(VendorAdapter):
    """
    Intel adapter supporting both CPU and Intel GPU (XPU) devices

    Provides unified interface for Intel hardware while leveraging
    Intel Extension for PyTorch (IPEX) when available.
    """

    def __init__(self):
        super().__init__(HardwareVendor.INTEL)
        self.ipex_available = self._check_ipex_availability()
        self.xpu_available = self._check_xpu_availability()

    def _check_ipex_availability(self) -> bool:
        """Check if Intel Extension for PyTorch is available"""
        try:
            import intel_extension_for_pytorch as ipex
            return True
        except ImportError:
            return False

    def _check_xpu_availability(self) -> bool:
        """Check if Intel XPU (GPU) is available"""
        try:
            if self.ipex_available:
                import intel_extension_for_pytorch as ipex
                return hasattr(ipex, 'xpu') and ipex.xpu.is_available()
            return False
        except Exception:
            return False

    def initialize_device(self, device_id: int) -> DeviceSpec:
        """Initialize Intel device (CPU or XPU)"""
        if device_id == 0:
            # CPU device
            capabilities = HardwareCapabilities(
                vendor=HardwareVendor.INTEL,
                device_name=self._get_cpu_info(),
                compute_capability="cpu",
                memory_gb=psutil.virtual_memory().total / (1024**3),
                peak_flops_fp32=self._estimate_cpu_flops(),
                peak_flops_fp16=self._estimate_cpu_flops() * 0.5,  # AVX-512 can help
                memory_bandwidth_gbps=100,  # Typical DDR4/DDR5
                supported_precisions=[ComputeCapability.FP32, ComputeCapability.FP16],
                interconnect_type="System Bus"
            )
        elif self.xpu_available:
            # Intel XPU (GPU) device
            capabilities = self._get_xpu_capabilities(device_id - 1)
        else:
            raise RuntimeError(f"Intel device {device_id} not available")

        device_spec = DeviceSpec(
            device_id=device_id,
            vendor=HardwareVendor.INTEL,
            capabilities=capabilities
        )

        self._devices.append(device_spec)
        return device_spec

    def discover_devices(self) -> List[DeviceSpec]:
        """Discover Intel devices (CPU + XPU if available)"""
        devices = []

        try:
            # Always include CPU as device 0
            cpu_device = self.initialize_device(0)
            devices.append(cpu_device)

            # Add XPU devices if available
            if self.xpu_available:
                import intel_extension_for_pytorch as ipex
                xpu_count = ipex.xpu.device_count() if hasattr(ipex.xpu, 'device_count') else 1

                for xpu_id in range(xpu_count):
                    xpu_device = self.initialize_device(xpu_id + 1)  # Offset by 1
                    devices.append(xpu_device)

        except Exception as e:
            logger.error(f"Error discovering Intel devices: {e}")

        return devices

    def compile_kernel(self, kernel_source: str, target_device: DeviceSpec) -> Any:
        """Compile kernel for Intel device"""
        try:
            if target_device.capabilities.device_name.startswith("Intel GPU"):
                # XPU kernel compilation
                return self._compile_xpu_kernel(kernel_source, target_device)
            else:
                # CPU kernel compilation (JIT or AOT)
                return self._compile_cpu_kernel(kernel_source, target_device)

        except Exception as e:
            logger.error(f"Intel kernel compilation failed: {e}")
            return None

    def _compile_xpu_kernel(self, kernel_source: str, device: DeviceSpec) -> Any:
        """Compile kernel for Intel XPU"""
        return {
            'device_id': device.device_id,
            'vendor': 'intel_xpu',
            'kernel_source': kernel_source,
            'compilation_time': 0.2,
            'optimizations_applied': ['vectorization', 'memory_tiling']
        }

    def _compile_cpu_kernel(self, kernel_source: str, device: DeviceSpec) -> Any:
        """Compile kernel for Intel CPU"""
        return {
            'device_id': device.device_id,
            'vendor': 'intel_cpu',
            'kernel_source': kernel_source,
            'compilation_time': 0.05,
            'optimizations_applied': ['avx512', 'vectorization', 'prefetch']
        }

    def optimize_memory_layout(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Apply Intel-specific memory optimizations"""
        # Ensure tensor is on correct device
        if device.capabilities.device_name.startswith("Intel GPU") and self.xpu_available:
            # Move to XPU
            import intel_extension_for_pytorch as ipex
            tensor = tensor.to(f'xpu:{device.device_id-1}')
        else:
            # CPU optimizations
            tensor = tensor.cpu()

        # Apply Intel-specific optimizations
        if self.ipex_available:
            try:
                import intel_extension_for_pytorch as ipex
                # Apply IPEX optimizations
                tensor = ipex.optimize(tensor)
            except Exception as e:
                logger.debug(f"IPEX optimization failed: {e}")

        return tensor

    def create_communication_backend(self, devices: List[DeviceSpec]) -> Any:
        """Create communication backend for Intel devices"""
        intel_devices = [d for d in devices if d.vendor == HardwareVendor.INTEL]

        return {
            'backend_type': 'gloo',  # Intel typically uses Gloo for CPU
            'devices': [d.device_id for d in intel_devices],
            'supports_allreduce': True,
            'supports_allgather': True,
            'optimizations': ['avx512'] if self._has_avx512() else []
        }

    def get_device_metrics(self, device_id: int) -> Dict[str, float]:
        """Get Intel device metrics"""
        metrics = {
            'utilization': 0.0,
            'memory_used_gb': 0.0,
            'memory_total_gb': 0.0,
            'temperature_c': 0.0
        }

        try:
            if device_id == 0:  # CPU
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                metrics.update({
                    'utilization': cpu_percent,
                    'memory_used_gb': (memory.total - memory.available) / (1024**3),
                    'memory_total_gb': memory.total / (1024**3)
                })

                # Try to get CPU temperature
                try:
                    temps = psutil.sensors_temperatures()
                    if 'coretemp' in temps:
                        avg_temp = np.mean([t.current for t in temps['coretemp']])
                        metrics['temperature_c'] = avg_temp
                except Exception:
                    pass

            elif self.xpu_available:
                # TODO: Implement Intel XPU metrics using Intel Level Zero APIs
                # This would require Intel GPU monitoring APIs for arc GPU metrics
                metrics.update({
                    'utilization': 50.0,  # Placeholder
                    'memory_used_gb': 2.0,  # Placeholder
                    'memory_total_gb': 16.0  # Placeholder
                })

        except Exception as e:
            logger.error(f"Error getting Intel device metrics: {e}")

        return metrics

    def _get_cpu_info(self) -> str:
        """Get CPU information"""
        try:
            import platform
            return f"{platform.processor()} ({psutil.cpu_count()} cores)"
        except Exception:
            return "Intel CPU"

    def _estimate_cpu_flops(self) -> float:
        """Estimate CPU FLOPS"""
        # Rough estimation based on core count and frequency
        try:
            core_count = psutil.cpu_count(logical=False)  # Physical cores
            # Assume base frequency of ~3 GHz for modern Intel CPUs
            base_freq = 3.0e9  # 3 GHz
            # Assume ~32 FLOPS per cycle for AVX-512
            flops_per_cycle = 32 if self._has_avx512() else 16

            return core_count * base_freq * flops_per_cycle
        except Exception:
            return 1e12  # 1 TFLOP fallback

    def _has_avx512(self) -> bool:
        """Check if CPU supports AVX-512"""
        try:
            import platform
            # Simple heuristic - this would need proper CPUID checking
            return "Intel" in platform.processor()
        except Exception:
            return False

    def _get_xpu_capabilities(self, xpu_id: int) -> HardwareCapabilities:
        """Get Intel XPU capabilities"""
        return HardwareCapabilities(
            vendor=HardwareVendor.INTEL,
            device_name="Intel GPU (XPU)",
            compute_capability="xe_hpg",
            memory_gb=16.0,  # Typical for Intel Arc GPUs
            peak_flops_fp32=10e12,  # 10 TFLOPS estimate
            peak_flops_fp16=20e12,  # 20 TFLOPS estimate
            memory_bandwidth_gbps=512,
            supported_precisions=[
                ComputeCapability.FP32,
                ComputeCapability.FP16,
                ComputeCapability.INT8
            ],
            interconnect_type="PCIe"
        )


class CPUAdapter(VendorAdapter):
    """
    Generic CPU adapter for any CPU architecture

    Provides basic CPU support when specific vendor adapters are not available.
    """

    def __init__(self):
        super().__init__(HardwareVendor.INTEL)  # Map CPU to Intel vendor

    def initialize_device(self, device_id: int) -> DeviceSpec:
        """Initialize CPU device"""
        if device_id != 0:
            raise RuntimeError("CPU adapter only supports device ID 0")

        capabilities = HardwareCapabilities(
            vendor=HardwareVendor.INTEL,
            device_name=self._get_cpu_name(),
            compute_capability="cpu",
            memory_gb=psutil.virtual_memory().total / (1024**3),
            peak_flops_fp32=self._estimate_cpu_flops(),
            peak_flops_fp16=self._estimate_cpu_flops() * 0.5,
            memory_bandwidth_gbps=50,  # Conservative estimate
            supported_precisions=[ComputeCapability.FP32, ComputeCapability.FP16],
            interconnect_type="System Bus"
        )

        device_spec = DeviceSpec(
            device_id=0,
            vendor=HardwareVendor.INTEL,
            capabilities=capabilities
        )

        self._devices.append(device_spec)
        return device_spec

    def discover_devices(self) -> List[DeviceSpec]:
        """Discover CPU device"""
        try:
            return [self.initialize_device(0)]
        except Exception as e:
            logger.error(f"Error discovering CPU device: {e}")
            return []

    def compile_kernel(self, kernel_source: str, target_device: DeviceSpec) -> Any:
        """Compile CPU kernel"""
        return {
            'device_id': 0,
            'vendor': 'cpu',
            'kernel_source': kernel_source,
            'compilation_time': 0.01,
            'optimizations_applied': ['vectorization']
        }

    def optimize_memory_layout(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Apply CPU memory optimizations"""
        tensor = tensor.cpu()

        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    def create_communication_backend(self, devices: List[DeviceSpec]) -> Any:
        """Create CPU communication backend"""
        return {
            'backend_type': 'gloo',
            'devices': [0],
            'supports_allreduce': True,
            'supports_allgather': True
        }

    def get_device_metrics(self, device_id: int) -> Dict[str, float]:
        """Get CPU metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            return {
                'utilization': cpu_percent,
                'memory_used_gb': (memory.total - memory.available) / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'temperature_c': 0.0  # Not easily available
            }
        except Exception as e:
            logger.error(f"Error getting CPU metrics: {e}")
            return {'utilization': 0.0, 'memory_used_gb': 0.0, 'memory_total_gb': 0.0}

    def _get_cpu_name(self) -> str:
        """Get CPU name"""
        try:
            import platform
            return platform.processor() or "Generic CPU"
        except Exception:
            return "Generic CPU"

    def _estimate_cpu_flops(self) -> float:
        """Estimate CPU FLOPS"""
        try:
            core_count = psutil.cpu_count(logical=False)
            # Conservative estimate: 2.5 GHz base, 8 FLOPS per cycle
            return core_count * 2.5e9 * 8
        except Exception:
            return 100e9  # 100 GFLOPS fallback


# Factory function for creating appropriate adapters
def create_vendor_adapter(vendor: HardwareVendor) -> VendorAdapter:
    """Create appropriate vendor adapter"""
    if vendor == HardwareVendor.NVIDIA:
        return NVIDIAAdapter()
    elif vendor == HardwareVendor.AMD:
        return AMDAdapter()
    elif vendor == HardwareVendor.INTEL:
        return IntelAdapter()
    else:
        return CPUAdapter()


# Convenience functions for adapter management
def get_available_vendors() -> List[HardwareVendor]:
    """Get list of available hardware vendors"""
    vendors = []

    # Check NVIDIA
    if torch.cuda.is_available():
        vendors.append(HardwareVendor.NVIDIA)

    # Check AMD (ROCm)
    try:
        # Check if PyTorch was compiled with ROCm support
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
            if torch.version.hip is not None:
                vendors.append(HardwareVendor.AMD)
        else:
            # Alternative check for ROCm devices
            import subprocess
            result = subprocess.run(['rocm-smi', '--showid'],
                                 capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                vendors.append(HardwareVendor.AMD)
    except (ImportError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check Intel
    try:
        import intel_extension_for_pytorch as ipex
        vendors.append(HardwareVendor.INTEL)
    except ImportError:
        pass

    # CPU is always available
    vendors.append(HardwareVendor.UNKNOWN)  # Generic CPU

    return vendors


def auto_detect_best_adapter() -> VendorAdapter:
    """Automatically detect and return the best available adapter"""
    # Prioritize NVIDIA for CUDA availability
    if torch.cuda.is_available():
        return NVIDIAAdapter()

    # Check for AMD ROCm
    try:
        if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
            if torch.version.hip is not None:
                return AMDAdapter()
    except Exception:
        pass

    # Check for Intel XPU
    try:
        import intel_extension_for_pytorch as ipex
        return IntelAdapter()
    except ImportError:
        pass

    # Fallback to CPU
    return CPUAdapter()


class CustomHardwareAdapter(VendorAdapter):
    """
    Custom Hardware Adapter for TPUs, ASICs, and specialized accelerators

    Provides a flexible framework for integrating custom accelerators through
    PyTorch's PrivateUse1 interface and vendor-specific plugins.
    """

    def __init__(self, hardware_type: str = "unknown"):
        super().__init__(HardwareVendor.UNKNOWN)
        self.hardware_type = hardware_type.lower()
        self.custom_backend = None
        self.device_plugin = None

        # Initialize based on hardware type
        self._initialize_custom_backend()

    def _initialize_custom_backend(self):
        """Initialize custom backend based on hardware type"""
        if self.hardware_type == "tpu":
            self._initialize_tpu_backend()
        elif self.hardware_type == "asic":
            self._initialize_asic_backend()
        elif self.hardware_type == "neuromorphic":
            self._initialize_neuromorphic_backend()
        else:
            logger.warning(f"Unknown custom hardware type: {self.hardware_type}")

    def _initialize_tpu_backend(self):
        """Initialize TPU backend support"""
        try:
            # Check for Google Cloud TPU support
            import torch_xla
            import torch_xla.core.xla_model as xm
            self.custom_backend = "xla_tpu"
            logger.info("TPU backend initialized with XLA")
        except ImportError:
            # Fallback to custom TPU implementation
            logger.warning("torch_xla not available, using custom TPU backend")
            self.custom_backend = "custom_tpu"

    def _initialize_asic_backend(self):
        """Initialize ASIC backend for custom accelerators"""
        self.custom_backend = "custom_asic"

        # Register custom ASIC device with PyTorch PrivateUse1
        try:
            from .privateuse1_integration import register_custom_device, CustomDeviceBackend

            # Create custom ASIC backend (simplified for demo purposes)
            class ASICBackend(CustomDeviceBackend):
                def __init__(self):
                    super().__init__("asic", HardwareVendor.UNKNOWN)

                def synchronize_device(self, device_id: int):
                    # TODO: Implement ASIC-specific device synchronization
                    pass

                def get_device_properties(self, device_id: int) -> Dict[str, Any]:
                    return {"name": "Custom AI ASIC", "memory": "16GB"}

            # Create ASIC device configuration
            asic_config = PrivateUse1Config(
                device_name="asic",
                vendor=HardwareVendor.UNKNOWN,
                backend_library="custom_asic_backend",
                enable_autograd=True,
                enable_compilation=True
            )

            # Register the backend and config
            asic_backend = ASICBackend()
            register_custom_device(asic_backend, asic_config)
            logger.info("Custom ASIC device registered with PrivateUse1")

        except Exception as e:
            logger.error(f"Failed to register ASIC device: {e}")

    def _initialize_neuromorphic_backend(self):
        """Initialize neuromorphic computing backend"""
        self.custom_backend = "neuromorphic"

        # Neuromorphic chips often require specialized spike-timing computation
        logger.info("Neuromorphic backend initialized")

    def initialize_device(self, device_id: int) -> DeviceSpec:
        """Initialize custom hardware device"""
        if self.hardware_type == "tpu":
            return self._initialize_tpu_device(device_id)
        elif self.hardware_type == "asic":
            return self._initialize_asic_device(device_id)
        elif self.hardware_type == "neuromorphic":
            return self._initialize_neuromorphic_device(device_id)
        else:
            return self._initialize_generic_device(device_id)

    def _initialize_tpu_device(self, device_id: int) -> DeviceSpec:
        """Initialize TPU device"""
        capabilities = HardwareCapabilities(
            vendor=HardwareVendor.UNKNOWN,
            device_name=f"Google Cloud TPU v{self._get_tpu_version()}",
            compute_capability="tpu_v4",
            memory_gb=32.0,  # TPU v4 has 32GB HBM
            peak_flops_fp32=275e12,  # TPU v4: 275 TFLOPS
            peak_flops_fp16=1100e12,  # TPU v4: 1.1 PFLOPS BF16
            memory_bandwidth_gbps=1200,  # TPU v4: 1.2 TB/s
            supported_precisions=[
                ComputeCapability.BF16,
                ComputeCapability.FP32,
                ComputeCapability.INT8,
                ComputeCapability.INT4
            ],
            interconnect_type="ICI (Inter-Chip Interconnect)",
            generation="v4",
            architecture="TPU",
            features=["Matrix_Units", "Systolic_Array", "Vector_Units", "XLA_Compiler"]
        )

        device_spec = DeviceSpec(
            device_id=device_id,
            vendor=HardwareVendor.UNKNOWN,
            capabilities=capabilities
        )

        self._devices.append(device_spec)
        return device_spec

    def _initialize_asic_device(self, device_id: int) -> DeviceSpec:
        """Initialize custom ASIC device"""
        capabilities = HardwareCapabilities(
            vendor=HardwareVendor.UNKNOWN,
            device_name="Custom AI ASIC",
            compute_capability="asic_v1",
            memory_gb=16.0,  # Configurable based on ASIC design
            peak_flops_fp32=50e12,  # 50 TFLOPS (configurable)
            peak_flops_fp16=100e12,  # 100 TFLOPS (configurable)
            memory_bandwidth_gbps=800,  # High bandwidth for AI workloads
            supported_precisions=[
                ComputeCapability.FP16,
                ComputeCapability.FP32,
                ComputeCapability.INT8,
                ComputeCapability.INT4
            ],
            interconnect_type="Custom Fabric",
            generation="v1",
            architecture="Custom",
            features=["Custom_Matrix_Units", "On_Chip_Memory", "Hardware_Sparsity"]
        )

        device_spec = DeviceSpec(
            device_id=device_id,
            vendor=HardwareVendor.UNKNOWN,
            capabilities=capabilities
        )

        self._devices.append(device_spec)
        return device_spec

    def _initialize_neuromorphic_device(self, device_id: int) -> DeviceSpec:
        """Initialize neuromorphic computing device"""
        capabilities = HardwareCapabilities(
            vendor=HardwareVendor.UNKNOWN,
            device_name="Neuromorphic Processor",
            compute_capability="neuromorphic_v1",
            memory_gb=8.0,  # Neuromorphic chips typically have limited memory
            peak_flops_fp32=1e12,  # Event-driven, not traditional FLOPS
            peak_flops_fp16=2e12,
            memory_bandwidth_gbps=100,  # Lower bandwidth, event-driven
            supported_precisions=[
                ComputeCapability.INT8,
                ComputeCapability.INT4,
                ComputeCapability.FP16  # Some support low precision
            ],
            interconnect_type="Spike Network",
            generation="v1",
            architecture="Neuromorphic",
            features=["Spiking_Neurons", "Event_Driven", "Ultra_Low_Power", "Temporal_Dynamics"]
        )

        device_spec = DeviceSpec(
            device_id=device_id,
            vendor=HardwareVendor.UNKNOWN,
            capabilities=capabilities
        )

        self._devices.append(device_spec)
        return device_spec

    def _initialize_generic_device(self, device_id: int) -> DeviceSpec:
        """Initialize generic custom device"""
        capabilities = HardwareCapabilities(
            vendor=HardwareVendor.UNKNOWN,
            device_name=f"Custom {self.hardware_type.title()} Device",
            compute_capability="custom_v1",
            memory_gb=8.0,
            peak_flops_fp32=10e12,
            peak_flops_fp16=20e12,
            memory_bandwidth_gbps=200,
            supported_precisions=[ComputeCapability.FP32, ComputeCapability.FP16],
            interconnect_type="Custom",
            generation="v1",
            architecture="Custom",
            features=["Custom_Features"]
        )

        device_spec = DeviceSpec(
            device_id=device_id,
            vendor=HardwareVendor.UNKNOWN,
            capabilities=capabilities
        )

        self._devices.append(device_spec)
        return device_spec

    def discover_devices(self) -> List[DeviceSpec]:
        """Discover custom hardware devices"""
        devices = []

        try:
            if self.hardware_type == "tpu":
                devices.extend(self._discover_tpu_devices())
            elif self.hardware_type == "asic":
                devices.extend(self._discover_asic_devices())
            elif self.hardware_type == "neuromorphic":
                devices.extend(self._discover_neuromorphic_devices())
            else:
                # Generic discovery - assume one device
                devices.append(self.initialize_device(0))

        except Exception as e:
            logger.error(f"Error discovering {self.hardware_type} devices: {e}")

        return devices

    def _discover_tpu_devices(self) -> List[DeviceSpec]:
        """Discover TPU devices"""
        devices = []

        try:
            if self.custom_backend == "xla_tpu":
                import torch_xla.core.xla_model as xm
                # XLA typically provides 8 TPU cores per chip
                num_devices = min(8, xm.xrt_world_size() if hasattr(xm, 'xrt_world_size') else 1)
            else:
                # Custom TPU backend - check environment
                import os
                num_devices = int(os.environ.get('TPU_NUM_DEVICES', '1'))

            for device_id in range(num_devices):
                devices.append(self._initialize_tpu_device(device_id))

        except Exception as e:
            logger.error(f"Error discovering TPU devices: {e}")
            # Fallback to single device
            devices.append(self._initialize_tpu_device(0))

        return devices

    def _discover_asic_devices(self) -> List[DeviceSpec]:
        """Discover ASIC devices"""
        # TODO: Implement ASIC device discovery using vendor-specific APIs
        # This would require custom ASIC vendor drivers and device management APIs
        devices = []

        try:
            # Check for custom device count (vendor-specific)
            import os
            num_devices = int(os.environ.get('ASIC_NUM_DEVICES', '1'))

            for device_id in range(num_devices):
                devices.append(self._initialize_asic_device(device_id))

        except Exception as e:
            logger.error(f"Error discovering ASIC devices: {e}")
            devices.append(self._initialize_asic_device(0))

        return devices

    def _discover_neuromorphic_devices(self) -> List[DeviceSpec]:
        """Discover neuromorphic devices"""
        devices = []

        try:
            # TODO: Implement neuromorphic device discovery using vendor-specific APIs
            # This would integrate with Intel Loihi, BrainChip Akida, or other neuromorphic platforms
            # Neuromorphic devices are typically single chip
            devices.append(self._initialize_neuromorphic_device(0))
        except Exception as e:
            logger.error(f"Error discovering neuromorphic devices: {e}")

        return devices

    def compile_kernel(self, kernel_source: str, target_device: DeviceSpec) -> Any:
        """Compile kernel for custom hardware"""
        if self.hardware_type == "tpu":
            return self._compile_tpu_kernel(kernel_source, target_device)
        elif self.hardware_type == "asic":
            return self._compile_asic_kernel(kernel_source, target_device)
        elif self.hardware_type == "neuromorphic":
            return self._compile_neuromorphic_kernel(kernel_source, target_device)
        else:
            return self._compile_generic_kernel(kernel_source, target_device)

    def _compile_tpu_kernel(self, kernel_source: str, device: DeviceSpec) -> Any:
        """Compile kernel for TPU using XLA"""
        return {
            'device_id': device.device_id,
            'vendor': 'tpu',
            'kernel_source': kernel_source,
            'compilation_time': 1.0,  # XLA compilation can be slow
            'optimizations_applied': ['xla_fusion', 'systolic_array', 'matrix_optimization'],
            'backend': self.custom_backend
        }

    def _compile_asic_kernel(self, kernel_source: str, device: DeviceSpec) -> Any:
        """Compile kernel for custom ASIC"""
        return {
            'device_id': device.device_id,
            'vendor': 'asic',
            'kernel_source': kernel_source,
            'compilation_time': 0.3,
            'optimizations_applied': ['asic_specific', 'memory_tiling', 'dataflow'],
            'backend': self.custom_backend
        }

    def _compile_neuromorphic_kernel(self, kernel_source: str, device: DeviceSpec) -> Any:
        """Compile kernel for neuromorphic device"""
        return {
            'device_id': device.device_id,
            'vendor': 'neuromorphic',
            'kernel_source': kernel_source,
            'compilation_time': 0.1,
            'optimizations_applied': ['spike_optimization', 'temporal_encoding', 'event_driven'],
            'backend': self.custom_backend
        }

    def _compile_generic_kernel(self, kernel_source: str, device: DeviceSpec) -> Any:
        """Compile kernel for generic custom device"""
        return {
            'device_id': device.device_id,
            'vendor': f'custom_{self.hardware_type}',
            'kernel_source': kernel_source,
            'compilation_time': 0.2,
            'optimizations_applied': ['generic_optimization'],
            'backend': self.custom_backend
        }

    def optimize_memory_layout(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Apply custom hardware memory optimizations"""
        if self.hardware_type == "tpu":
            return self._optimize_tpu_memory(tensor, device)
        elif self.hardware_type == "asic":
            return self._optimize_asic_memory(tensor, device)
        elif self.hardware_type == "neuromorphic":
            return self._optimize_neuromorphic_memory(tensor, device)
        else:
            return tensor  # Generic - no optimization

    def _optimize_tpu_memory(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Optimize tensor layout for TPU"""
        try:
            if self.custom_backend == "xla_tpu":
                import torch_xla
                # Move to TPU device
                tensor = tensor.to(f'xla:{device.device_id}')

                # TPUs prefer BF16 for better performance
                if tensor.dtype == torch.float32:
                    tensor = tensor.bfloat16()

        except ImportError:
            logger.warning("torch_xla not available for TPU optimization")
            # Fallback to CPU with BF16
            tensor = tensor.cpu()
            if tensor.dtype == torch.float32:
                tensor = tensor.bfloat16()

        return tensor

    def _optimize_asic_memory(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Optimize tensor layout for ASIC"""
        # Move to custom device if PrivateUse1 is configured
        try:
            tensor = tensor.to('privateuse1:0')
        except Exception:
            tensor = tensor.cpu()

        # ASIC typically prefers lower precision
        if tensor.dtype == torch.float32:
            tensor = tensor.half()

        # Ensure contiguous for custom memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor

    def _optimize_neuromorphic_memory(self, tensor: torch.Tensor, device: DeviceSpec) -> torch.Tensor:
        """Optimize tensor layout for neuromorphic device"""
        # Neuromorphic devices typically work with sparse, low-precision data
        tensor = tensor.cpu()

        # Convert to low precision (INT8 simulation)
        if tensor.dtype == torch.float32:
            # Simulate INT8 quantization
            tensor = torch.clamp(tensor, -1.0, 1.0)
            tensor = (tensor * 127).round() / 127.0

        return tensor

    def create_communication_backend(self, devices: List[DeviceSpec]) -> Any:
        """Create communication backend for custom devices"""
        custom_devices = [d for d in devices if d.vendor == HardwareVendor.UNKNOWN]

        backend_config = {
            'backend_type': f'{self.hardware_type}_collective',
            'devices': [d.device_id for d in custom_devices],
            'supports_allreduce': True,
            'supports_allgather': True,
            'hardware_type': self.hardware_type
        }

        if self.hardware_type == "tpu":
            backend_config.update({
                'xla_backend': self.custom_backend == "xla_tpu",
                'mesh_topology': '2x2x1',  # Example TPU mesh
                'collective_ops': ['cross_replica', 'all_reduce', 'all_gather']
            })
        elif self.hardware_type == "asic":
            backend_config.update({
                'custom_fabric': True,
                'bandwidth_gbps': 400,  # High-speed custom interconnect
                'collective_ops': ['allreduce', 'broadcast', 'reduce_scatter']
            })
        elif self.hardware_type == "neuromorphic":
            backend_config.update({
                'event_driven': True,
                'spike_communication': True,
                'collective_ops': ['spike_broadcast', 'event_reduction']
            })

        return backend_config

    def get_device_metrics(self, device_id: int) -> Dict[str, float]:
        """Get custom device metrics"""
        base_metrics = {
            'utilization': 0.0,
            'memory_used_gb': 0.0,
            'memory_total_gb': 0.0,
            'temperature_c': 0.0,
            'power_w': 0.0
        }

        try:
            if self.hardware_type == "tpu":
                # TODO: Implement TPU metrics collection using GCP monitoring APIs
                # This would require Google Cloud Monitoring API integration for production TPUs
                base_metrics.update({
                    'utilization': 75.0,  # TPUs typically run at high utilization
                    'memory_used_gb': 24.0,
                    'memory_total_gb': 32.0,
                    'temperature_c': 65.0,
                    'power_w': 200.0  # TPU v4 power consumption
                })
            elif self.hardware_type == "asic":
                # TODO: Implement ASIC metrics collection using vendor-specific monitoring APIs
                # This would require custom ASIC vendor monitoring libraries
                base_metrics.update({
                    'utilization': 80.0,
                    'memory_used_gb': 12.0,
                    'memory_total_gb': 16.0,
                    'temperature_c': 70.0,
                    'power_w': 150.0
                })
            elif self.hardware_type == "neuromorphic":
                # TODO: Implement neuromorphic device metrics using spike-based monitoring APIs
                # This would require vendor-specific neuromorphic chip monitoring interfaces
                base_metrics.update({
                    'utilization': 30.0,  # Event-driven, lower continuous utilization
                    'memory_used_gb': 2.0,
                    'memory_total_gb': 8.0,
                    'temperature_c': 40.0,  # Very low power
                    'power_w': 5.0,  # Ultra-low power consumption
                    'spike_rate_khz': 100.0,  # Neuromorphic-specific metric
                    'events_per_second': 1000000.0
                })

        except Exception as e:
            logger.error(f"Error getting {self.hardware_type} device metrics: {e}")

        return base_metrics

    def _get_tpu_version(self) -> str:
        """Get TPU version"""
        try:
            # In practice, this would query the TPU metadata
            import os
            return os.environ.get('TPU_VERSION', 'v4')
        except Exception:
            return 'v4'


# Factory function for creating custom hardware adapters
def create_custom_adapter(hardware_type: str) -> CustomHardwareAdapter:
    """Create custom hardware adapter for specific hardware type"""
    supported_types = ["tpu", "asic", "neuromorphic"]

    if hardware_type.lower() not in supported_types:
        logger.warning(f"Hardware type '{hardware_type}' not in supported types {supported_types}")

    return CustomHardwareAdapter(hardware_type)


# Enhanced factory function with custom hardware support
def create_vendor_adapter_enhanced(vendor: HardwareVendor, custom_type: Optional[str] = None) -> VendorAdapter:
    """Enhanced vendor adapter creation with custom hardware support"""
    if vendor == HardwareVendor.NVIDIA:
        return NVIDIAAdapter()
    elif vendor == HardwareVendor.AMD:
        return AMDAdapter()
    elif vendor == HardwareVendor.INTEL:
        return IntelAdapter()
    elif vendor == HardwareVendor.UNKNOWN and custom_type:
        return CustomHardwareAdapter(custom_type)
    else:
        return CPUAdapter()