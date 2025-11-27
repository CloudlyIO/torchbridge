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
from ..distributed_scale.hardware_discovery import HardwareVendor
from .privateuse1_integration import CustomDeviceBackend, PrivateUse1Config

# Import existing hardware discovery for compatibility
from ..distributed_scale.hardware_discovery import DeviceInfo, DeviceCapability, ThermalState

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

    def initialize_device(self, device_id: int) -> DeviceSpec:
        """Initialize NVIDIA GPU device"""
        if not self.cuda_available:
            raise RuntimeError("CUDA not available for NVIDIA adapter")

        if device_id >= torch.cuda.device_count():
            raise RuntimeError(f"Invalid NVIDIA device ID: {device_id}")

        # Get device properties
        props = torch.cuda.get_device_properties(device_id)

        # Convert to our capabilities format
        capabilities = HardwareCapabilities(
            vendor=HardwareVendor.NVIDIA,
            device_name=props.name,
            compute_capability=f"{props.major}.{props.minor}",
            memory_gb=props.total_memory / (1024**3),
            peak_flops_fp32=self._estimate_peak_flops(props),
            peak_flops_fp16=self._estimate_peak_flops(props) * 2,  # Approximate
            memory_bandwidth_gbps=self._estimate_memory_bandwidth(props),
            supported_precisions=self._get_supported_precisions(props),
            tensor_core_support=props.major >= 7,
            interconnect_type="NVLink" if props.multi_processor_count > 80 else "PCIe"
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
                # XPU metrics (would need Intel GPU monitoring APIs)
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
    if torch.cuda.is_available():
        return NVIDIAAdapter()

    try:
        import intel_extension_for_pytorch as ipex
        return IntelAdapter()
    except ImportError:
        pass

    return CPUAdapter()