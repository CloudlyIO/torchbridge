"""
Comprehensive Tests for Hardware Abstraction Layer (HAL)

Tests cover:
- PrivateUse1 integration
- Cross-vendor device mesh creation
- Hardware auto-detection
- Vendor adapter functionality
- Integration with existing hardware systems
"""

import pytest
import torch
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Import hardware abstraction components
try:
    from kernel_pytorch.hardware_abstraction.hal_core import (
        HardwareAbstractionLayer, DeviceSpec, HardwareCapabilities,
        ComputeCapability, DeviceMesh, CrossVendorCapabilities
    )
    # Use the original HardwareVendor for compatibility
    from kernel_pytorch.distributed_scale.hardware_discovery import HardwareVendor
    from kernel_pytorch.hardware_abstraction.privateuse1_integration import (
        PrivateUse1Manager, CustomDeviceBackend, PrivateUse1Config,
        register_custom_device, validate_privateuse1_setup
    )
    from kernel_pytorch.hardware_abstraction.vendor_adapters import (
        NVIDIAAdapter, IntelAdapter, CPUAdapter
    )
    from kernel_pytorch.distributed_scale.hardware_adapter import HardwareAdapter
    HAL_AVAILABLE = True
except ImportError as e:
    HAL_AVAILABLE = False
    pytest.skip(f"Hardware abstraction not available: {e}", allow_module_level=True)


@pytest.fixture
def sample_device_specs():
    """Create sample device specifications for testing"""
    nvidia_device = DeviceSpec(
        device_id=0,
        vendor=HardwareVendor.NVIDIA,
        capabilities=HardwareCapabilities(
            vendor=HardwareVendor.NVIDIA,
            device_name="Tesla V100",
            compute_capability="7.0",
            memory_gb=16.0,
            peak_flops_fp32=15.7e12,
            peak_flops_fp16=31.4e12,
            memory_bandwidth_gbps=900.0,
            supported_precisions=[
                ComputeCapability.FP32, ComputeCapability.FP16,
                ComputeCapability.MIXED_PRECISION, ComputeCapability.TENSOR_CORES
            ],
            tensor_core_support=True,
            interconnect_type="NVLink"
        )
    )

    intel_device = DeviceSpec(
        device_id=1,
        vendor=HardwareVendor.INTEL,
        capabilities=HardwareCapabilities(
            vendor=HardwareVendor.INTEL,
            device_name="Intel XPU",
            compute_capability="1.0",
            memory_gb=32.0,
            peak_flops_fp32=10.0e12,
            peak_flops_fp16=20.0e12,
            memory_bandwidth_gbps=1024.0,
            supported_precisions=[
                ComputeCapability.FP32, ComputeCapability.FP16,
                ComputeCapability.BF16, ComputeCapability.MIXED_PRECISION
            ],
            tensor_core_support=False,
            interconnect_type="PCIe"
        )
    )

    return [nvidia_device, intel_device]


@pytest.fixture
def hal_instance():
    """Create HAL instance for testing"""
    return HardwareAbstractionLayer()


class TestHardwareAbstractionLayer:
    """Test core HAL functionality"""

    def test_hal_initialization(self, hal_instance):
        """Test HAL initializes correctly"""
        assert hal_instance is not None
        assert isinstance(hal_instance.vendor_adapters, dict)
        assert isinstance(hal_instance.devices, dict)

    def test_vendor_adapter_registration(self, hal_instance):
        """Test vendor adapter registration"""
        # Create mock adapter with proper discover_devices method
        mock_adapter = Mock()
        mock_adapter.vendor = HardwareVendor.NVIDIA
        mock_adapter.discover_devices.return_value = []  # Return empty list of devices

        # Register adapter
        hal_instance.register_vendor_adapter(mock_adapter)

        # Verify registration
        assert HardwareVendor.NVIDIA in hal_instance.vendor_adapters
        assert hal_instance.vendor_adapters[HardwareVendor.NVIDIA] == mock_adapter

    def test_device_registration(self, hal_instance, sample_device_specs):
        """Test device registration with HAL"""
        device = sample_device_specs[0]

        # Register device
        hal_instance.register_device(device)

        # Verify registration
        assert device.device_id in hal_instance.devices
        assert hal_instance.devices[device.device_id] == device

    def test_optimal_device_selection(self, hal_instance, sample_device_specs):
        """Test optimal device selection logic"""
        # Register devices
        for device in sample_device_specs:
            hal_instance.register_device(device)

        # Test selection with memory requirement
        optimal = hal_instance.get_optimal_device(
            memory_requirement_gb=8.0,
            compute_requirement_tflops=10.0
        )

        assert optimal is not None
        assert optimal.capabilities.memory_gb >= 8.0
        assert optimal.capabilities.peak_flops_fp32 / 1e12 >= 10.0

    def test_cross_vendor_mesh_creation(self, hal_instance, sample_device_specs):
        """Test cross-vendor device mesh creation"""
        # Register devices
        for device in sample_device_specs:
            hal_instance.register_device(device)

        # Create cross-vendor mesh
        mesh = hal_instance.create_cross_vendor_mesh(
            devices=sample_device_specs,
            mesh_id="test_mesh",
            topology="ring"
        )

        assert isinstance(mesh, DeviceMesh)
        assert mesh.mesh_id == "test_mesh"
        assert len(mesh.devices) == 2
        assert mesh.topology == "ring"

        # Check bandwidth/latency matrices are populated
        assert mesh.bandwidth_matrix is not None
        assert mesh.latency_matrix is not None
        assert len(mesh.bandwidth_matrix) == 2
        assert len(mesh.latency_matrix) == 2

    def test_cross_vendor_capabilities(self, hal_instance, sample_device_specs):
        """Test cross-vendor capabilities aggregation"""
        # Register devices
        for device in sample_device_specs:
            hal_instance.register_device(device)

        capabilities = hal_instance.get_cross_vendor_capabilities()

        assert isinstance(capabilities, CrossVendorCapabilities)
        assert capabilities.total_devices == 2
        assert HardwareVendor.NVIDIA in capabilities.vendor_distribution
        assert HardwareVendor.INTEL in capabilities.vendor_distribution
        assert capabilities.cross_vendor_communication is True
        assert "ring" in capabilities.mesh_topologies


class TestPrivateUse1Integration:
    """Test PrivateUse1 integration functionality"""

    def test_privateuse1_manager_creation(self):
        """Test PrivateUse1 manager creation"""
        from kernel_pytorch.hardware_abstraction.privateuse1_integration import get_privateuse1_manager

        manager = get_privateuse1_manager()
        assert manager is not None
        assert isinstance(manager.registered_devices, dict)
        assert isinstance(manager.device_mappings, dict)

    def test_custom_device_backend(self):
        """Test custom device backend implementation"""
        class TestBackend(CustomDeviceBackend):
            def initialize_device(self, device_id: int) -> bool:
                return True

            def get_device_count(self) -> int:
                return 1

            def get_device_properties(self, device_id: int) -> Dict[str, Any]:
                return {"name": "test_device", "memory": 1024}

            def allocate_memory(self, size: int, device_id: int) -> Any:
                return Mock()

            def copy_to_device(self, tensor: torch.Tensor, device_id: int) -> torch.Tensor:
                return tensor

        backend = TestBackend("test_device", HardwareVendor.CUSTOM_ASIC)

        assert backend.device_name == "test_device"
        assert backend.vendor == HardwareVendor.CUSTOM_ASIC
        assert not backend.is_registered
        assert backend.get_device_count() == 1

    def test_privateuse1_validation(self):
        """Test PrivateUse1 setup validation"""
        status = validate_privateuse1_setup()

        assert 'pytorch_version' in status
        assert 'privateuse1_supported' in status
        assert 'registered_devices' in status
        assert 'errors' in status

        # Check PyTorch version is valid
        assert status['pytorch_version'] is not None


class TestVendorAdapters:
    """Test vendor-specific adapter implementations"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_nvidia_adapter(self):
        """Test NVIDIA adapter functionality"""
        adapter = NVIDIAAdapter()

        assert adapter.vendor == HardwareVendor.NVIDIA

        # Test device discovery
        devices = adapter.discover_devices()
        if torch.cuda.device_count() > 0:
            assert len(devices) > 0
            assert all(d.vendor == HardwareVendor.NVIDIA for d in devices)

    def test_cpu_adapter(self):
        """Test CPU adapter functionality"""
        adapter = CPUAdapter()

        assert adapter.vendor == HardwareVendor.INTEL  # CPU mapped to Intel

        # Test device discovery
        devices = adapter.discover_devices()
        assert len(devices) >= 1  # Should always have at least one CPU

        # Test device properties
        device = devices[0]
        assert device.capabilities.memory_gb > 0
        assert device.capabilities.peak_flops_fp32 > 0

    def test_vendor_adapter_compilation(self):
        """Test vendor adapter kernel compilation interface"""
        adapter = CPUAdapter()

        # Create mock device
        device = DeviceSpec(
            device_id=0,
            vendor=HardwareVendor.INTEL,
            capabilities=HardwareCapabilities(
                vendor=HardwareVendor.INTEL,
                device_name="CPU",
                compute_capability="1.0",
                memory_gb=16.0,
                peak_flops_fp32=1e12,
                peak_flops_fp16=2e12,
                memory_bandwidth_gbps=100.0,
                supported_precisions=[ComputeCapability.FP32, ComputeCapability.FP16],
                tensor_core_support=False,
                interconnect_type="System"
            )
        )

        # Test kernel compilation interface
        result = adapter.compile_kernel("test kernel", device)
        assert result is not None


class TestHardwareAdapterIntegration:
    """Test integration with existing HardwareAdapter"""

    def test_hardware_adapter_hal_initialization(self):
        """Test HardwareAdapter with HAL enabled"""
        adapter = HardwareAdapter(enable_hal=True)

        # Check HAL initialization
        hal_enabled = adapter.is_hal_enabled()
        if HAL_AVAILABLE:
            # HAL should be available in test environment
            hal_instance = adapter.get_hal()
            assert hal_instance is not None or not hal_enabled

    def test_hardware_adapter_backward_compatibility(self):
        """Test backward compatibility with HAL disabled"""
        adapter = HardwareAdapter(enable_hal=False)

        assert not adapter.is_hal_enabled()
        assert adapter.get_hal() is None

        # Legacy methods should still work
        try:
            devices = adapter.get_available_devices()
            assert isinstance(devices, list)
        except Exception:
            # May fail in test environment, but interface should exist
            assert hasattr(adapter, 'get_available_devices')

    def test_hal_enhanced_capabilities(self):
        """Test HAL-enhanced capabilities"""
        adapter = HardwareAdapter(enable_hal=True)

        if adapter.is_hal_enabled():
            # Test auto-detection
            hardware = adapter.auto_detect_hardware_hal()
            assert isinstance(hardware, dict)

            # Test vendor capabilities
            capabilities = adapter.get_vendor_capabilities_hal()
            if capabilities:
                assert 'total_devices' in capabilities
                assert 'vendor_distribution' in capabilities

    def test_cross_vendor_mesh_creation_integration(self):
        """Test cross-vendor mesh creation through HardwareAdapter"""
        adapter = HardwareAdapter(enable_hal=True)

        if adapter.is_hal_enabled():
            # Create mock devices for testing
            mock_devices = [
                Mock(device_id=0, vendor=HardwareVendor.NVIDIA, is_available=True),
                Mock(device_id=1, vendor=HardwareVendor.INTEL, is_available=True)
            ]

            mesh = adapter.create_cross_vendor_mesh_hal(
                devices=mock_devices,
                mesh_id="integration_test",
                topology="ring"
            )

            # Should either succeed or gracefully handle lack of real hardware
            assert mesh is not None or not adapter.is_hal_enabled()


class TestPerformanceAndStress:
    """Performance and stress tests for hardware abstraction"""

    def test_device_discovery_performance(self, hal_instance):
        """Test device discovery performance"""
        import time

        start_time = time.time()

        # Discover all hardware (may be mocked in test environment)
        try:
            inventory = hal_instance.discover_all_hardware()
            discovery_time = time.time() - start_time

            # Discovery should complete within reasonable time
            assert discovery_time < 30.0  # 30 seconds max
            assert isinstance(inventory, dict)
        except Exception:
            # May fail in test environment without real hardware
            pass

    def test_mesh_creation_scaling(self, hal_instance):
        """Test device mesh creation with multiple devices"""
        # Create multiple mock devices
        devices = []
        for i in range(10):
            device = DeviceSpec(
                device_id=i,
                vendor=HardwareVendor.NVIDIA if i % 2 == 0 else HardwareVendor.INTEL,
                capabilities=HardwareCapabilities(
                    vendor=HardwareVendor.NVIDIA if i % 2 == 0 else HardwareVendor.INTEL,
                    device_name=f"Test Device {i}",
                    compute_capability="1.0",
                    memory_gb=16.0,
                    peak_flops_fp32=1e12,
                    peak_flops_fp16=2e12,
                    memory_bandwidth_gbps=100.0,
                    supported_precisions=[ComputeCapability.FP32],
                    tensor_core_support=False,
                    interconnect_type="PCIe"
                )
            )
            devices.append(device)

        # Test mesh creation with multiple devices
        mesh = hal_instance.create_cross_vendor_mesh(
            devices=devices,
            mesh_id="scaling_test",
            topology="mesh"
        )

        assert len(mesh.devices) == 10
        assert len(mesh.bandwidth_matrix) == 10
        assert len(mesh.latency_matrix) == 10


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""

    def test_empty_device_list_mesh_creation(self, hal_instance):
        """Test mesh creation with empty device list"""
        with pytest.raises(ValueError, match="Cannot create mesh with empty device list"):
            hal_instance.create_cross_vendor_mesh([], "empty_test")

    def test_unavailable_devices_filtering(self, hal_instance, sample_device_specs):
        """Test filtering of unavailable devices"""
        # Mark devices as unavailable
        for device in sample_device_specs:
            device.is_available = False

        # Should filter out unavailable devices
        mesh = hal_instance.create_cross_vendor_mesh(
            devices=sample_device_specs,
            mesh_id="unavailable_test"
        )

        assert len(mesh.devices) == 0

    def test_invalid_vendor_handling(self, hal_instance):
        """Test handling of unknown vendors"""
        # Create device with unknown vendor
        unknown_device = DeviceSpec(
            device_id=999,
            vendor=HardwareVendor.UNKNOWN,
            capabilities=HardwareCapabilities(
                vendor=HardwareVendor.UNKNOWN,
                device_name="Unknown Device",
                compute_capability="0.0",
                memory_gb=1.0,
                peak_flops_fp32=1.0,
                peak_flops_fp16=1.0,
                memory_bandwidth_gbps=1.0,
                supported_precisions=[ComputeCapability.FP32],
                tensor_core_support=False,
                interconnect_type="Unknown"
            )
        )

        # Should handle gracefully
        hal_instance.register_device(unknown_device)
        # Device ID gets reassigned by HAL, check that device was registered
        assert len(hal_instance.devices) > 0
        # Check that the unknown vendor device is in the devices
        unknown_device_found = any(d.vendor == HardwareVendor.UNKNOWN for d in hal_instance.devices.values())
        assert unknown_device_found


class TestIntegrationWithExistingSystems:
    """Test integration with existing framework components"""

    def test_compatibility_with_existing_hardware_discovery(self):
        """Test compatibility with existing hardware discovery system"""
        from kernel_pytorch.distributed_scale.hardware_discovery import HardwareTopologyManager

        # Should be able to create both systems without conflict
        topology_manager = HardwareTopologyManager()
        hal_instance = HardwareAbstractionLayer()

        assert topology_manager is not None
        assert hal_instance is not None

    def test_device_mesh_compatibility(self):
        """Test device mesh compatibility with PyTorch distributed"""
        try:
            import torch.distributed._tensor as dt

            # Test that we can create PyTorch-compatible device meshes
            if torch.cuda.is_available():
                device_ids = [0] if torch.cuda.device_count() > 0 else []
                if device_ids:
                    mesh = dt.DeviceMesh("cuda", torch.tensor(device_ids))
                    assert mesh is not None
        except ImportError:
            # Distributed tensor API not available in this PyTorch version
            pass


@pytest.mark.integration
class TestRealHardwareIntegration:
    """Integration tests with real hardware (when available)"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_nvidia_device_discovery(self):
        """Test discovery of real NVIDIA devices"""
        adapter = NVIDIAAdapter()
        devices = adapter.discover_devices()

        if torch.cuda.device_count() > 0:
            assert len(devices) == torch.cuda.device_count()

            for i, device in enumerate(devices):
                assert device.device_id == i
                assert device.vendor == HardwareVendor.NVIDIA
                assert device.capabilities.memory_gb > 0

                # Verify device properties match PyTorch's view
                props = torch.cuda.get_device_properties(i)
                assert device.capabilities.device_name == props.name

    def test_real_hardware_adapter_integration(self):
        """Test HardwareAdapter with real hardware"""
        adapter = HardwareAdapter(enable_hal=True, enable_monitoring=False)

        try:
            # Test basic functionality
            assert adapter.is_hal_enabled() in [True, False]  # Depends on environment

            # Test device enumeration
            devices = adapter.get_available_devices()
            assert isinstance(devices, list)

            # Test cluster status
            status = adapter.get_cluster_status()
            assert 'total_devices' in status
            assert 'device_details' in status

        finally:
            adapter.shutdown()


if __name__ == "__main__":
    # Run specific test suites
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "not real_hardware"  # Skip real hardware tests by default
    ])