"""
Hardware Abstraction Layer for PyTorch Optimization Framework

Universal hardware abstraction supporting proprietary GPUs and AI chips through:
- PyTorch PrivateUse1 integration
- Triton compiler backend abstraction
- Vendor-specific optimization plugins
- Distributed training across heterogeneous hardware
- Real-time inference load balancing
"""

from .hal_core import (
    DeviceSpec,
    HardwareAbstractionLayer,
    HardwareCapabilities,
    VendorAdapter,
)
from .privateuse1_integration import (
    CustomDeviceBackend,
    PrivateUse1Manager,
    register_custom_device,
)
from .vendor_adapters import (
    AMDAdapter,
    CPUAdapter,
    CustomHardwareAdapter,
    IntelAdapter,
    NVIDIAAdapter,
    auto_detect_best_adapter,
    create_custom_adapter,
    create_vendor_adapter_enhanced,
    get_available_vendors,
)

__all__ = [
    # Core HAL
    'HardwareAbstractionLayer',
    'VendorAdapter',
    'HardwareCapabilities',
    'DeviceSpec',

    # PrivateUse1 Integration
    'CustomDeviceBackend',
    'PrivateUse1Manager',
    'register_custom_device',

    # Vendor Adapters
    'NVIDIAAdapter',
    'AMDAdapter',
    'IntelAdapter',
    'CPUAdapter',

    # Custom Hardware Support
    'CustomHardwareAdapter',
    'create_custom_adapter',
    'create_vendor_adapter_enhanced',

    # Utility Functions
    'get_available_vendors',
    'auto_detect_best_adapter'
]
