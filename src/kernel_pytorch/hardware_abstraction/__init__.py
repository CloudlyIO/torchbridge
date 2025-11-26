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
    HardwareAbstractionLayer,
    VendorAdapter,
    HardwareCapabilities,
    DeviceSpec
)

from .privateuse1_integration import (
    ProprietaryDeviceBackend,
    PrivateUse1Manager,
    register_custom_device
)

from .triton_backends import (
    ProprietaryTritonBackend,
    TritonCompilerManager,
    KernelOptimizationPipeline
)

from .plugin_system import (
    HardwarePluginManager,
    BaseHardwarePlugin,
    CapabilityMatrix
)

from .vendor_adapters import (
    NVIDIAAdapter,
    AMDAdapter,
    IntelAdapter,
    CustomASICAdapter
)

__all__ = [
    # Core HAL
    'HardwareAbstractionLayer',
    'VendorAdapter',
    'HardwareCapabilities',
    'DeviceSpec',

    # PrivateUse1 Integration
    'ProprietaryDeviceBackend',
    'PrivateUse1Manager',
    'register_custom_device',

    # Triton Backends
    'ProprietaryTritonBackend',
    'TritonCompilerManager',
    'KernelOptimizationPipeline',

    # Plugin System
    'HardwarePluginManager',
    'BaseHardwarePlugin',
    'CapabilityMatrix',

    # Vendor Adapters
    'NVIDIAAdapter',
    'AMDAdapter',
    'IntelAdapter',
    'CustomASICAdapter'
]