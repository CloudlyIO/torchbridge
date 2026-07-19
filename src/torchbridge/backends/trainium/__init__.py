# SPDX-License-Identifier: Apache-2.0
"""
Trainium Backend Module for TorchBridge

This module provides Trainium-specific backend implementations for cross-backend
validation and configuration intelligence using the AWS Neuron SDK (torch_neuronx).

The current integration (Neuron SDK 2.27, PyTorch 2.9) uses XLA under the
hood (same as TPU, with torch.device('xla')). Trainium is differentiated
from TPU via environment variables and torch_neuronx imports.

Key Components:
- Trainium Device Management
- Neuron SDK Integration
- Trainium-specific Memory Management
- Neuron Compilation and Graph Caching
- Trainium Model Deployment Support

Example:
    ```python
    from torchbridge.backends.trainium import TrainiumBackend, TrainiumAdapter

    # Initialize Trainium backend
    backend = TrainiumBackend()
    model = backend.prepare_model(your_model)

    # Optimize for Trainium
    adapter = TrainiumAdapter(backend.config)
    optimized_model = adapter.optimize(model)
    ```
"""

from .neuron_compiler import NeuronCompiler
from .neuron_utilities import (
    get_neuron_env_info,
    get_neuron_sdk_version,
    is_neuron_available,
)
from .trainium_adapter import TrainiumAdapter
from .trainium_backend import TrainiumBackend

__all__ = [
    "TrainiumBackend",
    "TrainiumAdapter",
    "NeuronCompiler",
    "is_neuron_available",
    "get_neuron_sdk_version",
    "get_neuron_env_info",
]

# Version compatibility
__trainium_support__ = True

try:
    import torch_neuronx

    __neuron_available__ = True
    __neuron_sdk_version__ = torch_neuronx.__version__
except ImportError:
    __neuron_available__ = False
    __neuron_sdk_version__ = None
