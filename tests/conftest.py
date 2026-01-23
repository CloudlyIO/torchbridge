"""
Shared pytest fixtures for KernelPyTorch test suite.

This module provides reusable fixtures for:
- Device detection and availability
- Common model architectures
- Sample input data
- Backend configurations
- Mock utilities for hardware simulation
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Tuple


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def cpu_device():
    """Force CPU device for tests requiring CPU."""
    return torch.device("cpu")


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def mps_available():
    """Check if MPS (Apple Silicon) is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


# ============================================================================
# Model Fixtures - Simple Models
# ============================================================================

@pytest.fixture
def simple_linear_model():
    """Simple linear model for basic testing."""
    return nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )


@pytest.fixture
def small_mlp():
    """Small MLP for quick tests."""
    return nn.Sequential(
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16)
    )


@pytest.fixture
def medium_mlp():
    """Medium MLP for more comprehensive tests."""
    return nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32)
    )


# ============================================================================
# Model Fixtures - Transformer Models
# ============================================================================

class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing attention-based models."""

    def __init__(self, d_model: int = 256, num_heads: int = 4, d_ff: int = 512):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


@pytest.fixture
def simple_transformer_block():
    """Simple transformer block for attention testing."""
    return SimpleTransformerBlock(d_model=256, num_heads=4, d_ff=512)


@pytest.fixture
def small_transformer_block():
    """Smaller transformer block for quick tests."""
    return SimpleTransformerBlock(d_model=64, num_heads=2, d_ff=128)


@pytest.fixture
def large_transformer_block():
    """Larger transformer block for comprehensive tests."""
    return SimpleTransformerBlock(d_model=512, num_heads=8, d_ff=2048)


# ============================================================================
# Model Fixtures - Vision Models
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN for vision testing."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


@pytest.fixture
def simple_cnn():
    """Simple CNN for vision testing."""
    return SimpleCNN(in_channels=3, num_classes=10)


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def linear_input():
    """Sample input for linear models (batch_size=32, features=128)."""
    return torch.randn(32, 128)


@pytest.fixture
def transformer_input():
    """Sample input for transformer models (batch=8, seq_len=64, d_model=256)."""
    return torch.randn(8, 64, 256)


@pytest.fixture
def small_transformer_input():
    """Small transformer input (batch=4, seq_len=32, d_model=64)."""
    return torch.randn(4, 32, 64)


@pytest.fixture
def vision_input():
    """Sample input for CNN models (batch=16, channels=3, height=32, width=32)."""
    return torch.randn(16, 3, 32, 32)


@pytest.fixture
def sample_data():
    """Dictionary of sample data for various test scenarios."""
    return {
        "linear_input": torch.randn(32, 128),
        "transformer_input": torch.randn(8, 64, 256),
        "vision_input": torch.randn(16, 3, 32, 32),
        "attention_q": torch.randn(4, 8, 64),  # batch, seq, head_dim
        "attention_k": torch.randn(4, 8, 64),
        "attention_v": torch.randn(4, 8, 64),
    }


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def kernel_config():
    """Default KernelPyTorchConfig for testing."""
    from kernel_pytorch.core.config import KernelPyTorchConfig
    return KernelPyTorchConfig()


@pytest.fixture
def production_config():
    """Production-optimized KernelPyTorchConfig."""
    from kernel_pytorch.core.config import KernelPyTorchConfig
    return KernelPyTorchConfig.for_production()


@pytest.fixture
def development_config():
    """Development-friendly KernelPyTorchConfig."""
    from kernel_pytorch.core.config import KernelPyTorchConfig
    return KernelPyTorchConfig.for_development()


# ============================================================================
# Backend Configuration Fixtures
# ============================================================================

@pytest.fixture
def nvidia_config():
    """Default NVIDIA configuration."""
    from kernel_pytorch.core.config import NVIDIAConfig
    return NVIDIAConfig()


@pytest.fixture
def tpu_config():
    """Default TPU configuration."""
    from kernel_pytorch.core.config import TPUConfig
    return TPUConfig()


@pytest.fixture
def amd_config():
    """Default AMD configuration."""
    from kernel_pytorch.core.config import AMDConfig
    return AMDConfig()


# ============================================================================
# Mock Fixtures for Hardware Simulation
# ============================================================================

@pytest.fixture
def mock_cuda_available():
    """Mock CUDA as available."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=1):
            mock_props = MagicMock()
            mock_props.name = "NVIDIA GeForce RTX 4090"
            mock_props.major = 8
            mock_props.minor = 9
            mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB
            with patch("torch.cuda.get_device_properties", return_value=mock_props):
                yield mock_props


@pytest.fixture
def mock_h100():
    """Mock NVIDIA H100 GPU."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=1):
            mock_props = MagicMock()
            mock_props.name = "NVIDIA H100 80GB"
            mock_props.major = 9
            mock_props.minor = 0
            mock_props.total_memory = 80 * 1024 * 1024 * 1024  # 80GB
            with patch("torch.cuda.get_device_properties", return_value=mock_props):
                yield mock_props


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_multi_gpu():
    """Mock multi-GPU setup (4 GPUs)."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=4):
            mock_props = MagicMock()
            mock_props.name = "NVIDIA A100 40GB"
            mock_props.major = 8
            mock_props.minor = 0
            mock_props.total_memory = 40 * 1024 * 1024 * 1024  # 40GB
            with patch("torch.cuda.get_device_properties", return_value=mock_props):
                yield mock_props


# ============================================================================
# Training Fixtures
# ============================================================================

@pytest.fixture
def training_batch():
    """Sample training batch with inputs and labels."""
    return {
        "inputs": torch.randn(32, 128),
        "labels": torch.randint(0, 10, (32,))
    }


@pytest.fixture
def transformer_training_batch():
    """Sample transformer training batch."""
    return {
        "inputs": torch.randn(8, 64, 256),
        "labels": torch.randint(0, 1000, (8, 64))  # Token-level labels
    }


@pytest.fixture
def vision_training_batch():
    """Sample vision training batch."""
    return {
        "inputs": torch.randn(16, 3, 224, 224),
        "labels": torch.randint(0, 1000, (16,))  # ImageNet-like labels
    }


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def tolerance():
    """Default numerical tolerance for floating point comparisons."""
    return {"rtol": 1e-3, "atol": 1e-5}


@pytest.fixture
def strict_tolerance():
    """Strict tolerance for precision-sensitive tests."""
    return {"rtol": 1e-5, "atol": 1e-7}


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for test outputs."""
    return tmp_path


# ============================================================================
# Benchmark Fixtures
# ============================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "warmup_runs": 3,
        "benchmark_runs": 10,
        "min_runtime_seconds": 0.1,
    }


@pytest.fixture
def quick_benchmark_config():
    """Quick benchmark configuration for CI."""
    return {
        "warmup_runs": 1,
        "benchmark_runs": 3,
        "min_runtime_seconds": 0.01,
    }


# ============================================================================
# Parametrize Helpers
# ============================================================================

# Common batch sizes for parametrized tests
BATCH_SIZES = [1, 4, 16, 32]

# Common model dimensions
MODEL_DIMS = [64, 128, 256, 512]

# Common sequence lengths
SEQ_LENGTHS = [32, 64, 128, 256]

# Precision types
PRECISIONS = ["fp32", "fp16", "bf16"]


@pytest.fixture(params=BATCH_SIZES)
def batch_size(request):
    """Parametrized batch size fixture."""
    return request.param


@pytest.fixture(params=MODEL_DIMS)
def model_dim(request):
    """Parametrized model dimension fixture."""
    return request.param


@pytest.fixture(params=SEQ_LENGTHS)
def seq_length(request):
    """Parametrized sequence length fixture."""
    return request.param


@pytest.fixture(params=PRECISIONS)
def precision(request):
    """Parametrized precision fixture."""
    return request.param
