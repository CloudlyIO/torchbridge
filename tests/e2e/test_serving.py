"""
Tests for the TorchBridge Serving Module (v0.3.9)

Tests cover:
- TorchServe handler functionality
- Triton configuration generation
- FastAPI server (when available)
- Integration between components
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

# ============================================================================
# Test Fixtures
# ============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 5):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TransformerModel(nn.Module):
    """Transformer-like model for testing."""

    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        return self.fc(attn_out)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def transformer_model():
    """Create a transformer model for testing."""
    return TransformerModel()


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(2, 10)


@pytest.fixture
def transformer_input():
    """Create transformer input tensor."""
    return torch.randn(2, 16, 64)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# TorchServe Handler Tests
# ============================================================================


class TestHandlerConfig:
    """Tests for HandlerConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from torchbridge.deployment.serving import torchserve_handler

        config = torchserve_handler.HandlerConfig()
        assert config.optimization_level == "balanced"
        assert config.enable_fp16 is True
        assert config.enable_fp8 is False
        assert config.max_batch_size == 32
        assert config.device == "auto"

    def test_config_to_dict(self):
        """Test config serialization."""
        from torchbridge.deployment.serving import torchserve_handler

        config = torchserve_handler.HandlerConfig(
            optimization_level="aggressive",
            max_batch_size=64,
        )
        config_dict = config.to_dict()

        assert config_dict["optimization_level"] == "aggressive"
        assert config_dict["max_batch_size"] == 64

    def test_config_from_dict(self):
        """Test config deserialization."""
        from torchbridge.deployment.serving import torchserve_handler

        data = {
            "optimization_level": "conservative",
            "enable_fp16": False,
            "max_batch_size": 16,
        }
        config = torchserve_handler.HandlerConfig.from_dict(data)

        assert config.optimization_level == "conservative"
        assert config.enable_fp16 is False
        assert config.max_batch_size == 16


class TestTorchBridgeHandler:
    """Tests for TorchBridgeHandler."""

    def test_handler_creation(self):
        """Test handler instantiation."""
        from torchbridge.deployment.serving import TorchBridgeHandler

        handler = TorchBridgeHandler()
        assert handler.model is None
        assert handler.initialized is False

    def test_preprocess_dict_input(self):
        """Test preprocessing dictionary input."""
        from torchbridge.deployment.serving import TorchBridgeHandler

        handler = TorchBridgeHandler()
        data = [{"input": [1.0, 2.0, 3.0, 4.0, 5.0]}]

        tensor = handler.preprocess(data)
        assert tensor.shape == (1, 5)
        assert tensor[0, 0].item() == 1.0

    def test_preprocess_batch(self):
        """Test preprocessing batch input."""
        from torchbridge.deployment.serving import TorchBridgeHandler

        handler = TorchBridgeHandler()
        data = [
            {"input": [1.0, 2.0, 3.0]},
            {"input": [4.0, 5.0, 6.0]},
        ]

        tensor = handler.preprocess(data)
        assert tensor.shape == (2, 3)

    def test_postprocess(self):
        """Test postprocessing output."""
        from torchbridge.deployment.serving import TorchBridgeHandler

        handler = TorchBridgeHandler()
        handler._last_inference_time = 0.005  # 5ms

        output = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        results = handler.postprocess(output)

        assert len(results) == 2
        assert results[0]["output"] == [1.0, 2.0, 3.0]
        assert "inference_time_ms" in results[0]

    def test_get_metrics(self):
        """Test metrics collection."""
        from torchbridge.deployment.serving import TorchBridgeHandler

        handler = TorchBridgeHandler()
        handler._inference_count = 100
        handler._total_inference_time = 0.5
        handler._last_inference_time = 0.005

        metrics = handler.get_metrics()

        assert metrics["inference_count"] == 100
        assert metrics["average_inference_time_ms"] == 5.0
        assert metrics["last_inference_time_ms"] == 5.0


class TestCreateTorchServeHandler:
    """Tests for handler creation utility."""

    def test_create_default_handler(self):
        """Test creating default handler."""
        from torchbridge.deployment.serving import create_torchserve_handler

        handler = create_torchserve_handler()
        assert handler is not None

    def test_create_with_config(self):
        """Test creating handler with config."""
        from torchbridge.deployment.serving import (
            create_torchserve_handler,
            torchserve_handler,
        )

        config = torchserve_handler.HandlerConfig(
            optimization_level="aggressive",
            max_batch_size=64,
        )
        handler = create_torchserve_handler(config=config)

        assert handler.config.optimization_level == "aggressive"
        assert handler.config.max_batch_size == 64


class TestPackageForTorchServe:
    """Tests for TorchServe packaging."""

    def test_package_model(self, simple_model, sample_input, temp_dir):
        """Test packaging a model for TorchServe."""
        from torchbridge.deployment.serving import (
            package_for_torchserve,
            torchserve_handler,
        )

        output_path = os.path.join(temp_dir, "model.mar")
        config = torchserve_handler.TorchServePackageConfig(
            model_name="test_model",
            version="1.0",
        )

        mar_path = package_for_torchserve(
            model=simple_model,
            output_path=output_path,
            config=config,
            sample_input=sample_input,
        )

        assert os.path.exists(mar_path)
        assert mar_path.endswith(".mar")


# ============================================================================
# Triton Configuration Tests
# ============================================================================


class TestTritonDataType:
    """Tests for TritonDataType."""

    def test_from_torch_dtype(self):
        """Test conversion from PyTorch dtype."""
        from torchbridge.deployment.serving import TritonDataType

        assert TritonDataType.from_torch_dtype(torch.float32) == TritonDataType.FP32
        assert TritonDataType.from_torch_dtype(torch.float16) == TritonDataType.FP16
        assert TritonDataType.from_torch_dtype(torch.int64) == TritonDataType.INT64

    def test_from_string(self):
        """Test conversion from string."""
        from torchbridge.deployment.serving import TritonDataType

        assert TritonDataType.from_string("FP32") == TritonDataType.FP32
        assert TritonDataType.from_string("float32") == TritonDataType.FP32
        assert TritonDataType.from_string("TYPE_FP16") == TritonDataType.FP16


class TestTritonInput:
    """Tests for TritonInput."""

    def test_input_to_config_str(self):
        """Test input config string generation."""
        from torchbridge.deployment.serving import triton_config

        inp = triton_config.TritonInput(
            name="input_tensor",
            data_type=triton_config.TritonDataType.FP32,
            dims=[1, 512],
        )
        config_str = inp.to_config_str()

        assert 'name: "input_tensor"' in config_str
        assert "data_type: TYPE_FP32" in config_str
        assert "dims: [1, 512]" in config_str


class TestTritonOutput:
    """Tests for TritonOutput."""

    def test_output_to_config_str(self):
        """Test output config string generation."""
        from torchbridge.deployment.serving import triton_config

        out = triton_config.TritonOutput(
            name="output_tensor",
            data_type=triton_config.TritonDataType.FP32,
            dims=[1, 10],
        )
        config_str = out.to_config_str()

        assert 'name: "output_tensor"' in config_str
        assert "data_type: TYPE_FP32" in config_str


class TestTritonDynamicBatching:
    """Tests for TritonDynamicBatching."""

    def test_dynamic_batching_config(self):
        """Test dynamic batching config generation."""
        from torchbridge.deployment.serving import triton_config

        batching = triton_config.TritonDynamicBatching(
            preferred_batch_size=[4, 8, 16],
            max_queue_delay_microseconds=50000,
        )
        config_str = batching.to_config_str()

        assert "dynamic_batching" in config_str
        assert "preferred_batch_size: [4, 8, 16]" in config_str
        assert "max_queue_delay_microseconds: 50000" in config_str


class TestTritonModelConfig:
    """Tests for TritonModelConfig."""

    def test_full_config_generation(self):
        """Test full configuration generation."""
        from torchbridge.deployment.serving import triton_config

        config = triton_config.TritonModelConfig(
            name="test_model",
            backend=triton_config.TritonBackend.PYTORCH,
            max_batch_size=32,
            inputs=[
                triton_config.TritonInput(
                    name="input",
                    data_type=triton_config.TritonDataType.FP32,
                    dims=[512],
                )
            ],
            outputs=[
                triton_config.TritonOutput(
                    name="output",
                    data_type=triton_config.TritonDataType.FP32,
                    dims=[10],
                )
            ],
        )

        config_str = config.to_config_str()

        assert 'name: "test_model"' in config_str
        assert 'backend: "pytorch"' in config_str
        assert "max_batch_size: 32" in config_str

    def test_config_save(self, temp_dir):
        """Test saving configuration to file."""
        from torchbridge.deployment.serving import triton_config

        config = triton_config.TritonModelConfig(
            name="test_model",
            max_batch_size=16,
        )

        config_path = os.path.join(temp_dir, "config.pbtxt")
        config.save(config_path)

        assert os.path.exists(config_path)
        with open(config_path) as f:
            content = f.read()
            assert 'name: "test_model"' in content


class TestCreateTritonConfig:
    """Tests for create_triton_config utility."""

    def test_create_basic_config(self):
        """Test creating basic Triton config."""
        from torchbridge.deployment.serving import create_triton_config

        config = create_triton_config(
            model_name="my_model",
            inputs=[("input", "FP32", [512])],
            outputs=[("output", "FP32", [10])],
            max_batch_size=32,
        )

        assert config.name == "my_model"
        assert config.max_batch_size == 32
        assert len(config.inputs) == 1
        assert len(config.outputs) == 1

    def test_create_with_dynamic_batching(self):
        """Test creating config with dynamic batching."""
        from torchbridge.deployment.serving import create_triton_config

        config = create_triton_config(
            model_name="batched_model",
            inputs=[("input", "FP32", [256])],
            outputs=[("output", "FP32", [10])],
            enable_dynamic_batching=True,
        )

        assert config.dynamic_batching is not None

    def test_create_multi_gpu(self):
        """Test creating config for multi-GPU."""
        from torchbridge.deployment.serving import create_triton_config

        config = create_triton_config(
            model_name="multi_gpu_model",
            inputs=[("input", "FP32", [512])],
            outputs=[("output", "FP32", [10])],
            gpu_count=4,
            instances_per_gpu=2,
        )

        assert len(config.instance_groups) == 4


class TestGenerateTritonModelRepository:
    """Tests for model repository generation."""

    def test_generate_torchscript_repo(self, simple_model, sample_input, temp_dir):
        """Test generating TorchScript model repository."""
        from torchbridge.deployment.serving import (
            create_triton_config,
            generate_triton_model_repository,
        )

        config = create_triton_config(
            model_name="ts_model",
            inputs=[("input", "FP32", [10])],
            outputs=[("output", "FP32", [5])],
        )

        model_dir = generate_triton_model_repository(
            model=simple_model,
            output_dir=temp_dir,
            config=config,
            sample_input=sample_input,
            export_format="torchscript",
        )

        # Check directory structure
        assert os.path.exists(model_dir)
        assert os.path.exists(os.path.join(model_dir, "config.pbtxt"))
        assert os.path.exists(os.path.join(model_dir, "1", "model.pt"))

    def test_generate_onnx_repo(self, simple_model, sample_input, temp_dir):
        """Test generating ONNX model repository."""
        pytest.importorskip("onnx")

        from torchbridge.deployment.serving import (
            TritonBackend,
            create_triton_config,
            generate_triton_model_repository,
        )

        config = create_triton_config(
            model_name="onnx_model",
            inputs=[("input", "FP32", [10])],
            outputs=[("output", "FP32", [5])],
            backend=TritonBackend.ONNXRUNTIME,
        )

        model_dir = generate_triton_model_repository(
            model=simple_model,
            output_dir=temp_dir,
            config=config,
            sample_input=sample_input,
            export_format="onnx",
        )

        assert os.path.exists(os.path.join(model_dir, "1", "model.onnx"))


# ============================================================================
# FastAPI Server Tests
# ============================================================================


class TestServerConfig:
    """Tests for ServerConfig."""

    def test_default_config(self):
        """Test default server configuration."""
        from torchbridge.deployment.serving import ServerConfig

        config = ServerConfig()
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.enable_fp16 is True
        assert config.max_batch_size == 32

    def test_config_to_dict(self):
        """Test config serialization."""
        from torchbridge.deployment.serving import ServerConfig

        config = ServerConfig(model_name="test", port=9000)
        config_dict = config.to_dict()

        assert config_dict["model_name"] == "test"
        assert config_dict["port"] == 9000


class TestInferenceServer:
    """Tests for InferenceServer."""

    @pytest.mark.skipif(
        not pytest.importorskip("fastapi", reason="FastAPI not available"),
        reason="FastAPI required",
    )
    def test_server_creation(self, simple_model):
        """Test server instantiation."""
        try:
            from torchbridge.deployment.serving import InferenceServer

            server = InferenceServer(model=simple_model)
            assert server.model is not None
            assert server.app is not None
        except ImportError:
            pytest.skip("FastAPI not available")

    @pytest.mark.skipif(
        not pytest.importorskip("fastapi", reason="FastAPI not available"),
        reason="FastAPI required",
    )
    def test_server_with_config(self, simple_model):
        """Test server with custom config."""
        try:
            from torchbridge.deployment.serving import InferenceServer, ServerConfig

            config = ServerConfig(
                model_name="custom_model",
                enable_fp16=False,
            )
            server = InferenceServer(model=simple_model, config=config)

            assert server.config.model_name == "custom_model"
            assert server.config.enable_fp16 is False
        except ImportError:
            pytest.skip("FastAPI not available")


class TestCreateFastAPIServer:
    """Tests for create_fastapi_server utility."""

    @pytest.mark.skipif(
        not pytest.importorskip("fastapi", reason="FastAPI not available"),
        reason="FastAPI required",
    )
    def test_create_server(self, simple_model):
        """Test creating FastAPI server."""
        try:
            from torchbridge.deployment.serving import create_fastapi_server

            server = create_fastapi_server(
                model=simple_model,
                model_name="test_model",
                model_version="2.0",
            )

            assert server.config.model_name == "test_model"
            assert server.config.model_version == "2.0"
        except ImportError:
            pytest.skip("FastAPI not available")


# ============================================================================
# Integration Tests
# ============================================================================


class TestServingIntegration:
    """Integration tests for serving module."""

    def test_import_all(self):
        """Test importing all serving components."""
        from torchbridge.deployment.serving import (
            TorchBridgeHandler,
            TritonModelConfig,
            create_triton_config,
        )

        assert TorchBridgeHandler is not None
        assert TritonModelConfig is not None
        assert create_triton_config is not None

    def test_deployment_module_imports(self):
        """Test importing from parent deployment module."""
        from torchbridge.deployment import (
            # Serving components
            TorchBridgeHandler,
            create_triton_config,
        )

        assert TorchBridgeHandler is not None
        assert create_triton_config is not None

    def test_full_workflow_torchscript(self, simple_model, sample_input, temp_dir):
        """Test full workflow: export -> configure -> package."""
        from torchbridge.deployment import (
            create_triton_config,
            export_to_torchscript,
        )

        # Export model
        ts_path = os.path.join(temp_dir, "model.pt")
        result = export_to_torchscript(
            model=simple_model,
            output_path=ts_path,
            sample_input=sample_input,
        )

        assert result.success
        assert os.path.exists(ts_path)

        # Create Triton config
        config = create_triton_config(
            model_name="workflow_model",
            inputs=[("input", "FP32", [10])],
            outputs=[("output", "FP32", [5])],
        )

        config_path = os.path.join(temp_dir, "config.pbtxt")
        config.save(config_path)

        assert os.path.exists(config_path)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
