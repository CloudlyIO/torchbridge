"""Tests for checkpoint metadata and cross-backend portability."""

import json

import pytest
import torch

from torchbridge.checkpoint.metadata import (
    _PORTABLE_DTYPE_MAP,
    CheckpointMetadata,
    PortabilityNormalizer,
    _backend_supports_dtype,
    _dtype_to_str,
    _str_to_dtype,
)
from torchbridge.core.config import HardwareBackend


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def _make_metadata(self, **overrides):
        defaults = {
            "checkpoint_id": "/ckpts/ckpt_001",
            "timestamp": "2026-02-19T12:00:00+00:00",
            "torchbridge_version": "0.5.28",
            "pytorch_version": "2.7.0",
            "backend": "cuda",
            "architecture": "hopper",
            "world_size": 8,
            "local_world_size": 8,
            "model_params": int(7e9),
            "dtype_map": {"weight": "bfloat16"},
            "device_map": {"weight": "cuda:0"},
        }
        defaults.update(overrides)
        return CheckpointMetadata(**defaults)

    def test_creation(self):
        m = self._make_metadata()
        assert m.checkpoint_id == "/ckpts/ckpt_001"
        assert m.backend == "cuda"
        assert m.world_size == 8
        assert m.model_params == int(7e9)

    def test_frozen(self):
        m = self._make_metadata()
        with pytest.raises(AttributeError):
            m.backend = "amd"  # type: ignore[misc]

    def test_to_dict(self):
        m = self._make_metadata()
        d = m.to_dict()
        assert d["checkpoint_id"] == "/ckpts/ckpt_001"
        assert d["backend"] == "cuda"
        assert d["architecture"] == "hopper"
        assert d["world_size"] == 8
        assert d["dtype_map"] == {"weight": "bfloat16"}
        assert d["device_map"] == {"weight": "cuda:0"}

    def test_from_dict_roundtrip(self):
        original = self._make_metadata()
        d = original.to_dict()
        restored = CheckpointMetadata.from_dict(d)
        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.backend == original.backend
        assert restored.world_size == original.world_size
        assert restored.model_params == original.model_params
        assert restored.dtype_map == original.dtype_map

    def test_from_dict_minimal(self):
        d = {
            "checkpoint_id": "ckpt",
            "timestamp": "2026-01-01T00:00:00",
            "torchbridge_version": "0.5.28",
            "pytorch_version": "2.7.0",
            "backend": "cpu",
            "world_size": 1,
        }
        m = CheckpointMetadata.from_dict(d)
        assert m.architecture is None
        assert m.local_world_size == 1
        assert m.model_params is None
        assert m.dtype_map == {}

    def test_save_and_load(self, tmp_path):
        m = self._make_metadata()
        filepath = tmp_path / "meta.json"
        m.save(filepath)

        assert filepath.exists()
        loaded = CheckpointMetadata.load(filepath)
        assert loaded.checkpoint_id == m.checkpoint_id
        assert loaded.dtype_map == m.dtype_map

    def test_save_creates_parent_dirs(self, tmp_path):
        m = self._make_metadata()
        filepath = tmp_path / "deep" / "nested" / "meta.json"
        m.save(filepath)
        assert filepath.exists()

    def test_json_format(self, tmp_path):
        m = self._make_metadata()
        filepath = tmp_path / "meta.json"
        m.save(filepath)
        data = json.loads(filepath.read_text())
        assert isinstance(data, dict)
        assert data["backend"] == "cuda"


class TestDtypeConversion:
    """Tests for dtype string conversion helpers."""

    def test_dtype_to_str(self):
        assert _dtype_to_str(torch.float32) == "float32"
        assert _dtype_to_str(torch.bfloat16) == "bfloat16"
        assert _dtype_to_str(torch.int8) == "int8"
        assert _dtype_to_str(torch.float8_e4m3fn) == "float8_e4m3fn"

    def test_str_to_dtype(self):
        assert _str_to_dtype("float32") == torch.float32
        assert _str_to_dtype("bfloat16") == torch.bfloat16
        assert _str_to_dtype("float8_e4m3fn") == torch.float8_e4m3fn

    def test_str_to_dtype_unknown_fallback(self):
        assert _str_to_dtype("unknown_type") == torch.float32


class TestPortableDtypeMap:
    """Tests for the portable dtype mapping."""

    def test_fp8_e4m3_maps_to_fp16(self):
        assert _PORTABLE_DTYPE_MAP[torch.float8_e4m3fn] == torch.float16

    def test_fp8_e5m2_maps_to_fp16(self):
        assert _PORTABLE_DTYPE_MAP[torch.float8_e5m2] == torch.float16

    def test_standard_dtypes_not_in_map(self):
        assert torch.float32 not in _PORTABLE_DTYPE_MAP
        assert torch.bfloat16 not in _PORTABLE_DTYPE_MAP
        assert torch.float16 not in _PORTABLE_DTYPE_MAP


class TestBackendSupportsDtype:
    """Tests for _backend_supports_dtype."""

    def test_fp8_supported_on_cuda(self):
        assert _backend_supports_dtype(HardwareBackend.CUDA, torch.float8_e4m3fn)

    def test_fp8_supported_on_amd(self):
        assert _backend_supports_dtype(HardwareBackend.AMD, torch.float8_e5m2)

    def test_fp8_not_supported_on_cpu(self):
        assert not _backend_supports_dtype(HardwareBackend.CPU, torch.float8_e4m3fn)

    def test_fp8_not_supported_on_tpu(self):
        assert not _backend_supports_dtype(HardwareBackend.TPU, torch.float8_e5m2)

    def test_standard_dtypes_supported_everywhere(self):
        for backend in [
            HardwareBackend.CUDA,
            HardwareBackend.AMD,
            HardwareBackend.CPU,
            HardwareBackend.TPU,
        ]:
            assert _backend_supports_dtype(backend, torch.float32)
            assert _backend_supports_dtype(backend, torch.bfloat16)
            assert _backend_supports_dtype(backend, torch.int8)


class TestPortabilityNormalizer:
    """Tests for PortabilityNormalizer."""

    def test_normalize_cpu_tensors(self):
        state = {
            "weight": torch.randn(4, 4),
            "bias": torch.randn(4),
        }
        normalized, dtype_map, device_map = PortabilityNormalizer.normalize_state_dict(
            state, HardwareBackend.CUDA
        )
        assert "weight" in dtype_map
        assert dtype_map["weight"] == "float32"
        assert normalized["weight"].device == torch.device("cpu")

    def test_normalize_preserves_values(self):
        original = torch.tensor([1.0, 2.0, 3.0])
        state = {"param": original.clone()}
        normalized, _, _ = PortabilityNormalizer.normalize_state_dict(
            state, HardwareBackend.CPU
        )
        torch.testing.assert_close(normalized["param"], original)

    def test_normalize_non_tensor_passthrough(self):
        state = {
            "step": 100,
            "lr": 0.001,
            "name": "test",
        }
        normalized, dtype_map, device_map = PortabilityNormalizer.normalize_state_dict(
            state, HardwareBackend.CPU
        )
        assert normalized["step"] == 100
        assert normalized["lr"] == 0.001
        assert normalized["name"] == "test"
        assert len(dtype_map) == 0

    def test_normalize_nested_dict(self):
        state = {
            "optimizer": {
                "param_groups": torch.randn(2),
                "step": 50,
            }
        }
        normalized, dtype_map, _ = PortabilityNormalizer.normalize_state_dict(
            state, HardwareBackend.CUDA
        )
        assert "optimizer.param_groups" in dtype_map
        assert isinstance(normalized["optimizer"], dict)
        assert normalized["optimizer"]["step"] == 50

    def test_restore_to_cpu(self):
        metadata = CheckpointMetadata(
            checkpoint_id="test",
            timestamp="2026-01-01",
            torchbridge_version="0.5.28",
            pytorch_version="2.7.0",
            backend="cuda",
            architecture=None,
            world_size=1,
            local_world_size=1,
            dtype_map={"weight": "float32"},
            device_map={"weight": "cuda:0"},
        )
        state = {"weight": torch.randn(4, 4)}
        restored = PortabilityNormalizer.restore_state_dict(
            state, metadata, HardwareBackend.CPU, torch.device("cpu")
        )
        assert restored["weight"].device == torch.device("cpu")

    def test_restore_non_tensor_passthrough(self):
        metadata = CheckpointMetadata(
            checkpoint_id="test",
            timestamp="2026-01-01",
            torchbridge_version="0.5.28",
            pytorch_version="2.7.0",
            backend="cpu",
            architecture=None,
            world_size=1,
            local_world_size=1,
        )
        state = {"step": 42, "lr": 0.01}
        restored = PortabilityNormalizer.restore_state_dict(
            state, metadata, HardwareBackend.CPU, torch.device("cpu")
        )
        assert restored["step"] == 42
        assert restored["lr"] == 0.01

    def test_normalize_and_restore_roundtrip(self):
        original = torch.randn(8, 8)
        state = {"weight": original.clone()}

        normalized, dtype_map, device_map = PortabilityNormalizer.normalize_state_dict(
            state, HardwareBackend.CPU
        )

        metadata = CheckpointMetadata(
            checkpoint_id="test",
            timestamp="2026-01-01",
            torchbridge_version="0.5.28",
            pytorch_version="2.7.0",
            backend="cpu",
            architecture=None,
            world_size=1,
            local_world_size=1,
            dtype_map=dtype_map,
            device_map=device_map,
        )

        restored = PortabilityNormalizer.restore_state_dict(
            normalized, metadata, HardwareBackend.CPU, torch.device("cpu")
        )
        torch.testing.assert_close(restored["weight"], original)
