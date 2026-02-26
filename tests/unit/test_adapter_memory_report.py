"""
Tests for Adapter Memory Reporting

Tests that AdapterResult includes correct before/after memory stats.
"""

import torch.nn as nn

from torchbridge.adapters.config import AdapterConfig
from torchbridge.adapters.engine import AdapterEngine, AdapterResult, _model_size_mb


class _SimpleModel(nn.Module):
    """Simple model for memory testing."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.fc = nn.Linear(256, 256)


class TestModelSizeMb:
    """Tests for _model_size_mb helper."""

    def test_returns_positive(self):
        """Model size should be positive."""
        model = _SimpleModel()
        size = _model_size_mb(model)
        assert size > 0

    def test_larger_model_has_more_memory(self):
        """Larger model should report more memory."""

        class Small(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

        class Large(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(1024, 1024)

        small_size = _model_size_mb(Small())
        large_size = _model_size_mb(Large())
        assert large_size > small_size


class TestAdapterMemoryReport:
    """Tests for memory reporting in AdapterResult."""

    def test_memory_before_positive(self):
        """memory_before_mb should be positive."""
        config = AdapterConfig(auto_detect_targets=False)
        engine = AdapterEngine(config)
        model = _SimpleModel()
        result = engine.inject(model)
        assert result.memory_before_mb > 0

    def test_memory_after_gte_before(self):
        """memory_after_mb should be >= memory_before_mb (adapters add params)."""
        config = AdapterConfig(auto_detect_targets=False)
        engine = AdapterEngine(config)
        model = _SimpleModel()
        result = engine.inject(model)
        assert result.memory_after_mb >= result.memory_before_mb

    def test_result_to_dict_has_memory(self):
        """AdapterResult.to_dict should include memory fields."""
        result = AdapterResult(
            success=True,
            method_applied=AdapterConfig().method,
            method_requested=AdapterConfig().method,
            memory_before_mb=100.0,
            memory_after_mb=105.0,
        )
        d = result.to_dict()
        assert "memory_before_mb" in d
        assert "memory_after_mb" in d
        assert d["memory_before_mb"] == 100.0
        assert d["memory_after_mb"] == 105.0
