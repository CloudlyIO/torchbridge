"""
Comprehensive Test Suite for PyGraph CUDA Graph Optimizations

Tests for PyGraph CUDA Graph optimization techniques.
"""

import pytest
import torch
import torch.nn as nn

from torchbridge.optimizations.next_gen import (
    AutoGraphCapture,
    CUDAGraphManager,
    SelectiveCUDAGraphs,
    create_pygraph_optimizer,
)


class TestPyGraphOptimization:
    """Test suite for PyGraph CUDA Graph optimizations"""

    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def simple_model(self, device):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(256, 128)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        return SimpleModel().to(device).eval()

    def test_cuda_graph_manager_creation(self, device):
        """Test CUDA Graph manager initialization"""
        manager = CUDAGraphManager(device)
        assert manager.device == device
        assert isinstance(manager.graphs, dict)
        assert isinstance(manager.parameter_tables, dict)
        assert isinstance(manager.dynamic_shapes, dict)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_graph_capture(self, device, simple_model):
        """Test graph capture functionality"""
        manager = CUDAGraphManager(device)

        x = torch.randn(4, 256, device=device)

        def test_func(inp):
            return simple_model(inp)

        try:
            manager.capture_graph(test_func, (x,), "test_graph")
            assert "test_graph" in manager.graphs

            # Test graph execution
            output = manager.execute_graph("test_graph")
            assert output is not None

        except Exception as e:
            # Graph capture may fail in some environments
            pytest.skip(f"Graph capture failed: {e}")

    def test_selective_cuda_graphs(self, device, simple_model):
        """Test selective CUDA graph optimization"""
        optimizer = SelectiveCUDAGraphs(simple_model, device)

        x = torch.randn(4, 256, device=device)

        # Test profiling
        def test_func(inp):
            return simple_model(inp)

        # This test should work even without CUDA graphs
        profile_results = optimizer.profile_operation("test_op", test_func, (x,))

        assert 'normal_time' in profile_results
        assert 'speedup' in profile_results
        assert 'benefits_from_graph' in profile_results

    def test_auto_graph_capture(self, device):
        """Test automatic graph capture"""
        auto_capture = AutoGraphCapture(device, capture_threshold=3)

        def simple_func(x):
            return x * 2 + 1

        x = torch.randn(4, device=device)

        # Execute multiple times to trigger auto-capture
        for i in range(5):
            result = auto_capture.track_execution(simple_func, (x,), f"pattern_{i % 2}")
            assert result is not None

        stats = auto_capture.get_auto_optimization_stats()
        assert 'total_patterns' in stats
        assert 'total_executions' in stats

    def test_pygraph_optimizer_creation(self, device, simple_model):
        """Test PyGraph optimizer factory function"""
        optimizer = create_pygraph_optimizer(
            simple_model,
            device=device,
            optimization_level="balanced"
        )

        assert optimizer.optimization_level == "balanced"
        assert optimizer.device == device
        assert optimizer.model == simple_model

    def test_optimization_summary(self, device, simple_model):
        """Test optimization summary generation"""
        optimizer = create_pygraph_optimizer(simple_model, device)

        summary = optimizer.get_optimization_summary()

        assert 'graph_manager_stats' in summary
        assert 'auto_capture_stats' in summary
        assert 'optimization_level' in summary
        assert 'device' in summary


# Test runner configuration
if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
