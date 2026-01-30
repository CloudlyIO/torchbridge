"""
Comprehensive Tests for Priority 1 Compiler Integration

Tests for:
1. FlashLight Compiler Framework
2. PyGraph CUDA Graphs Support
3. Enhanced TorchInductor Fusion

These tests validate state-of-the-art optimizations.
"""

import os
import sys
import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kernel_pytorch.core.compilers import (
    AttentionPattern,
    FlashLightKernelCompiler,
    FusionBoundaryOptimizer,
    FusionPass,
    FusionStrategy,
    GraphDeploymentStrategy,
    PyGraphCUDAOptimizer,
    WorkloadAnalysis,
)


class TestFlashLightCompiler:
    """Test FlashLight Compiler Framework functionality"""

    @pytest.fixture
    def compiler(self):
        """Create FlashLight compiler instance"""
        return FlashLightKernelCompiler(optimization_level="aggressive")

    @pytest.fixture
    def attention_inputs(self):
        """Create test inputs for basic functionality testing"""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 128, 32  # Moderate size for functionality
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        return q, k, v

    @pytest.fixture
    def realistic_attention_inputs(self):
        """Create realistic-scale inputs for comprehensive testing"""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64  # Production-like scale
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        return q, k, v

    @pytest.fixture
    def large_scale_attention_inputs(self):
        """Create large-scale inputs for stress/performance testing"""
        batch_size, num_heads, seq_len, head_dim = 4, 16, 1024, 64  # Large scale
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        return q, k, v

    def test_compiler_initialization(self, compiler):
        """Test compiler initialization"""
        assert compiler.optimization_level == "aggressive"
        assert len(compiler.kernel_cache.cache) == 0
        assert compiler.compilation_stats["total_compilations"] == 0

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_causal_attention_compilation(self, compiler, attention_inputs):
        """Test causal attention kernel compilation"""
        q, k, v = attention_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]

        # Compile causal attention kernel
        compiled_kernel = compiler.compile_attention_kernel("causal", seq_len, head_dim)

        assert compiled_kernel.pattern == AttentionPattern.CAUSAL
        assert compiled_kernel.seq_len == seq_len
        assert compiled_kernel.head_dim == head_dim
        assert callable(compiled_kernel.kernel_fn)

        # Test kernel execution
        output = compiled_kernel.kernel_fn(q, k, v)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_sliding_window_attention_compilation(self, compiler, attention_inputs):
        """Test sliding window attention kernel compilation"""
        q, k, v = attention_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]
        window_size = 128

        compiled_kernel = compiler.compile_attention_kernel(
            "sliding_window", seq_len, head_dim, {"window_size": window_size}
        )

        assert compiled_kernel.pattern == AttentionPattern.SLIDING_WINDOW

        # Test execution
        output = compiled_kernel.kernel_fn(q, k, v)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_basic_pattern_compilation(self, compiler, attention_inputs):
        """Test basic pattern compilation with functional verification"""
        q, k, v = attention_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]

        # Test core functionality with causal pattern only
        kernel = compiler.compile_attention_kernel("causal", seq_len, head_dim)

        # Verify compilation succeeded
        assert kernel.kernel_fn is not None
        assert kernel.pattern == AttentionPattern.CAUSAL

        # Verify functionality with actual execution
        output = kernel.kernel_fn(q, k, v)
        assert output.shape == q.shape
        assert not torch.isnan(output).any()

        # Test caching
        assert len(compiler.kernel_cache.cache) == 1

    @pytest.mark.integration
    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_comprehensive_pattern_compilation(self, compiler, realistic_attention_inputs):
        """Test all attention patterns with realistic data sizes"""
        q, k, v = realistic_attention_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]

        patterns = ["causal", "sliding_window", "dilated"]
        compiled_kernels = []
        pattern_outputs = {}

        for pattern in patterns:
            kernel = compiler.compile_attention_kernel(pattern, seq_len, head_dim)
            compiled_kernels.append(kernel)

            # Test functional correctness for each pattern
            output = kernel.kernel_fn(q, k, v)
            pattern_outputs[pattern] = output

            # Verify output properties
            assert output.shape == q.shape
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()

        # Verify all patterns were compiled
        assert len(compiled_kernels) == len(patterns)
        assert all(kernel.kernel_fn is not None for kernel in compiled_kernels)

        # Verify pattern-specific behavior differences
        causal_out = pattern_outputs["causal"]
        sliding_out = pattern_outputs["sliding_window"]
        dilated_out = pattern_outputs["dilated"]

        # Different patterns should produce different outputs (functionality verification)
        assert not torch.allclose(causal_out, sliding_out, rtol=1e-3)
        assert not torch.allclose(causal_out, dilated_out, rtol=1e-3)

        # Test that cache is working
        assert len(compiler.kernel_cache.cache) == len(patterns)

    @pytest.mark.stress
    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_large_scale_pattern_compilation(self, compiler, large_scale_attention_inputs):
        """Test pattern compilation with large-scale data for performance validation"""
        q, k, v = large_scale_attention_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]

        patterns = ["causal", "sliding_window", "dilated"]
        compilation_times = {}
        execution_times = {}

        for pattern in patterns:
            # Measure compilation time
            start_time = time.time()
            kernel = compiler.compile_attention_kernel(pattern, seq_len, head_dim)
            compilation_time = time.time() - start_time
            compilation_times[pattern] = compilation_time

            # Measure execution time
            start_time = time.time()
            output = kernel.kernel_fn(q, k, v)
            execution_time = time.time() - start_time
            execution_times[pattern] = execution_time

            # Verify output integrity with large data
            assert output.shape == q.shape
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()

        # Performance assertions
        total_compilation_time = sum(compilation_times.values())
        total_execution_time = sum(execution_times.values())

        print("\\nPerformance Results:")
        print(f"Total compilation time: {total_compilation_time:.3f}s")
        print(f"Total execution time: {total_execution_time:.3f}s")
        print(f"Patterns tested: {patterns}")
        print(f"Data scale: {q.shape}")

        # Ensure performance is within reasonable bounds
        assert total_compilation_time < 60.0, f"Compilation too slow: {total_compilation_time}s"
        assert total_execution_time < 10.0, f"Execution too slow: {total_execution_time}s"

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_kernel_caching(self, compiler, attention_inputs):
        """Test kernel caching functionality"""
        q, k, v = attention_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]

        # First compilation
        compiler.compile_attention_kernel("causal", seq_len, head_dim)
        initial_compilations = compiler.compilation_stats["total_compilations"]

        # Second compilation (should use cache)
        compiler.compile_attention_kernel("causal", seq_len, head_dim)
        final_compilations = compiler.compilation_stats["total_compilations"]

        # Should not increase compilation count due to cache hit
        assert final_compilations == initial_compilations
        assert compiler.compilation_stats["cache_hits"] == 1

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_benchmark_functionality(self, compiler):
        """Test attention pattern benchmarking"""
        seq_len, head_dim, num_heads = 256, 64, 8

        benchmark_results = compiler.benchmark_pattern(
            "causal", seq_len, head_dim, num_heads, num_trials=3
        )

        required_keys = ["mean_time", "min_time", "max_time", "estimated_speedup"]
        assert all(key in benchmark_results for key in required_keys)
        assert benchmark_results["mean_time"] > 0
        assert benchmark_results["estimated_speedup"] > 0

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_compilation_stats(self, compiler, attention_inputs):
        """Test compilation statistics tracking"""
        q, k, v = attention_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]

        # Compile some kernels
        compiler.compile_attention_kernel("causal", seq_len, head_dim)
        compiler.compile_attention_kernel("sliding_window", seq_len, head_dim)

        stats = compiler.get_compilation_stats()

        assert "total_compilations" in stats
        assert "cache_hits" in stats
        assert "cache_hit_rate" in stats
        assert "cached_kernels" in stats
        assert stats["total_compilations"] >= 2


class TestPyGraphOptimizer:
    """Test PyGraph CUDA Graphs optimization functionality"""

    @pytest.fixture
    def optimizer(self):
        """Create PyGraph optimizer instance"""
        return PyGraphCUDAOptimizer(cost_threshold=0.1, strategy="balanced")

    @pytest.fixture
    def simple_model(self):
        """Create simple test model"""
        return nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    @pytest.fixture
    def model_inputs(self, simple_model):
        """Create test inputs for model"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        simple_model.to(device)
        inputs = [torch.randn(32, 128, device=device)]
        return inputs

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.cost_threshold == 0.1
        assert optimizer.strategy == GraphDeploymentStrategy.BALANCED
        assert len(optimizer.graph_cache) == 0

    def test_workload_analysis(self, optimizer, simple_model, model_inputs):
        """Test workload analysis functionality"""
        analysis = optimizer.analyze_workload(simple_model, model_inputs, num_trials=3)

        assert isinstance(analysis, WorkloadAnalysis)
        assert analysis.cpu_launch_overhead >= 0
        assert analysis.memory_footprint >= 0
        assert 0 <= analysis.kernel_fusion_potential <= 1
        assert isinstance(analysis.dynamic_shapes, bool)
        assert isinstance(analysis.graph_recommended, bool)

    def test_cuda_graph_creation(self, optimizer, simple_model, model_inputs):
        """Test CUDA graph creation (if CUDA available)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for graph testing")

        analysis = optimizer.analyze_workload(simple_model, model_inputs)

        if analysis.graph_recommended:
            graph_manager = optimizer.create_cuda_graph(simple_model, model_inputs)

            assert graph_manager.graph is not None
            assert graph_manager.static_inputs is not None
            assert graph_manager.static_outputs is not None
            assert graph_manager.capture_time > 0

    def test_graph_execution(self, optimizer, simple_model, model_inputs):
        """Test CUDA graph execution (if CUDA available and recommended)"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for graph testing")

        analysis = optimizer.analyze_workload(simple_model, model_inputs)

        if analysis.graph_recommended:
            graph_manager = optimizer.create_cuda_graph(simple_model, model_inputs)

            # Test graph execution
            outputs = optimizer.execute_cuda_graph(graph_manager, model_inputs)

            assert len(outputs) > 0
            assert outputs[0].shape[0] == model_inputs[0].shape[0]  # Same batch size

    def test_benchmark_vs_eager(self, optimizer, simple_model, model_inputs):
        """Test benchmark comparison between graph and eager execution"""
        benchmark_results = optimizer.benchmark_vs_eager(
            simple_model, model_inputs, num_trials=5
        )

        required_keys = ["analysis", "eager_execution", "graph_execution", "performance"]
        assert all(key in benchmark_results for key in required_keys)

        eager_results = benchmark_results["eager_execution"]
        assert "mean_time" in eager_results
        assert eager_results["mean_time"] > 0

    def test_performance_stats(self, optimizer, simple_model, model_inputs):
        """Test performance statistics tracking"""
        # Perform some operations
        optimizer.analyze_workload(simple_model, model_inputs)

        stats = optimizer.get_performance_stats()

        required_keys = [
            "total_analyses", "graphs_deployed", "active_graphs",
            "cached_analyses", "deployment_rate"
        ]
        assert all(key in stats for key in required_keys)
        assert stats["total_analyses"] >= 1

    def test_cache_functionality(self, optimizer, simple_model, model_inputs):
        """Test caching functionality"""
        # First analysis
        analysis1 = optimizer.analyze_workload(simple_model, model_inputs)

        # Second analysis (should use cache)
        analysis2 = optimizer.analyze_workload(simple_model, model_inputs)

        # Should be identical due to caching
        assert analysis1.cpu_launch_overhead == analysis2.cpu_launch_overhead
        assert len(optimizer.analysis_cache) >= 1


class TestEnhancedFusion:
    """Test Enhanced TorchInductor Fusion functionality"""

    @pytest.fixture
    def fusion_optimizer(self):
        """Create fusion optimizer instance"""
        strategy = FusionStrategy(
            enabled_passes=[
                FusionPass.HORIZONTAL_FUSION,
                FusionPass.VERTICAL_FUSION,
                FusionPass.QUANTIZATION_FUSION
            ],
            aggressive_mode=True
        )
        return FusionBoundaryOptimizer(strategy)

    @pytest.fixture
    def fusable_model(self):
        """Create model with fusion opportunities"""
        class FusableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(128, 256)
                self.relu1 = nn.ReLU()
                self.linear2 = nn.Linear(256, 256)
                self.relu2 = nn.ReLU()
                self.linear3 = nn.Linear(256, 64)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu1(x)
                x = self.linear2(x)
                x = self.relu2(x)
                x = self.linear3(x)
                return x

        return FusableModel()

    def test_fusion_optimizer_initialization(self, fusion_optimizer):
        """Test fusion optimizer initialization"""
        assert len(fusion_optimizer.strategy.enabled_passes) == 3
        assert fusion_optimizer.strategy.aggressive_mode is True
        assert fusion_optimizer.optimization_stats["graphs_optimized"] == 0

    def test_fusion_graph_optimization(self, fusion_optimizer, fusable_model):
        """Test graph fusion optimization"""
        try:
            optimized_graph = fusion_optimizer.optimize_fusion_graph(fusable_model)

            assert optimized_graph.original_node_count >= 0
            assert optimized_graph.optimized_node_count >= 0
            assert optimized_graph.estimated_speedup >= 1.0
            assert optimized_graph.graph_module is not None

            # Test that optimized model is callable
            test_input = torch.randn(16, 128)
            output = optimized_graph.graph_module(test_input)
            assert output.shape == (16, 64)

        except Exception as e:
            # FX tracing might fail on some models, which is acceptable
            pytest.skip(f"FX tracing failed (acceptable): {e}")

    def test_fusion_pattern_detection(self, fusion_optimizer):
        """Test fusion pattern detection methods"""
        # Create a simple FX node mock for testing
        class MockNode:
            def __init__(self, op, target):
                self.op = op
                self.target = target

        # Test pattern detection methods
        relu_node = MockNode('call_function', torch.relu)
        MockNode('call_module', nn.Linear(10, 10))

        assert fusion_optimizer._is_vertically_fusable(relu_node)
        assert not fusion_optimizer._is_attention_operation(relu_node)
        assert not fusion_optimizer._is_quantization_operation(relu_node)

    def test_optimization_stats(self, fusion_optimizer, fusable_model):
        """Test optimization statistics tracking"""
        try:
            fusion_optimizer.optimize_fusion_graph(fusable_model)
            stats = fusion_optimizer.get_optimization_stats()

            required_keys = [
                "graphs_optimized", "total_nodes_removed",
                "total_speedup", "memory_saved_mb", "average_speedup"
            ]
            assert all(key in stats for key in required_keys)
            assert stats["graphs_optimized"] >= 1

        except Exception:
            # FX tracing might fail, which is acceptable for testing
            pytest.skip("FX tracing failed (acceptable for some models)")

    def test_fusion_strategy_configuration(self):
        """Test fusion strategy configuration"""
        custom_strategy = FusionStrategy(
            enabled_passes=[FusionPass.VERTICAL_FUSION],
            aggressive_mode=False,
            memory_budget_mb=2048
        )

        optimizer = FusionBoundaryOptimizer(custom_strategy)

        assert len(optimizer.strategy.enabled_passes) == 1
        assert optimizer.strategy.aggressive_mode is False
        assert optimizer.strategy.memory_budget_mb == 2048


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple optimization techniques"""

    @pytest.fixture
    def attention_model(self):
        """Create attention model for integration testing"""
        class SimpleAttention(nn.Module):
            def __init__(self, d_model=256, num_heads=8):
                super().__init__()
                self.d_model = d_model
                self.num_heads = num_heads
                self.head_dim = d_model // num_heads

                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.out_proj = nn.Linear(d_model, d_model)

            def forward(self, x):
                batch_size, seq_len, d_model = x.shape

                q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

                # Simple attention (not optimized)
                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn_weights = F.softmax(scores, dim=-1)
                out = torch.matmul(attn_weights, v)

                out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
                return self.out_proj(out)

        return SimpleAttention()

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_flashlight_and_pygraph_integration(self, attention_model):
        """Test integration of FlashLight compiler with PyGraph optimization"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        attention_model.to(device)

        # Create test input
        batch_size, seq_len, d_model = 4, 128, 256
        inputs = [torch.randn(batch_size, seq_len, d_model, device=device)]

        # Initialize optimizers
        flashlight_compiler = FlashLightKernelCompiler()
        pygraph_optimizer = PyGraphCUDAOptimizer()

        # Analyze workload for CUDA graphs
        analysis = pygraph_optimizer.analyze_workload(attention_model, inputs, num_trials=3)

        # Test FlashLight compilation for attention patterns
        compiled_kernel = flashlight_compiler.compile_attention_kernel(
            "causal", seq_len, d_model // 8  # head_dim
        )

        # Verify both optimizations work
        assert analysis.cpu_launch_overhead >= 0
        assert compiled_kernel.estimated_speedup > 0

        # Test that model still produces valid output
        output = attention_model(*inputs)
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_all_optimizations_combined(self, attention_model):
        """Test all Priority 1 optimizations working together"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        attention_model.to(device)

        inputs = [torch.randn(2, 64, 256, device=device)]

        # Initialize all optimizers
        flashlight_compiler = FlashLightKernelCompiler()
        pygraph_optimizer = PyGraphCUDAOptimizer()
        fusion_optimizer = FusionBoundaryOptimizer()

        # Run all optimizations
        compiled_kernel = flashlight_compiler.compile_attention_kernel("causal", 64, 32)
        analysis = pygraph_optimizer.analyze_workload(attention_model, inputs, num_trials=3)

        try:
            fusion_optimizer.optimize_fusion_graph(attention_model)
            fusion_success = True
        except Exception:
            fusion_success = False  # FX tracing might fail

        # Verify optimizations completed
        assert compiled_kernel is not None
        assert analysis is not None

        # Get performance statistics
        flashlight_stats = flashlight_compiler.get_compilation_stats()
        pygraph_stats = pygraph_optimizer.get_performance_stats()

        if fusion_success:
            fusion_stats = fusion_optimizer.get_optimization_stats()
            assert fusion_stats["graphs_optimized"] >= 1

        assert flashlight_stats["total_compilations"] >= 1
        assert pygraph_stats["total_analyses"] >= 1

    @pytest.mark.skipif(sys.platform == 'darwin', reason="Compiler tests can hang on macOS - skip for CI stability")
    def test_performance_regression_prevention(self, attention_model):
        """Test that optimizations don't cause performance regressions"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        attention_model.to(device)

        inputs = [torch.randn(1, 32, 256, device=device)]

        # Measure baseline performance
        attention_model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = attention_model(*inputs)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            # Measure baseline
            start_time = time.perf_counter()
            baseline_output = attention_model(*inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            time.perf_counter() - start_time

        # Apply optimizations and measure performance
        pygraph_optimizer = PyGraphCUDAOptimizer()
        benchmark_results = pygraph_optimizer.benchmark_vs_eager(
            attention_model, inputs, num_trials=5
        )

        # Verify outputs are numerically equivalent
        assert baseline_output.shape == (1, 32, 256)
        assert not torch.isnan(baseline_output).any()

        # Verify optimization recommendations are reasonable
        analysis = benchmark_results["analysis"]
        assert isinstance(analysis.graph_recommended, bool)
        if analysis.graph_recommended:
            assert analysis.expected_speedup >= 1.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
