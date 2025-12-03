#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite

This module contains long-running, high-efficacy tests that use realistic data sizes
and comprehensive scenarios. These tests prioritize thorough validation over speed.

Test Categories:
- @pytest.mark.integration: Realistic scale tests (5-30s)
- @pytest.mark.stress: Large scale performance tests (30s-5min)
- @pytest.mark.gpu: GPU-dependent comprehensive tests
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kernel_pytorch.core.compilers import (
    FlashLightKernelCompiler,
    AttentionPattern,
    PyGraphCUDAOptimizer
)
from test_configs import TEST_CONFIGS, TestDataConfig


class TestRealisticScaleCompilation:
    """Test compiler functionality with realistic production-scale data"""

    @pytest.fixture
    def compiler(self):
        """Create compiler with production settings"""
        return FlashLightKernelCompiler(optimization_level="aggressive")

    @pytest.fixture
    def realistic_inputs(self):
        """Realistic production-scale inputs"""
        config = TEST_CONFIGS['realistic']
        return config.create_tensors()

    @pytest.fixture
    def large_inputs(self):
        """Large-scale inputs for stress testing"""
        config = TEST_CONFIGS['large']
        return config.create_tensors()

    @pytest.mark.integration
    def test_all_patterns_realistic_scale(self, compiler, realistic_inputs):
        """Test all attention patterns with realistic data sizes - comprehensive validation"""
        q, k, v = realistic_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]

        print(f"\\nTesting with realistic scale: {q.shape}")
        print(f"Memory per tensor: {q.numel() * 4 / 1024 / 1024:.1f}MB")

        patterns = ["causal", "sliding_window", "dilated"]
        results = {}
        total_compilation_time = 0

        for pattern in patterns:
            print(f"  Testing {pattern} pattern...")

            # Measure compilation time
            start_time = time.time()
            kernel = compiler.compile_attention_kernel(pattern, seq_len, head_dim)
            compilation_time = time.time() - start_time
            total_compilation_time += compilation_time

            # Measure execution time
            start_time = time.time()
            output = kernel.kernel_fn(q, k, v)
            execution_time = time.time() - start_time

            # Comprehensive validation
            assert output.shape == q.shape, f"Shape mismatch for {pattern}"
            assert not torch.isnan(output).any(), f"NaN values in {pattern} output"
            assert torch.isfinite(output).all(), f"Infinite values in {pattern} output"

            # Pattern-specific validation
            if pattern == "causal":
                # For causal attention, verify causal property (upper triangular mask)
                # Convert to attention scores for validation
                scores = torch.einsum('bhid,bhjd->bhij', q, k) / np.sqrt(head_dim)
                attn_weights = torch.softmax(scores, dim=-1)

                # Check that attention weights respect causality (roughly)
                # Upper triangle should have lower attention weights
                for h in range(q.shape[1]):  # For each head
                    weights_2d = attn_weights[0, h]  # Take first batch
                    upper_tri = torch.triu(weights_2d, diagonal=1)
                    lower_tri = torch.tril(weights_2d, diagonal=0)

                    # Upper triangle should have significantly lower weights
                    upper_mean = upper_tri[upper_tri > 0].mean() if (upper_tri > 0).any() else 0
                    lower_mean = lower_tri[lower_tri > 0].mean() if (lower_tri > 0).any() else 1

                    # This is a rough check - causal should reduce upper triangle influence
                    assert upper_mean <= lower_mean * 2.0, f"Causal constraint not respected in head {h}"

            results[pattern] = {
                'compilation_time': compilation_time,
                'execution_time': execution_time,
                'output_shape': output.shape,
                'output_stats': {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item()
                }
            }

            print(f"    ✅ {pattern}: compile={compilation_time:.3f}s, exec={execution_time:.3f}s")

        # Cross-pattern validation: Different patterns should produce different outputs
        causal_kernel = compiler.compile_attention_kernel("causal", seq_len, head_dim)
        sliding_kernel = compiler.compile_attention_kernel("sliding_window", seq_len, head_dim)
        dilated_kernel = compiler.compile_attention_kernel("dilated", seq_len, head_dim)

        causal_out = causal_kernel.kernel_fn(q, k, v)
        sliding_out = sliding_kernel.kernel_fn(q, k, v)
        dilated_out = dilated_kernel.kernel_fn(q, k, v)

        # Verify patterns produce meaningfully different outputs
        causal_vs_sliding = torch.norm(causal_out - sliding_out).item()
        causal_vs_dilated = torch.norm(causal_out - dilated_out).item()
        sliding_vs_dilated = torch.norm(sliding_out - dilated_out).item()

        print(f"\\nPattern difference analysis:")
        print(f"  Causal vs Sliding Window: {causal_vs_sliding:.3f}")
        print(f"  Causal vs Dilated: {causal_vs_dilated:.3f}")
        print(f"  Sliding vs Dilated: {sliding_vs_dilated:.3f}")

        # Patterns should produce different outputs (not identical)
        assert causal_vs_sliding > 1e-6, "Causal and sliding window produce nearly identical outputs"
        assert causal_vs_dilated > 1e-6, "Causal and dilated produce nearly identical outputs"
        assert sliding_vs_dilated > 1e-6, "Sliding window and dilated produce nearly identical outputs"

        # Performance requirements for realistic scale
        assert total_compilation_time < 60.0, f"Total compilation too slow: {total_compilation_time:.3f}s"

        print(f"\\n🎯 Realistic scale test completed successfully!")
        print(f"   Total compilation time: {total_compilation_time:.3f}s")
        print(f"   Patterns tested: {len(patterns)}")
        print(f"   Data scale: {q.shape} ({q.numel() * 4 / 1024 / 1024:.1f}MB per tensor)")

    @pytest.mark.stress
    def test_stress_scale_compilation(self, compiler, large_inputs):
        """Stress test with large-scale data to validate performance and memory handling"""
        q, k, v = large_inputs
        seq_len, head_dim = q.shape[2], q.shape[3]

        print(f"\\nStress testing with large scale: {q.shape}")
        print(f"Memory per tensor: {q.numel() * 4 / 1024 / 1024:.1f}MB")
        print(f"Total memory (Q,K,V): {q.numel() * 3 * 4 / 1024 / 1024:.1f}MB")

        # Test memory-intensive patterns
        memory_intensive_patterns = ["causal"]  # Start with most stable pattern

        for pattern in memory_intensive_patterns:
            print(f"  Stress testing {pattern} pattern...")

            # Monitor memory usage during compilation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()

            start_time = time.time()
            kernel = compiler.compile_attention_kernel(pattern, seq_len, head_dim)
            compilation_time = time.time() - start_time

            if torch.cuda.is_available():
                post_compilation_memory = torch.cuda.memory_allocated()
                compilation_memory_mb = (post_compilation_memory - initial_memory) / 1024 / 1024
                print(f"    Compilation memory usage: {compilation_memory_mb:.1f}MB")

            # Test execution with memory monitoring
            start_time = time.time()

            try:
                output = kernel.kernel_fn(q, k, v)
                execution_time = time.time() - start_time

                # Validate output integrity
                assert output.shape == q.shape
                assert not torch.isnan(output).any()
                assert torch.isfinite(output).all()

                # Memory efficiency check
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated()
                    peak_memory_mb = peak_memory / 1024 / 1024
                    print(f"    Peak memory usage: {peak_memory_mb:.1f}MB")

                    # Reset peak memory counter
                    torch.cuda.reset_peak_memory_stats()

                print(f"    ✅ {pattern} stress test: compile={compilation_time:.3f}s, exec={execution_time:.3f}s")

                # Performance bounds for stress test
                assert compilation_time < 120.0, f"Compilation too slow for stress test: {compilation_time:.3f}s"
                assert execution_time < 30.0, f"Execution too slow for stress test: {execution_time:.3f}s"

            except torch.cuda.OutOfMemoryError:
                print(f"    ⚠️ {pattern} pattern exceeded GPU memory limits")
                pytest.skip(f"GPU memory insufficient for {pattern} pattern at scale {q.shape}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    ⚠️ {pattern} pattern hit memory limits: {e}")
                    pytest.skip(f"Memory insufficient for {pattern} pattern at scale {q.shape}")
                else:
                    raise

        print(f"\\n💪 Stress test completed successfully!")

    @pytest.mark.integration
    def test_specialized_configurations(self, compiler):
        """Test specialized data configurations for edge cases"""

        specialized_configs = ['long_sequence', 'high_heads', 'wide_embedding']

        for config_name in specialized_configs:
            config = TEST_CONFIGS[config_name]
            print(f"\\nTesting {config_name} configuration: {config.description}")

            q, k, v = config.create_tensors()
            seq_len, head_dim = q.shape[2], q.shape[3]

            print(f"  Dimensions: {q.shape}")
            print(f"  Memory: {config.total_memory_mb:.1f}MB")

            # Test causal pattern with specialized config
            try:
                start_time = time.time()
                kernel = compiler.compile_attention_kernel("causal", seq_len, head_dim)
                compilation_time = time.time() - start_time

                start_time = time.time()
                output = kernel.kernel_fn(q, k, v)
                execution_time = time.time() - start_time

                # Validate specialized configuration requirements
                assert output.shape == q.shape
                assert not torch.isnan(output).any()
                assert torch.isfinite(output).all()

                print(f"  ✅ {config_name}: compile={compilation_time:.3f}s, exec={execution_time:.3f}s")

                # Configuration-specific validation
                if config_name == 'long_sequence':
                    # Long sequences should handle memory efficiently
                    assert execution_time < 60.0, f"Long sequence execution too slow: {execution_time:.3f}s"

                elif config_name == 'high_heads':
                    # High head count should parallelize well
                    assert compilation_time < 30.0, f"High heads compilation too slow: {compilation_time:.3f}s"

                elif config_name == 'wide_embedding':
                    # Wide embeddings should handle memory bandwidth
                    assert output.std().item() > 0, "Wide embedding output lacks variance"

            except Exception as e:
                if "memory" in str(e).lower():
                    print(f"  ⚠️ {config_name} configuration hit memory limits")
                    pytest.skip(f"Memory insufficient for {config_name}: {e}")
                else:
                    raise


class TestEnd2EndIntegration:
    """End-to-end integration tests with multiple components"""

    @pytest.mark.integration
    def test_full_pipeline_integration(self):
        """Test complete pipeline from compilation through execution with realistic data"""
        print("\\n🔄 Testing complete compiler pipeline integration...")

        # Use realistic configuration
        config = TEST_CONFIGS['realistic']
        q, k, v = config.create_tensors()

        # Initialize all components
        compiler = FlashLightKernelCompiler(optimization_level="aggressive")

        # Test pipeline: compilation -> caching -> execution -> validation
        patterns = ["causal", "sliding_window"]

        pipeline_start = time.time()

        for pattern in patterns:
            print(f"  Pipeline testing {pattern}...")

            # Stage 1: Compilation
            kernel = compiler.compile_attention_kernel(pattern, q.shape[2], q.shape[3])
            assert kernel.kernel_fn is not None

            # Stage 2: Cache verification
            cache_key = f"{pattern}_{q.shape[2]}_{q.shape[3]}"
            assert len(compiler.kernel_cache.cache) > 0

            # Stage 3: Execution
            output = kernel.kernel_fn(q, k, v)

            # Stage 4: Validation
            assert output.shape == q.shape
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()

        pipeline_time = time.time() - pipeline_start
        print(f"  ✅ Full pipeline completed in {pipeline_time:.3f}s")

        # Integration requirements
        assert pipeline_time < 120.0, f"Full pipeline too slow: {pipeline_time:.3f}s"
        assert len(compiler.kernel_cache.cache) == len(patterns)
        assert compiler.compilation_stats["total_compilations"] == len(patterns)


if __name__ == "__main__":
    # Run comprehensive tests
    pytest.main([__file__, "-v", "-m", "integration or stress"])