#!/usr/bin/env python3
"""
Test Configuration Framework

Defines realistic test data configurations for different test categories
to ensure proper test efficacy while maintaining performance.
"""

from dataclasses import dataclass

import torch


@dataclass
class TestDataConfig:
    """Configuration for test data dimensions"""
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    description: str
    target_time: str
    use_case: str

    @property
    def memory_mb(self) -> float:
        """Estimated memory usage per tensor in MB"""
        elements = self.batch_size * self.num_heads * self.seq_len * self.head_dim
        return (elements * 4) / (1024 * 1024)  # 4 bytes per float32

    @property
    def total_memory_mb(self) -> float:
        """Total memory for Q, K, V tensors"""
        return self.memory_mb * 3

    def create_tensors(self, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create Q, K, V tensors with this configuration"""
        device = torch.device(device)
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=device)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=device)
        v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim, device=device)
        return q, k, v


# Test Data Configuration Hierarchy
TEST_CONFIGS = {

    # === UNIT TEST LEVEL ===
    'micro': TestDataConfig(
        batch_size=1, num_heads=2, seq_len=32, head_dim=16,
        description="Micro-scale for algorithm correctness",
        target_time="< 0.1s",
        use_case="Unit tests, algorithm validation"
    ),

    'small': TestDataConfig(
        batch_size=1, num_heads=4, seq_len=64, head_dim=32,
        description="Small-scale for functional testing",
        target_time="< 0.5s",
        use_case="Basic functionality, edge case testing"
    ),

    # === INTEGRATION TEST LEVEL ===
    'medium': TestDataConfig(
        batch_size=2, num_heads=8, seq_len=128, head_dim=64,
        description="Medium-scale for integration testing",
        target_time="< 5s",
        use_case="Integration tests, pattern verification"
    ),

    'realistic': TestDataConfig(
        batch_size=2, num_heads=8, seq_len=512, head_dim=64,
        description="Production-realistic scale",
        target_time="< 30s",
        use_case="End-to-end integration, realistic scenarios"
    ),

    # === STRESS TEST LEVEL ===
    'large': TestDataConfig(
        batch_size=4, num_heads=16, seq_len=1024, head_dim=64,
        description="Large-scale for performance testing",
        target_time="< 60s",
        use_case="Performance validation, stress testing"
    ),

    'xlarge': TestDataConfig(
        batch_size=8, num_heads=32, seq_len=2048, head_dim=128,
        description="Extra-large for stress testing",
        target_time="< 300s",
        use_case="Extreme stress testing, memory limits"
    ),

    # === SPECIALIZED CONFIGURATIONS ===
    'long_sequence': TestDataConfig(
        batch_size=1, num_heads=8, seq_len=4096, head_dim=64,
        description="Long sequence testing",
        target_time="< 120s",
        use_case="Long context handling, memory efficiency"
    ),

    'high_heads': TestDataConfig(
        batch_size=2, num_heads=64, seq_len=256, head_dim=64,
        description="Many attention heads testing",
        target_time="< 60s",
        use_case="Multi-head scaling, parallelization"
    ),

    'wide_embedding': TestDataConfig(
        batch_size=2, num_heads=8, seq_len=512, head_dim=256,
        description="Wide embedding dimension testing",
        target_time="< 60s",
        use_case="Large model compatibility, memory bandwidth"
    )
}


def get_config_for_test_type(test_type: str) -> TestDataConfig:
    """Get appropriate configuration for test type"""
    config_map = {
        'unit': 'small',
        'integration': 'realistic',
        'stress': 'large',
        'micro': 'micro',
        'performance': 'xlarge'
    }

    config_name = config_map.get(test_type, 'medium')
    return TEST_CONFIGS[config_name]


def print_config_summary():
    """Print summary of all test configurations"""
    print(" Test Data Configuration Summary")
    print("=" * 80)

    print(f"{'Config':<12} {'Dims (B×H×S×D)':<20} {'Memory':<10} {'Target':<10} {'Use Case'}")
    print("-" * 80)

    for name, config in TEST_CONFIGS.items():
        dims = f"{config.batch_size}×{config.num_heads}×{config.seq_len}×{config.head_dim}"
        memory = f"{config.total_memory_mb:.1f}MB"

        print(f"{name:<12} {dims:<20} {memory:<10} {config.target_time:<10} {config.use_case}")

    print()
    print(" Test Category Recommendations:")
    print("• Unit tests:        micro, small configs")
    print("• Integration tests: medium, realistic configs")
    print("• Stress tests:      large, xlarge configs")
    print("• Specialized:       long_sequence, high_heads, wide_embedding")


if __name__ == "__main__":
    print_config_summary()
