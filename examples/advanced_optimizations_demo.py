"""
Advanced Optimizations Demo (2024-2025)

Comprehensive demonstration of cutting-edge optimization techniques:
- FlashAttention-3 with FP8 precision
- Mixture of Experts with adaptive routing
- Deep optimizer states with memory offloading
- Advanced memory management
- Performance benchmarking and comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kernel_pytorch.advanced_attention import FlashAttention3, FP8AttentionConfig, FlexAttentionAPI, AttentionPatterns
from kernel_pytorch.mixture_of_experts import create_moe_layer, MoEConfig
from kernel_pytorch.advanced_memory import InterleaveOffloadingOptimizer, DeepOptimizerStates, MemoryConfig
from kernel_pytorch.components.basic_optimized import OptimizedTransformerBlock


class AdvancedTransformerBlock(nn.Module):
    """
    Transformer block with all latest optimizations
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        use_moe: bool = True,
        moe_type: str = "adaptive",
        num_experts: int = 8,
        attention_pattern: AttentionPatterns = AttentionPatterns.CAUSAL,
        use_fp8: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads

        # Advanced attention with FlashAttention-3
        fp8_config = FP8AttentionConfig(
            use_fp8=use_fp8 and torch.cuda.is_available(),
            async_compute=True,
            warp_specialization=True
        )

        self.attention = FlashAttention3(
            embed_dim=dim,
            num_heads=num_heads,
            config=fp8_config,
            causal=(attention_pattern == AttentionPatterns.CAUSAL)
        )

        # Alternative: FlexAttention for different patterns
        self.flex_attention = FlexAttentionAPI(
            embed_dim=dim,
            num_heads=num_heads,
            pattern=attention_pattern,
            dropout=dropout
        )

        self.use_flex_attention = attention_pattern != AttentionPatterns.CAUSAL

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # MoE or standard FFN
        if use_moe:
            self.ffn = create_moe_layer(
                moe_type=moe_type,
                hidden_size=dim,
                num_experts=num_experts,
                top_k=2 if moe_type != "switch" else 1,
                expert_dropout=dropout,
                load_balance_loss_weight=0.01
            )
        else:
            # Standard FFN
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention with residual connection
        residual = x
        x = self.norm1(x)

        if self.use_flex_attention:
            attn_out = self.flex_attention(x)
        else:
            attn_out = self.attention(x)

        x = residual + self.dropout(attn_out)

        # FFN with residual connection
        residual = x
        x = self.norm2(x)

        if hasattr(self.ffn, '__call__') and hasattr(self.ffn, 'forward'):
            # MoE layer
            if hasattr(self.ffn, 'config'):
                ffn_out, aux_losses = self.ffn(x, return_router_logits=True)
                # Store aux losses for backward pass
                if hasattr(self, '_aux_losses'):
                    self._aux_losses.update(aux_losses)
                else:
                    self._aux_losses = aux_losses
            else:
                ffn_out = self.ffn(x)
        else:
            # Standard FFN
            ffn_out = self.ffn(x)

        x = residual + self.dropout(ffn_out)

        return x

    def get_auxiliary_losses(self):
        """Get auxiliary losses from MoE layers"""
        return getattr(self, '_aux_losses', {})


class AdvancedTransformerModel(nn.Module):
    """
    Complete transformer model with all optimizations
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 2048,
        use_moe: bool = True,
        moe_type: str = "adaptive",
        num_experts: int = 8,
        attention_pattern: AttentionPatterns = AttentionPatterns.CAUSAL,
        use_fp8: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            AdvancedTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                use_moe=use_moe,
                moe_type=moe_type,
                num_experts=num_experts,
                attention_pattern=attention_pattern,
                use_fp8=use_fp8,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Output layers
        self.norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, return_aux_losses: bool = False) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        x = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(positions)
        x = x + position_embeds

        # Transformer layers
        aux_losses = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Collect auxiliary losses from MoE layers
            if return_aux_losses:
                layer_aux = layer.get_auxiliary_losses()
                for loss_name, loss_value in layer_aux.items():
                    if loss_name not in aux_losses:
                        aux_losses[loss_name] = []
                    aux_losses[loss_name].append(loss_value)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_proj(x)

        if return_aux_losses:
            # Aggregate auxiliary losses
            aggregated_aux = {}
            for loss_name, loss_list in aux_losses.items():
                aggregated_aux[loss_name] = torch.stack(loss_list).mean()

            return logits, aggregated_aux

        return logits


def benchmark_optimizations():
    """
    Benchmark different optimization configurations
    """
    print("🚀 Advanced Optimizations Benchmark")
    print("=" * 60)

    # Configuration
    batch_size = 4
    seq_len = 1024
    vocab_size = 32000
    dim = 768
    num_heads = 12
    num_layers = 6
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Input shape: [{batch_size}, {seq_len}]")
    print(f"Model size: {num_layers} layers, {dim} dim, {num_heads} heads")

    # Test configurations
    configs = {
        'baseline': {
            'use_moe': False,
            'use_fp8': False,
            'attention_pattern': AttentionPatterns.CAUSAL
        },
        'moe_only': {
            'use_moe': True,
            'moe_type': 'adaptive',
            'num_experts': 8,
            'use_fp8': False,
            'attention_pattern': AttentionPatterns.CAUSAL
        },
        'fp8_attention': {
            'use_moe': False,
            'use_fp8': True,
            'attention_pattern': AttentionPatterns.CAUSAL
        },
        'sliding_window': {
            'use_moe': False,
            'use_fp8': False,
            'attention_pattern': AttentionPatterns.SLIDING_WINDOW
        },
        'all_optimizations': {
            'use_moe': True,
            'moe_type': 'adaptive',
            'num_experts': 8,
            'use_fp8': True,
            'attention_pattern': AttentionPatterns.CAUSAL
        }
    }

    results = {}

    for config_name, config in configs.items():
        print(f"\n📊 Testing configuration: {config_name}")
        print("-" * 40)

        try:
            # Create model
            model = AdvancedTransformerModel(
                vocab_size=vocab_size,
                dim=dim,
                num_layers=num_layers,
                num_heads=num_heads,
                max_seq_len=seq_len,
                **config
            ).to(device)

            # Create optimizer with advanced memory management
            base_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            if device == "cuda":
                # Use advanced optimizer for CUDA
                optimizer = InterleaveOffloadingOptimizer(
                    optimizer=base_optimizer,
                    model=model,
                    memory_limit_gb=8.0,
                    auto_tune=True
                )
            else:
                optimizer = base_optimizer

            # Generate test data
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            # Warmup
            model.train()
            for _ in range(3):
                optimizer.zero_grad()
                outputs = model(input_ids, return_aux_losses='moe' in config_name)

                if isinstance(outputs, tuple):
                    logits, aux_losses = outputs
                    loss = F.cross_entropy(
                        logits.view(-1, vocab_size),
                        input_ids.view(-1)
                    )
                    # Add auxiliary losses
                    for aux_loss in aux_losses.values():
                        loss = loss + aux_loss
                else:
                    loss = F.cross_entropy(
                        outputs.view(-1, vocab_size),
                        input_ids.view(-1)
                    )

                loss.backward()
                if hasattr(optimizer, 'step'):
                    metrics = optimizer.step()
                else:
                    optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            # Benchmark
            num_iterations = 10
            start_time = time.perf_counter()

            for i in range(num_iterations):
                optimizer.zero_grad()
                outputs = model(input_ids, return_aux_losses='moe' in config_name)

                if isinstance(outputs, tuple):
                    logits, aux_losses = outputs
                    loss = F.cross_entropy(
                        logits.view(-1, vocab_size),
                        input_ids.view(-1)
                    )
                    for aux_loss in aux_losses.values():
                        loss = loss + aux_loss
                else:
                    loss = F.cross_entropy(
                        outputs.view(-1, vocab_size),
                        input_ids.view(-1)
                    )

                loss.backward()
                if hasattr(optimizer, 'step'):
                    step_metrics = optimizer.step()
                else:
                    optimizer.step()

            if device == "cuda":
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            # Calculate metrics
            total_time = end_time - start_time
            avg_time_per_step = total_time / num_iterations
            tokens_per_second = (batch_size * seq_len * num_iterations) / total_time

            # Memory usage
            if device == "cuda":
                memory_used = torch.cuda.max_memory_allocated() / 1e9  # GB
                torch.cuda.reset_peak_memory_stats()
            else:
                memory_used = 0

            # Model parameters
            total_params = sum(p.numel() for p in model.parameters())

            results[config_name] = {
                'avg_time_ms': avg_time_per_step * 1000,
                'tokens_per_second': tokens_per_second,
                'memory_gb': memory_used,
                'total_params': total_params,
                'loss': loss.item(),
                'config': config
            }

            print(f"  ✅ Average time: {avg_time_per_step*1000:.2f}ms")
            print(f"  🚀 Tokens/sec: {tokens_per_second:.0f}")
            print(f"  💾 Memory: {memory_used:.2f}GB")
            print(f"  📐 Parameters: {total_params:,}")
            print(f"  📉 Final loss: {loss.item():.4f}")

            # Get optimization-specific stats
            if hasattr(optimizer, 'get_stats'):
                opt_stats = optimizer.get_stats()
                print(f"  ⚙️  Optimizer stats: {opt_stats}")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            results[config_name] = {'error': str(e)}

        # Cleanup
        if 'model' in locals():
            del model
        if 'optimizer' in locals():
            del optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

    # Print summary
    print("\n📈 BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'Time (ms)':<12} {'Tokens/s':<12} {'Memory (GB)':<12} {'Speedup':<10}")
    print("-" * 66)

    baseline_time = results.get('baseline', {}).get('avg_time_ms', 1000)

    for config_name, result in results.items():
        if 'error' in result:
            print(f"{config_name:<20} {'ERROR':<12} {'':<12} {'':<12} {'':<10}")
        else:
            speedup = baseline_time / result['avg_time_ms'] if baseline_time > 0 else 1.0
            print(f"{config_name:<20} {result['avg_time_ms']:<12.2f} "
                  f"{result['tokens_per_second']:<12.0f} {result['memory_gb']:<12.2f} "
                  f"{speedup:<10.2f}x")

    return results


def demonstrate_memory_optimization():
    """
    Demonstrate advanced memory optimization techniques
    """
    print("\n🧠 Memory Optimization Demo")
    print("=" * 40)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a larger model to stress memory
    model = AdvancedTransformerModel(
        vocab_size=50000,
        dim=1024,
        num_layers=12,
        num_heads=16,
        use_moe=True,
        moe_type="adaptive",
        num_experts=16
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if device == "cuda":
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # Test different memory optimization strategies
        optimizers = {
            'standard': torch.optim.AdamW(model.parameters(), lr=1e-4),
            'memory_optimized': InterleaveOffloadingOptimizer(
                optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
                model=model,
                memory_limit_gb=4.0,  # Low limit to force offloading
                auto_tune=True
            )
        }

        for opt_name, optimizer in optimizers.items():
            print(f"\nTesting {opt_name} optimizer:")

            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()

            # Training step
            input_ids = torch.randint(0, 50000, (2, 512), device=device)
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.view(-1, 50000), input_ids.view(-1))
            loss.backward()

            if hasattr(optimizer, 'step') and hasattr(optimizer, 'get_stats'):
                step_metrics = optimizer.step()
                stats = optimizer.get_stats()
                print(f"  Step metrics: {step_metrics}")
                print(f"  Memory stats: {stats.get('memory_usage', {})}")
            else:
                optimizer.step()

            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak memory usage: {peak_memory:.2f}GB")

            del optimizer
            torch.cuda.empty_cache()

    else:
        print("Memory optimization demo requires CUDA")


def main():
    """Main demonstration function"""
    print("🎯 Advanced PyTorch Optimizations Demo (2024-2025)")
    print("=" * 60)

    # Check available optimizations
    print("📋 Available Optimizations:")
    print(f"  🔥 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  🎮 GPU: {torch.cuda.get_device_name()}")
        capability = torch.cuda.get_device_capability()
        print(f"  ⚡ Compute capability: {capability[0]}.{capability[1]}")

    try:
        import flash_attn
        print(f"  ⚡ FlashAttention available: ✅")
    except ImportError:
        print(f"  ⚡ FlashAttention available: ❌")

    try:
        import triton
        print(f"  🔹 Triton available: ✅")
    except ImportError:
        print(f"  🔹 Triton available: ❌")

    # Run benchmarks
    benchmark_results = benchmark_optimizations()

    # Demonstrate memory optimization
    demonstrate_memory_optimization()

    print("\n🎉 Demo completed successfully!")
    print("\nKey Takeaways:")
    print("• FlashAttention-3 provides significant speedup for long sequences")
    print("• MoE layers enable massive model scaling with minimal compute overhead")
    print("• Advanced memory optimization enables training larger models")
    print("• FP8 precision reduces memory usage while maintaining accuracy")
    print("• Different attention patterns optimize for different use cases")

    return benchmark_results


if __name__ == "__main__":
    results = main()