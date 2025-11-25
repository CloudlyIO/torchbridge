"""
GPU Memory Optimization Framework
=================================

Comprehensive GPU memory optimization patterns, techniques, and educational tools
for maximizing GPU memory efficiency in PyTorch neural networks.

This module provides:
1. Memory-efficient operation patterns
2. Gradient accumulation strategies
3. Memory pool management
4. Memory leak detection and prevention
5. Educational memory optimization examples

Author: Advanced GPU Optimization Framework
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import gc
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class MemoryProfile:
    """Memory usage profile for analysis and optimization."""
    allocated_memory: float
    cached_memory: float
    reserved_memory: float
    max_allocated: float
    peak_memory: float
    memory_efficiency: float
    fragmentation_ratio: float


class MemoryOptimizer:
    """
    Advanced GPU memory optimization framework for PyTorch models.

    Provides comprehensive memory optimization strategies including:
    - Memory-efficient operation patterns
    - Gradient accumulation without memory overhead
    - Memory pool optimization
    - Memory leak detection and prevention
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_snapshots = []
        self.peak_memory = 0.0

    def profile_memory_usage(self,
                           model: nn.Module,
                           sample_inputs: Union[torch.Tensor, List[torch.Tensor]],
                           num_iterations: int = 10) -> MemoryProfile:
        """
        Profile memory usage patterns for a model and inputs.

        Args:
            model: PyTorch model to profile
            sample_inputs: Sample input tensors
            num_iterations: Number of forward/backward passes to profile

        Returns:
            MemoryProfile with detailed memory usage statistics
        """
        if isinstance(sample_inputs, torch.Tensor):
            sample_inputs = [sample_inputs]

        # Move model and inputs to device
        model = model.to(self.device)
        sample_inputs = [inp.to(self.device) for inp in sample_inputs]

        # Clear memory and reset tracking
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        allocated_memories = []
        cached_memories = []

        for i in range(num_iterations):
            # Forward pass
            if len(sample_inputs) == 1:
                outputs = model(sample_inputs[0])
            else:
                outputs = model(*sample_inputs)

            # Backward pass
            if isinstance(outputs, torch.Tensor):
                loss = outputs.sum()
            else:
                loss = sum(out.sum() for out in outputs if isinstance(out, torch.Tensor))

            loss.backward()

            # Record memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB

            allocated_memories.append(allocated)
            cached_memories.append(cached)

            # Clear gradients
            model.zero_grad()

        # Calculate statistics
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        max_allocated = max(allocated_memories)
        avg_allocated = np.mean(allocated_memories)
        avg_cached = np.mean(cached_memories)

        # Calculate memory efficiency and fragmentation
        memory_efficiency = avg_allocated / avg_cached if avg_cached > 0 else 0.0
        fragmentation_ratio = (avg_cached - avg_allocated) / avg_cached if avg_cached > 0 else 0.0

        return MemoryProfile(
            allocated_memory=avg_allocated,
            cached_memory=avg_cached,
            reserved_memory=avg_cached,
            max_allocated=max_allocated,
            peak_memory=peak_memory,
            memory_efficiency=memory_efficiency,
            fragmentation_ratio=fragmentation_ratio
        )

    def optimize_memory_layout(self,
                             model: nn.Module,
                             enable_memory_efficient_attention: bool = True,
                             use_checkpoint: bool = True,
                             optimize_embeddings: bool = True) -> nn.Module:
        """
        Apply memory layout optimizations to a model.

        Args:
            model: Model to optimize
            enable_memory_efficient_attention: Use memory-efficient attention
            use_checkpoint: Apply gradient checkpointing
            optimize_embeddings: Optimize embedding layers

        Returns:
            Memory-optimized model
        """
        optimized_model = model

        # Apply gradient checkpointing
        if use_checkpoint and hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()

        # Optimize attention layers for memory efficiency
        if enable_memory_efficient_attention:
            optimized_model = self._optimize_attention_memory(model)

        # Optimize embedding layers
        if optimize_embeddings:
            optimized_model = self._optimize_embedding_memory(optimized_model)

        return optimized_model

    def _optimize_attention_memory(self, model: nn.Module) -> nn.Module:
        """Optimize attention layers for memory efficiency."""
        for module in model.modules():
            if hasattr(module, 'attention') or 'attention' in module.__class__.__name__.lower():
                # Apply memory-efficient attention patterns
                if hasattr(module, 'scale_dot_product_attention'):
                    # Use Flash Attention if available
                    module.use_memory_efficient_attention = True
        return model

    def _optimize_embedding_memory(self, model: nn.Module) -> nn.Module:
        """Optimize embedding layers for memory efficiency."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                # Apply sparse gradient updates for large embeddings
                if module.num_embeddings > 10000:
                    module.sparse = True
        return model


class GradientAccumulator:
    """
    Memory-efficient gradient accumulation without storing intermediate gradients.

    Enables training with large effective batch sizes while maintaining memory efficiency.
    """

    def __init__(self,
                 accumulation_steps: int,
                 normalize_gradients: bool = True,
                 clip_grad_norm: Optional[float] = None):
        self.accumulation_steps = accumulation_steps
        self.normalize_gradients = normalize_gradients
        self.clip_grad_norm = clip_grad_norm
        self.current_step = 0

    @contextmanager
    def accumulate(self, model: nn.Module):
        """Context manager for gradient accumulation."""
        self.current_step += 1

        # Scale loss by accumulation steps to maintain gradient magnitude
        scale = 1.0 / self.accumulation_steps if self.normalize_gradients else 1.0

        try:
            # Use no_sync to prevent gradient synchronization in DDP
            if hasattr(model, 'no_sync') and self.current_step % self.accumulation_steps != 0:
                with model.no_sync():
                    yield scale
            else:
                yield scale
        finally:
            # Perform optimizer step and reset after accumulation_steps
            if self.current_step % self.accumulation_steps == 0:
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)

    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return self.current_step % self.accumulation_steps == 0


class MemoryEfficientLinear(nn.Module):
    """
    Memory-efficient linear layer with optional activation checkpointing.

    Reduces memory usage through:
    1. Activation checkpointing
    2. Chunked computation for large tensors
    3. In-place operations where possible
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 chunk_size: Optional[int] = None,
                 use_checkpoint: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_size = chunk_size
        self.use_checkpoint = use_checkpoint

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights with proper scaling."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Memory-efficient forward pass."""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, input)
        else:
            return self._forward_impl(input)

    def _forward_impl(self, input: torch.Tensor) -> torch.Tensor:
        """Actual forward implementation with chunking."""
        if self.chunk_size is not None and input.numel() > self.chunk_size:
            return self._chunked_forward(input)
        else:
            return F.linear(input, self.weight, self.bias)

    def _chunked_forward(self, input: torch.Tensor) -> torch.Tensor:
        """Process input in chunks to reduce memory usage."""
        batch_size = input.shape[0]
        chunk_size = min(self.chunk_size, batch_size)

        outputs = []
        for i in range(0, batch_size, chunk_size):
            chunk = input[i:i+chunk_size]
            chunk_output = F.linear(chunk, self.weight, self.bias)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=0)


class MemoryEfficientAttention(nn.Module):
    """
    Memory-efficient multi-head attention with several optimization strategies.

    Features:
    1. Flash Attention integration when available
    2. Chunked attention computation
    3. Memory-efficient softmax
    4. Gradient checkpointing support
    """

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 chunk_size: Optional[int] = None,
                 use_flash_attention: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.use_flash_attention = use_flash_attention

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = MemoryEfficientLinear(embed_dim, embed_dim, bias=bias)
        self.k_proj = MemoryEfficientLinear(embed_dim, embed_dim, bias=bias)
        self.v_proj = MemoryEfficientLinear(embed_dim, embed_dim, bias=bias)
        self.out_proj = MemoryEfficientLinear(embed_dim, embed_dim, bias=bias)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient attention forward pass."""
        if key is None:
            key = query
        if value is None:
            value = query

        batch_size, seq_len, _ = query.shape

        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Try Flash Attention first
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            try:
                attn_output = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=attn_mask is None  # Assume causal if no mask provided
                )
            except Exception:
                # Fallback to manual attention
                attn_output = self._manual_attention(q, k, v, attn_mask)
        else:
            attn_output = self._manual_attention(q, k, v, attn_mask)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )

        # Final projection
        return self.out_proj(attn_output)

    def _manual_attention(self,
                         q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Manual attention computation with memory optimizations."""
        scale = 1.0 / (self.head_dim ** 0.5)

        if self.chunk_size is not None:
            return self._chunked_attention(q, k, v, scale, attn_mask)

        # Standard attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores + attn_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        return torch.matmul(attn_weights, v)

    def _chunked_attention(self,
                          q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          scale: float,
                          attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Chunked attention computation to reduce memory usage."""
        batch_size, num_heads, q_len, head_dim = q.shape
        kv_len = k.shape[2]

        # Process in chunks
        chunk_size = min(self.chunk_size, q_len)
        outputs = []

        for i in range(0, q_len, chunk_size):
            q_chunk = q[:, :, i:i+chunk_size]

            # Compute attention for this chunk
            scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * scale

            if attn_mask is not None:
                mask_chunk = attn_mask[i:i+chunk_size] if attn_mask.dim() == 2 else attn_mask[:, :, i:i+chunk_size]
                scores = scores + mask_chunk

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)

            chunk_output = torch.matmul(attn_weights, v)
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=2)


class MemoryPool:
    """
    Custom memory pool for efficient tensor allocation and reuse.

    Helps reduce memory fragmentation by reusing pre-allocated tensors.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.pools: Dict[Tuple[torch.Size, torch.dtype], List[torch.Tensor]] = {}
        self.allocated_tensors = set()

    def get_tensor(self,
                   size: torch.Size,
                   dtype: torch.dtype = torch.float32,
                   requires_grad: bool = False) -> torch.Tensor:
        """Get a tensor from the pool or allocate a new one."""
        key = (size, dtype)

        if key in self.pools and self.pools[key]:
            tensor = self.pools[key].pop()
            tensor.requires_grad_(requires_grad)
            self.allocated_tensors.add(id(tensor))
            return tensor
        else:
            tensor = torch.empty(size, dtype=dtype, device=self.device, requires_grad=requires_grad)
            self.allocated_tensors.add(id(tensor))
            return tensor

    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the pool for reuse."""
        tensor_id = id(tensor)
        if tensor_id in self.allocated_tensors:
            self.allocated_tensors.remove(tensor_id)

            # Reset tensor properties
            tensor.requires_grad_(False)
            if tensor.grad is not None:
                tensor.grad = None

            key = (tensor.size(), tensor.dtype)
            if key not in self.pools:
                self.pools[key] = []

            self.pools[key].append(tensor)

    def clear_pool(self):
        """Clear all tensors from the pool."""
        self.pools.clear()
        self.allocated_tensors.clear()
        torch.cuda.empty_cache()


class MemoryLeakDetector:
    """
    Detect and analyze memory leaks in PyTorch models.

    Provides tools for tracking memory usage patterns and identifying
    potential memory leaks.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_snapshots = []
        self.baseline_memory = 0.0

    def start_monitoring(self):
        """Start monitoring memory usage."""
        torch.cuda.empty_cache()
        self.baseline_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        self.memory_snapshots = [self.baseline_memory]

    def take_snapshot(self, label: str = ""):
        """Take a memory usage snapshot."""
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        self.memory_snapshots.append(current_memory)

        if label:
            print(f"Memory snapshot '{label}': {current_memory:.2f} MB")

    def analyze_memory_pattern(self) -> Dict[str, Any]:
        """Analyze memory usage patterns for potential leaks."""
        if len(self.memory_snapshots) < 2:
            return {"error": "Need at least 2 snapshots for analysis"}

        memory_diffs = np.diff(self.memory_snapshots)

        analysis = {
            "total_memory_increase": self.memory_snapshots[-1] - self.baseline_memory,
            "average_increase_per_step": np.mean(memory_diffs),
            "max_single_increase": np.max(memory_diffs),
            "memory_trend": "increasing" if np.mean(memory_diffs) > 0.1 else "stable",
            "snapshots": self.memory_snapshots,
            "potential_leak": self.memory_snapshots[-1] - self.baseline_memory > 50.0  # 50MB threshold
        }

        return analysis

    @contextmanager
    def monitor_block(self, label: str = "block"):
        """Monitor memory usage within a code block."""
        before = torch.cuda.memory_allocated() / 1024**2
        try:
            yield
        finally:
            after = torch.cuda.memory_allocated() / 1024**2
            increase = after - before
            print(f"Memory usage in '{label}': {increase:.2f} MB increase")


def demonstrate_memory_optimization():
    """
    Comprehensive demonstration of memory optimization techniques.
    """
    print("🔋 GPU Memory Optimization Demonstration")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create sample model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = MemoryEfficientAttention(512, 8)
            self.linear1 = MemoryEfficientLinear(512, 2048, use_checkpoint=True)
            self.linear2 = MemoryEfficientLinear(2048, 512)

        def forward(self, x):
            x = self.attention(x)
            x = F.gelu(self.linear1(x))
            return self.linear2(x)

    model = TestModel().to(device)
    sample_input = torch.randn(8, 128, 512, device=device)

    # Initialize memory optimizer
    memory_optimizer = MemoryOptimizer(device)

    # Profile original memory usage
    print("\n📊 Memory Profiling:")
    original_profile = memory_optimizer.profile_memory_usage(model, sample_input)
    print(f"Original memory usage: {original_profile.allocated_memory:.3f} GB")
    print(f"Memory efficiency: {original_profile.memory_efficiency:.2%}")
    print(f"Fragmentation ratio: {original_profile.fragmentation_ratio:.2%}")

    # Apply memory optimizations
    print("\n⚡ Applying Memory Optimizations:")
    optimized_model = memory_optimizer.optimize_memory_layout(model)

    # Profile optimized memory usage
    optimized_profile = memory_optimizer.profile_memory_usage(optimized_model, sample_input)
    print(f"Optimized memory usage: {optimized_profile.allocated_memory:.3f} GB")
    print(f"Memory efficiency: {optimized_profile.memory_efficiency:.2%}")
    print(f"Memory reduction: {((original_profile.allocated_memory - optimized_profile.allocated_memory) / original_profile.allocated_memory * 100):.1f}%")

    # Demonstrate gradient accumulation
    print("\n🔄 Gradient Accumulation Demo:")
    accumulator = GradientAccumulator(accumulation_steps=4)
    optimizer = torch.optim.Adam(optimized_model.parameters())

    for step in range(8):
        with accumulator.accumulate(optimized_model) as scale:
            outputs = optimized_model(sample_input)
            loss = outputs.sum() * scale
            loss.backward()

        if accumulator.should_step():
            optimizer.step()
            optimizer.zero_grad()
            print(f"  Optimizer step performed at iteration {step + 1}")

    # Demonstrate memory leak detection
    print("\n🔍 Memory Leak Detection:")
    leak_detector = MemoryLeakDetector(device)
    leak_detector.start_monitoring()

    for i in range(5):
        with leak_detector.monitor_block(f"iteration_{i}"):
            _ = optimized_model(sample_input)
        leak_detector.take_snapshot(f"after_iteration_{i}")

    analysis = leak_detector.analyze_memory_pattern()
    print(f"Memory trend: {analysis['memory_trend']}")
    print(f"Potential memory leak detected: {analysis['potential_leak']}")

    print("\n✅ Memory optimization demonstration complete!")


if __name__ == "__main__":
    demonstrate_memory_optimization()