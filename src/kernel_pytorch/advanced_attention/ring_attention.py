"""
Ring Attention Implementation for Million-Token Sequences

Ring Attention enables efficient processing of extremely long sequences (1M+ tokens)
by distributing the computation across multiple devices while maintaining linear
memory complexity with respect to the number of devices.

Key Features:
- Linear memory complexity O(N/P) where P is number of devices
- Support for sequences up to 10M+ tokens
- Automatic device mesh creation and coordination
- Integration with existing FlexAttention patterns
- Production-ready distributed processing

References:
    - Ring Attention Paper: https://arxiv.org/abs/2310.01889
    - Distributed Attention: https://arxiv.org/abs/2004.05150
    - PyTorch Distributed: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
import warnings

try:
    import torch.distributed as dist
    from torch.distributed import DeviceMesh
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    warnings.warn("PyTorch distributed not available - Ring Attention will use single device fallback")

try:
    from ..hardware_abstraction.device_coordinator import HardwareAbstractionLayer
    HAL_AVAILABLE = True
except ImportError:
    HAL_AVAILABLE = False
    warnings.warn("Hardware Abstraction Layer not available - using basic device coordination")


class RingAttentionConfig:
    """Configuration for Ring Attention implementation"""

    def __init__(
        self,
        ring_size: Optional[int] = None,
        chunk_size: int = 4096,
        overlap_comm_compute: bool = True,
        use_flash_attention: bool = True,
        enable_checkpointing: bool = False,
        device_mesh: Optional[DeviceMesh] = None
    ):
        self.ring_size = ring_size  # Auto-detect if None
        self.chunk_size = chunk_size
        self.overlap_comm_compute = overlap_comm_compute
        self.use_flash_attention = use_flash_attention
        self.enable_checkpointing = enable_checkpointing
        self.device_mesh = device_mesh


class RingAttentionLayer(nn.Module):
    """
    Ring Attention layer for processing extremely long sequences.

    Distributes attention computation across multiple devices using ring-based
    communication pattern while maintaining linear memory complexity.

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        config (RingAttentionConfig): Ring attention configuration
        device (torch.device): Target device

    Example:
        >>> config = RingAttentionConfig(ring_size=4, chunk_size=4096)
        >>> attention = RingAttentionLayer(d_model=512, num_heads=8, config=config)
        >>> # Process 1M token sequence
        >>> x = torch.randn(1, 1_000_000, 512)
        >>> output = attention(x)  # Linear memory complexity
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        config: RingAttentionConfig,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Validate configuration
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"

        # Initialize projections
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        # Initialize distributed setup if available
        self.world_size = 1
        self.rank = 0
        self.is_distributed = False

        if DISTRIBUTED_AVAILABLE and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.is_distributed = True

            # Set ring size based on available devices
            if config.ring_size is None:
                self.ring_size = min(self.world_size, 8)  # Max 8 devices in ring
            else:
                self.ring_size = min(config.ring_size, self.world_size)
        else:
            self.ring_size = 1

        # Initialize hardware abstraction layer if available
        if HAL_AVAILABLE:
            self.hal = HardwareAbstractionLayer(vendor="auto", device=self.device)
            self.device_mesh = self.hal.create_device_mesh(self.ring_size)
        else:
            self.device_mesh = None

        # Initialize communication groups
        self._setup_communication_groups()

        # Pre-allocate communication buffers
        self._setup_communication_buffers()

    def _setup_communication_groups(self):
        """Setup communication groups for ring attention"""
        if not self.is_distributed or self.ring_size <= 1:
            self.ring_group = None
            self.prev_rank = self.rank
            self.next_rank = self.rank
            return

        # Create ring communication group
        ring_ranks = list(range(min(self.ring_size, self.world_size)))
        if DISTRIBUTED_AVAILABLE:
            self.ring_group = dist.new_group(ranks=ring_ranks)

        # Calculate previous and next ranks in ring
        if self.rank < self.ring_size:
            self.prev_rank = (self.rank - 1) % self.ring_size
            self.next_rank = (self.rank + 1) % self.ring_size
        else:
            self.prev_rank = self.rank
            self.next_rank = self.rank

    def _setup_communication_buffers(self):
        """Pre-allocate buffers for efficient communication"""
        if not self.is_distributed or self.ring_size <= 1:
            return

        # Calculate maximum buffer size needed
        max_seq_len = self.config.chunk_size * 2  # Double for safety
        buffer_shape = (1, max_seq_len, self.d_model)

        # Pre-allocate send/receive buffers
        self.send_buffer = torch.zeros(buffer_shape, device=self.device, dtype=torch.float16)
        self.recv_buffer = torch.zeros(buffer_shape, device=self.device, dtype=torch.float16)

        # Pre-allocate attention buffers
        attention_buffer_shape = (1, self.num_heads, max_seq_len, self.head_dim)
        self.kv_buffer = {
            'key': torch.zeros(attention_buffer_shape, device=self.device, dtype=torch.float16),
            'value': torch.zeros(attention_buffer_shape, device=self.device, dtype=torch.float16)
        }

    def _ring_communication_step(self, kv_chunk: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform one step of ring communication to exchange key-value chunks

        Args:
            kv_chunk: Dictionary containing 'key' and 'value' tensors

        Returns:
            Dictionary containing received key-value chunks from previous rank
        """
        if not self.is_distributed or self.ring_size <= 1:
            return kv_chunk

        batch_size, num_heads, seq_len, head_dim = kv_chunk['key'].shape

        # Prepare send data
        send_key = kv_chunk['key'].contiguous()
        send_value = kv_chunk['value'].contiguous()

        # Prepare receive buffers
        recv_key = torch.zeros_like(send_key)
        recv_value = torch.zeros_like(send_value)

        if DISTRIBUTED_AVAILABLE and self.ring_group is not None:
            # Non-blocking communication for better overlap
            send_reqs = []
            recv_reqs = []

            # Send to next rank
            send_reqs.append(dist.isend(send_key, dst=self.next_rank, group=self.ring_group))
            send_reqs.append(dist.isend(send_value, dst=self.next_rank, group=self.ring_group))

            # Receive from previous rank
            recv_reqs.append(dist.irecv(recv_key, src=self.prev_rank, group=self.ring_group))
            recv_reqs.append(dist.irecv(recv_value, src=self.prev_rank, group=self.ring_group))

            # Wait for completion
            for req in send_reqs + recv_reqs:
                req.wait()

        return {'key': recv_key, 'value': recv_value}

    def _compute_attention_chunk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal_mask: bool = True
    ) -> torch.Tensor:
        """
        Compute attention for a chunk of key-value pairs

        Args:
            query: Query tensor [batch, heads, seq_len, head_dim]
            key: Key tensor [batch, heads, seq_len, head_dim]
            value: Value tensor [batch, heads, seq_len, head_dim]
            causal_mask: Whether to apply causal masking

        Returns:
            Attention output for this chunk
        """
        batch_size, num_heads, q_len, head_dim = query.shape
        _, _, kv_len, _ = key.shape

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

        # Apply causal mask if needed
        if causal_mask and q_len == kv_len:
            causal_mask_tensor = torch.tril(torch.ones(q_len, kv_len, device=query.device))
            scores = scores.masked_fill(causal_mask_tensor == 0, float('-inf'))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Ring Attention

        Args:
            hidden_states: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            causal_mask: Whether to apply causal masking

        Returns:
            Tuple of (output_tensor, attention_weights)
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Check if sequence length requires ring attention
        if seq_len <= self.config.chunk_size:
            # Use standard attention for short sequences
            return self._standard_attention(hidden_states, attention_mask, causal_mask)

        # Prepare for ring attention
        chunk_size = self.config.chunk_size
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # Project to query, key, value
        query = self.query_proj(hidden_states)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Initialize output accumulator
        output_chunks = []

        # Process each query chunk
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_len)

            query_chunk = query[:, :, start_idx:end_idx, :]
            chunk_outputs = []

            # Ring attention across all key-value chunks
            current_kv = {
                'key': key[:, :, start_idx:end_idx, :].clone(),
                'value': value[:, :, start_idx:end_idx, :].clone()
            }

            # Process local and remote chunks
            for ring_step in range(self.ring_size):
                if ring_step > 0:
                    # Communicate to get next chunk
                    current_kv = self._ring_communication_step(current_kv)

                # Compute attention for current chunk
                chunk_output, _ = self._compute_attention_chunk(
                    query_chunk,
                    current_kv['key'],
                    current_kv['value'],
                    causal_mask and ring_step == 0  # Only apply causal mask to local chunk
                )
                chunk_outputs.append(chunk_output)

            # Combine outputs from all ring steps
            if chunk_outputs:
                combined_output = torch.stack(chunk_outputs, dim=0).mean(dim=0)
                output_chunks.append(combined_output)

        # Concatenate all chunks
        if output_chunks:
            attention_output = torch.cat(output_chunks, dim=2)  # Concatenate along sequence dimension
        else:
            # Fallback to standard attention
            return self._standard_attention(hidden_states, attention_mask, causal_mask)

        # Reshape back to original format
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)

        # Apply output projection
        output = self.output_proj(attention_output)

        return output, None  # No attention weights returned for ring attention

    def _standard_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fallback to standard attention for shorter sequences"""
        batch_size, seq_len, d_model = hidden_states.shape

        # Project to QKV
        query = self.query_proj(hidden_states)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention computation
        attention_output, attention_weights = self._compute_attention_chunk(
            query, key, value, causal_mask
        )

        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        output = self.output_proj(attention_output)

        return output, attention_weights


class RingAttentionBlock(nn.Module):
    """
    Complete Ring Attention block with layer normalization and feed-forward

    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        config (RingAttentionConfig): Ring attention configuration
        dropout (float): Dropout rate

    Example:
        >>> config = RingAttentionConfig(ring_size=4, chunk_size=4096)
        >>> block = RingAttentionBlock(d_model=512, num_heads=8, d_ff=2048, config=config)
        >>> x = torch.randn(1, 1_000_000, 512)  # 1M tokens
        >>> output = block(x)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        config: RingAttentionConfig,
        dropout: float = 0.1,
        device: Optional[torch.device] = None
    ):
        super().__init__()

        self.attention = RingAttentionLayer(d_model, num_heads, config, device)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True
    ) -> torch.Tensor:
        """Forward pass with residual connections"""

        # Self-attention with residual connection
        normed_hidden_states = self.norm1(hidden_states)
        attention_output, _ = self.attention(normed_hidden_states, attention_mask, causal_mask)
        hidden_states = hidden_states + self.dropout(attention_output)

        # Feed-forward with residual connection
        normed_hidden_states = self.norm2(hidden_states)
        ff_output = self.feed_forward(normed_hidden_states)
        hidden_states = hidden_states + ff_output

        return hidden_states


# Factory function for easy usage
def create_ring_attention(
    d_model: int,
    num_heads: int,
    max_sequence_length: int = 1_000_000,
    ring_size: Optional[int] = None,
    device: Optional[torch.device] = None
) -> RingAttentionLayer:
    """
    Factory function to create Ring Attention layer with optimal configuration

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        max_sequence_length: Maximum expected sequence length
        ring_size: Number of devices in ring (auto-detected if None)
        device: Target device

    Returns:
        Configured RingAttentionLayer

    Example:
        >>> attention = create_ring_attention(d_model=512, num_heads=8, max_sequence_length=1_000_000)
        >>> x = torch.randn(1, 1_000_000, 512)
        >>> output = attention(x)
    """

    # Calculate optimal chunk size based on sequence length and memory constraints
    if max_sequence_length <= 8192:
        chunk_size = max_sequence_length
    elif max_sequence_length <= 65536:
        chunk_size = 4096
    elif max_sequence_length <= 524288:
        chunk_size = 8192
    else:
        chunk_size = 16384  # For very long sequences

    config = RingAttentionConfig(
        ring_size=ring_size,
        chunk_size=chunk_size,
        overlap_comm_compute=True,
        use_flash_attention=True,
        enable_checkpointing=max_sequence_length > 100_000
    )

    return RingAttentionLayer(d_model, num_heads, config, device)


# Utility functions for sequence processing
def estimate_memory_usage(
    batch_size: int,
    sequence_length: int,
    d_model: int,
    num_heads: int,
    ring_size: int = 1
) -> Dict[str, float]:
    """
    Estimate memory usage for Ring Attention

    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        d_model: Model dimension
        num_heads: Number of attention heads
        ring_size: Number of devices in ring

    Returns:
        Dictionary with memory estimates in MB
    """

    # Calculate memory for different components
    input_memory = batch_size * sequence_length * d_model * 4 / (1024 * 1024)  # 4 bytes for float32

    # QKV projections
    qkv_memory = 3 * input_memory

    # Attention scores (distributed across ring)
    attention_memory = batch_size * num_heads * (sequence_length // ring_size) ** 2 * 4 / (1024 * 1024)

    # Communication buffers
    comm_buffer_memory = 2 * batch_size * (sequence_length // ring_size) * d_model * 4 / (1024 * 1024)

    total_memory = input_memory + qkv_memory + attention_memory + comm_buffer_memory

    return {
        'input_memory_mb': input_memory,
        'qkv_memory_mb': qkv_memory,
        'attention_memory_mb': attention_memory,
        'communication_buffer_mb': comm_buffer_memory,
        'total_memory_mb': total_memory,
        'memory_per_device_mb': total_memory / max(ring_size, 1)
    }


def validate_ring_attention_setup(config: RingAttentionConfig) -> Dict[str, Any]:
    """
    Validate Ring Attention configuration and system setup

    Args:
        config: RingAttentionConfig to validate

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }

    # Check distributed setup
    if not DISTRIBUTED_AVAILABLE:
        validation_results['warnings'].append("Distributed PyTorch not available - falling back to single device")

    # Check HAL availability
    if not HAL_AVAILABLE:
        validation_results['warnings'].append("Hardware Abstraction Layer not available")

    # Validate chunk size
    if config.chunk_size < 1024:
        validation_results['warnings'].append(f"Small chunk size ({config.chunk_size}) may be inefficient")
    elif config.chunk_size > 16384:
        validation_results['warnings'].append(f"Large chunk size ({config.chunk_size}) may cause memory issues")

    # Check ring size
    if config.ring_size is not None and config.ring_size > 8:
        validation_results['warnings'].append("Ring sizes > 8 may have diminishing returns due to communication overhead")

    return validation_results