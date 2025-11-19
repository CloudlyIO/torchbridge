"""
Expert Network Implementations

Various expert architectures for MoE systems:
- Feed-forward experts (standard MLP)
- Convolutional experts for spatial processing
- Attention-based experts for sequence modeling
- Parameter-efficient experts for memory optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math


class FeedForwardExpert(nn.Module):
    """
    Standard feed-forward expert network

    A simple MLP expert that can be used as the building block
    for MoE layers. Supports various activation functions and
    dropout patterns.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        dropout: float = 0.1,
        activation_fn: str = "relu",
        use_bias: bool = True,
        use_layernorm: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        output_size = output_size or input_size
        factory_kwargs = {'device': device, 'dtype': dtype}

        # First linear layer
        self.fc1 = nn.Linear(input_size, hidden_size, bias=use_bias, **factory_kwargs)

        # Activation function
        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "swish" or activation_fn == "silu":
            self.activation = nn.SiLU()
        elif activation_fn == "mish":
            self.activation = nn.Mish()
        elif activation_fn == "geglu":
            # GLU variant with GELU
            self.activation = None
            self.fc1 = nn.Linear(input_size, hidden_size * 2, bias=use_bias, **factory_kwargs)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        # Optional layer normalization
        self.layernorm = nn.LayerNorm(hidden_size, **factory_kwargs) if use_layernorm else None

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Second linear layer
        self.fc2 = nn.Linear(hidden_size, output_size, bias=use_bias, **factory_kwargs)

        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert"""
        # First linear transformation
        hidden = self.fc1(x)

        # Apply activation
        if self.activation_fn == "geglu":
            # GLU: split hidden into two parts and apply gating
            hidden, gate = hidden.chunk(2, dim=-1)
            hidden = hidden * F.gelu(gate)
        else:
            hidden = self.activation(hidden)

        # Optional layer normalization
        if self.layernorm is not None:
            hidden = self.layernorm(hidden)

        # Optional dropout
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        # Second linear transformation
        output = self.fc2(hidden)

        return output


class ConvolutionalExpert(nn.Module):
    """
    Convolutional expert for spatial/local pattern processing

    Uses 1D convolutions for sequence data or can be adapted
    for 2D convolutions for image-like data.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation_fn: str = "relu",
        use_bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        output_size = output_size or input_size
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Input projection to match conv dimensions
        self.input_proj = nn.Linear(input_size, hidden_size, bias=use_bias, **factory_kwargs)

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=use_bias,
                    **factory_kwargs
                )
            )

        # Activation function
        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size, bias=use_bias, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolutional expert

        Args:
            x: Input tensor [batch_size, seq_len, input_size]
                or [batch_size * seq_len, input_size]

        Returns:
            Output tensor with same shape pattern as input
        """
        original_shape = x.shape

        # Handle both 2D and 3D input tensors
        if x.dim() == 2:
            # Reshape to [batch_size, seq_len, input_size] for conv processing
            # Note: This is a simplified case - in practice, seq_len would need to be known
            batch_size = x.size(0)
            seq_len = 1  # Assume single sequence per batch for flattened case
            x = x.view(batch_size, seq_len, -1)
        else:
            batch_size, seq_len = x.shape[:2]

        # Project to hidden dimension
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_size]

        # Transpose for conv1d: [batch_size, hidden_size, seq_len]
        x = x.transpose(1, 2)

        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            residual = x
            x = conv_layer(x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
            # Residual connection
            x = x + residual

        # Transpose back: [batch_size, seq_len, hidden_size]
        x = x.transpose(1, 2)

        # Output projection
        x = self.output_proj(x)

        # Reshape to match original input shape
        if len(original_shape) == 2:
            x = x.view(original_shape[0], -1)

        return x


class AttentionExpert(nn.Module):
    """
    Attention-based expert for sequence modeling

    Uses self-attention to process sequences, allowing the expert
    to focus on relevant parts of the input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_bias: bool = True,
        max_seq_len: int = 2048,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        output_size = output_size or input_size
        factory_kwargs = {'device': device, 'dtype': dtype}

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size, bias=use_bias, **factory_kwargs)

        # Attention projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias, **factory_kwargs)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias, **factory_kwargs)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias, **factory_kwargs)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias, **factory_kwargs)

        # Final projection to output size
        self.final_proj = nn.Linear(hidden_size, output_size, bias=use_bias, **factory_kwargs)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size, **factory_kwargs)

        # Positional encoding (optional, for sequence processing)
        self.max_seq_len = max_seq_len
        self.register_buffer(
            'pos_encoding',
            self._create_positional_encoding(max_seq_len, hidden_size)
        )

    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention expert

        Args:
            x: Input tensor [batch_size, seq_len, input_size]
                or [batch_size * seq_len, input_size]

        Returns:
            Output tensor with same shape pattern as input
        """
        original_shape = x.shape

        # Handle both 2D and 3D input tensors
        if x.dim() == 2:
            # For MoE, we typically get flattened tokens
            # Reshape assuming seq_len=1 for each token
            batch_size = x.size(0)
            seq_len = 1
            x = x.view(batch_size, seq_len, -1)
        else:
            batch_size, seq_len = x.shape[:2]

        # Project to hidden dimension
        x = self.input_proj(x)  # [batch_size, seq_len, hidden_size]

        # Add positional encoding if sequence length allows
        if seq_len <= self.max_seq_len:
            x = x + self.pos_encoding[:, :seq_len, :]

        # Self-attention
        residual = x

        # Compute Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, hidden_size]
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1
        )
        attn_output = self.out_proj(attn_output)

        # Residual connection and layer norm
        x = self.layer_norm(attn_output + residual)

        # Final projection
        x = self.final_proj(x)

        # Reshape to match original input shape
        if len(original_shape) == 2:
            x = x.view(original_shape[0], -1)

        return x


class ParameterEfficientExpert(nn.Module):
    """
    Parameter-efficient expert using techniques like LoRA or adapters

    Designed for memory efficiency in large-scale MoE systems.
    Uses low-rank decomposition to reduce parameter count while
    maintaining expressiveness.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        rank: int = 32,
        dropout: float = 0.1,
        activation_fn: str = "relu",
        use_bias: bool = True,
        adaptation_type: str = "lora",  # "lora", "adapter", "prefix"
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        output_size = output_size or input_size
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.adaptation_type = adaptation_type
        self.rank = rank

        if adaptation_type == "lora":
            # Low-rank adaptation
            self.lora_a = nn.Linear(input_size, rank, bias=False, **factory_kwargs)
            self.lora_b = nn.Linear(rank, hidden_size, bias=use_bias, **factory_kwargs)
            self.base_proj = nn.Linear(input_size, hidden_size, bias=use_bias, **factory_kwargs)

            # Initialize LoRA layers
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

        elif adaptation_type == "adapter":
            # Adapter layers
            self.down_proj = nn.Linear(input_size, rank, bias=use_bias, **factory_kwargs)
            self.up_proj = nn.Linear(rank, hidden_size, bias=use_bias, **factory_kwargs)

        else:
            # Standard linear layer as fallback
            self.linear = nn.Linear(input_size, hidden_size, bias=use_bias, **factory_kwargs)

        # Activation function
        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation = nn.GELU()
        elif activation_fn == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size, bias=use_bias, **factory_kwargs)

        # Scaling factor for LoRA
        self.scaling = 1.0 / rank if adaptation_type == "lora" else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through parameter-efficient expert"""

        if self.adaptation_type == "lora":
            # LoRA: h = x @ W + (x @ A @ B) * scaling
            base_output = self.base_proj(x)
            lora_output = self.lora_b(self.lora_a(x)) * self.scaling
            hidden = base_output + lora_output

        elif self.adaptation_type == "adapter":
            # Adapter: h = down_proj(x) -> activation -> up_proj
            hidden = self.down_proj(x)
            hidden = self.activation(hidden)
            if self.dropout is not None:
                hidden = self.dropout(hidden)
            hidden = self.up_proj(hidden)

        else:
            # Standard linear transformation
            hidden = self.linear(x)

        # Apply activation (for non-adapter cases or final activation)
        if self.adaptation_type != "adapter":
            hidden = self.activation(hidden)

        # Optional dropout
        if self.dropout is not None and self.adaptation_type != "adapter":
            hidden = self.dropout(hidden)

        # Output projection
        output = self.output_proj(hidden)

        return output

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown"""
        total_params = sum(p.numel() for p in self.parameters())

        if self.adaptation_type == "lora":
            lora_params = (
                self.lora_a.weight.numel() +
                self.lora_b.weight.numel()
            )
            base_params = self.base_proj.weight.numel()
            if self.base_proj.bias is not None:
                base_params += self.base_proj.bias.numel()

            return {
                'total': total_params,
                'lora': lora_params,
                'base': base_params,
                'output': self.output_proj.weight.numel(),
                'compression_ratio': lora_params / (base_params + lora_params)
            }
        else:
            return {'total': total_params}


class HybridExpert(nn.Module):
    """
    Hybrid expert that combines multiple expert types

    Allows for more sophisticated expert architectures by
    combining different processing paradigms.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        expert_types: Dict[str, Dict[str, Any]] = None,
        combination_method: str = "concat",  # "concat", "add", "attention"
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        output_size = output_size or input_size
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.combination_method = combination_method

        # Default expert types if not provided
        if expert_types is None:
            expert_types = {
                'ffn': {'class': FeedForwardExpert, 'kwargs': {}},
                'conv': {'class': ConvolutionalExpert, 'kwargs': {}},
            }

        # Create sub-experts
        self.experts = nn.ModuleDict()
        expert_output_sizes = []

        for name, config in expert_types.items():
            expert_class = config['class']
            expert_kwargs = config.get('kwargs', {})

            # Adjust hidden size for concatenation
            if combination_method == "concat":
                expert_hidden = hidden_size // len(expert_types)
                expert_output_size = expert_hidden
            else:
                expert_hidden = hidden_size
                expert_output_size = output_size

            self.experts[name] = expert_class(
                input_size=input_size,
                hidden_size=expert_hidden,
                output_size=expert_output_size,
                **expert_kwargs,
                **factory_kwargs
            )
            expert_output_sizes.append(expert_output_size)

        # Combination layer
        if combination_method == "concat":
            total_concat_size = sum(expert_output_sizes)
            self.combination_proj = nn.Linear(
                total_concat_size, output_size, **factory_kwargs
            )
        elif combination_method == "attention":
            self.attention_weights = nn.Linear(
                input_size, len(expert_types), **factory_kwargs
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid expert"""

        # Get outputs from all sub-experts
        expert_outputs = []
        for expert in self.experts.values():
            expert_outputs.append(expert(x))

        # Combine outputs
        if self.combination_method == "concat":
            combined = torch.cat(expert_outputs, dim=-1)
            output = self.combination_proj(combined)
        elif self.combination_method == "add":
            output = sum(expert_outputs)
        elif self.combination_method == "attention":
            # Attention-weighted combination
            attn_weights = F.softmax(self.attention_weights(x), dim=-1)
            attn_weights = attn_weights.unsqueeze(-1)  # [batch, seq, num_experts, 1]

            stacked_outputs = torch.stack(expert_outputs, dim=-2)  # [batch, seq, num_experts, hidden]
            output = (stacked_outputs * attn_weights).sum(dim=-2)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

        return output


if __name__ == "__main__":
    # Test different expert types
    input_size = 768
    hidden_size = 1024
    batch_size = 4
    seq_len = 128

    expert_types = {
        'FeedForward': FeedForwardExpert,
        'Convolutional': ConvolutionalExpert,
        'Attention': AttentionExpert,
        'ParameterEfficient': ParameterEfficientExpert,
        'Hybrid': HybridExpert
    }

    x = torch.randn(batch_size, seq_len, input_size)

    for name, expert_class in expert_types.items():
        print(f"\nTesting {name} Expert:")

        if name == "Hybrid":
            expert = expert_class(input_size, hidden_size)
        else:
            expert = expert_class(input_size, hidden_size)

        if torch.cuda.is_available():
            x_test = x.cuda()
            expert = expert.cuda()
        else:
            x_test = x

        # Test forward pass
        output = expert(x_test)
        print(f"  Input shape: {x_test.shape}")
        print(f"  Output shape: {output.shape}")

        # Test with flattened input (common in MoE)
        x_flat = x_test.view(-1, input_size)
        output_flat = expert(x_flat)
        print(f"  Flattened input shape: {x_flat.shape}")
        print(f"  Flattened output shape: {output_flat.shape}")

        # Parameter count for parameter-efficient expert
        if name == "ParameterEfficient":
            param_stats = expert.get_parameter_count()
            print(f"  Parameter stats: {param_stats}")

    print("\nAll expert types tested successfully!")