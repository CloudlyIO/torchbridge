"""
Semantic ML Model Examples

This module demonstrates how kernel-optimized components can be used to build
complete ML models that showcase different semantic concepts in machine learning,
while maintaining optimal performance through efficient kernel patterns.

Each model demonstrates different ML concepts:
1. Language Modeling - Autoregressive generation and attention patterns
2. Vision Transformer - Patch processing and spatial attention
3. Graph Neural Network - Message passing and aggregation
4. Recommendation System - Embedding interactions and collaborative filtering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from ..components.basic_optimized import (
    OptimizedTransformerBlock,
    OptimizedMultiHeadAttention,
    OptimizedLayerNorm,
    OptimizedMLP,
    PositionalEncoding
)

try:
    from ..triton_kernels.fused_ops import TritonOptimizedTransformerBlock
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class KernelOptimizedLanguageModel(nn.Module):
    """
    Language Model demonstrating autoregressive generation semantics
    with kernel-optimized attention patterns.

    Key ML Concepts Demonstrated:
    - Autoregressive modeling
    - Causal attention masks
    - Token prediction and sequence generation
    - Gradient flow in deep networks
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
        use_triton: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_length = max_seq_length

        # Token and position embeddings - efficient lookup operations
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = PositionalEncoding(dim, max_seq_length)

        # Transformer layers with kernel optimization
        if use_triton and TRITON_AVAILABLE:
            self.layers = nn.ModuleList([
                TritonOptimizedTransformerBlock(dim, num_heads)
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                OptimizedTransformerBlock(dim, num_heads)
                for _ in range(num_layers)
            ])

        self.norm = OptimizedLayerNorm(dim)
        self.output_projection = nn.Linear(dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following best practices for deep networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass demonstrating autoregressive language modeling.

        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Mask for padding tokens [batch_size, seq_len]
            return_hidden_states: Whether to return intermediate representations
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create causal mask for autoregressive generation
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(device)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # Expand masks to [batch, heads, seq_len, seq_len] format
            padding_mask = ~attention_mask.bool()
            combined_mask = causal_mask.unsqueeze(0) | padding_mask.unsqueeze(-1)
        else:
            combined_mask = causal_mask

        # Embedding lookup - efficient for sparse token indices
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.position_embedding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Store hidden states for analysis if requested
        all_hidden_states = [] if return_hidden_states else None

        # Pass through transformer layers
        for layer in self.layers:
            if return_hidden_states:
                all_hidden_states.append(hidden_states.clone())

            hidden_states = layer(hidden_states)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        # Project to vocabulary for next token prediction
        logits = self.output_projection(hidden_states)

        result = {"logits": logits}
        if return_hidden_states:
            result["hidden_states"] = all_hidden_states

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Autoregressive text generation demonstrating kernel-optimized inference.
        """
        self.eval()
        batch_size = input_ids.shape[0]

        for _ in range(max_new_tokens):
            # Get predictions for the current sequence
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :]  # Last token logits

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Sample next token
            if do_sample:
                # Top-p (nucleus) sampling
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits.scatter_(-1, indices_to_remove, float('-inf'))

                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            # Truncate if exceeding max length
            if input_ids.shape[1] > self.max_seq_length:
                input_ids = input_ids[:, -self.max_seq_length:]

        return input_ids


class KernelOptimizedVisionTransformer(nn.Module):
    """
    Vision Transformer demonstrating spatial attention and patch processing.

    Key ML Concepts Demonstrated:
    - Image patch embedding
    - Spatial positional encoding
    - Global attention across patches
    - Classification head
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_classes: int = 1000,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch embedding - converts image patches to tokens
        self.patch_embedding = nn.Conv2d(
            num_channels, dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Learnable positional embeddings for spatial awareness
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))

        # Classification token (CLS token) for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer layers optimized for vision tasks
        self.layers = nn.ModuleList([
            OptimizedTransformerBlock(dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

        self.norm = OptimizedLayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights specifically for vision tasks"""
        # Initialize patch embedding like a linear layer
        nn.init.xavier_uniform_(self.patch_embedding.weight.view(self.patch_embedding.weight.size(0), -1))
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process images through patch embedding and spatial attention.

        Args:
            images: Input images [batch_size, channels, height, width]
        """
        batch_size = images.shape[0]

        # Convert images to patches - efficient kernel operation
        patches = self.patch_embedding(images)  # [batch, dim, H/patch_size, W/patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [batch, num_patches, dim]

        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, patches], dim=1)

        # Add positional encoding for spatial awareness
        tokens = tokens + self.pos_embedding
        tokens = self.dropout(tokens)

        # Pass through transformer layers with spatial attention
        for layer in self.layers:
            tokens = layer(tokens)

        # Final normalization
        tokens = self.norm(tokens)

        # Extract global representation from CLS token
        cls_representation = tokens[:, 0]  # First token is CLS
        patch_representations = tokens[:, 1:]  # Remaining are patches

        # Classification
        logits = self.classifier(cls_representation)

        return {
            "logits": logits,
            "cls_representation": cls_representation,
            "patch_representations": patch_representations,
            "spatial_attention_maps": self._compute_attention_maps(tokens)
        }

    def _compute_attention_maps(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention maps for visualization.
        This demonstrates how attention weights reveal spatial relationships.
        """
        # This is a simplified version - in practice, you'd extract attention
        # weights from the transformer layers during forward pass
        batch_size, num_tokens, dim = tokens.shape

        # Simulate attention between CLS token and patches
        cls_token = tokens[:, 0:1]  # [batch, 1, dim]
        patch_tokens = tokens[:, 1:]  # [batch, num_patches, dim]

        # Compute attention scores
        attention_scores = torch.matmul(cls_token, patch_tokens.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores / math.sqrt(dim), dim=-1)

        # Reshape to spatial dimensions
        spatial_size = int(math.sqrt(self.num_patches))
        attention_maps = attention_weights.view(batch_size, spatial_size, spatial_size)

        return attention_maps


class KernelOptimizedGraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network demonstrating message passing and aggregation.

    Key ML Concepts Demonstrated:
    - Node and edge representations
    - Message passing between connected nodes
    - Graph-level pooling and aggregation
    - Inductive vs transductive learning
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_classes: int = 10,
        pool_method: str = "attention"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pool_method = pool_method

        # Node and edge embedding layers
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)

        # Message passing layers using optimized attention
        self.message_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Graph-level pooling
        if pool_method == "attention":
            self.global_attention = OptimizedMultiHeadAttention(hidden_dim, num_heads)

        # Classification head
        self.classifier = nn.Sequential(
            OptimizedLayerNorm(hidden_dim),
            OptimizedMLP(hidden_dim, hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass demonstrating graph neural network computations.

        Args:
            node_features: Node feature matrix [num_nodes, node_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_features: Edge feature matrix [num_edges, edge_features]
            batch: Batch assignment for each node [num_nodes]
        """
        # Embed nodes and edges
        node_embeddings = self.node_embedding(node_features)
        if edge_features is not None:
            edge_embeddings = self.edge_embedding(edge_features)
        else:
            edge_embeddings = None

        # Message passing through graph layers
        node_representations = node_embeddings
        for layer in self.message_layers:
            node_representations = layer(
                node_representations, edge_index, edge_embeddings
            )

        # Graph-level pooling
        if batch is not None:
            # Multiple graphs in batch
            graph_representations = self._pool_graphs(node_representations, batch)
        else:
            # Single graph
            graph_representations = self._pool_single_graph(node_representations)

        # Classification
        logits = self.classifier(graph_representations)

        return {
            "logits": logits,
            "node_representations": node_representations,
            "graph_representations": graph_representations
        }

    def _pool_graphs(self, node_representations: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Pool multiple graphs in a batch"""
        num_graphs = batch.max().item() + 1
        graph_representations = []

        for i in range(num_graphs):
            mask = (batch == i)
            graph_nodes = node_representations[mask]
            graph_rep = self._pool_single_graph(graph_nodes)
            graph_representations.append(graph_rep)

        return torch.stack(graph_representations)

    def _pool_single_graph(self, node_representations: torch.Tensor) -> torch.Tensor:
        """Pool a single graph to get graph-level representation"""
        if self.pool_method == "mean":
            return node_representations.mean(dim=0)
        elif self.pool_method == "max":
            return node_representations.max(dim=0)[0]
        elif self.pool_method == "attention":
            # Use attention pooling for learnable aggregation
            # Add batch dimension for attention mechanism
            nodes = node_representations.unsqueeze(0)
            # Use self-attention to weight node contributions
            pooled = self.global_attention(nodes)
            return pooled.mean(dim=1).squeeze(0)
        else:
            raise ValueError(f"Unknown pooling method: {self.pool_method}")


class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer implementing message passing with optimized kernels.
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm = OptimizedLayerNorm(dim)
        self.mlp = OptimizedMLP(dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Message passing with attention mechanism.
        """
        residual = node_features
        num_nodes = node_features.shape[0]

        # Compute attention queries, keys, values
        q = self.q_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)

        # Message passing using edge connectivity
        src_nodes, dst_nodes = edge_index

        # Compute attention scores between connected nodes
        queries = q[dst_nodes]  # [num_edges, num_heads, head_dim]
        keys = k[src_nodes]     # [num_edges, num_heads, head_dim]
        values = v[src_nodes]   # [num_edges, num_heads, head_dim]

        # Attention computation
        scale = self.head_dim ** -0.5
        attention_scores = (queries * keys).sum(dim=-1) * scale  # [num_edges, num_heads]

        # Apply softmax over edges for each destination node
        attention_weights = torch.zeros_like(attention_scores)
        for node_idx in range(num_nodes):
            edge_mask = (dst_nodes == node_idx)
            if edge_mask.any():
                node_edges = attention_scores[edge_mask]
                node_weights = F.softmax(node_edges, dim=0)
                attention_weights[edge_mask] = node_weights

        # Aggregate messages
        messages = attention_weights.unsqueeze(-1) * values  # [num_edges, num_heads, head_dim]

        # Aggregate by destination node
        aggregated = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=node_features.device)
        aggregated.index_add_(0, dst_nodes, messages)

        # Reshape and project
        aggregated = aggregated.view(num_nodes, self.dim)
        output = self.out_proj(aggregated)

        # Residual connection and MLP
        output = self.norm(output + residual)
        output = output + self.mlp(output)

        return output


# Example usage and demonstration functions
def demonstrate_language_model():
    """Demonstrate language model capabilities"""
    print("=== Language Model Demonstration ===")

    vocab_size = 10000
    model = KernelOptimizedLanguageModel(vocab_size, dim=512, num_layers=6)

    # Example input sequence
    input_ids = torch.randint(0, vocab_size, (2, 64))

    # Forward pass
    outputs = model(input_ids, return_hidden_states=True)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Number of hidden states: {len(outputs['hidden_states'])}")

    # Text generation
    prompt = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, do_sample=True)
    print(f"Generated sequence length: {generated.shape[1]}")


def demonstrate_vision_transformer():
    """Demonstrate vision transformer capabilities"""
    print("=== Vision Transformer Demonstration ===")

    model = KernelOptimizedVisionTransformer(
        image_size=224, patch_size=16, num_classes=1000
    )

    # Example images
    images = torch.randn(2, 3, 224, 224)

    outputs = model(images)
    print(f"Classification logits shape: {outputs['logits'].shape}")
    print(f"CLS representation shape: {outputs['cls_representation'].shape}")
    print(f"Attention maps shape: {outputs['spatial_attention_maps'].shape}")


def demonstrate_graph_neural_network():
    """Demonstrate graph neural network capabilities"""
    print("=== Graph Neural Network Demonstration ===")

    model = KernelOptimizedGraphNeuralNetwork(
        node_features=64, edge_features=32, num_classes=10
    )

    # Example graph
    num_nodes = 100
    num_edges = 200

    node_features = torch.randn(num_nodes, 64)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_features = torch.randn(num_edges, 32)

    outputs = model(node_features, edge_index, edge_features)
    print(f"Graph classification logits shape: {outputs['logits'].shape}")
    print(f"Node representations shape: {outputs['node_representations'].shape}")


def run_semantic_models_demo():
    """Run all semantic model demonstrations"""
    print("Semantic ML Models with Kernel Optimization")
    print("="*60)

    demonstrate_language_model()
    print()
    demonstrate_vision_transformer()
    print()
    demonstrate_graph_neural_network()

    print("\nKey Semantic Concepts Demonstrated:")
    print("1. Autoregressive modeling in language generation")
    print("2. Spatial attention in vision transformers")
    print("3. Message passing in graph neural networks")
    print("4. Efficient kernel patterns across all architectures")


if __name__ == "__main__":
    run_semantic_models_demo()