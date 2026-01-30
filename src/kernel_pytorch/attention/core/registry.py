"""
Attention Registry System

Provides a centralized registry for all attention implementations,
enabling dynamic creation and configuration of attention layers.
"""

import warnings
from functools import wraps

from .base import BaseAttention
from .config import AttentionConfig, AttentionPatterns

# Global registry for attention implementations
_ATTENTION_REGISTRY: dict[str, type[BaseAttention]] = {}


def register_attention(name: str):
    """
    Decorator to register an attention implementation.

    Args:
        name: Unique name for the attention implementation

    Example:
        @register_attention('flash_attention3')
        class FlashAttention3(BaseAttention):
            ...
    """
    def decorator(cls: type[BaseAttention]):
        if name in _ATTENTION_REGISTRY:
            warnings.warn(f"Overriding existing attention implementation '{name}'", stacklevel=2)

        _ATTENTION_REGISTRY[name] = cls

        # Add name to class for introspection
        cls._registry_name = name

        @wraps(cls)
        def wrapper(*args, **kwargs):
            return cls(*args, **kwargs)

        return cls

    return decorator


def get_attention_registry() -> dict[str, type[BaseAttention]]:
    """Get the current attention registry"""
    return _ATTENTION_REGISTRY.copy()


def list_available_attention() -> list:
    """List all registered attention implementations"""
    return list(_ATTENTION_REGISTRY.keys())


def create_attention(config: AttentionConfig,
                    implementation: str | None = None,
                    **kwargs) -> BaseAttention:
    """
    Factory function to create attention layers.

    Args:
        config: Attention configuration
        implementation: Specific implementation to use (auto-selected if None)
        **kwargs: Additional arguments passed to the attention constructor

    Returns:
        Configured attention layer

    Example:
        config = AttentionConfig(embed_dim=512, num_heads=8)
        attention = create_attention(config, implementation='flash_attention3')
    """
    if implementation is None:
        implementation = _select_best_implementation(config)

    if implementation not in _ATTENTION_REGISTRY:
        available = ', '.join(_ATTENTION_REGISTRY.keys())
        raise ValueError(f"Unknown attention implementation '{implementation}'. "
                        f"Available implementations: {available}")

    attention_cls = _ATTENTION_REGISTRY[implementation]
    return attention_cls(config, **kwargs)


def _select_best_implementation(config: AttentionConfig) -> str:
    """
    Automatically select the best attention implementation based on config.

    This implements a heuristic to choose the most appropriate attention
    implementation based on the configuration parameters.
    """
    # Pattern-specific selections
    if config.pattern == AttentionPatterns.RING:
        if 'ring_attention' in _ATTENTION_REGISTRY:
            return 'ring_attention'

    elif config.pattern in [AttentionPatterns.SPARSE, AttentionPatterns.DYNAMIC_SPARSE]:
        if 'dynamic_sparse_attention' in _ATTENTION_REGISTRY:
            return 'dynamic_sparse_attention'
        elif 'sparse_attention' in _ATTENTION_REGISTRY:
            return 'sparse_attention'

    # Memory-efficient selection
    if config.use_memory_efficient:
        if 'memory_efficient_attention' in _ATTENTION_REGISTRY:
            return 'memory_efficient_attention'

    # Flash attention selection (prefer newer versions)
    if config.use_flash_attention:
        # Try FlashAttention-3 first, fall back to v2
        for flash_impl in ['flash_attention3', 'flash_attention_3', 'flash_attention2', 'flash_attention']:
            if flash_impl in _ATTENTION_REGISTRY:
                return flash_impl

    # Default fallback - try common implementations
    for default_impl in ['flash_attention3', 'memory_efficient_attention', 'standard_attention']:
        if default_impl in _ATTENTION_REGISTRY:
            return default_impl

    # Last resort - use any available implementation
    if _ATTENTION_REGISTRY:
        return next(iter(_ATTENTION_REGISTRY.keys()))

    raise RuntimeError("No attention implementations registered")


def unregister_attention(name: str):
    """Remove an attention implementation from the registry"""
    if name in _ATTENTION_REGISTRY:
        del _ATTENTION_REGISTRY[name]
    else:
        warnings.warn(f"Attention implementation '{name}' not found in registry", stacklevel=2)


def clear_registry():
    """Clear all registered attention implementations (mainly for testing)"""
    global _ATTENTION_REGISTRY
    _ATTENTION_REGISTRY = {}


# Convenience functions for common patterns
def create_flash_attention(embed_dim: int, num_heads: int, **kwargs) -> BaseAttention:
    """Create FlashAttention with minimal configuration"""
    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_flash_attention=True,
        **kwargs
    )
    return create_attention(config, implementation='flash_attention3')


def create_memory_efficient_attention(embed_dim: int, num_heads: int, **kwargs) -> BaseAttention:
    """Create memory-efficient attention with minimal configuration"""
    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_memory_efficient=True,
        **kwargs
    )
    return create_attention(config, implementation='memory_efficient_attention')


def create_sparse_attention(embed_dim: int, num_heads: int, sparsity: float = 0.1, **kwargs) -> BaseAttention:
    """Create sparse attention with minimal configuration"""
    from .config import DynamicSparseConfig

    config = AttentionConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        pattern=AttentionPatterns.DYNAMIC_SPARSE,
        sparse_config=DynamicSparseConfig(sparsity_threshold=sparsity),
        **kwargs
    )
    return create_attention(config, implementation='dynamic_sparse_attention')
