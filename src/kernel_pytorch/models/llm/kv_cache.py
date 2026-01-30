"""
KV-Cache Optimization Module

Provides optimized key-value cache management for LLM inference:
- Standard KV-cache management
- Paged attention (vLLM-style)
- Sliding window cache for efficient long contexts

Version: 0.4.12
"""

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for KV-cache."""
    max_length: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    # Paging options
    page_size: int = 16  # Tokens per page
    num_pages: int = 256

    # Sliding window
    window_size: int = 4096


class KVCacheManager:
    """
    Standard KV-cache manager for LLM inference.

    Manages key-value caches across transformer layers for efficient
    autoregressive generation.

    Example:
        >>> cache_manager = KVCacheManager(config)
        >>> cache = cache_manager.create_cache(batch_size=4)
        >>> cache = cache_manager.update_cache(cache, new_keys, new_values, layer_idx)
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize KV-cache manager.

        Args:
            config: Cache configuration
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    def create_cache(self, batch_size: int = 1) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Create empty KV-cache for all layers.

        Args:
            batch_size: Batch size for generation

        Returns:
            List of (key_cache, value_cache) tuples for each layer
        """
        cache = []
        for _ in range(self.config.num_layers):
            key_cache = torch.zeros(
                batch_size,
                self.config.num_heads,
                0,  # Will grow during generation
                self.config.head_dim,
                dtype=self.config.dtype,
                device=self.device
            )
            value_cache = torch.zeros(
                batch_size,
                self.config.num_heads,
                0,
                self.config.head_dim,
                dtype=self.config.dtype,
                device=self.device
            )
            cache.append((key_cache, value_cache))

        return cache

    def update_cache(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        layer_idx: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Update cache with new keys and values.

        Args:
            cache: Current cache state
            new_keys: New keys to append [batch, heads, seq, head_dim]
            new_values: New values to append [batch, heads, seq, head_dim]
            layer_idx: Layer index to update

        Returns:
            Updated cache
        """
        key_cache, value_cache = cache[layer_idx]

        # Concatenate new keys/values
        updated_keys = torch.cat([key_cache, new_keys], dim=2)
        updated_values = torch.cat([value_cache, new_values], dim=2)

        # Truncate if exceeds max length
        if updated_keys.size(2) > self.config.max_length:
            updated_keys = updated_keys[:, :, -self.config.max_length:, :]
            updated_values = updated_values[:, :, -self.config.max_length:, :]

        cache[layer_idx] = (updated_keys, updated_values)
        return cache

    def get_cache_length(self, cache: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
        """Get current cache length."""
        if not cache or not cache[0][0].numel():
            return 0
        return cache[0][0].size(2)

    def clear_cache(self, cache: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Clear cache memory."""
        for i in range(len(cache)):
            cache[i] = (
                cache[i][0][:, :, :0, :],
                cache[i][1][:, :, :0, :]
            )

    def get_memory_usage(self, cache: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
        """Get cache memory usage in MB."""
        total_bytes = 0
        for key_cache, value_cache in cache:
            total_bytes += key_cache.numel() * key_cache.element_size()
            total_bytes += value_cache.numel() * value_cache.element_size()

        return {
            "cache_memory_mb": total_bytes / (1024 * 1024),
            "cache_length": self.get_cache_length(cache),
            "num_layers": len(cache),
        }


class PagedKVCache:
    """
    Paged KV-cache for memory-efficient long sequences.

    Implements vLLM-style paged attention where KV-cache is stored
    in fixed-size pages for efficient memory allocation.

    Example:
        >>> paged_cache = PagedKVCache(config)
        >>> paged_cache.allocate_pages(batch_size=4, num_pages_per_seq=10)
        >>> paged_cache.write_to_cache(keys, values, page_indices, offsets)
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize paged KV-cache.

        Args:
            config: Cache configuration with paging settings
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Page table: maps (batch, seq_pos) -> page_id
        self.page_table: torch.Tensor | None = None

        # Physical pages: [num_pages, 2, num_layers, page_size, num_heads, head_dim]
        # 2 for keys and values
        self.physical_pages: torch.Tensor | None = None

        # Free page list
        self.free_pages: list[int] = []

    def initialize_pages(self) -> None:
        """Initialize the physical page pool."""
        # Allocate physical pages
        self.physical_pages = torch.zeros(
            self.config.num_pages,
            2,  # keys and values
            self.config.num_layers,
            self.config.page_size,
            self.config.num_heads,
            self.config.head_dim,
            dtype=self.config.dtype,
            device=self.device
        )

        # Initialize free page list
        self.free_pages = list(range(self.config.num_pages))

        logger.info(
            f"Initialized {self.config.num_pages} pages, "
            f"page_size={self.config.page_size}, "
            f"memory={self.get_memory_usage()['total_mb']:.1f}MB"
        )

    def allocate_pages(self, batch_size: int, num_pages_per_seq: int) -> torch.Tensor:
        """
        Allocate pages for a batch of sequences.

        Args:
            batch_size: Number of sequences
            num_pages_per_seq: Pages per sequence

        Returns:
            Page table [batch_size, num_pages_per_seq]
        """
        if self.physical_pages is None:
            self.initialize_pages()

        total_pages_needed = batch_size * num_pages_per_seq
        if len(self.free_pages) < total_pages_needed:
            raise RuntimeError(
                f"Not enough free pages: need {total_pages_needed}, "
                f"have {len(self.free_pages)}"
            )

        # Allocate pages
        allocated = []
        for _ in range(total_pages_needed):
            page_id = self.free_pages.pop(0)
            allocated.append(page_id)

        # Create page table
        self.page_table = torch.tensor(
            allocated,
            dtype=torch.long,
            device=self.device
        ).reshape(batch_size, num_pages_per_seq)

        return self.page_table

    def free_pages_for_sequence(self, batch_idx: int) -> None:
        """Free all pages for a completed sequence."""
        if self.page_table is None:
            return

        pages = self.page_table[batch_idx].tolist()
        self.free_pages.extend(pages)

    def write_to_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int,
        page_indices: torch.Tensor,
        slot_indices: torch.Tensor
    ) -> None:
        """
        Write keys and values to specified page slots.

        Args:
            keys: Keys to write [batch, heads, seq, head_dim]
            values: Values to write [batch, heads, seq, head_dim]
            layer_idx: Layer index
            page_indices: Page IDs [batch, num_tokens]
            slot_indices: Slot offsets within pages [batch, num_tokens]
        """
        if self.physical_pages is None:
            raise RuntimeError("Pages not initialized")

        # Reshape for scatter
        batch_size, num_heads, seq_len, head_dim = keys.shape

        for b in range(batch_size):
            for s in range(seq_len):
                page_id = page_indices[b, s].item()
                slot = slot_indices[b, s].item()

                self.physical_pages[page_id, 0, layer_idx, slot] = keys[b, :, s, :]
                self.physical_pages[page_id, 1, layer_idx, slot] = values[b, :, s, :]

    def read_from_cache(
        self,
        layer_idx: int,
        page_indices: torch.Tensor,
        slot_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Read keys and values from cache.

        Args:
            layer_idx: Layer index
            page_indices: Page IDs [batch, num_tokens]
            slot_indices: Slot offsets [batch, num_tokens]

        Returns:
            Tuple of (keys, values) tensors
        """
        if self.physical_pages is None:
            raise RuntimeError("Pages not initialized")

        batch_size, seq_len = page_indices.shape
        keys = torch.zeros(
            batch_size, self.config.num_heads, seq_len, self.config.head_dim,
            dtype=self.config.dtype, device=self.device
        )
        values = torch.zeros_like(keys)

        for b in range(batch_size):
            for s in range(seq_len):
                page_id = page_indices[b, s].item()
                slot = slot_indices[b, s].item()

                keys[b, :, s, :] = self.physical_pages[page_id, 0, layer_idx, slot]
                values[b, :, s, :] = self.physical_pages[page_id, 1, layer_idx, slot]

        return keys, values

    def get_memory_usage(self) -> dict[str, float]:
        """Get memory usage statistics."""
        if self.physical_pages is None:
            return {"total_mb": 0, "used_pages": 0, "free_pages": 0}

        total_bytes = self.physical_pages.numel() * self.physical_pages.element_size()
        used_pages = self.config.num_pages - len(self.free_pages)

        return {
            "total_mb": total_bytes / (1024 * 1024),
            "used_pages": used_pages,
            "free_pages": len(self.free_pages),
            "utilization": used_pages / self.config.num_pages if self.config.num_pages > 0 else 0,
        }


class SlidingWindowCache:
    """
    Sliding window KV-cache for efficient long context handling.

    Maintains only the most recent tokens within a sliding window,
    useful for models with sliding window attention (e.g., Mistral).

    Example:
        >>> sw_cache = SlidingWindowCache(config)
        >>> cache = sw_cache.create_cache(batch_size=4)
        >>> cache = sw_cache.update_cache(cache, new_keys, new_values, layer_idx)
    """

    def __init__(self, config: CacheConfig):
        """
        Initialize sliding window cache.

        Args:
            config: Cache configuration with window_size
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.window_size = config.window_size

    def create_cache(self, batch_size: int = 1) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Create empty sliding window cache."""
        cache = []
        for _ in range(self.config.num_layers):
            key_cache = torch.zeros(
                batch_size,
                self.config.num_heads,
                0,
                self.config.head_dim,
                dtype=self.config.dtype,
                device=self.device
            )
            value_cache = torch.zeros_like(key_cache)
            cache.append((key_cache, value_cache))

        return cache

    def update_cache(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]],
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        layer_idx: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Update cache with sliding window truncation.

        Args:
            cache: Current cache state
            new_keys: New keys [batch, heads, seq, head_dim]
            new_values: New values [batch, heads, seq, head_dim]
            layer_idx: Layer index

        Returns:
            Updated cache (within window size)
        """
        key_cache, value_cache = cache[layer_idx]

        # Concatenate
        updated_keys = torch.cat([key_cache, new_keys], dim=2)
        updated_values = torch.cat([value_cache, new_values], dim=2)

        # Apply sliding window
        if updated_keys.size(2) > self.window_size:
            updated_keys = updated_keys[:, :, -self.window_size:, :]
            updated_values = updated_values[:, :, -self.window_size:, :]

        cache[layer_idx] = (updated_keys, updated_values)
        return cache

    def get_window_mask(self, seq_len: int, cache_len: int) -> torch.Tensor:
        """
        Get attention mask for sliding window.

        Args:
            seq_len: Query sequence length
            cache_len: Current cache length

        Returns:
            Attention mask tensor
        """
        total_len = cache_len + seq_len
        mask = torch.ones(seq_len, total_len, dtype=torch.bool, device=self.device)

        # Mask out positions outside the window
        for i in range(seq_len):
            query_pos = cache_len + i
            start_pos = max(0, query_pos - self.window_size + 1)
            mask[i, :start_pos] = False

        return mask

    def get_memory_usage(
        self,
        cache: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, float]:
        """Get cache memory usage."""
        total_bytes = 0
        cache_len = 0

        for key_cache, value_cache in cache:
            total_bytes += key_cache.numel() * key_cache.element_size()
            total_bytes += value_cache.numel() * value_cache.element_size()
            if key_cache.numel() > 0:
                cache_len = key_cache.size(2)

        return {
            "cache_memory_mb": total_bytes / (1024 * 1024),
            "cache_length": cache_len,
            "window_size": self.window_size,
            "window_utilization": cache_len / self.window_size if self.window_size > 0 else 0,
        }


__all__ = [
    "CacheConfig",
    "KVCacheManager",
    "PagedKVCache",
    "SlidingWindowCache",
]
