"""
Zero-Copy Memory Shard
=======================

Memory shard with text reference instead of text copy.
Enables zero-copy data flow from spinner → graph → query.

Author: Claude Code
Date: November 17, 2025
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .storage import WoolStorage
    from .text_reference import TextReference


@dataclass
class ZeroCopyMemoryShard:
    """
    Zero-copy memory shard using TextReference.

    Key Properties:
    - text_ref: TextReference to wool storage (NO COPY)
    - Lazy evaluation of entities/motifs (compute on first access)
    - Minimal metadata storage (only what's essential)

    Memory Footprint:
    - Current MemoryShard: ~500 bytes + len(text)
    - ZeroCopyMemoryShard: ~200 bytes (just pointers!)

    Example:
        # Store text in wool
        wool = WoolStorage()
        text_bytes = "Thompson Sampling balances...".encode('utf-8')
        wool_ref = wool.store(text_bytes)

        # Create TextReference
        text_ref = TextReference(
            file_id=wool_ref.file_id,
            offset=0,
            length=len(text_bytes)
        )

        # Create zero-copy shard
        shard = ZeroCopyMemoryShard(
            id="thompson_sampling_intro",
            episode="rl_tutorial",
            text_ref=text_ref,
            entities=["thompson_sampling", "exploration"],
            motifs=["reinforcement_learning"]
        )

        # Later, resolve text (lazy)
        text = shard.get_text(wool)
    """

    # Core data (zero-copy)
    id: str                 # Shard ID
    episode: str            # Episode/document ID
    text_ref: 'TextReference'  # Reference to wool storage (NO COPY!)

    # Extracted semantics
    entities: List[str] = field(default_factory=list)
    motifs: List[str] = field(default_factory=list)

    # Minimal metadata
    metadata: Dict = field(default_factory=dict)

    # Lazy text cache (only materialized on first access)
    _text_cache: Optional[str] = field(default=None, init=False, repr=False)

    def get_text(self, wool_storage: 'WoolStorage') -> str:
        """
        Get text (lazy, cached).

        First call: Resolves from wool storage
        Subsequent calls: Returns cached value

        Args:
            wool_storage: WoolStorage instance

        Returns:
            Decoded text string
        """
        if self._text_cache is None:
            self._text_cache = self.text_ref.resolve(wool_storage)
        return self._text_cache

    @property
    def has_text_cached(self) -> bool:
        """Check if text is cached."""
        return self._text_cache is not None

    def clear_text_cache(self):
        """Clear cached text (free memory)."""
        self._text_cache = None

    def to_dict(self, include_text: bool = False, wool_storage: Optional['WoolStorage'] = None) -> dict:
        """
        Serialize shard.

        Args:
            include_text: If True, include resolved text (requires wool_storage)
            wool_storage: WoolStorage instance (required if include_text=True)

        Returns:
            Dictionary representation
        """
        result = {
            'id': self.id,
            'episode': self.episode,
            'text_ref': self.text_ref.to_dict(),
            'entities': self.entities,
            'motifs': self.motifs,
            'metadata': self.metadata
        }

        if include_text:
            if wool_storage is None:
                raise ValueError("wool_storage required when include_text=True")
            result['text'] = self.get_text(wool_storage)

        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'ZeroCopyMemoryShard':
        """
        Deserialize shard.

        Args:
            data: Dictionary representation

        Returns:
            ZeroCopyMemoryShard instance
        """
        from .text_reference import TextReference

        text_ref = TextReference.from_dict(data['text_ref'])

        return cls(
            id=data['id'],
            episode=data['episode'],
            text_ref=text_ref,
            entities=data.get('entities', []),
            motifs=data.get('motifs', []),
            metadata=data.get('metadata', {})
        )

    def __sizeof__(self) -> int:
        """
        Report actual memory usage (excludes wool storage).

        Returns:
            Size in bytes (approximate)
        """
        # id: ~50 bytes
        # episode: ~50 bytes
        # text_ref: ~88 bytes (TextReference)
        # entities: ~10 bytes per entity
        # motifs: ~10 bytes per motif
        # metadata: ~100 bytes (estimate)
        # _text_cache: 0 if not cached, len(text) if cached

        base_size = 200
        entities_size = len(self.entities) * 10
        motifs_size = len(self.motifs) * 10
        text_cache_size = len(self._text_cache) if self._text_cache else 0

        return base_size + entities_size + motifs_size + text_cache_size

    def __repr__(self) -> str:
        """String representation."""
        cached = "cached" if self.has_text_cached else "not cached"
        return (
            f"ZeroCopyMemoryShard(id={self.id}, episode={self.episode}, "
            f"text_ref={self.text_ref.file_id[:8]}..., "
            f"entities={len(self.entities)}, motifs={len(self.motifs)}, "
            f"text={cached})"
        )


# ============================================================================
# Conversion Utilities
# ============================================================================

def convert_memory_shard_to_zerocopy(
    shard: 'MemoryShard',
    wool_storage: 'WoolStorage'
) -> ZeroCopyMemoryShard:
    """
    Convert legacy MemoryShard to ZeroCopyMemoryShard.

    Args:
        shard: Legacy MemoryShard with full text
        wool_storage: WoolStorage instance

    Returns:
        ZeroCopyMemoryShard with TextReference
    """
    from .text_reference import TextReference

    # Store text in wool
    text_bytes = shard.text.encode('utf-8')
    wool_ref = wool_storage.store(text_bytes, content_type='text/plain')

    # Create TextReference
    text_ref = TextReference(
        file_id=wool_ref.file_id,
        offset=0,
        length=len(text_bytes),
        encoding='utf-8'
    )

    # Create zero-copy shard
    return ZeroCopyMemoryShard(
        id=shard.id,
        episode=shard.episode,
        text_ref=text_ref,
        entities=shard.entities,
        motifs=shard.motifs,
        metadata=shard.metadata
    )
