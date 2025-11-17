"""
Zero-Copy Spinner Support
==========================

Helper utilities for enabling zero-copy output in spinners.

Allows gradual migration from legacy MemoryShard to ZeroCopyMemoryShard.

Usage:
    # In your spinner __init__:
    def __init__(self, wool_storage: Optional[WoolStorage] = None, **kwargs):
        super().__init__(**kwargs)
        self.wool = wool_storage
        self.enable_zerocopy = wool_storage is not None

    # In your _spin_impl:
    shard = create_zerocopy_shard(
        text=text,
        id_suffix=str(idx),
        episode=episode,
        entities=entities,
        motifs=motifs,
        metadata=metadata,
        wool=self.wool,
        spinner_name=self.name
    )

Author: Claude Code
Date: November 17, 2025 (Phase 2)
"""

from typing import Optional, List, Dict, Any, Union
from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.wool import WoolStorage, TextReference, ZeroCopyMemoryShard


def create_zerocopy_shard(
    text: str,
    id_suffix: str,
    episode: str,
    entities: List[str],
    motifs: List[str],
    metadata: Dict[str, Any],
    wool: Optional[WoolStorage],
    spinner_name: str = "unknown"
) -> Union[MemoryShard, ZeroCopyMemoryShard]:
    """
    Create shard with zero-copy support if wool storage available.

    Args:
        text: Shard text content
        id_suffix: Unique suffix for shard ID
        episode: Episode identifier
        entities: Extracted entities
        motifs: Extracted motifs
        metadata: Additional metadata
        wool: WoolStorage instance (None = use legacy MemoryShard)
        spinner_name: Name of spinner creating the shard

    Returns:
        ZeroCopyMemoryShard if wool provided, else MemoryShard
    """
    # Add spinner metadata
    metadata['spinner'] = spinner_name

    # Create shard ID
    shard_id = f"{spinner_name}_{id_suffix}"

    if wool is not None:
        # Zero-copy path: store text in wool, create reference
        text_bytes = text.encode('utf-8')
        wool_ref = wool.store(text_bytes, content_type='text/plain')

        text_ref = TextReference(
            file_id=wool_ref.file_id,
            offset=0,
            length=len(text_bytes)
        )

        return ZeroCopyMemoryShard(
            id=shard_id,
            episode=episode,
            text_ref=text_ref,
            entities=entities[:20],  # Limit to 20
            motifs=motifs[:10],  # Limit to 10
            metadata=metadata
        )
    else:
        # Legacy path: full text in shard
        return MemoryShard(
            id=shard_id,
            text=text,
            episode=episode,
            entities=entities[:20],
            motifs=motifs[:10],
            metadata=metadata
        )


def batch_create_zerocopy_shards(
    texts: List[str],
    id_prefix: str,
    episode: str,
    entities_list: List[List[str]],
    motifs_list: List[List[str]],
    metadata_list: List[Dict[str, Any]],
    wool: Optional[WoolStorage],
    spinner_name: str = "unknown"
) -> List[Union[MemoryShard, ZeroCopyMemoryShard]]:
    """
    Batch create shards with zero-copy support.

    Args:
        texts: List of text content
        id_prefix: Prefix for shard IDs
        episode: Episode identifier
        entities_list: List of entity lists (one per text)
        motifs_list: List of motif lists (one per text)
        metadata_list: List of metadata dicts (one per text)
        wool: WoolStorage instance (None = use legacy)
        spinner_name: Name of spinner

    Returns:
        List of shards (ZeroCopy if wool provided, else legacy)
    """
    shards = []

    for idx, text in enumerate(texts):
        shard = create_zerocopy_shard(
            text=text,
            id_suffix=f"{id_prefix}_{idx}",
            episode=episode,
            entities=entities_list[idx] if idx < len(entities_list) else [],
            motifs=motifs_list[idx] if idx < len(motifs_list) else [],
            metadata=metadata_list[idx] if idx < len(metadata_list) else {},
            wool=wool,
            spinner_name=spinner_name
        )
        shards.append(shard)

    return shards
