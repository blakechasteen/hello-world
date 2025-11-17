"""
Hybrid Knowledge Graph - Gradual Migration Support
===================================================

Knowledge graph wrapper supporting both legacy and zero-copy nodes.
Enables gradual migration from copy-heavy to zero-copy architecture.

Features:
- Transparent support for both MemoryShard (legacy) and ZeroCopyMemoryShard
- Automatic text resolution (works for both text and text_ref)
- Migration utilities (convert legacy → zero-copy)
- Statistics tracking (legacy vs zero-copy distribution)

Author: Claude Code
Date: November 17, 2025
"""

import logging
from typing import Optional, List, Union, Dict
from dataclasses import dataclass

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.Documentation.types import MemoryShard

from .storage import WoolStorage
from .text_reference import TextReference
from .zerocopy_shard import ZeroCopyMemoryShard, convert_memory_shard_to_zerocopy

logger = logging.getLogger(__name__)


# ============================================================================
# Statistics
# ============================================================================

@dataclass
class HybridKGStats:
    """Statistics for hybrid KG."""
    legacy_nodes: int = 0          # Nodes with full text
    zerocopy_nodes: int = 0        # Nodes with TextReference
    total_nodes: int = 0
    legacy_bytes: int = 0          # Memory used by legacy nodes
    zerocopy_bytes: int = 0        # Memory used by zero-copy nodes
    migrated_nodes: int = 0        # Nodes migrated from legacy

    @property
    def zerocopy_percentage(self) -> float:
        """Percentage of nodes that are zero-copy."""
        if self.total_nodes == 0:
            return 0.0
        return (self.zerocopy_nodes / self.total_nodes) * 100

    @property
    def memory_savings_bytes(self) -> int:
        """Estimated memory savings from zero-copy nodes."""
        # Assume average text size of 500 bytes
        # Legacy: 500 bytes (text) + 100 bytes (overhead) = 600 bytes
        # Zero-copy: 88 bytes (TextReference) + 100 bytes (overhead) = 188 bytes
        # Savings per node: 412 bytes
        return self.zerocopy_nodes * 412

    @property
    def memory_savings_mb(self) -> float:
        """Memory savings in MB."""
        return self.memory_savings_bytes / (1024 * 1024)

    def to_dict(self) -> dict:
        """Serialize statistics."""
        return {
            'legacy_nodes': self.legacy_nodes,
            'zerocopy_nodes': self.zerocopy_nodes,
            'total_nodes': self.total_nodes,
            'zerocopy_percentage': self.zerocopy_percentage,
            'legacy_bytes': self.legacy_bytes,
            'zerocopy_bytes': self.zerocopy_bytes,
            'migrated_nodes': self.migrated_nodes,
            'memory_savings_bytes': self.memory_savings_bytes,
            'memory_savings_mb': self.memory_savings_mb
        }


# ============================================================================
# Hybrid Knowledge Graph
# ============================================================================

class HybridKG:
    """
    Knowledge graph supporting both legacy and zero-copy nodes.

    Usage:
        wool = WoolStorage()
        kg = HybridKG(wool_storage=wool)

        # Add legacy shard (copy-heavy)
        legacy_shard = MemoryShard(...)
        kg.add_shard(legacy_shard)

        # Add zero-copy shard
        zerocopy_shard = ZeroCopyMemoryShard(...)
        kg.add_shard(zerocopy_shard)

        # Get text (works for both!)
        text = kg.get_text("node_id")

        # Migrate legacy → zero-copy
        kg.migrate_node("node_id")

        # Get statistics
        stats = kg.get_stats()
    """

    def __init__(
        self,
        wool_storage: Optional[WoolStorage] = None,
        base_kg: Optional[KG] = None
    ):
        """
        Initialize hybrid KG.

        Args:
            wool_storage: WoolStorage instance (creates new if None)
            base_kg: Base KG instance (creates new if None)
        """
        self.wool = wool_storage or WoolStorage()
        self.kg = base_kg or KG()

        # Statistics
        self.stats = HybridKGStats()

        logger.info("Initialized HybridKG")

    def add_shard(
        self,
        shard: Union[MemoryShard, ZeroCopyMemoryShard],
        auto_convert: bool = False
    ):
        """
        Add memory shard (supports both legacy and zero-copy).

        Args:
            shard: MemoryShard or ZeroCopyMemoryShard
            auto_convert: If True, convert legacy → zero-copy automatically

        Returns:
            None
        """
        if isinstance(shard, ZeroCopyMemoryShard):
            self._add_zerocopy_shard(shard)
        elif isinstance(shard, MemoryShard):
            if auto_convert:
                # Convert to zero-copy
                zerocopy_shard = convert_memory_shard_to_zerocopy(shard, self.wool)
                self._add_zerocopy_shard(zerocopy_shard)
            else:
                # Add as legacy
                self._add_legacy_shard(shard)
        else:
            raise TypeError(f"Invalid shard type: {type(shard)}")

    def _add_legacy_shard(self, shard: MemoryShard):
        """Add legacy shard with full text."""
        # Add node with full text (copy-heavy)
        self.kg.graph.add_node(
            shard.id,
            text=shard.text,  # Full text copied!
            entities=shard.entities,
            motifs=shard.motifs,
            metadata=shard.metadata,
            episode=shard.episode,
            _is_legacy=True  # Mark as legacy
        )

        # Update stats
        self.stats.legacy_nodes += 1
        self.stats.total_nodes += 1
        self.stats.legacy_bytes += len(shard.text)

        logger.debug(f"Added legacy shard: {shard.id}")

    def _add_zerocopy_shard(self, shard: ZeroCopyMemoryShard):
        """Add zero-copy shard with TextReference."""
        # Add node with TextReference (zero-copy)
        self.kg.graph.add_node(
            shard.id,
            text_ref=shard.text_ref.to_dict(),  # Reference only!
            entities=shard.entities,
            motifs=shard.motifs,
            metadata=shard.metadata,
            episode=shard.episode,
            _is_legacy=False  # Mark as zero-copy
        )

        # Update stats
        self.stats.zerocopy_nodes += 1
        self.stats.total_nodes += 1
        self.stats.zerocopy_bytes += 88  # TextReference size

        logger.debug(f"Added zero-copy shard: {shard.id}")

    def get_text(self, node_id: str) -> Optional[str]:
        """
        Get text for node (works for both legacy and zero-copy).

        Args:
            node_id: Node ID

        Returns:
            Text string, or None if node doesn't exist

        Raises:
            FileNotFoundError: If zero-copy node references missing file
        """
        if node_id not in self.kg.graph:
            return None

        node = self.kg.graph.nodes[node_id]

        # Check if legacy (has full text)
        if 'text' in node:
            return node['text']

        # Check if zero-copy (has text_ref)
        if 'text_ref' in node:
            text_ref = TextReference.from_dict(node['text_ref'])
            return text_ref.resolve(self.wool)

        # No text available
        return None

    def migrate_node(self, node_id: str) -> bool:
        """
        Migrate node from legacy to zero-copy.

        Args:
            node_id: Node ID to migrate

        Returns:
            True if migrated, False if already zero-copy or doesn't exist
        """
        if node_id not in self.kg.graph:
            logger.warning(f"Node not found: {node_id}")
            return False

        node = self.kg.graph.nodes[node_id]

        # Check if already zero-copy
        if not node.get('_is_legacy', False):
            logger.debug(f"Node already zero-copy: {node_id}")
            return False

        # Get text
        text = node.get('text')
        if not text:
            logger.warning(f"Node has no text: {node_id}")
            return False

        # Store in wool
        text_bytes = text.encode('utf-8')
        wool_ref = self.wool.store(text_bytes, content_type='text/plain')

        # Create TextReference
        text_ref = TextReference(
            file_id=wool_ref.file_id,
            offset=0,
            length=len(text_bytes),
            encoding='utf-8'
        )

        # Update node (replace text with text_ref)
        node['text_ref'] = text_ref.to_dict()
        del node['text']  # Remove full text
        node['_is_legacy'] = False

        # Update stats
        self.stats.legacy_nodes -= 1
        self.stats.zerocopy_nodes += 1
        self.stats.migrated_nodes += 1
        self.stats.legacy_bytes -= len(text)
        self.stats.zerocopy_bytes += 88

        logger.info(f"Migrated node: {node_id}")
        return True

    def migrate_all(self, batch_size: int = 100) -> int:
        """
        Migrate all legacy nodes to zero-copy.

        Args:
            batch_size: Process nodes in batches (for progress logging)

        Returns:
            Number of nodes migrated
        """
        legacy_nodes = [
            node_id for node_id, data in self.kg.graph.nodes(data=True)
            if data.get('_is_legacy', False)
        ]

        total = len(legacy_nodes)
        migrated = 0

        logger.info(f"Starting migration of {total} legacy nodes...")

        for i, node_id in enumerate(legacy_nodes, 1):
            if self.migrate_node(node_id):
                migrated += 1

            # Progress logging
            if i % batch_size == 0:
                logger.info(f"Migrated {i}/{total} nodes ({(i/total)*100:.1f}%)")

        logger.info(f"Migration complete: {migrated}/{total} nodes migrated")
        return migrated

    def get_stats(self) -> HybridKGStats:
        """
        Get KG statistics.

        Returns:
            HybridKGStats instance
        """
        return self.stats

    def add_edges(self, edges: List[KGEdge]):
        """
        Add edges to graph.

        Args:
            edges: List of KGEdge objects
        """
        self.kg.add_edges(edges)

    def get_neighbors(self, entity: str) -> List[str]:
        """
        Get neighbors of entity.

        Args:
            entity: Entity name

        Returns:
            List of neighbor entity names
        """
        return self.kg.get_neighbors(entity)

    def search(
        self,
        query: str,
        limit: int = 10,
        include_text: bool = False
    ) -> List[Dict]:
        """
        Search for nodes matching query.

        Args:
            query: Search query (matches entities/motifs)
            limit: Maximum results
            include_text: If True, include resolved text in results

        Returns:
            List of node dictionaries
        """
        results = []

        for node_id, data in self.kg.graph.nodes(data=True):
            # Search in entities
            if any(query.lower() in entity.lower() for entity in data.get('entities', [])):
                result = {
                    'id': node_id,
                    'entities': data.get('entities', []),
                    'motifs': data.get('motifs', []),
                    'metadata': data.get('metadata', {}),
                    'is_legacy': data.get('_is_legacy', False)
                }

                if include_text:
                    result['text'] = self.get_text(node_id)

                results.append(result)

                if len(results) >= limit:
                    break

        return results

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HybridKG(total_nodes={self.stats.total_nodes}, "
            f"zerocopy={self.stats.zerocopy_nodes}, "
            f"legacy={self.stats.legacy_nodes}, "
            f"zerocopy_pct={self.stats.zerocopy_percentage:.1f}%)"
        )
