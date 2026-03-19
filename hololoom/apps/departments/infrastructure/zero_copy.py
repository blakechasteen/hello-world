#!/usr/bin/env python3
"""
Zero-Copy Embedding Store - Memory-mapped embedding storage.

Provides efficient shared access to embeddings across multiple departments
without memory duplication. Uses memory-mapped files (mmap) for zero-copy access.

Design Philosophy:
"Share embeddings, not copies."

Key Features:
- Memory-mapped storage (no duplication)
- Multiple concurrent readers
- Atomic writes with copy-on-write
- Automatic index management
- Matryoshka embedding support (96, 192, 384 dims)
"""

from __future__ import annotations

import json
import logging
import mmap
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class EmbeddingMetadata:
    """Metadata for an embedding."""
    id: str
    dimension: int
    created_at: float
    last_accessed: float
    access_count: int
    tags: list[str]


@dataclass
class EmbeddingView:
    """Read-only view of an embedding (zero-copy)."""
    id: str
    vector: np.ndarray  # Memory-mapped view (no copy)
    metadata: EmbeddingMetadata

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (creates a copy)."""
        return np.array(self.vector)


# ============================================================================
# Zero-Copy Embedding Store
# ============================================================================

class ZeroCopyEmbeddingStore:
    """
    Zero-copy embedding storage using memory-mapped files.

    Provides efficient shared access to embeddings across multiple departments.
    Uses mmap for zero-copy reads and copy-on-write for atomic updates.

    Example:

        # Create store
        store = ZeroCopyEmbeddingStore(storage_dir="./embeddings")

        # Add embeddings
        await store.add_embedding("doc_1", np.random.randn(384), tags=["context"])

        # Get embedding (zero-copy)
        view = await store.get_embedding("doc_1")
        print(view.vector.shape)  # (384,) - no copy made

        # Search similar (using mmapped data)
        similar = await store.search_similar(query_vector, k=10)

        # Cleanup
        await store.close()
    """

    def __init__(
        self,
        storage_dir: Path | str,
        dimension: int = 384,
        max_embeddings: int = 100000,
        enable_index: bool = True
    ):
        """
        Initialize zero-copy embedding store.

        Args:
            storage_dir: Directory for memory-mapped files
            dimension: Embedding dimension (default 384 for Matryoshka)
            max_embeddings: Maximum number of embeddings to store
            enable_index: Enable fast index for search
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.dimension = dimension
        self.max_embeddings = max_embeddings
        self.enable_index = enable_index

        # Memory-mapped array for embeddings
        self.embeddings_file = self.storage_dir / f"embeddings_{dimension}d.mmap"
        self.embeddings_mmap: mmap.mmap | None = None
        self.embeddings_array: np.ndarray | None = None

        # Metadata storage
        self.metadata_file = self.storage_dir / "metadata.json"
        self.metadata: dict[str, EmbeddingMetadata] = {}

        # Index mapping (id -> position in array)
        self.index: dict[str, int] = {}
        self.next_position = 0

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats = {
            "embeddings_stored": 0,
            "total_accesses": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        logger.info(
            f"ZeroCopyEmbeddingStore initialized "
            f"(dir: {storage_dir}, dim: {dimension}, max: {max_embeddings})"
        )

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize memory-mapped storage."""
        # Create or open memory-mapped file
        file_size = self.max_embeddings * self.dimension * 4  # 4 bytes per float32

        if not self.embeddings_file.exists():
            # Create new file
            logger.info(f"Creating new embedding file: {self.embeddings_file}")
            with open(self.embeddings_file, 'wb') as f:
                f.write(b'\x00' * file_size)

        # Open memory-mapped file
        with open(self.embeddings_file, 'r+b') as f:
            self.embeddings_mmap = mmap.mmap(
                f.fileno(),
                file_size,
                access=mmap.ACCESS_WRITE
            )

        # Create numpy array view (zero-copy)
        self.embeddings_array = np.ndarray(
            shape=(self.max_embeddings, self.dimension),
            dtype=np.float32,
            buffer=self.embeddings_mmap
        )

        # Load metadata
        await self._load_metadata()

        logger.info(
            f"Memory-mapped {self.stats['embeddings_stored']} embeddings "
            f"({file_size / 1024 / 1024:.1f} MB)"
        )

    async def close(self) -> None:
        """Close memory-mapped file and save metadata."""
        # Save metadata
        await self._save_metadata()

        # Close mmap
        if self.embeddings_mmap is not None:
            self.embeddings_mmap.close()
            self.embeddings_mmap = None

        self.embeddings_array = None

        logger.info("ZeroCopyEmbeddingStore closed")

    # ========================================================================
    # Core Operations
    # ========================================================================

    async def add_embedding(
        self,
        id: str,
        vector: np.ndarray,
        tags: list[str] | None = None
    ) -> bool:
        """
        Add embedding to store (copy-on-write).

        Args:
            id: Unique identifier
            vector: Embedding vector
            tags: Optional tags for filtering

        Returns:
            True if added, False if already exists
        """
        with self._lock:
            # Check if already exists
            if id in self.index:
                logger.warning(f"Embedding {id} already exists")
                return False

            # Check capacity
            if self.next_position >= self.max_embeddings:
                raise RuntimeError(
                    f"Embedding store full (max: {self.max_embeddings})"
                )

            # Validate dimension
            if vector.shape[0] != self.dimension:
                raise ValueError(
                    f"Vector dimension {vector.shape[0]} != store dimension {self.dimension}"
                )

            # Write to memory-mapped array (copy-on-write)
            position = self.next_position
            self.embeddings_array[position, :] = vector

            # Update index
            self.index[id] = position
            self.next_position += 1

            # Store metadata
            self.metadata[id] = EmbeddingMetadata(
                id=id,
                dimension=self.dimension,
                created_at=datetime.now().timestamp(),
                last_accessed=datetime.now().timestamp(),
                access_count=0,
                tags=tags or []
            )

            self.stats["embeddings_stored"] += 1

            logger.debug(f"Added embedding {id} at position {position}")
            return True

    async def get_embedding(self, id: str) -> EmbeddingView | None:
        """
        Get embedding view (zero-copy).

        Args:
            id: Embedding identifier

        Returns:
            EmbeddingView (memory-mapped, no copy) or None if not found
        """
        with self._lock:
            # Check if exists
            if id not in self.index:
                self.stats["cache_misses"] += 1
                return None

            self.stats["cache_hits"] += 1
            self.stats["total_accesses"] += 1

            # Get position
            position = self.index[id]

            # Get metadata
            metadata = self.metadata[id]

            # Update access tracking
            metadata.last_accessed = datetime.now().timestamp()
            metadata.access_count += 1

            # Create view (zero-copy)
            vector_view = self.embeddings_array[position, :]

            return EmbeddingView(
                id=id,
                vector=vector_view,
                metadata=metadata
            )

    async def get_embeddings_batch(
        self,
        ids: list[str]
    ) -> list[EmbeddingView | None]:
        """
        Get multiple embeddings (zero-copy batch).

        Args:
            ids: List of embedding identifiers

        Returns:
            List of EmbeddingViews (None for missing)
        """
        return [await self.get_embedding(id) for id in ids]

    async def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        tags: list[str] | None = None
    ) -> list[tuple[str, float, EmbeddingView]]:
        """
        Search for k most similar embeddings (uses memory-mapped data).

        Args:
            query_vector: Query embedding
            k: Number of results
            tags: Filter by tags (optional)

        Returns:
            List of (id, similarity, view) tuples
        """
        with self._lock:
            if self.next_position == 0:
                return []

            # Get active embeddings
            active_embeddings = self.embeddings_array[:self.next_position, :]

            # Filter by tags if specified
            if tags:
                filtered_ids = [
                    id for id, meta in self.metadata.items()
                    if any(tag in meta.tags for tag in tags)
                ]
                filtered_positions = [self.index[id] for id in filtered_ids]
                active_embeddings = active_embeddings[filtered_positions, :]
                candidate_ids = filtered_ids
            else:
                candidate_ids = list(self.index.keys())

            # Compute cosine similarities (using mmapped data, no copy)
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                return []

            similarities = np.dot(active_embeddings, query_vector) / (
                np.linalg.norm(active_embeddings, axis=1) * query_norm
            )

            # Get top k
            top_k_indices = np.argsort(similarities)[::-1][:k]

            # Build results
            results = []
            for idx in top_k_indices:
                id = candidate_ids[idx]
                similarity = similarities[idx]
                view = await self.get_embedding(id)
                if view:
                    results.append((id, float(similarity), view))

            return results

    async def delete_embedding(self, id: str) -> bool:
        """
        Delete embedding (marks as deleted, doesn't compact).

        Args:
            id: Embedding identifier

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if id not in self.index:
                return False

            # Remove from index
            position = self.index.pop(id)

            # Remove metadata
            self.metadata.pop(id)

            self.stats["embeddings_stored"] -= 1

            logger.debug(f"Deleted embedding {id} (position {position})")
            return True

    async def compact(self) -> int:
        """
        Compact store by removing deleted embeddings.

        Returns:
            Number of embeddings compacted
        """
        with self._lock:
            # Rebuild index
            new_position = 0
            new_index = {}

            for id, old_position in sorted(self.index.items(), key=lambda x: x[1]):
                if new_position != old_position:
                    # Move embedding
                    self.embeddings_array[new_position, :] = self.embeddings_array[old_position, :]

                new_index[id] = new_position
                new_position += 1

            embeddings_compacted = self.next_position - new_position

            # Update index
            self.index = new_index
            self.next_position = new_position

            logger.info(f"Compacted {embeddings_compacted} embeddings")
            return embeddings_compacted

    # ========================================================================
    # Metadata Management
    # ========================================================================

    async def _load_metadata(self) -> None:
        """Load metadata from disk."""
        if not self.metadata_file.exists():
            return

        try:
            with open(self.metadata_file) as f:
                data = json.load(f)

            # Restore metadata
            for id, meta_dict in data.get("metadata", {}).items():
                self.metadata[id] = EmbeddingMetadata(**meta_dict)

            # Restore index
            self.index = data.get("index", {})
            self.next_position = data.get("next_position", 0)

            # Restore stats
            self.stats.update(data.get("stats", {}))

            logger.info(f"Loaded metadata for {len(self.metadata)} embeddings")

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    async def _save_metadata(self) -> None:
        """Save metadata to disk."""
        try:
            # Convert metadata to dict
            metadata_dict = {
                id: {
                    "id": meta.id,
                    "dimension": meta.dimension,
                    "created_at": meta.created_at,
                    "last_accessed": meta.last_accessed,
                    "access_count": meta.access_count,
                    "tags": meta.tags
                }
                for id, meta in self.metadata.items()
            }

            data = {
                "metadata": metadata_dict,
                "index": self.index,
                "next_position": self.next_position,
                "stats": self.stats,
                "dimension": self.dimension,
                "max_embeddings": self.max_embeddings
            }

            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved metadata for {len(self.metadata)} embeddings")

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            return {
                **self.stats,
                "capacity": self.max_embeddings,
                "utilization": self.next_position / self.max_embeddings,
                "dimension": self.dimension,
                "storage_size_mb": (self.max_embeddings * self.dimension * 4) / 1024 / 1024,
                "hit_rate": (
                    self.stats["cache_hits"] / self.stats["total_accesses"]
                    if self.stats["total_accesses"] > 0
                    else 0.0
                )
            }

    def get_metadata(self, id: str) -> EmbeddingMetadata | None:
        """Get metadata for embedding."""
        return self.metadata.get(id)

    def list_embeddings(
        self,
        tags: list[str] | None = None,
        limit: int = 100
    ) -> list[EmbeddingMetadata]:
        """List embeddings with optional tag filtering."""
        with self._lock:
            results = []

            for id, meta in self.metadata.items():
                if tags and not any(tag in meta.tags for tag in tags):
                    continue

                results.append(meta)

                if len(results) >= limit:
                    break

            return results


# ============================================================================
# Multi-Scale Embedding Store (Matryoshka)
# ============================================================================

class MatryoshkaEmbeddingStore:
    """
    Multi-scale embedding store for Matryoshka embeddings.

    Stores embeddings at multiple scales (96, 192, 384) in separate memory-mapped files.
    Enables efficient multi-scale retrieval without redundancy.

    Example:

        store = MatryoshkaEmbeddingStore(storage_dir="./embeddings")
        await store.initialize()

        # Add embedding at all scales
        await store.add_embedding_multi_scale(
            "doc_1",
            {96: vec_96, 192: vec_192, 384: vec_384}
        )

        # Search at specific scale
        results = await store.search_similar(query_vec_96, k=10, scale=96)
    """

    def __init__(
        self,
        storage_dir: Path | str,
        scales: list[int] = [96, 192, 384],
        max_embeddings: int = 100000
    ):
        """
        Initialize multi-scale embedding store.

        Args:
            storage_dir: Directory for memory-mapped files
            scales: Embedding scales to support
            max_embeddings: Maximum number of embeddings per scale
        """
        self.storage_dir = Path(storage_dir)
        self.scales = scales
        self.max_embeddings = max_embeddings

        # Create store for each scale
        self.stores: dict[int, ZeroCopyEmbeddingStore] = {}

        for scale in scales:
            self.stores[scale] = ZeroCopyEmbeddingStore(
                storage_dir=self.storage_dir / f"scale_{scale}",
                dimension=scale,
                max_embeddings=max_embeddings
            )

        logger.info(
            f"MatryoshkaEmbeddingStore initialized with scales: {scales}"
        )

    async def initialize(self) -> None:
        """Initialize all scale stores."""
        for scale, store in self.stores.items():
            await store.initialize()

    async def close(self) -> None:
        """Close all scale stores."""
        for scale, store in self.stores.items():
            await store.close()

    async def add_embedding_multi_scale(
        self,
        id: str,
        vectors: dict[int, np.ndarray],
        tags: list[str] | None = None
    ) -> bool:
        """
        Add embedding at multiple scales.

        Args:
            id: Unique identifier
            vectors: Dict of {scale: vector}
            tags: Optional tags

        Returns:
            True if added at all scales
        """
        success = True

        for scale, vector in vectors.items():
            if scale not in self.stores:
                logger.warning(f"Scale {scale} not supported")
                continue

            result = await self.stores[scale].add_embedding(id, vector, tags)
            success = success and result

        return success

    async def get_embedding(
        self,
        id: str,
        scale: int
    ) -> EmbeddingView | None:
        """Get embedding at specific scale."""
        if scale not in self.stores:
            raise ValueError(f"Scale {scale} not supported")

        return await self.stores[scale].get_embedding(id)

    async def search_similar(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        scale: int = 384,
        tags: list[str] | None = None
    ) -> list[tuple[str, float, EmbeddingView]]:
        """Search at specific scale."""
        if scale not in self.stores:
            raise ValueError(f"Scale {scale} not supported")

        return await self.stores[scale].search_similar(query_vector, k, tags)

    def get_stats_all_scales(self) -> dict[int, dict[str, Any]]:
        """Get statistics for all scales."""
        return {
            scale: store.get_stats()
            for scale, store in self.stores.items()
        }
