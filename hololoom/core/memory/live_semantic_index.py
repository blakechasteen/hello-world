from __future__ import annotations
"""
LiveSemanticIndex — Self-updating embedding index for the Memory Bus.

Provides Level 4 (SEMANTIC) search by maintaining a live index of
sentence-transformers embeddings over items stored in the LiteMemoryBus.
Unlike semantic_bridge.py (which wraps an external retriever), this module
owns its own index and updates it as items flow through the bus.

Model: all-MiniLM-L6-v2 (384d, Matryoshka-truncatable to 96/192)

Usage:
    from hololoom.memory.live_semantic_index import LiveSemanticIndex

    index = LiveSemanticIndex()
    bus = LiteMemoryBus(semantic_fn=index.as_semantic_fn())

    # Items indexed automatically when stored via bus — but you can
    # also index a StoredItem directly:
    index.index_item(stored_item)
"""

import logging
from collections.abc import Callable, Coroutine
from typing import Any

logger = logging.getLogger(__name__)

# Optional imports — graceful degradation (Design Principle #2)
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _HAVE_EMBEDDINGS = True
except ImportError:
    SentenceTransformer = None
    np = None
    _HAVE_EMBEDDINGS = False
    logger.warning(
        "sentence-transformers not available — LiveSemanticIndex will "
        "fall back to keyword matching"
    )


class LiveSemanticIndex:
    """Self-updating embedding index backed by all-MiniLM-L6-v2.

    Maintains parallel arrays of IDs and normalized embeddings.
    Cosine similarity = dot product (vectors are L2-normalized at encode time).

    Graceful degradation: if sentence-transformers is missing, falls back to
    Jaccard keyword matching over stored content.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        score_threshold: float = 0.0,
    ):
        """
        Args:
            model_name: Sentence-transformers model to load.
            embedding_dim: Truncate embeddings to this dimension (Matryoshka).
                           96 = fast, 192 = balanced, 384 = precise.
            score_threshold: Minimum cosine similarity to include in results.
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.score_threshold = score_threshold

        self._model: Any | None = None
        self._ids: list[str] = []
        self._embeddings: Any | None = None  # np.ndarray (N, dim) or None
        self._content_map: dict[str, str] = {}  # id → content
        self._meta_map: dict[str, dict[str, Any]] = {}  # id → metadata

        self.available = _HAVE_EMBEDDINGS
        if self.available:
            try:
                self._model = SentenceTransformer(model_name)
                logger.info("LiveSemanticIndex: loaded %s (%dd)", model_name, embedding_dim)
            except Exception as e:
                logger.warning("LiveSemanticIndex: failed to load model: %s", e)
                self._model = None
                self.available = False

    # =========================================================================
    # Embedding
    # =========================================================================

    def _embed(self, text: str) -> Any | None:
        """Embed a single text, truncate to embedding_dim, return normalized vector."""
        if not self.available or self._model is None:
            return None
        vec = self._model.encode(text, normalize_embeddings=True)
        if len(vec) > self.embedding_dim:
            vec = vec[:self.embedding_dim]
            # Re-normalize after truncation
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def _embed_batch(self, texts: list[str]) -> Any | None:
        """Embed a batch of texts. Returns (N, dim) ndarray or None."""
        if not self.available or self._model is None or not texts:
            return None
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=64)
        if vecs.shape[1] > self.embedding_dim:
            vecs = vecs[:, :self.embedding_dim]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            vecs = vecs / norms
        return vecs

    # =========================================================================
    # Index Management
    # =========================================================================

    def index_item(
        self,
        item_id: str,
        content: str,
        memory_type: str = "factual",
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Add or update a single item in the index.

        Returns True if indexed (embedding computed), False if fallback-only.
        """
        self._content_map[item_id] = content
        self._meta_map[item_id] = {
            "memory_type": memory_type,
            "importance": importance,
            **(metadata or {}),
        }

        vec = self._embed(content)
        if vec is None:
            # Still stored in content_map for keyword fallback
            return False

        if item_id in self._ids:
            # Update existing
            idx = self._ids.index(item_id)
            self._embeddings[idx] = vec
        else:
            # Append
            self._ids.append(item_id)
            if self._embeddings is None:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])

        return True

    def index_batch(
        self,
        items: list[tuple[str, str, str, float, dict[str, Any] | None]],
    ) -> int:
        """Bulk-index items: list of (id, content, memory_type, importance, metadata).

        Returns count of items successfully embedded.
        """
        if not items:
            return 0

        # Store metadata for all items (keyword fallback)
        for item_id, content, memory_type, importance, metadata in items:
            self._content_map[item_id] = content
            self._meta_map[item_id] = {
                "memory_type": memory_type,
                "importance": importance,
                **(metadata or {}),
            }

        texts = [content for _, content, _, _, _ in items]
        vecs = self._embed_batch(texts)
        if vecs is None:
            return 0

        ids = [item_id for item_id, _, _, _, _ in items]
        # Filter out items already in the index (update them)
        new_ids = []
        new_vecs = []
        for i, item_id in enumerate(ids):
            if item_id in self._ids:
                idx = self._ids.index(item_id)
                self._embeddings[idx] = vecs[i]
            else:
                new_ids.append(item_id)
                new_vecs.append(vecs[i])

        if new_ids:
            self._ids.extend(new_ids)
            new_matrix = np.array(new_vecs)
            if self._embeddings is None:
                self._embeddings = new_matrix
            else:
                self._embeddings = np.vstack([self._embeddings, new_matrix])

        return len(ids)

    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the index. Returns True if found."""
        self._content_map.pop(item_id, None)
        self._meta_map.pop(item_id, None)

        if item_id not in self._ids:
            return False

        idx = self._ids.index(item_id)
        self._ids.pop(idx)
        if self._embeddings is not None and len(self._ids) > 0:
            self._embeddings = np.delete(self._embeddings, idx, axis=0)
        else:
            self._embeddings = None
        return True

    @property
    def count(self) -> int:
        """Number of items in the index."""
        return len(self._ids)

    # =========================================================================
    # Search
    # =========================================================================

    def search_sync(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Synchronous search — returns list of dicts matching semantic_fn contract."""
        if self.available and self._embeddings is not None and len(self._ids) > 0:
            return self._search_vector(query, max_results)
        return self._search_keyword(query, max_results)

    async def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Async search — the interface LiteMemoryBus.semantic_fn expects."""
        return self.search_sync(query, max_results)

    def _search_vector(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Cosine similarity search over the embedding index."""
        query_vec = self._embed(query)
        if query_vec is None:
            return self._search_keyword(query, max_results)

        # dot product = cosine similarity (normalized vectors)
        scores = self._embeddings @ query_vec
        top_k = min(max_results, len(self._ids))
        if top_k >= len(self._ids):
            # All items fit — just sort directly
            top_indices = np.argsort(-scores)[:top_k]
        else:
            top_indices = np.argpartition(-scores, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < self.score_threshold:
                continue
            item_id = self._ids[idx]
            meta = self._meta_map.get(item_id, {})
            results.append({
                "id": item_id,
                "content": self._content_map.get(item_id, ""),
                "memory_type": meta.get("memory_type", "factual"),
                "importance": meta.get("importance", 0.5),
                "_search_score": score,
                "_resolution": "semantic",
                "_model": self.model_name,
                "_dim": self.embedding_dim,
            })

        return results

    def _search_keyword(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Jaccard keyword fallback when embeddings are unavailable."""
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return []

        scored = []
        for item_id, content in self._content_map.items():
            content_tokens = set(content.lower().split())
            if not content_tokens:
                continue
            intersection = query_tokens & content_tokens
            union = query_tokens | content_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard > 0:
                scored.append((jaccard, item_id))

        scored.sort(reverse=True)
        results = []
        for score, item_id in scored[:max_results]:
            if score < self.score_threshold:
                continue
            meta = self._meta_map.get(item_id, {})
            results.append({
                "id": item_id,
                "content": self._content_map.get(item_id, ""),
                "memory_type": meta.get("memory_type", "factual"),
                "importance": meta.get("importance", 0.5),
                "_search_score": score,
                "_resolution": "keyword_fallback",
            })

        return results

    # =========================================================================
    # Bus Integration
    # =========================================================================

    def as_semantic_fn(self) -> Callable[[str, int], Coroutine[Any, Any, list[dict]]]:
        """Return an async function matching LiteMemoryBus.semantic_fn signature.

        Usage:
            index = LiveSemanticIndex()
            bus = LiteMemoryBus(semantic_fn=index.as_semantic_fn())
        """
        return self.search

    def hook_store(self, item_id: str, stored_item: Any) -> None:
        """Call after bus.store() to keep index in sync.

        Accepts any object with .content, .memory_type, .importance, .properties
        attributes (i.e. a StoredItem).
        """
        content = getattr(stored_item, "content", None)
        if content is None:
            return
        self.index_item(
            item_id=item_id,
            content=content,
            memory_type=getattr(stored_item, "memory_type", "factual"),
            importance=getattr(stored_item, "importance", 0.5),
            metadata=getattr(stored_item, "properties", None),
        )

    def hook_delete(self, item_id: str) -> None:
        """Call after bus.delete() or bus.archive() to keep index in sync."""
        self.remove_item(item_id)

    # =========================================================================
    # Diagnostics
    # =========================================================================

    def health(self) -> dict[str, Any]:
        """Return index health status."""
        return {
            "available": self.available,
            "model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "indexed_items": self.count,
            "content_items": len(self._content_map),
            "fallback_only": not self.available,
        }


# =============================================================================
# Factory: wire index + bus in one call
# =============================================================================

def create_semantic_bus(
    model_name: str = "all-MiniLM-L6-v2",
    embedding_dim: int = 384,
    score_threshold: float = 0.0,
    bus_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, "LiveSemanticIndex"]:
    """Create a LiteMemoryBus with a live semantic index wired in.

    Returns:
        (bus, index) — the bus has semantic_fn set and a post-store hook
        registered so items are indexed automatically.

    Usage:
        from hololoom.memory.live_semantic_index import create_semantic_bus

        bus, index = create_semantic_bus()
        await bus.initialize()
        node_id = await bus.store(MemoryItem(content="...", memory_type="factual"))
        results = await bus.query(MemoryQuery(intent="...", semantic_text="..."))
    """
    from .lite_bus import LiteMemoryBus

    index = LiveSemanticIndex(
        model_name=model_name,
        embedding_dim=embedding_dim,
        score_threshold=score_threshold,
    )

    kwargs = bus_kwargs or {}
    bus = LiteMemoryBus(semantic_fn=index.as_semantic_fn(), **kwargs)
    bus._post_store_hooks.append(index.hook_store)

    return bus, index
