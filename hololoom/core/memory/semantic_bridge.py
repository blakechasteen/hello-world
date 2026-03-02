"""
Semantic Bridge — Connects the existing shard embedding retriever to the Memory Bus.

This is the Level 4 implementation for both LiteMemoryBus and MemoryBus.
Wraps RetrieverMS (multi-scale Matryoshka embeddings + BM25) into the
async fn(query_text, max_results) → List[Dict] interface that the bus expects.

Usage:
    from hololoom.memory.semantic_bridge import create_semantic_fn

    retriever = create_retriever(shards=shards, emb=embedder)
    semantic_fn = create_semantic_fn(retriever)
    bus = LiteMemoryBus(semantic_fn=semantic_fn)
"""

import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


def create_semantic_fn(
    retriever: Any,
    score_threshold: float = 0.0,
) -> Callable[[str, int], Coroutine[Any, Any, List[Dict]]]:
    """Create a Level 4 semantic search function from an existing retriever.

    Args:
        retriever: Any object with async search(query, k, fast) method
                  (e.g. RetrieverMS or anything matching the Retriever protocol)
        score_threshold: Minimum score to include (0.0 = include all)

    Returns:
        Async function compatible with LiteMemoryBus.semantic_fn
    """

    async def semantic_fn(query_text: str, max_results: int = 5) -> List[Dict]:
        try:
            hits = await retriever.search(
                query=query_text,
                k=max_results,
                fast=False,
            )
        except Exception as e:
            logger.warning("Semantic search failed: %s", e)
            return []

        results = []
        for shard, score in hits:
            if score < score_threshold:
                continue

            item = {
                "id": shard.id if hasattr(shard, 'id') else str(id(shard)),
                "content": shard.text if hasattr(shard, 'text') else str(shard),
                "memory_type": "factual",
                "importance": float(score),
                "_search_score": float(score),
                "_resolution": "semantic",
            }
            # Merge shard metadata
            if hasattr(shard, 'metadata') and shard.metadata:
                item.update(shard.metadata)

            results.append(item)

        return results

    return semantic_fn


def create_shard_semantic_fn(
    shards: List[Any],
    embedder: Any,
) -> Callable[[str, int], Coroutine[Any, Any, List[Dict]]]:
    """Create a Level 4 semantic function directly from shards + embedder.

    Convenience wrapper that creates a RetrieverMS internally.
    Use this when you have raw shards but no retriever yet.

    Args:
        shards: List of MemoryShard objects
        embedder: Matryoshka embedder instance

    Returns:
        Async function compatible with LiteMemoryBus.semantic_fn
    """
    from .cache import create_retriever

    retriever = create_retriever(shards=shards, emb=embedder)
    return create_semantic_fn(retriever)
