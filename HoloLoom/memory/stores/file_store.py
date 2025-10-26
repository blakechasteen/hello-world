"""
File-Based Memory Store - Simple Persistence Fallback
=====================================================
When Qdrant/Neo4j unavailable, fall back to file persistence.

Architecture:
- memories.jsonl: One JSON object per line (append-only)
- embeddings.npy: Numpy array of embeddings
- Simple linear scan for retrieval (fine for <10k memories)

Benefits:
- Zero dependencies (just stdlib + numpy)
- Portable (works everywhere)
- Human-readable (JSONL format)
- Fast enough for demos
"""

from __future__ import annotations  # Enable string type hints

import logging
import json
import os
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

import numpy as np

# Handle imports for both package and standalone usage
try:
    from ..protocol import Memory, MemoryQuery, RetrievalResult, Strategy, MemoryStore
except ImportError:
    # Standalone execution - will be imported after path setup
    pass


logger = logging.getLogger(__name__)


class FileMemoryStore:
    """
    File-based memory store with async interface.

    Storage:
    - {data_dir}/memories.jsonl: Memory objects (one per line)
    - {data_dir}/embeddings.npy: Embedding vectors

    Thread-safe via asyncio locks.
    """

    def __init__(
        self,
        data_dir: str = "./memory_data",
        embedder = None
    ):
        """
        Initialize file store.

        Args:
            data_dir: Directory for data files
            embedder: Optional embedder for semantic search
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.memories_path = self.data_dir / "memories.jsonl"
        self.embeddings_path = self.data_dir / "embeddings.npy"

        self.embedder = embedder

        # In-memory cache (loaded on init)
        self.memories: List[Memory] = []
        self.embeddings: Optional[np.ndarray] = None

        # Async lock
        self.lock = asyncio.Lock()

        # Load existing data
        self._load()

        logger.info(
            f"File store initialized: {len(self.memories)} memories in {data_dir}"
        )

    def _load(self):
        """Load memories and embeddings from disk."""
        # Load memories
        if self.memories_path.exists():
            self.memories = []
            with open(self.memories_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        mem = Memory.from_dict(data)
                        self.memories.append(mem)
                    except Exception as e:
                        logger.warning(f"Failed to load memory: {e}")

            logger.info(f"Loaded {len(self.memories)} memories from disk")

        # Load embeddings
        if self.embeddings_path.exists():
            try:
                self.embeddings = np.load(self.embeddings_path)
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")

                # Verify shape matches
                if len(self.embeddings) != len(self.memories):
                    logger.warning(
                        f"Embedding count mismatch: {len(self.embeddings)} != {len(self.memories)}"
                    )
                    self.embeddings = None
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                self.embeddings = None

    def _save_memory(self, memory: Memory):
        """Append memory to file."""
        with open(self.memories_path, 'a', encoding='utf-8') as f:
            json.dump(memory.to_dict(), f)
            f.write('\n')

    def _save_embeddings(self):
        """Save all embeddings to disk."""
        if self.embeddings is not None:
            np.save(self.embeddings_path, self.embeddings)

    async def store(self, memory: Memory) -> str:
        """Store a memory (async interface)."""
        async with self.lock:
            # Add to in-memory cache
            self.memories.append(memory)

            # Compute embedding if embedder available
            if self.embedder is not None:
                # Run embedder in thread pool (blocking operation)
                loop = asyncio.get_event_loop()
                embed = await loop.run_in_executor(
                    None,
                    lambda: self.embedder.encode([memory.text])[0]
                )

                # Add to embeddings array
                if self.embeddings is None:
                    self.embeddings = embed.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, embed])

            # Save to disk (blocking I/O in thread pool)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._save_memory, memory)

            if self.embeddings is not None:
                await loop.run_in_executor(None, self._save_embeddings)

            logger.info(f"Stored memory: {memory.id} (total: {len(self.memories)})")
            return memory.id

    async def retrieve(
        self,
        query: MemoryQuery,
        strategy = None  # Type hint removed to avoid import issues
    ) -> RetrievalResult:
        """
        Retrieve memories based on query.

        Strategies:
        - TEMPORAL: Most recent memories
        - SEMANTIC: Cosine similarity (requires embedder)
        - FUSED: Combine temporal + semantic (default)
        """
        # Default to FUSED if not specified
        if strategy is None:
            strategy_name = "fused"
        else:
            strategy_name = strategy.value if hasattr(strategy, 'value') else str(strategy)

        async with self.lock:
            if len(self.memories) == 0:
                return RetrievalResult(
                    memories=[],
                    scores=[],
                    strategy_used=strategy_name,
                    metadata={'total_memories': 0}
                )

            # Compute scores based on strategy
            if strategy_name == "temporal":
                scores = self._temporal_scores()

            elif strategy_name == "semantic":
                if self.embedder is None or self.embeddings is None:
                    logger.warning("No embedder available, falling back to temporal")
                    scores = self._temporal_scores()
                else:
                    loop = asyncio.get_event_loop()
                    scores = await loop.run_in_executor(
                        None,
                        self._semantic_scores,
                        query.text
                    )

            elif strategy_name == "fused":
                # Combine temporal + semantic (70% semantic, 30% temporal)
                temporal_scores = self._temporal_scores()

                if self.embedder is not None and self.embeddings is not None:
                    loop = asyncio.get_event_loop()
                    semantic_scores = await loop.run_in_executor(
                        None,
                        self._semantic_scores,
                        query.text
                    )

                    # Normalize both
                    t_norm = self._normalize_scores(temporal_scores)
                    s_norm = self._normalize_scores(semantic_scores)

                    scores = 0.7 * s_norm + 0.3 * t_norm
                else:
                    scores = temporal_scores

            else:
                # Default to temporal
                scores = self._temporal_scores()

            # Get top-K
            top_k = min(query.limit, len(self.memories))
            top_indices = np.argsort(scores)[-top_k:][::-1]

            top_memories = [self.memories[i] for i in top_indices]
            top_scores = [float(scores[i]) for i in top_indices]

            return RetrievalResult(
                memories=top_memories,
                scores=top_scores,
                strategy_used=strategy_name,
                metadata={
                    'total_memories': len(self.memories),
                    'backend': 'file',
                    'has_embeddings': self.embeddings is not None
                }
            )

    def _temporal_scores(self) -> np.ndarray:
        """Score based on recency (newer = higher score)."""
        # Simple: index-based (later = newer)
        scores = np.arange(len(self.memories), dtype=float)

        # Normalize to [0, 1]
        if len(scores) > 1:
            scores = scores / scores.max()

        return scores

    def _semantic_scores(self, query_text: str) -> np.ndarray:
        """Score based on cosine similarity."""
        # Encode query
        query_embed = self.embedder.encode([query_text])[0]

        # Normalize
        query_norm = query_embed / (np.linalg.norm(query_embed) + 1e-8)
        mem_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        )

        # Cosine similarity
        scores = mem_norm @ query_norm

        return scores

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]."""
        if len(scores) == 0:
            return scores

        min_score = scores.min()
        max_score = scores.max()

        if max_score - min_score < 1e-8:
            return np.ones_like(scores)

        return (scores - min_score) / (max_score - min_score)

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Note: For file store, we need to rewrite the entire file.
        This is slow but simple.
        """
        async with self.lock:
            # Find index
            idx = None
            for i, mem in enumerate(self.memories):
                if mem.id == memory_id:
                    idx = i
                    break

            if idx is None:
                return False

            # Remove from cache
            del self.memories[idx]

            if self.embeddings is not None:
                self.embeddings = np.delete(self.embeddings, idx, axis=0)

            # Rewrite files (in thread pool)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._rewrite_files)

            logger.info(f"Deleted memory: {memory_id}")
            return True

    def _rewrite_files(self):
        """Rewrite memory and embedding files."""
        # Rewrite memories
        with open(self.memories_path, 'w', encoding='utf-8') as f:
            for mem in self.memories:
                json.dump(mem.to_dict(), f)
                f.write('\n')

        # Rewrite embeddings
        if self.embeddings is not None:
            np.save(self.embeddings_path, self.embeddings)

    async def health_check(self) -> Dict:
        """Check store health."""
        return {
            'status': 'healthy',
            'backend': 'file',
            'data_dir': str(self.data_dir),
            'memory_count': len(self.memories),
            'has_embeddings': self.embeddings is not None,
            'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else None
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    import os

    # Add repo root to path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )))
    sys.path.insert(0, repo_root)

    # Now import after path is set
    from HoloLoom.memory.protocol import Memory, MemoryQuery, RetrievalResult, Strategy
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

    async def main():
        print("="*80)
        print("FILE MEMORY STORE DEMO")
        print("="*80)

        # Create embedder
        print("\nInitializing embedder...")
        embedder = MatryoshkaEmbeddings(sizes=[96, 192, 384])

        # Create store
        print("Initializing file store...")
        store = FileMemoryStore(
            data_dir="./test_memory_data",
            embedder=embedder
        )

        # Add some memories
        print("\nAdding memories...")
        knowledge_base = [
            ("Machine learning is a subset of AI", {"source": "ml_docs"}),
            ("Python is a popular programming language", {"source": "python_docs"}),
            ("Neural networks mimic biological neurons", {"source": "dl_docs"}),
            ("Reinforcement learning uses rewards", {"source": "rl_docs"}),
        ]

        for i, (text, metadata) in enumerate(knowledge_base):
            mem = Memory(
                id=f"mem_{i}",
                text=text,
                timestamp=datetime.now(),
                context={},
                metadata=metadata
            )
            await store.store(mem)

        # Query
        print("\nQuerying memories...")
        query = MemoryQuery(
            text="What is machine learning?",
            limit=3
        )

        result = await store.retrieve(query, strategy=Strategy.FUSED)

        print(f"\nTop {len(result.memories)} results:")
        for i, (mem, score) in enumerate(zip(result.memories, result.scores), 1):
            print(f"{i}. [{score:.3f}] {mem.text}")
            print(f"   Source: {mem.metadata.get('source', 'unknown')}")

        print(f"\nMetadata: {result.metadata}")

        # Health check
        print("\n" + "="*80)
        print("HEALTH CHECK")
        print("="*80)
        health = await store.health_check()
        for key, value in health.items():
            print(f"{key}: {value}")

        print("\nFile store operational!")

    # Run
    asyncio.run(main())
