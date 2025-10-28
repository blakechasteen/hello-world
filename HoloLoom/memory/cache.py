"""
HoloLoom Memory Management - Cache & Retrieval
===============================================
Multi-scale retrieval with working memory, episodic buffer, and persistence.

This is a "warp thread" module - independent memory management.

Architecture:
- Protocol-based design (Retriever, MemoryStore)
- Multi-scale vector retrieval with BM25 fusion
- Working memory cache + episodic buffer
- Async persistence to PDV/MemoAI
- Zero dependencies on other HoloLoom modules (except types, embedding)

Philosophy:
Memory is the "loom's yarn reserve" - what we've woven before and can access quickly.
We maintain multiple memory tiers for different access patterns and persistence needs.
"""

import json
import time
import asyncio
import warnings
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Protocol

import numpy as np

# Import only from shared types and embedding
from HoloLoom.documentation.types import Query, Context, Features, Vector
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

# Optional BM25 dependency
try:
    from rank_bm25 import BM25Okapi
    _HAVE_BM25 = True
except ImportError:
    BM25Okapi = None
    _HAVE_BM25 = False
    warnings.warn("rank-bm25 not available. Install with: pip install rank-bm25")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MemoryShard:
    """
    A unit of memory - can be a document, conversation turn, or knowledge snippet.
    
    Shards are the atomic units we retrieve and compose into context.
    """
    id: str
    text: str
    episode: str  # Episode/session this shard belongs to
    entities: List[str] = field(default_factory=list)
    motifs: List[str] = field(default_factory=list)
    scales: Dict[str, List[float]] = field(default_factory=dict)  # Pre-computed embeddings
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            "id": self.id,
            "text": self.text,
            "episode": self.episode,
            "entities": self.entities,
            "motifs": self.motifs,
            "scales": self.scales,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryShard':
        """Deserialize from storage."""
        return cls(
            id=data.get("id", ""),
            text=data.get("text", ""),
            episode=data.get("episode", ""),
            entities=data.get("entities", []),
            motifs=data.get("motifs", []),
            scales=data.get("scales", {}),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# Protocols
# ============================================================================

class Retriever(Protocol):
    """Protocol for retrieval implementations."""
    
    async def search(
        self,
        query: str,
        k: int = 6,
        fast: bool = False
    ) -> List[Tuple[MemoryShard, float]]:
        """
        Search for relevant memory shards.
        
        Args:
            query: Query text
            k: Number of results to return
            fast: Use fast mode (smallest scale only)
            
        Returns:
            List of (shard, score) tuples, sorted by relevance
        """
        ...


# ============================================================================
# Multi-Scale Retriever with BM25 Fusion
# ============================================================================

@dataclass
class RetrieverMS:
    """
    Multi-scale retriever using Matryoshka embeddings + BM25 fusion.
    
    Strategy:
    - Fast mode: Use smallest scale only (96d) for speed
    - Full mode: Fuse all scales (96d + 192d + 384d) + BM25 for quality
    
    Scoring:
    - Each scale gets normalized scores (z-score → sigmoid)
    - Weighted fusion based on scale importance
    - Optional BM25 boost for lexical matching
    
    This implements coarse-to-fine retrieval: quick filtering with small
    embeddings, refined ranking with large embeddings.
    """
    
    shards: List[MemoryShard]
    emb: MatryoshkaEmbeddings
    fusion_weights: Optional[Dict[int, float]] = None
    bm25_weight: float = 0.15  # BM25 contribution to final score
    
    def __post_init__(self):
        self.texts = [s.text for s in self.shards]
        
        # Refresh embedder for this corpus
        self.emb.refresh_runtime_qr(self.texts)
        
        # Pre-compute embeddings at all scales
        self.vecs_per_scale: Dict[int, np.ndarray] = {}
        for d in self.emb.sizes:
            self.vecs_per_scale[d] = self.emb.encode_scales(self.texts, size=d)
        
        # Set default fusion weights if not provided
        if self.fusion_weights is None:
            # Default: larger scales get more weight
            n_scales = len(self.emb.sizes)
            self.fusion_weights = {
                d: (i + 1) / sum(range(1, n_scales + 1))
                for i, d in enumerate(self.emb.sizes)
            }
        
        # Initialize BM25 if available
        if _HAVE_BM25:
            tokenized = [t.lower().split() for t in self.texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores to [0, 1] using z-score + sigmoid.
        
        This makes scores from different scales comparable.
        """
        mu = float(scores.mean())
        sd = float(scores.std() + 1e-9)
        z = (scores - mu) / sd
        return 1.0 / (1.0 + np.exp(-z))
    
    async def search(
        self,
        query: str,
        k: int = 6,
        fast: bool = False
    ) -> List[Tuple[MemoryShard, float]]:
        """
        Search for relevant shards using multi-scale retrieval.
        
        Args:
            query: Query text
            k: Number of results
            fast: If True, use only smallest scale (fastest)
            
        Returns:
            List of (shard, score) tuples
        """
        if fast:
            return await self._fast_search(query, k)
        else:
            return await self._fused_search(query, k)
    
    async def _fast_search(self, query: str, k: int) -> List[Tuple[MemoryShard, float]]:
        """
        Fast retrieval using smallest scale only.
        
        Use for:
        - High-throughput scenarios
        - Initial filtering in multi-stage retrieval
        - When speed > accuracy
        """
        d = min(self.emb.sizes)  # Smallest dimension
        mat = self.vecs_per_scale[d]
        
        # Encode query at this scale
        q = self.emb.encode_scales([query], size=d)[0]
        
        # Compute similarities
        scores = mat @ q
        scores = self._normalize(scores)
        
        # Get top-k
        idx = np.argsort(-scores)[:k]
        return [(self.shards[i], float(scores[i])) for i in idx]
    
    async def _fused_search(self, query: str, k: int) -> List[Tuple[MemoryShard, float]]:
        """
        Full retrieval with multi-scale fusion + BM25.
        
        Use for:
        - High-quality retrieval
        - When accuracy > speed
        - Final ranking in multi-stage systems
        """
        fused = np.zeros(len(self.texts))
        
        # Fuse scores from all scales
        for d, mat in self.vecs_per_scale.items():
            # Encode query at this scale
            q = self.emb.encode_scales([query], size=d)[0]
            
            # Compute similarities
            scores = mat @ q
            
            # Normalize and weight
            scores = self._normalize(scores)
            weight = self.fusion_weights.get(d, 0.0)
            fused += weight * scores
        
        # Add BM25 scores if available
        if self.bm25:
            bm_scores = self.bm25.get_scores(query.lower().split())
            bm_scores = self._normalize(bm_scores)
            fused = (1 - self.bm25_weight) * fused + self.bm25_weight * bm_scores
        
        # Get top-k
        idx = np.argsort(-fused)[:k]
        return [(self.shards[i], float(fused[i])) for i in idx]


# ============================================================================
# Persistence Clients
# ============================================================================

@dataclass
class PDVClient:
    """
    Personal Data Vault client - stores raw memory shards.
    
    PDV is the "long-term memory" - durable storage of all interactions.
    Writes are append-only JSONL for simplicity and durability.
    """
    
    root: str = "data"
    
    def __post_init__(self):
        self.root_path = Path(self.root)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.shard_file = self.root_path / "pdv_shards.jsonl"
    
    async def store_shard(self, shard: MemoryShard):
        """
        Store a memory shard to PDV.
        
        Async to not block the main pipeline.
        """
        with self.shard_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps(shard.to_dict()) + "\n")
    
    async def load_all_shards(self) -> List[MemoryShard]:
        """Load all shards from PDV."""
        if not self.shard_file.exists():
            return []
        
        shards = []
        with self.shard_file.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    shards.append(MemoryShard.from_dict(data))
                except Exception as e:
                    warnings.warn(f"Failed to load shard: {e}")
        
        return shards


@dataclass
class MemoAIClient:
    """
    MemoAI vector store client - stores pre-computed embeddings.
    
    MemoAI is the "semantic index" - fast vector retrieval.
    Stores embeddings at multiple scales for Matryoshka retrieval.
    """
    
    root: str = "data"
    
    def __post_init__(self):
        self.root_path = Path(self.root)
        self.root_path.mkdir(parents=True, exist_ok=True)
        self.vec_file = self.root_path / "memoai_vectors.jsonl"
    
    async def upsert_vectors(
        self,
        shard_id: str,
        scale_vectors: Dict[str, List[float]]
    ):
        """
        Upsert pre-computed vectors for a shard.
        
        Args:
            shard_id: Unique shard identifier
            scale_vectors: Dict mapping scale (str) to vector (list)
        """
        rec = {
            "id": shard_id,
            "vectors": scale_vectors,
            "ts": int(time.time())
        }
        
        with self.vec_file.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + "\n")


# ============================================================================
# Memory Manager - Orchestrates All Memory Tiers
# ============================================================================

@dataclass
class MemoryManager:
    """
    Unified memory management with multiple tiers:
    
    1. Working Memory - Hot cache of recent queries (hash → Context)
    2. Episodic Buffer - Recent interactions (bounded deque)
    3. PDV - Persistent raw storage (disk)
    4. MemoAI - Persistent vector index (disk)
    
    Architecture:
    - Retrieval checks working memory first (O(1) cache hit)
    - Falls back to vector retrieval if cache miss
    - Asynchronously persists to PDV/MemoAI (non-blocking)
    """
    
    retriever: RetrieverMS
    pdv: PDVClient
    memo: MemoAIClient
    working_memory_size: int = 100
    episodic_buffer_size: int = 100
    
    def __post_init__(self):
        # Tier 1: Working memory (fast cache)
        self.working_memory: Dict[int, Context] = {}
        
        # Tier 2: Episodic buffer (recent interactions)
        self.episodic_buffer = deque(maxlen=self.episodic_buffer_size)
        
        # Async persistence queue
        self.persistence_queue = asyncio.Queue()
        
        # Start background archiver
        self._archiver_task = None
        self._start_archiver()
    
    def _start_archiver(self):
        """Start background task for async persistence."""
        async def archiver():
            while True:
                try:
                    item = await self.persistence_queue.get()
                    
                    # Persist to both PDV (raw) and MemoAI (vectors)
                    await asyncio.gather(
                        self.pdv.store_shard(item['shard']),
                        self.memo.upsert_vectors(item['shard'].id, item['shard'].scales)
                    )
                    
                    self.persistence_queue.task_done()
                except Exception as e:
                    warnings.warn(f"Persistence failed: {e}")
        
        self._archiver_task = asyncio.create_task(archiver())
    
    async def retrieve(self, query: Query, kg_sub, fast: bool = False) -> Context:
        """
        Retrieve context for a query.
        
        Pipeline:
        1. Check working memory cache (by query hash)
        2. If miss, perform vector retrieval
        3. Build context with hits + KG subgraph
        4. Cache in working memory
        
        Args:
            query: Query object
            kg_sub: Knowledge graph subgraph for this query
            fast: Use fast retrieval mode
            
        Returns:
            Context with retrieved shards and metadata
        """
        # Check cache first
        q_hash = hash(query.text)
        if q_hash in self.working_memory:
            return self.working_memory[q_hash]
        
        # Cache miss - perform retrieval
        hits = await self.retriever.search(query.text, k=6, fast=fast)
        shard_texts = [s.text for s, _ in hits]
        
        # Calculate relevance score
        relevance = float(np.mean([score for _, score in hits])) if hits else 0.0
        
        # Build context
        context = Context(
            hits=[(s, score) for s, score in hits],
            kg_sub=kg_sub,
            shard_texts=shard_texts,
            relevance=relevance
        )
        
        # Cache in working memory
        self.working_memory[q_hash] = context
        
        # Prune cache if too large
        if len(self.working_memory) > self.working_memory_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.working_memory))
            del self.working_memory[oldest_key]
        
        return context
    
    async def persist(
        self,
        query: Query,
        results: Dict,
        features: Features
    ):
        """
        Persist query and results to long-term memory.
        
        Non-blocking - queues for background persistence.
        
        Args:
            query: The query that was processed
            results: Results/actions taken
            features: Extracted features
        """
        # Add to episodic buffer
        self.episodic_buffer.append({
            'query': query.text,
            'results': results,
            'features': features,
            'timestamp': time.time()
        })
        
        # Extract entities (simple heuristic: capitalized words)
        entities = [w for w in query.text.split() if w and w[0].isupper()]
        
        # Pre-compute embeddings for all scales
        scales = {
            str(d): self.retriever.emb.encode_scales([query.text], size=d)[0].tolist()
            for d in self.retriever.emb.sizes
        }
        
        # Create memory shard
        shard = MemoryShard(
            id=f"q_{hash(query.text)}",
            text=query.text,
            episode="query",
            entities=entities[:10],  # Limit entities
            motifs=features.motifs,
            scales=scales,
            metadata={
                'timestamp': time.time(),
                'confidence': features.confidence,
                'results': results
            }
        )
        
        # Queue for async persistence
        await self.persistence_queue.put({'shard': shard})
    
    async def shutdown(self):
        """Graceful shutdown - wait for persistence queue to drain."""
        await self.persistence_queue.join()
        if self._archiver_task:
            self._archiver_task.cancel()


# ============================================================================
# Factory Functions
# ============================================================================

def create_retriever(
    shards: List[MemoryShard],
    emb: MatryoshkaEmbeddings,
    fusion_weights: Optional[Dict[int, float]] = None
) -> RetrieverMS:
    """
    Factory function to create a retriever.
    
    Args:
        shards: Memory shards to index
        emb: Embeddings instance
        fusion_weights: Optional weights for scale fusion
        
    Returns:
        Configured RetrieverMS
    """
    return RetrieverMS(
        shards=shards,
        emb=emb,
        fusion_weights=fusion_weights
    )


def create_memory_manager(
    shards: List[MemoryShard],
    emb: MatryoshkaEmbeddings,
    fusion_weights: Optional[Dict[int, float]] = None,
    root: str = "data"
) -> MemoryManager:
    """
    Factory function to create a memory manager.
    
    Args:
        shards: Initial memory shards
        emb: Embeddings instance
        fusion_weights: Optional weights for scale fusion
        root: Root directory for persistence
        
    Returns:
        Configured MemoryManager
    """
    retriever = create_retriever(shards, emb, fusion_weights)
    pdv = PDVClient(root=root)
    memo = MemoAIClient(root=root)
    
    return MemoryManager(
        retriever=retriever,
        pdv=pdv,
        memo=memo
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    from embedding.spectral import MatryoshkaEmbeddings
    
    async def demo():
        print("=== Memory Cache Demo ===\n")
        
        # Create sample shards
        shards = [
            MemoryShard(
                id="s1",
                text="Multi-head attention processes multiple representation subspaces",
                episode="ML_basics",
                entities=["attention"],
                motifs=["explanation"]
            ),
            MemoryShard(
                id="s2",
                text="Transformers use self-attention mechanisms for sequence processing",
                episode="ML_basics",
                entities=["Transformers"],
                motifs=["explanation"]
            ),
            MemoryShard(
                id="s3",
                text="Neural networks learn hierarchical feature representations",
                episode="ML_basics",
                entities=["Neural networks"],
                motifs=["explanation"]
            ),
        ]
        
        # Create embedder and memory manager
        emb = MatryoshkaEmbeddings(sizes=[96, 192, 384])
        memory = create_memory_manager(shards, emb, root="demo_data")
        
        # Test retrieval
        query_text = "How does attention work in transformers?"
        
        print("Fast retrieval:")
        fast_results = await memory.retriever.search(query_text, k=2, fast=True)
        for shard, score in fast_results:
            print(f"  [{score:.3f}] {shard.text[:60]}...")
        
        print("\nFused retrieval:")
        fused_results = await memory.retriever.search(query_text, k=2, fast=False)
        for shard, score in fused_results:
            print(f"  [{score:.3f}] {shard.text[:60]}...")
        
        # Cleanup
        await memory.shutdown()
        print("\n✓ Demo complete!")
    
    asyncio.run(demo())