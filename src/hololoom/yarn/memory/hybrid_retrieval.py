"""
Hybrid Retrieval System (Week 4)
=================================

Best-of-all-worlds memory retrieval combining:
1. Semantic search (sentence-transformers embeddings + cosine similarity)
2. BM25 keyword search (traditional information retrieval)
3. Graph traversal (multi-hop knowledge expansion)
4. Reciprocal Rank Fusion (RRF) for score combination

Based on Research:
- LangMem: "Hybrid retrieval combines semantic + keyword + graph"
- Graphiti: "Multi-hop traversal enriches context"
- Mem0: "Rank fusion gives best of all worlds"

Architecture:
- SemanticRetriever: Embeddings + cosine similarity
- BM25Retriever: Keyword-based scoring
- GraphRetriever: Multi-hop traversal from entities
- HybridRetriever: Combines all three with RRF
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
import re
from collections import Counter

from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.memory.graph import KG

logger = logging.getLogger(__name__)


# ============================================================================
# Retrieval Result Types
# ============================================================================

@dataclass
class RetrievalScore:
    """Score breakdown for a retrieved memory."""
    memory_id: str
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    combined_score: float = 0.0
    retrieval_method: str = "unknown"  # "semantic", "bm25", "graph", "hybrid"


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    memories: List[MemoryShard]
    scores: List[RetrievalScore]
    total_candidates: int
    retrieval_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Semantic Search (Sentence-Transformers)
# ============================================================================

class SemanticRetriever:
    """
    Semantic search using sentence-transformers embeddings.

    Key Features:
    - Multi-scale Matryoshka embeddings (96, 192, 384 dims)
    - Cosine similarity for ranking
    - Graceful fallback if sentence-transformers unavailable
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        enable_fallback: bool = True
    ):
        """
        Initialize semantic retriever.

        Args:
            model_name: Sentence-transformers model (default: all-MiniLM-L6-v2)
            embedding_dim: Embedding dimension (96, 192, or 384)
            enable_fallback: Fall back to keyword matching if unavailable
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.enable_fallback = enable_fallback

        # Try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            self.model = SentenceTransformer(model_name)
            self.np = np
            self.available = True
            logger.info(f"Loaded sentence-transformers model: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback")
            self.model = None
            self.np = None
            self.available = False

        # Cache embeddings
        self.embedding_cache: Dict[str, Any] = {}

    def embed(self, text: str) -> Optional[Any]:
        """
        Embed text into vector space.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if unavailable
        """
        if not self.available:
            return None

        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Encode
        embedding = self.model.encode(text, normalize_embeddings=True)

        # Truncate to desired dimension (Matryoshka)
        if len(embedding) > self.embedding_dim:
            embedding = embedding[:self.embedding_dim]

        # Cache
        self.embedding_cache[text] = embedding

        return embedding

    def cosine_similarity(self, vec1: Any, vec2: Any) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity (0.0-1.0)
        """
        if vec1 is None or vec2 is None:
            return 0.0

        # Vectors are already normalized, so dot product = cosine similarity
        return float(self.np.dot(vec1, vec2))

    async def retrieve(
        self,
        query: str,
        memories: List[MemoryShard],
        limit: int = 10
    ) -> List[Tuple[MemoryShard, float]]:
        """
        Retrieve memories using semantic similarity.

        Args:
            query: Search query
            memories: Candidate memories
            limit: Maximum memories to return

        Returns:
            List of (memory, score) tuples sorted by score
        """
        if not self.available:
            if self.enable_fallback:
                logger.info("Semantic search unavailable, using keyword fallback")
                return await self._retrieve_fallback(query, memories, limit)
            return []

        # Embed query
        query_embedding = self.embed(query)
        if query_embedding is None:
            return []

        # Score all memories
        scored_memories = []
        for memory in memories:
            # Embed memory text
            memory_embedding = self.embed(memory.text)
            if memory_embedding is None:
                continue

            # Compute similarity
            score = self.cosine_similarity(query_embedding, memory_embedding)
            scored_memories.append((memory, score))

        # Sort by score (descending)
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return scored_memories[:limit]

    async def _retrieve_fallback(
        self,
        query: str,
        memories: List[MemoryShard],
        limit: int
    ) -> List[Tuple[MemoryShard, float]]:
        """Fallback to simple keyword matching."""
        query_words = set(query.lower().split())

        scored_memories = []
        for memory in memories:
            memory_words = set(memory.text.lower().split())
            # Jaccard similarity
            intersection = len(query_words & memory_words)
            union = len(query_words | memory_words)
            score = intersection / union if union > 0 else 0.0
            scored_memories.append((memory, score))

        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return scored_memories[:limit]


# ============================================================================
# BM25 Keyword Search
# ============================================================================

class BM25Retriever:
    """
    BM25 keyword-based retrieval (Okapi BM25 algorithm).

    Key Features:
    - Term frequency scoring with saturation
    - Inverse document frequency (IDF)
    - Document length normalization
    - No external dependencies (pure Python)
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b

        # Document statistics
        self.doc_freqs: Dict[str, int] = {}  # Term → doc count
        self.doc_lengths: Dict[str, int] = {}  # Doc ID → length
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase terms
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        text = text.lower()
        terms = re.findall(r'\w+', text)
        return terms

    def index_documents(self, memories: List[MemoryShard]):
        """
        Index documents for BM25 scoring.

        Args:
            memories: Documents to index
        """
        self.doc_freqs.clear()
        self.doc_lengths.clear()
        self.num_docs = len(memories)

        total_length = 0

        for memory in memories:
            terms = self.tokenize(memory.text)
            doc_length = len(terms)

            self.doc_lengths[memory.id] = doc_length
            total_length += doc_length

            # Track term frequencies
            unique_terms = set(terms)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Average document length
        self.avg_doc_length = total_length / self.num_docs if self.num_docs > 0 else 0.0

    def idf(self, term: str) -> float:
        """
        Compute inverse document frequency for term.

        Args:
            term: Term to score

        Returns:
            IDF score
        """
        df = self.doc_freqs.get(term, 0)
        if df == 0:
            return 0.0

        # BM25 IDF: log((N - df + 0.5) / (df + 0.5))
        return math.log((self.num_docs - df + 0.5) / (df + 0.5))

    def score_document(self, query_terms: List[str], doc_terms: List[str], doc_id: str) -> float:
        """
        Score document for query using BM25.

        Args:
            query_terms: Query terms
            doc_terms: Document terms
            doc_id: Document ID

        Returns:
            BM25 score
        """
        doc_length = self.doc_lengths.get(doc_id, 0)
        if doc_length == 0:
            return 0.0

        # Term frequencies in document
        term_freqs = Counter(doc_terms)

        score = 0.0
        for term in query_terms:
            if term not in term_freqs:
                continue

            # Term frequency in document
            tf = term_freqs[term]

            # IDF
            idf_score = self.idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))

            score += idf_score * (numerator / denominator)

        return score

    async def retrieve(
        self,
        query: str,
        memories: List[MemoryShard],
        limit: int = 10
    ) -> List[Tuple[MemoryShard, float]]:
        """
        Retrieve memories using BM25.

        Args:
            query: Search query
            memories: Candidate memories
            limit: Maximum memories to return

        Returns:
            List of (memory, score) tuples sorted by score
        """
        # Index documents
        self.index_documents(memories)

        # Tokenize query
        query_terms = self.tokenize(query)

        # Score all documents
        scored_memories = []
        for memory in memories:
            doc_terms = self.tokenize(memory.text)
            score = self.score_document(query_terms, doc_terms, memory.id)
            scored_memories.append((memory, score))

        # Sort by score (descending)
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return scored_memories[:limit]


# ============================================================================
# Graph Traversal Retrieval
# ============================================================================

class GraphRetriever:
    """
    Graph-based retrieval using multi-hop traversal.

    Key Features:
    - Start from query entities
    - Multi-hop traversal (1-3 hops)
    - Score based on path length and edge weights
    - Context expansion via connected entities
    """

    def __init__(
        self,
        kg: KG,
        max_hops: int = 2,
        hop_decay: float = 0.5
    ):
        """
        Initialize graph retriever.

        Args:
            kg: Knowledge graph
            max_hops: Maximum traversal hops (default: 2)
            hop_decay: Score decay per hop (default: 0.5)
        """
        self.kg = kg
        self.max_hops = max_hops
        self.hop_decay = hop_decay

    def extract_entities(self, query: str) -> List[str]:
        """
        Extract entities from query (simple approach).

        Args:
            query: Search query

        Returns:
            List of potential entities
        """
        # Simple approach: Title-case words might be entities
        words = query.split()
        entities = []

        for word in words:
            # Check if word (or variants) exist in graph
            if word in self.kg.G.nodes:
                entities.append(word)
            elif word.lower() in self.kg.G.nodes:
                entities.append(word.lower())
            elif word.capitalize() in self.kg.G.nodes:
                entities.append(word.capitalize())

        return entities

    def traverse(self, start_entity: str, max_hops: int) -> Dict[str, float]:
        """
        Multi-hop traversal from start entity.

        Args:
            start_entity: Starting entity
            max_hops: Maximum hops

        Returns:
            Dict of {entity: score} for reachable entities
        """
        if start_entity not in self.kg.G.nodes:
            return {}

        # BFS traversal
        visited = {start_entity: 1.0}  # Entity → score
        frontier = [(start_entity, 0)]  # (entity, hop_count)

        while frontier:
            current_entity, hops = frontier.pop(0)

            if hops >= max_hops:
                continue

            # Get neighbors
            neighbors = list(self.kg.G.neighbors(current_entity))

            for neighbor in neighbors:
                if neighbor not in visited:
                    # Score decays with distance
                    score = visited[current_entity] * self.hop_decay
                    visited[neighbor] = score
                    frontier.append((neighbor, hops + 1))

        return visited

    async def retrieve(
        self,
        query: str,
        memories: List[MemoryShard],
        limit: int = 10
    ) -> List[Tuple[MemoryShard, float]]:
        """
        Retrieve memories using graph traversal.

        Args:
            query: Search query
            memories: Candidate memories
            limit: Maximum memories to return

        Returns:
            List of (memory, score) tuples sorted by score
        """
        # Extract entities from query
        query_entities = self.extract_entities(query)
        if not query_entities:
            logger.info("No entities found in query for graph retrieval")
            return []

        # Traverse graph from each query entity
        all_reachable = {}
        for entity in query_entities:
            reachable = self.traverse(entity, self.max_hops)
            # Merge scores (take max)
            for node, score in reachable.items():
                all_reachable[node] = max(all_reachable.get(node, 0.0), score)

        # Score memories based on entity overlap
        scored_memories = []
        for memory in memories:
            # Check if memory contains reachable entities
            memory_entities = set(memory.entities)
            reachable_entities = set(all_reachable.keys())

            overlap = memory_entities & reachable_entities
            if overlap:
                # Score = sum of reachable entity scores
                score = sum(all_reachable[entity] for entity in overlap)
                scored_memories.append((memory, score))

        # Sort by score (descending)
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        return scored_memories[:limit]


# ============================================================================
# Reciprocal Rank Fusion (RRF)
# ============================================================================

def reciprocal_rank_fusion(
    rankings: List[List[Tuple[MemoryShard, float]]],
    k: int = 60
) -> List[Tuple[MemoryShard, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion.

    RRF Formula: score = sum(1 / (k + rank_i)) for all rankings

    Args:
        rankings: List of ranked results from different retrievers
        k: RRF constant (default: 60, from research)

    Returns:
        Fused ranking
    """
    # Aggregate scores
    memory_scores: Dict[str, float] = {}
    memory_objs: Dict[str, MemoryShard] = {}

    for ranking in rankings:
        for rank, (memory, _) in enumerate(ranking, start=1):
            rrf_score = 1.0 / (k + rank)

            memory_scores[memory.id] = memory_scores.get(memory.id, 0.0) + rrf_score
            memory_objs[memory.id] = memory

    # Sort by fused score
    sorted_memories = sorted(
        memory_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Return (memory, score) tuples
    return [(memory_objs[mem_id], score) for mem_id, score in sorted_memories]


# ============================================================================
# Hybrid Retriever (Combines All Three)
# ============================================================================

class HybridRetriever:
    """
    Hybrid retrieval combining semantic + BM25 + graph with RRF fusion.

    Best-of-all-worlds approach:
    - Semantic search: Finds conceptually similar content
    - BM25: Finds keyword matches
    - Graph: Expands context via entity relationships
    - RRF: Combines rankings for robust results
    """

    def __init__(
        self,
        kg: KG,
        semantic_model: str = "all-MiniLM-L6-v2",
        enable_semantic: bool = True,
        enable_bm25: bool = True,
        enable_graph: bool = True
    ):
        """
        Initialize hybrid retriever.

        Args:
            kg: Knowledge graph
            semantic_model: Sentence-transformers model
            enable_semantic: Enable semantic search
            enable_bm25: Enable BM25 search
            enable_graph: Enable graph traversal
        """
        self.kg = kg

        # Initialize retrievers
        self.semantic_retriever = SemanticRetriever(model_name=semantic_model) if enable_semantic else None
        self.bm25_retriever = BM25Retriever() if enable_bm25 else None
        self.graph_retriever = GraphRetriever(kg=kg) if enable_graph else None

    async def retrieve(
        self,
        query: str,
        memories: List[MemoryShard],
        limit: int = 10
    ) -> RetrievalResult:
        """
        Hybrid retrieval using all enabled methods + RRF fusion.

        Args:
            query: Search query
            memories: Candidate memories
            limit: Maximum memories to return

        Returns:
            RetrievalResult with fused rankings
        """
        start_time = datetime.now()

        rankings = []
        methods_used = []

        # Semantic search
        if self.semantic_retriever:
            semantic_results = await self.semantic_retriever.retrieve(query, memories, limit=limit*2)
            if semantic_results:
                rankings.append(semantic_results)
                methods_used.append("semantic")
                logger.info(f"Semantic retrieval: {len(semantic_results)} results")

        # BM25 search
        if self.bm25_retriever:
            bm25_results = await self.bm25_retriever.retrieve(query, memories, limit=limit*2)
            if bm25_results:
                rankings.append(bm25_results)
                methods_used.append("bm25")
                logger.info(f"BM25 retrieval: {len(bm25_results)} results")

        # Graph traversal
        if self.graph_retriever:
            graph_results = await self.graph_retriever.retrieve(query, memories, limit=limit*2)
            if graph_results:
                rankings.append(graph_results)
                methods_used.append("graph")
                logger.info(f"Graph retrieval: {len(graph_results)} results")

        # Fuse rankings with RRF
        if not rankings:
            logger.warning("No retrieval methods produced results")
            return RetrievalResult(
                memories=[],
                scores=[],
                total_candidates=len(memories),
                retrieval_time_ms=0.0
            )

        fused_results = reciprocal_rank_fusion(rankings)[:limit]

        # Extract memories and create scores
        final_memories = []
        final_scores = []

        for memory, rrf_score in fused_results:
            final_memories.append(memory)

            # Create detailed score breakdown
            score = RetrievalScore(
                memory_id=memory.id,
                combined_score=rrf_score,
                retrieval_method="+".join(methods_used)
            )
            final_scores.append(score)

        retrieval_time_ms = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Hybrid retrieval complete: {len(final_memories)} results, "
            f"{retrieval_time_ms:.1f}ms, methods={methods_used}"
        )

        return RetrievalResult(
            memories=final_memories,
            scores=final_scores,
            total_candidates=len(memories),
            retrieval_time_ms=retrieval_time_ms,
            metadata={"methods": methods_used}
        )


# ============================================================================
# Factory Functions
# ============================================================================

def create_hybrid_retriever(
    kg: KG,
    semantic_model: str = "all-MiniLM-L6-v2",
    enable_all: bool = True
) -> HybridRetriever:
    """
    Create hybrid retriever with sensible defaults.

    Args:
        kg: Knowledge graph
        semantic_model: Sentence-transformers model
        enable_all: Enable all retrieval methods

    Returns:
        HybridRetriever
    """
    return HybridRetriever(
        kg=kg,
        semantic_model=semantic_model,
        enable_semantic=enable_all,
        enable_bm25=enable_all,
        enable_graph=enable_all
    )
