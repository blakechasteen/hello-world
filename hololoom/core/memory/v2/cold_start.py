"""
Cold Start Handler — Dual-mode fallback for queries with weak graph anchors.

When extract_seeds() finds few or no graph anchors, normal PPR can't work
because there's nothing to start walking from. The cold start handler
provides an alternative: content similarity to find seed nodes.

Strategy:
  1. If extract_seeds() returns < min_seeds: trigger cold start
  2. Compute content similarity against all node contents
  3. Top-k similar nodes become PPR seeds (weight = similarity)
  4. Resume normal PPR from these seeds

Two similarity backends:
  - Embedding similarity (if bus._embeddings available)
  - Keyword overlap (fallback, no dependencies)

The result feeds into the normal PPR pipeline — cold start just
provides better seeds, everything downstream is the same.

No external dependencies beyond what's already available.
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ColdStartConfig:
    """Configuration for cold start handling."""
    min_seeds: int = 2                # Trigger cold start if fewer seeds
    vector_k: int = 5                 # Top-k similar nodes for vector fallback
    keyword_k: int = 5               # Top-k for keyword overlap fallback
    min_similarity: float = 0.1      # Minimum similarity to accept as seed
    keyword_weight: float = 0.3      # Seed weight for keyword-based seeds
    vector_weight: float = 0.5       # Seed weight for vector-based seeds


class ColdStartHandler:
    """Handles queries with weak graph anchors.

    When the navigator's extract_seeds() finds too few anchors in the graph,
    the cold start handler provides alternative seed nodes via content
    similarity matching.

    Usage:
        handler = ColdStartHandler(graph=LiteBusGraph(bus))
        seeds = extract_seeds(query, graph)

        if handler.needs_cold_start(seeds):
            cold_seeds = handler.find_seeds(query)
            seeds.extend(cold_seeds)

        # Continue with normal PPR using enriched seeds
    """

    def __init__(
        self,
        graph,  # GraphBackend
        config: ColdStartConfig | None = None,
        embeddings: dict[str, Any] | None = None,
    ):
        self.graph = graph
        self.config = config or ColdStartConfig()
        self._embeddings = embeddings  # Optional: {node_id: embedding_vector}

    def needs_cold_start(self, seeds: list) -> bool:
        """Check if we need cold start (too few seeds)."""
        return len(seeds) < self.config.min_seeds

    def find_seeds(self, query: str) -> list:
        """Find seed nodes via content similarity (no graph structure needed).

        Tries embedding similarity first, falls back to keyword overlap.

        Returns:
            List of SeedNode objects (imported lazily to avoid circular deps)
        """
        if self._embeddings:
            seeds = self._vector_seeds(query)
            if seeds:
                logger.info("ColdStart: %d vector seeds for '%s'", len(seeds), query[:50])
                return seeds

        seeds = self._keyword_seeds(query)
        logger.info("ColdStart: %d keyword seeds for '%s'", len(seeds), query[:50])
        return seeds

    def _vector_seeds(self, query: str) -> list:
        """Find seeds via embedding cosine similarity.

        Requires self._embeddings to be a dict of {node_id: numpy_array}.
        Also requires a query embedding — if not available, returns empty.
        """
        try:
            import numpy as np
        except ImportError:
            return []

        if not self._embeddings:
            return []

        # Check if we have a query embedding function
        query_embedding = self._embeddings.get("__query_fn__")
        if callable(query_embedding):
            q_vec = query_embedding(query)
        elif "__query__" in self._embeddings:
            q_vec = self._embeddings["__query__"]
        else:
            return []

        # Cosine similarity against all node embeddings
        similarities = []
        for node_id, embedding in self._embeddings.items():
            if node_id.startswith("__"):
                continue
            try:
                sim = self._cosine_similarity(q_vec, embedding)
                if sim >= self.config.min_similarity:
                    similarities.append((node_id, sim))
            except (ValueError, TypeError):
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = similarities[:self.config.vector_k]

        # Convert to SeedNodes
        from .navigator import SeedNode
        return [
            SeedNode(
                node_id=nid,
                weight=sim * self.config.vector_weight,
                source="cold_start_vector",
            )
            for nid, sim in top_k
        ]

    def _keyword_seeds(self, query: str) -> list:
        """Find seeds via keyword overlap (no dependencies needed).

        Uses TF-IDF-like scoring: query words matched against node content,
        weighted by inverse frequency across all nodes.
        """
        query_words = self._tokenize(query)
        if not query_words:
            return []

        # Score each node by keyword overlap
        scores: list[tuple] = []
        all_nodes = self.graph.nodes()

        # Build document frequency for IDF-like weighting
        doc_freq: Counter = Counter()
        node_words: dict[str, set[str]] = {}
        for nid in all_nodes:
            data = self.graph.node_data(nid)
            content = data.get("content", "")
            words = self._tokenize(content)
            node_words[nid] = words
            for w in words:
                doc_freq[w] += 1

        n_docs = max(1, len(all_nodes))

        for nid in all_nodes:
            words = node_words.get(nid, set())
            if not words:
                continue

            # TF-IDF overlap: sum of IDF weights for matching query words
            score = 0.0
            for qw in query_words:
                if qw in words:
                    idf = math.log(1 + n_docs / max(1, doc_freq.get(qw, 1)))
                    score += idf

            if score > 0:
                # Normalize by query length
                score /= len(query_words)
                scores.append((nid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k = scores[:self.config.keyword_k]

        if not top_k:
            return []

        # Normalize scores to [0, 1]
        max_score = top_k[0][1]
        if max_score <= 0:
            return []

        from .navigator import SeedNode
        return [
            SeedNode(
                node_id=nid,
                weight=(score / max_score) * self.config.keyword_weight,
                source="cold_start_keyword",
            )
            for nid, score in top_k
            if score / max_score >= self.config.min_similarity
        ]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple word tokenization (lowercase, strip punctuation, min length)."""
        words = set()
        for word in text.lower().split():
            clean = word.strip(".,;:!?()[]\"'-")
            if len(clean) > 2:
                words.add(clean)
        return words

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity between two vectors."""
        import numpy as np
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
