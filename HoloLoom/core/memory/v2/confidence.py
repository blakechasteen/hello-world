"""
Confidence — Dual confidence estimation for the Matryoshka shell loop.

Two independent confidence signals combined via geometric mean:

  1. STRUCTURAL confidence — Is the retrieved context well-grounded?
     - Seed coverage: fraction of query seeds found in results
     - Entity linkage: fraction of results connected to an entity node
     - Graph connectivity: fraction of result nodes connected to others
     - PPR concentration: how focused the retrieval is (Gini of scores)
     - Content density: fraction of results with substantial content

  2. NARRATIVE confidence — Does the context form a coherent story?
     - Pairwise density: fraction of all result pairs with graph edges
     - Source diversity: mix of memory types (entity/fact/episode)
     - Topical coherence: word overlap between retrieved items
     - Contradiction detection: opposing claims about same topic

Combined via geometric mean: both must be decent for high confidence.
This prevents a single strong axis from masking a weak one.

No LLM calls — pure graph/structural analysis.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .navigator import GraphBackend, NavigatorResult

logger = logging.getLogger(__name__)


# =============================================================================
# Semantic Contradiction Detection (shared by ConfidenceEstimator + TMS)
# =============================================================================

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "so", "yet", "both", "either", "neither", "each", "every",
    "all", "any", "few", "more", "most", "other", "some", "such", "than",
    "too", "very", "just", "also", "about", "up", "out", "if", "this",
    "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "which",
    "who", "whom", "what", "where", "when", "why", "how", "there",
    "here", "then", "now", "only", "own",
})

_NEGATION_WORDS = frozenset({
    "not", "never", "no", "cannot", "doesn't", "don't", "isn't",
    "won't", "fails", "unable", "without", "lack", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't", "couldn't",
    "shouldn't", "wouldn't", "mustn't",
})


def _simple_stem(word: str) -> str:
    """Minimal suffix stripping for morphological matching."""
    if len(word) <= 3:
        return word
    if word.endswith("ing") and len(word) > 5:
        return word[:-3]
    if word.endswith("ed") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _content_stems(text: str) -> Set[str]:
    """Tokenize, remove stop words, and stem content words."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {
        _simple_stem(w)
        for w in tokens
        if w not in _STOP_WORDS and len(w) > 2
    }


def find_contradicting_pairs(ranked_nodes, node_data, graph=None):
    """Semantic contradiction detection via entity-anchored opposing claims.

    Two items contradict when:
      1. They reference the same entity (share an entity neighbor in the
         graph, or share entity-name substrings as fallback)
      2. One contains negation words, the other doesn't
      3. Their content words have significant Jaccard overlap

    Replaces the naive lexical heuristic (overlap >= 3 + negation +
    comparison words) which over-triggered in domains where comparison
    language is universal.

    Args:
        ranked_nodes: List of (node_id, score) from NavigatorResult.
        node_data: Dict of node_id -> data dicts.
        graph: Optional GraphBackend for entity neighbor lookup.

    Returns:
        List of (node_id_a, node_id_b) contradicting pairs.
    """
    # --- Precompute entity neighbors per retrieved node ---
    entity_neighbors: Dict[str, Set[str]] = {}
    entity_names: Set[str] = set()

    if graph is not None:
        try:
            all_nids = set(graph.nodes())
        except Exception:
            all_nids = set()

        for nid in all_nids:
            try:
                data = graph.node_data(nid)
            except Exception:
                continue
            if data.get("memory_type") == "entity":
                name = data.get("content", "").strip().lower().split("\n")[0]
                if len(name) >= 3:
                    entity_names.add(name)

        for nid, _ in ranked_nodes:
            ent_nbrs: Set[str] = set()
            try:
                for nbr_id, _, _ in graph.neighbors(nid):
                    try:
                        nbr_data = graph.node_data(nbr_id)
                    except Exception:
                        continue
                    if nbr_data.get("memory_type") == "entity":
                        ent_nbrs.add(nbr_id)
            except Exception:
                pass
            # If this node IS an entity, include itself
            try:
                self_data = graph.node_data(nid)
                if self_data.get("memory_type") == "entity":
                    ent_nbrs.add(nid)
            except Exception:
                pass
            entity_neighbors[nid] = ent_nbrs

    has_graph_entities = any(entity_neighbors.values())
    jaccard_threshold = 0.15 if has_graph_entities else 0.25

    # --- Build per-item analysis ---
    items = []
    for nid, _ in ranked_nodes:
        data = node_data.get(nid, {})
        content = data.get("content", "").lower()
        if not content:
            continue

        all_words = set(re.findall(r"\b\w+\b", content))
        content_words = _content_stems(content)
        has_negation = bool(all_words & _NEGATION_WORDS)
        ent_nbrs = entity_neighbors.get(nid, set())

        # Fallback: content-based entity mention detection
        mentioned = set()
        if not ent_nbrs and entity_names:
            for ent_name in entity_names:
                if ent_name in content:
                    mentioned.add(ent_name)

        items.append((nid, content_words, has_negation, ent_nbrs, mentioned))

    # --- Pairwise comparison ---
    pairs = []
    for i in range(len(items)):
        nid_i, words_i, neg_i, ent_i, ment_i = items[i]
        for j in range(i + 1, len(items)):
            nid_j, words_j, neg_j, ent_j, ment_j = items[j]

            # Must have negation asymmetry
            if neg_i == neg_j:
                continue

            # Must reference the same entity
            shared = bool(ent_i & ent_j) if (ent_i and ent_j) else False
            if not shared and ment_i and ment_j:
                shared = bool(ment_i & ment_j)
            if not shared:
                continue

            # Must have significant content word overlap (Jaccard)
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union == 0:
                continue
            if intersection / union < jaccard_threshold:
                continue

            pairs.append((nid_i, nid_j))

    return pairs


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ConfidenceConfig:
    """Thresholds for confidence estimation."""
    # Structural weights (must sum to 1.0)
    coverage_weight: float = 0.25
    entity_linkage_weight: float = 0.25
    connectivity_weight: float = 0.20
    ppr_concentration_weight: float = 0.15
    content_density_weight: float = 0.15

    # Narrative weights (must sum to 1.0)
    pairwise_density_weight: float = 0.30
    source_diversity_weight: float = 0.20
    topical_coherence_weight: float = 0.25
    contradiction_penalty: float = 0.3

    # Combined
    high_threshold: float = 0.75   # Above this → skip VERIFY
    low_threshold: float = 0.35    # Below this → trigger FLAG

    # Content density threshold (chars)
    min_content_length: int = 20


# =============================================================================
# Confidence Signals
# =============================================================================

@dataclass
class StructuralConfidence:
    """Structural confidence analysis result."""
    score: float                    # 0.0–1.0
    seed_coverage: float            # Fraction of seeds found in results
    entity_grounding: float         # Entity linkage (connected to entity)
    graph_connectivity: float       # Fraction of results connected to others
    ppr_concentration: float = 0.0  # Gini of PPR scores (0=uniform, 1=focused)
    content_density: float = 0.0    # Fraction of results with substantial content
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeConfidence:
    """Narrative confidence analysis result."""
    score: float                    # 0.0–1.0
    edge_coherence: float           # Pairwise edge density across all result pairs
    temporal_order: float           # Source diversity (mix of types)
    contradiction_count: int        # Number of detected contradictions
    topical_coherence: float = 0.0  # Average word overlap between items
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DualConfidence:
    """Combined structural + narrative confidence."""
    structural: StructuralConfidence
    narrative: NarrativeConfidence
    combined: float                  # Geometric mean
    decision: str                    # "sufficient", "verify", "flag"
    per_node: Dict[str, float]       # Per-node confidence scores


# =============================================================================
# Estimator
# =============================================================================

class ConfidenceEstimator:
    """Dual confidence estimator with rich graph signals.

    Optionally includes a cross-encoder reranking signal: a small model
    that scores (query, context) pairs for relevance. When available,
    this provides a calibrated relevance signal without a full LLM call.

    Usage:
        estimator = ConfidenceEstimator(graph=nav.graph)
        confidence = estimator.estimate(navigator_result)
        if confidence.decision == "sufficient":
            # Skip VERIFY shell
        elif confidence.decision == "flag":
            # Trigger FLAG shell
    """

    def __init__(
        self,
        graph: GraphBackend,
        config: Optional[ConfidenceConfig] = None,
        reranker_fn=None,  # Optional: (query: str, text: str) → float (0-1)
    ):
        self.graph = graph
        self.config = config or ConfidenceConfig()
        self.reranker_fn = reranker_fn  # Cross-encoder reranker

    def estimate(self, nav_result: NavigatorResult, query: str = "") -> DualConfidence:
        """Run dual confidence estimation on navigator output.

        Args:
            nav_result: Navigator output with ranked nodes
            query: Original query (needed for cross-encoder reranking)
        """
        if not nav_result.ranked_nodes:
            empty_structural = StructuralConfidence(0.0, 0.0, 0.0, 0.0)
            empty_narrative = NarrativeConfidence(0.0, 0.0, 0.0, 0)
            return DualConfidence(
                structural=empty_structural,
                narrative=empty_narrative,
                combined=0.0,
                decision="flag",
                per_node={},
            )

        structural = self._structural_confidence(nav_result)
        narrative = self._narrative_confidence(nav_result)

        cfg = self.config

        # Cross-encoder reranking signal (optional, blended in)
        reranker_score = self._reranker_confidence(query, nav_result)

        # Geometric mean: both axes must be decent
        if structural.score > 0 and narrative.score > 0:
            combined = math.sqrt(structural.score * narrative.score)
        else:
            combined = 0.0

        # Blend reranker signal if available (20% weight, redistribute)
        if reranker_score is not None:
            combined = 0.8 * combined + 0.2 * reranker_score

        # Decision
        if combined >= cfg.high_threshold:
            decision = "sufficient"
        elif combined <= cfg.low_threshold:
            decision = "flag"
        else:
            decision = "verify"

        per_node = self._per_node_confidence(nav_result)

        return DualConfidence(
            structural=structural,
            narrative=narrative,
            combined=combined,
            decision=decision,
            per_node=per_node,
        )

    # =========================================================================
    # Cross-Encoder Reranking
    # =========================================================================

    def _reranker_confidence(self, query: str, nav_result: NavigatorResult) -> Optional[float]:
        """Average cross-encoder reranker score across top-k retrieved items.

        When a small reranker model is available (e.g., via sentence-transformers
        or Ollama), scores (query, item_content) pairs. This provides a calibrated
        relevance signal independent of graph structure.

        Returns None when no reranker is available.
        """
        if not self.reranker_fn or not query:
            return None

        scores = []
        for nid, _ in nav_result.ranked_nodes[:10]:  # Top-10 only for speed
            data = nav_result.node_data.get(nid, {})
            content = data.get("content", data.get("text", ""))
            if not content:
                continue
            try:
                score = self.reranker_fn(query, content[:500])  # Truncate for speed
                scores.append(float(score))
            except Exception:
                continue

        if not scores:
            return None

        return sum(scores) / len(scores)

    # =========================================================================
    # Structural Confidence
    # =========================================================================

    def _structural_confidence(self, nav_result: NavigatorResult) -> StructuralConfidence:
        """Assess structural grounding of retrieved context.

        Five signals:
          1. Seed coverage — seeds found in results
          2. Entity linkage — results connected to entity nodes (not just ARE entities)
          3. Graph connectivity — results connected to other results
          4. PPR concentration — Gini coefficient of PPR scores
          5. Content density — results with substantial text content
        """
        cfg = self.config
        result_ids = set(nid for nid, _ in nav_result.ranked_nodes)
        seed_ids = set(s.node_id for s in nav_result.seed_nodes)

        # 1. Seed coverage
        seeds_in_results = len(seed_ids & result_ids)
        seed_coverage = seeds_in_results / max(1, len(seed_ids))

        # 2. Entity linkage: fraction of results that are entities OR connected to entities
        entity_nodes = set()
        for nid in self.graph.nodes():
            data = self.graph.node_data(nid)
            if data.get("memory_type") == "entity" or data.get("entity_type"):
                entity_nodes.add(nid)

        linked_count = 0
        for nid in result_ids:
            if nid in entity_nodes:
                linked_count += 1
            else:
                # Check if any neighbor is an entity
                neighbors = self.graph.neighbors(nid)
                neighbor_ids = set(n[0] for n in neighbors)
                if neighbor_ids & entity_nodes:
                    linked_count += 1
        entity_linkage = linked_count / max(1, len(result_ids))

        # 3. Graph connectivity
        connected_count = 0
        for nid in result_ids:
            neighbors = self.graph.neighbors(nid)
            neighbor_ids = set(n[0] for n in neighbors)
            if neighbor_ids & (result_ids - {nid}):
                connected_count += 1
        graph_connectivity = connected_count / max(1, len(result_ids))

        # 4. PPR concentration (Gini coefficient)
        scores = [s for _, s in nav_result.ranked_nodes]
        ppr_concentration = self._gini(scores)

        # 5. Content density
        dense_count = 0
        for nid in result_ids:
            data = nav_result.node_data.get(nid, {})
            content = data.get("content", data.get("text", ""))
            if len(content) >= cfg.min_content_length:
                dense_count += 1
        content_density = dense_count / max(1, len(result_ids))

        score = (
            cfg.coverage_weight * seed_coverage +
            cfg.entity_linkage_weight * entity_linkage +
            cfg.connectivity_weight * graph_connectivity +
            cfg.ppr_concentration_weight * ppr_concentration +
            cfg.content_density_weight * content_density
        )

        return StructuralConfidence(
            score=min(1.0, score),
            seed_coverage=seed_coverage,
            entity_grounding=entity_linkage,
            graph_connectivity=graph_connectivity,
            ppr_concentration=ppr_concentration,
            content_density=content_density,
            details={
                "seeds_total": len(seed_ids),
                "seeds_in_results": seeds_in_results,
                "entity_linked": linked_count,
                "connected_nodes": connected_count,
                "dense_nodes": dense_count,
                "total_results": len(result_ids),
            },
        )

    # =========================================================================
    # Narrative Confidence
    # =========================================================================

    def _narrative_confidence(self, nav_result: NavigatorResult) -> NarrativeConfidence:
        """Assess narrative coherence of retrieved context.

        Three signals (temporal_order repurposed as source_diversity):
          1. Pairwise density — fraction of ALL result pairs with edges
          2. Source diversity — mix of memory types (entity/fact/episode)
          3. Topical coherence — average word overlap between items
          4. Contradictions — negation-based detection (penalty)
        """
        cfg = self.config
        ranked = nav_result.ranked_nodes
        result_ids = [nid for nid, _ in ranked]

        # 1. Pairwise edge density: fraction of all pairs with direct edges
        #    (fixes the bug where only adjacent-in-ranking pairs were checked)
        total_pairs = 0
        connected_pairs = 0
        neighbor_cache: Dict[str, Set[str]] = {}

        for i, nid_a in enumerate(result_ids):
            if nid_a not in neighbor_cache:
                neighbor_cache[nid_a] = set(n[0] for n in self.graph.neighbors(nid_a))
            for j in range(i + 1, len(result_ids)):
                nid_b = result_ids[j]
                total_pairs += 1
                if nid_b in neighbor_cache[nid_a]:
                    connected_pairs += 1
                else:
                    if nid_b not in neighbor_cache:
                        neighbor_cache[nid_b] = set(n[0] for n in self.graph.neighbors(nid_b))
                    if nid_a in neighbor_cache[nid_b]:
                        connected_pairs += 1

        pairwise_density = connected_pairs / max(1, total_pairs)

        # 2. Source diversity: Shannon entropy of memory type distribution, normalized
        type_counts: Dict[str, int] = {}
        for nid in result_ids:
            data = nav_result.node_data.get(nid, {})
            mtype = data.get("memory_type", "unknown")
            type_counts[mtype] = type_counts.get(mtype, 0) + 1

        n_items = max(1, len(result_ids))
        n_types = len(type_counts)
        if n_types > 1:
            entropy = -sum(
                (c / n_items) * math.log2(c / n_items)
                for c in type_counts.values() if c > 0
            )
            max_entropy = math.log2(n_types)
            source_diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            source_diversity = 0.0  # Single type = no diversity

        # 3. Topical coherence: average Jaccard similarity between content pairs
        contents = []
        word_sets = []
        for nid in result_ids:
            data = nav_result.node_data.get(nid, {})
            content = data.get("content", data.get("text", "")).lower()
            if content:
                contents.append(content)
                words = set(w for w in content.split() if len(w) > 2)
                word_sets.append(words)

        topical_coherence = 0.0
        if len(word_sets) >= 2:
            jaccard_sum = 0.0
            pair_count = 0
            for i in range(len(word_sets)):
                for j in range(i + 1, min(len(word_sets), i + 10)):  # Cap at 10 neighbors
                    intersection = len(word_sets[i] & word_sets[j])
                    union = len(word_sets[i] | word_sets[j])
                    if union > 0:
                        jaccard_sum += intersection / union
                    pair_count += 1
            topical_coherence = jaccard_sum / max(1, pair_count)

        # 4. Contradictions
        contradictions = self._detect_contradictions(nav_result)

        # Score
        contradiction_factor = max(0.0, 1.0 - cfg.contradiction_penalty * contradictions)

        # Remaining weight (0.25) is implicit from 1.0 - weights
        remaining_weight = 1.0 - cfg.pairwise_density_weight - cfg.source_diversity_weight - cfg.topical_coherence_weight
        score = (
            cfg.pairwise_density_weight * pairwise_density +
            cfg.source_diversity_weight * source_diversity +
            cfg.topical_coherence_weight * topical_coherence +
            max(0, remaining_weight) * 1.0  # Baseline presence signal
        ) * contradiction_factor

        return NarrativeConfidence(
            score=min(1.0, score),
            edge_coherence=pairwise_density,
            temporal_order=source_diversity,  # Repurposed field
            contradiction_count=contradictions,
            topical_coherence=topical_coherence,
            details={
                "total_pairs": total_pairs,
                "connected_pairs": connected_pairs,
                "type_counts": type_counts,
                "n_content_items": len(contents),
            },
        )

    def _detect_contradictions(self, nav_result: NavigatorResult) -> int:
        """Semantic contradiction detection via entity-anchored opposing claims.

        Uses find_contradicting_pairs() which checks:
          1. Shared entity neighbor in the graph (same subject)
          2. Negation asymmetry (opposing predicate)
          3. Significant Jaccard overlap on content words
        """
        pairs = find_contradicting_pairs(
            nav_result.ranked_nodes,
            nav_result.node_data,
            graph=self.graph,
        )
        return len(pairs)

    # =========================================================================
    # Per-Node Confidence
    # =========================================================================

    def _per_node_confidence(self, nav_result: NavigatorResult) -> Dict[str, float]:
        """Compute per-node confidence scores."""
        result_ids = set(nid for nid, _ in nav_result.ranked_nodes)
        per_node: Dict[str, float] = {}

        max_ppr = max(s for _, s in nav_result.ranked_nodes) if nav_result.ranked_nodes else 1.0

        # Precompute entity nodes
        entity_nodes = set()
        for nid in self.graph.nodes():
            data = self.graph.node_data(nid)
            if data.get("memory_type") == "entity" or data.get("entity_type"):
                entity_nodes.add(nid)

        for nid, ppr_score in nav_result.ranked_nodes:
            data = nav_result.node_data.get(nid, {})

            ppr_norm = ppr_score / max(1e-10, max_ppr)

            # Entity linkage (is entity or connected to one)
            neighbors = self.graph.neighbors(nid)
            neighbor_ids = set(n[0] for n in neighbors)
            is_linked = 1.0 if (nid in entity_nodes or neighbor_ids & entity_nodes) else 0.0

            # Local connectivity
            if neighbor_ids:
                local_connectivity = len(neighbor_ids & result_ids - {nid}) / len(neighbor_ids)
            else:
                local_connectivity = 0.0

            # Content richness
            content = data.get("content", data.get("text", ""))
            has_content = 1.0 if len(content) >= self.config.min_content_length else 0.0

            score = 0.35 * ppr_norm + 0.25 * is_linked + 0.20 * local_connectivity + 0.20 * has_content
            per_node[nid] = min(1.0, score)

        return per_node

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _gini(values: List[float]) -> float:
        """Gini coefficient: 0 = uniform, 1 = maximally concentrated."""
        if not values or len(values) < 2:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        cumulative = 0.0
        gini_sum = 0.0
        for i, v in enumerate(sorted_vals):
            cumulative += v
            gini_sum += (2 * (i + 1) - n - 1) * v
        return gini_sum / (n * total)
