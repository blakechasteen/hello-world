"""
Retrieval Quality Metrics
=========================

Standard Information Retrieval metrics for evaluating memory-augmented retrieval.

Metrics Implemented:
- NDCG@k: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank
- MAP: Mean Average Precision
- Precision@k: Precision at k retrieved documents
- Recall@k: Recall at k retrieved documents
- Hit Rate: Did relevant doc appear in top-k?

Integration Points:
- Uses LLMJudge for relevance scoring when ground truth unavailable
- Supports multi-hop retrieval path evaluation
- Handles temporal retrieval (recent vs old knowledge)

Author: HoloLoom Architecture Team
Date: December 2025
"""

import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Set
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RetrievalMetrics:
    """
    Complete retrieval quality metrics for a single query or batch.

    Attributes:
        ndcg_at_k: Normalized Discounted Cumulative Gain at k={1,3,5,10}
        mrr: Mean Reciprocal Rank (position of first relevant result)
        map_score: Mean Average Precision
        precision_at_k: Precision at k={1,3,5,10}
        recall_at_k: Recall at k={1,3,5,10}
        hit_rate: Fraction of queries with at least one relevant result in top-k
        total_queries: Number of queries evaluated
        metadata: Additional evaluation metadata
    """
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_score: float = 0.0
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate: float = 0.0
    total_queries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "map_score": self.map_score,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "hit_rate": self.hit_rate,
            "total_queries": self.total_queries,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Human-readable summary of metrics."""
        lines = [
            f"Retrieval Metrics ({self.total_queries} queries)",
            f"  MRR: {self.mrr:.4f}",
            f"  MAP: {self.map_score:.4f}",
            f"  Hit Rate: {self.hit_rate:.4f}",
        ]
        for k in sorted(self.ndcg_at_k.keys()):
            lines.append(f"  NDCG@{k}: {self.ndcg_at_k[k]:.4f}")
        for k in sorted(self.precision_at_k.keys()):
            lines.append(f"  P@{k}: {self.precision_at_k[k]:.4f}")
        for k in sorted(self.recall_at_k.keys()):
            lines.append(f"  R@{k}: {self.recall_at_k[k]:.4f}")
        return "\n".join(lines)


@dataclass
class RetrievalResult:
    """Single retrieval result with relevance score."""
    doc_id: str
    score: float  # Retrieval score (similarity, etc.)
    relevance: Optional[float] = None  # Ground truth relevance (0-1 or binary)
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalQuery:
    """A query with its retrieved results and ground truth."""
    query_id: str
    query_text: str
    retrieved: List[RetrievalResult]
    relevant_doc_ids: Set[str]  # Ground truth relevant documents
    relevance_grades: Optional[Dict[str, float]] = None  # Graded relevance (for NDCG)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Individual Metric Functions
# =============================================================================

def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Discounted Cumulative Gain at k.

    DCG@k = Σ (2^rel_i - 1) / log2(i + 1) for i in 1..k

    Args:
        relevances: List of relevance scores in rank order
        k: Cutoff position

    Returns:
        DCG score
    """
    if not relevances:
        return 0.0

    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        # Numerator: gain, Denominator: discount
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because log2(1) = 0

    return dcg


def ndcg_at_k(
    retrieved: List[RetrievalResult],
    relevance_grades: Dict[str, float],
    k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k

    Where IDCG is the DCG of the ideal ranking (sorted by relevance).

    Args:
        retrieved: List of retrieved results in rank order
        relevance_grades: Dict mapping doc_id to relevance grade (0-1 or higher)
        k: Cutoff position

    Returns:
        NDCG score (0-1)
    """
    if not retrieved or not relevance_grades:
        return 0.0

    # Get actual relevances in retrieved order
    actual_rels = [
        relevance_grades.get(r.doc_id, 0.0)
        for r in retrieved[:k]
    ]

    # Ideal relevances (sorted descending)
    ideal_rels = sorted(relevance_grades.values(), reverse=True)[:k]

    dcg = dcg_at_k(actual_rels, k)
    idcg = dcg_at_k(ideal_rels, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def mrr(queries: List[RetrievalQuery]) -> float:
    """
    Mean Reciprocal Rank.

    MRR = (1/|Q|) Σ 1/rank_i

    Where rank_i is the position of the first relevant document for query i.

    Args:
        queries: List of queries with retrieved results and ground truth

    Returns:
        MRR score (0-1)
    """
    if not queries:
        return 0.0

    reciprocal_ranks = []

    for query in queries:
        for i, result in enumerate(query.retrieved):
            if result.doc_id in query.relevant_doc_ids:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            # No relevant document found
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def average_precision(
    retrieved: List[RetrievalResult],
    relevant_doc_ids: Set[str]
) -> float:
    """
    Average Precision for a single query.

    AP = (1/|R|) Σ P@k × rel(k)

    Where R is the set of relevant documents.

    Args:
        retrieved: List of retrieved results in rank order
        relevant_doc_ids: Set of relevant document IDs

    Returns:
        AP score (0-1)
    """
    if not retrieved or not relevant_doc_ids:
        return 0.0

    num_relevant = len(relevant_doc_ids)
    if num_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_count = 0

    for i, result in enumerate(retrieved):
        if result.doc_id in relevant_doc_ids:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    return precision_sum / num_relevant


def map_score(queries: List[RetrievalQuery]) -> float:
    """
    Mean Average Precision.

    MAP = (1/|Q|) Σ AP(q)

    Args:
        queries: List of queries with retrieved results and ground truth

    Returns:
        MAP score (0-1)
    """
    if not queries:
        return 0.0

    aps = [
        average_precision(q.retrieved, q.relevant_doc_ids)
        for q in queries
    ]

    return sum(aps) / len(aps)


def precision_at_k(
    retrieved: List[RetrievalResult],
    relevant_doc_ids: Set[str],
    k: int
) -> float:
    """
    Precision at k.

    P@k = |{relevant docs in top-k}| / k

    Args:
        retrieved: List of retrieved results in rank order
        relevant_doc_ids: Set of relevant document IDs
        k: Cutoff position

    Returns:
        P@k score (0-1)
    """
    if not retrieved or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for r in top_k if r.doc_id in relevant_doc_ids)

    return relevant_in_top_k / k


def recall_at_k(
    retrieved: List[RetrievalResult],
    relevant_doc_ids: Set[str],
    k: int
) -> float:
    """
    Recall at k.

    R@k = |{relevant docs in top-k}| / |{all relevant docs}|

    Args:
        retrieved: List of retrieved results in rank order
        relevant_doc_ids: Set of relevant document IDs
        k: Cutoff position

    Returns:
        R@k score (0-1)
    """
    if not retrieved or not relevant_doc_ids or k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for r in top_k if r.doc_id in relevant_doc_ids)

    return relevant_in_top_k / len(relevant_doc_ids)


def hit_rate(queries: List[RetrievalQuery], k: int = 10) -> float:
    """
    Hit Rate at k.

    HR@k = |{queries with at least one relevant doc in top-k}| / |{all queries}|

    Args:
        queries: List of queries with retrieved results and ground truth
        k: Cutoff position

    Returns:
        Hit rate (0-1)
    """
    if not queries:
        return 0.0

    hits = 0
    for query in queries:
        top_k = query.retrieved[:k]
        if any(r.doc_id in query.relevant_doc_ids for r in top_k):
            hits += 1

    return hits / len(queries)


# =============================================================================
# Evaluator Class
# =============================================================================

class RetrievalEvaluator:
    """
    Comprehensive retrieval evaluator with support for:
    - Standard IR metrics (NDCG, MRR, MAP, P@k, R@k)
    - LLM-based relevance scoring (via LLMJudge integration)
    - Multi-hop retrieval path evaluation
    - Temporal retrieval analysis
    """

    DEFAULT_K_VALUES = [1, 3, 5, 10]

    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        llm_judge: Optional[Any] = None,  # LLMJudge instance for relevance scoring
        use_graded_relevance: bool = True,
    ):
        """
        Initialize retrieval evaluator.

        Args:
            k_values: List of k values for @k metrics (default: [1, 3, 5, 10])
            llm_judge: Optional LLMJudge for scoring relevance when ground truth unavailable
            use_graded_relevance: Whether to use graded (0-1) or binary relevance
        """
        self.k_values = k_values or self.DEFAULT_K_VALUES
        self.llm_judge = llm_judge
        self.use_graded_relevance = use_graded_relevance

    def evaluate(
        self,
        queries: List[RetrievalQuery],
        aggregate: bool = True
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval quality across multiple queries.

        Args:
            queries: List of queries with retrieved results and ground truth
            aggregate: Whether to return aggregated metrics (True) or per-query

        Returns:
            RetrievalMetrics with all computed metrics
        """
        if not queries:
            return RetrievalMetrics()

        # Compute NDCG@k for each k
        ndcg_scores: Dict[int, List[float]] = {k: [] for k in self.k_values}
        precision_scores: Dict[int, List[float]] = {k: [] for k in self.k_values}
        recall_scores: Dict[int, List[float]] = {k: [] for k in self.k_values}

        for query in queries:
            # Get relevance grades (graded or binary)
            if query.relevance_grades and self.use_graded_relevance:
                rel_grades = query.relevance_grades
            else:
                # Convert to binary relevance
                rel_grades = {doc_id: 1.0 for doc_id in query.relevant_doc_ids}

            for k in self.k_values:
                # NDCG@k
                ndcg = ndcg_at_k(query.retrieved, rel_grades, k)
                ndcg_scores[k].append(ndcg)

                # P@k
                precision = precision_at_k(query.retrieved, query.relevant_doc_ids, k)
                precision_scores[k].append(precision)

                # R@k
                recall = recall_at_k(query.retrieved, query.relevant_doc_ids, k)
                recall_scores[k].append(recall)

        # Aggregate
        metrics = RetrievalMetrics(
            ndcg_at_k={k: sum(v) / len(v) for k, v in ndcg_scores.items()},
            mrr=mrr(queries),
            map_score=map_score(queries),
            precision_at_k={k: sum(v) / len(v) for k, v in precision_scores.items()},
            recall_at_k={k: sum(v) / len(v) for k, v in recall_scores.items()},
            hit_rate=hit_rate(queries, k=max(self.k_values)),
            total_queries=len(queries),
            metadata={
                "k_values": self.k_values,
                "use_graded_relevance": self.use_graded_relevance,
            }
        )

        return metrics

    async def evaluate_with_llm(
        self,
        queries: List[Tuple[str, List[str]]],  # (query_text, retrieved_contents)
        k: int = 10,
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval using LLM-based relevance scoring.

        When ground truth is unavailable, uses LLMJudge to score relevance
        of retrieved documents to the query.

        Args:
            queries: List of (query_text, retrieved_contents) tuples
            k: Maximum number of results to evaluate per query

        Returns:
            RetrievalMetrics with LLM-scored relevance
        """
        if self.llm_judge is None:
            raise ValueError("LLM judge required for evaluate_with_llm()")

        retrieval_queries = []

        for query_text, contents in queries:
            # Score each retrieved document
            results = []
            relevance_grades = {}
            relevant_docs = set()

            for i, content in enumerate(contents[:k]):
                doc_id = f"doc_{i}"

                # Use LLM judge to score relevance
                score = await self._score_relevance_llm(query_text, content)

                results.append(RetrievalResult(
                    doc_id=doc_id,
                    score=1.0 - i / k,  # Rank-based score
                    relevance=score,
                    content=content
                ))

                relevance_grades[doc_id] = score
                if score >= 0.5:  # Threshold for "relevant"
                    relevant_docs.add(doc_id)

            retrieval_queries.append(RetrievalQuery(
                query_id=f"q_{len(retrieval_queries)}",
                query_text=query_text,
                retrieved=results,
                relevant_doc_ids=relevant_docs,
                relevance_grades=relevance_grades
            ))

        return self.evaluate(retrieval_queries)

    async def _score_relevance_llm(self, query: str, document: str) -> float:
        """
        Score relevance of a document to a query using LLM judge.

        Args:
            query: Query text
            document: Document content

        Returns:
            Relevance score (0-1)
        """
        # Use LLMJudge if available
        if hasattr(self.llm_judge, 'score_relevance'):
            return await self.llm_judge.score_relevance(query, document)

        # Fallback: Use evaluate method
        if hasattr(self.llm_judge, 'evaluate'):
            result = await self.llm_judge.evaluate(
                output=document,
                context={"query": query}
            )
            # Extract relevance score
            if hasattr(result, 'get_score_by_criterion'):
                score = result.get_score_by_criterion('relevance')
                return score.score if score else 0.5
            return result.overall_score if hasattr(result, 'overall_score') else 0.5

        return 0.5  # Default neutral score

    def evaluate_multihop(
        self,
        queries: List[RetrievalQuery],
        paths: List[List[str]],  # Retrieval paths (sequences of doc IDs)
    ) -> Dict[str, Any]:
        """
        Evaluate multi-hop retrieval paths.

        Args:
            queries: Standard retrieval queries
            paths: Retrieval paths for each query

        Returns:
            Dict with standard metrics + path-specific metrics
        """
        base_metrics = self.evaluate(queries)

        # Path-specific metrics
        path_lengths = [len(p) for p in paths]
        avg_path_length = sum(path_lengths) / len(path_lengths) if path_lengths else 0

        # Path success rate (reached relevant doc)
        path_successes = 0
        for query, path in zip(queries, paths):
            if any(doc_id in query.relevant_doc_ids for doc_id in path):
                path_successes += 1
        path_success_rate = path_successes / len(queries) if queries else 0

        return {
            **base_metrics.to_dict(),
            "path_metrics": {
                "avg_path_length": avg_path_length,
                "path_success_rate": path_success_rate,
                "min_path_length": min(path_lengths) if path_lengths else 0,
                "max_path_length": max(path_lengths) if path_lengths else 0,
            }
        }

    def evaluate_temporal(
        self,
        queries: List[RetrievalQuery],
        timestamps: Dict[str, float],  # doc_id -> timestamp
        recency_window_days: float = 7.0,
    ) -> Dict[str, Any]:
        """
        Evaluate temporal retrieval (recency bias analysis).

        Args:
            queries: Standard retrieval queries
            timestamps: Document timestamps (Unix timestamps)
            recency_window_days: Window for "recent" classification

        Returns:
            Dict with standard metrics + temporal analysis
        """
        import time

        base_metrics = self.evaluate(queries)

        current_time = time.time()
        recency_threshold = current_time - (recency_window_days * 24 * 3600)

        # Analyze retrieval by recency
        recent_relevant = 0
        old_relevant = 0
        recent_retrieved = 0
        old_retrieved = 0

        for query in queries:
            for result in query.retrieved:
                ts = timestamps.get(result.doc_id, 0)
                is_recent = ts >= recency_threshold
                is_relevant = result.doc_id in query.relevant_doc_ids

                if is_recent:
                    recent_retrieved += 1
                    if is_relevant:
                        recent_relevant += 1
                else:
                    old_retrieved += 1
                    if is_relevant:
                        old_relevant += 1

        return {
            **base_metrics.to_dict(),
            "temporal_metrics": {
                "recent_precision": recent_relevant / recent_retrieved if recent_retrieved else 0,
                "old_precision": old_relevant / old_retrieved if old_retrieved else 0,
                "recency_bias": recent_retrieved / (recent_retrieved + old_retrieved) if (recent_retrieved + old_retrieved) else 0,
                "recent_retrieved": recent_retrieved,
                "old_retrieved": old_retrieved,
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_retrieval_query(
    query_id: str,
    query_text: str,
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
    relevance_grades: Optional[Dict[str, float]] = None,
) -> RetrievalQuery:
    """
    Convenience function to create a RetrievalQuery.

    Args:
        query_id: Unique query identifier
        query_text: Query text
        retrieved_doc_ids: List of retrieved document IDs (in rank order)
        relevant_doc_ids: List of ground truth relevant document IDs
        relevance_grades: Optional graded relevance (doc_id -> grade)

    Returns:
        RetrievalQuery instance
    """
    return RetrievalQuery(
        query_id=query_id,
        query_text=query_text,
        retrieved=[
            RetrievalResult(doc_id=doc_id, score=1.0 - i / len(retrieved_doc_ids))
            for i, doc_id in enumerate(retrieved_doc_ids)
        ],
        relevant_doc_ids=set(relevant_doc_ids),
        relevance_grades=relevance_grades,
    )


def evaluate_simple(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
    k_values: Optional[List[int]] = None,
) -> RetrievalMetrics:
    """
    Simple evaluation with a single query.

    Args:
        retrieved_doc_ids: Retrieved document IDs in rank order
        relevant_doc_ids: Ground truth relevant document IDs
        k_values: k values for @k metrics

    Returns:
        RetrievalMetrics
    """
    query = create_retrieval_query(
        query_id="single_query",
        query_text="",
        retrieved_doc_ids=retrieved_doc_ids,
        relevant_doc_ids=relevant_doc_ids,
    )

    evaluator = RetrievalEvaluator(k_values=k_values)
    return evaluator.evaluate([query])
