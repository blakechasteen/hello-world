"""
Tests for Retrieval Quality Evaluation
======================================

Comprehensive tests for NDCG, MRR, MAP, Precision@k, Recall@k metrics.

Author: HoloLoom Architecture Team
Date: December 2025
"""

import pytest
from typing import List, Dict

from HoloLoom.model_extension.eval.retrieval_metrics import (
    RetrievalMetrics,
    RetrievalQuery,
    RetrievalResult,
    RetrievalEvaluator,
    ndcg_at_k,
    mrr,
    map_score,
    precision_at_k,
    recall_at_k,
    hit_rate,
    dcg_at_k,
    average_precision,
    create_retrieval_query,
    evaluate_simple,
)


# =============================================================================
# DCG and NDCG Tests
# =============================================================================

class TestDCG:
    """Tests for DCG (Discounted Cumulative Gain)."""

    def test_dcg_perfect_ranking(self):
        """Test DCG with perfect ranking."""
        relevances = [3, 2, 1, 0]  # Perfect descending order
        dcg = dcg_at_k(relevances, k=4)
        # DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4) + 0
        # DCG = 7/1 + 3/1.585 + 1/2 = 7 + 1.893 + 0.5 = 9.393
        assert dcg > 9.0  # Approximate check

    def test_dcg_reverse_ranking(self):
        """Test DCG with reverse ranking (worst case)."""
        relevances = [0, 1, 2, 3]  # Worst order
        dcg = dcg_at_k(relevances, k=4)
        # DCG should be lower than perfect ranking
        perfect_dcg = dcg_at_k([3, 2, 1, 0], k=4)
        assert dcg < perfect_dcg

    def test_dcg_empty(self):
        """Test DCG with empty list."""
        assert dcg_at_k([], k=10) == 0.0

    def test_dcg_k_truncation(self):
        """Test that DCG respects k cutoff."""
        relevances = [3, 2, 1, 0, 0, 0]
        dcg_k3 = dcg_at_k(relevances, k=3)
        dcg_k6 = dcg_at_k(relevances, k=6)
        # DCG@6 should equal DCG@3 (remaining docs have 0 relevance)
        assert abs(dcg_k3 - dcg_k6) < 0.01


class TestNDCG:
    """Tests for NDCG (Normalized DCG)."""

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking should be 1.0."""
        retrieved = [
            RetrievalResult(doc_id="d1", score=0.9),
            RetrievalResult(doc_id="d2", score=0.8),
            RetrievalResult(doc_id="d3", score=0.7),
        ]
        relevance_grades = {"d1": 1.0, "d2": 0.8, "d3": 0.6}

        ndcg = ndcg_at_k(retrieved, relevance_grades, k=3)
        assert ndcg == pytest.approx(1.0, rel=0.01)

    def test_ndcg_reverse_ranking(self):
        """Test NDCG with reverse ranking should be less than 1."""
        retrieved = [
            RetrievalResult(doc_id="d3", score=0.9),
            RetrievalResult(doc_id="d2", score=0.8),
            RetrievalResult(doc_id="d1", score=0.7),
        ]
        relevance_grades = {"d1": 1.0, "d2": 0.8, "d3": 0.6}

        ndcg = ndcg_at_k(retrieved, relevance_grades, k=3)
        assert ndcg < 1.0
        assert ndcg > 0.5  # Still reasonable

    def test_ndcg_no_relevant_docs(self):
        """Test NDCG when no relevant docs are retrieved."""
        retrieved = [
            RetrievalResult(doc_id="irrelevant1", score=0.9),
            RetrievalResult(doc_id="irrelevant2", score=0.8),
        ]
        relevance_grades = {"relevant1": 1.0, "relevant2": 0.8}

        ndcg = ndcg_at_k(retrieved, relevance_grades, k=2)
        assert ndcg == 0.0

    def test_ndcg_empty_inputs(self):
        """Test NDCG with empty inputs."""
        assert ndcg_at_k([], {}, k=10) == 0.0
        assert ndcg_at_k([], {"d1": 1.0}, k=10) == 0.0


# =============================================================================
# MRR Tests
# =============================================================================

class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_mrr_first_relevant(self):
        """Test MRR when first doc is relevant."""
        queries = [
            RetrievalQuery(
                query_id="q1",
                query_text="test",
                retrieved=[
                    RetrievalResult(doc_id="relevant", score=0.9),
                    RetrievalResult(doc_id="irrelevant", score=0.8),
                ],
                relevant_doc_ids={"relevant"},
            )
        ]
        assert mrr(queries) == 1.0

    def test_mrr_second_relevant(self):
        """Test MRR when second doc is relevant."""
        queries = [
            RetrievalQuery(
                query_id="q1",
                query_text="test",
                retrieved=[
                    RetrievalResult(doc_id="irrelevant", score=0.9),
                    RetrievalResult(doc_id="relevant", score=0.8),
                ],
                relevant_doc_ids={"relevant"},
            )
        ]
        assert mrr(queries) == 0.5

    def test_mrr_no_relevant(self):
        """Test MRR when no relevant docs found."""
        queries = [
            RetrievalQuery(
                query_id="q1",
                query_text="test",
                retrieved=[
                    RetrievalResult(doc_id="irrelevant1", score=0.9),
                    RetrievalResult(doc_id="irrelevant2", score=0.8),
                ],
                relevant_doc_ids={"relevant"},
            )
        ]
        assert mrr(queries) == 0.0

    def test_mrr_multiple_queries(self):
        """Test MRR with multiple queries."""
        queries = [
            RetrievalQuery(
                query_id="q1",
                query_text="test1",
                retrieved=[
                    RetrievalResult(doc_id="rel1", score=0.9),
                ],
                relevant_doc_ids={"rel1"},
            ),
            RetrievalQuery(
                query_id="q2",
                query_text="test2",
                retrieved=[
                    RetrievalResult(doc_id="irrel", score=0.9),
                    RetrievalResult(doc_id="rel2", score=0.8),
                ],
                relevant_doc_ids={"rel2"},
            ),
        ]
        # (1/1 + 1/2) / 2 = 0.75
        assert mrr(queries) == pytest.approx(0.75, rel=0.01)

    def test_mrr_empty(self):
        """Test MRR with empty queries list."""
        assert mrr([]) == 0.0


# =============================================================================
# MAP Tests
# =============================================================================

class TestMAP:
    """Tests for Mean Average Precision."""

    def test_average_precision_perfect(self):
        """Test AP with perfect retrieval."""
        retrieved = [
            RetrievalResult(doc_id="rel1", score=0.9),
            RetrievalResult(doc_id="rel2", score=0.8),
            RetrievalResult(doc_id="rel3", score=0.7),
        ]
        relevant_ids = {"rel1", "rel2", "rel3"}

        ap = average_precision(retrieved, relevant_ids)
        # AP = (1/1 + 2/2 + 3/3) / 3 = (1 + 1 + 1) / 3 = 1.0
        assert ap == pytest.approx(1.0, rel=0.01)

    def test_average_precision_alternating(self):
        """Test AP with alternating relevant/irrelevant."""
        retrieved = [
            RetrievalResult(doc_id="rel1", score=0.9),
            RetrievalResult(doc_id="irrel1", score=0.8),
            RetrievalResult(doc_id="rel2", score=0.7),
        ]
        relevant_ids = {"rel1", "rel2"}

        ap = average_precision(retrieved, relevant_ids)
        # AP = (1/1 + 2/3) / 2 = (1 + 0.667) / 2 = 0.833
        assert ap == pytest.approx(0.833, rel=0.01)

    def test_map_multiple_queries(self, mixed_retrieval_queries):
        """Test MAP with multiple queries."""
        map_result = map_score(mixed_retrieval_queries)
        # Should be between 0 and 1
        assert 0.0 <= map_result <= 1.0

    def test_map_empty(self):
        """Test MAP with empty queries."""
        assert map_score([]) == 0.0


# =============================================================================
# Precision@k Tests
# =============================================================================

class TestPrecisionAtK:
    """Tests for Precision@k."""

    def test_precision_all_relevant(self):
        """Test P@k when all top-k are relevant."""
        retrieved = [
            RetrievalResult(doc_id="rel1", score=0.9),
            RetrievalResult(doc_id="rel2", score=0.8),
            RetrievalResult(doc_id="rel3", score=0.7),
        ]
        relevant_ids = {"rel1", "rel2", "rel3", "rel4", "rel5"}

        assert precision_at_k(retrieved, relevant_ids, k=3) == 1.0

    def test_precision_none_relevant(self):
        """Test P@k when none of top-k are relevant."""
        retrieved = [
            RetrievalResult(doc_id="irrel1", score=0.9),
            RetrievalResult(doc_id="irrel2", score=0.8),
        ]
        relevant_ids = {"rel1", "rel2"}

        assert precision_at_k(retrieved, relevant_ids, k=2) == 0.0

    def test_precision_mixed(self):
        """Test P@k with mixed relevance."""
        retrieved = [
            RetrievalResult(doc_id="rel1", score=0.9),
            RetrievalResult(doc_id="irrel1", score=0.8),
            RetrievalResult(doc_id="rel2", score=0.7),
            RetrievalResult(doc_id="irrel2", score=0.6),
        ]
        relevant_ids = {"rel1", "rel2"}

        assert precision_at_k(retrieved, relevant_ids, k=4) == 0.5

    def test_precision_k_larger_than_retrieved(self):
        """Test P@k when k > number of retrieved docs."""
        retrieved = [
            RetrievalResult(doc_id="rel1", score=0.9),
        ]
        relevant_ids = {"rel1"}

        # Should use min(k, len(retrieved)) effectively
        assert precision_at_k(retrieved, relevant_ids, k=10) == 0.1


# =============================================================================
# Recall@k Tests
# =============================================================================

class TestRecallAtK:
    """Tests for Recall@k."""

    def test_recall_all_found(self):
        """Test R@k when all relevant found in top-k."""
        retrieved = [
            RetrievalResult(doc_id="rel1", score=0.9),
            RetrievalResult(doc_id="rel2", score=0.8),
        ]
        relevant_ids = {"rel1", "rel2"}

        assert recall_at_k(retrieved, relevant_ids, k=2) == 1.0

    def test_recall_none_found(self):
        """Test R@k when none of relevant found."""
        retrieved = [
            RetrievalResult(doc_id="irrel1", score=0.9),
            RetrievalResult(doc_id="irrel2", score=0.8),
        ]
        relevant_ids = {"rel1", "rel2"}

        assert recall_at_k(retrieved, relevant_ids, k=2) == 0.0

    def test_recall_partial(self):
        """Test R@k with partial recall."""
        retrieved = [
            RetrievalResult(doc_id="rel1", score=0.9),
            RetrievalResult(doc_id="irrel1", score=0.8),
        ]
        relevant_ids = {"rel1", "rel2", "rel3"}

        assert recall_at_k(retrieved, relevant_ids, k=2) == pytest.approx(1/3, rel=0.01)


# =============================================================================
# Hit Rate Tests
# =============================================================================

class TestHitRate:
    """Tests for Hit Rate."""

    def test_hit_rate_all_hits(self, perfect_retrieval_query):
        """Test hit rate when all queries have hits."""
        queries = [perfect_retrieval_query] * 3
        assert hit_rate(queries, k=5) == 1.0

    def test_hit_rate_no_hits(self, poor_retrieval_query):
        """Test hit rate when no queries have hits in top-3."""
        queries = [poor_retrieval_query]
        # relevant1 is at position 4, so no hit in top-3
        assert hit_rate(queries, k=3) == 0.0

    def test_hit_rate_mixed(self, perfect_retrieval_query, poor_retrieval_query):
        """Test hit rate with mixed results."""
        queries = [perfect_retrieval_query, poor_retrieval_query]
        # Perfect query has hit at position 1
        # Poor query has hit at position 4, so no hit in top-3
        assert hit_rate(queries, k=3) == 0.5


# =============================================================================
# RetrievalEvaluator Tests
# =============================================================================

class TestRetrievalEvaluator:
    """Tests for the main RetrievalEvaluator class."""

    def test_evaluate_basic(self, mixed_retrieval_queries, retrieval_evaluator):
        """Test basic evaluation."""
        metrics = retrieval_evaluator.evaluate(mixed_retrieval_queries)

        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.total_queries == len(mixed_retrieval_queries)
        assert 0.0 <= metrics.mrr <= 1.0
        assert 0.0 <= metrics.map_score <= 1.0
        assert 0.0 <= metrics.hit_rate <= 1.0

    def test_evaluate_ndcg_at_all_k(self, mixed_retrieval_queries, retrieval_evaluator):
        """Test NDCG is computed at all k values."""
        metrics = retrieval_evaluator.evaluate(mixed_retrieval_queries)

        for k in [1, 3, 5, 10]:
            assert k in metrics.ndcg_at_k
            assert 0.0 <= metrics.ndcg_at_k[k] <= 1.0

    def test_evaluate_precision_recall_at_all_k(self, mixed_retrieval_queries, retrieval_evaluator):
        """Test precision and recall at all k values."""
        metrics = retrieval_evaluator.evaluate(mixed_retrieval_queries)

        for k in [1, 3, 5, 10]:
            assert k in metrics.precision_at_k
            assert k in metrics.recall_at_k
            assert 0.0 <= metrics.precision_at_k[k] <= 1.0
            assert 0.0 <= metrics.recall_at_k[k] <= 1.0

    def test_evaluate_empty(self, retrieval_evaluator):
        """Test evaluation with empty query list."""
        metrics = retrieval_evaluator.evaluate([])

        assert metrics.total_queries == 0
        assert metrics.mrr == 0.0
        assert metrics.map_score == 0.0

    def test_evaluate_with_graded_relevance(self, perfect_retrieval_query, retrieval_evaluator):
        """Test evaluation with graded relevance."""
        retrieval_evaluator.use_graded_relevance = True
        metrics = retrieval_evaluator.evaluate([perfect_retrieval_query])

        # With graded relevance, NDCG should use actual grades
        assert metrics.ndcg_at_k[3] > 0.0

    def test_evaluate_multihop(self, mixed_retrieval_queries, retrieval_evaluator):
        """Test multi-hop evaluation."""
        paths = [
            ["start", "middle", relevant_id]
            for q in mixed_retrieval_queries
            for relevant_id in list(q.relevant_doc_ids)[:1]
        ][:len(mixed_retrieval_queries)]

        result = retrieval_evaluator.evaluate_multihop(
            mixed_retrieval_queries,
            paths
        )

        assert "path_metrics" in result
        assert "avg_path_length" in result["path_metrics"]
        assert "path_success_rate" in result["path_metrics"]

    def test_to_dict(self, mixed_retrieval_queries, retrieval_evaluator):
        """Test metrics serialization."""
        metrics = retrieval_evaluator.evaluate(mixed_retrieval_queries)
        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert "ndcg_at_k" in result
        assert "mrr" in result
        assert "map_score" in result

    def test_summary(self, mixed_retrieval_queries, retrieval_evaluator):
        """Test human-readable summary."""
        metrics = retrieval_evaluator.evaluate(mixed_retrieval_queries)
        summary = metrics.summary()

        assert isinstance(summary, str)
        assert "MRR" in summary
        assert "MAP" in summary
        assert "NDCG" in summary


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_retrieval_query(self):
        """Test create_retrieval_query helper."""
        query = create_retrieval_query(
            query_id="test_q1",
            query_text="What is X?",
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids=["d1", "d3"],
        )

        assert query.query_id == "test_q1"
        assert len(query.retrieved) == 3
        assert query.relevant_doc_ids == {"d1", "d3"}

    def test_evaluate_simple(self):
        """Test simple evaluation helper."""
        metrics = evaluate_simple(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids=["d1", "d2"],
            k_values=[1, 3],
        )

        assert isinstance(metrics, RetrievalMetrics)
        assert metrics.total_queries == 1
        assert 1 in metrics.ndcg_at_k
        assert 3 in metrics.ndcg_at_k
