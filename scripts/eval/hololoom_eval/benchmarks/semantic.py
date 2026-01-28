"""
Semantic Quality Benchmark

Tests retrieval quality beyond keyword matching:
- Semantic similarity of retrieved results to query intent
- Diversity of retrieved results (coverage of relevant sub-topics)
- Ranking quality (most relevant first)
- Paraphrase invariance (rephrased queries should get similar results)
"""

import math
import time
from typing import Dict, List, Any, Optional

from ..runner import BenchmarkResult, EvalMode
from .datasets import get_corpus
from .retrieval import BM25Retriever, VectorRetriever, HybridRetriever


# ============================================================================
# SEMANTIC TEST QUERIES
# ============================================================================

# Paraphrase pairs: semantically equivalent queries
PARAPHRASE_PAIRS: List[Dict[str, Any]] = [
    {
        "id": "para_1",
        "original": "What is Thompson Sampling?",
        "paraphrase": "Explain the Thompson Sampling algorithm",
        "expected_overlap_ratio": 0.6,  # At least 60% overlap in results
    },
    {
        "id": "para_2",
        "original": "How do transformers work?",
        "paraphrase": "Describe the transformer architecture mechanism",
        "expected_overlap_ratio": 0.6,
    },
    {
        "id": "para_3",
        "original": "What retrieval methods are used in RAG systems?",
        "paraphrase": "Which search techniques do retrieval-augmented generation systems use?",
        "expected_overlap_ratio": 0.4,
    },
    {
        "id": "para_4",
        "original": "How do bandit algorithms balance exploration and exploitation?",
        "paraphrase": "What is the explore-exploit tradeoff in multi-armed bandits?",
        "expected_overlap_ratio": 0.4,
    },
]

# Semantic intent queries: test understanding vs keyword matching
INTENT_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "intent_1",
        "query": "I need to decide between trying new options and sticking with what works",
        "description": "Exploration-exploitation without technical terms",
        "relevant_ids": ["ml_1", "ml_3", "ml_4", "ml_5", "hl_3"],
        "requires_semantic": True,
    },
    {
        "id": "intent_2",
        "query": "How can a system find relevant information from a large knowledge base?",
        "description": "Retrieval described in plain language",
        "relevant_ids": ["rag_1", "rag_3", "rag_4", "ir_1", "ir_4"],
        "requires_semantic": True,
    },
    {
        "id": "intent_3",
        "query": "What approach combines the best of both keyword and meaning-based search?",
        "description": "Hybrid retrieval in plain language",
        "relevant_ids": ["rag_3", "ir_1", "ir_4"],
        "requires_semantic": True,
    },
    {
        "id": "intent_4",
        "query": "How does a computer learn from its mistakes?",
        "description": "Backpropagation in plain language",
        "relevant_ids": ["nn_1", "nn_4", "cs_1"],
        "requires_semantic": True,
    },
    {
        "id": "intent_5",
        "query": "updating beliefs with new evidence",
        "description": "Bayesian inference without technical terms",
        "relevant_ids": ["ml_6", "ml_7", "ml_8"],
        "requires_semantic": True,
    },
]

# Diversity test queries: should retrieve across sub-topics
DIVERSITY_QUERIES: List[Dict[str, Any]] = [
    {
        "id": "div_1",
        "query": "machine learning algorithms",
        "subtopics": {
            "bandits": ["ml_1", "ml_2", "ml_3", "ml_4"],
            "bayesian": ["ml_6", "ml_7", "ml_8"],
            "neural": ["nn_1", "nn_2", "nn_3", "nn_4"],
            "rl": ["cs_1"],
        },
        "min_subtopics_covered": 2,
    },
    {
        "id": "div_2",
        "query": "information retrieval techniques",
        "subtopics": {
            "traditional_ir": ["ir_1", "ir_3"],
            "semantic_ir": ["ir_2", "ir_4"],
            "rag": ["rag_1", "rag_3", "rag_4"],
        },
        "min_subtopics_covered": 2,
    },
]


def _jaccard_similarity(set1: set, set2: set) -> float:
    """Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


def _overlap_ratio(list1: List[str], list2: List[str]) -> float:
    """Fraction of list1 elements that appear in list2."""
    if not list1:
        return 0.0
    set2 = set(list2)
    return sum(1 for x in list1 if x in set2) / len(list1)


def _reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """MRR: 1/position of first relevant result."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """NDCG at k."""
    relevant_set = set(relevant)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg > 0 else 0.0


async def run_semantic_benchmark(mode: EvalMode, deps: Dict[str, bool]) -> BenchmarkResult:
    """
    Run semantic quality benchmark.

    Evaluates:
    1. Paraphrase invariance: do rephrased queries get similar results?
    2. Intent understanding: can the system understand non-technical queries?
    3. Result diversity: does retrieval cover multiple sub-topics?
    4. Ranking quality: are most relevant results ranked first?
    """
    corpus = get_corpus()
    k = 5

    # Set up retrievers
    bm25 = BM25Retriever(corpus)
    hybrid = HybridRetriever(corpus)

    has_vector = deps.get("sentence_transformers", False) or deps.get("numpy", False)

    # Select test sets based on mode
    if mode == EvalMode.QUICK:
        paraphrase_tests = PARAPHRASE_PAIRS[:2]
        intent_tests = INTENT_QUERIES[:2]
        diversity_tests = DIVERSITY_QUERIES[:1]
    elif mode == EvalMode.STANDARD:
        paraphrase_tests = PARAPHRASE_PAIRS
        intent_tests = INTENT_QUERIES[:3]
        diversity_tests = DIVERSITY_QUERIES
    else:
        paraphrase_tests = PARAPHRASE_PAIRS
        intent_tests = INTENT_QUERIES
        diversity_tests = DIVERSITY_QUERIES

    scores = {
        "paraphrase_invariance": [],
        "intent_understanding": [],
        "diversity": [],
        "bm25_vs_hybrid_intent": [],
    }

    # 1. Paraphrase invariance
    for pair in paraphrase_tests:
        orig_results = hybrid.retrieve(pair["original"], k=k)
        para_results = hybrid.retrieve(pair["paraphrase"], k=k)

        overlap = _overlap_ratio(orig_results, para_results)
        bi_overlap = (_overlap_ratio(orig_results, para_results) + _overlap_ratio(para_results, orig_results)) / 2

        passed = bi_overlap >= pair["expected_overlap_ratio"]
        scores["paraphrase_invariance"].append(bi_overlap)

    # 2. Intent understanding (BM25 vs Hybrid)
    for intent_q in intent_tests:
        bm25_results = bm25.retrieve(intent_q["query"], k=k)
        hybrid_results = hybrid.retrieve(intent_q["query"], k=k)

        bm25_ndcg = _ndcg_at_k(bm25_results, intent_q["relevant_ids"], k)
        hybrid_ndcg = _ndcg_at_k(hybrid_results, intent_q["relevant_ids"], k)

        # Semantic understanding = how much better hybrid does on intent queries
        scores["intent_understanding"].append(hybrid_ndcg)
        scores["bm25_vs_hybrid_intent"].append(hybrid_ndcg - bm25_ndcg)

    # 3. Result diversity
    for div_q in diversity_tests:
        results = hybrid.retrieve(div_q["query"], k=k * 2)  # Get more for diversity
        results_set = set(results)

        subtopics_covered = 0
        for subtopic_name, subtopic_ids in div_q["subtopics"].items():
            if results_set & set(subtopic_ids):
                subtopics_covered += 1

        total_subtopics = len(div_q["subtopics"])
        coverage = subtopics_covered / total_subtopics if total_subtopics > 0 else 0
        scores["diversity"].append(coverage)

    # Calculate aggregate scores
    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    para_score = safe_mean(scores["paraphrase_invariance"])
    intent_score = safe_mean(scores["intent_understanding"])
    diversity_score = safe_mean(scores["diversity"])
    semantic_advantage = safe_mean(scores["bm25_vs_hybrid_intent"])

    # Overall: weighted combination
    overall = 0.35 * para_score + 0.35 * intent_score + 0.30 * diversity_score

    # Verdict
    if overall >= 0.7 and para_score >= 0.5:
        verdict = f"Strong semantic quality (intent={intent_score:.2f}, paraphrase={para_score:.2f}, diversity={diversity_score:.2f})"
        passed = True
    elif overall >= 0.4:
        verdict = f"Moderate semantic quality (intent={intent_score:.2f}, paraphrase={para_score:.2f})"
        passed = True
    else:
        verdict = f"Weak semantic quality (intent={intent_score:.2f}, paraphrase={para_score:.2f})"
        passed = False

    if not has_vector:
        verdict += " [BM25 only - install sentence-transformers for semantic search]"

    return BenchmarkResult(
        name="Semantic Quality",
        passed=passed,
        score=overall,
        verdict=verdict,
        details={
            "paraphrase_invariance": para_score,
            "intent_understanding": intent_score,
            "diversity_coverage": diversity_score,
            "semantic_advantage_over_bm25": semantic_advantage,
            "has_vector_search": has_vector,
            "num_paraphrase_tests": len(paraphrase_tests),
            "num_intent_tests": len(intent_tests),
            "num_diversity_tests": len(diversity_tests),
            "raw_scores": {k: [round(v, 4) for v in vs] for k, vs in scores.items()},
        }
    )
