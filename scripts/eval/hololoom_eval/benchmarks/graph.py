"""
Knowledge Graph Value Benchmark

Tests: Does graph traversal improve retrieval?
- Compare vector-only vs vector+graph
- Test multi-hop retrieval benefit
- Measure entity relationship utilization
"""

import asyncio
from typing import Dict, List, Any

from ..runner import BenchmarkResult, EvalMode
from .datasets import get_corpus, get_queries_for_mode
from .retrieval import VectorRetriever, ndcg_at_k, reciprocal_rank


# Queries specifically designed to benefit from graph traversal
GRAPH_TEST_QUERIES = [
    {
        "query": "How does Thompson Sampling relate to Bayesian inference?",
        "relevant_ids": ["ml_1", "ml_6", "ml_7", "hl_3"],
        "requires_graph": True,
        "reason": "Requires connecting Thompson Sampling → Bayesian concepts",
    },
    {
        "query": "What techniques combine keyword and semantic search?",
        "relevant_ids": ["rag_3", "ir_1", "ir_2", "ir_4"],
        "requires_graph": True,
        "reason": "Requires connecting hybrid retrieval → BM25 → embeddings",
    },
    {
        "query": "How do exploration strategies affect reinforcement learning?",
        "relevant_ids": ["ml_1", "ml_3", "ml_4", "ml_5", "cs_1"],
        "requires_graph": True,
        "reason": "Requires connecting multiple exploration methods → RL",
    },
    {
        "query": "What is the relationship between transformers and attention?",
        "relevant_ids": ["nn_2", "nn_3"],
        "requires_graph": False,  # Control: should work without graph
        "reason": "Direct relationship, graph not needed",
    },
]


async def run_graph_benchmark(mode: EvalMode, deps: Dict[str, bool]) -> BenchmarkResult:
    """
    Test whether knowledge graph traversal adds value.

    Methodology:
    1. Test queries that should benefit from graph relationships
    2. Compare vector-only vs HoloLoom (which includes graph)
    3. Measure improvement specifically on graph-dependent queries
    """

    corpus = get_corpus()
    k = 5

    # Use specialized graph test queries
    queries = GRAPH_TEST_QUERIES if mode != EvalMode.QUICK else GRAPH_TEST_QUERIES[:2]

    # Vector-only baseline
    vector = VectorRetriever(corpus)

    vector_results = {"graph_queries": [], "control_queries": []}

    for q in queries:
        retrieved = vector.retrieve(q["query"], k)
        ndcg = ndcg_at_k(retrieved, q["relevant_ids"], k)
        mrr = reciprocal_rank(retrieved, q["relevant_ids"])

        result = {"query": q["query"], "ndcg": ndcg, "mrr": mrr}

        if q["requires_graph"]:
            vector_results["graph_queries"].append(result)
        else:
            vector_results["control_queries"].append(result)

    # HoloLoom with graph (if available)
    hololoom_results = {"graph_queries": [], "control_queries": []}

    if deps.get("hololoom") and deps.get("numpy"):
        try:
            from HoloLoom import HoloLoom
            from HoloLoom.config import Config

            async with HoloLoom(config=Config.fused()) as loom:
                # Ingest with explicit entity relationships
                for doc in corpus:
                    await loom.experience(doc["text"], metadata={"id": doc["id"]})

                for q in queries:
                    memories = await loom.recall(q["query"], k=k)
                    retrieved = [
                        m.metadata.get("id") for m in memories
                        if hasattr(m, "metadata") and m.metadata.get("id")
                    ]

                    ndcg = ndcg_at_k(retrieved, q["relevant_ids"], k)
                    mrr = reciprocal_rank(retrieved, q["relevant_ids"])

                    result = {"query": q["query"], "ndcg": ndcg, "mrr": mrr}

                    if q["requires_graph"]:
                        hololoom_results["graph_queries"].append(result)
                    else:
                        hololoom_results["control_queries"].append(result)

        except Exception as e:
            return BenchmarkResult(
                name="Knowledge Graph Value",
                passed=False,
                score=0.3,
                verdict=f"HoloLoom error: {str(e)[:40]}",
                details={"error": str(e)}
            )
    else:
        # Can't test HoloLoom - report vector-only baseline
        avg_ndcg = sum(r["ndcg"] for r in vector_results["graph_queries"]) / max(1, len(vector_results["graph_queries"]))

        return BenchmarkResult(
            name="Knowledge Graph Value",
            passed=avg_ndcg > 0.5,
            score=avg_ndcg,
            verdict=f"Vector-only baseline: NDCG={avg_ndcg:.3f} (graph not tested)",
            details={
                "vector_results": vector_results,
                "hololoom_tested": False,
            }
        )

    # Compare results
    def avg_ndcg(results: List[Dict]) -> float:
        return sum(r["ndcg"] for r in results) / max(1, len(results))

    vector_graph_ndcg = avg_ndcg(vector_results["graph_queries"])
    vector_control_ndcg = avg_ndcg(vector_results["control_queries"])
    hololoom_graph_ndcg = avg_ndcg(hololoom_results["graph_queries"])
    hololoom_control_ndcg = avg_ndcg(hololoom_results["control_queries"])

    # Calculate graph benefit
    graph_improvement = (hololoom_graph_ndcg - vector_graph_ndcg) / vector_graph_ndcg if vector_graph_ndcg > 0 else 0
    control_improvement = (hololoom_control_ndcg - vector_control_ndcg) / vector_control_ndcg if vector_control_ndcg > 0 else 0

    # Graph value = improvement on graph queries - improvement on control queries
    # (If graph helps everywhere, it's just better retrieval, not graph-specific)
    graph_specific_value = graph_improvement - control_improvement

    # Determine verdict
    if graph_specific_value > 0.1:
        verdict = f"Graph adds value: +{graph_specific_value*100:.1f}% on relationship queries"
        passed = True
        score = 0.6 + min(0.4, graph_specific_value)
    elif graph_improvement > 0:
        verdict = f"HoloLoom better overall (+{graph_improvement*100:.1f}%), but not graph-specific"
        passed = True
        score = 0.5 + min(0.2, graph_improvement)
    else:
        verdict = "Graph traversal provides no measurable benefit"
        passed = False
        score = max(0.3, hololoom_graph_ndcg)

    return BenchmarkResult(
        name="Knowledge Graph Value",
        passed=passed,
        score=score,
        verdict=verdict,
        details={
            "vector_results": vector_results,
            "hololoom_results": hololoom_results,
            "graph_improvement_pct": graph_improvement * 100,
            "control_improvement_pct": control_improvement * 100,
            "graph_specific_value_pct": graph_specific_value * 100,
        }
    )
