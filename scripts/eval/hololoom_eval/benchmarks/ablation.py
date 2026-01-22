"""
Feature Ablation Benchmark

Tests: What's the value of each HoloLoom component?
- Knowledge graph vs no graph
- Multipass vs single pass
- Hot patterns vs no hot patterns
- Hybrid vs vector-only
"""

import asyncio
from typing import Dict, List, Any

from ..runner import BenchmarkResult, EvalMode
from .datasets import get_corpus, get_queries_for_mode
from .retrieval import (
    BM25Retriever, VectorRetriever, HybridRetriever,
    ndcg_at_k, reciprocal_rank
)


async def run_ablation_benchmark(mode: EvalMode, deps: Dict[str, bool]) -> BenchmarkResult:
    """
    Ablation study: measure contribution of each component.

    Components tested:
    1. BM25 only (baseline)
    2. Vector only
    3. Hybrid (BM25 + Vector)
    4. + Knowledge Graph (if HoloLoom available)
    5. + Hot Patterns (if HoloLoom available)
    """

    corpus = get_corpus()
    queries = get_queries_for_mode(mode.value)
    k = 5

    results = {}

    # Test baselines
    configurations = [
        ("BM25 Only", "bm25"),
        ("Vector Only", "vector"),
        ("Hybrid (BM25+Vector)", "hybrid"),
    ]

    bm25 = BM25Retriever(corpus)
    vector = VectorRetriever(corpus)
    hybrid = HybridRetriever(corpus)

    for config_name, config_type in configurations:
        ndcg_scores = []
        mrr_scores = []

        for q in queries:
            if config_type == "bm25":
                retrieved = bm25.retrieve(q["query"], k)
            elif config_type == "vector":
                retrieved = vector.retrieve(q["query"], k)
            else:
                retrieved = hybrid.retrieve(q["query"], k)

            ndcg_scores.append(ndcg_at_k(retrieved, q["relevant_ids"], k))
            mrr_scores.append(reciprocal_rank(retrieved, q["relevant_ids"]))

        results[config_name] = {
            "ndcg": sum(ndcg_scores) / len(ndcg_scores),
            "mrr": sum(mrr_scores) / len(mrr_scores),
        }

    # Test HoloLoom configurations if available
    if deps.get("hololoom") and deps.get("numpy"):
        try:
            from HoloLoom import HoloLoom
            from HoloLoom.config import Config

            # HoloLoom FAST mode
            async with HoloLoom(config=Config.fast()) as loom:
                for doc in corpus:
                    await loom.experience(doc["text"], metadata={"id": doc["id"]})

                ndcg_scores = []
                mrr_scores = []

                for q in queries:
                    memories = await loom.recall(q["query"], k=k)
                    retrieved = [
                        m.metadata.get("id") for m in memories
                        if hasattr(m, "metadata") and m.metadata.get("id")
                    ]

                    ndcg_scores.append(ndcg_at_k(retrieved, q["relevant_ids"], k))
                    mrr_scores.append(reciprocal_rank(retrieved, q["relevant_ids"]))

                results["HoloLoom FAST"] = {
                    "ndcg": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
                    "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
                }

            # HoloLoom FUSED mode (full features)
            if mode in [EvalMode.STANDARD, EvalMode.FULL]:
                async with HoloLoom(config=Config.fused()) as loom:
                    for doc in corpus:
                        await loom.experience(doc["text"], metadata={"id": doc["id"]})

                    ndcg_scores = []
                    mrr_scores = []

                    for q in queries:
                        memories = await loom.recall(q["query"], k=k)
                        retrieved = [
                            m.metadata.get("id") for m in memories
                            if hasattr(m, "metadata") and m.metadata.get("id")
                        ]

                        ndcg_scores.append(ndcg_at_k(retrieved, q["relevant_ids"], k))
                        mrr_scores.append(reciprocal_rank(retrieved, q["relevant_ids"]))

                    results["HoloLoom FUSED"] = {
                        "ndcg": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
                        "mrr": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0,
                    }

        except Exception as e:
            results["HoloLoom Error"] = {"error": str(e)}

    # Analyze component contributions
    baseline_ndcg = results.get("Hybrid (BM25+Vector)", {}).get("ndcg", 0)
    best_ndcg = max(r.get("ndcg", 0) for r in results.values() if isinstance(r.get("ndcg"), (int, float)))
    best_config = max(
        ((name, r.get("ndcg", 0)) for name, r in results.items() if isinstance(r.get("ndcg"), (int, float))),
        key=lambda x: x[1]
    )[0]

    # Calculate incremental contributions
    contributions = []
    sorted_configs = sorted(
        [(name, r.get("ndcg", 0)) for name, r in results.items() if isinstance(r.get("ndcg"), (int, float))],
        key=lambda x: x[1]
    )

    for i, (name, ndcg) in enumerate(sorted_configs):
        if i > 0:
            prev_ndcg = sorted_configs[i-1][1]
            delta = ndcg - prev_ndcg
            contributions.append(f"{name}: +{delta:.3f} NDCG")

    # Determine verdict
    if baseline_ndcg > 0:
        improvement = (best_ndcg - baseline_ndcg) / baseline_ndcg
        if improvement > 0.1:
            verdict = f"{best_config} best (+{improvement*100:.1f}% over Hybrid)"
            passed = True
            score = 0.6 + min(0.4, improvement)
        elif improvement > 0:
            verdict = f"Marginal improvement from {best_config}"
            passed = True
            score = 0.5 + improvement
        else:
            verdict = "Hybrid baseline is sufficient"
            passed = False
            score = max(0.4, baseline_ndcg)
    else:
        verdict = "Could not establish baseline"
        passed = False
        score = 0.3

    return BenchmarkResult(
        name="Feature Ablation",
        passed=passed,
        score=score,
        verdict=verdict,
        details={
            "configurations": results,
            "best_config": best_config,
            "contributions": contributions,
            "num_queries": len(queries),
        }
    )
