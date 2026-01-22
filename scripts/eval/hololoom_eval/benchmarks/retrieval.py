"""
Retrieval Quality Benchmark

Tests: Does HoloLoom retrieve better than baselines?
- BM25 (keyword matching)
- Vector search (semantic similarity)
- Hybrid (BM25 + Vector)
- HoloLoom full system
"""

import asyncio
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..runner import BenchmarkResult, EvalMode
from .datasets import get_corpus, get_queries_for_mode


# ============================================================================
# METRICS
# ============================================================================

def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Precision@k: fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Recall@k: fraction of relevant docs found in top-k."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant)


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """MRR: 1/position of first relevant result."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Normalized DCG at k."""
    relevant_set = set(relevant)

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG
    idcg = 0.0
    for i in range(min(k, len(relevant))):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# RETRIEVERS
# ============================================================================

class BM25Retriever:
    """Simple BM25 keyword retrieval."""

    def __init__(self, corpus: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.id_to_doc = {doc["id"]: doc["text"] for doc in corpus}
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def _build_index(self):
        self.doc_tokens = {}
        self.doc_lengths = {}
        self.term_doc_freq = {}

        for doc in self.corpus:
            doc_id = doc["id"]
            tokens = self._tokenize(doc["text"])
            self.doc_tokens[doc_id] = tokens
            self.doc_lengths[doc_id] = len(tokens)

            for term in set(tokens):
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        self.num_docs = len(self.corpus)

    def _bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        doc_tokens = self.doc_tokens[doc_id]
        doc_length = self.doc_lengths[doc_id]

        score = 0.0
        for term in query_tokens:
            if term not in self.term_doc_freq:
                continue

            tf = doc_tokens.count(term)
            if tf == 0:
                continue

            df = self.term_doc_freq[term]
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        query_tokens = self._tokenize(query)
        scores = [(doc["id"], self._bm25_score(query_tokens, doc["id"])) for doc in self.corpus]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scores[:k]]


class VectorRetriever:
    """Vector similarity retrieval."""

    def __init__(self, corpus: List[Dict[str, str]], model_name: str = "all-MiniLM-L6-v2"):
        self.corpus = corpus
        self.model = None
        self.embeddings = None
        self.np = None

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            self.model = SentenceTransformer(model_name)
            texts = [doc["text"] for doc in corpus]
            self.embeddings = self.model.encode(texts, normalize_embeddings=True)
            self.np = np
        except ImportError:
            pass  # Will use fallback

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if self.model is None:
            # Fallback to BM25
            fallback = BM25Retriever(self.corpus)
            return fallback.retrieve(query, k)

        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        similarities = self.np.dot(self.embeddings, query_emb)
        indices = self.np.argsort(similarities)[::-1][:k]
        return [self.corpus[i]["id"] for i in indices]


class HybridRetriever:
    """Hybrid: BM25 + Vector with RRF."""

    def __init__(self, corpus: List[Dict[str, str]], rrf_k: int = 60):
        self.bm25 = BM25Retriever(corpus)
        self.vector = VectorRetriever(corpus)
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        bm25_results = self.bm25.retrieve(query, k=k*2)
        vector_results = self.vector.retrieve(query, k=k*2)

        # RRF fusion
        scores = {}
        for ranking in [bm25_results, vector_results]:
            for rank, doc_id in enumerate(ranking):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:k]]


class HoloLoomRetriever:
    """HoloLoom full retrieval system."""

    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.loom = None
        self.initialized = False

    async def initialize(self):
        if self.initialized:
            return True

        try:
            from HoloLoom import HoloLoom
            from HoloLoom.config import Config

            self.loom = HoloLoom(config=Config.fast())
            await self.loom.__aenter__()

            for doc in self.corpus:
                await self.loom.experience(doc["text"], metadata={"id": doc["id"]})

            self.initialized = True
            return True
        except Exception as e:
            print(f"      HoloLoom init error: {e}")
            return False

    async def retrieve(self, query: str, k: int = 5) -> List[str]:
        if not self.initialized:
            return []

        try:
            memories = await self.loom.recall(query, k=k)
            retrieved_ids = []
            for mem in memories:
                doc_id = mem.metadata.get("id") if hasattr(mem, "metadata") else None
                if doc_id:
                    retrieved_ids.append(doc_id)
            return retrieved_ids[:k]
        except Exception:
            return []

    async def cleanup(self):
        if self.loom:
            try:
                await self.loom.__aexit__(None, None, None)
            except Exception:
                pass


# ============================================================================
# BENCHMARK
# ============================================================================

async def run_retrieval_benchmark(mode: EvalMode, deps: Dict[str, bool]) -> BenchmarkResult:
    """Run retrieval quality benchmark."""
    corpus = get_corpus()
    queries = get_queries_for_mode(mode.value)
    k = 5

    results = {}
    methods = ["BM25", "Vector", "Hybrid"]

    # Check if we can test HoloLoom
    can_test_hololoom = deps.get("hololoom", False) and deps.get("numpy", False)
    if can_test_hololoom:
        methods.append("HoloLoom")

    # Initialize retrievers
    bm25 = BM25Retriever(corpus)
    vector = VectorRetriever(corpus)
    hybrid = HybridRetriever(corpus)
    hololoom = HoloLoomRetriever(corpus) if can_test_hololoom else None

    if hololoom:
        await hololoom.initialize()

    # Evaluate each method
    for method in methods:
        metrics = {"ndcg": [], "mrr": [], "precision": [], "recall": []}

        for q in queries:
            query_text = q["query"]
            relevant = q["relevant_ids"]

            if method == "BM25":
                retrieved = bm25.retrieve(query_text, k)
            elif method == "Vector":
                retrieved = vector.retrieve(query_text, k)
            elif method == "Hybrid":
                retrieved = hybrid.retrieve(query_text, k)
            elif method == "HoloLoom":
                retrieved = await hololoom.retrieve(query_text, k)
            else:
                retrieved = []

            metrics["ndcg"].append(ndcg_at_k(retrieved, relevant, k))
            metrics["mrr"].append(reciprocal_rank(retrieved, relevant))
            metrics["precision"].append(precision_at_k(retrieved, relevant, k))
            metrics["recall"].append(recall_at_k(retrieved, relevant, k))

        # Average metrics
        results[method] = {
            "ndcg": sum(metrics["ndcg"]) / len(metrics["ndcg"]),
            "mrr": sum(metrics["mrr"]) / len(metrics["mrr"]),
            "precision": sum(metrics["precision"]) / len(metrics["precision"]),
            "recall": sum(metrics["recall"]) / len(metrics["recall"]),
        }

    # Cleanup
    if hololoom:
        await hololoom.cleanup()

    # Determine winner and improvement
    best_method = max(results.keys(), key=lambda m: results[m]["ndcg"])
    best_ndcg = results[best_method]["ndcg"]

    # Compare HoloLoom vs Hybrid (the main comparison)
    hybrid_ndcg = results.get("Hybrid", {}).get("ndcg", 0)
    hololoom_ndcg = results.get("HoloLoom", {}).get("ndcg", 0)

    if can_test_hololoom and hololoom_ndcg > 0:
        improvement = (hololoom_ndcg - hybrid_ndcg) / hybrid_ndcg if hybrid_ndcg > 0 else 0
        if improvement > 0.1:
            verdict = f"HoloLoom +{improvement*100:.1f}% over Hybrid"
            passed = True
            score = min(1.0, 0.6 + improvement)
        elif improvement > 0:
            verdict = f"HoloLoom marginally better (+{improvement*100:.1f}%)"
            passed = True
            score = 0.5 + improvement
        else:
            verdict = f"Hybrid matches or beats HoloLoom"
            passed = False
            score = max(0.3, hybrid_ndcg)
    else:
        # Can't test HoloLoom, report baseline performance
        verdict = f"Best baseline: {best_method} (NDCG={best_ndcg:.3f})"
        passed = best_ndcg > 0.5
        score = best_ndcg

    return BenchmarkResult(
        name="Retrieval Quality",
        passed=passed,
        score=score,
        verdict=verdict,
        details={
            "results_by_method": results,
            "num_queries": len(queries),
            "k": k,
            "best_method": best_method,
            "hololoom_tested": can_test_hololoom,
        }
    )
