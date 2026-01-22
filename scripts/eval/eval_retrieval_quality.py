#!/usr/bin/env python3
"""
Retrieval Quality Evaluation: Does HoloLoom beat baselines?

This script compares HoloLoom's retrieval against simpler approaches to
determine if the complexity provides real value.

Baselines:
1. Vector-only (sentence-transformers + cosine similarity)
2. BM25-only (keyword matching)
3. Simple hybrid (vector + BM25, no graph)
4. HoloLoom full (hybrid + graph + multipass)

Metrics:
- Precision@k: % of top-k results that are relevant
- Recall@k: % of relevant docs found in top-k
- MRR: Mean Reciprocal Rank (position of first relevant result)
- NDCG@k: Normalized Discounted Cumulative Gain

Usage:
    python scripts/eval/eval_retrieval_quality.py
    python scripts/eval/eval_retrieval_quality.py --dataset custom --k 5
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import math

# ============================================================================
# TEST DATASET
# ============================================================================
#
# A small, manually-verifiable dataset. Each query has:
# - query: the search query
# - relevant_ids: IDs of documents that SHOULD be retrieved
# - corpus: the documents to search over
#
# You can replace this with your own domain-specific data.

EVALUATION_CORPUS = [
    {"id": "doc_1", "text": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It balances exploration and exploitation by sampling from posterior distributions."},
    {"id": "doc_2", "text": "The multi-armed bandit problem is a classic reinforcement learning problem where an agent must choose between multiple options with uncertain rewards."},
    {"id": "doc_3", "text": "UCB (Upper Confidence Bound) is another algorithm for the multi-armed bandit problem that uses confidence intervals to balance exploration."},
    {"id": "doc_4", "text": "Bayesian inference updates prior beliefs with observed data to form posterior distributions. It is fundamental to probabilistic machine learning."},
    {"id": "doc_5", "text": "Epsilon-greedy is a simple exploration strategy that explores randomly with probability epsilon and exploits the best known option otherwise."},
    {"id": "doc_6", "text": "Neural networks are computational models inspired by biological neurons. They learn representations through backpropagation."},
    {"id": "doc_7", "text": "Transformers use self-attention mechanisms to process sequences. They are the foundation of modern language models like GPT and BERT."},
    {"id": "doc_8", "text": "Reinforcement learning trains agents to make decisions by maximizing cumulative rewards through trial and error."},
    {"id": "doc_9", "text": "Knowledge graphs represent entities and relationships as nodes and edges. They enable structured reasoning over facts."},
    {"id": "doc_10", "text": "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models to ground responses in external knowledge."},
    {"id": "doc_11", "text": "Matryoshka embeddings encode information at multiple scales, allowing flexible precision vs computation tradeoffs."},
    {"id": "doc_12", "text": "BM25 is a probabilistic ranking function used in information retrieval based on term frequency and document length."},
    {"id": "doc_13", "text": "Cosine similarity measures the angle between two vectors, commonly used to compare text embeddings."},
    {"id": "doc_14", "text": "The exploration-exploitation tradeoff is fundamental to decision making under uncertainty. Too much exploration wastes resources, too little misses better options."},
    {"id": "doc_15", "text": "Posterior distributions in Bayesian statistics represent updated beliefs after observing data, combining prior knowledge with evidence."},
]

EVALUATION_QUERIES = [
    {
        "query": "What is Thompson Sampling?",
        "relevant_ids": ["doc_1", "doc_2", "doc_4", "doc_15"],  # Thompson, bandits, Bayesian, posterior
        "difficulty": "easy",
    },
    {
        "query": "How do bandit algorithms balance exploration and exploitation?",
        "relevant_ids": ["doc_1", "doc_3", "doc_5", "doc_14"],  # Thompson, UCB, epsilon-greedy, tradeoff
        "difficulty": "medium",
    },
    {
        "query": "Explain Bayesian approaches to decision making",
        "relevant_ids": ["doc_1", "doc_4", "doc_15"],  # Thompson, Bayesian inference, posterior
        "difficulty": "medium",
    },
    {
        "query": "What retrieval methods are used in RAG systems?",
        "relevant_ids": ["doc_10", "doc_12", "doc_13"],  # RAG, BM25, cosine similarity
        "difficulty": "medium",
    },
    {
        "query": "How do neural language models work?",
        "relevant_ids": ["doc_6", "doc_7"],  # Neural networks, transformers
        "difficulty": "easy",
    },
    {
        "query": "Compare different exploration strategies",
        "relevant_ids": ["doc_1", "doc_3", "doc_5", "doc_14"],  # Thompson, UCB, epsilon-greedy, tradeoff
        "difficulty": "hard",
    },
    {
        "query": "What is the role of embeddings in search?",
        "relevant_ids": ["doc_11", "doc_13"],  # Matryoshka, cosine similarity
        "difficulty": "medium",
    },
    {
        "query": "How do knowledge graphs help reasoning?",
        "relevant_ids": ["doc_9"],  # Knowledge graphs
        "difficulty": "easy",
    },
]


# ============================================================================
# METRICS
# ============================================================================

@dataclass
class RetrievalResult:
    """Result from a single retrieval."""
    retrieved_ids: List[str]
    latency_ms: float
    method: str


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    latency_ms: float = 0.0
    hit: bool = False  # Did we find at least one relevant doc?


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all queries."""
    method: str
    num_queries: int
    avg_precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    avg_ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


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
    """MRR: 1/position of first relevant result (0 if none found)."""
    relevant_set = set(relevant)
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def dcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    relevant_set = set(relevant)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant_set:
            # Binary relevance: 1 if relevant, 0 otherwise
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Normalized DCG at k."""
    dcg = dcg_at_k(retrieved, relevant, k)
    # Ideal DCG: all relevant docs at top
    ideal_retrieved = relevant[:k] + [f"dummy_{i}" for i in range(k - len(relevant))]
    idcg = dcg_at_k(ideal_retrieved, relevant, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_query_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    latency_ms: float,
    ks: List[int] = [1, 3, 5, 10]
) -> QueryMetrics:
    """Compute all metrics for a single query."""
    metrics = QueryMetrics(
        query="",
        latency_ms=latency_ms,
        hit=any(doc_id in set(relevant_ids) for doc_id in retrieved_ids)
    )

    for k in ks:
        metrics.precision_at_k[k] = precision_at_k(retrieved_ids, relevant_ids, k)
        metrics.recall_at_k[k] = recall_at_k(retrieved_ids, relevant_ids, k)
        metrics.ndcg_at_k[k] = ndcg_at_k(retrieved_ids, relevant_ids, k)

    metrics.mrr = reciprocal_rank(retrieved_ids, relevant_ids)
    return metrics


def aggregate_metrics(
    query_metrics: List[QueryMetrics],
    method: str,
    ks: List[int] = [1, 3, 5, 10]
) -> AggregateMetrics:
    """Aggregate metrics across all queries."""
    n = len(query_metrics)
    if n == 0:
        return AggregateMetrics(method=method, num_queries=0)

    agg = AggregateMetrics(
        method=method,
        num_queries=n,
        mrr=sum(m.mrr for m in query_metrics) / n,
        hit_rate=sum(1 for m in query_metrics if m.hit) / n,
        avg_latency_ms=sum(m.latency_ms for m in query_metrics) / n,
    )

    latencies = sorted([m.latency_ms for m in query_metrics])
    agg.p95_latency_ms = latencies[int(0.95 * n)] if n > 0 else 0.0

    for k in ks:
        agg.avg_precision_at_k[k] = sum(m.precision_at_k.get(k, 0) for m in query_metrics) / n
        agg.avg_recall_at_k[k] = sum(m.recall_at_k.get(k, 0) for m in query_metrics) / n
        agg.avg_ndcg_at_k[k] = sum(m.ndcg_at_k.get(k, 0) for m in query_metrics) / n

    return agg


# ============================================================================
# BASELINE RETRIEVERS
# ============================================================================

class BaseRetriever:
    """Base class for retrieval methods."""

    def __init__(self, corpus: List[Dict[str, str]]):
        self.corpus = corpus
        self.id_to_doc = {doc["id"]: doc["text"] for doc in corpus}

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    """Simple BM25 keyword retrieval."""

    def __init__(self, corpus: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        super().__init__(corpus)
        self.k1 = k1
        self.b = b
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()

    def _build_index(self):
        """Build inverted index and compute IDF."""
        self.doc_tokens = {}
        self.doc_lengths = {}
        self.term_doc_freq = {}  # term -> number of docs containing term

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
        """Compute BM25 score for a document."""
        doc_tokens = self.doc_tokens[doc_id]
        doc_length = self.doc_lengths[doc_id]

        score = 0.0
        for term in query_tokens:
            if term not in self.term_doc_freq:
                continue

            # Term frequency in document
            tf = doc_tokens.count(term)
            if tf == 0:
                continue

            # Inverse document frequency
            df = self.term_doc_freq[term]
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator

        return score

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        start = time.perf_counter()

        query_tokens = self._tokenize(query)
        scores = []
        for doc in self.corpus:
            score = self._bm25_score(query_tokens, doc["id"])
            scores.append((doc["id"], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        retrieved_ids = [doc_id for doc_id, _ in scores[:k]]

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(retrieved_ids=retrieved_ids, latency_ms=latency_ms, method="bm25")


class VectorRetriever(BaseRetriever):
    """Simple vector similarity retrieval using sentence-transformers."""

    def __init__(self, corpus: List[Dict[str, str]], model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(corpus)
        self.model_name = model_name
        self.embeddings = None
        self.model = None
        self._build_index()

    def _build_index(self):
        """Embed all documents."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            self.model = SentenceTransformer(self.model_name)
            texts = [doc["text"] for doc in self.corpus]
            self.embeddings = self.model.encode(texts, normalize_embeddings=True)
            self.np = np
        except ImportError:
            print("WARNING: sentence-transformers not installed. Vector retrieval will use fallback.")
            self.model = None
            self.embeddings = None

    def _cosine_similarity(self, query_embedding, doc_embeddings):
        """Compute cosine similarity (embeddings assumed normalized)."""
        return self.np.dot(doc_embeddings, query_embedding)

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        start = time.perf_counter()

        if self.model is None:
            # Fallback: use BM25
            fallback = BM25Retriever(self.corpus)
            result = fallback.retrieve(query, k)
            result.method = "vector_fallback_bm25"
            return result

        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        similarities = self._cosine_similarity(query_embedding, self.embeddings)

        indices = self.np.argsort(similarities)[::-1][:k]
        retrieved_ids = [self.corpus[i]["id"] for i in indices]

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(retrieved_ids=retrieved_ids, latency_ms=latency_ms, method="vector")


class HybridRetriever(BaseRetriever):
    """Simple hybrid: BM25 + Vector with Reciprocal Rank Fusion."""

    def __init__(self, corpus: List[Dict[str, str]], rrf_k: int = 60):
        super().__init__(corpus)
        self.bm25 = BM25Retriever(corpus)
        self.vector = VectorRetriever(corpus)
        self.rrf_k = rrf_k

    def _reciprocal_rank_fusion(self, rankings: List[List[str]], k: int) -> List[str]:
        """Combine rankings using RRF."""
        scores = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += 1.0 / (self.rrf_k + rank + 1)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs[:k]]

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        start = time.perf_counter()

        # Get rankings from both methods
        bm25_result = self.bm25.retrieve(query, k=k*2)  # Get more for fusion
        vector_result = self.vector.retrieve(query, k=k*2)

        # Fuse rankings
        fused = self._reciprocal_rank_fusion(
            [bm25_result.retrieved_ids, vector_result.retrieved_ids],
            k
        )

        latency_ms = (time.perf_counter() - start) * 1000
        return RetrievalResult(retrieved_ids=fused, latency_ms=latency_ms, method="hybrid")


# ============================================================================
# HOLOLOOM RETRIEVER
# ============================================================================

class HoloLoomRetriever(BaseRetriever):
    """HoloLoom's full retrieval system."""

    def __init__(self, corpus: List[Dict[str, str]]):
        super().__init__(corpus)
        self.loom = None
        self.initialized = False

    async def _initialize(self):
        """Initialize HoloLoom with corpus."""
        if self.initialized:
            return

        try:
            from HoloLoom import HoloLoom
            from HoloLoom.config import Config

            self.loom = HoloLoom(config=Config.fast())
            await self.loom.__aenter__()

            # Ingest corpus
            for doc in self.corpus:
                await self.loom.experience(doc["text"], metadata={"id": doc["id"]})

            self.initialized = True
        except Exception as e:
            print(f"WARNING: Could not initialize HoloLoom: {e}")
            self.loom = None

    async def retrieve_async(self, query: str, k: int = 5) -> RetrievalResult:
        start = time.perf_counter()

        if not self.initialized:
            await self._initialize()

        if self.loom is None:
            # Fallback to hybrid
            fallback = HybridRetriever(self.corpus)
            result = fallback.retrieve(query, k)
            result.method = "hololoom_fallback_hybrid"
            return result

        try:
            memories = await self.loom.recall(query, k=k)

            # Map memories back to doc IDs
            retrieved_ids = []
            for mem in memories:
                # Try to get doc ID from metadata
                doc_id = mem.metadata.get("id") if hasattr(mem, "metadata") else None
                if doc_id:
                    retrieved_ids.append(doc_id)
                else:
                    # Fallback: match by text content
                    for doc in self.corpus:
                        if doc["text"] in str(mem) or str(mem) in doc["text"]:
                            retrieved_ids.append(doc["id"])
                            break

            latency_ms = (time.perf_counter() - start) * 1000
            return RetrievalResult(retrieved_ids=retrieved_ids[:k], latency_ms=latency_ms, method="hololoom")

        except Exception as e:
            print(f"HoloLoom retrieval error: {e}")
            fallback = HybridRetriever(self.corpus)
            result = fallback.retrieve(query, k)
            result.method = "hololoom_error_fallback"
            return result

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Sync wrapper for async retrieve."""
        return asyncio.run(self.retrieve_async(query, k))

    async def cleanup(self):
        """Cleanup HoloLoom resources."""
        if self.loom:
            await self.loom.__aexit__(None, None, None)


# ============================================================================
# EVALUATION RUNNER
# ============================================================================

async def evaluate_retriever(
    retriever: BaseRetriever,
    queries: List[Dict],
    k: int = 5,
    method_name: str = ""
) -> AggregateMetrics:
    """Evaluate a retriever on all queries."""
    query_metrics = []

    for q in queries:
        query_text = q["query"]
        relevant_ids = q["relevant_ids"]

        # Handle async vs sync retrievers
        if hasattr(retriever, "retrieve_async"):
            result = await retriever.retrieve_async(query_text, k=k)
        else:
            result = retriever.retrieve(query_text, k=k)

        metrics = compute_query_metrics(
            retrieved_ids=result.retrieved_ids,
            relevant_ids=relevant_ids,
            latency_ms=result.latency_ms
        )
        metrics.query = query_text
        query_metrics.append(metrics)

    return aggregate_metrics(query_metrics, method_name or retriever.__class__.__name__)


def print_comparison_table(results: List[AggregateMetrics], k: int = 5):
    """Print a comparison table of all methods."""
    print("\n" + "=" * 80)
    print(f"RETRIEVAL QUALITY COMPARISON (k={k})")
    print("=" * 80)

    # Header
    print(f"\n{'Method':<25} {'P@{k}':<10} {'R@{k}':<10} {'NDCG@{k}':<10} {'MRR':<10} {'Hit Rate':<10} {'Latency':<10}")
    print(f"{'Method':<25} {'P@'+str(k):<10} {'R@'+str(k):<10} {'NDCG@'+str(k):<10} {'MRR':<10} {'Hit Rate':<10} {'(ms)':<10}")
    print("-" * 85)

    # Sort by NDCG@k descending
    sorted_results = sorted(results, key=lambda x: x.avg_ndcg_at_k.get(k, 0), reverse=True)

    for r in sorted_results:
        print(f"{r.method:<25} "
              f"{r.avg_precision_at_k.get(k, 0):<10.3f} "
              f"{r.avg_recall_at_k.get(k, 0):<10.3f} "
              f"{r.avg_ndcg_at_k.get(k, 0):<10.3f} "
              f"{r.mrr:<10.3f} "
              f"{r.hit_rate:<10.3f} "
              f"{r.avg_latency_ms:<10.1f}")

    print("-" * 85)

    # Winner analysis
    print("\n📊 ANALYSIS:")
    best = sorted_results[0]
    print(f"   Best method: {best.method}")
    print(f"   NDCG@{k}: {best.avg_ndcg_at_k.get(k, 0):.3f}")

    if len(sorted_results) > 1:
        second = sorted_results[1]
        delta = best.avg_ndcg_at_k.get(k, 0) - second.avg_ndcg_at_k.get(k, 0)
        print(f"   Improvement over {second.method}: +{delta:.3f} ({delta/second.avg_ndcg_at_k.get(k, 0.001)*100:.1f}%)")


async def main():
    """Run the full evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--output", type=str, default="scripts/eval/results/retrieval_quality.json")
    args = parser.parse_args()

    print("=" * 80)
    print("HOLOLOOM RETRIEVAL QUALITY EVALUATION")
    print("=" * 80)
    print(f"\nCorpus size: {len(EVALUATION_CORPUS)} documents")
    print(f"Test queries: {len(EVALUATION_QUERIES)}")
    print(f"Retrieving top-{args.k} results per query")

    corpus = EVALUATION_CORPUS
    queries = EVALUATION_QUERIES

    # Initialize retrievers
    print("\n📦 Initializing retrievers...")

    bm25 = BM25Retriever(corpus)
    print("   ✓ BM25 ready")

    vector = VectorRetriever(corpus)
    print(f"   ✓ Vector ready (model: {vector.model_name if vector.model else 'fallback'})")

    hybrid = HybridRetriever(corpus)
    print("   ✓ Hybrid (BM25+Vector) ready")

    hololoom = HoloLoomRetriever(corpus)
    print("   ✓ HoloLoom ready (will initialize on first query)")

    # Run evaluations
    print("\n🔍 Running evaluations...")

    results = []

    print("   Evaluating BM25...")
    results.append(await evaluate_retriever(bm25, queries, k=args.k, method_name="BM25 (keyword)"))

    print("   Evaluating Vector...")
    results.append(await evaluate_retriever(vector, queries, k=args.k, method_name="Vector (semantic)"))

    print("   Evaluating Hybrid...")
    results.append(await evaluate_retriever(hybrid, queries, k=args.k, method_name="Hybrid (BM25+Vector)"))

    print("   Evaluating HoloLoom...")
    results.append(await evaluate_retriever(hololoom, queries, k=args.k, method_name="HoloLoom (full)"))

    # Cleanup
    await hololoom.cleanup()

    # Print results
    print_comparison_table(results, k=args.k)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_dict = {
        "metadata": {
            "corpus_size": len(corpus),
            "num_queries": len(queries),
            "k": args.k,
        },
        "results": [
            {
                "method": r.method,
                "precision_at_k": r.avg_precision_at_k,
                "recall_at_k": r.avg_recall_at_k,
                "ndcg_at_k": r.avg_ndcg_at_k,
                "mrr": r.mrr,
                "hit_rate": r.hit_rate,
                "avg_latency_ms": r.avg_latency_ms,
                "p95_latency_ms": r.p95_latency_ms,
            }
            for r in results
        ]
    }

    with open(output_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    hololoom_result = next((r for r in results if "HoloLoom" in r.method), None)
    hybrid_result = next((r for r in results if "Hybrid" in r.method), None)

    if hololoom_result and hybrid_result:
        holo_ndcg = hololoom_result.avg_ndcg_at_k.get(args.k, 0)
        hybrid_ndcg = hybrid_result.avg_ndcg_at_k.get(args.k, 0)

        if holo_ndcg > hybrid_ndcg * 1.1:  # >10% improvement
            print(f"✅ HoloLoom provides meaningful improvement (+{(holo_ndcg/hybrid_ndcg - 1)*100:.1f}%)")
            print("   The complexity appears justified for retrieval quality.")
        elif holo_ndcg > hybrid_ndcg:
            print(f"🟡 HoloLoom provides marginal improvement (+{(holo_ndcg/hybrid_ndcg - 1)*100:.1f}%)")
            print("   Consider if the complexity is worth the small gain.")
        else:
            print(f"❌ HoloLoom does NOT outperform simple hybrid retrieval")
            print("   The additional complexity may not be justified.")

        # Latency comparison
        holo_lat = hololoom_result.avg_latency_ms
        hybrid_lat = hybrid_result.avg_latency_ms
        print(f"\n   Latency: HoloLoom {holo_lat:.1f}ms vs Hybrid {hybrid_lat:.1f}ms")
        if holo_lat > hybrid_lat * 2:
            print(f"   ⚠️  HoloLoom is {holo_lat/hybrid_lat:.1f}x slower")


if __name__ == "__main__":
    asyncio.run(main())
