"""
HoloLoom Recall Accuracy Benchmark - Wikipedia Dataset
=======================================================

Tests how well HoloLoom retrieves relevant memories from Wikipedia articles.

Dataset:
- Source: Synthetic Wikipedia-style articles (for demo)
- Size: 100-10,000 articles
- Queries: Factual questions with known answers

Metrics:
- Precision@K (K=1,5,10)
- Recall@K (K=5,10)
- MRR (Mean Reciprocal Rank)

Usage:
    PYTHONPATH=. python benchmarks/recall_accuracy/wikipedia_benchmark.py

Output:
    - JSON results: benchmarks/results/recall_accuracy_wikipedia_YYYY-MM-DD.json
    - Markdown report: benchmarks/results/recall_accuracy_wikipedia_YYYY-MM-DD.md

Author: HoloLoom Team
Date: 2025-11-17
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.benchmark_base import (
    BaseBenchmark,
    BenchmarkMetrics,
    BenchmarkConfig,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_percentile
)


# ============================================================================
# Wikipedia Benchmark Dataset (Synthetic for Demo)
# ============================================================================

WIKIPEDIA_ARTICLES = [
    {
        'id': 'ml_001',
        'title': 'Machine Learning',
        'content': 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.'
    },
    {
        'id': 'ts_001',
        'title': 'Thompson Sampling',
        'content': 'Thompson Sampling is a heuristic for choosing actions in multi-armed bandit problems. It balances exploration and exploitation by sampling from the posterior distribution of each arm\'s expected reward. Named after William R. Thompson who proposed it in 1933.'
    },
    {
        'id': 'nn_001',
        'title': 'Neural Networks',
        'content': 'Artificial neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Deep learning uses multiple layers to progressively extract higher-level features from raw input.'
    },
    {
        'id': 'rl_001',
        'title': 'Reinforcement Learning',
        'content': 'Reinforcement learning is an area of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties and aims to maximize cumulative reward over time. PPO and SAC are popular RL algorithms.'
    },
    {
        'id': 'py_001',
        'title': 'Python Programming',
        'content': 'Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, it has become one of the most popular languages for data science, web development, and automation.'
    },
    {
        'id': 'ts_002',
        'title': 'TypeScript',
        'content': 'TypeScript is a strongly typed programming language that builds on JavaScript. Developed by Microsoft, it adds optional static typing to JavaScript, making it easier to write and maintain large-scale applications. TypeScript code compiles to plain JavaScript.'
    },
    {
        'id': 'kg_001',
        'title': 'Knowledge Graphs',
        'content': 'A knowledge graph is a structured representation of knowledge as a network of entities and their relationships. Each edge represents a typed relationship between entities. Knowledge graphs enable semantic search, inference, and reasoning over connected data.'
    },
    {
        'id': 'emb_001',
        'title': 'Embeddings',
        'content': 'Embeddings are dense vector representations of discrete objects like words, sentences, or images. They map high-dimensional sparse data into lower-dimensional continuous spaces where similar items are close together. Word2Vec and BERT produce popular text embeddings.'
    },
    {
        'id': 'att_001',
        'title': 'Attention Mechanism',
        'content': 'The attention mechanism allows neural networks to focus on relevant parts of the input when producing output. Transformers use self-attention to process sequences in parallel, enabling models like GPT and BERT to achieve state-of-the-art performance.'
    },
    {
        'id': 'mat_001',
        'title': 'Matryoshka Embeddings',
        'content': 'Matryoshka Representation Learning produces embeddings where the first k dimensions contain a valid k-dimensional representation. This nesting property enables flexible granularity: use small embeddings for efficiency or full embeddings for accuracy.'
    }
]

WIKIPEDIA_QUERIES = [
    {
        'query': 'What is Thompson Sampling?',
        'relevant': ['ts_001']
    },
    {
        'query': 'How do neural networks work?',
        'relevant': ['nn_001', 'att_001']
    },
    {
        'query': 'What programming languages are popular for ML?',
        'relevant': ['py_001', 'ml_001']
    },
    {
        'query': 'Explain reinforcement learning',
        'relevant': ['rl_001', 'ts_001']
    },
    {
        'query': 'What are knowledge graphs?',
        'relevant': ['kg_001']
    },
    {
        'query': 'How do embeddings work?',
        'relevant': ['emb_001', 'mat_001']
    },
    {
        'query': 'What is attention in transformers?',
        'relevant': ['att_001', 'nn_001']
    },
    {
        'query': 'Difference between Python and TypeScript',
        'relevant': ['py_001', 'ts_002']
    },
    {
        'query': 'Multi-armed bandit algorithms',
        'relevant': ['ts_001', 'rl_001']
    },
    {
        'query': 'What is Matryoshka representation learning?',
        'relevant': ['mat_001', 'emb_001']
    }
]


# ============================================================================
# Wikipedia Recall Accuracy Benchmark
# ============================================================================

class WikipediaRecallBenchmark(BaseBenchmark):
    """Recall accuracy benchmark on Wikipedia-style articles."""

    def __init__(self):
        super().__init__(
            name="recall_accuracy_wikipedia",
            version="1.0",
            seed=42
        )

    def load_dataset(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Load Wikipedia dataset.

        Returns:
            (articles, info) tuple
        """
        # In production, this would download from a URL or read from disk
        # For demo, we use the synthetic dataset above

        dataset_info = {
            'name': 'wikipedia_synthetic',
            'version': '1.0',
            'num_articles': len(WIKIPEDIA_ARTICLES),
            'num_queries': len(WIKIPEDIA_QUERIES),
            'source': 'synthetic'
        }

        return WIKIPEDIA_ARTICLES, dataset_info

    def run_benchmark(
        self,
        data: List[Dict],
        config: BenchmarkConfig
    ) -> BenchmarkMetrics:
        """
        Run recall accuracy benchmark.

        Simulates HoloLoom recall() without requiring full dependencies.
        """
        print(f"Simulating recall on {len(data)} articles...")
        print(f"Testing {len(WIKIPEDIA_QUERIES)} queries...\n")

        # Simulate storing memories
        memory_store = {}
        for article in data:
            memory_store[article['id']] = article['content']

        # Run queries
        all_retrieved = []
        all_relevant = []
        latencies = []

        for i, query_item in enumerate(WIKIPEDIA_QUERIES, 1):
            query = query_item['query']
            relevant = query_item['relevant']

            # Simulate retrieval (simple keyword matching for demo)
            start_time = time.time()
            retrieved = self._simulate_retrieval(query, data, k=10)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)

            all_retrieved.append(retrieved)
            all_relevant.append(relevant)

            # Progress
            if i % 2 == 0:
                print(f"  Progress: {i}/{len(WIKIPEDIA_QUERIES)} queries")

        print()

        # Calculate metrics
        metrics = BenchmarkMetrics(
            precision_at_1=self._avg_precision(all_retrieved, all_relevant, k=1),
            precision_at_5=self._avg_precision(all_retrieved, all_relevant, k=5),
            precision_at_10=self._avg_precision(all_retrieved, all_relevant, k=10),
            recall_at_5=self._avg_recall(all_retrieved, all_relevant, k=5),
            recall_at_10=self._avg_recall(all_retrieved, all_relevant, k=10),
            mrr=calculate_mrr(all_retrieved, all_relevant),
            latency_p50_ms=calculate_percentile(latencies, 50),
            latency_p95_ms=calculate_percentile(latencies, 95),
            latency_p99_ms=calculate_percentile(latencies, 99),
            throughput_qps=len(WIKIPEDIA_QUERIES) / sum(latencies) * 1000 if latencies else 0
        )

        return metrics

    def _simulate_retrieval(
        self,
        query: str,
        articles: List[Dict],
        k: int = 10
    ) -> List[str]:
        """
        Simulate HoloLoom retrieval using simple keyword matching.

        In production, this would use actual HoloLoom recall().
        """
        # Simple scoring: count matching words
        query_words = set(query.lower().split())

        scores = []
        for article in articles:
            content_words = set(article['content'].lower().split())
            score = len(query_words & content_words)
            scores.append((article['id'], score))

        # Sort by score and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [article_id for article_id, _ in scores[:k]]

    def _avg_precision(
        self,
        retrieved_lists: List[List[str]],
        relevant_lists: List[List[str]],
        k: int
    ) -> float:
        """Calculate average Precision@K across all queries."""
        if not retrieved_lists:
            return 0.0

        precisions = [
            calculate_precision_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(retrieved_lists, relevant_lists)
        ]

        return sum(precisions) / len(precisions)

    def _avg_recall(
        self,
        retrieved_lists: List[List[str]],
        relevant_lists: List[List[str]],
        k: int
    ) -> float:
        """Calculate average Recall@K across all queries."""
        if not retrieved_lists:
            return 0.0

        recalls = [
            calculate_recall_at_k(retrieved, relevant, k)
            for retrieved, relevant in zip(retrieved_lists, relevant_lists)
        ]

        return sum(recalls) / len(recalls)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run Wikipedia recall accuracy benchmark."""
    # Create benchmark
    benchmark = WikipediaRecallBenchmark()

    # Run benchmark
    result = benchmark.run(
        num_memories=10,  # Use all 10 articles
        num_queries=10,   # Use all 10 queries
        config_mode="fast"
    )

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = result.timestamp.split('T')[0]  # YYYY-MM-DD
    json_file = output_dir / f"recall_accuracy_wikipedia_{timestamp}.json"
    result.save(json_file)

    print(f"✓ Results saved to: {json_file}")

    # Create markdown report
    md_file = output_dir / f"recall_accuracy_wikipedia_{timestamp}.md"
    with open(md_file, 'w') as f:
        f.write(f"# Wikipedia Recall Accuracy Benchmark\n\n")
        f.write(f"**Date:** {result.timestamp}\n\n")
        f.write(f"**Dataset:** {result.config.dataset_name}\n")
        f.write(f"**Memories:** {result.config.num_memories}\n")
        f.write(f"**Queries:** {result.config.num_queries}\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Precision@1 | {result.metrics.precision_at_1:.3f} |\n")
        f.write(f"| Precision@5 | {result.metrics.precision_at_5:.3f} |\n")
        f.write(f"| Precision@10 | {result.metrics.precision_at_10:.3f} |\n")
        f.write(f"| Recall@5 | {result.metrics.recall_at_5:.3f} |\n")
        f.write(f"| Recall@10 | {result.metrics.recall_at_10:.3f} |\n")
        f.write(f"| MRR | {result.metrics.mrr:.3f} |\n")
        f.write(f"| Latency p95 | {result.metrics.latency_p95_ms:.1f}ms |\n")
        f.write(f"| Throughput | {result.metrics.throughput_qps:.1f} queries/sec |\n")

    print(f"✓ Report saved to: {md_file}\n")

    return 0 if not result.errors else 1


if __name__ == '__main__':
    sys.exit(main())
