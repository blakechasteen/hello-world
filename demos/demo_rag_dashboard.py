#!/usr/bin/env python3
"""
RAG Performance Dashboard Demo
==============================

Demonstrates automated dashboard construction from RAG query history.
Generates sample queries and builds a complete performance dashboard.

Example:
    python demo_rag_dashboard.py

Output:
    demos/output/rag_dashboard.html - Complete interactive dashboard
"""

import sys
import random
from pathlib import Path
from datetime import datetime

# Add repository to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.visualization.rag_dashboard import RAGDashboard, RAGResult


def generate_sample_queries(n_queries: int = 15) -> list:
    """
    Generate sample RAG query results for demo.

    Args:
        n_queries: Number of sample queries

    Returns:
        List of RAGResult objects with realistic data
    """
    sample_topics = [
        "Thompson Sampling",
        "Bayesian inference",
        "Reinforcement learning",
        "Multi-armed bandits",
        "Exploration-exploitation",
        "Neural networks",
        "Transformers",
        "Attention mechanisms",
        "BERT embeddings",
        "RAG systems"
    ]

    sample_sources = [
        "arxiv_2019_thompsons_sampling.pdf",
        "reinforcement_learning_textbook.pdf",
        "neural_networks_deep_learning.pdf",
        "attention_is_all_you_need.pdf",
        "bert_paper_2018.pdf",
        "bandit_algorithms_2020.pdf",
        "rag_survey_2023.pdf",
        "bayesian_methods_handbook.pdf",
        "deep_rl_tutorial_2021.pdf",
        "transformer_architecture_guide.pdf"
    ]

    queries = []
    confidence_trend = 0.85
    cache_hit_rate = 0.3

    for i in range(n_queries):
        # Generate realistic confidence with trends and anomalies
        if i % 5 == 0 and i > 0:
            confidence_trend *= random.uniform(0.95, 1.05)
        confidence = min(1.0, max(0.5, confidence_trend + random.gauss(0, 0.08)))

        # Occasional anomalies
        if random.random() < 0.1:
            confidence *= 0.6  # Sudden drop
        if random.random() < 0.1:
            confidence *= 1.1  # Sudden spike (but keep it <= 1.0)

        # Ensure confidence is in valid range
        confidence = min(1.0, max(0.0, confidence))

        # Cache hits more common in later queries
        cache_hit = random.random() < (cache_hit_rate + i * 0.02)

        # Latency depends on cache
        if cache_hit:
            latency_ms = random.uniform(20, 50)
        else:
            latency_ms = random.uniform(100, 300)

        # Number of sources decreases with later queries (cache helps)
        n_sources = max(2, int(6 - i * 0.2 + random.gauss(0, 1)))

        # Create RAGResult
        query = RAGResult(
            response=f"Response about {random.choice(sample_topics)} with comprehensive analysis...",
            sources=random.sample(sample_sources, min(n_sources, len(sample_sources))),
            confidence=confidence,
            reasoning_mode=random.choice(['direct', 'verify', 'research']),
            metadata={
                'latency_ms': latency_ms,
                'cache_hit': cache_hit,
                'timestamp': datetime.now().timestamp(),
                'tokens_used': random.randint(100, 500)
            }
        )
        queries.append(query)

    return queries


def main():
    """Generate demo dashboard and save to HTML."""
    print("RAG Performance Dashboard Demo")
    print("=" * 50)

    # Generate sample queries
    print("\n1. Generating sample RAG queries...")
    queries = generate_sample_queries(n_queries=15)
    print(f"   Created {len(queries)} sample queries")

    # Calculate some stats
    avg_confidence = sum(q.confidence for q in queries) / len(queries)
    avg_sources = sum(len(q.sources) for q in queries) / len(queries)
    cache_hits = sum(1 for q in queries if q.metadata.get('cache_hit', False))
    hit_rate = cache_hits / len(queries)

    print(f"   Avg confidence: {avg_confidence:.2f}")
    print(f"   Avg sources: {avg_sources:.1f}")
    print(f"   Cache hit rate: {hit_rate:.1%}")

    # Build dashboard
    print("\n2. Building dashboard from query history...")
    dashboard = RAGDashboard.from_query_history(
        queries,
        title="RAG Performance Analysis - Demo",
        detect_anomalies=True
    )
    print(f"   Dashboard created with {len(dashboard.dashboard.panels)} panels")

    # Render to HTML
    print("\n3. Rendering dashboard to HTML...")
    html = dashboard.render()
    print(f"   HTML generated ({len(html)} characters)")

    # Save to file
    output_path = Path("demos/output/rag_dashboard.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dashboard.save(str(output_path))
    print(f"   Saved to: {output_path}")

    # Print metrics summary
    print("\n4. Dashboard Metrics:")
    print(f"   {len(queries)} queries analyzed")
    print(f"   Avg confidence: {avg_confidence:.2f}")
    print(f"   Confidence range: [{min(q.confidence for q in queries):.2f}, {max(q.confidence for q in queries):.2f}]")
    print(f"   Avg sources retrieved: {avg_sources:.1f}")
    print(f"   Total unique sources: {len(set(s for q in queries for s in q.sources))}")
    print(f"   Cache hit rate: {hit_rate:.1%}")
    print(f"   Avg latency: {sum(q.metadata.get('latency_ms', 0) for q in queries) / len(queries):.1f}ms")

    # Print panel information
    print("\n5. Dashboard Panels:")
    for i, panel in enumerate(dashboard.dashboard.panels, 1):
        print(f"   {i}. {panel.title} ({panel.type.value})")
        if panel.subtitle:
            # Encode Unicode safely for Windows console
            subtitle = panel.subtitle.encode('utf-8', errors='replace').decode('utf-8')
            print(f"      {subtitle}")

    print("\n[OK] Demo complete! Open the HTML file in a browser to view the dashboard.")


if __name__ == "__main__":
    main()
