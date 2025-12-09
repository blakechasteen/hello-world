"""
Phase 5: Information-Theoretic Context Packing Demo
====================================================

Demonstrates the information-theoretic improvements from Phase 5:
1. 7-signal importance scoring (INFORMATION_CONTENT at 25% weight)
2. Information budget compression (Tishby's Information Bottleneck)
3. MI-aware Matryoshka scale assignment (384D/256D/128D)
4. Cache effectiveness for <5ms overhead

Author: Claude Code
Date: December 2025
"""

import time
import sys
from typing import Dict, List, Any
from collections import Counter

# Add parent directory for imports
sys.path.insert(0, '.')

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available, using mock graph")


def create_sample_knowledge_graph():
    """Create a sample knowledge graph with varied content."""
    if not NETWORKX_AVAILABLE:
        return None

    G = nx.MultiDiGraph()

    # Add nodes with content about Thompson Sampling and related topics
    nodes = {
        "thompson_sampling": {
            "content": "Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff. "
                      "It maintains probability distributions over reward parameters and samples from them.",
            "timestamp": time.time() - 3600,  # 1 hour ago
            "access_count": 10,
            "confidence": 0.9
        },
        "bayesian_methods": {
            "content": "Bayesian methods use prior probability distributions updated with evidence to form posteriors. "
                      "Beta distributions are commonly used for binary outcomes.",
            "timestamp": time.time() - 7200,
            "access_count": 8,
            "confidence": 0.85
        },
        "multi_armed_bandit": {
            "content": "The multi-armed bandit problem involves choosing between multiple options with uncertain rewards. "
                      "Classic exploration strategies include UCB, epsilon-greedy, and Thompson Sampling.",
            "timestamp": time.time() - 1800,
            "access_count": 15,
            "confidence": 0.92
        },
        "exploration_exploitation": {
            "content": "Exploration discovers new information while exploitation uses known good options. "
                      "Balancing these is crucial for optimal decision-making under uncertainty.",
            "timestamp": time.time() - 900,
            "access_count": 12,
            "confidence": 0.88
        },
        "beta_distribution": {
            "content": "Beta distribution Beta(alpha, beta) is conjugate to binomial likelihood. "
                      "Used in Thompson Sampling to model binary outcome probabilities.",
            "timestamp": time.time() - 5400,
            "access_count": 5,
            "confidence": 0.95
        },
        "ucb_algorithm": {
            "content": "Upper Confidence Bound algorithms add exploration bonus based on uncertainty. "
                      "UCB1 uses sqrt(2*ln(n)/n_a) as confidence width.",
            "timestamp": time.time() - 10800,
            "access_count": 3,
            "confidence": 0.82
        },
        "regret_bounds": {
            "content": "Regret measures cumulative difference from optimal strategy. "
                      "Thompson Sampling achieves logarithmic regret bounds asymptotically.",
            "timestamp": time.time() - 14400,
            "access_count": 2,
            "confidence": 0.78
        },
        "contextual_bandits": {
            "content": "Contextual bandits incorporate context features into arm selection. "
                      "LinUCB and contextual Thompson Sampling are popular approaches.",
            "timestamp": time.time() - 18000,
            "access_count": 4,
            "confidence": 0.80
        },
        "neural_networks": {
            "content": "Neural networks are computational models inspired by biological neurons. "
                      "Deep learning uses multiple layers for hierarchical feature extraction.",
            "timestamp": time.time() - 86400,  # 1 day ago (less relevant)
            "access_count": 20,
            "confidence": 0.95
        },
        "reinforcement_learning": {
            "content": "Reinforcement learning trains agents through reward signals. "
                      "Bandit problems are simplified RL with single-state environments.",
            "timestamp": time.time() - 3000,
            "access_count": 7,
            "confidence": 0.85
        }
    }

    # Add nodes to graph
    for node_id, data in nodes.items():
        G.add_node(node_id, **data)

    # Add edges (relationships)
    edges = [
        ("thompson_sampling", "bayesian_methods", "USES"),
        ("thompson_sampling", "multi_armed_bandit", "SOLVES"),
        ("thompson_sampling", "exploration_exploitation", "ADDRESSES"),
        ("thompson_sampling", "beta_distribution", "USES"),
        ("multi_armed_bandit", "exploration_exploitation", "INVOLVES"),
        ("multi_armed_bandit", "ucb_algorithm", "HAS_METHOD"),
        ("multi_armed_bandit", "regret_bounds", "HAS_METRIC"),
        ("contextual_bandits", "multi_armed_bandit", "EXTENDS"),
        ("reinforcement_learning", "multi_armed_bandit", "INCLUDES"),
        ("bayesian_methods", "beta_distribution", "USES"),
    ]

    for src, dst, rel in edges:
        G.add_edge(src, dst, relation=rel, weight=1.0)

    return G


def demo_standard_vs_information_packing():
    """Compare standard packing vs information-theoretic packing."""
    print("=" * 70)
    print("DEMO 1: Standard vs Information-Theoretic Packing")
    print("=" * 70)

    from HoloLoom.context_packing import (
        ContextPacker,
        ContextPackerConfig,
        pack_context,
        information_budget_pack
    )

    # Create sample data
    graph = create_sample_knowledge_graph()
    if graph is None:
        print("Skipping demo - networkx not available")
        return

    candidate_nodes = list(graph.nodes())
    node_contents = {n: graph.nodes[n].get("content", "") for n in candidate_nodes}
    query = "What is Thompson Sampling and how does it balance exploration?"

    print(f"\nQuery: {query}")
    print(f"Candidate nodes: {len(candidate_nodes)}")

    # Standard packing (6 signals)
    print("\n--- Standard Packing (6 Signals) ---")
    config_standard = ContextPackerConfig.balanced()
    config_standard.importance_scorer.enable_information_scoring = False

    packer_standard = ContextPacker(config_standard)

    start = time.perf_counter()
    result_standard = packer_standard.pack(
        query=query,
        candidate_nodes=candidate_nodes,
        graph=graph,
        target_tokens=500
    )
    standard_time = (time.perf_counter() - start) * 1000

    print(f"  Kept nodes: {result_standard.compressed_count}/{result_standard.original_count}")
    print(f"  Compression ratio: {result_standard.compression_ratio:.1%}")
    print(f"  Token savings: {result_standard.token_savings}")
    print(f"  Time: {standard_time:.2f}ms")
    print(f"  Nodes: {result_standard.compressed_nodes}")

    # Information-theoretic packing (7 signals)
    print("\n--- Information Budget Packing (7 Signals) ---")
    config_info = ContextPackerConfig.balanced()
    config_info.importance_scorer.enable_information_scoring = True
    config_info.compression.enable_information_budget = True

    packer_info = ContextPacker(config_info)

    start = time.perf_counter()
    result_info, scales, mi_scores = packer_info.pack_with_information_budget(
        query=query,
        candidate_nodes=candidate_nodes,
        graph=graph,
        node_contents=node_contents,
        information_budget=5.0
    )
    info_time = (time.perf_counter() - start) * 1000

    print(f"  Kept nodes: {result_info.compressed_count}/{result_info.original_count}")
    print(f"  Compression ratio: {result_info.compression_ratio:.1%}")
    print(f"  Token savings: {result_info.token_savings}")
    print(f"  Time: {info_time:.2f}ms")
    print(f"  Nodes: {result_info.compressed_nodes}")

    # Show MI scores
    print("\n  MI Scores (sorted by relevance to query):")
    sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    for node, mi in sorted_mi[:5]:
        scale = scales.get(node, "dropped")
        print(f"    {node}: MI={mi:.3f}, Scale={scale}")

    # Scale distribution
    print("\n  Matryoshka Scale Distribution:")
    scale_counts = Counter(scales.values())
    for scale in [384, 256, 128]:
        count = scale_counts.get(scale, 0)
        print(f"    {scale}D: {count} nodes")

    return mi_scores, scales


def demo_budget_comparison():
    """Compare different information budgets."""
    print("\n" + "=" * 70)
    print("DEMO 2: Information Budget Comparison")
    print("=" * 70)

    from HoloLoom.context_packing import ContextPacker, ContextPackerConfig

    graph = create_sample_knowledge_graph()
    if graph is None:
        print("Skipping demo - networkx not available")
        return

    candidate_nodes = list(graph.nodes())
    node_contents = {n: graph.nodes[n].get("content", "") for n in candidate_nodes}
    query = "Explain Thompson Sampling for multi-armed bandits"

    budgets = [2.0, 5.0, 10.0, 20.0]

    print(f"\nQuery: {query}")
    print(f"Testing budgets: {budgets}")
    print()

    config = ContextPackerConfig.balanced()
    config.importance_scorer.enable_information_scoring = True
    config.compression.enable_information_budget = True
    packer = ContextPacker(config)

    print(f"{'Budget':>8} | {'Nodes':>6} | {'Ratio':>8} | {'Tokens':>8} | {'MI Used':>8}")
    print("-" * 50)

    for budget in budgets:
        result, scales, mi_scores = packer.pack_with_information_budget(
            query=query,
            candidate_nodes=candidate_nodes,
            graph=graph,
            node_contents=node_contents,
            information_budget=budget
        )

        mi_used = sum(mi_scores.get(n, 0) for n in result.compressed_nodes)

        print(f"{budget:>8.1f} | {result.compressed_count:>6} | "
              f"{result.compression_ratio:>7.1%} | {result.token_savings:>8} | {mi_used:>8.3f}")


def demo_cache_effectiveness():
    """Demonstrate MI cache effectiveness."""
    print("\n" + "=" * 70)
    print("DEMO 3: Cache Effectiveness")
    print("=" * 70)

    from HoloLoom.context_packing import ImportanceScorer, ImportanceScorerConfig

    config = ImportanceScorerConfig()
    config.enable_information_scoring = True
    scorer = ImportanceScorer(config)

    # Sample content
    node_contents = {
        f"node_{i}": f"Content about topic {i} with some Thompson Sampling concepts"
        for i in range(50)
    }

    query = "Thompson Sampling exploration"

    # Cold cache
    scorer.invalidate_mi_cache()
    start = time.perf_counter()
    mi_scores_cold = scorer.batch_compute_mi_scores(
        node_ids=list(node_contents.keys()),
        query=query,
        node_contents=node_contents
    )
    cold_time = (time.perf_counter() - start) * 1000

    stats_after_cold = scorer.get_stats()

    # Warm cache (same query)
    start = time.perf_counter()
    mi_scores_warm = scorer.batch_compute_mi_scores(
        node_ids=list(node_contents.keys()),
        query=query,
        node_contents=node_contents
    )
    warm_time = (time.perf_counter() - start) * 1000

    stats_after_warm = scorer.get_stats()

    print(f"\nNodes: {len(node_contents)}")
    print(f"\nCold Cache:")
    print(f"  Time: {cold_time:.2f}ms")
    print(f"  Cache misses: {stats_after_cold['mi_cache_misses']}")

    print(f"\nWarm Cache (same query):")
    print(f"  Time: {warm_time:.2f}ms")
    print(f"  Cache hits: {stats_after_warm['mi_cache_hits'] - stats_after_cold['mi_cache_hits']}")

    if cold_time > 0:
        speedup = cold_time / warm_time if warm_time > 0 else float('inf')
        print(f"\nSpeedup: {speedup:.1f}x")

    # Verify results are identical
    assert mi_scores_cold == mi_scores_warm, "Cache should return identical results"
    print("\nCache integrity verified: Results identical")


def demo_mi_score_distribution():
    """Visualize MI score distribution."""
    print("\n" + "=" * 70)
    print("DEMO 4: MI Score Distribution")
    print("=" * 70)

    from HoloLoom.context_packing import ImportanceScorer, ImportanceScorerConfig

    config = ImportanceScorerConfig()
    config.enable_information_scoring = True
    scorer = ImportanceScorer(config)

    # Create varied content with different relevance levels
    node_contents = {
        # High relevance (should have high MI)
        "ts_core": "Thompson Sampling samples from posterior distributions to balance exploration",
        "ts_beta": "Beta distributions model binary outcomes in Thompson Sampling",
        "ts_exploit": "Thompson Sampling exploits high-confidence arms while exploring uncertain ones",

        # Medium relevance
        "bandit_general": "Multi-armed bandits involve sequential decision making under uncertainty",
        "bayesian_intro": "Bayesian inference updates beliefs based on evidence",

        # Low relevance
        "ml_intro": "Machine learning algorithms learn patterns from data",
        "neural_net": "Neural networks use layers of interconnected nodes",
        "deep_learning": "Deep learning uses multiple hidden layers for complex patterns",

        # Very low relevance
        "cooking": "Italian pasta recipes use various types of noodles",
        "sports": "Basketball requires teamwork and individual skill",
    }

    query = "How does Thompson Sampling balance exploration and exploitation?"

    mi_scores = scorer.batch_compute_mi_scores(
        node_ids=list(node_contents.keys()),
        query=query,
        node_contents=node_contents
    )

    print(f"\nQuery: {query}\n")

    # Sort by MI score
    sorted_scores = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)

    # Create simple histogram
    print("MI Score Distribution:")
    print()

    for node, mi in sorted_scores:
        bar_len = int(mi * 40)
        bar = "#" * bar_len

        # Determine scale assignment
        if mi >= 0.7:
            scale = "384D"
        elif mi >= 0.4:
            scale = "256D"
        elif mi >= 0.2:
            scale = "128D"
        else:
            scale = "DROP"

        print(f"  {node:<15} |{bar:<40}| MI={mi:.3f} [{scale}]")

    # Summary
    print("\n  Thresholds: HIGH=0.7 (384D), MID=0.4 (256D), LOW=0.2 (128D)")

    high_mi = sum(1 for _, mi in sorted_scores if mi >= 0.7)
    mid_mi = sum(1 for _, mi in sorted_scores if 0.4 <= mi < 0.7)
    low_mi = sum(1 for _, mi in sorted_scores if 0.2 <= mi < 0.4)
    dropped = sum(1 for _, mi in sorted_scores if mi < 0.2)

    print(f"\n  High MI (384D): {high_mi} nodes")
    print(f"  Mid MI (256D): {mid_mi} nodes")
    print(f"  Low MI (128D): {low_mi} nodes")
    print(f"  Dropped (<0.2): {dropped} nodes")


def demo_convenience_function():
    """Demonstrate the convenience function API."""
    print("\n" + "=" * 70)
    print("DEMO 5: Convenience Function API")
    print("=" * 70)

    from HoloLoom.context_packing import information_budget_pack

    graph = create_sample_knowledge_graph()
    if graph is None:
        print("Skipping demo - networkx not available")
        return

    candidate_nodes = list(graph.nodes())
    node_contents = {n: graph.nodes[n].get("content", "") for n in candidate_nodes}
    query = "Thompson Sampling for exploration"

    print(f"\nQuery: {query}")
    print("\nUsing information_budget_pack() convenience function:")
    print()
    print("  nodes, scales, mi_scores = information_budget_pack(")
    print('      query="Thompson Sampling for exploration",')
    print("      candidate_nodes=candidates,")
    print("      graph=kg,")
    print("      node_contents=contents,")
    print("      information_budget=5.0")
    print("  )")
    print()

    start = time.perf_counter()
    nodes, scales, mi_scores = information_budget_pack(
        query=query,
        candidate_nodes=candidate_nodes,
        graph=graph,
        node_contents=node_contents,
        information_budget=5.0
    )
    elapsed = (time.perf_counter() - start) * 1000

    print(f"Result:")
    print(f"  Packed nodes: {len(nodes)}")
    print(f"  Nodes: {nodes}")
    print(f"  Time: {elapsed:.2f}ms")
    print()
    print("  Scale assignments:")
    for node in nodes[:5]:
        print(f"    {node}: {scales.get(node, 'N/A')}D (MI={mi_scores.get(node, 0):.3f})")


def main():
    """Run all demos."""
    print()
    print("=" * 70)
    print("   Phase 5: Information-Theoretic Context Packing Demo")
    print("   Tishby's Information Bottleneck for Context Compression")
    print("=" * 70)
    print()
    print("This demo showcases the information-theoretic improvements:")
    print("  - 7-signal importance scoring (25% MI weight)")
    print("  - Information budget compression (greedy MI selection)")
    print("  - MI-aware Matryoshka scales (384D/256D/128D)")
    print("  - MI caching for <5ms overhead")
    print()

    try:
        # Run demos
        demo_standard_vs_information_packing()
        demo_budget_comparison()
        demo_cache_effectiveness()
        demo_mi_score_distribution()
        demo_convenience_function()

        print("\n" + "=" * 70)
        print("   All demos completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
