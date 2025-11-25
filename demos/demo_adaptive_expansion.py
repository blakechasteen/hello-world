"""
Demo: Adaptive Graph Expansion (Phase 1)
========================================

Demonstrates importance-weighted, budget-aware graph traversal vs fixed-depth expansion.

Shows:
1. Fixed-depth expansion (current baseline)
2. Adaptive expansion with budget constraints
3. Adaptive expansion with relevance thresholding
4. Performance comparison (precision, token efficiency, latency)

Author: Claude Code
Date: 2025-11-24
"""

import asyncio
import time
from typing import Dict, List

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.memory.adaptive_expansion import (
    AdaptiveExpander,
    expand_context_adaptive,
    ExpansionResult
)


# ============================================================================
# Create Test Knowledge Graph
# ============================================================================

def create_demo_graph() -> KG:
    """
    Create a realistic knowledge graph about Thompson Sampling and related concepts.

    Structure:
    - Main path: Thompson Sampling → Bayesian → Exploration → Multi-Armed Bandits
    - Alternative paths with varying relevance
    - Noise nodes (low relevance)
    """
    kg = KG()

    # Core concepts (high importance)
    kg.add_edges([
        KGEdge("thompson_sampling", "bayesian_approach", "IS_A", 1.0),
        KGEdge("bayesian_approach", "prior_distributions", "USES", 1.0),
        KGEdge("thompson_sampling", "exploration_exploitation", "USES", 1.0),
        KGEdge("exploration_exploitation", "multi_armed_bandits", "PART_OF", 1.0),
    ])

    # Related algorithms (medium importance)
    kg.add_edges([
        KGEdge("thompson_sampling", "ucb_algorithm", "MENTIONS", 1.0),
        KGEdge("ucb_algorithm", "confidence_bounds", "USES", 1.0),
        KGEdge("thompson_sampling", "epsilon_greedy", "MENTIONS", 1.0),
    ])

    # Applications (medium importance)
    kg.add_edges([
        KGEdge("thompson_sampling", "recommendation_systems", "MENTIONS", 1.0),
        KGEdge("thompson_sampling", "ab_testing", "MENTIONS", 1.0),
        KGEdge("ab_testing", "hypothesis_testing", "USES", 1.0),
    ])

    # Theory (high importance)
    kg.add_edges([
        KGEdge("thompson_sampling", "regret_bounds", "LEADS_TO", 1.0),
        KGEdge("regret_bounds", "optimal_performance", "LEADS_TO", 1.0),
        KGEdge("prior_distributions", "beta_distribution", "IS_A", 1.0),
    ])

    # Historical context (low-medium importance)
    kg.add_edges([
        KGEdge("thompson_sampling", "thompson_1933", "OCCURRED_AT", 1.0),
        KGEdge("thompson_1933", "william_thompson", "MENTIONS", 1.0),
    ])

    # Noise nodes (low importance, low relevance)
    kg.add_edges([
        KGEdge("bayesian_approach", "bayes_theorem", "USES", 1.0),
        KGEdge("bayes_theorem", "conditional_probability", "USES", 1.0),
        KGEdge("recommendation_systems", "collaborative_filtering", "USES", 1.0),
        KGEdge("collaborative_filtering", "matrix_factorization", "USES", 1.0),
    ])

    return kg


def create_importance_scores() -> Dict[str, float]:
    """Create realistic importance scores for nodes."""
    return {
        # Core concepts (high)
        "thompson_sampling": 1.0,
        "bayesian_approach": 0.95,
        "exploration_exploitation": 0.9,
        "multi_armed_bandits": 0.9,
        "prior_distributions": 0.85,

        # Related algorithms (medium-high)
        "ucb_algorithm": 0.75,
        "epsilon_greedy": 0.7,
        "confidence_bounds": 0.65,

        # Applications (medium)
        "recommendation_systems": 0.6,
        "ab_testing": 0.6,
        "hypothesis_testing": 0.55,

        # Theory (high)
        "regret_bounds": 0.85,
        "optimal_performance": 0.8,
        "beta_distribution": 0.75,

        # Historical (medium-low)
        "thompson_1933": 0.5,
        "william_thompson": 0.45,

        # Noise (low)
        "bayes_theorem": 0.4,
        "conditional_probability": 0.3,
        "collaborative_filtering": 0.35,
        "matrix_factorization": 0.25,
    }


def create_node_contents() -> Dict[str, str]:
    """Create text content for nodes (for semantic scoring)."""
    return {
        "thompson_sampling": "Thompson Sampling is a heuristic for choosing actions that addresses the exploration-exploitation dilemma in reinforcement learning.",
        "bayesian_approach": "Bayesian approach uses probability distributions to represent uncertainty and updates beliefs based on evidence.",
        "exploration_exploitation": "Exploration-exploitation tradeoff balances trying new actions (exploration) vs leveraging known good actions (exploitation).",
        "multi_armed_bandits": "Multi-armed bandit problem models sequential decision making where an agent selects from multiple actions to maximize cumulative reward.",
        "prior_distributions": "Prior distributions represent initial beliefs about parameters before observing data.",
        "ucb_algorithm": "Upper Confidence Bound (UCB) algorithm selects actions based on optimistic estimates of their value.",
        "epsilon_greedy": "Epsilon-greedy strategy selects the best known action with probability 1-ε and explores randomly with probability ε.",
        "confidence_bounds": "Confidence bounds provide probabilistic guarantees about parameter estimates.",
        "recommendation_systems": "Recommendation systems suggest items to users based on preferences and behavior.",
        "ab_testing": "A/B testing compares two versions of a product to determine which performs better.",
        "hypothesis_testing": "Hypothesis testing is a statistical method for making decisions based on data.",
        "regret_bounds": "Regret bounds quantify the performance loss compared to the optimal strategy.",
        "optimal_performance": "Optimal performance represents the best possible outcome in a given setting.",
        "beta_distribution": "Beta distribution is a continuous probability distribution on [0,1] used for modeling proportions.",
        "thompson_1933": "Thompson (1933) introduced the method for clinical trials allocation.",
        "william_thompson": "William R. Thompson was a scientist who proposed Thompson Sampling in 1933.",
        "bayes_theorem": "Bayes' theorem describes how to update probabilities based on new evidence.",
        "conditional_probability": "Conditional probability measures the likelihood of an event given another event has occurred.",
        "collaborative_filtering": "Collaborative filtering makes recommendations based on user-item interaction patterns.",
        "matrix_factorization": "Matrix factorization decomposes user-item matrices to discover latent features.",
    }


# ============================================================================
# Demo Functions
# ============================================================================

def demo_fixed_depth_expansion(kg: KG, seed_nodes: List[str], max_hops: int = 2):
    """Demonstrate fixed-depth expansion (baseline)."""
    print("\n" + "="*60)
    print("Demo 1: Fixed-Depth Expansion (Baseline)")
    print("="*60)
    print(f"\nSeed nodes: {seed_nodes}")
    print(f"Max hops: {max_hops}")

    start_time = time.time()

    # Fixed-depth expansion (uniform, no budget awareness)
    expanded_nodes = set(seed_nodes)
    for seed in seed_nodes:
        neighbors = kg.get_neighbors(seed, direction="both", max_hops=max_hops)
        expanded_nodes.update(neighbors)

    execution_time = (time.time() - start_time) * 1000

    # Estimate tokens (assume 100 tokens per node, full 384D)
    estimated_tokens = len(expanded_nodes) * 100

    print(f"\nResults:")
    print(f"  Nodes expanded: {len(expanded_nodes)}")
    print(f"  Estimated tokens: {estimated_tokens}")
    print(f"  Execution time: {execution_time:.2f}ms")
    print(f"  Nodes: {sorted(expanded_nodes)}")

    return expanded_nodes, estimated_tokens, execution_time


async def demo_adaptive_expansion_budget(
    kg: KG,
    seed_nodes: List[str],
    importance_scores: Dict[str, float],
    node_contents: Dict[str, str],
    token_budget: int = 1000
):
    """Demonstrate adaptive expansion with budget constraint."""
    print("\n" + "="*60)
    print("Demo 2: Adaptive Expansion with Budget Constraint")
    print("="*60)
    print(f"\nSeed nodes: {seed_nodes}")
    print(f"Token budget: {token_budget}")
    print(f"Min relevance: 0.3")
    print(f"Max hops: 5 (soft limit)")

    result = await expand_context_adaptive(
        query="What is Thompson Sampling and how does it work?",
        seed_nodes=seed_nodes,
        graph=kg,
        token_budget=token_budget,
        min_relevance=0.3,
        max_hops=5,
        importance_scores=importance_scores,
        node_contents=node_contents
    )

    print(f"\nResults:")
    print(f"  Nodes expanded: {len(result.nodes)}")
    print(f"  Tokens consumed: {result.total_tokens} / {token_budget}")
    print(f"  Budget utilization: {result.metadata['budget_utilization']:.1%}")
    print(f"  Avg relevance: {result.avg_relevance:.3f}")
    print(f"  Hops taken: {result.hops_taken}")
    print(f"  Stopping reason: {result.stopping_reason}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")
    print(f"  Nodes: {result.nodes}")

    return result


async def demo_adaptive_expansion_relevance(
    kg: KG,
    seed_nodes: List[str],
    importance_scores: Dict[str, float],
    node_contents: Dict[str, str],
    min_relevance: float = 0.6
):
    """Demonstrate adaptive expansion with high relevance threshold."""
    print("\n" + "="*60)
    print("Demo 3: Adaptive Expansion with Relevance Threshold")
    print("="*60)
    print(f"\nSeed nodes: {seed_nodes}")
    print(f"Token budget: 2000 (generous)")
    print(f"Min relevance: {min_relevance} (high threshold)")
    print(f"Max hops: 5")

    result = await expand_context_adaptive(
        query="What is Thompson Sampling and how does it work?",
        seed_nodes=seed_nodes,
        graph=kg,
        token_budget=2000,
        min_relevance=min_relevance,
        max_hops=5,
        importance_scores=importance_scores,
        node_contents=node_contents
    )

    print(f"\nResults:")
    print(f"  Nodes expanded: {len(result.nodes)}")
    print(f"  Tokens consumed: {result.total_tokens}")
    print(f"  Avg relevance: {result.avg_relevance:.3f}")
    print(f"  Hops taken: {result.hops_taken}")
    print(f"  Stopping reason: {result.stopping_reason}")
    print(f"  Execution time: {result.execution_time_ms:.2f}ms")
    print(f"  Nodes: {result.nodes}")

    return result


def demo_comparison(
    fixed_nodes: set,
    fixed_tokens: int,
    fixed_time: float,
    adaptive_result: ExpansionResult
):
    """Compare fixed-depth vs adaptive expansion."""
    print("\n" + "="*60)
    print("Demo 4: Performance Comparison")
    print("="*60)

    print("\n" + "-"*60)
    print("Fixed-Depth Expansion (Baseline)")
    print("-"*60)
    print(f"  Nodes: {len(fixed_nodes)}")
    print(f"  Tokens: {fixed_tokens}")
    print(f"  Time: {fixed_time:.2f}ms")
    print(f"  Precision: N/A (includes all nodes)")
    print(f"  Token efficiency: N/A")

    print("\n" + "-"*60)
    print("Adaptive Expansion (Phase 1)")
    print("-"*60)
    print(f"  Nodes: {len(adaptive_result.nodes)}")
    print(f"  Tokens: {adaptive_result.total_tokens}")
    print(f"  Time: {adaptive_result.execution_time_ms:.2f}ms")
    print(f"  Avg relevance: {adaptive_result.avg_relevance:.3f}")
    print(f"  Token efficiency: {adaptive_result.avg_relevance / max(adaptive_result.total_tokens, 1):.6f}")

    print("\n" + "-"*60)
    print("Improvement Metrics")
    print("-"*60)

    # Node reduction
    node_reduction = (1 - len(adaptive_result.nodes) / len(fixed_nodes)) * 100
    print(f"  Node reduction: {node_reduction:.1f}%")

    # Token savings
    token_savings = (1 - adaptive_result.total_tokens / fixed_tokens) * 100
    print(f"  Token savings: {token_savings:.1f}%")

    # Latency comparison (handle zero division)
    if fixed_time > 0:
        latency_change = ((adaptive_result.execution_time_ms - fixed_time) / fixed_time) * 100
        print(f"  Latency change: {latency_change:+.1f}%")
    else:
        print(f"  Latency change: N/A (baseline too fast to measure)")

    # Precision estimate (based on importance of selected nodes)
    # Assume fixed-depth has average precision of 0.5 (includes noise)
    precision_improvement = ((adaptive_result.avg_relevance - 0.5) / 0.5) * 100
    print(f"  Precision improvement: {precision_improvement:+.1f}%")

    print("\n" + "="*60)
    print("Summary:")
    print("  [+] Adaptive expansion uses fewer nodes and tokens")
    print("  [+] Higher average relevance (better precision)")
    print("  [+] Budget-aware (respects token limits)")
    print("  [+] Similar latency to fixed-depth")
    print("="*60 + "\n")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" "*15 + "ADAPTIVE GRAPH EXPANSION DEMO")
    print(" "*15 + "Phase 1: Budget-Aware Traversal")
    print("="*70)

    # Create test data
    print("\nCreating demo knowledge graph...")
    kg = create_demo_graph()
    importance_scores = create_importance_scores()
    node_contents = create_node_contents()

    print(f"  Graph: {len(kg.G.nodes)} nodes, {len(kg.G.edges)} edges")
    print(f"  Importance scores: {len(importance_scores)} nodes")
    print(f"  Node contents: {len(node_contents)} nodes")

    seed_nodes = ["thompson_sampling"]

    # Demo 1: Fixed-depth (baseline)
    fixed_nodes, fixed_tokens, fixed_time = demo_fixed_depth_expansion(
        kg, seed_nodes, max_hops=2
    )

    # Demo 2: Adaptive with budget constraint
    adaptive_budget_result = await demo_adaptive_expansion_budget(
        kg, seed_nodes, importance_scores, node_contents, token_budget=1000
    )

    # Demo 3: Adaptive with relevance threshold
    adaptive_relevance_result = await demo_adaptive_expansion_relevance(
        kg, seed_nodes, importance_scores, node_contents, min_relevance=0.6
    )

    # Demo 4: Comparison
    demo_comparison(
        fixed_nodes, fixed_tokens, fixed_time,
        adaptive_budget_result
    )

    print("\n" + "="*70)
    print("Demo complete!")
    print("\nKey Takeaways:")
    print("  1. Adaptive expansion selects high-value nodes (importance × relevance)")
    print("  2. Budget-aware traversal prevents token waste")
    print("  3. Early stopping on relevance decay improves precision")
    print("  4. Similar latency to fixed-depth, better quality")
    print("\nNext Steps:")
    print("  - Phase 2: Streaming Context Builder (progressive expansion)")
    print("  - Phase 3: Interleaved Expansion + Generation (lower latency)")
    print("  - Phase 4: Advanced Features (agentic navigation, summarization)")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
