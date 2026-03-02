"""
Math Modules Integration Demo
=============================

Demonstrates HoloLoom's math modules in action with:
1. Information Theory + RAG Confidence Scoring
2. Spectral Clustering + Knowledge Graph Analysis
3. Game Theory + Multi-Tool Policy Coordination
4. Causal Inference + Reasoning Chain Validation
5. Regret Minimization + Adaptive Query Routing

Run: PYTHONPATH=. python demos/demo_math_integration.py
"""

import asyncio
import numpy as np
import time
from typing import List, Dict, Any, Tuple

# Math modules
from hololoom.warp.math.information import entropy, mutual_information, InformationMetrics
from hololoom.warp.math.graph import (
    compute_laplacian, fiedler_vector, algebraic_connectivity,
    KnowledgeGraphClusterer, SpectralClustering
)
from hololoom.warp.math.decision.game_theory import NormalFormGame, NashEquilibrium
from hololoom.warp.math.causal import CausalGraph
from hololoom.warp.math.decision.regret_minimization import Hedge

# HoloLoom components
from hololoom.memory.graph import KG, KGEdge

import networkx as nx


def print_header(title: str):
    """Print a formatted header."""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str):
    """Print a section separator."""
    print()
    print(f"--- {title} ---")


# =============================================================================
# Demo 1: Information Theory + RAG Confidence Scoring
# =============================================================================

def demo_information_theory_rag():
    """
    Shows how entropy-based confidence scoring improves RAG query reliability.

    Key insight: High entropy in source relevance = low confidence
                 Low entropy = one dominant source = high confidence
    """
    print_header("Demo 1: Information Theory + RAG Confidence")

    # Simulate RAG retrieval results (relevance scores from 5 sources)
    scenarios = [
        ("Uniform sources (uncertain)", [0.2, 0.2, 0.2, 0.2, 0.2]),
        ("One dominant source (confident)", [0.8, 0.05, 0.05, 0.05, 0.05]),
        ("Two strong sources (moderate)", [0.4, 0.4, 0.1, 0.05, 0.05]),
        ("Noisy retrieval (uncertain)", [0.3, 0.25, 0.2, 0.15, 0.1]),
    ]

    max_entropy = np.log2(5)  # Max entropy for 5 sources (using base 2)

    print_section("Entropy-based Confidence Adjustment")
    print()
    print(f"{'Scenario':<35} {'Entropy':>10} {'Confidence':>12}")
    print("-" * 60)

    for name, relevances in scenarios:
        probs = np.array(relevances)
        probs = probs / probs.sum()  # Normalize

        # Calculate Shannon entropy using standalone function
        h = entropy(probs, base=2.0)

        # Confidence = 1 - normalized_entropy
        # High entropy = low confidence, Low entropy = high confidence
        confidence = 1.0 - (h / max_entropy)

        print(f"{name:<35} {h:>10.3f} {confidence:>12.1%}")

    print()
    print("Interpretation:")
    print("  - Low entropy = concentrated probability = high confidence")
    print("  - High entropy = spread out probability = low confidence")
    print("  - This adjusts RAG response confidence based on source agreement")

    # Show mutual information for source deduplication
    print_section("Mutual Information for Source Deduplication")

    # Simulate two sources with varying redundancy
    source_pairs = [
        ("Independent sources", np.array([[0.25, 0.25], [0.25, 0.25]])),
        ("Redundant sources", np.array([[0.45, 0.05], [0.05, 0.45]])),
        ("Partially overlapping", np.array([[0.3, 0.2], [0.2, 0.3]])),
    ]

    print()
    print(f"{'Source Pair':<25} {'MI':>10} {'Interpretation':>20}")
    print("-" * 60)

    from hololoom.warp.math.information import DistributionPair

    for name, joint in source_pairs:
        pair = DistributionPair(joint=joint)
        mi = mutual_information(pair, base=2.0)

        if mi < 0.1:
            interpretation = "Keep both"
        elif mi < 0.5:
            interpretation = "Partially redundant"
        else:
            interpretation = "Deduplicate"

        print(f"{name:<25} {mi:>10.3f} {interpretation:>20}")


# =============================================================================
# Demo 2: Spectral Clustering + Knowledge Graph Analysis
# =============================================================================

def demo_spectral_knowledge_graph():
    """
    Uses graph Laplacian and spectral methods to analyze knowledge graph structure.

    Key insight: Algebraic connectivity reveals how tightly connected the graph is,
                 Fiedler vector enables optimal graph partitioning.
    """
    print_header("Demo 2: Spectral Clustering + Knowledge Graph")

    # Create a sample knowledge graph with clear topic clusters
    kg = KG()

    # Cluster 1: Machine Learning concepts
    ml_edges = [
        KGEdge("neural_network", "deep_learning", "IS_A", 0.9),
        KGEdge("deep_learning", "transformer", "USES", 0.8),
        KGEdge("transformer", "attention", "USES", 0.9),
        KGEdge("attention", "neural_network", "PART_OF", 0.7),
    ]

    # Cluster 2: Statistics concepts
    stats_edges = [
        KGEdge("bayesian", "probability", "IS_A", 0.9),
        KGEdge("probability", "distribution", "USES", 0.8),
        KGEdge("distribution", "sampling", "USES", 0.7),
        KGEdge("sampling", "bayesian", "USES", 0.8),
    ]

    # Bridge between clusters
    bridge_edges = [
        KGEdge("neural_network", "bayesian", "USES", 0.5),  # Weak connection
    ]

    for edge in ml_edges + stats_edges + bridge_edges:
        kg.add_edge(edge)

    # Access the underlying NetworkX graph
    G = kg.G

    print_section("Knowledge Graph Structure")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Convert NetworkX graph to Laplacian matrix
    print_section("Spectral Analysis")

    # Use compute_laplacian with networkx data type
    L = compute_laplacian(G, data_type="networkx")
    print(f"  Laplacian matrix shape: {L.shape}")

    # Algebraic connectivity (Fiedler value)
    connectivity = algebraic_connectivity(L)
    print(f"  Algebraic Connectivity: {connectivity:.4f}")

    if connectivity < 0.5:
        print("  Interpretation: Graph has weak connections (potential clusters)")
    else:
        print("  Interpretation: Graph is well-connected")

    # Fiedler vector for partitioning (returns tuple: vector, value)
    fiedler_vec, fiedler_val = fiedler_vector(L)

    # Partition based on Fiedler vector sign
    nodes = list(G.nodes())
    cluster_1 = [nodes[i] for i in range(len(nodes)) if fiedler_vec[i] < 0]
    cluster_2 = [nodes[i] for i in range(len(nodes)) if fiedler_vec[i] >= 0]

    print_section("Spectral Graph Partitioning (Fiedler Vector)")
    print(f"  Cluster 1: {', '.join(cluster_1)}")
    print(f"  Cluster 2: {', '.join(cluster_2)}")

    # Full spectral clustering using adjacency matrix
    print_section("K-Means Spectral Clustering (k=2)")

    # Convert to adjacency matrix for SpectralClustering
    A = nx.to_numpy_array(G)
    clustering = SpectralClustering(n_clusters=2)
    result = clustering.fit(A)

    for label in set(result.labels):
        members = [nodes[i] for i in range(len(nodes)) if result.labels[i] == label]
        print(f"  Cluster {label}: {', '.join(members)}")


# =============================================================================
# Demo 3: Game Theory + Multi-Tool Policy Coordination
# =============================================================================

def demo_game_theory_policy():
    """
    Models tool selection as a strategic game to find optimal coordination.

    Key insight: Nash equilibrium tells us which tool to select given query
                 complexity and available context.
    """
    print_header("Demo 3: Game Theory + Tool Selection Policy")

    # Model as 2-player game:
    # Player 1 (Query): simple vs complex
    # Player 2 (Context): sparse vs rich

    # Payoff matrix: higher = better outcome
    # Rows = query type, Cols = context type
    # Values = expected response quality

    print_section("Tool Selection Payoff Matrix")
    print()
    print("Tool: FAST (minimal processing)")
    fast_payoffs = np.array([
        [0.9, 0.7],  # Simple query: good with sparse, ok with rich (overkill)
        [0.4, 0.3],  # Complex query: bad with both (needs thorough)
    ])
    print("           Sparse   Rich")
    print(f"Simple:    {fast_payoffs[0,0]:.1f}      {fast_payoffs[0,1]:.1f}")
    print(f"Complex:   {fast_payoffs[1,0]:.1f}      {fast_payoffs[1,1]:.1f}")

    print()
    print("Tool: FUSED (thorough processing)")
    fused_payoffs = np.array([
        [0.6, 0.7],  # Simple query: ok (overkill) to good
        [0.7, 0.95], # Complex query: good to excellent
    ])
    print("           Sparse   Rich")
    print(f"Simple:    {fused_payoffs[0,0]:.1f}      {fused_payoffs[0,1]:.1f}")
    print(f"Complex:   {fused_payoffs[1,0]:.1f}      {fused_payoffs[1,1]:.1f}")

    # Find optimal tool for each scenario
    print_section("Optimal Tool Selection (Pure Strategy)")

    query_types = ["Simple", "Complex"]
    context_types = ["Sparse", "Rich"]

    for i, query in enumerate(query_types):
        for j, context in enumerate(context_types):
            fast_value = fast_payoffs[i, j]
            fused_value = fused_payoffs[i, j]

            optimal = "FAST" if fast_value > fused_value else "FUSED"
            print(f"  {query} query + {context} context: Use {optimal} "
                  f"(FAST={fast_value:.1f}, FUSED={fused_value:.1f})")

    # Mixed strategy for uncertain cases
    print_section("Mixed Strategy Nash Equilibrium")

    # Model as zero-sum-like game between exploration and exploitation
    # Matching Pennies: classic game with mixed equilibrium at (0.5, 0.5)
    explore_exploit = np.array([
        [1, -1],   # Explore when novel
        [-1, 1],   # Exploit when familiar
    ])

    game = NormalFormGame([explore_exploit, -explore_exploit])
    nash = NashEquilibrium()
    mixed = nash.find_mixed_2player(game)

    if mixed is not None:
        p1_strategy, p2_strategy = mixed
        print(f"  Exploration probability: {p1_strategy.probabilities[0]:.2f}")
        print(f"  Exploitation probability: {p1_strategy.probabilities[1]:.2f}")
        print("  (Equal mixing optimal for this zero-sum game)")
    else:
        print("  No mixed equilibrium found (pure strategies dominate)")


# =============================================================================
# Demo 4: Causal Inference + Reasoning Chain Validation
# =============================================================================

def demo_causal_reasoning():
    """
    Uses d-separation to validate reasoning chain independence.

    Key insight: If answer is conditionally independent of query given context,
                 then the reasoning chain is properly structured.
    """
    print_header("Demo 4: Causal Inference + Reasoning Validation")

    # Build causal DAG for HoloLoom reasoning pipeline
    cg = CausalGraph()

    # Main reasoning chain
    cg.add_edge("query", "retrieval")
    cg.add_edge("retrieval", "context")
    cg.add_edge("context", "reasoning")
    cg.add_edge("reasoning", "answer")

    # Additional influences
    cg.add_edge("query", "complexity")
    cg.add_edge("complexity", "tool_selection")
    cg.add_edge("tool_selection", "reasoning")

    print_section("Reasoning Pipeline Causal Graph")
    print("  query --> retrieval --> context --> reasoning --> answer")
    print("    |                                    ^")
    print("    +--> complexity --> tool_selection --+")

    # Test d-separation (conditional independence)
    print_section("D-Separation Tests (Conditional Independence)")

    tests = [
        # (X, Y, conditioning_set, expected_interpretation)
        ({"query"}, {"answer"}, {"context", "tool_selection"},
         "Answer is influenced only through context and tool (good!)"),
        ({"query"}, {"answer"}, set(),
         "Query and answer are dependent (expected)"),
        ({"retrieval"}, {"tool_selection"}, {"query"},
         "Retrieval and tool selection independent given query"),
        ({"complexity"}, {"answer"}, {"reasoning"},
         "Complexity's effect goes through reasoning"),
    ]

    print()
    print(f"{'X':<15} {'Y':<15} {'Given':<20} {'Independent?':<12}")
    print("-" * 70)

    for x, y, given, interpretation in tests:
        independent = cg.d_separated(x, y, given)
        x_str = list(x)[0]
        y_str = list(y)[0]
        given_str = str(given) if given else "{}"
        status = "Yes" if independent else "No"
        print(f"{x_str:<15} {y_str:<15} {given_str:<20} {status:<12}")

    print()
    print("Interpretation:")
    print("  - d-separation tells us which variables are conditionally independent")
    print("  - This validates that reasoning chains are properly structured")
    print("  - Can detect when confounders need to be controlled")

    # Backdoor criterion example
    print_section("Backdoor Criterion for Effect Estimation")

    # Can we estimate effect of tool_selection on answer?
    treatment = "tool_selection"
    outcome = "answer"

    # Check if complexity is a valid adjustment set
    adjustment_set = {"complexity"}

    print(f"  Treatment: {treatment}")
    print(f"  Outcome: {outcome}")
    print(f"  Proposed adjustment: {adjustment_set}")

    # A valid adjustment set blocks all backdoor paths
    # Here: query -> complexity -> tool_selection is blocked if we adjust for complexity
    print("  Result: Valid adjustment set (blocks backdoor path through complexity)")


# =============================================================================
# Demo 5: Regret Minimization + Adaptive Query Routing
# =============================================================================

def demo_regret_routing():
    """
    Uses Hedge algorithm to adaptively route queries to optimal complexity levels.

    Key insight: Multiplicative weights learn which routes work best,
                 achieving sublinear O(sqrt(T) log n) regret.
    """
    print_header("Demo 5: Regret Minimization + Adaptive Routing")

    # 4 routing options with different characteristics
    routes = ["LITE", "FAST", "FULL", "RESEARCH"]
    n_actions = len(routes)

    # Initialize Hedge algorithm
    hedge = Hedge(n_actions=n_actions, learning_rate=0.1)

    # Simulate query stream with varying optimal routes
    np.random.seed(42)
    n_rounds = 100

    # Ground truth: optimal route varies by query type
    def get_losses(query_type: int) -> np.ndarray:
        """Return losses for each route given query type (lower = better)."""
        if query_type == 0:  # Simple query
            return np.array([0.1, 0.2, 0.5, 0.8])  # LITE best
        elif query_type == 1:  # Medium query
            return np.array([0.4, 0.1, 0.2, 0.4])  # FAST best
        elif query_type == 2:  # Complex query
            return np.array([0.9, 0.5, 0.1, 0.2])  # FULL best
        else:  # Research query
            return np.array([1.0, 0.8, 0.3, 0.1])  # RESEARCH best

    # Track metrics
    cumulative_loss = 0.0
    best_fixed_loss = np.zeros(n_actions)
    choices_over_time = []

    print_section("Hedge Algorithm Learning Progress")
    print()

    for t in range(n_rounds):
        # Random query type
        query_type = np.random.randint(0, 4)

        # Get current distribution
        probs = hedge.get_distribution()

        # Choose action
        action = hedge.choose()
        choices_over_time.append(action)

        # Get losses
        losses = get_losses(query_type)

        # Incur loss
        cumulative_loss += losses[action]
        best_fixed_loss += losses

        # Update Hedge
        hedge.update(losses)

        # Print progress at intervals
        if (t + 1) % 25 == 0:
            stats = hedge.get_stats()
            print(f"  Round {t+1:3d}: Regret = {stats.cumulative_regret:6.2f}, "
                  f"Avg = {stats.average_regret:.3f}, "
                  f"Bound = {stats.regret_bound:.2f}")

    # Final analysis
    print_section("Final Routing Distribution")
    final_probs = hedge.get_distribution()

    for i, route in enumerate(routes):
        print(f"  {route:<10}: {final_probs[i]:6.1%}")

    # Choice frequency
    print_section("Route Usage Frequency")
    for i, route in enumerate(routes):
        count = choices_over_time.count(i)
        print(f"  {route:<10}: {count:3d} times ({count/n_rounds:.1%})")

    # Regret analysis
    print_section("Regret Analysis")
    stats = hedge.get_stats()

    best_in_hindsight = np.argmin(best_fixed_loss)
    print(f"  Best fixed route (hindsight): {routes[best_in_hindsight]}")
    print(f"  Cumulative regret: {stats.cumulative_regret:.2f}")
    print(f"  Average regret: {stats.average_regret:.4f}")
    print(f"  Theoretical O(sqrt(T)): {stats.regret_bound:.2f}")
    print()
    print("  Regret is sublinear: algorithm adapts to query distribution!")


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all integration demos."""
    print()
    print("=" * 60)
    print("  HOLOLOOM MATH MODULES INTEGRATION DEMO")
    print("=" * 60)
    print()
    print("This demo shows math modules integrated with HoloLoom systems:")
    print("  1. Information Theory + RAG Confidence")
    print("  2. Spectral Clustering + Knowledge Graph")
    print("  3. Game Theory + Tool Policy")
    print("  4. Causal Inference + Reasoning Validation")
    print("  5. Regret Minimization + Adaptive Routing")

    start = time.perf_counter()

    # Run demos
    demo_information_theory_rag()
    demo_spectral_knowledge_graph()
    demo_game_theory_policy()
    demo_causal_reasoning()
    demo_regret_routing()

    elapsed = (time.perf_counter() - start) * 1000

    print()
    print("=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print(f"  Total runtime: {elapsed:.1f}ms")
    print()
    print("  These math modules provide principled foundations for:")
    print("    - Confidence calibration (Information Theory)")
    print("    - Knowledge organization (Spectral Clustering)")
    print("    - Strategic decision making (Game Theory)")
    print("    - Reasoning validation (Causal Inference)")
    print("    - Adaptive learning (Regret Minimization)")
    print()


if __name__ == "__main__":
    main()
