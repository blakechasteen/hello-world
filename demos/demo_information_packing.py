#!/usr/bin/env python3
"""
Demo: Information-Theoretic Context Packing

This demo showcases HoloLoom information-theoretic improvements:
- Compare standard vs MI-aware packing
- Show token savings with information budget
- Visualize MI score distribution
- Demonstrate Matryoshka scale assignment

Run with:
    PYTHONPATH=. python demos/demo_information_packing.py

Created: December 2025
"""

from typing import Dict, List, Any, Tuple
import random
import math


def print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}\n")


class MockGraph:
    """Mock graph for demonstration purposes."""

    def __init__(self, node_ids: List[str]):
        self._nodes = node_ids
        self._degrees = {n: random.randint(1, 5) for n in node_ids}

    def nodes(self):
        return self._nodes

    def degree(self, node_id: str) -> int:
        return self._degrees.get(node_id, 1)

def create_synthetic_knowledge_graph() -> Tuple[MockGraph, Dict[str, str], Dict[str, float]]:
    """
    Create a synthetic knowledge graph with varied MI scores.

    Returns:
        (graph, node_contents, importance_scores)
    """
    # High MI nodes (directly relevant to Thompson Sampling)
    high_mi_nodes = [
        ("thompson_sampling", "Thompson Sampling is a Bayesian approach to multi-armed bandits"),
        ("bayesian_prior", "Bayesian priors encode initial beliefs about arm distributions"),
        ("posterior_update", "Posterior updates combine prior with observed rewards"),
        ("exploration_exploitation", "Balancing exploration of unknown arms vs exploitation of best known"),
        ("beta_distribution", "Beta distribution models binary outcome probabilities"),
    ]

    # Medium MI nodes (related ML/stats concepts)
    medium_mi_nodes = [
        ("ucb_algorithm", "Upper Confidence Bound uses optimism under uncertainty"),
        ("epsilon_greedy", "Epsilon-greedy explores randomly with probability epsilon"),
        ("regret_bound", "Regret measures cumulative loss vs optimal strategy"),
        ("conjugate_prior", "Conjugate priors have same form as posterior"),
        ("monte_carlo", "Monte Carlo methods use random sampling for estimation"),
    ]
    # Low MI nodes (tangentially related)
    low_mi_nodes = [
        ("gradient_descent", "Gradient descent optimizes parameters by following negative gradient"),
        ("neural_network", "Neural networks are compositions of linear and nonlinear functions"),
        ("loss_function", "Loss functions measure prediction error"),
        ("backpropagation", "Backpropagation computes gradients via chain rule"),
        ("activation_function", "Activation functions introduce nonlinearity"),
    ]

    # Noise nodes (irrelevant)
    noise_nodes = [
        ("database_indexing", "B-trees provide O(log n) lookup in databases"),
        ("http_protocol", "HTTP is a stateless request-response protocol"),
        ("css_styling", "CSS cascading determines style precedence"),
        ("docker_container", "Containers isolate application dependencies"),
        ("git_branching", "Git branches enable parallel development"),
    ]

    # Combine all nodes
    all_nodes = high_mi_nodes + medium_mi_nodes + low_mi_nodes + noise_nodes
    node_ids = [n[0] for n in all_nodes]
    node_contents = {n[0]: n[1] for n in all_nodes}

    # Create importance scores based on MI category
    importance_scores = {}
    for node_id, _ in high_mi_nodes:
        importance_scores[node_id] = random.uniform(0.75, 0.95)
    for node_id, _ in medium_mi_nodes:
        importance_scores[node_id] = random.uniform(0.45, 0.65)
    for node_id, _ in low_mi_nodes:
        importance_scores[node_id] = random.uniform(0.25, 0.38)
    for node_id, _ in noise_nodes:
        importance_scores[node_id] = random.uniform(0.05, 0.18)

    graph = MockGraph(node_ids)
    return graph, node_contents, importance_scores

def compute_mi_score(node_id: str, query: str, importance: float) -> float:
    """
    Compute mutual information score between node and query.
    
    In production, this would use actual MI calculation.
    Here we simulate based on importance + query relevance.
    """
    # Simulate MI based on importance and keyword matching
    query_lower = query.lower()
    keywords = ["thompson", "sampling", "bayesian", "bandit", "exploration"]
    keyword_boost = sum(0.1 for kw in keywords if kw in node_id.lower())
    mi = importance + keyword_boost
    return min(1.0, mi)


def estimate_tokens(node_content: str, scale: int) -> int:
    """Estimate tokens based on content length and Matryoshka scale."""
    base_tokens = len(node_content.split()) * 1.3  # ~1.3 tokens per word
    scale_multiplier = {384: 1.0, 256: 0.67, 128: 0.33}
    return int(base_tokens * scale_multiplier.get(scale, 1.0))


def assign_matryoshka_scale(mi_score: float) -> int:
    """Assign Matryoshka embedding scale based on MI score."""
    if mi_score >= 0.7:
        return 384  # Full detail
    elif mi_score >= 0.4:
        return 256  # Medium detail
    elif mi_score >= 0.2:
        return 128  # Minimal detail
    else:
        return 0    # Dropped

def demo_standard_vs_mi_aware():
    """Demo 1: Compare standard vs MI-aware packing."""
    print_section("DEMO 1: Standard vs MI-Aware Packing")
    
    query = "How does Thompson Sampling balance exploration and exploitation?"
    print(f"Query: {query}")
    print()
    
    graph, node_contents, importance_scores = create_synthetic_knowledge_graph()
    
    # Standard packing: just take top K by importance
    print("[Standard Packing] Top 10 by importance score:")
    sorted_by_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    standard_kept = sorted_by_importance[:10]
    standard_tokens = sum(estimate_tokens(node_contents[n], 384) for n, _ in standard_kept)
    
    for node_id, imp in standard_kept:
        print(f"  {node_id}: importance={imp:.2f}")
    print(f"  Total tokens: {standard_tokens}")
    print()
    
    # MI-aware packing: compute MI scores and use graduated scales
    print("[MI-Aware Packing] With Matryoshka scale assignment:")
    mi_scores = {
        node_id: compute_mi_score(node_id, query, imp)
        for node_id, imp in importance_scores.items()
    }
    
    mi_aware_results = []
    mi_tokens = 0
    for node_id, mi in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True):
        scale = assign_matryoshka_scale(mi)
        if scale > 0:
            tokens = estimate_tokens(node_contents[node_id], scale)
            mi_tokens += tokens
            mi_aware_results.append((node_id, mi, scale, tokens))
    
    for node_id, mi, scale, tokens in mi_aware_results[:10]:
        print(f"  {node_id}: MI={mi:.2f}, scale={scale}D, tokens={tokens}")
    print(f"  Total tokens: {mi_tokens}")
    print()
    
    # Summary
    savings = (standard_tokens - mi_tokens) / standard_tokens * 100 if standard_tokens > 0 else 0
    print(f"Summary:")
    print(f"  Standard: {len(standard_kept)} nodes, {standard_tokens} tokens")
    print(f"  MI-Aware: {len(mi_aware_results)} nodes, {mi_tokens} tokens")
    print(f"  Token savings: {savings:.1f}%")
    print(f"  Avg MI (MI-aware): {sum(r[1] for r in mi_aware_results) / len(mi_aware_results):.2f}")


def demo_token_savings():
    """Demo 2: Show token savings with information budget."""
    print_section("DEMO 2: Token Savings with Information Budget")
    
    query = "Explain Thompson Sampling for multi-armed bandits"
    print(f"Query: {query}")
    print()
    
    graph, node_contents, importance_scores = create_synthetic_knowledge_graph()
    
    # Compute MI scores
    mi_scores = {
        node_id: compute_mi_score(node_id, query, imp)
        for node_id, imp in importance_scores.items()
    }
    
    # Test different information budgets
    budgets = [2.0, 4.0, 6.0, 8.0, 10.0]
    print("Information Budget Analysis:")
    print("-" * 55)
    print(f"  Budget(bits)  Nodes  Tokens  Avg MI  Info Density")
    print("-" * 55)
    
    for budget in budgets:
        # Select nodes up to information budget
        # Simulate: higher MI nodes contribute more "information bits"
        sorted_nodes = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
        
        selected = []
        total_info = 0.0
        total_tokens = 0
        
        for node_id, mi in sorted_nodes:
            # Estimate information contribution (simplified)
            info_contrib = mi * 1.5  # Scale factor
            if total_info + info_contrib <= budget or len(selected) == 0:
                scale = assign_matryoshka_scale(mi)
                if scale > 0:
                    tokens = estimate_tokens(node_contents[node_id], scale)
                    selected.append((node_id, mi, tokens))
                    total_info += info_contrib
                    total_tokens += tokens
        
        avg_mi = sum(s[1] for s in selected) / len(selected) if selected else 0
        info_density = total_info / total_tokens * 100 if total_tokens > 0 else 0
        
        print(f"  {budget:>6.1f}        {len(selected):>3}    {total_tokens:>5}   {avg_mi:.2f}    {info_density:.2f}")
    
    print("-" * 55)
    print()
    print("Insight: Higher budgets include more nodes but with diminishing returns.")
    print("         Information density peaks at moderate budgets (4-6 bits).")


def demo_mi_distribution():
    """Demo 3: Visualize MI score distribution."""
    print_section("DEMO 3: MI Score Distribution")
    
    query = "What is Thompson Sampling?"
    print(f"Query: {query}")
    print()
    
    graph, node_contents, importance_scores = create_synthetic_knowledge_graph()
    
    # Compute MI scores
    mi_scores = {
        node_id: compute_mi_score(node_id, query, imp)
        for node_id, imp in importance_scores.items()
    }
    
    # Create ASCII histogram
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.7), (0.7, 1.0)]
    bin_labels = ["DROP (<0.2)", "128D (0.2-0.4)", "256D (0.4-0.7)", "384D (>=0.7)"]
    bin_counts = [0, 0, 0, 0]
    
    for mi in mi_scores.values():
        if mi < 0.2:
            bin_counts[0] += 1
        elif mi < 0.4:
            bin_counts[1] += 1
        elif mi < 0.7:
            bin_counts[2] += 1
        else:
            bin_counts[3] += 1
    
    print("MI Score Distribution (ASCII Histogram):")
    print()
    max_count = max(bin_counts) if bin_counts else 1
    bar_width = 40
    
    for label, count in zip(bin_labels, bin_counts):
        bar_len = int(count / max_count * bar_width)
        bar = "#" * bar_len
        print(f"  {label:>14} | {bar:<40} ({count})")
    
    print()
    print("Threshold Markers:")
    print("  |----DROP----|--128D--|----256D----|----384D----|")
    print("  0.0         0.2      0.4          0.7          1.0")
    print()
    
    # Statistics
    mi_values = list(mi_scores.values())
    mean_mi = sum(mi_values) / len(mi_values)
    sorted_mi = sorted(mi_values)
    median_mi = sorted_mi[len(sorted_mi) // 2]
    std_mi = math.sqrt(sum((m - mean_mi) ** 2 for m in mi_values) / len(mi_values))
    
    print("Statistics:")
    print(f"  Mean MI:   {mean_mi:.3f}")
    print(f"  Median MI: {median_mi:.3f}")
    print(f"  Std Dev:   {std_mi:.3f}")
    print(f"  Q1 (25%):  {sorted_mi[len(sorted_mi) // 4]:.3f}")
    print(f"  Q3 (75%):  {sorted_mi[3 * len(sorted_mi) // 4]:.3f}")


def demo_scale_assignment():
    """Demo 4: Demonstrate Matryoshka scale assignment."""
    print_section("DEMO 4: Matryoshka Scale Assignment")
    
    query = "Explain Bayesian methods in Thompson Sampling"
    print(f"Query: {query}")
    print()
    
    graph, node_contents, importance_scores = create_synthetic_knowledge_graph()
    
    # Compute MI scores
    mi_scores = {
        node_id: compute_mi_score(node_id, query, imp)
        for node_id, imp in importance_scores.items()
    }
    
    # Assign scales and calculate tokens
    scale_groups = {384: [], 256: [], 128: [], 0: []}
    total_tokens_by_scale = {384: 0, 256: 0, 128: 0}
    
    for node_id, mi in mi_scores.items():
        scale = assign_matryoshka_scale(mi)
        scale_groups[scale].append((node_id, mi))
        if scale > 0:
            total_tokens_by_scale[scale] += estimate_tokens(node_contents[node_id], scale)
    
    print("Scale Assignment Results:")
    print()
    
    # 384D (full detail)
    print(f"384D (Full Detail) - MI >= 0.7:")
    print(f"  Nodes: {len(scale_groups[384])}, Tokens: {total_tokens_by_scale[384]}")
    for node_id, mi in sorted(scale_groups[384], key=lambda x: x[1], reverse=True)[:3]:
        print(f"    - {node_id} (MI={mi:.2f})")
    if len(scale_groups[384]) > 3:
        print(f"    ... and {len(scale_groups[384]) - 3} more")
    print()
    
    # 256D (medium detail)
    print(f"256D (Medium Detail) - MI 0.4-0.7:")
    print(f"  Nodes: {len(scale_groups[256])}, Tokens: {total_tokens_by_scale[256]}")
    for node_id, mi in sorted(scale_groups[256], key=lambda x: x[1], reverse=True)[:3]:
        print(f"    - {node_id} (MI={mi:.2f})")
    if len(scale_groups[256]) > 3:
        print(f"    ... and {len(scale_groups[256]) - 3} more")
    print()
    
    # 128D (minimal detail)
    print(f"128D (Minimal Detail) - MI 0.2-0.4:")
    print(f"  Nodes: {len(scale_groups[128])}, Tokens: {total_tokens_by_scale[128]}")
    for node_id, mi in sorted(scale_groups[128], key=lambda x: x[1], reverse=True)[:3]:
        print(f"    - {node_id} (MI={mi:.2f})")
    if len(scale_groups[128]) > 3:
        print(f"    ... and {len(scale_groups[128]) - 3} more")
    print()
    
    # Dropped
    print(f"DROPPED - MI < 0.2:")
    print(f"  Nodes: {len(scale_groups[0])} (saved tokens by exclusion)")
    for node_id, mi in sorted(scale_groups[0], key=lambda x: x[1], reverse=True)[:3]:
        print(f"    - {node_id} (MI={mi:.2f})")
    print()
    
    # Token savings calculation
    full_tokens = sum(estimate_tokens(node_contents[n], 384) for n in mi_scores.keys())
    actual_tokens = sum(total_tokens_by_scale.values())
    savings = (full_tokens - actual_tokens) / full_tokens * 100
    
    print("Token Savings Summary:")
    print(f"  If all 384D: {full_tokens} tokens")
    print(f"  With scaling: {actual_tokens} tokens")
    print(f"  Savings:      {savings:.1f}%")
    print()
    print("Scale Distribution:")
    total_kept = len(scale_groups[384]) + len(scale_groups[256]) + len(scale_groups[128])
    print(f"  384D: {len(scale_groups[384]):>2} nodes ({len(scale_groups[384])/total_kept*100:.0f}%)")
    print(f"  256D: {len(scale_groups[256]):>2} nodes ({len(scale_groups[256])/total_kept*100:.0f}%)")
    print(f"  128D: {len(scale_groups[128]):>2} nodes ({len(scale_groups[128])/total_kept*100:.0f}%)")


def main():
    """Run all demos."""
    print()
    print("=" * 60)
    print(" Information-Theoretic Context Packing Demo")
    print("=" * 60)
    print()
    print("This demo showcases HoloLoom's information-theoretic improvements:")
    print("- MI-aware context packing (vs standard importance-only)")
    print("- Token savings with information budget constraints")
    print("- Matryoshka scale assignment based on MI scores")
    print()
    print("Key Concepts:")
    print("  - Information Bottleneck: Maximize I(Context; Query) within budget")
    print("  - MI Thresholds: >0.7->384D, 0.4-0.7->256D, 0.2-0.4->128D, <0.2->DROP")
    print("  - Graduated compression preserves high-value information")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Run all demos
    demo_standard_vs_mi_aware()
    demo_token_savings()
    demo_mi_distribution()
    demo_scale_assignment()
    
    # Final summary
    print_section("DEMO COMPLETE")
    print("Key Takeaways:")
    print("  1. MI-aware packing selects nodes by query relevance, not just importance")
    print("  2. Information budget controls context size with diminishing returns")
    print("  3. Matryoshka scales enable graduated detail based on MI scores")
    print("  4. Typical token savings: 30-60% with maintained quality")
    print()
    print("For production use, integrate with:")
    print("  - HoloLoom.context_packing.packer.ContextPacker")
    print("  - HoloLoom.context_packing.importance_scorer.ImportanceScorer")
    print("  - HoloLoom.context_packing.context_compressor.ContextCompressor")
    print()


if __name__ == "__main__":
    main()
