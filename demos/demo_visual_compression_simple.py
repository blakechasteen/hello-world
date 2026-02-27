#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Compression Demo (Simplified)
====================================

Quick demonstration of visual compression without full HoloLoom initialization.

This shows the compression module working directly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import networkx as nx
    import matplotlib
    matplotlib.use('Agg')
    DEPS_AVAILABLE = True
except ImportError:
    print("Install: pip install matplotlib networkx")
    sys.exit(1)

from hololoom.memory.visual_compression import (
    compress_to_visual,
    CompressionType,
    CompressionMetrics
)


def create_knowledge_graph():
    """Create a larger knowledge graph for meaningful compression."""
    G = nx.DiGraph()

    # Core concepts (20 nodes)
    concepts = [
        "Thompson Sampling", "Multi-Armed Bandit", "Bayesian Optimization",
        "Reinforcement Learning", "Exploration-Exploitation", "Regret Bounds",
        "UCB", "Epsilon-Greedy", "Softmax Selection", "Contextual Bandits",
        "Neural Networks", "Deep Learning", "Policy Gradient", "Q-Learning",
        "SARSA", "Actor-Critic", "PPO", "TRPO", "A3C", "DQN"
    ]

    for i, concept in enumerate(concepts):
        G.add_node(concept, type='concept', importance=1.0 - (i * 0.02))

    # Relationships (15 edges)
    edges = [
        ("Thompson Sampling", "Multi-Armed Bandit", "IS_A"),
        ("Thompson Sampling", "Bayesian Optimization", "USES"),
        ("Thompson Sampling", "Exploration-Exploitation", "ADDRESSES"),
        ("Multi-Armed Bandit", "Reinforcement Learning", "IS_A"),
        ("UCB", "Multi-Armed Bandit", "IS_A"),
        ("Epsilon-Greedy", "Multi-Armed Bandit", "IS_A"),
        ("Contextual Bandits", "Multi-Armed Bandit", "EXTENDS"),
        ("Deep Learning", "Neural Networks", "USES"),
        ("Policy Gradient", "Reinforcement Learning", "IS_A"),
        ("PPO", "Policy Gradient", "IS_A"),
        ("TRPO", "Policy Gradient", "IS_A"),
        ("A3C", "Actor-Critic", "IS_A"),
        ("DQN", "Q-Learning", "EXTENDS"),
        ("Q-Learning", "Reinforcement Learning", "IS_A"),
        ("Actor-Critic", "Reinforcement Learning", "IS_A"),
    ]

    for source, target, rel in edges:
        G.add_edge(source, target, type=rel, weight=1.0)

    return G


def main():
    """Run simplified visual compression demo."""

    print("=" * 80)
    print("Visual Compression Demo (Simplified)")
    print("=" * 80)
    print()

    # =====================================================================
    # Demo 1: Knowledge Graph Compression
    # =====================================================================
    print("Demo 1: Knowledge Graph Compression")
    print("-" * 80)
    graph = create_knowledge_graph()
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    print("Compressing...")
    image, metrics = compress_to_visual(graph, compression_type=CompressionType.KNOWLEDGE_GRAPH)

    print(f"Original tokens: {metrics.original_tokens:,}")
    print(f"Visual tokens: {metrics.visual_tokens:,}")
    print(f"Compression: {metrics.compression_ratio:.1f}x")
    print(f"Tokens saved: {metrics.original_tokens - metrics.visual_tokens:,}")
    print()

    # =====================================================================
    # Demo 2: Table Compression
    # =====================================================================
    print("Demo 2: Table Compression")
    print("-" * 80)
    table_data = {
        'headers': ["Model", "Accuracy", "Latency", "Memory", "Training", "Params"],
        'rows': [
            ["GPT-4", "95.2%", "450ms", "18GB", "21d", "1.76T"],
            ["GPT-3.5", "89.7%", "120ms", "4GB", "14d", "175B"],
            ["BERT-L", "87.3%", "80ms", "1.3GB", "4d", "340M"],
            ["BERT-B", "82.1%", "35ms", "440MB", "11h", "110M"],
            ["DistilB", "79.5%", "18ms", "250MB", "3h", "66M"],
            ["RoBERTa", "88.9%", "95ms", "1.4GB", "5d", "355M"],
            ["ALBERT", "86.4%", "60ms", "890MB", "2d", "235M"],
            ["ELECTRA", "85.8%", "45ms", "670MB", "1d", "110M"],
            ["T5-L", "91.2%", "380ms", "3GB", "8d", "770M"],
            ["T5-B", "85.6%", "140ms", "890MB", "2d", "220M"],
        ]
    }
    print(f"Table: {len(table_data['headers'])} columns, {len(table_data['rows'])} rows")

    print("Compressing...")
    image, metrics = compress_to_visual(table_data, compression_type=CompressionType.TABLE)

    print(f"Original tokens: {metrics.original_tokens:,}")
    print(f"Visual tokens: {metrics.visual_tokens:,}")
    print(f"Compression: {metrics.compression_ratio:.1f}x")
    print(f"Tokens saved: {metrics.original_tokens - metrics.visual_tokens:,}")
    print()

    # =====================================================================
    # Demo 3: Code Compression
    # =====================================================================
    print("Demo 3: Code Compression")
    print("-" * 80)
    code = '''class ThompsonSamplingBandit:
    """Thompson Sampling for multi-armed bandits."""

    def __init__(self, n_arms: int, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms) * prior_alpha
        self.beta = np.ones(n_arms) * prior_beta
        self.counts = np.zeros(n_arms)
        self.total_reward = 0.0

    def select_arm(self) -> int:
        """Select arm by sampling from Beta posteriors."""
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """Update posterior based on observed reward."""
        self.counts[arm] += 1
        self.total_reward += reward
        if reward > 0:
            self.alpha[arm] += reward
        else:
            self.beta[arm] += (1 - reward)

    def get_stats(self) -> Dict[str, Any]:
        """Get bandit statistics."""
        return {
            'total_pulls': int(sum(self.counts)),
            'total_reward': float(self.total_reward),
            'arm_counts': self.counts.tolist(),
            'expected_rewards': (self.alpha / (self.alpha + self.beta)).tolist()
        }'''
    print(f"Code: {len(code.split(chr(10)))} lines")

    print("Compressing...")
    image, metrics = compress_to_visual(code, compression_type=CompressionType.CODE)

    print(f"Original tokens: {metrics.original_tokens:,}")
    print(f"Visual tokens: {metrics.visual_tokens:,}")
    print(f"Compression: {metrics.compression_ratio:.1f}x")
    print(f"Tokens saved: {metrics.original_tokens - metrics.visual_tokens:,}")
    print()

    print("=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print()
    print("Visual compression working correctly.")
    print("For full integration demo, run: python demos/demo_multimodal_memory.py")


if __name__ == "__main__":
    main()
