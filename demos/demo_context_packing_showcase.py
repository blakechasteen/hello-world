#!/usr/bin/env python3
"""
HoloLoom Context Packing Showcase
==================================
Demonstrates physics-inspired context compression that achieves 40-90% token
savings while preserving information density.

Three-stage pipeline:
  1. Beta Wave Activation Spreading - neuroscience-inspired propagation
  2. Multi-Signal Importance Scoring - 7 weighted signals
  3. Matryoshka-Aware Compression   - multi-scale embedding assignment

Run:
    python demos/demo_context_packing_showcase.py

No heavy dependencies required (numpy only).
"""

import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class KnowledgeGraph:
    """Lightweight directed multigraph for demonstration."""

    _adj: Dict[str, List[Tuple[str, str, float]]] = field(default_factory=lambda: defaultdict(list))
    _rev: Dict[str, List[Tuple[str, str, float]]] = field(default_factory=lambda: defaultdict(list))
    _nodes: Set[str] = field(default_factory=set)
    _contents: Dict[str, str] = field(default_factory=dict)

    def add_edge(self, edge: Edge):
        self._adj[edge.source].append((edge.target, edge.relation, edge.weight))
        self._rev[edge.target].append((edge.source, edge.relation, edge.weight))
        self._nodes.update([edge.source, edge.target])

    def add_content(self, node: str, content: str):
        self._contents[node] = content
        self._nodes.add(node)

    @property
    def nodes(self) -> List[str]:
        return sorted(self._nodes)

    def neighbors(self, node: str) -> List[Tuple[str, str, float]]:
        return self._adj.get(node, [])

    def degree(self, node: str) -> int:
        return len(self._adj.get(node, [])) + len(self._rev.get(node, []))

    def content(self, node: str) -> str:
        return self._contents.get(node, node)


# ---------------------------------------------------------------------------
# Edge type weights (semantic importance)
# ---------------------------------------------------------------------------

EDGE_WEIGHTS = {
    "IS_A": 1.0,
    "USES": 0.9,
    "PART_OF": 0.85,
    "LEADS_TO": 0.8,
    "MENTIONS": 0.5,
    "IN_TIME": 0.4,
    "OCCURRED_AT": 0.3,
}


# ---------------------------------------------------------------------------
# Demo 1: Beta Wave Activation Spreading
# ---------------------------------------------------------------------------

def demo_beta_wave_spreading(graph: KnowledgeGraph):
    """
    Neuroscience-inspired activation propagation.

    Beta waves (12-30 Hz) represent focused attention in the brain.
    HoloLoom models this as exponential decay over graph hops:

        activation(node, hop) = initial * decay^hop * edge_weight

    This naturally prioritises closely related concepts while allowing
    distant but strongly-connected ideas to survive.
    """
    print("=" * 64)
    print("Demo 1: Beta Wave Activation Spreading")
    print("=" * 64)
    print()
    print("  Beta waves (12-30 Hz) represent focused attention.")
    print("  Activation decays exponentially across graph hops:")
    print("    A(node, hop) = initial * decay^hop * edge_weight")
    print()

    source_nodes = ["thompson_sampling"]
    initial_activation = 1.0
    decay_rate = 0.7
    max_hops = 4

    # BFS-based spreading
    activation: Dict[str, float] = {}
    frontier = [(s, 0) for s in source_nodes]
    visited: Set[str] = set()

    for src in source_nodes:
        activation[src] = initial_activation

    while frontier:
        node, hop = frontier.pop(0)
        if node in visited or hop > max_hops:
            continue
        visited.add(node)

        for neighbor, relation, weight in graph.neighbors(node):
            edge_w = EDGE_WEIGHTS.get(relation, 0.5) * weight
            prop = activation[node] * decay_rate * edge_w
            if prop > 0.01:  # threshold
                if neighbor not in activation or prop > activation[neighbor]:
                    activation[neighbor] = prop
                frontier.append((neighbor, hop + 1))

    # Display
    sorted_act = sorted(activation.items(), key=lambda x: -x[1])
    print(f"  Source: {source_nodes[0]}")
    print(f"  Decay rate: {decay_rate}, Max hops: {max_hops}")
    print()
    print(f"  {'Node':<25} {'Activation':>10}  Visualisation")
    print(f"  {'─' * 25} {'─' * 10}  {'─' * 25}")
    for node, act in sorted_act:
        bar_len = int(act * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"  {node:<25} {act:>10.4f}  {bar}")

    print()
    print(f"  Activated {len(activation)} of {len(graph.nodes)} nodes")
    print(f"  Highest hop-2+ activation: ", end="")
    hop2 = [(n, a) for n, a in sorted_act if n not in source_nodes]
    if hop2:
        print(f"{hop2[0][0]} ({hop2[0][1]:.4f})")
    print()
    return activation


# ---------------------------------------------------------------------------
# Demo 2: Multi-Signal Importance Scoring
# ---------------------------------------------------------------------------

@dataclass
class ImportanceSignals:
    recency: float = 0.0
    relevance: float = 0.0
    centrality: float = 0.0
    access_frequency: float = 0.0
    confidence: float = 0.0
    heat: float = 0.0
    information_content: float = 0.0


# Default weights (sum to 1.0)
SIGNAL_WEIGHTS = {
    "recency": 0.15,
    "relevance": 0.20,
    "centrality": 0.12,
    "access_frequency": 0.08,
    "confidence": 0.12,
    "heat": 0.08,
    "information_content": 0.25,
}


def compute_signals(
    node: str,
    query: str,
    graph: KnowledgeGraph,
    activation: Dict[str, float],
) -> ImportanceSignals:
    """Compute 7 importance signals for a single node."""

    # Recency: exponential decay from simulated timestamps
    rng = random.Random(hash(node) + 1)
    days_old = rng.uniform(0, 90)
    recency = math.exp(-0.02 * days_old)

    # Relevance: word overlap as proxy for semantic similarity
    node_words = set(graph.content(node).lower().split())
    query_words = set(query.lower().split())
    overlap = len(node_words & query_words)
    relevance = min(overlap / max(len(query_words), 1), 1.0)

    # Centrality: degree-based (proxy for PageRank)
    max_degree = max(graph.degree(n) for n in graph.nodes) or 1
    centrality = graph.degree(node) / max_degree

    # Access frequency: simulated log-scaled access count
    access_count = rng.randint(1, 100)
    access_frequency = math.log1p(access_count) / math.log1p(100)

    # Confidence: simulated historical confidence
    confidence = rng.uniform(0.5, 1.0)

    # Heat: from activation spreading
    heat = activation.get(node, 0.0)

    # Information content: mutual information proxy
    content_length = len(graph.content(node))
    unique_terms = len(set(graph.content(node).lower().split()))
    information_content = min(unique_terms / 20, 1.0) * (0.5 + 0.5 * relevance)

    return ImportanceSignals(
        recency=recency,
        relevance=relevance,
        centrality=centrality,
        access_frequency=access_frequency,
        confidence=confidence,
        heat=heat,
        information_content=information_content,
    )


def aggregate_importance(signals: ImportanceSignals) -> float:
    """Weighted sum of 7 signals."""
    return (
        SIGNAL_WEIGHTS["recency"] * signals.recency
        + SIGNAL_WEIGHTS["relevance"] * signals.relevance
        + SIGNAL_WEIGHTS["centrality"] * signals.centrality
        + SIGNAL_WEIGHTS["access_frequency"] * signals.access_frequency
        + SIGNAL_WEIGHTS["confidence"] * signals.confidence
        + SIGNAL_WEIGHTS["heat"] * signals.heat
        + SIGNAL_WEIGHTS["information_content"] * signals.information_content
    )


def demo_importance_scoring(graph: KnowledgeGraph, activation: Dict[str, float]):
    """
    7 weighted signals blended into a single importance score.

    Signals: recency, relevance, centrality, access_frequency,
             confidence, heat, information_content
    """
    print("=" * 64)
    print("Demo 2: Multi-Signal Importance Scoring")
    print("=" * 64)
    print()
    print("  7 signals blended per node (weights sum to 1.0):")
    print()
    for name, w in SIGNAL_WEIGHTS.items():
        bar = "█" * int(w * 40)
        print(f"    {name:<22} {w:.2f}  {bar}")
    print()

    query = "What is Thompson Sampling?"
    scores: Dict[str, Tuple[float, ImportanceSignals]] = {}

    for node in graph.nodes:
        signals = compute_signals(node, query, graph, activation)
        score = aggregate_importance(signals)
        scores[node] = (score, signals)

    ranked = sorted(scores.items(), key=lambda x: -x[1][0])

    print(f"  Query: \"{query}\"")
    print()
    print(f"  {'Rank':<5} {'Node':<25} {'Score':>6}  {'Rec':>5} {'Rel':>5} {'Cen':>5} {'Frq':>5} {'Con':>5} {'Hea':>5} {'MI':>5}")
    print(f"  {'─' * 5} {'─' * 25} {'─' * 6}  {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5} {'─' * 5}")

    for rank, (node, (score, sig)) in enumerate(ranked, 1):
        print(
            f"  {rank:<5} {node:<25} {score:>6.3f}  "
            f"{sig.recency:>5.2f} {sig.relevance:>5.2f} {sig.centrality:>5.2f} "
            f"{sig.access_frequency:>5.2f} {sig.confidence:>5.2f} {sig.heat:>5.2f} "
            f"{sig.information_content:>5.2f}"
        )

    print()
    return {n: s for n, (s, _) in scores.items()}


# ---------------------------------------------------------------------------
# Demo 3: Matryoshka-Aware Compression
# ---------------------------------------------------------------------------

MATRYOSHKA_SCALES = {
    "high": {"min_importance": 0.75, "dims": 384, "tokens": 100, "label": "Full detail"},
    "medium": {"min_importance": 0.50, "dims": 256, "tokens": 67, "label": "Moderate"},
    "low": {"min_importance": 0.25, "dims": 128, "tokens": 33, "label": "Minimal"},
    "dropped": {"min_importance": 0.0, "dims": 0, "tokens": 0, "label": "Dropped"},
}


def assign_matryoshka_scale(importance: float) -> str:
    if importance >= 0.75:
        return "high"
    elif importance >= 0.50:
        return "medium"
    elif importance >= 0.25:
        return "low"
    else:
        return "dropped"


def demo_matryoshka_compression(importance_scores: Dict[str, float]):
    """
    Multi-scale embedding assignment based on importance.

    High-importance nodes keep full 384D embeddings.
    Medium nodes compress to 256D.  Low nodes to 128D.
    Very low importance nodes are dropped entirely.

    This leverages the Matryoshka "prefix property": the first k
    dimensions of a 384D embedding contain the k-dimensional
    representation, enabling zero-copy slicing.
    """
    print("=" * 64)
    print("Demo 3: Matryoshka-Aware Compression")
    print("=" * 64)
    print()
    print("  Scale assignment based on importance:")
    print()
    print(f"    {'Scale':<10} {'Importance':>12} {'Dims':>6} {'Tokens':>8}  Description")
    print(f"    {'─' * 10} {'─' * 12} {'─' * 6} {'─' * 8}  {'─' * 20}")
    for scale, info in MATRYOSHKA_SCALES.items():
        min_imp = info["min_importance"]
        dims = info["dims"]
        tokens = info["tokens"]
        label = info["label"]
        print(f"    {scale:<10} {'≥' + f'{min_imp:.2f}':>12} {dims:>6} {tokens:>8}  {label}")
    print()

    assignments: Dict[str, str] = {}
    scale_counts = defaultdict(int)
    total_tokens_full = 0
    total_tokens_compressed = 0

    ranked = sorted(importance_scores.items(), key=lambda x: -x[1])

    for node, imp in ranked:
        scale = assign_matryoshka_scale(imp)
        assignments[node] = scale
        scale_counts[scale] += 1
        total_tokens_full += 100  # Full 384D for everything
        total_tokens_compressed += MATRYOSHKA_SCALES[scale]["tokens"]

    print(f"  {'Node':<25} {'Importance':>10} {'Scale':>8} {'Dims':>6} {'Tokens':>8}")
    print(f"  {'─' * 25} {'─' * 10} {'─' * 8} {'─' * 6} {'─' * 8}")
    for node, imp in ranked:
        scale = assignments[node]
        info = MATRYOSHKA_SCALES[scale]
        marker = " *" if scale == "dropped" else ""
        print(f"  {node:<25} {imp:>10.3f} {scale:>8} {info['dims']:>6} {info['tokens']:>8}{marker}")

    savings = (1 - total_tokens_compressed / max(total_tokens_full, 1)) * 100
    print()
    print(f"  Scale distribution:")
    for scale in ["high", "medium", "low", "dropped"]:
        count = scale_counts[scale]
        bar = "█" * count + " " * (20 - count)
        print(f"    {scale:<10} {count:>3} nodes  {bar}")
    print()
    print(f"  Token budget: {total_tokens_full} (uncompressed) -> {total_tokens_compressed} (compressed)")
    print(f"  Token savings: {savings:.1f}%")
    print()
    return assignments, total_tokens_compressed


# ---------------------------------------------------------------------------
# Demo 4: Full Context Packing Pipeline
# ---------------------------------------------------------------------------

def demo_full_pipeline(graph: KnowledgeGraph):
    """
    End-to-end context packing: spread -> score -> compress.

    Runs the complete three-stage pipeline and shows how each stage
    filters and prioritises information.
    """
    print("=" * 64)
    print("Demo 4: Full Context Packing Pipeline")
    print("=" * 64)
    print()

    query = "Explain Thompson Sampling and Bayesian exploration"
    target_tokens = 400
    source_nodes = ["thompson_sampling", "bayesian"]

    print(f"  Query: \"{query}\"")
    print(f"  Target token budget: {target_tokens}")
    print(f"  Candidate nodes: {len(graph.nodes)}")
    print()

    # Stage 1: Activation spreading
    t0 = time.perf_counter()
    activation: Dict[str, float] = {}
    frontier = [(s, 0) for s in source_nodes]
    visited: Set[str] = set()
    for src in source_nodes:
        activation[src] = 1.0

    while frontier:
        node, hop = frontier.pop(0)
        if node in visited or hop > 3:
            continue
        visited.add(node)
        for neighbor, relation, weight in graph.neighbors(node):
            edge_w = EDGE_WEIGHTS.get(relation, 0.5) * weight
            prop = activation[node] * 0.7 * edge_w
            if prop > 0.01:
                if neighbor not in activation or prop > activation[neighbor]:
                    activation[neighbor] = prop
                frontier.append((neighbor, hop + 1))
    t1 = time.perf_counter()
    spread_ms = (t1 - t0) * 1000

    print(f"  Stage 1: Beta Wave Spreading ({spread_ms:.2f}ms)")
    print(f"    Activated: {len(activation)} / {len(graph.nodes)} nodes")
    print()

    # Stage 2: Importance scoring
    t0 = time.perf_counter()
    scores: Dict[str, float] = {}
    for node in graph.nodes:
        signals = compute_signals(node, query, graph, activation)
        scores[node] = aggregate_importance(signals)
    t1 = time.perf_counter()
    score_ms = (t1 - t0) * 1000

    print(f"  Stage 2: Importance Scoring ({score_ms:.2f}ms)")
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top3 = ranked[:3]
    print(f"    Top 3: {', '.join(f'{n} ({s:.3f})' for n, s in top3)}")
    print()

    # Stage 3: Matryoshka compression with budget
    t0 = time.perf_counter()
    kept_nodes = []
    scale_map = {}
    tokens_used = 0

    for node, imp in ranked:
        scale = assign_matryoshka_scale(imp)
        tok = MATRYOSHKA_SCALES[scale]["tokens"]
        if scale == "dropped":
            continue
        if tokens_used + tok > target_tokens:
            break
        kept_nodes.append(node)
        scale_map[node] = scale
        tokens_used += tok
    t1 = time.perf_counter()
    compress_ms = (t1 - t0) * 1000

    original_tokens = len(graph.nodes) * 100
    savings = (1 - tokens_used / max(original_tokens, 1)) * 100

    print(f"  Stage 3: Matryoshka Compression ({compress_ms:.2f}ms)")
    print(f"    Kept: {len(kept_nodes)} / {len(graph.nodes)} nodes")
    print(f"    Tokens: {original_tokens} -> {tokens_used} ({savings:.1f}% savings)")
    print()

    # Summary
    total_ms = spread_ms + score_ms + compress_ms
    print(f"  Pipeline Summary:")
    print(f"  ┌───────────────────┬───────────┬────────────┐")
    print(f"  │ Stage             │  Time     │ Nodes      │")
    print(f"  ├───────────────────┼───────────┼────────────┤")
    print(f"  │ 1. Spreading      │ {spread_ms:>7.2f}ms │ {len(graph.nodes):>3} -> {len(activation):>3}  │")
    print(f"  │ 2. Scoring        │ {score_ms:>7.2f}ms │ {len(activation):>3} scored │")
    print(f"  │ 3. Compression    │ {compress_ms:>7.2f}ms │ {len(activation):>3} -> {len(kept_nodes):>3}  │")
    print(f"  ├───────────────────┼───────────┼────────────┤")
    print(f"  │ Total             │ {total_ms:>7.2f}ms │ {original_tokens:>4} -> {tokens_used:>3} tok │")
    print(f"  └───────────────────┴───────────┴────────────┘")
    print()

    print(f"  Final context ({len(kept_nodes)} nodes, {tokens_used} tokens):")
    for node in kept_nodes:
        scale = scale_map[node]
        dims = MATRYOSHKA_SCALES[scale]["dims"]
        print(f"    [{dims}D] {node}: {graph.content(node)[:60]}...")
    print()


# ---------------------------------------------------------------------------
# Demo 5: Compression Preset Comparison
# ---------------------------------------------------------------------------

def demo_preset_comparison(graph: KnowledgeGraph):
    """
    Compare AGGRESSIVE, BALANCED, CONSERVATIVE, and RESEARCH presets
    across the same knowledge graph.
    """
    print("=" * 64)
    print("Demo 5: Compression Preset Comparison")
    print("=" * 64)
    print()

    presets = {
        "AGGRESSIVE":   {"keep_ratio": 0.30, "description": "Tight budgets, fast queries"},
        "BALANCED":     {"keep_ratio": 0.50, "description": "General use (default)"},
        "CONSERVATIVE": {"keep_ratio": 0.70, "description": "Quality-critical queries"},
        "RESEARCH":     {"keep_ratio": 0.90, "description": "Full exploration mode"},
    }

    query = "What is Thompson Sampling?"
    all_nodes = graph.nodes

    # Score all nodes once
    activation = {"thompson_sampling": 1.0, "bayesian": 0.7, "beta_dist": 0.49}
    scores: Dict[str, float] = {}
    for node in all_nodes:
        signals = compute_signals(node, query, graph, activation)
        scores[node] = aggregate_importance(signals)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    total_uncompressed = len(all_nodes) * 100

    print(f"  Query: \"{query}\"")
    print(f"  Total nodes: {len(all_nodes)}")
    print(f"  Uncompressed tokens: {total_uncompressed}")
    print()

    print(f"  {'Preset':<15} {'Keep':>5} {'Nodes':>6} {'Tokens':>7} {'Savings':>8}  Description")
    print(f"  {'─' * 15} {'─' * 5} {'─' * 6} {'─' * 7} {'─' * 8}  {'─' * 30}")

    for name, preset in presets.items():
        keep_n = max(1, int(len(all_nodes) * preset["keep_ratio"]))
        kept = ranked[:keep_n]
        tokens = sum(
            MATRYOSHKA_SCALES[assign_matryoshka_scale(s)]["tokens"]
            for _, s in kept
        )
        savings = (1 - tokens / max(total_uncompressed, 1)) * 100
        print(
            f"  {name:<15} {preset['keep_ratio']:>4.0%} {keep_n:>6} {tokens:>7} {savings:>7.1f}%  {preset['description']}"
        )

    print()
    print("  Token savings visualised:")
    print()
    for name, preset in presets.items():
        keep_n = max(1, int(len(all_nodes) * preset["keep_ratio"]))
        kept = ranked[:keep_n]
        tokens = sum(
            MATRYOSHKA_SCALES[assign_matryoshka_scale(s)]["tokens"]
            for _, s in kept
        )
        savings = (1 - tokens / max(total_uncompressed, 1)) * 100
        bar_saved = "░" * int(savings / 2)
        bar_kept = "█" * (50 - int(savings / 2))
        print(f"    {name:<15} {bar_kept}{bar_saved} {savings:.0f}%")
    print()


# ---------------------------------------------------------------------------
# Demo 6: Information Budget Packing (MI-Aware)
# ---------------------------------------------------------------------------

def demo_information_budget(graph: KnowledgeGraph):
    """
    Tishby's Information Bottleneck applied to context packing.

    Maximise I(Context; Query) subject to a token budget.
    Low-MI nodes are aggressively compressed or dropped;
    high-MI nodes keep full resolution.
    """
    print("=" * 64)
    print("Demo 6: Information Budget Packing (MI-Aware)")
    print("=" * 64)
    print()

    query = "Explain Thompson Sampling for decision making"
    query_terms = set(query.lower().split())
    budget_bits = 5.0

    print(f"  Query: \"{query}\"")
    print(f"  Information budget: {budget_bits:.1f} bits")
    print()

    # Compute MI proxy for each node
    mi_scores: Dict[str, float] = {}
    for node in graph.nodes:
        content = graph.content(node).lower()
        content_terms = set(content.split())

        # Mutual information proxy: term overlap + unique info
        overlap = len(query_terms & content_terms)
        unique = len(content_terms - query_terms)
        total = len(content_terms) or 1

        # MI = overlap contribution + diversity bonus
        mi = (overlap / max(len(query_terms), 1)) * 0.7 + min(unique / total, 1.0) * 0.3
        mi_scores[node] = mi

    # Entropy-aware weighting
    print("  MI scores with entropy weighting:")
    print()
    print(f"    {'Node':<25} {'Raw MI':>7} {'Entropy':>8} {'Weighted':>9} {'Scale':>8}")
    print(f"    {'─' * 25} {'─' * 7} {'─' * 8} {'─' * 9} {'─' * 8}")

    weighted_mi: Dict[str, float] = {}
    for node in sorted(graph.nodes):
        mi = mi_scores[node]
        # Entropy: high entropy (uncertain) = penalty
        content_terms = graph.content(node).lower().split()
        term_freq = defaultdict(int)
        for t in content_terms:
            term_freq[t] += 1
        total = len(content_terms) or 1
        entropy = -sum((c / total) * math.log2(c / total) for c in term_freq.values() if c > 0)
        entropy_weight = 1.0 / (1.0 + entropy)

        final = 0.7 * mi + 0.3 * (mi * entropy_weight)
        weighted_mi[node] = final

        scale = assign_matryoshka_scale(final)
        dims = MATRYOSHKA_SCALES[scale]["dims"]
        print(f"    {node:<25} {mi:>7.3f} {entropy:>8.3f} {final:>9.4f} {dims:>5}D")

    # Pack under budget
    ranked = sorted(weighted_mi.items(), key=lambda x: -x[1])
    cumulative_mi = 0.0
    selected = []

    print()
    print(f"  Budget packing (greedy, {budget_bits:.1f} bits):")
    print()

    for node, mi in ranked:
        if cumulative_mi + mi > budget_bits and selected:
            print(f"    [budget exhausted at {cumulative_mi:.3f} bits]")
            break
        selected.append(node)
        cumulative_mi += mi
        bar = "█" * int(cumulative_mi / budget_bits * 30)
        print(f"    + {node:<23} MI={mi:.3f}  cumulative={cumulative_mi:.3f}  {bar}")

    print()
    print(f"  Selected: {len(selected)} / {len(graph.nodes)} nodes")
    print(f"  Information preserved: {cumulative_mi:.3f} / {sum(weighted_mi.values()):.3f} bits")
    print(f"  Efficiency: {cumulative_mi / max(len(selected), 1):.3f} bits/node (avg)")
    print()


# ---------------------------------------------------------------------------
# Knowledge graph construction
# ---------------------------------------------------------------------------

def build_knowledge_graph() -> KnowledgeGraph:
    """Build a demonstration knowledge graph about ML concepts."""
    graph = KnowledgeGraph()

    edges = [
        Edge("thompson_sampling", "bandit_algorithm", "IS_A"),
        Edge("thompson_sampling", "bayesian", "USES"),
        Edge("thompson_sampling", "beta_dist", "USES"),
        Edge("thompson_sampling", "exploration", "LEADS_TO"),
        Edge("thompson_sampling", "exploitation", "LEADS_TO"),
        Edge("bandit_algorithm", "decision_making", "PART_OF"),
        Edge("bandit_algorithm", "reward", "USES"),
        Edge("bayesian", "prior", "USES"),
        Edge("bayesian", "posterior", "LEADS_TO"),
        Edge("beta_dist", "posterior", "PART_OF"),
        Edge("exploration", "uncertainty", "USES"),
        Edge("exploitation", "reward", "USES"),
        Edge("decision_making", "policy", "USES"),
        Edge("policy", "reinforcement_learning", "PART_OF"),
        Edge("reinforcement_learning", "reward", "USES"),
        Edge("reinforcement_learning", "exploration", "USES"),
        Edge("prior", "belief", "IS_A"),
        Edge("posterior", "belief", "IS_A"),
        Edge("ucb", "bandit_algorithm", "IS_A"),
        Edge("ucb", "confidence_bound", "USES"),
        Edge("epsilon_greedy", "bandit_algorithm", "IS_A"),
        Edge("epsilon_greedy", "exploration", "USES"),
    ]

    contents = {
        "thompson_sampling": "Thompson Sampling is a Bayesian algorithm that balances exploration and exploitation by sampling from posterior distributions",
        "bandit_algorithm": "Multi-armed bandit algorithms solve the exploration-exploitation dilemma in sequential decision making",
        "bayesian": "Bayesian inference updates beliefs about parameters using Bayes theorem with prior and likelihood",
        "beta_dist": "The beta distribution Beta(alpha, beta) is the conjugate prior for Bernoulli observations",
        "exploration": "Exploration tries new actions to gather information about their reward distributions",
        "exploitation": "Exploitation selects the action believed to have the highest expected reward",
        "decision_making": "Decision making under uncertainty requires balancing information gathering and reward maximisation",
        "reward": "Reward signals indicate the quality of an action taken in a given state",
        "prior": "A prior distribution encodes beliefs about parameters before observing data",
        "posterior": "The posterior distribution represents updated beliefs after incorporating observed data",
        "uncertainty": "Uncertainty quantifies how much we don't know about the environment or parameters",
        "policy": "A policy maps states to actions and defines the agent's behaviour in an environment",
        "reinforcement_learning": "Reinforcement learning trains agents to make sequential decisions to maximise cumulative reward",
        "belief": "A belief state represents the agent's probabilistic knowledge about the world",
        "ucb": "Upper Confidence Bound selects arms based on optimistic estimates of expected reward",
        "confidence_bound": "Confidence bounds provide intervals that contain the true parameter with high probability",
        "epsilon_greedy": "Epsilon-greedy explores uniformly at random with probability epsilon and exploits otherwise",
    }

    for edge in edges:
        graph.add_edge(edge)
    for node, content in contents.items():
        graph.add_content(node, content)

    return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║       HoloLoom Context Packing Showcase                   ║")
    print("  ║       Physics-Inspired Context Compression                ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print()

    graph = build_knowledge_graph()
    print(f"  Knowledge graph: {len(graph.nodes)} nodes, "
          f"{sum(len(v) for v in graph._adj.values())} edges")
    print()

    # Demo 1: Beta wave activation
    activation = demo_beta_wave_spreading(graph)

    # Demo 2: Multi-signal importance
    importance_scores = demo_importance_scoring(graph, activation)

    # Demo 3: Matryoshka compression
    demo_matryoshka_compression(importance_scores)

    # Demo 4: Full pipeline
    demo_full_pipeline(graph)

    # Demo 5: Preset comparison
    demo_preset_comparison(graph)

    # Demo 6: MI-aware budget packing
    demo_information_budget(graph)

    print("  " + "═" * 62)
    print("  All demonstrations complete.")
    print()


if __name__ == "__main__":
    main()
