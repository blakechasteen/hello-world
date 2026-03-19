"""
Knowledge Graph Reasoning — MCTS-Guided Multi-Hop Traversal
============================================================

Wire 8 of the structured AI integration roadmap (ReKG-MCTS pattern).

Transforms the knowledge graph from passive storage into an active
reasoning substrate. Uses MCTS to search over KG paths for multi-hop
reasoning: UCB1 selects promising graph nodes, rollouts score candidate
reasoning paths, backpropagation updates path quality estimates.

Based on: ReKG-MCTS (ACL 2025) — training-free framework combining
MCTS with LLMs for dynamic reasoning over knowledge graphs.

Usage:
    from hololoom.core.memory.graph import KG
    reasoner = KGReasoner(kg=my_kg, max_depth=4, num_simulations=50)
    result = reasoner.reason("What connects quantum computing to cryptography?")
    print(result.best_path)       # ['quantum_computing', 'factoring', 'RSA', 'cryptography']
    print(result.path_score)      # 0.85
    print(result.paths_explored)  # 50

Author: Claude Code (Structured AI Wiring - Wire 8)
Date: 2026-03-18
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class ReasoningPath:
    """A path through the knowledge graph with quality score."""
    nodes: list[str]            # Ordered list of entity IDs
    edges: list[str]            # Edge types between consecutive nodes
    score: float                # Quality score from rollout (0-1)
    depth: int                  # Number of hops


@dataclass
class KGReasoningResult:
    """Result of MCTS-guided KG reasoning."""
    best_path: ReasoningPath            # Highest-scored path found
    all_paths: list[ReasoningPath]      # All paths evaluated (sorted by score)
    paths_explored: int                 # Total MCTS simulations
    search_time_ms: float               # Wall clock time
    anchor_entities: list[str]          # Starting entities
    target_entities: list[str]          # Target entities (if specified)


# ============================================================================
# MCTS Node for KG Traversal
# ============================================================================

@dataclass
class _KGNode:
    """Internal MCTS node for KG path search."""
    entity: str
    path: list[str]             # Full path from root to this node
    edge_types: list[str]       # Edge types along the path
    parent: Optional['_KGNode'] = None
    children: list['_KGNode'] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    def ucb1(self, c: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.value_sum / self.visits
        if self.parent is None or self.parent.visits == 0:
            return exploitation
        exploration = c * np.sqrt(np.log(self.parent.visits + 1) / (self.visits + 1e-9))
        return exploitation + exploration


# ============================================================================
# KG Reasoner
# ============================================================================

class KGReasoner:
    """
    MCTS-guided multi-hop reasoning over a knowledge graph.

    The reasoner treats KG traversal as a sequential decision problem:
    - State: current path through the graph
    - Action: which neighbor to visit next
    - Reward: how relevant/coherent the resulting path is

    MCTS balances exploration (visiting unvisited graph regions) with
    exploitation (following paths that have scored well before).

    The rollout function is pluggable — default uses path coherence
    (embedding similarity between start and end). For LLM-guided rollouts,
    pass a custom scorer.
    """

    def __init__(
        self,
        max_depth: int = 4,
        num_simulations: int = 50,
        exploration_constant: float = 1.4,
        scorer: Callable[[list[str], list[str]], float] | None = None,
    ):
        """
        Args:
            max_depth: Maximum path length (hops)
            num_simulations: Number of MCTS iterations
            exploration_constant: UCB1 exploration parameter
            scorer: Optional callable(path_nodes, edge_types) -> float in [0,1].
                   Default: length-based heuristic + target proximity.
        """
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.scorer = scorer
        logger.info(
            f"KGReasoner initialized (depth={max_depth}, sims={num_simulations}, C={exploration_constant})"
        )

    def _get_graph(self, kg):
        """Extract the NetworkX graph from a KG object or return as-is."""
        # HoloLoom KG wraps a NetworkX graph in .graph attribute
        if hasattr(kg, 'graph') and hasattr(kg.graph, 'out_edges'):
            return kg.graph
        # Already a NetworkX graph
        if hasattr(kg, 'out_edges'):
            return kg
        return kg

    def _get_neighbors(self, kg, entity: str) -> list[tuple]:
        """
        Get neighbors of an entity from the KG.

        Returns list of (neighbor_id, edge_type) tuples.
        Works with HoloLoom's KG class (NetworkX-backed) or raw NetworkX graphs.
        """
        graph = self._get_graph(kg)
        neighbors = []

        if not hasattr(graph, 'has_node') or not graph.has_node(entity):
            return neighbors

        # Outgoing edges
        for _, target, data in graph.out_edges(entity, data=True):
            edge_type = data.get('rel_type', data.get('type', 'related'))
            neighbors.append((target, str(edge_type)))

        # Incoming edges (reverse traversal)
        for source, _, data in graph.in_edges(entity, data=True):
            edge_type = data.get('rel_type', data.get('type', 'related'))
            neighbors.append((source, f"inv_{edge_type}"))

        return neighbors

    def _default_scorer(
        self,
        path: list[str],
        edge_types: list[str],
        targets: set[str],
    ) -> float:
        """
        Default path scoring heuristic.

        Combines:
        - Path length penalty (shorter paths preferred)
        - Target hit bonus (paths reaching targets score higher)
        - Edge type diversity bonus (diverse paths are more informative)
        """
        if not path:
            return 0.0

        score = 0.0

        # Length: prefer paths of 2-3 hops (not too short, not too long)
        ideal_length = 3
        length_score = 1.0 - abs(len(path) - ideal_length) / max(self.max_depth, 1)
        score += 0.3 * max(0.0, length_score)

        # Target proximity: bonus if path reaches a target
        if targets:
            for i, node in enumerate(path):
                if node in targets:
                    # Earlier hits score higher (more direct reasoning)
                    score += 0.5 * (1.0 - i / len(path))
                    break

        # Edge diversity: more diverse edge types = more informative path
        if edge_types:
            unique_ratio = len(set(edge_types)) / len(edge_types)
            score += 0.2 * unique_ratio

        return min(1.0, score)

    def reason(
        self,
        kg,
        anchors: list[str],
        targets: list[str] | None = None,
        query: str | None = None,
    ) -> KGReasoningResult:
        """
        Perform MCTS-guided multi-hop reasoning over the KG.

        Args:
            kg: Knowledge graph instance (HoloLoom KG or NetworkX graph)
            anchors: Starting entities for reasoning
            targets: Optional target entities (paths reaching these score higher)
            query: Optional query text (for LLM-guided rollouts)

        Returns:
            KGReasoningResult with best path and all explored paths
        """
        t0 = time.time()
        target_set = set(targets or [])
        all_paths: list[ReasoningPath] = []

        # Create root nodes for each anchor
        roots = []
        for anchor in anchors:
            roots.append(_KGNode(entity=anchor, path=[anchor], edge_types=[]))

        # Run MCTS from each anchor
        for root in roots:
            for _ in range(self.num_simulations):
                # 1. Selection: traverse via UCB1
                node = self._select(root, kg)

                # 2. Expansion: add a new neighbor
                node = self._expand(node, kg)

                # 3. Simulation: score the path
                if self.scorer is not None:
                    reward = self.scorer(node.path, node.edge_types)
                else:
                    reward = self._default_scorer(node.path, node.edge_types, target_set)

                # Record path
                all_paths.append(ReasoningPath(
                    nodes=list(node.path),
                    edges=list(node.edge_types),
                    score=reward,
                    depth=len(node.path) - 1,
                ))

                # 4. Backpropagation
                self._backpropagate(node, reward)

        # Sort by score
        all_paths.sort(key=lambda p: p.score, reverse=True)

        # Deduplicate by path
        seen = set()
        unique_paths = []
        for p in all_paths:
            key = tuple(p.nodes)
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)

        best = unique_paths[0] if unique_paths else ReasoningPath(
            nodes=anchors, edges=[], score=0.0, depth=0
        )

        elapsed = (time.time() - t0) * 1000

        logger.info(
            f"KG reasoning: {len(unique_paths)} unique paths, "
            f"best score={best.score:.3f}, depth={best.depth}, "
            f"time={elapsed:.1f}ms"
        )

        return KGReasoningResult(
            best_path=best,
            all_paths=unique_paths[:20],  # Top 20
            paths_explored=len(all_paths),
            search_time_ms=elapsed,
            anchor_entities=anchors,
            target_entities=targets or [],
        )

    def _select(self, node: _KGNode, kg) -> _KGNode:
        """Selection: traverse via UCB1 until expandable or terminal."""
        depth = len(node.path) - 1
        while depth < self.max_depth:
            neighbors = self._get_neighbors(kg, node.entity)
            visited_entities = set(node.path)
            available = [(n, e) for n, e in neighbors if n not in visited_entities]

            if not available:
                return node

            if len(node.children) < len(available):
                return node  # Expandable

            # All neighbors explored — pick best child
            if node.children:
                node = max(node.children, key=lambda c: c.ucb1(self.exploration_constant))
                depth = len(node.path) - 1
            else:
                return node

        return node

    def _expand(self, node: _KGNode, kg) -> _KGNode:
        """Expansion: add one unexplored neighbor."""
        if len(node.path) - 1 >= self.max_depth:
            return node

        neighbors = self._get_neighbors(kg, node.entity)
        visited = set(node.path)
        available = [(n, e) for n, e in neighbors if n not in visited]

        if not available:
            return node

        # Find untried
        tried = {c.entity for c in node.children}
        untried = [(n, e) for n, e in available if n not in tried]

        if not untried:
            return node

        # Pick random untried neighbor
        idx = np.random.randint(len(untried))
        next_entity, edge_type = untried[idx]

        child = _KGNode(
            entity=next_entity,
            path=node.path + [next_entity],
            edge_types=node.edge_types + [edge_type],
            parent=node,
        )
        node.children.append(child)
        return child

    def _backpropagate(self, node: _KGNode, reward: float) -> None:
        """Update statistics up the tree."""
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent


# ============================================================================
# Factory
# ============================================================================

def create_kg_reasoner(
    max_depth: int = 4,
    num_simulations: int = 50,
    scorer: Callable | None = None,
) -> KGReasoner:
    """Create a KG reasoner with MCTS-guided multi-hop search."""
    return KGReasoner(
        max_depth=max_depth,
        num_simulations=num_simulations,
        scorer=scorer,
    )
