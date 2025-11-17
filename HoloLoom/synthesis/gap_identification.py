"""
DreamEngine Gap Identification
================================

Analyzes knowledge graph structure to find missing connections and suggest learning.

**Philosophy:** Humans naturally seek to fill knowledge gaps. DreamEngine identifies
these gaps and suggests relevant learning paths.

Author: HoloLoom Team
Date: 2025-11-17
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

from HoloLoom.synthesis.types import (
    KnowledgeGap,
    GapType
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GapIdentificationConfig:
    """Configuration for gap identification."""
    enabled: bool = True
    min_importance: float = 0.5  # Minimum importance score
    max_gaps_per_run: int = 5  # Limit to avoid overload
    trigger_frequency: str = "weekly"  # weekly, daily, manual


# ============================================================================
# Gap Identification Engine
# ============================================================================

class GapIdentifier:
    """
    Identifies knowledge gaps using graph structure analysis.

    Examples:
    - Bridge gap: "Feature Engineering" mentioned but not explained
    - Missing prerequisite: Know "Neural Networks" but not "Linear Algebra"
    - Incomplete category: Know "Python" and "JavaScript" but not "TypeScript"
    """

    def __init__(self, config: Optional[GapIdentificationConfig] = None):
        """
        Initialize gap identifier.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or GapIdentificationConfig()

        # Knowledge domain ontology (simple MVP - can be expanded)
        self.related_concepts = {
            # Machine Learning domain
            'machine learning': ['supervised learning', 'unsupervised learning', 'reinforcement learning',
                                'feature engineering', 'model evaluation', 'data preprocessing'],
            'neural networks': ['backpropagation', 'activation functions', 'loss functions',
                               'optimization', 'regularization', 'dropout'],
            'deep learning': ['convolutional networks', 'recurrent networks', 'transformers',
                            'attention mechanism', 'transfer learning'],

            # Programming domain
            'python': ['data types', 'functions', 'classes', 'modules',
                      'decorators', 'generators', 'async/await'],
            'javascript': ['promises', 'async/await', 'closures', 'prototypes',
                          'event loop', 'dom manipulation'],
            'typescript': ['types', 'interfaces', 'generics', 'decorators',
                          'type inference', 'modules'],

            # Data structures
            'algorithms': ['sorting', 'searching', 'graph algorithms',
                          'dynamic programming', 'greedy algorithms'],
            'data structures': ['arrays', 'linked lists', 'trees', 'graphs',
                               'hash tables', 'heaps', 'stacks', 'queues'],

            # Databases
            'databases': ['sql', 'nosql', 'indexing', 'transactions',
                         'normalization', 'query optimization'],

            # Web development
            'web development': ['html', 'css', 'javascript', 'react',
                               'backend', 'apis', 'authentication'],
        }

        # Prerequisite relationships (concept A requires concept B)
        self.prerequisites = {
            'deep learning': ['machine learning', 'linear algebra', 'calculus'],
            'neural networks': ['machine learning', 'linear algebra'],
            'reinforcement learning': ['machine learning', 'probability'],
            'transformers': ['neural networks', 'attention mechanism'],
            'async/await': ['promises', 'callbacks'],
            'typescript': ['javascript'],
            'react': ['javascript', 'html', 'css'],
        }

    def identify_gaps(
        self,
        knowledge_graph: Dict[str, Any]
    ) -> List[KnowledgeGap]:
        """
        Identify knowledge gaps in the graph.

        Args:
            knowledge_graph: Dict representing knowledge graph with keys:
                - nodes: List of concept nodes
                - edges: List of (source, target, type) edges
                - node_counts: Dict of node -> access count

        Returns:
            List of identified knowledge gaps
        """
        if not self.config.enabled:
            return []

        gaps = []

        # Extract graph data
        nodes = set(knowledge_graph.get('nodes', []))
        edges = knowledge_graph.get('edges', [])
        node_counts = knowledge_graph.get('node_counts', {})

        # Build adjacency lists
        in_degree, out_degree = self._build_degree_maps(nodes, edges)

        # Gap Type 1: Bridge gaps (mentioned but not explained)
        bridge_gaps = self._find_bridge_gaps(nodes, in_degree, out_degree, node_counts)
        gaps.extend(bridge_gaps)

        # Gap Type 2: Missing prerequisites
        prerequisite_gaps = self._find_missing_prerequisites(nodes)
        gaps.extend(prerequisite_gaps)

        # Gap Type 3: Incomplete categories
        category_gaps = self._find_incomplete_categories(nodes)
        gaps.extend(category_gaps)

        # Sort by importance and limit
        gaps.sort(key=lambda g: g.importance, reverse=True)
        return gaps[:self.config.max_gaps_per_run]

    def _build_degree_maps(
        self,
        nodes: Set[str],
        edges: List[Tuple[str, str, str]]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Build in-degree and out-degree maps for nodes.

        Args:
            nodes: Set of node names
            edges: List of (source, target, type) edges

        Returns:
            (in_degree, out_degree) dicts
        """
        in_degree = {node: 0 for node in nodes}
        out_degree = {node: 0 for node in nodes}

        for source, target, _ in edges:
            if target in in_degree:
                in_degree[target] += 1
            if source in out_degree:
                out_degree[source] += 1

        return in_degree, out_degree

    def _find_bridge_gaps(
        self,
        nodes: Set[str],
        in_degree: Dict[str, int],
        out_degree: Dict[str, int],
        node_counts: Dict[str, int]
    ) -> List[KnowledgeGap]:
        """
        Find bridge gaps: nodes with high in-degree but low out-degree.

        These are concepts mentioned frequently but not well explained.

        Args:
            nodes: Set of node names
            in_degree: In-degree map
            out_degree: Out-degree map
            node_counts: Access counts per node

        Returns:
            List of bridge gap instances
        """
        gaps = []

        for node in nodes:
            # Bridge characteristics:
            # - High in-degree (mentioned often)
            # - Low out-degree (not explained/expanded)
            # - Moderate to high access count (users are interested)

            if in_degree[node] >= 3 and out_degree[node] <= 1:
                # Calculate importance based on access count and degree ratio
                access_count = node_counts.get(node, 0)
                degree_ratio = in_degree[node] / max(out_degree[node], 1)

                importance = min(1.0, (access_count / 10) * 0.5 + (degree_ratio / 10) * 0.5)

                if importance >= self.config.min_importance:
                    # Find related concepts that might be missing
                    missing = self._get_related_concepts_not_in_graph(node, nodes)

                    gaps.append(KnowledgeGap(
                        bridge_concept=node,
                        missing_concepts=missing[:3],  # Top 3
                        importance=importance,
                        gap_type=GapType.BRIDGE_GAP,
                        suggested_queries=self._generate_queries(node, missing[:3])
                    ))

        return gaps

    def _find_missing_prerequisites(
        self,
        nodes: Set[str]
    ) -> List[KnowledgeGap]:
        """
        Find missing prerequisites: know advanced topic but missing basics.

        Args:
            nodes: Set of node names in graph

        Returns:
            List of missing prerequisite gaps
        """
        gaps = []

        for concept, prereqs in self.prerequisites.items():
            # Normalize concept name for comparison
            concept_lower = concept.lower()
            nodes_lower = {n.lower() for n in nodes}

            # Check if we know the advanced concept
            if concept_lower in nodes_lower:
                # Check which prerequisites are missing
                missing_prereqs = [
                    p for p in prereqs
                    if p.lower() not in nodes_lower
                ]

                if missing_prereqs:
                    # High importance - prerequisites are foundational
                    importance = 0.8

                    gaps.append(KnowledgeGap(
                        bridge_concept=concept,
                        missing_concepts=missing_prereqs,
                        importance=importance,
                        gap_type=GapType.MISSING_PREREQUISITE,
                        suggested_queries=self._generate_prerequisite_queries(
                            concept, missing_prereqs
                        )
                    ))

        return gaps

    def _find_incomplete_categories(
        self,
        nodes: Set[str]
    ) -> List[KnowledgeGap]:
        """
        Find incomplete categories: know some items, missing others.

        Args:
            nodes: Set of node names in graph

        Returns:
            List of incomplete category gaps
        """
        gaps = []

        for category, related in self.related_concepts.items():
            nodes_lower = {n.lower() for n in nodes}

            # Count how many related concepts we know
            known_concepts = [
                c for c in related
                if c.lower() in nodes_lower
            ]

            # If we know some but not all, it's an incomplete category
            if len(known_concepts) >= 2 and len(known_concepts) < len(related):
                missing_concepts = [
                    c for c in related
                    if c.lower() not in nodes_lower
                ]

                # Importance based on completeness ratio
                completeness = len(known_concepts) / len(related)
                importance = 0.6 * (1 - completeness)  # More missing = more important

                if importance >= self.config.min_importance:
                    gaps.append(KnowledgeGap(
                        bridge_concept=category,
                        missing_concepts=missing_concepts[:3],
                        importance=importance,
                        gap_type=GapType.INCOMPLETE_CATEGORY,
                        suggested_queries=self._generate_category_queries(
                            category, missing_concepts[:3]
                        )
                    ))

        return gaps

    def _get_related_concepts_not_in_graph(
        self,
        concept: str,
        existing_nodes: Set[str]
    ) -> List[str]:
        """
        Get related concepts that are not in the graph.

        Args:
            concept: The bridge concept
            existing_nodes: Set of existing nodes

        Returns:
            List of missing related concepts
        """
        concept_lower = concept.lower()
        nodes_lower = {n.lower() for n in existing_nodes}

        # Look up in ontology
        related = self.related_concepts.get(concept_lower, [])

        # Filter out concepts already in graph
        missing = [
            c for c in related
            if c.lower() not in nodes_lower
        ]

        return missing

    def _generate_queries(
        self,
        bridge_concept: str,
        missing_concepts: List[str]
    ) -> List[str]:
        """
        Generate suggested queries to fill a bridge gap.

        Args:
            bridge_concept: The bridge concept
            missing_concepts: Missing related concepts

        Returns:
            List of suggested query strings
        """
        queries = []

        # General query about the concept
        queries.append(f"What is {bridge_concept}?")

        # Queries about missing related concepts
        for missing in missing_concepts[:2]:  # Top 2
            queries.append(f"How does {missing} relate to {bridge_concept}?")

        # Practical query
        queries.append(f"How to use {bridge_concept} in practice?")

        return queries

    def _generate_prerequisite_queries(
        self,
        concept: str,
        missing_prereqs: List[str]
    ) -> List[str]:
        """
        Generate queries for missing prerequisites.

        Args:
            concept: The advanced concept
            missing_prereqs: Missing prerequisite concepts

        Returns:
            List of suggested queries
        """
        queries = []

        for prereq in missing_prereqs[:2]:
            queries.append(f"What is {prereq}?")
            queries.append(f"Why is {prereq} important for {concept}?")

        return queries

    def _generate_category_queries(
        self,
        category: str,
        missing_concepts: List[str]
    ) -> List[str]:
        """
        Generate queries for incomplete categories.

        Args:
            category: The category
            missing_concepts: Missing concepts in category

        Returns:
            List of suggested queries
        """
        queries = []

        for concept in missing_concepts[:2]:
            queries.append(f"What is {concept} in {category}?")

        queries.append(f"Overview of {category}")

        return queries

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get gap identification statistics.

        Returns:
            Dict with statistics
        """
        return {
            'min_importance': self.config.min_importance,
            'max_gaps_per_run': self.config.max_gaps_per_run,
            'trigger_frequency': self.config.trigger_frequency,
            'ontology_size': len(self.related_concepts),
            'prerequisites_defined': len(self.prerequisites)
        }


# ============================================================================
# Standalone Functions
# ============================================================================

def identify_knowledge_gaps(
    knowledge_graph: Dict[str, Any],
    config: Optional[GapIdentificationConfig] = None
) -> List[KnowledgeGap]:
    """
    Identify knowledge gaps in a graph (standalone function).

    Args:
        knowledge_graph: Dict with nodes, edges, node_counts
        config: Optional configuration

    Returns:
        List of identified gaps

    Example:
        >>> graph = {
        ...     'nodes': ['machine learning', 'neural networks', 'python'],
        ...     'edges': [('ml', 'nn', 'RELATED')],
        ...     'node_counts': {'machine learning': 10, 'neural networks': 5}
        ... }
        >>> gaps = identify_knowledge_gaps(graph)
        >>> print(len(gaps))
        2  # Missing prerequisites + incomplete categories
    """
    identifier = GapIdentifier(config)
    return identifier.identify_gaps(knowledge_graph)
