"""
Causal Chain Finder

Uses Layer 1 causal reasoning to find HOW to achieve goals.

Key Insight: Causal paths tell us which actions lead to desired outcomes!
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from HoloLoom.causal import CausalDAG

logger = logging.getLogger(__name__)


@dataclass
class CausalPath:
    """
    Causal path from source to target.

    Attributes:
        nodes: Variables in path
        strength: Cumulative causal strength
        length: Number of edges
    """
    nodes: List[str]
    strength: float
    length: int

    def __repr__(self):
        return f"CausalPath({' → '.join(self.nodes)}, strength={self.strength:.2f})"


class CausalChainFinder:
    """
    Finds causal chains for planning.

    Integrates Layer 1 (causal reasoning) with Layer 2 (planning).

    Usage:
        finder = CausalChainFinder(causal_dag)

        # Find how to achieve outcome
        paths = finder.find_paths_to_goal("recovery")

        # Find strongest path
        best = finder.find_strongest_path("treatment", "recovery")
    """

    def __init__(self, dag: CausalDAG):
        """
        Initialize finder.

        Args:
            dag: Causal DAG from Layer 1
        """
        self.dag = dag

    def find_paths_to_goal(
        self,
        goal_var: str,
        max_length: int = 5
    ) -> List[CausalPath]:
        """
        Find all causal paths leading to goal variable.

        Args:
            goal_var: Target variable
            max_length: Maximum path length

        Returns:
            List of causal paths
        """
        paths = []

        # Find all ancestors (variables that cause goal)
        ancestors = self.dag.ancestors(goal_var)

        for ancestor in ancestors:
            # Find all paths from ancestor to goal
            simple_paths = self.dag.get_paths(ancestor, goal_var)

            for path in simple_paths:
                if len(path) <= max_length + 1:  # +1 because path includes both ends
                    # Calculate path strength
                    strength = self._calculate_path_strength(path)

                    paths.append(CausalPath(
                        nodes=path,
                        strength=strength,
                        length=len(path) - 1
                    ))

        # Sort by strength
        paths.sort(key=lambda p: p.strength, reverse=True)

        return paths

    def find_strongest_path(
        self,
        source: str,
        target: str
    ) -> Optional[CausalPath]:
        """
        Find strongest causal path from source to target.

        Args:
            source: Starting variable
            target: Goal variable

        Returns:
            Strongest path or None
        """
        # Get all paths
        simple_paths = self.dag.get_paths(source, target)

        if not simple_paths:
            return None

        # Calculate strength for each
        path_strengths = []
        for path in simple_paths:
            strength = self._calculate_path_strength(path)
            path_strengths.append((path, strength))

        # Return strongest
        best_path, best_strength = max(path_strengths, key=lambda x: x[1])

        return CausalPath(
            nodes=best_path,
            strength=best_strength,
            length=len(best_path) - 1
        )

    def find_controllable_causes(
        self,
        goal_var: str,
        controllable_vars: Set[str]
    ) -> List[CausalPath]:
        """
        Find controllable variables that cause goal.

        Args:
            goal_var: Target variable
            controllable_vars: Variables we can intervene on

        Returns:
            Paths from controllable variables to goal
        """
        paths = []

        for controllable in controllable_vars:
            # Check if path exists
            simple_paths = self.dag.get_paths(controllable, goal_var)

            for path in simple_paths:
                strength = self._calculate_path_strength(path)
                paths.append(CausalPath(
                    nodes=path,
                    strength=strength,
                    length=len(path) - 1
                ))

        # Sort by strength
        paths.sort(key=lambda p: p.strength, reverse=True)

        return paths

    def _calculate_path_strength(self, path: List[str]) -> float:
        """
        Calculate cumulative strength of causal path.

        Uses product of edge strengths (weakest link matters).
        """
        if len(path) < 2:
            return 1.0

        strength = 1.0

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]

            # Get edge strength
            edge = self.dag.edges.get((source, target))
            if edge:
                strength *= edge.strength
            else:
                # No edge found (shouldn't happen if path is valid)
                strength = 0.0
                break

        return strength

    def explain_path(self, path: CausalPath) -> str:
        """
        Generate human-readable explanation of causal path.

        Args:
            path: Causal path to explain

        Returns:
            Natural language explanation
        """
        lines = []

        lines.append(f"Causal Chain: {' → '.join(path.nodes)}")
        lines.append(f"Strength: {path.strength:.2f}")
        lines.append(f"Length: {path.length} steps")
        lines.append("")
        lines.append("Reasoning:")

        # Explain each step
        for i in range(len(path.nodes) - 1):
            source = path.nodes[i]
            target = path.nodes[i + 1]

            edge = self.dag.edges.get((source, target))
            if edge:
                lines.append(
                    f"  {i+1}. {source} causes {target} "
                    f"(strength={edge.strength:.2f})"
                )
                if edge.mechanism:
                    lines.append(f"      Mechanism: {edge.mechanism}")

        return "\n".join(lines)
