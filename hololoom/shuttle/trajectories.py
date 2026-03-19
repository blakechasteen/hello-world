"""
HoloLoom Shuttle - Trajectory Strategies

Trajectories define how to enter and traverse the Yarn graph from Warp anchors.
Each trajectory encodes a different strategy for expanding the graph.

NOTE: Renamed from "policies" to "trajectories" to avoid collision with
HoloLoom's existing policy system (tool selection policies in policy/unified.py).
"""

from dataclasses import dataclass
from typing import Protocol

# ============================================================================
# Core Data Structures
# ============================================================================

@dataclass
class Anchor:
    """An anchor point from Warp search results into the Yarn graph."""
    name: str
    type: str  # e.g., "Project", "Task", "Person", "System", "Artifact"
    node_id: str | None = None  # Yarn node ID if known


@dataclass
class TraversalConfig:
    """Configuration for how to expand the Yarn graph from anchors."""
    max_depth: int
    max_nodes: int
    allowed_edge_types: list[str]


# ============================================================================
# Trajectory Protocol
# ============================================================================

class TrajectoryStrategy(Protocol):
    """
    Protocol for trajectory strategies.

    A trajectory defines how to expand the Yarn graph from Warp anchors.
    It's essentially a strategy pattern for graph traversal.
    """
    name: str

    def build_config(self, anchors: list[Anchor]) -> TraversalConfig:
        """
        Given anchors, return parameters for how to explore the Yarn graph.

        Args:
            anchors: Anchor points from Warp search

        Returns:
            TraversalConfig for Yarn graph expansion
        """
        ...


# ============================================================================
# Concrete Trajectory Strategies
# ============================================================================

class ProjectBlockersTrajectory:
    """
    Follow blocking and dependency relationships.
    Good for: "What's blocking X?", "Why is Y delayed?"
    """
    name = "project_blockers"

    def build_config(self, anchors: list[Anchor]) -> TraversalConfig:
        return TraversalConfig(
            max_depth=2,
            max_nodes=40,
            allowed_edge_types=["BLOCKED_BY", "DEPENDS_ON"],
        )


class OwnershipTrajectory:
    """
    Follow ownership and assignment relationships.
    Good for: "Who owns X?", "What is Y working on?"
    """
    name = "who_owns_this"

    def build_config(self, anchors: list[Anchor]) -> TraversalConfig:
        return TraversalConfig(
            max_depth=1,
            max_nodes=30,
            allowed_edge_types=["ASSIGNED_TO", "OWNS"],
        )


class TimelineTrajectory:
    """
    Follow chronological and temporal relationships.
    Good for: "What happened before X?", "What's the history of Y?"
    """
    name = "timeline"

    def build_config(self, anchors: list[Anchor]) -> TraversalConfig:
        return TraversalConfig(
            max_depth=3,
            max_nodes=50,
            allowed_edge_types=["HAPPENED_BEFORE", "HAPPENED_AFTER", "PRECEDES", "FOLLOWS"],
        )


class ConceptualTrajectory:
    """
    Follow semantic and conceptual relationships.
    Good for: "What's related to X?", "Similar concepts to Y?"
    """
    name = "conceptual"

    def build_config(self, anchors: list[Anchor]) -> TraversalConfig:
        return TraversalConfig(
            max_depth=2,
            max_nodes=35,
            allowed_edge_types=["RELATED_TO", "SIMILAR_TO", "USES_CONCEPT"],
        )


class HierarchicalTrajectory:
    """
    Follow organizational and hierarchical structure.
    Good for: "What's above X in the hierarchy?", "Sub-components of Y?"
    """
    name = "hierarchical"

    def build_config(self, anchors: list[Anchor]) -> TraversalConfig:
        return TraversalConfig(
            max_depth=2,
            max_nodes=40,
            allowed_edge_types=["PARENT_OF", "CHILD_OF", "CONTAINS", "PART_OF"],
        )


class ExploratoryTrajectory:
    """
    Broad, undirected expansion for open-ended queries.
    Good for: "Tell me about X", "What do we know about Y?"
    """
    name = "exploratory"

    def build_config(self, anchors: list[Anchor]) -> TraversalConfig:
        return TraversalConfig(
            max_depth=2,
            max_nodes=50,
            allowed_edge_types=["RELATED_TO", "CONNECTED_TO", "ASSOCIATED_WITH"],
        )


# ============================================================================
# Trajectory Registry
# ============================================================================

ALL_TRAJECTORIES: list[TrajectoryStrategy] = [
    ProjectBlockersTrajectory(),
    OwnershipTrajectory(),
    TimelineTrajectory(),
    ConceptualTrajectory(),
    HierarchicalTrajectory(),
    ExploratoryTrajectory(),
]

TRAJECTORY_BY_NAME = {trajectory.name: trajectory for trajectory in ALL_TRAJECTORIES}
