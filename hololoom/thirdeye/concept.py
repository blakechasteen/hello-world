"""
Concept - Core data type for 3rdEye visualizations.

A Concept is a semantic unit extracted from chat messages that can be
visualized in 1D (sparklines), 2D (Tufte), or 3D (immersive) space.

Created: 2025-11-30
Author: HoloLoom Team
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ConceptType(Enum):
    """Types of concepts that can be visualized."""

    # Core cognitive types
    ENTITY = auto()          # Named things (Thompson Sampling, Python, etc.)
    RELATIONSHIP = auto()    # Connections between entities
    PROCESS = auto()         # Sequences, algorithms, workflows
    COMPARISON = auto()      # A vs B, tradeoffs, alternatives
    HIERARCHY = auto()       # Taxonomies, IS_A relationships
    TEMPORAL = auto()        # Time-based patterns, evolution

    # Abstract types
    PROBABILITY = auto()     # Distributions, uncertainty, confidence
    DECISION = auto()        # Choice points, branches, outcomes
    ACTIVATION = auto()      # Spreading activation, attention flow
    MEMORY = auto()          # Retrieval, storage, consolidation

    # Composite types
    CLUSTER = auto()         # Group of related concepts
    THREAD = auto()          # Narrative chain, reasoning path
    WORLD = auto()           # Full Thoughtspace visualization


class VisualStyle(Enum):
    """Visual rendering styles for concepts."""

    MINIMAL = "minimal"        # Simple shapes, low detail
    STANDARD = "standard"      # Balanced detail and performance
    RICH = "rich"              # Full detail, animations
    IMMERSIVE = "immersive"    # Full 3D with effects


@dataclass
class SemanticPosition:
    """Position in 228D semantic space (or projected to 3D)."""

    # 3D projection for visualization
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # Full semantic embedding (optional, for distance calculations)
    embedding: list[float] | None = None

    # Interpretable axes (from 16-axis projection)
    axes: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_embedding(cls, embedding: list[float]) -> "SemanticPosition":
        """Create position from full embedding with UMAP projection."""
        # Simple projection for now - use first 3 components scaled
        if len(embedding) >= 3:
            scale = 10.0  # World units
            return cls(
                x=embedding[0] * scale,
                y=embedding[1] * scale,
                z=embedding[2] * scale,
                embedding=embedding
            )
        return cls(embedding=embedding)


@dataclass
class Concept:
    """
    A semantic unit that can be visualized in the 3rdEye.

    Concepts are extracted from chat messages and smoothly transition
    as the conversation evolves. Each concept knows:
    - What it represents (type, label, metadata)
    - Where it sits in semantic space (position)
    - How it relates to other concepts (connections)
    - How it should be visualized (style, importance)
    """

    # Identity
    id: str                              # Unique identifier
    label: str                           # Display name
    concept_type: ConceptType            # Type for visualization

    # Semantic position
    position: SemanticPosition = field(default_factory=SemanticPosition)

    # Visual properties
    importance: float = 0.5              # 0.0-1.0, affects size/prominence
    confidence: float = 1.0              # 0.0-1.0, affects opacity/certainty
    style: VisualStyle = VisualStyle.STANDARD

    # Connections to other concepts
    connections: list[str] = field(default_factory=list)  # IDs of connected concepts
    parent_id: str | None = None      # Parent concept (for hierarchies)

    # Content
    description: str | None = None    # Brief explanation
    metadata: dict[str, Any] = field(default_factory=dict)

    # Temporal
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # Source tracking
    source_message_id: str | None = None  # Chat message that created this

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = time.time()

    def age_seconds(self) -> float:
        """How long since this concept was last active."""
        return time.time() - self.last_active

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for WebSocket streaming."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.concept_type.name.lower(),
            "position": {
                "x": self.position.x,
                "y": self.position.y,
                "z": self.position.z,
            },
            "importance": self.importance,
            "confidence": self.confidence,
            "style": self.style.value,
            "connections": self.connections,
            "parent_id": self.parent_id,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Concept":
        """Create concept from dict."""
        return cls(
            id=data["id"],
            label=data["label"],
            concept_type=ConceptType[data["type"].upper()],
            position=SemanticPosition(
                x=data.get("position", {}).get("x", 0),
                y=data.get("position", {}).get("y", 0),
                z=data.get("position", {}).get("z", 0),
            ),
            importance=data.get("importance", 0.5),
            confidence=data.get("confidence", 1.0),
            style=VisualStyle(data.get("style", "standard")),
            connections=data.get("connections", []),
            parent_id=data.get("parent_id"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            last_active=data.get("last_active", time.time()),
        )


@dataclass
class ConceptWorld:
    """
    A collection of concepts forming a Thoughtspace world.

    The world manages the spatial layout of concepts and their
    relationships, enabling the full-screen Thoughtspace mode.
    """

    name: str
    concepts: dict[str, Concept] = field(default_factory=dict)
    focal_concept_id: str | None = None  # Current focus

    # World bounds
    min_x: float = -50.0
    max_x: float = 50.0
    min_y: float = -50.0
    max_y: float = 50.0
    min_z: float = -50.0
    max_z: float = 50.0

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the world."""
        self.concepts[concept.id] = concept

    def remove_concept(self, concept_id: str) -> Concept | None:
        """Remove and return a concept from the world."""
        return self.concepts.pop(concept_id, None)

    def get_concept(self, concept_id: str) -> Concept | None:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)

    def set_focus(self, concept_id: str) -> None:
        """Set the focal concept (camera centers on this)."""
        if concept_id in self.concepts:
            self.focal_concept_id = concept_id
            self.concepts[concept_id].touch()

    def get_connected_concepts(self, concept_id: str) -> list[Concept]:
        """Get all concepts connected to the given concept."""
        concept = self.concepts.get(concept_id)
        if not concept:
            return []
        return [
            self.concepts[cid]
            for cid in concept.connections
            if cid in self.concepts
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert world to JSON-serializable dict."""
        return {
            "name": self.name,
            "concepts": {
                cid: c.to_dict() for cid, c in self.concepts.items()
            },
            "focal_concept_id": self.focal_concept_id,
            "bounds": {
                "min": [self.min_x, self.min_y, self.min_z],
                "max": [self.max_x, self.max_y, self.max_z],
            }
        }


# Convenience factory functions
def create_entity_concept(
    id: str,
    label: str,
    importance: float = 0.5,
    description: str | None = None,
    embedding: list[float] | None = None,
) -> Concept:
    """Create an entity concept (named thing)."""
    position = SemanticPosition.from_embedding(embedding) if embedding else SemanticPosition()
    return Concept(
        id=id,
        label=label,
        concept_type=ConceptType.ENTITY,
        position=position,
        importance=importance,
        description=description,
    )


def create_comparison_concept(
    id: str,
    label: str,
    concepts_compared: list[str],
    importance: float = 0.6,
) -> Concept:
    """Create a comparison concept (A vs B)."""
    return Concept(
        id=id,
        label=label,
        concept_type=ConceptType.COMPARISON,
        connections=concepts_compared,
        importance=importance,
        metadata={"compared": concepts_compared},
    )


def create_process_concept(
    id: str,
    label: str,
    steps: list[str],
    importance: float = 0.5,
) -> Concept:
    """Create a process concept (algorithm, workflow)."""
    return Concept(
        id=id,
        label=label,
        concept_type=ConceptType.PROCESS,
        connections=steps,
        importance=importance,
        metadata={"steps": steps},
    )
