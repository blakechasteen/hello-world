"""
DreamWeaver: Open-Source World Building Component for HoloLoom

The DreamWeaver is the generative layer that transforms narrative threads into
coherent, persistent worlds. It weaves story, character, location, and history
into a living knowledge graph that evolves with each interaction.

Core Philosophy:
---------------
"Dreams are not random - they are recursive narratives seeking coherence."

The DreamWeaver treats world building as a constraint satisfaction problem in
narrative space, where consistency, causality, and creativity must balance.

Architecture:
------------
1. **Narrative Threads** - Story elements (character arcs, plot lines, themes)
2. **World Fabric** - Persistent world state (locations, history, rules)
3. **Consistency Engine** - Maintains logical/causal coherence
4. **Generative Loom** - Procedurally generates new content
5. **Narrative Memory** - Story-aware knowledge graph

Integration Points:
------------------
- Yarn Graph: Stores world entities/relationships
- Warp Space: Tensioned narrative manifolds for generation
- Convergence Engine: Collapses narrative possibilities to concrete events
- Reflection Buffer: Learns from player/reader reactions
- Thompson Sampling: Explores narrative branches

World Building Features:
-----------------------
- Procedural generation with narrative constraints
- Character personality/goal tracking
- Location interconnectedness (geography, politics, culture)
- Historical causality (past → present → future)
- Multi-scale consistency (scene → chapter → book → series)
- Collaborative authoring (human + AI co-creation)

Safety & Alignment:
------------------
- Content filtering for sensitive topics
- Authorship transparency (human vs AI generated)
- Version control for world states
- Rollback/branch for alternative timelines
"""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

__version__ = "0.1.0"

# Public API exports
__all__ = [
    # Core components
    "DreamWeaver",
    "NarrativeThread",
    "WorldFabric",
    "ConsistencyEngine",
    "GenerativeLoom",

    # Data structures
    "WorldEntity",
    "NarrativeEvent",
    "WorldState",
    "ConsistencyRule",

    # Enums
    "EntityType",
    "EventType",
    "ConsistencyLevel",
    "GenerationMode",

    # Protocols
    "WorldBuilderProtocol",
    "ConsistencyCheckerProtocol",
    "GeneratorProtocol",
]


class EntityType(Enum):
    """Types of world entities."""
    CHARACTER = "character"
    LOCATION = "location"
    FACTION = "faction"
    ARTIFACT = "artifact"
    EVENT = "event"
    CONCEPT = "concept"
    CUSTOM = "custom"


class EventType(Enum):
    """Types of narrative events."""
    DIALOGUE = "dialogue"
    ACTION = "action"
    DISCOVERY = "discovery"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"
    TRANSITION = "transition"
    REFLECTION = "reflection"


class ConsistencyLevel(Enum):
    """Strictness of consistency checking."""
    STRICT = "strict"        # No contradictions allowed
    BALANCED = "balanced"    # Minor inconsistencies allowed if narratively valuable
    LOOSE = "loose"          # Prioritize creativity over consistency
    DREAMLIKE = "dreamlike"  # Embrace surrealism and non-linearity


class GenerationMode(Enum):
    """World generation strategies."""
    PROCEDURAL = "procedural"      # Rule-based generation
    NARRATIVE = "narrative"        # Story-driven generation
    COLLABORATIVE = "collaborative" # Human-AI co-creation
    EMERGENT = "emergent"          # Player/reader-driven
    HYBRID = "hybrid"              # Mix of above


@dataclass
class WorldEntity:
    """
    An entity in the world (character, location, faction, etc).

    Attributes:
        entity_id: Unique identifier
        entity_type: Type of entity
        name: Human-readable name
        description: Textual description
        attributes: Key-value attributes (stats, traits, etc)
        relationships: Connections to other entities
        history: Timeline of events involving this entity
        metadata: Generation metadata (source, timestamp, etc)
    """
    entity_id: str
    entity_type: EntityType
    name: str
    description: str
    attributes: Dict[str, Any]
    relationships: List[Dict[str, Any]]  # {target_id, relation_type, strength}
    history: List["NarrativeEvent"]
    metadata: Dict[str, Any]


@dataclass
class NarrativeEvent:
    """
    An event in the narrative timeline.

    Attributes:
        event_id: Unique identifier
        event_type: Type of event
        timestamp: When event occurred (in-world time)
        participants: Entities involved
        location_id: Where event occurred
        description: What happened
        consequences: Effects on world state
        causal_links: Events that caused/resulted from this
        metadata: Generation metadata
    """
    event_id: str
    event_type: EventType
    timestamp: float  # In-world time (arbitrary units)
    participants: List[str]  # Entity IDs
    location_id: Optional[str]
    description: str
    consequences: Dict[str, Any]
    causal_links: Dict[str, List[str]]  # {causes: [...], effects: [...]}
    metadata: Dict[str, Any]


@dataclass
class WorldState:
    """
    Complete snapshot of world at a point in time.

    Attributes:
        timestamp: Current in-world time
        entities: All entities in world
        active_threads: Ongoing narrative threads
        recent_events: Recent history (last N events)
        global_state: World-level state (politics, economy, etc)
        consistency_violations: Known inconsistencies
        generation_context: Context for next generation
    """
    timestamp: float
    entities: Dict[str, WorldEntity]
    active_threads: List["NarrativeThread"]
    recent_events: List[NarrativeEvent]
    global_state: Dict[str, Any]
    consistency_violations: List[Dict[str, Any]]
    generation_context: Dict[str, Any]


@dataclass
class ConsistencyRule:
    """
    A rule that world states must satisfy.

    Attributes:
        rule_id: Unique identifier
        rule_type: Category of rule (physical, logical, narrative)
        description: Human-readable description
        check_function: Validation function (callable or LLM prompt)
        severity: How important is this rule?
        auto_fix: Can violations be automatically corrected?
    """
    rule_id: str
    rule_type: str  # "physical", "logical", "narrative", "stylistic"
    description: str
    check_function: Any  # Callable[[WorldState], bool] or str (LLM prompt)
    severity: float  # 0.0 (guideline) to 1.0 (hard constraint)
    auto_fix: bool


@dataclass
class NarrativeThread:
    """
    An ongoing story thread (plot line, character arc, theme).

    Attributes:
        thread_id: Unique identifier
        thread_type: Category (plot, character, theme, subplot)
        title: Human-readable name
        participants: Entities involved
        events: Events in this thread
        status: Current state (active, resolved, abandoned)
        tension: Narrative tension level (0.0-1.0)
        foreshadowing: Future hints/setup
        metadata: Generation metadata
    """
    thread_id: str
    thread_type: str  # "plot", "character", "theme", "subplot"
    title: str
    participants: List[str]  # Entity IDs
    events: List[str]  # Event IDs
    status: str  # "active", "resolved", "abandoned", "dormant"
    tension: float  # 0.0 (no tension) to 1.0 (climax)
    foreshadowing: List[str]  # Setup for future events
    metadata: Dict[str, Any]


# Protocol definitions for dependency injection

class WorldBuilderProtocol(Protocol):
    """Protocol for world building components."""

    async def generate_entity(
        self,
        entity_type: EntityType,
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> WorldEntity:
        """Generate a new world entity."""
        ...

    async def generate_event(
        self,
        event_type: EventType,
        participants: List[str],
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> NarrativeEvent:
        """Generate a narrative event."""
        ...

    async def evolve_world(
        self,
        world_state: WorldState,
        user_input: Optional[str] = None,
        mode: GenerationMode = GenerationMode.HYBRID
    ) -> WorldState:
        """Evolve world state forward."""
        ...


class ConsistencyCheckerProtocol(Protocol):
    """Protocol for consistency checking."""

    async def check_consistency(
        self,
        world_state: WorldState,
        rules: List[ConsistencyRule]
    ) -> List[Dict[str, Any]]:
        """Check world state against consistency rules."""
        ...

    async def resolve_contradiction(
        self,
        world_state: WorldState,
        violation: Dict[str, Any]
    ) -> Optional[WorldState]:
        """Attempt to resolve a consistency violation."""
        ...


class GeneratorProtocol(Protocol):
    """Protocol for generative components."""

    async def generate_text(
        self,
        prompt: str,
        context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate text (description, dialogue, etc)."""
        ...

    async def expand_context(
        self,
        entity: WorldEntity,
        depth: int = 1
    ) -> Dict[str, Any]:
        """Expand context around an entity."""
        ...


# Placeholder implementations (to be filled in core.py)

class DreamWeaver:
    """Main orchestrator for world building. See core.py for implementation."""
    pass


class NarrativeMemory:
    """Story-aware knowledge graph. See narrative_memory.py for implementation."""
    pass


class WorldFabric:
    """Persistent world state manager. See world_fabric.py for implementation."""
    pass


class ConsistencyEngine:
    """Maintains logical/causal coherence. See consistency.py for implementation."""
    pass


class GenerativeLoom:
    """Procedural content generation. See generative.py for implementation."""
    pass
