"""
Conversation Memory
===================

Tracks entities, relationships, and events across the conversation,
enabling scenes that evolve naturally as discussion progresses.

The hero mentioned in message 1 should still be visible in message 10.
The database we discussed earlier should maintain its position when
we add a new service that connects to it.

Philosophy: "Conversations have continuity. Visualizations should too."

Created: 2025-12-01
Author: HoloLoom Team
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class EntityType(Enum):
    """Types of entities that can persist in conversation."""
    # Narrative entities
    CHARACTER = "character"
    LOCATION = "location"
    OBJECT = "object"
    EVENT = "event"

    # Technical entities
    SERVICE = "service"
    DATABASE = "database"
    API = "api"
    COMPONENT = "component"

    # UI entities
    SCREEN = "screen"
    FORM = "form"
    BUTTON = "button"
    PANEL = "panel"

    # Data entities
    METRIC = "metric"
    DATASET = "dataset"

    # Abstract
    CONCEPT = "concept"


class RelationType(Enum):
    """Types of relationships between entities."""
    # Spatial
    CONTAINS = "contains"       # room contains table
    NEAR = "near"              # hero near castle
    INSIDE = "inside"          # data inside database

    # Directional
    CONNECTS_TO = "connects_to"  # service connects to database
    FLOWS_TO = "flows_to"        # data flows to analytics
    LEADS_TO = "leads_to"        # button leads to screen

    # Hierarchical
    PART_OF = "part_of"          # button part of form
    OWNS = "owns"                # user owns settings

    # Temporal
    FOLLOWS = "follows"          # scene follows scene
    CAUSES = "causes"            # action causes event

    # Semantic
    IS_A = "is_a"               # postgres is a database
    SIMILAR_TO = "similar_to"   # redis similar to memcached


@dataclass
class EntityState:
    """Current state of an entity in the conversation."""
    id: str
    name: str
    entity_type: EntityType

    # Spatial state
    position: dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0, "z": 0})
    scale: float = 1.0
    rotation: float = 0.0

    # Visual state
    color: str | None = None
    style: str | None = None
    visible: bool = True
    highlighted: bool = False

    # Semantic state
    attributes: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    # Temporal state
    first_mentioned: datetime = field(default_factory=datetime.now)
    last_mentioned: datetime = field(default_factory=datetime.now)
    mention_count: int = 1

    # Evolution
    previous_states: list[dict[str, Any]] = field(default_factory=list)

    def update_mention(self) -> None:
        """Record that entity was mentioned again."""
        self.last_mentioned = datetime.now()
        self.mention_count += 1

    def move_to(self, x: float, y: float, z: float) -> None:
        """Move entity, preserving history."""
        self.previous_states.append({
            "position": self.position.copy(),
            "timestamp": datetime.now().isoformat()
        })
        self.position = {"x": x, "y": y, "z": z}

    @property
    def importance(self) -> float:
        """Calculate importance based on mentions and recency."""
        recency_hours = (datetime.now() - self.last_mentioned).total_seconds() / 3600
        recency_factor = 1.0 / (1.0 + recency_hours * 0.1)  # Decay over hours
        mention_factor = min(self.mention_count / 10, 1.0)  # Cap at 10 mentions
        return 0.5 * recency_factor + 0.5 * mention_factor


@dataclass
class Relationship:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float = 1.0  # 0.0 to 1.0
    bidirectional: bool = False
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def id(self) -> str:
        return f"{self.source_id}-{self.relation_type.value}-{self.target_id}"


@dataclass
class ConversationEvent:
    """An event in the conversation timeline."""
    id: str
    description: str
    entities_involved: list[str]
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = "action"  # action, state_change, appearance, disappearance


class ConversationMemory:
    """
    Maintains memory of entities across the conversation.

    Enables:
    - Entity persistence (the hero stays visible)
    - Relationship tracking (service connects to database)
    - Spatial continuity (positions maintained)
    - Temporal evolution (scenes build on each other)
    """

    def __init__(self, max_entities: int = 100):
        self.max_entities = max_entities
        self.entities: dict[str, EntityState] = {}
        self.relationships: list[Relationship] = []
        self.timeline: list[ConversationEvent] = []
        self._entity_aliases: dict[str, str] = {}  # "the hero" -> "hero_123"

    def _generate_id(self, name: str) -> str:
        """Generate stable ID from name."""
        normalized = name.lower().strip()
        hash_suffix = hashlib.md5(normalized.encode()).hexdigest()[:6]
        return f"{normalized.replace(' ', '_')}_{hash_suffix}"

    def _find_entity_by_name(self, name: str) -> EntityState | None:
        """Find entity by name or alias."""
        normalized = name.lower().strip()

        # Check aliases first
        if normalized in self._entity_aliases:
            entity_id = self._entity_aliases[normalized]
            return self.entities.get(entity_id)

        # Check direct matches
        for entity in self.entities.values():
            if entity.name.lower() == normalized:
                return entity

        # Check partial matches
        for entity in self.entities.values():
            if normalized in entity.name.lower() or entity.name.lower() in normalized:
                return entity

        return None

    def remember_entity(
        self,
        name: str,
        entity_type: EntityType,
        attributes: dict[str, Any] | None = None,
        position: dict[str, float] | None = None,
    ) -> EntityState:
        """
        Remember an entity, creating or updating as needed.

        If the entity already exists, updates its mention count.
        If new, creates it with auto-assigned position.
        """
        # Check if entity already exists
        existing = self._find_entity_by_name(name)
        if existing:
            existing.update_mention()
            if attributes:
                existing.attributes.update(attributes)
            if position:
                existing.move_to(**position)
            return existing

        # Create new entity
        entity_id = self._generate_id(name)

        # Auto-assign position based on entity type and count
        if position is None:
            position = self._auto_position(entity_type)

        entity = EntityState(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            position=position,
            attributes=attributes or {},
        )

        self.entities[entity_id] = entity
        self._entity_aliases[name.lower().strip()] = entity_id

        # Prune if too many entities
        if len(self.entities) > self.max_entities:
            self._prune_least_important()

        # Record event
        self.timeline.append(ConversationEvent(
            id=f"appear_{entity_id}",
            description=f"{name} appeared",
            entities_involved=[entity_id],
            event_type="appearance"
        ))

        return entity

    def _auto_position(self, entity_type: EntityType) -> dict[str, float]:
        """Calculate automatic position based on entity type and existing entities."""
        # Count entities of same type
        same_type = [e for e in self.entities.values() if e.entity_type == entity_type]
        index = len(same_type)

        # Position strategies by type
        if entity_type in [EntityType.CHARACTER]:
            # Characters spread horizontally at ground level
            return {"x": (index - 2) * 3.0, "y": 0, "z": 0}

        elif entity_type in [EntityType.SERVICE, EntityType.API]:
            # Services in a grid, middle layer
            row = index // 4
            col = index % 4
            return {"x": (col - 1.5) * 4.0, "y": 2.0, "z": row * -3.0}

        elif entity_type in [EntityType.DATABASE]:
            # Databases at bottom, spread out
            return {"x": (index - 1) * 5.0, "y": -2.0, "z": 0}

        elif entity_type in [EntityType.SCREEN, EntityType.PANEL]:
            # UI elements floating in front
            return {"x": (index - 2) * 4.0, "y": 1.0, "z": 3.0}

        elif entity_type in [EntityType.LOCATION]:
            # Locations as backdrops
            return {"x": 0, "y": 0, "z": -10.0 - index * 5.0}

        else:
            # Default: spiral outward
            import math
            angle = index * 0.8
            radius = 2 + index * 0.5
            return {
                "x": math.cos(angle) * radius,
                "y": 0,
                "z": math.sin(angle) * radius
            }

    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relation_type: RelationType,
        strength: float = 1.0,
        bidirectional: bool = False,
    ) -> Relationship | None:
        """Add a relationship between two entities."""
        source = self._find_entity_by_name(source_name)
        target = self._find_entity_by_name(target_name)

        if not source or not target:
            return None

        # Check if relationship already exists
        for rel in self.relationships:
            if rel.source_id == source.id and rel.target_id == target.id:
                rel.strength = max(rel.strength, strength)  # Strengthen
                return rel

        relationship = Relationship(
            source_id=source.id,
            target_id=target.id,
            relation_type=relation_type,
            strength=strength,
            bidirectional=bidirectional,
        )

        self.relationships.append(relationship)

        # Record event
        self.timeline.append(ConversationEvent(
            id=f"rel_{relationship.id}",
            description=f"{source.name} {relation_type.value} {target.name}",
            entities_involved=[source.id, target.id],
            event_type="relationship"
        ))

        return relationship

    def record_event(
        self,
        description: str,
        entity_names: list[str],
        event_type: str = "action"
    ) -> ConversationEvent:
        """Record an event involving entities."""
        entity_ids = []
        for name in entity_names:
            entity = self._find_entity_by_name(name)
            if entity:
                entity.update_mention()
                entity_ids.append(entity.id)

        event = ConversationEvent(
            id=f"event_{len(self.timeline)}",
            description=description,
            entities_involved=entity_ids,
            event_type=event_type,
        )

        self.timeline.append(event)
        return event

    def _prune_least_important(self) -> None:
        """Remove least important entities when over capacity."""
        if len(self.entities) <= self.max_entities:
            return

        # Sort by importance
        sorted_entities = sorted(
            self.entities.values(),
            key=lambda e: e.importance,
            reverse=True
        )

        # Keep only the most important
        keep_ids = {e.id for e in sorted_entities[:self.max_entities]}

        # Remove others
        to_remove = [eid for eid in self.entities if eid not in keep_ids]
        for eid in to_remove:
            del self.entities[eid]

        # Clean up relationships
        self.relationships = [
            r for r in self.relationships
            if r.source_id in keep_ids and r.target_id in keep_ids
        ]

    def get_active_entities(self, min_importance: float = 0.1) -> list[EntityState]:
        """Get entities above importance threshold."""
        return [
            e for e in self.entities.values()
            if e.importance >= min_importance
        ]

    def get_entity_graph(self) -> dict[str, Any]:
        """Get entities and relationships as a graph structure."""
        return {
            "entities": {
                eid: {
                    "name": e.name,
                    "type": e.entity_type.value,
                    "position": e.position,
                    "importance": e.importance,
                    "attributes": e.attributes,
                }
                for eid, e in self.entities.items()
            },
            "relationships": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.relation_type.value,
                    "strength": r.strength,
                }
                for r in self.relationships
            ]
        }

    def get_scene_context(self) -> dict[str, Any]:
        """Get context suitable for scene composition."""
        active = self.get_active_entities()

        # Group by type
        by_type: dict[str, list[EntityState]] = {}
        for entity in active:
            type_name = entity.entity_type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(entity)

        return {
            "entity_count": len(active),
            "entities_by_type": {
                t: [{"name": e.name, "id": e.id, "position": e.position}
                    for e in entities]
                for t, entities in by_type.items()
            },
            "relationships": [
                {
                    "from": self.entities[r.source_id].name if r.source_id in self.entities else r.source_id,
                    "to": self.entities[r.target_id].name if r.target_id in self.entities else r.target_id,
                    "type": r.relation_type.value,
                }
                for r in self.relationships
                if r.source_id in self.entities and r.target_id in self.entities
            ],
            "recent_events": [
                {"description": e.description, "type": e.event_type}
                for e in self.timeline[-5:]  # Last 5 events
            ]
        }

    def clear(self) -> None:
        """Clear all memory."""
        self.entities.clear()
        self.relationships.clear()
        self.timeline.clear()
        self._entity_aliases.clear()


# Entity extraction patterns
ENTITY_PATTERNS = {
    # Characters
    EntityType.CHARACTER: [
        r'\b(?:the\s+)?(hero|villain|protagonist|antagonist|character|person|user|admin|customer)\b',
        r'\b([A-Z][a-z]+)\s+(?:said|walked|ran|looked|entered|left)\b',
    ],

    # Locations
    EntityType.LOCATION: [
        r'\b(?:the\s+)?(room|castle|forest|city|office|building|house|cave|mountain|beach|space)\b',
        r'\b(?:in|at|to)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
    ],

    # Services
    EntityType.SERVICE: [
        r'\b(\w+)\s+(?:service|api|microservice|server)\b',
        r'\b(\w+)Service\b',
    ],

    # Databases
    EntityType.DATABASE: [
        r'\b(PostgreSQL|MySQL|MongoDB|Redis|Cassandra|Neo4j|Qdrant|SQLite)\b',
        r'\b(\w+)\s+(?:database|db|storage)\b',
    ],

    # UI Elements
    EntityType.SCREEN: [
        r'\b(\w+)\s+(?:screen|page|view)\b',
        r'\b(?:the\s+)?(\w+)\s+(?:dashboard|panel|modal)\b',
    ],

    EntityType.BUTTON: [
        r'\b(\w+)\s+button\b',
        r'\bbutton\s+(?:for|to)\s+(\w+)\b',
    ],

    EntityType.FORM: [
        r'\b(\w+)\s+form\b',
        r'\bform\s+(?:for|to)\s+(\w+)\b',
    ],
}

# Relationship extraction patterns
RELATIONSHIP_PATTERNS = [
    (r'(\w+)\s+connects?\s+to\s+(\w+)', RelationType.CONNECTS_TO),
    (r'(\w+)\s+(?:sends?|writes?)\s+(?:data\s+)?to\s+(\w+)', RelationType.FLOWS_TO),
    (r'(\w+)\s+(?:reads?|fetches?|queries?)\s+(?:from\s+)?(\w+)', RelationType.FLOWS_TO),
    (r'(\w+)\s+(?:contains?|includes?)\s+(\w+)', RelationType.CONTAINS),
    (r'(\w+)\s+(?:is\s+)?inside\s+(\w+)', RelationType.INSIDE),
    (r'(\w+)\s+(?:is\s+)?(?:part\s+of|belongs?\s+to)\s+(\w+)', RelationType.PART_OF),
    (r'(\w+)\s+(?:leads?\s+to|opens?)\s+(\w+)', RelationType.LEADS_TO),
    (r'(\w+)\s+(?:is\s+a|is\s+an)\s+(\w+)', RelationType.IS_A),
    (r'(\w+)\s+(?:near|next\s+to|beside)\s+(\w+)', RelationType.NEAR),
]


def extract_entities_from_text(
    text: str,
    memory: ConversationMemory
) -> list[EntityState]:
    """Extract and remember entities from text."""
    extracted = []
    text_lower = text.lower()

    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match if isinstance(match, str) else match[0]
                if name and len(name) > 1:
                    entity = memory.remember_entity(name, entity_type)
                    extracted.append(entity)

    return extracted


def extract_relationships_from_text(
    text: str,
    memory: ConversationMemory
) -> list[Relationship]:
    """Extract and remember relationships from text."""
    extracted = []

    for pattern, rel_type in RELATIONSHIP_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:
                source_name, target_name = match[0], match[1]
                rel = memory.add_relationship(source_name, target_name, rel_type)
                if rel:
                    extracted.append(rel)

    return extracted


# Singleton memory instance
_conversation_memory: ConversationMemory | None = None


def get_conversation_memory() -> ConversationMemory:
    """Get or create the conversation memory singleton."""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory


__all__ = [
    'EntityType',
    'EntityState',
    'RelationType',
    'Relationship',
    'ConversationEvent',
    'ConversationMemory',
    'get_conversation_memory',
    'extract_entities_from_text',
    'extract_relationships_from_text',
]
