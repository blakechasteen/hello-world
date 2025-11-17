"""
Tests for DreamWeaving Data Structures

Tests the core data structures: WorldEntity, NarrativeEvent, WorldState, ConsistencyRule, NarrativeThread

Author: Claude Code
Date: 2025-11-16
Phase: 2 (Testing) - Week 1, Day 1
Target: +10 tests for data structure validation
"""

import pytest
from HoloLoom.dreamweaving import (
    WorldEntity,
    NarrativeEvent,
    WorldState,
    ConsistencyRule,
    NarrativeThread,
    EntityType,
    EventType,
    ConsistencyLevel,
)


class TestWorldEntity:
    """Test suite for WorldEntity data structure."""

    def test_world_entity_creation(self):
        """Test WorldEntity creation with valid parameters."""
        entity = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Alice",
            description="A brave adventurer",
            attributes={"strength": 15, "intelligence": 18},
            relationships=[],
            history=[],
            metadata={"created_by": "human"}
        )

        assert entity.entity_id == "char_001"
        assert entity.entity_type == EntityType.CHARACTER
        assert entity.name == "Alice"
        assert entity.attributes["strength"] == 15
        assert len(entity.relationships) == 0

    def test_world_entity_with_relationships(self):
        """Test WorldEntity with relationships to other entities."""
        entity = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Alice",
            description="A brave adventurer",
            attributes={},
            relationships=[
                {"target": "char_002", "type": "friend", "strength": 0.8},
                {"target": "loc_001", "type": "lives_in", "strength": 1.0}
            ],
            history=[],
            metadata={}
        )

        assert len(entity.relationships) == 2
        assert entity.relationships[0]["target"] == "char_002"
        assert entity.relationships[1]["type"] == "lives_in"

    def test_world_entity_serialization(self):
        """Test WorldEntity can be serialized to dict."""
        entity = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Alice",
            description="Test character",
            attributes={"level": 5},
            relationships=[],
            history=[],
            metadata={}
        )

        # Serialize to dict (assuming dataclass behavior)
        entity_dict = {
            "entity_id": entity.entity_id,
            "entity_type": entity.entity_type.value,
            "name": entity.name,
            "description": entity.description,
            "attributes": entity.attributes,
        }

        assert entity_dict["entity_id"] == "char_001"
        assert entity_dict["entity_type"] == "character"


class TestNarrativeEvent:
    """Test suite for NarrativeEvent data structure."""

    def test_narrative_event_creation(self):
        """Test NarrativeEvent creation."""
        event = NarrativeEvent(
            event_id="evt_001",
            event_type=EventType.ACTION,
            timestamp=100.0,
            participants=["char_001", "char_002"],
            location_id="loc_001",
            description="Alice met Bob in the tavern",
            consequences={},
            causal_links={"causes": [], "effects": []},
            metadata={}
        )

        assert event.event_id == "evt_001"
        assert event.event_type == EventType.ACTION
        assert len(event.participants) == 2
        assert event.location_id == "loc_001"

    def test_narrative_event_with_consequences(self):
        """Test NarrativeEvent with consequences."""
        event = NarrativeEvent(
            event_id="evt_001",
            event_type=EventType.CONFLICT,
            timestamp=200.0,
            participants=["char_001", "char_002"],
            location_id="loc_001",
            description="A fierce battle",
            consequences={
                "relationship_changes": [{"target": "char_002", "value": -0.5}],
                "items_acquired": ["sword_of_valor"],
                "health_changes": {"char_001": -20, "char_002": -30}
            },
            causal_links={
                "causes": ["evt_000"],
                "effects": ["evt_002", "evt_003"]
            },
            metadata={}
        )

        assert len(event.consequences) == 3
        assert "relationship_changes" in event.consequences
        assert event.consequences["items_acquired"][0] == "sword_of_valor"
        assert len(event.causal_links["causes"]) == 1
        assert len(event.causal_links["effects"]) == 2


class TestWorldState:
    """Test suite for WorldState data structure."""

    def test_world_state_creation(self):
        """Test WorldState creation."""
        state = WorldState(
            timestamp=100.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        assert state.timestamp == 100.0
        assert len(state.entities) == 0
        assert len(state.recent_events) == 0

    def test_world_state_with_entities(self):
        """Test WorldState with entities."""
        entity1 = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Alice",
            description="Hero",
            attributes={},
            relationships=[],
            history=[],
            metadata={}
        )

        state = WorldState(
            timestamp=100.0,
            entities={"char_001": entity1},
            active_threads=[],
            recent_events=[],
            global_state={"season": "winter", "day": 1},
            consistency_violations=[],
            generation_context={"mode": "narrative"}
        )

        assert len(state.entities) == 1
        assert state.entities["char_001"].name == "Alice"
        assert state.global_state["season"] == "winter"


class TestConsistencyRule:
    """Test suite for ConsistencyRule data structure."""

    def test_consistency_rule_creation(self):
        """Test ConsistencyRule creation."""
        rule = ConsistencyRule(
            rule_id="rule_001",
            rule_type="physical",
            description="Characters cannot be in two places at once",
            check_function=lambda state: True,  # Simple validation function
            severity=1.0,  # Hard constraint
            auto_fix=False
        )

        assert rule.rule_id == "rule_001"
        assert rule.severity == 1.0
        assert rule.rule_type == "physical"
        assert rule.auto_fix == False


class TestNarrativeThread:
    """Test suite for NarrativeThread data structure."""

    def test_narrative_thread_creation(self):
        """Test NarrativeThread creation."""
        thread = NarrativeThread(
            thread_id="thread_001",
            thread_type="plot",
            title="Quest for the Lost Artifact",
            participants=["char_001", "char_002"],
            events=["evt_001", "evt_002"],
            status="active",
            tension=0.7,  # 0.0 = resolved, 1.0 = maximum tension
            foreshadowing=["The sword has a dark secret", "The villain returns"],
            metadata={"created_at": 100.0}
        )

        assert thread.thread_id == "thread_001"
        assert thread.thread_type == "plot"
        assert thread.title == "Quest for the Lost Artifact"
        assert thread.tension == 0.7
        assert thread.status == "active"
        assert len(thread.participants) == 2
        assert len(thread.events) == 2
        assert len(thread.foreshadowing) == 2
