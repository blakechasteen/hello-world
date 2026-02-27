"""
DreamWeaver Core Tests
======================

Comprehensive tests for HoloLoom's world-building component.

Test Coverage:
- Enums: EntityType, EventType, ConsistencyLevel, GenerationMode
- Dataclasses: WorldEntity, NarrativeEvent, WorldState, ConsistencyRule, NarrativeThread
- Config: DreamWeaverConfig
- Core: DreamWeaver class (init, context manager, bootstrap_world, step, query_world)

Created: 2025-12-30
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

# Import enums
from hololoom.dreamweaving import (
    EntityType,
    EventType,
    ConsistencyLevel,
    GenerationMode,
)

# Import dataclasses
from hololoom.dreamweaving import (
    WorldEntity,
    NarrativeEvent,
    WorldState,
    ConsistencyRule,
    NarrativeThread,
)

# Import core classes
from hololoom.dreamweaving.core import DreamWeaver, DreamWeaverConfig


# ============================================================================
# Enum Tests
# ============================================================================

class TestEntityType:
    """Tests for EntityType enum."""

    def test_all_entity_types_exist(self):
        """All expected entity types are defined."""
        assert EntityType.CHARACTER.value == "character"
        assert EntityType.LOCATION.value == "location"
        assert EntityType.FACTION.value == "faction"
        assert EntityType.ARTIFACT.value == "artifact"
        assert EntityType.EVENT.value == "event"
        assert EntityType.CONCEPT.value == "concept"
        assert EntityType.CUSTOM.value == "custom"

    def test_entity_type_count(self):
        """Correct number of entity types."""
        assert len(EntityType) == 7

    def test_entity_type_from_string(self):
        """Can create enum from string value."""
        assert EntityType("character") == EntityType.CHARACTER
        assert EntityType("location") == EntityType.LOCATION

    def test_invalid_entity_type_raises(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            EntityType("invalid_type")


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        """All expected event types are defined."""
        assert EventType.DIALOGUE.value == "dialogue"
        assert EventType.ACTION.value == "action"
        assert EventType.DISCOVERY.value == "discovery"
        assert EventType.CONFLICT.value == "conflict"
        assert EventType.RESOLUTION.value == "resolution"
        assert EventType.TRANSITION.value == "transition"
        assert EventType.REFLECTION.value == "reflection"

    def test_event_type_count(self):
        """Correct number of event types."""
        assert len(EventType) == 7

    def test_event_type_from_string(self):
        """Can create enum from string value."""
        assert EventType("dialogue") == EventType.DIALOGUE
        assert EventType("conflict") == EventType.CONFLICT


class TestConsistencyLevel:
    """Tests for ConsistencyLevel enum."""

    def test_all_consistency_levels_exist(self):
        """All expected consistency levels are defined."""
        assert ConsistencyLevel.STRICT.value == "strict"
        assert ConsistencyLevel.BALANCED.value == "balanced"
        assert ConsistencyLevel.LOOSE.value == "loose"
        assert ConsistencyLevel.DREAMLIKE.value == "dreamlike"

    def test_consistency_level_count(self):
        """Correct number of consistency levels."""
        assert len(ConsistencyLevel) == 4


class TestGenerationMode:
    """Tests for GenerationMode enum."""

    def test_all_generation_modes_exist(self):
        """All expected generation modes are defined."""
        assert GenerationMode.PROCEDURAL.value == "procedural"
        assert GenerationMode.NARRATIVE.value == "narrative"
        assert GenerationMode.COLLABORATIVE.value == "collaborative"
        assert GenerationMode.EMERGENT.value == "emergent"
        assert GenerationMode.HYBRID.value == "hybrid"

    def test_generation_mode_count(self):
        """Correct number of generation modes."""
        assert len(GenerationMode) == 5


# ============================================================================
# Dataclass Tests
# ============================================================================

class TestWorldEntity:
    """Tests for WorldEntity dataclass."""

    def test_create_basic_entity(self):
        """Can create entity with required fields."""
        entity = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Hero",
            description="The protagonist of the story",
            attributes={"strength": 10, "wisdom": 8},
            relationships=[],
            history=[],
            metadata={}
        )
        assert entity.entity_id == "char_001"
        assert entity.entity_type == EntityType.CHARACTER
        assert entity.name == "Hero"
        assert entity.attributes["strength"] == 10

    def test_entity_with_relationships(self):
        """Entity can have relationships."""
        entity = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Hero",
            description="Main character",
            attributes={},
            relationships=[
                {"target_id": "char_002", "relation_type": "ALLY", "strength": 0.9},
                {"target_id": "loc_001", "relation_type": "RESIDES_IN", "strength": 1.0}
            ],
            history=[],
            metadata={"created": "2025-12-30"}
        )
        assert len(entity.relationships) == 2
        assert entity.relationships[0]["relation_type"] == "ALLY"

    def test_entity_all_types(self):
        """Can create entities of all types."""
        for entity_type in EntityType:
            entity = WorldEntity(
                entity_id=f"test_{entity_type.value}",
                entity_type=entity_type,
                name=f"Test {entity_type.value.title()}",
                description=f"A test {entity_type.value}",
                attributes={},
                relationships=[],
                history=[],
                metadata={}
            )
            assert entity.entity_type == entity_type


class TestNarrativeEvent:
    """Tests for NarrativeEvent dataclass."""

    def test_create_basic_event(self):
        """Can create event with required fields."""
        event = NarrativeEvent(
            event_id="event_001",
            event_type=EventType.ACTION,
            timestamp=100.0,
            participants=["char_001", "char_002"],
            location_id="loc_001",
            description="Hero defeats the dragon",
            consequences={"dragon_defeated": True},
            causal_links={"causes": ["event_000"], "effects": []},
            metadata={}
        )
        assert event.event_id == "event_001"
        assert event.event_type == EventType.ACTION
        assert len(event.participants) == 2
        assert event.consequences["dragon_defeated"] is True

    def test_event_without_location(self):
        """Event can have None location."""
        event = NarrativeEvent(
            event_id="event_002",
            event_type=EventType.REFLECTION,
            timestamp=50.0,
            participants=["char_001"],
            location_id=None,
            description="Internal reflection",
            consequences={},
            causal_links={"causes": [], "effects": []},
            metadata={}
        )
        assert event.location_id is None

    def test_event_all_types(self):
        """Can create events of all types."""
        for event_type in EventType:
            event = NarrativeEvent(
                event_id=f"test_{event_type.value}",
                event_type=event_type,
                timestamp=0.0,
                participants=[],
                location_id=None,
                description=f"A {event_type.value} event",
                consequences={},
                causal_links={"causes": [], "effects": []},
                metadata={}
            )
            assert event.event_type == event_type


class TestWorldState:
    """Tests for WorldState dataclass."""

    def test_create_empty_world_state(self):
        """Can create world state with empty collections."""
        state = WorldState(
            timestamp=0.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )
        assert state.timestamp == 0.0
        assert len(state.entities) == 0
        assert len(state.active_threads) == 0

    def test_world_state_with_entities(self):
        """World state can contain entities."""
        entity = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Hero",
            description="Main character",
            attributes={},
            relationships=[],
            history=[],
            metadata={}
        )
        state = WorldState(
            timestamp=100.0,
            entities={"char_001": entity},
            active_threads=[],
            recent_events=[],
            global_state={"world_name": "Fantasy Realm"},
            consistency_violations=[],
            generation_context={}
        )
        assert "char_001" in state.entities
        assert state.global_state["world_name"] == "Fantasy Realm"


class TestConsistencyRule:
    """Tests for ConsistencyRule dataclass."""

    def test_create_consistency_rule(self):
        """Can create consistency rule."""
        rule = ConsistencyRule(
            rule_id="temporal_causality",
            rule_type="physical",
            description="Events must respect temporal order",
            check_function="check_temporal_order",
            severity=0.9,
            auto_fix=False
        )
        assert rule.rule_id == "temporal_causality"
        assert rule.severity == 0.9
        assert rule.auto_fix is False

    def test_rule_with_callable_check(self):
        """Rule can have callable check function."""
        def my_check(world_state: WorldState) -> bool:
            return True

        rule = ConsistencyRule(
            rule_id="custom_rule",
            rule_type="logical",
            description="Custom check",
            check_function=my_check,
            severity=0.5,
            auto_fix=True
        )
        assert callable(rule.check_function)


class TestNarrativeThread:
    """Tests for NarrativeThread dataclass."""

    def test_create_narrative_thread(self):
        """Can create narrative thread."""
        thread = NarrativeThread(
            thread_id="thread_001",
            thread_type="plot",
            title="The Dragon Quest",
            participants=["char_001", "char_002"],
            events=["event_001", "event_002"],
            status="active",
            tension=0.7,
            foreshadowing=["The ancient prophecy speaks of fire..."],
            metadata={}
        )
        assert thread.thread_id == "thread_001"
        assert thread.thread_type == "plot"
        assert thread.tension == 0.7
        assert len(thread.foreshadowing) == 1

    def test_thread_status_values(self):
        """Thread can have different status values."""
        for status in ["active", "resolved", "abandoned", "dormant"]:
            thread = NarrativeThread(
                thread_id=f"thread_{status}",
                thread_type="subplot",
                title=f"{status.title()} Thread",
                participants=[],
                events=[],
                status=status,
                tension=0.5,
                foreshadowing=[],
                metadata={}
            )
            assert thread.status == status


# ============================================================================
# DreamWeaverConfig Tests
# ============================================================================

class TestDreamWeaverConfig:
    """Tests for DreamWeaverConfig dataclass."""

    def test_default_config(self):
        """Config has sensible defaults."""
        config = DreamWeaverConfig()
        assert config.world_id == "default_world"
        assert config.save_path is None
        assert config.auto_save is True
        assert config.generation_mode == GenerationMode.HYBRID
        assert config.consistency_level == ConsistencyLevel.BALANCED
        assert config.max_entities == 10000
        assert config.max_events == 100000
        assert config.use_thompson_sampling is True

    def test_custom_config(self):
        """Can create config with custom values."""
        config = DreamWeaverConfig(
            world_id="my_world",
            save_path="/tmp/worlds",
            generation_mode=GenerationMode.NARRATIVE,
            consistency_level=ConsistencyLevel.STRICT,
            max_entities=5000,
            creativity_temperature=0.5,
            llm_provider="anthropic",
            llm_model="claude-3-5-sonnet-20241022"
        )
        assert config.world_id == "my_world"
        assert config.save_path == "/tmp/worlds"
        assert config.generation_mode == GenerationMode.NARRATIVE
        assert config.consistency_level == ConsistencyLevel.STRICT
        assert config.llm_provider == "anthropic"

    def test_config_safety_defaults(self):
        """Safety features are enabled by default."""
        config = DreamWeaverConfig()
        assert config.enable_content_filtering is True
        assert config.track_authorship is True
        assert config.enable_version_control is True


# ============================================================================
# DreamWeaver Core Tests (Mocked)
# ============================================================================

class TestDreamWeaverInit:
    """Tests for DreamWeaver initialization."""

    def test_init_with_defaults(self):
        """Can initialize with default config."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()

        weaver = DreamWeaver(config, mock_holo_config)

        assert weaver.config == config
        assert weaver.hololoom_config == mock_holo_config
        assert weaver.orchestrator is None  # Not initialized until __aenter__
        assert weaver.current_world_state is None
        assert len(weaver.consistency_rules) > 0  # Default rules loaded

    def test_init_with_existing_kg(self):
        """Can initialize with existing knowledge graph."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        mock_kg = MagicMock()

        weaver = DreamWeaver(config, mock_holo_config, knowledge_graph=mock_kg)

        assert weaver.kg == mock_kg

    def test_default_rules_loaded(self):
        """Default consistency rules are loaded."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()

        weaver = DreamWeaver(config, mock_holo_config)

        # Check default rules exist
        rule_ids = [r.rule_id for r in weaver.consistency_rules]
        assert "physical_causality" in rule_ids
        assert "entity_identity" in rule_ids
        assert "character_personality" in rule_ids

    def test_initial_metrics(self):
        """Metrics are initialized to zero."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()

        weaver = DreamWeaver(config, mock_holo_config)

        assert weaver.metrics["entities_generated"] == 0
        assert weaver.metrics["events_generated"] == 0
        assert weaver.metrics["consistency_checks"] == 0
        assert weaver.metrics["user_interactions"] == 0


class TestDreamWeaverHelpers:
    """Tests for DreamWeaver helper methods."""

    def test_create_initial_shards(self):
        """Can create initial memory shards."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        shards = weaver._create_initial_shards()

        assert len(shards) >= 1
        assert shards[0].id == "worldbuilding_primer"
        assert "world building" in shards[0].text.lower()

    def test_build_entity_generation_prompt(self):
        """Can build entity generation prompt."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        prompt = weaver._build_entity_generation_prompt(
            EntityType.CHARACTER,
            {"setting": "medieval"},
            None
        )

        assert "character" in prompt.lower()

    def test_build_event_generation_prompt(self):
        """Can build event generation prompt."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        prompt = weaver._build_event_generation_prompt(
            EventType.CONFLICT,
            ["char_001", "char_002"],
            {},
            None
        )

        assert "conflict" in prompt.lower()
        assert "2" in prompt  # 2 participants


class TestDreamWeaverMethods:
    """Tests for DreamWeaver methods (with mocking)."""

    def test_get_metrics_empty(self):
        """Get metrics on fresh weaver."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        metrics = weaver.get_metrics()

        assert metrics["entities_generated"] == 0
        assert metrics["world_states"] == 0
        assert metrics["total_entities"] == 0

    def test_get_metrics_with_world_state(self):
        """Get metrics with world state."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        # Create a world state
        entity = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Hero",
            description="Main character",
            attributes={},
            relationships=[],
            history=[],
            metadata={}
        )
        thread = NarrativeThread(
            thread_id="thread_001",
            thread_type="plot",
            title="Main Quest",
            participants=["char_001"],
            events=[],
            status="active",
            tension=0.5,
            foreshadowing=[],
            metadata={}
        )
        weaver.current_world_state = WorldState(
            timestamp=10.0,
            entities={"char_001": entity},
            active_threads=[thread],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        metrics = weaver.get_metrics()

        assert metrics["total_entities"] == 1
        assert metrics["active_threads"] == 1


@pytest.mark.asyncio
class TestDreamWeaverAsync:
    """Async tests for DreamWeaver (with mocking)."""

    async def test_retrieve_world_context(self):
        """Can retrieve world context."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        # Setup world state
        world_state = WorldState(
            timestamp=100.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        context = await weaver._retrieve_world_context("What happened?", world_state)

        assert "recent_events" in context
        assert "active_threads" in context
        assert "entity_count" in context

    async def test_apply_events_updates_timestamp(self):
        """Applying events updates world timestamp."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        world_state = WorldState(
            timestamp=100.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        new_state = await weaver._apply_events(world_state, [])

        assert new_state.timestamp == 101.0  # Incremented by 1.0
        assert new_state is not world_state  # Deep copy

    async def test_check_consistency_tracks_metrics(self):
        """Consistency check updates metrics."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        world_state = WorldState(
            timestamp=0.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        await weaver._check_consistency(world_state)

        assert weaver.metrics["consistency_checks"] == 1

    async def test_parse_entity_from_text(self):
        """Can parse entity from text."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        # Set initial world state for timestamp
        weaver.current_world_state = WorldState(
            timestamp=50.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        entity = await weaver._parse_entity_from_text(
            "A brave warrior from the northern mountains",
            EntityType.CHARACTER,
            {}
        )

        assert entity.entity_id == "entity_0"
        assert entity.entity_type == EntityType.CHARACTER
        assert "brave warrior" in entity.description

    async def test_parse_event_from_text(self):
        """Can parse event from text."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        # Set initial world state for timestamp
        weaver.current_world_state = WorldState(
            timestamp=50.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        event = await weaver._parse_event_from_text(
            "The hero faced the dragon in battle",
            EventType.CONFLICT,
            ["char_001", "dragon_001"],
            {}
        )

        assert event.event_id == "event_0"
        assert event.event_type == EventType.CONFLICT
        assert "dragon" in event.description
        assert event.timestamp == 50.0

    async def test_create_thread(self):
        """Can create narrative thread."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        # Set world state
        weaver.current_world_state = WorldState(
            timestamp=0.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        thread = await weaver.create_thread(
            thread_type="plot",
            title="Main Quest",
            participants=["char_001"]
        )

        assert thread.thread_id == "thread_0"
        assert thread.thread_type == "plot"
        assert thread.title == "Main Quest"
        assert thread.status == "active"
        assert thread.tension == 0.5

        # Thread should be added to world state
        assert len(weaver.current_world_state.active_threads) == 1


# ============================================================================
# Integration Tests (Mocked Orchestrator)
# ============================================================================

@pytest.mark.asyncio
class TestDreamWeaverIntegration:
    """Integration tests with mocked HoloLoom orchestrator."""

    async def test_context_manager_lifecycle(self):
        """Context manager properly initializes and cleans up."""
        config = DreamWeaverConfig(auto_save=False)
        mock_holo_config = MagicMock()

        with patch('hololoom.dreamweaving.core.WeavingOrchestrator') as MockOrchestrator:
            mock_instance = AsyncMock()
            MockOrchestrator.return_value = mock_instance

            async with DreamWeaver(config, mock_holo_config) as weaver:
                assert weaver.orchestrator is not None
                mock_instance.__aenter__.assert_called_once()

            mock_instance.__aexit__.assert_called_once()

    async def test_bootstrap_world_creates_state(self):
        """Bootstrap world creates initial state."""
        config = DreamWeaverConfig(auto_save=False)
        mock_holo_config = MagicMock()

        with patch('hololoom.dreamweaving.core.WeavingOrchestrator') as MockOrchestrator:
            mock_instance = AsyncMock()
            mock_instance.weave = AsyncMock(return_value=MagicMock(
                response="A medieval fantasy world with kingdoms and magic"
            ))
            MockOrchestrator.return_value = mock_instance

            async with DreamWeaver(config, mock_holo_config) as weaver:
                world_state = await weaver.bootstrap_world(
                    "A medieval fantasy world"
                )

                assert world_state is not None
                assert world_state.timestamp == 0.0
                assert "seed_prompt" in world_state.global_state
                assert weaver.current_world_state == world_state

    async def test_step_requires_world_state(self):
        """Step requires existing world state."""
        config = DreamWeaverConfig(auto_save=False)
        mock_holo_config = MagicMock()

        with patch('hololoom.dreamweaving.core.WeavingOrchestrator') as MockOrchestrator:
            mock_instance = AsyncMock()
            MockOrchestrator.return_value = mock_instance

            async with DreamWeaver(config, mock_holo_config) as weaver:
                with pytest.raises(ValueError, match="No world state"):
                    await weaver.step("Explore the forest")

    async def test_query_world_requires_world_state(self):
        """Query world requires existing world state."""
        config = DreamWeaverConfig(auto_save=False)
        mock_holo_config = MagicMock()

        with patch('hololoom.dreamweaving.core.WeavingOrchestrator') as MockOrchestrator:
            mock_instance = AsyncMock()
            MockOrchestrator.return_value = mock_instance

            async with DreamWeaver(config, mock_holo_config) as weaver:
                with pytest.raises(ValueError, match="No world state"):
                    await weaver.query_world("What exists here?")

    async def test_step_updates_metrics(self):
        """Step updates user interaction metrics."""
        config = DreamWeaverConfig(auto_save=False)
        mock_holo_config = MagicMock()

        with patch('hololoom.dreamweaving.core.WeavingOrchestrator') as MockOrchestrator:
            mock_instance = AsyncMock()
            mock_instance.weave = AsyncMock(return_value=MagicMock(
                response="You enter the dark forest"
            ))
            MockOrchestrator.return_value = mock_instance

            async with DreamWeaver(config, mock_holo_config) as weaver:
                # Bootstrap first
                weaver.current_world_state = WorldState(
                    timestamp=0.0,
                    entities={},
                    active_threads=[],
                    recent_events=[],
                    global_state={},
                    consistency_violations=[],
                    generation_context={}
                )

                await weaver.step("Enter the forest")

                assert weaver.metrics["user_interactions"] == 1

    async def test_generate_entity_updates_metrics(self):
        """Generate entity updates metrics."""
        config = DreamWeaverConfig(auto_save=False)
        mock_holo_config = MagicMock()

        with patch('hololoom.dreamweaving.core.WeavingOrchestrator') as MockOrchestrator:
            mock_instance = AsyncMock()
            mock_instance.weave = AsyncMock(return_value=MagicMock(
                response="A wise old wizard named Gandor"
            ))
            MockOrchestrator.return_value = mock_instance

            async with DreamWeaver(config, mock_holo_config) as weaver:
                entity = await weaver.generate_entity(
                    EntityType.CHARACTER,
                    {"role": "mentor"},
                    None
                )

                assert weaver.metrics["entities_generated"] == 1
                assert entity.entity_type == EntityType.CHARACTER

    async def test_generate_event_updates_metrics(self):
        """Generate event updates metrics."""
        config = DreamWeaverConfig(auto_save=False)
        mock_holo_config = MagicMock()

        with patch('hololoom.dreamweaving.core.WeavingOrchestrator') as MockOrchestrator:
            mock_instance = AsyncMock()
            mock_instance.weave = AsyncMock(return_value=MagicMock(
                response="A fierce battle erupts"
            ))
            MockOrchestrator.return_value = mock_instance

            async with DreamWeaver(config, mock_holo_config) as weaver:
                # Setup world state
                weaver.current_world_state = WorldState(
                    timestamp=100.0,
                    entities={},
                    active_threads=[],
                    recent_events=[],
                    global_state={},
                    consistency_violations=[],
                    generation_context={}
                )

                event = await weaver.generate_event(
                    EventType.CONFLICT,
                    ["char_001", "char_002"],
                    {},
                    None
                )

                assert weaver.metrics["events_generated"] == 1
                assert event.event_type == EventType.CONFLICT


# ============================================================================
# Serialization Tests
# ============================================================================

class TestWorldStateSerialization:
    """Tests for world state serialization."""

    def test_serialize_world_state(self):
        """Can serialize world state to dict."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        world_state = WorldState(
            timestamp=100.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        data = weaver._serialize_world_state(world_state)

        assert "timestamp" in data
        assert data["timestamp"] == 100.0

    def test_deserialize_world_state(self):
        """Can deserialize world state from dict."""
        config = DreamWeaverConfig()
        mock_holo_config = MagicMock()
        weaver = DreamWeaver(config, mock_holo_config)

        data = {"timestamp": 100.0, "entity_count": 5}

        world_state = weaver._deserialize_world_state(data)

        assert world_state.timestamp == 100.0
        assert isinstance(world_state, WorldState)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_world_state(self):
        """Handle empty world state gracefully."""
        state = WorldState(
            timestamp=0.0,
            entities={},
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )
        assert len(state.entities) == 0
        assert len(state.active_threads) == 0

    def test_large_world_state(self):
        """Handle large world state."""
        # Create many entities
        entities = {}
        for i in range(100):
            entities[f"entity_{i}"] = WorldEntity(
                entity_id=f"entity_{i}",
                entity_type=EntityType.CHARACTER,
                name=f"Character {i}",
                description=f"Description for character {i}",
                attributes={"level": i},
                relationships=[],
                history=[],
                metadata={}
            )

        state = WorldState(
            timestamp=1000.0,
            entities=entities,
            active_threads=[],
            recent_events=[],
            global_state={},
            consistency_violations=[],
            generation_context={}
        )

        assert len(state.entities) == 100

    def test_nested_relationships(self):
        """Handle complex nested relationships."""
        entity = WorldEntity(
            entity_id="char_001",
            entity_type=EntityType.CHARACTER,
            name="Hero",
            description="Main character",
            attributes={
                "stats": {"str": 10, "dex": 8, "int": 12},
                "inventory": [{"item": "sword", "count": 1}]
            },
            relationships=[
                {"target_id": "char_002", "relation_type": "ALLY", "strength": 0.9,
                 "metadata": {"since": "chapter_1"}},
                {"target_id": "faction_001", "relation_type": "MEMBER_OF", "strength": 1.0}
            ],
            history=[],
            metadata={"created": "2025-12-30", "author": "system"}
        )

        assert entity.attributes["stats"]["str"] == 10
        assert entity.relationships[0]["metadata"]["since"] == "chapter_1"
