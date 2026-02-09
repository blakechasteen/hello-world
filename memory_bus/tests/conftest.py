"""
Shared fixtures for memory_bus tests — 2026-02-09
"""

from datetime import datetime

import pytest

from memory_bus.models import (
    DetailLevel,
    Entity,
    EntityAlias,
    EntityType,
    Episode,
    EpisodeType,
    MemoryItem,
    MemoryQuery,
    MemoryStoreRequest,
    PlanRow,
    PlanState,
    ResolutionPath,
)


@pytest.fixture
def sample_entity() -> Entity:
    return Entity(
        id="ent_aurora",
        type=EntityType.OBJECT,
        name="Aurora",
        importance=0.8,
        loom_origin="test_loom",
        properties={"species": "bee_hive", "location": "north_yard"},
    )


@pytest.fixture
def sample_episode(sample_entity: Entity) -> Episode:
    return Episode(
        id="ep_inspection_001",
        type=EpisodeType.OBSERVATION,
        timestamp=datetime(2026, 2, 1, 10, 0, 0),
        summary="Inspected Aurora hive: 8 frames of brood, healthy queen spotted",
        content="Full inspection notes: 8 frames brood, 3 frames honey, queen seen on frame 5.",
        significance=0.7,
        loom_origin="test_loom",
        entity_ids=[sample_entity.id],
    )


@pytest.fixture
def sample_alias(sample_entity: Entity) -> EntityAlias:
    return EntityAlias(
        alias="aurora",
        entity_id=sample_entity.id,
        loom_id="test_loom",
        confidence=0.95,
    )


@pytest.fixture
def sample_plan() -> PlanRow:
    return PlanRow(
        id="plan_spring_treatment",
        name="Spring treatment plan for Aurora",
        state=PlanState.ACTIVE,
        progress=0.3,
        loom_id="test_loom",
    )


@pytest.fixture
def sample_items() -> list[MemoryItem]:
    """A list of MemoryItems for formatter tests."""
    return [
        MemoryItem(
            id="ent_aurora",
            kind="entity",
            title="Aurora",
            summary="object: Aurora",
            importance=0.8,
            resolution_path=ResolutionPath.EXACT,
            properties={"species": "bee_hive"},
        ),
        MemoryItem(
            id="ep_inspection_001",
            kind="episode",
            title="Inspected Aurora hive",
            summary="Inspected Aurora hive: 8 frames of brood, healthy queen spotted",
            content="Full inspection notes: 8 frames brood, 3 frames honey.",
            timestamp=datetime(2026, 2, 1, 10, 0, 0),
            importance=0.7,
            resolution_path=ResolutionPath.STRUCTURED,
            related_entity_ids=["ent_aurora"],
        ),
        MemoryItem(
            id="ep_treatment_002",
            kind="episode",
            title="Applied oxalic acid treatment",
            summary="Applied oxalic acid vaporization to Aurora for varroa mite control",
            content="Treatment log: OA vaporization, 2g dose, ambient temp 12C.",
            timestamp=datetime(2026, 1, 15, 14, 30, 0),
            importance=0.6,
            resolution_path=ResolutionPath.STRUCTURED,
            related_entity_ids=["ent_aurora"],
        ),
    ]


@pytest.fixture
def sample_query() -> MemoryQuery:
    return MemoryQuery(
        text="What treatment should I apply to Aurora next?",
        entity_names=["Aurora"],
        loom_id="test_loom",
        loop_id="loop_001",
    )


@pytest.fixture
def sample_store_request(sample_entity: Entity, sample_episode: Episode, sample_alias: EntityAlias) -> MemoryStoreRequest:
    return MemoryStoreRequest(
        entities=[sample_entity],
        episodes=[sample_episode],
        aliases=[sample_alias],
        loom_id="test_loom",
        loop_id="loop_001",
    )
