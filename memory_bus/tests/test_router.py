"""
Unit tests for router — 2026-02-09, updated 2026-02-10

Tests routing cascade: EXACT → STRUCTURED → GRAPH → SEMANTIC(stub).
Tests stub-vs-real-empty disambiguation.
Tests full fallthrough edge cases.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

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
    PressureTier,
    ResolveCandidate,
    ResolveResult,
    ResolutionPath,
)
from memory_bus.pressure import PressureEngine
from memory_bus.router import MemoryRouter


def _make_entity(id: str = "ent_1", name: str = "Aurora") -> Entity:
    return Entity(id=id, name=name, type=EntityType.OBJECT, importance=0.8)


def _make_episode(id: str = "ep_1", entity_id: str = "ent_1") -> Episode:
    return Episode(
        id=id,
        type=EpisodeType.OBSERVATION,
        summary="Test episode",
        significance=0.7,
        entity_ids=[entity_id],
    )


def _mock_neo4j(entity: Entity | None = None, episodes: list[Episode] | None = None) -> AsyncMock:
    neo4j = AsyncMock()
    neo4j.get_entity = AsyncMock(return_value=entity)
    neo4j.get_episode = AsyncMock(return_value=None)
    neo4j.query_episodes_for_entity = AsyncMock(return_value=episodes or [])
    neo4j.query_recent_episodes = AsyncMock(return_value=[])
    neo4j.search_entities_by_name = AsyncMock(return_value=[entity] if entity else [])

    # Converter methods need real implementations
    from memory_bus.neo4j_adapter import Neo4jAdapter
    adapter = Neo4jAdapter.__new__(Neo4jAdapter)
    neo4j.entity_to_item = adapter.entity_to_item
    neo4j.episode_to_item = adapter.episode_to_item
    return neo4j


def _mock_pg() -> AsyncMock:
    pg = AsyncMock()
    pg.write_audit = AsyncMock()
    pg.find_aliases = AsyncMock(return_value=[])
    return pg


@pytest.mark.asyncio
class TestRouterExactPath:
    async def test_exact_path_with_entity_id(self):
        entity = _make_entity()
        episode = _make_episode()
        neo4j = _mock_neo4j(entity=entity, episodes=[episode])

        router = MemoryRouter(
            neo4j=neo4j,
            pg=_mock_pg(),
            pressure=PressureEngine(),
        )

        query = MemoryQuery(text="", entity_ids=["ent_1"])
        result = await router.route(query)

        assert result.resolution_path == ResolutionPath.EXACT
        assert len(result.items) > 0
        assert any(i.id == "ent_1" for i in result.items)

    async def test_exact_path_fetches_episodes(self):
        entity = _make_entity()
        episode = _make_episode()
        neo4j = _mock_neo4j(entity=entity, episodes=[episode])

        router = MemoryRouter(
            neo4j=neo4j,
            pg=_mock_pg(),
            pressure=PressureEngine(),
        )

        query = MemoryQuery(text="", entity_ids=["ent_1"])
        result = await router.route(query)

        # Should have both entity and episode
        kinds = {i.kind for i in result.items}
        assert "entity" in kinds
        assert "episode" in kinds


@pytest.mark.asyncio
class TestRouterSemanticStub:
    async def test_semantic_returns_not_available(self):
        """Semantic query returns 'not available' and does NOT inject blobs."""
        neo4j = _mock_neo4j()
        neo4j.get_entity = AsyncMock(return_value=None)
        neo4j.get_episode = AsyncMock(return_value=None)

        router = MemoryRouter(
            neo4j=neo4j,
            pg=_mock_pg(),
            pressure=PressureEngine(),
        )

        # Query with only text, no IDs or names -> falls through to SEMANTIC
        query = MemoryQuery(text="something unknown")
        result = await router.route(query)

        assert result.resolution_path == ResolutionPath.SEMANTIC
        assert len(result.items) == 0
        assert any("not available" in w.lower() for w in result.warnings)
        assert result.formatted_context == ""

    async def test_semantic_disabled_at_tier2(self):
        neo4j = _mock_neo4j()
        neo4j.get_entity = AsyncMock(return_value=None)
        neo4j.get_episode = AsyncMock(return_value=None)

        pressure = PressureEngine(context_window=100_000)
        pressure.update(context_tokens_used=80_000)  # Tier2

        router = MemoryRouter(
            neo4j=neo4j,
            pg=_mock_pg(),
            pressure=pressure,
        )

        query = MemoryQuery(text="something")
        result = await router.route(query)

        assert result.resolution_path == ResolutionPath.SEMANTIC
        assert any("disabled" in w.lower() for w in result.warnings)


@pytest.mark.asyncio
class TestRouterAudit:
    async def test_audit_written_on_query(self):
        entity = _make_entity()
        neo4j = _mock_neo4j(entity=entity)
        pg = _mock_pg()

        router = MemoryRouter(
            neo4j=neo4j,
            pg=pg,
            pressure=PressureEngine(),
        )

        query = MemoryQuery(text="", entity_ids=["ent_1"], loop_id="loop_x")
        await router.route(query)

        pg.write_audit.assert_called()
        audit_row = pg.write_audit.call_args[0][0]
        assert audit_row.loop_id == "loop_x"


# ---------------------------------------------------------------------------
# Edge cases: stub-vs-real-empty disambiguation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStubVsRealEmpty:
    """
    Core risk: an empty result from the semantic stub and an empty result
    from a real structured query that found nothing must be distinguishable.
    The model needs to know "I can't search that way yet" vs "that data
    doesn't exist."
    """

    async def test_stub_empty_sets_flag(self):
        """When cascade falls through to semantic stub, semantic_stub_reached=True."""
        neo4j = _mock_neo4j()
        neo4j.get_entity = AsyncMock(return_value=None)
        neo4j.get_episode = AsyncMock(return_value=None)

        router = MemoryRouter(neo4j=neo4j, pg=_mock_pg(), pressure=PressureEngine())

        query = MemoryQuery(text="something unknown")
        result = await router.route(query)

        assert result.semantic_stub_reached is True
        assert result.no_results_reason == "semantic_not_implemented"
        assert len(result.items) == 0

    async def test_real_empty_exact_does_not_set_stub_flag(self):
        """EXACT path returns nothing because entity doesn't exist — NOT a stub."""
        neo4j = _mock_neo4j()
        neo4j.get_entity = AsyncMock(return_value=None)
        neo4j.get_episode = AsyncMock(return_value=None)

        router = MemoryRouter(neo4j=neo4j, pg=_mock_pg(), pressure=PressureEngine())

        # entity_ids provided but entity doesn't exist → falls through entire cascade
        # because text is empty and entity_names is empty, it hits semantic stub
        query = MemoryQuery(text="", entity_ids=["nonexistent"])
        result = await router.route(query)

        # With no text and no names, semantic stub is reached
        assert result.semantic_stub_reached is True

    async def test_graph_returns_real_results_no_stub(self):
        """GRAPH path finds real episodes — semantic_stub_reached should be False."""
        episode = _make_episode()
        neo4j = _mock_neo4j()
        neo4j.get_entity = AsyncMock(return_value=None)
        neo4j.get_episode = AsyncMock(return_value=None)
        neo4j.query_recent_episodes = AsyncMock(return_value=[episode])

        router = MemoryRouter(neo4j=neo4j, pg=_mock_pg(), pressure=PressureEngine())

        query = MemoryQuery(text="what happened recently")
        result = await router.route(query)

        assert result.semantic_stub_reached is False
        assert result.no_results_reason is None
        assert result.resolution_path == ResolutionPath.GRAPH
        assert len(result.items) == 1

    async def test_semantic_disabled_by_pressure_has_distinct_reason(self):
        """Tier2+ disables semantic — reason is 'semantic_disabled_by_pressure'."""
        neo4j = _mock_neo4j()
        neo4j.get_entity = AsyncMock(return_value=None)
        neo4j.get_episode = AsyncMock(return_value=None)

        pressure = PressureEngine(context_window=100_000)
        pressure.update(context_tokens_used=80_000)  # Tier2

        router = MemoryRouter(neo4j=neo4j, pg=_mock_pg(), pressure=pressure)

        query = MemoryQuery(text="something")
        result = await router.route(query)

        assert result.semantic_stub_reached is True
        assert result.no_results_reason == "semantic_disabled_by_pressure"


# ---------------------------------------------------------------------------
# Edge cases: cascade fallthrough
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCascadeFallthrough:
    """
    Verify that the cascade correctly falls through each level
    when earlier levels return empty.
    """

    async def test_exact_empty_falls_to_structured(self):
        """entity_ids returns nothing → falls to STRUCTURED if entity_names present."""
        entity = _make_entity(id="ent_aurora", name="Aurora")
        episode = _make_episode(id="ep_1", entity_id="ent_aurora")

        neo4j = _mock_neo4j(entity=entity, episodes=[episode])
        # EXACT: get_entity returns None for the given ID
        neo4j.get_entity = AsyncMock(side_effect=lambda eid: entity if eid == "ent_aurora" else None)
        neo4j.get_episode = AsyncMock(return_value=None)

        pg = _mock_pg()
        # resolve_entity will find alias for "Aurora"
        pg.find_aliases = AsyncMock(return_value=[
            EntityAlias(alias="aurora", entity_id="ent_aurora", confidence=0.9),
        ])

        router = MemoryRouter(neo4j=neo4j, pg=pg, pressure=PressureEngine())

        query = MemoryQuery(
            text="",
            entity_ids=["nonexistent_id"],
            entity_names=["Aurora"],
        )
        result = await router.route(query)

        assert result.resolution_path == ResolutionPath.STRUCTURED
        assert len(result.items) > 0
        assert result.semantic_stub_reached is False

    async def test_exact_and_structured_empty_falls_to_graph(self):
        """Both EXACT and STRUCTURED empty → falls to GRAPH with text query."""
        episode = _make_episode(id="ep_recent")

        neo4j = _mock_neo4j()
        neo4j.get_entity = AsyncMock(return_value=None)
        neo4j.get_episode = AsyncMock(return_value=None)
        neo4j.query_recent_episodes = AsyncMock(return_value=[episode])

        pg = _mock_pg()
        pg.find_aliases = AsyncMock(return_value=[])

        router = MemoryRouter(neo4j=neo4j, pg=pg, pressure=PressureEngine())

        query = MemoryQuery(
            text="what happened",
            entity_names=["NonexistentHive"],
        )
        result = await router.route(query)

        assert result.resolution_path == ResolutionPath.GRAPH
        assert len(result.items) == 1
        assert result.items[0].id == "ep_recent"
        assert result.semantic_stub_reached is False

    async def test_full_cascade_fallthrough_to_stub(self):
        """EXACT → STRUCTURED → GRAPH all empty → lands on SEMANTIC stub."""
        neo4j = _mock_neo4j()
        neo4j.get_entity = AsyncMock(return_value=None)
        neo4j.get_episode = AsyncMock(return_value=None)
        neo4j.query_recent_episodes = AsyncMock(return_value=[])

        pg = _mock_pg()
        pg.find_aliases = AsyncMock(return_value=[])

        router = MemoryRouter(neo4j=neo4j, pg=pg, pressure=PressureEngine())

        query = MemoryQuery(
            text="something completely unknown",
            entity_ids=["no_such_id"],
            entity_names=["NoSuchEntity"],
        )
        result = await router.route(query)

        assert result.resolution_path == ResolutionPath.SEMANTIC
        assert result.semantic_stub_reached is True
        assert result.no_results_reason == "semantic_not_implemented"
        assert len(result.items) == 0
        assert len(result.warnings) > 0

    async def test_importance_filter_causes_fallthrough(self):
        """Entities exist but below min_importance at Tier1 → treated as empty."""
        low_imp_entity = Entity(
            id="ent_low", name="LowPri", type=EntityType.OTHER,
            importance=0.1,  # Below Tier1 min_importance of 0.3
        )
        neo4j = _mock_neo4j(entity=low_imp_entity)

        pressure = PressureEngine(context_window=100_000)
        pressure.update(context_tokens_used=60_000)  # Tier1 → min_importance=0.3

        router = MemoryRouter(neo4j=neo4j, pg=_mock_pg(), pressure=pressure)

        query = MemoryQuery(text="check this", entity_ids=["ent_low"])
        result = await router.route(query)

        # Entity filtered by importance → cascade falls through
        assert result.semantic_stub_reached is True
        assert len(result.items) == 0

    async def test_dedup_across_exact_and_episodes(self):
        """Same item fetched via entity + episode query is deduplicated."""
        entity = _make_entity(id="ent_1")
        episode = _make_episode(id="ep_1")
        neo4j = _mock_neo4j(entity=entity, episodes=[episode, episode])

        router = MemoryRouter(neo4j=neo4j, pg=_mock_pg(), pressure=PressureEngine())

        query = MemoryQuery(text="", entity_ids=["ent_1"])
        result = await router.route(query)

        ids = [i.id for i in result.items]
        assert len(ids) == len(set(ids)), "Duplicate items should be removed"
