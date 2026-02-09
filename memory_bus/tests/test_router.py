"""
Unit tests for router — 2026-02-09

Tests routing uses EXACT/STRUCTURED when entity_ids present.
Tests semantic stub returns "not available".
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_bus.models import (
    DetailLevel,
    Entity,
    EntityType,
    Episode,
    EpisodeType,
    MemoryItem,
    MemoryQuery,
    PressureTier,
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
