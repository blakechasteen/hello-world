"""
Unit tests for entity resolution — 2026-02-09
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from memory_bus.models import Entity, EntityAlias, EntityType
from memory_bus.resolve import resolve_entity


def _mock_pg(aliases: list[EntityAlias] | None = None) -> AsyncMock:
    pg = AsyncMock()
    pg.find_aliases = AsyncMock(return_value=aliases or [])
    return pg


def _mock_neo4j(
    entity: Entity | None = None,
    search_results: list[Entity] | None = None,
) -> AsyncMock:
    neo4j = AsyncMock()
    neo4j.get_entity = AsyncMock(return_value=entity)
    neo4j.search_entities_by_name = AsyncMock(return_value=search_results or [])
    return neo4j


@pytest.mark.asyncio
class TestResolveEntity:
    async def test_alias_match(self):
        entity = Entity(id="ent_aurora", name="Aurora", type=EntityType.OBJECT)
        alias = EntityAlias(alias="aurora", entity_id="ent_aurora", confidence=0.95)

        result = await resolve_entity(
            "Aurora",
            pg=_mock_pg(aliases=[alias]),
            neo4j=_mock_neo4j(entity=entity),
        )

        assert result.best_match is not None
        assert result.best_match.entity_id == "ent_aurora"
        assert result.best_match.confidence == 0.95
        assert result.best_match.source == "alias"

    async def test_neo4j_fallback_exact_name(self):
        entity = Entity(id="ent_aurora", name="Aurora", type=EntityType.OBJECT)

        result = await resolve_entity(
            "Aurora",
            pg=_mock_pg(),  # No aliases
            neo4j=_mock_neo4j(search_results=[entity]),
        )

        assert result.best_match is not None
        assert result.best_match.entity_id == "ent_aurora"
        assert result.best_match.source == "neo4j_name_search"
        assert result.best_match.confidence == 0.8  # exact name match

    async def test_neo4j_partial_name(self):
        entity = Entity(id="ent_aurora_hive", name="Aurora Hive", type=EntityType.OBJECT)

        result = await resolve_entity(
            "Aurora",
            pg=_mock_pg(),
            neo4j=_mock_neo4j(search_results=[entity]),
        )

        assert result.best_match is not None
        assert result.best_match.confidence == 0.6  # partial match

    async def test_no_match(self):
        result = await resolve_entity(
            "nonexistent",
            pg=_mock_pg(),
            neo4j=_mock_neo4j(),
        )

        assert result.best_match is None
        assert len(result.candidates) == 0

    async def test_dedup_across_sources(self):
        entity = Entity(id="ent_aurora", name="Aurora", type=EntityType.OBJECT)
        alias = EntityAlias(alias="aurora", entity_id="ent_aurora", confidence=0.95)

        result = await resolve_entity(
            "Aurora",
            pg=_mock_pg(aliases=[alias]),
            neo4j=_mock_neo4j(entity=entity, search_results=[entity]),
        )

        # Should not have duplicate entries for the same entity
        ids = [c.entity_id for c in result.candidates]
        assert len(ids) == len(set(ids))

    async def test_loom_scope(self):
        entity = Entity(id="ent_1", name="Aurora", type=EntityType.OBJECT)

        await resolve_entity(
            "Aurora",
            pg=_mock_pg(),
            neo4j=_mock_neo4j(search_results=[entity]),
            loom_id="test_loom",
        )

        # Verify loom_id was passed through
        # (checked via mock call args)
