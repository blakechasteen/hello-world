"""
MERGE idempotency tests — 2026-02-10

Validates that Neo4j MERGE-based writes in neo4j_adapter.py are truly
idempotent under concurrent async execution.

Key invariant: multiple concurrent create_entity / create_episode calls
with the same ID must:
  1. Use MERGE (not CREATE) — single node regardless of race
  2. Use unconditional SET — last-writer-wins, deterministic final state
  3. Not duplicate ABOUT relationships (MERGE on relationship too)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from memory_bus.models import Entity, EntityType, Episode, EpisodeType
from memory_bus.neo4j_adapter import Neo4jAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_session():
    """Create a mock Neo4j session that records all Cypher queries."""
    session = AsyncMock()
    session.run = AsyncMock()
    return session


def _mock_driver(session):
    """Create a mock driver whose .session() returns a context manager."""
    driver = MagicMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    driver.session.return_value = ctx
    return driver


# ---------------------------------------------------------------------------
# MERGE pattern validation
# ---------------------------------------------------------------------------


class TestMergeIdempotency:
    """Verify that Cypher uses MERGE not CREATE for entity/episode writes."""

    @pytest.mark.asyncio
    async def test_create_entity_uses_merge(self):
        """create_entity must use MERGE {id: $id} pattern."""
        session = _mock_session()
        driver = _mock_driver(session)

        adapter = Neo4jAdapter.__new__(Neo4jAdapter)
        adapter._driver = driver

        entity = Entity(
            id="ent_test_1", name="Aurora", type=EntityType.OBJECT,
            importance=0.8,
        )

        await adapter.create_entity(entity)

        # Exactly one call to session.run
        assert session.run.call_count == 1
        cypher = session.run.call_args[0][0]

        # Must be MERGE, not CREATE
        assert "MERGE" in cypher
        assert "CREATE" not in cypher.replace("CREATE CONSTRAINT", "").replace("CREATE INDEX", "")

        # Must SET unconditionally (last-writer-wins)
        assert "SET" in cypher

        # Must match on id
        assert "{id: $id}" in cypher

    @pytest.mark.asyncio
    async def test_create_episode_uses_merge(self):
        """create_episode must use MERGE for both node and ABOUT relationship."""
        session = _mock_session()
        driver = _mock_driver(session)

        adapter = Neo4jAdapter.__new__(Neo4jAdapter)
        adapter._driver = driver

        episode = Episode(
            id="ep_test_1",
            type=EpisodeType.OBSERVATION,
            summary="Test observation",
            significance=0.7,
            entity_ids=["ent_1"],
        )

        await adapter.create_episode(episode)

        # Two calls: one for MERGE episode, one for MERGE ABOUT relationship
        assert session.run.call_count == 2

        # First call: MERGE the episode node
        ep_cypher = session.run.call_args_list[0][0][0]
        assert "MERGE" in ep_cypher
        assert "Episode" in ep_cypher

        # Second call: MERGE the ABOUT relationship
        rel_cypher = session.run.call_args_list[1][0][0]
        assert "MERGE" in rel_cypher
        assert "ABOUT" in rel_cypher

    @pytest.mark.asyncio
    async def test_create_episode_no_entities_single_merge(self):
        """Episode with no entity_ids should only produce one MERGE call."""
        session = _mock_session()
        driver = _mock_driver(session)

        adapter = Neo4jAdapter.__new__(Neo4jAdapter)
        adapter._driver = driver

        episode = Episode(
            id="ep_test_2", summary="No entities", entity_ids=[],
        )

        await adapter.create_episode(episode)
        assert session.run.call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_create_entity_same_id(self):
        """
        Concurrent create_entity with same ID should produce
        identical MERGE queries — Neo4j deduplicates at the database.

        This test verifies the application layer sends MERGE (not CREATE)
        for each concurrent call. Actual Neo4j serialization is an
        integration concern, but the Cypher pattern must be correct.
        """
        session = _mock_session()
        driver = _mock_driver(session)

        adapter = Neo4jAdapter.__new__(Neo4jAdapter)
        adapter._driver = driver

        entity = Entity(
            id="ent_concurrent_1", name="Aurora", type=EntityType.OBJECT,
        )

        # Simulate 5 concurrent writes with the same entity ID
        await asyncio.gather(
            adapter.create_entity(entity),
            adapter.create_entity(entity),
            adapter.create_entity(entity),
            adapter.create_entity(entity),
            adapter.create_entity(entity),
        )

        # All 5 calls should use MERGE
        assert session.run.call_count == 5
        for c in session.run.call_args_list:
            cypher = c[0][0]
            assert "MERGE" in cypher
            assert "CREATE" not in cypher.replace("CREATE CONSTRAINT", "").replace("CREATE INDEX", "")

    @pytest.mark.asyncio
    async def test_concurrent_create_entity_different_ids(self):
        """Concurrent writes with different IDs should all succeed independently."""
        session = _mock_session()
        driver = _mock_driver(session)

        adapter = Neo4jAdapter.__new__(Neo4jAdapter)
        adapter._driver = driver

        entities = [
            Entity(id=f"ent_{i}", name=f"Entity {i}") for i in range(5)
        ]

        await asyncio.gather(*[
            adapter.create_entity(e) for e in entities
        ])

        assert session.run.call_count == 5
        ids_passed = set()
        for c in session.run.call_args_list:
            kwargs = c[1]
            ids_passed.add(kwargs["id"])

        assert len(ids_passed) == 5

    @pytest.mark.asyncio
    async def test_concurrent_create_episode_same_id(self):
        """Concurrent episode creates with same ID should all use MERGE."""
        session = _mock_session()
        driver = _mock_driver(session)

        adapter = Neo4jAdapter.__new__(Neo4jAdapter)
        adapter._driver = driver

        episode = Episode(
            id="ep_concurrent_1", summary="Race test",
            entity_ids=["ent_1"],
        )

        await asyncio.gather(
            adapter.create_episode(episode),
            adapter.create_episode(episode),
            adapter.create_episode(episode),
        )

        # 3 episode MERGEs + 3 ABOUT MERGEs = 6 total
        assert session.run.call_count == 6
        for c in session.run.call_args_list:
            cypher = c[0][0]
            assert "MERGE" in cypher

    @pytest.mark.asyncio
    async def test_merge_set_is_unconditional(self):
        """
        MERGE ... SET must be unconditional (no ON CREATE SET / ON MATCH SET).

        Unconditional SET means last-writer-wins, which is the correct
        idempotent behavior — the final state is deterministic regardless
        of execution order.
        """
        session = _mock_session()
        driver = _mock_driver(session)

        adapter = Neo4jAdapter.__new__(Neo4jAdapter)
        adapter._driver = driver

        entity = Entity(id="ent_set_test", name="Test")
        await adapter.create_entity(entity)

        cypher = session.run.call_args[0][0]

        # Should NOT use conditional SET variants
        assert "ON CREATE SET" not in cypher
        assert "ON MATCH SET" not in cypher

        # Should use plain SET
        assert "\n                SET " in cypher or "SET e." in cypher


# ---------------------------------------------------------------------------
# Postgres upsert idempotency
# ---------------------------------------------------------------------------


class TestPostgresUpsertIdempotency:
    """Verify Postgres adapters use ON CONFLICT (upsert) patterns."""

    def test_alias_upsert_sql_pattern(self):
        """upsert_alias uses INSERT ... ON CONFLICT DO UPDATE."""
        import inspect
        from memory_bus.postgres_adapter import PostgresAdapter

        source = inspect.getsource(PostgresAdapter.upsert_alias)
        assert "ON CONFLICT" in source
        assert "DO UPDATE" in source

    def test_plan_upsert_sql_pattern(self):
        """upsert_plan uses INSERT ... ON CONFLICT DO UPDATE."""
        import inspect
        from memory_bus.postgres_adapter import PostgresAdapter

        source = inspect.getsource(PostgresAdapter.upsert_plan)
        assert "ON CONFLICT" in source
        assert "DO UPDATE" in source

    def test_config_upsert_sql_pattern(self):
        """set_config uses INSERT ... ON CONFLICT DO UPDATE."""
        import inspect
        from memory_bus.postgres_adapter import PostgresAdapter

        source = inspect.getsource(PostgresAdapter.set_config)
        assert "ON CONFLICT" in source
        assert "DO UPDATE" in source

    def test_audit_insert_not_upsert(self):
        """
        write_audit uses plain INSERT (no ON CONFLICT).

        Audit log is append-only — duplicate IDs should be rejected,
        not silently upserted. This is intentional: audit rows have
        UUID primary keys that should never collide.
        """
        import inspect
        from memory_bus.postgres_adapter import PostgresAdapter

        source = inspect.getsource(PostgresAdapter.write_audit)
        assert "INSERT INTO" in source
        assert "ON CONFLICT" not in source
