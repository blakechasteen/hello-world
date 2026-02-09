"""
Integration tests (Docker-dependent) — 2026-02-09

Requires: docker compose -f docker-compose.memory.yml up -d

Tests:
1. Seed entity + episode
2. resolve_entity("Aurora") returns hive_aurora candidate
3. Query episodes for hive_aurora returns results
4. Audit row written for resolve + query + store
5. memory://pressure returns tier and signals
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

# Skip entire module if not running integration tests
INTEGRATION = os.environ.get("MEMORY_BUS_INTEGRATION", "0") == "1"
pytestmark = pytest.mark.skipif(not INTEGRATION, reason="Set MEMORY_BUS_INTEGRATION=1")


@pytest_asyncio.fixture
async def bus():
    """Create a MemoryBus connected to local Docker services."""
    from memory_bus.tools import MemoryBus

    bus = MemoryBus(
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "hololoom"),
        pg_dsn=os.environ.get(
            "PG_DSN", "postgresql://hololoom:hololoom@localhost:5432/hololoom"
        ),
    )
    await bus.connect()
    yield bus
    await bus.close()


@pytest.mark.asyncio
async def test_seed_entity_and_episode(bus):
    """1. Seed entity + episode via memory_store."""
    result = await bus.memory_store(
        entities=[
            {
                "id": "ent_hive_aurora",
                "type": "object",
                "name": "Hive Aurora",
                "importance": 0.8,
                "loom_origin": "integration_test",
                "properties": {"species": "apis_mellifera", "location": "north_yard"},
            }
        ],
        episodes=[
            {
                "id": "ep_integration_001",
                "type": "observation",
                "summary": "Inspected Hive Aurora: 8 frames brood, healthy queen spotted",
                "content": "Full inspection: 8 brood, 3 honey, queen on frame 5, no signs of disease.",
                "significance": 0.7,
                "loom_origin": "integration_test",
                "entity_ids": ["ent_hive_aurora"],
            }
        ],
        aliases=[
            {
                "alias": "aurora",
                "entity_id": "ent_hive_aurora",
                "loom_id": "integration_test",
                "confidence": 0.95,
            }
        ],
        loom_id="integration_test",
        loop_id="loop_int_001",
    )

    assert "ent_hive_aurora" in result["entities"]
    assert "ep_integration_001" in result["episodes"]


@pytest.mark.asyncio
async def test_resolve_entity_aurora(bus):
    """2. resolve_entity("Aurora") returns hive_aurora candidate."""
    # Ensure seeded
    await test_seed_entity_and_episode(bus)

    resolve = await bus.memory_resolve_entity(
        name="Aurora",
        loom_id="integration_test",
        loop_id="loop_int_002",
    )

    assert resolve.best_match is not None
    assert resolve.best_match.entity_id == "ent_hive_aurora"
    assert resolve.best_match.confidence >= 0.8


@pytest.mark.asyncio
async def test_query_episodes_for_aurora(bus):
    """3. Query episodes for hive_aurora returns results."""
    await test_seed_entity_and_episode(bus)

    result = await bus.memory_query(
        text="",
        entity_ids=["ent_hive_aurora"],
        loom_id="integration_test",
        loop_id="loop_int_003",
    )

    assert len(result.items) > 0
    episode_items = [i for i in result.items if i.kind == "episode"]
    assert len(episode_items) > 0
    assert result.formatted_context != ""


@pytest.mark.asyncio
async def test_audit_rows_written(bus):
    """4. Audit row written for resolve + query + store."""
    await test_seed_entity_and_episode(bus)
    await test_resolve_entity_aurora(bus)
    await test_query_episodes_for_aurora(bus)

    rows = await bus._pg.get_audit_rows(loom_id="integration_test", limit=20)
    actions = {r.action.value for r in rows}

    assert "promote" in actions  # from store
    assert "resolve" in actions  # from resolve_entity
    assert "query" in actions  # from memory_query


@pytest.mark.asyncio
async def test_pressure_resource(bus):
    """5. memory://pressure returns tier and signals."""
    status = bus.memory_pressure()

    assert status.tier is not None
    assert status.signals.context_window_size > 0
    assert "priming_strategy" in status.policy
    assert "plan_write_through" in status.policy


@pytest.mark.asyncio
async def test_memory_status(bus):
    """memory://status returns health info."""
    status = await bus.memory_status()
    assert status["status"] == "healthy"
    assert status["neo4j_connected"] is True
    assert status["postgres_connected"] is True


@pytest.mark.asyncio
async def test_memory_explain(bus):
    """memory_explain returns provenance for a known item."""
    await test_seed_entity_and_episode(bus)

    result = await bus.memory_explain(
        item_id="ep_integration_001",
        loom_id="integration_test",
        loop_id="loop_int_explain",
    )

    assert result.item_id == "ep_integration_001"
    assert result.confidence > 0
    assert result.status == "confirmed"
    # Episode should derive from entity
    assert "ent_hive_aurora" in result.derived_from


@pytest.mark.asyncio
async def test_semantic_stub(bus):
    """Semantic query returns 'not available' and does NOT inject blobs."""
    result = await bus.memory_query(
        text="something with no matching entities at all xyz123",
        loom_id="integration_test_semantic",
    )

    # Should fall through to semantic stub
    assert len(result.items) == 0
    assert any("not available" in w.lower() or "semantic" in w.lower() for w in result.warnings)
    assert result.formatted_context == ""
