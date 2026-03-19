"""
Memory Bus Integration Tests — 8 tests validating the spec against real backends.

Requires Docker: Neo4j 5 (bolt://localhost:7687) + Postgres 16 (localhost:5432/hololoom).
Start with: docker compose up -d neo4j postgres

All tests marked @requires_docker + @integration.
"""

import pytest

from hololoom.memory.bus import MemoryBus, MemoryItem, MemoryQuery
from hololoom.memory.bus_config import MemoryBusConfig, ResolutionPath, AuditAction
from hololoom.memory.neo4j_schema import ensure_schema, NODE_ENTITY, NODE_EPISODE
from hololoom.memory.bee_data import (
    load_bee_data,
    HIVE_ENTITIES,
    ENTITY_ALIASES,
    EPISODES,
    CAUSAL_LINKS,
)


pytestmark = [pytest.mark.requires_docker, pytest.mark.integration]


# =========================================================================
# Test 1: Neo4j Connection
# =========================================================================

class TestNeo4jConnection:
    """Verify Neo4j is reachable and responsive."""

    async def test_neo4j_connection(self, neo4j_driver):
        """Connect to Neo4j, run RETURN 1, verify response."""
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 AS n")
            record = result.single()
            assert record["n"] == 1

    async def test_neo4j_schema_setup(self, neo4j_driver):
        """Verify schema constraints and indexes are created."""
        result = ensure_schema(neo4j_driver)
        # Either executed or skipped (idempotent)
        assert result["executed"] + result["skipped"] > 0
        assert len(result["errors"]) == 0


# =========================================================================
# Test 2: Postgres Connection
# =========================================================================

class TestPostgresConnection:
    """Verify Postgres is reachable and tables exist."""

    async def test_postgres_connection(self, postgres_store):
        """Connect to Postgres, verify all 5 tables exist."""
        health = await postgres_store.health_check()
        assert health["status"] == "connected"

        expected_tables = ["facts", "plans", "configs", "memory_audit", "entity_aliases"]
        for table in expected_tables:
            assert table in health["tables"], f"Missing table: {table}"
            assert health["tables"][table]["exists"], f"Table {table} does not exist"


# =========================================================================
# Test 3: Entity Creation and Retrieval
# =========================================================================

class TestEntityCreationRetrieval:
    """Create Entity nodes via bus.store(), retrieve by ID."""

    async def test_store_and_query_entity(self, memory_bus):
        """Store an entity and retrieve it by exact ID."""
        # Store
        node_id = await memory_bus.store(MemoryItem(
            content="Test Hive",
            memory_type="entity",
            properties={"type": "Hive", "genetics": "Carniolan"},
            importance=0.8,
            loom_id="bee_test",
        ))
        assert node_id is not None

        # Query by exact ID
        result = await memory_bus.query(MemoryQuery(
            intent="Get test hive",
            entity_ids=[node_id],
        ))
        assert result.resolution_path == ResolutionPath.EXACT.value
        assert len(result.items) == 1
        assert result.items[0]["id"] == node_id

    async def test_store_episode_linked_to_entity(self, memory_bus):
        """Store an episode linked to an entity via ABOUT relationship."""
        # Create entity first
        entity_id = await memory_bus.store(MemoryItem(
            content="Linked Hive",
            memory_type="entity",
            properties={"type": "Hive"},
            loom_id="bee_test",
        ))

        # Create episode linked to entity
        episode_id = await memory_bus.store(MemoryItem(
            content="Inspection of Linked Hive — strong population",
            memory_type="episodic",
            entity_ids=[entity_id],
            properties={"type": "inspection"},
            importance=0.7,
            loom_id="bee_test",
        ))
        assert episode_id is not None

        # Verify the ABOUT relationship exists
        with memory_bus._neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (ep:Episode {id: $ep_id})-[:ABOUT]->(e:Entity {id: $e_id})
                RETURN count(*) AS cnt
                """,
                ep_id=episode_id,
                e_id=entity_id,
            )
            assert result.single()["cnt"] == 1


# =========================================================================
# Test 4: Entity Resolution with Aliases
# =========================================================================

class TestEntityResolution:
    """Resolve names/aliases via Postgres alias table."""

    async def test_resolve_entity_exact_alias(self, memory_bus):
        """resolve_entity('Aurora') returns hive_aurora with confidence."""
        # Load aliases
        await memory_bus._postgres.add_aliases_batch([
            ("Aurora", "hive_aurora", "bee", 1.0),
            ("Hive Aurora", "hive_aurora", "bee", 1.0),
            ("aurora", "hive_aurora", "bee", 0.9),
        ])

        candidates = await memory_bus.resolve_entity("Aurora")
        assert len(candidates) >= 1
        assert candidates[0]["entity_id"] == "hive_aurora"
        assert candidates[0]["confidence"] == 1.0

    async def test_resolve_entity_case_insensitive(self, memory_bus):
        """resolve_entity('aurora') matches case-insensitively."""
        await memory_bus._postgres.add_alias("Aurora", "hive_aurora", "bee", 1.0)

        candidates = await memory_bus.resolve_entity("aurora")
        assert len(candidates) >= 1
        assert candidates[0]["entity_id"] == "hive_aurora"

    async def test_resolve_entity_no_match(self, memory_bus):
        """resolve_entity('nonexistent') returns empty list."""
        candidates = await memory_bus.resolve_entity("nonexistent_hive_xyz")
        assert candidates == []


# =========================================================================
# Test 5: Query Cascade — Structured
# =========================================================================

class TestQueryCascadeStructured:
    """Verify structured Cypher queries resolve at Level 2."""

    async def test_query_by_entity_type(self, memory_bus):
        """Query entities by type resolves as 'structured'."""
        # Create some entities
        for name in ["Cascade Hive A", "Cascade Hive B"]:
            await memory_bus.store(MemoryItem(
                content=name,
                memory_type="entity",
                properties={"type": "Hive"},
                loom_id="bee_test",
            ))

        result = await memory_bus.query(MemoryQuery(
            intent="Find all hives",
            entity_type="Hive",
        ))
        assert result.resolution_path == ResolutionPath.STRUCTURED.value
        assert len(result.items) >= 2

    async def test_query_episodes_by_type(self, memory_bus):
        """Query episodes by memory_type resolves as 'structured'."""
        await memory_bus.store(MemoryItem(
            content="Test inspection episode",
            memory_type="episodic",
            properties={"type": "inspection"},
            importance=0.6,
            loom_id="bee_test",
        ))

        result = await memory_bus.query(MemoryQuery(
            intent="Recent inspections",
            memory_type="episodic",
        ))
        assert result.resolution_path == ResolutionPath.STRUCTURED.value
        assert len(result.items) >= 1


# =========================================================================
# Test 6: Audit Log Captures Operations
# =========================================================================

class TestAuditLog:
    """Verify every query and store operation is audited."""

    async def test_query_creates_audit_entry(self, memory_bus):
        """A query operation logs to memory_audit."""
        # Run a query
        await memory_bus.query(MemoryQuery(intent="audit test query"))

        # Check audit log
        audits = await memory_bus._postgres.query_audit(action=AuditAction.QUERY.value)
        assert len(audits) >= 1
        latest = audits[0]
        assert latest["action"] == AuditAction.QUERY.value
        assert latest["details"] is not None

    async def test_store_creates_audit_entry(self, memory_bus):
        """A store operation logs to memory_audit."""
        await memory_bus.store(MemoryItem(
            content="Audited entity",
            memory_type="entity",
            properties={"type": "Test"},
            loom_id="bee_test",
        ))

        audits = await memory_bus._postgres.query_audit(action=AuditAction.PROMOTE.value)
        assert len(audits) >= 1
        latest = audits[0]
        assert latest["action"] == AuditAction.PROMOTE.value

    async def test_audit_captures_resolution_path(self, memory_bus):
        """Audit log records which resolution path was used."""
        # Create entity, then query by exact ID
        node_id = await memory_bus.store(MemoryItem(
            content="Path-tracked entity",
            memory_type="entity",
            properties={"type": "Test"},
            loom_id="bee_test",
        ))
        await memory_bus.query(MemoryQuery(
            intent="exact lookup",
            entity_ids=[node_id],
        ))

        audits = await memory_bus._postgres.query_audit(action=AuditAction.QUERY.value)
        exact_audits = [a for a in audits if a.get("resolution_path") == ResolutionPath.EXACT.value]
        assert len(exact_audits) >= 1


# =========================================================================
# Test 7: Bee Data Load and Query
# =========================================================================

class TestBeeDataLoadAndQuery:
    """Load real bee inspection data and validate queries."""

    async def test_load_bee_data(self, memory_bus):
        """load_bee_data() populates Neo4j and Postgres."""
        stats = await load_bee_data(memory_bus)

        assert stats["entities"] == len(HIVE_ENTITIES) + 3 + 1  # hives + locations + apiary
        assert stats["episodes"] == len(EPISODES)
        assert stats["aliases"] == len(ENTITY_ALIASES)
        assert stats["facts"] > 0
        assert stats["relationships"] > 0

    async def test_query_aurora_by_name(self, memory_bus):
        """After loading, resolve_entity('Aurora') finds hive_aurora."""
        await load_bee_data(memory_bus)

        candidates = await memory_bus.resolve_entity("Aurora")
        assert len(candidates) >= 1
        assert candidates[0]["entity_id"] == "hive_aurora"

    async def test_query_aurora_episodes(self, memory_bus):
        """Query episodes about Aurora returns inspections and treatments."""
        await load_bee_data(memory_bus)

        result = await memory_bus.query(MemoryQuery(
            intent="Aurora inspection history",
            entity_ids=["hive_aurora"],
        ))
        assert len(result.items) >= 1
        assert result.resolution_path == ResolutionPath.EXACT.value

    async def test_causal_chain_exists(self, memory_bus):
        """Treatment → inspection CAUSED relationship exists."""
        await load_bee_data(memory_bus)

        with memory_bus._neo4j_driver.session() as session:
            result = session.run(
                """
                MATCH (a:Episode {id: $src})-[:CAUSED]->(b:Episode {id: $dst})
                RETURN count(*) AS cnt
                """,
                src=CAUSAL_LINKS[0][0],
                dst=CAUSAL_LINKS[0][1],
            )
            assert result.single()["cnt"] == 1

    async def test_entity_graph_navigable(self, memory_bus):
        """Hive → Location via LOCATED_AT, Hive → Apiary via PART_OF."""
        await load_bee_data(memory_bus)

        entities = await memory_bus.entities(connected_to="hive_aurora", max_hops=2)
        entity_ids = [e.get("id") for e in entities]

        # Aurora should connect to north_yard (LOCATED_AT) and apiary_main (PART_OF)
        assert "loc_north_yard" in entity_ids
        assert "apiary_main" in entity_ids


# =========================================================================
# Test 8: Invariants
# =========================================================================

class TestInvariants:
    """Verify the 7 architectural invariants hold."""

    async def test_structured_before_semantic(self, memory_bus):
        """Invariant 2: Structured resolves before semantic.

        A semantic-only query returns empty (not available) while
        structured queries return results.
        """
        await memory_bus.store(MemoryItem(
            content="Invariant Test Hive",
            memory_type="entity",
            properties={"type": "Hive"},
            loom_id="bee_test",
        ))

        # Semantic-only query: should return "not available" (empty, semantic path)
        semantic_result = await memory_bus.query(MemoryQuery(
            intent="",
            semantic_text="something about hives and bees",
        ))
        assert semantic_result.resolution_path == ResolutionPath.SEMANTIC.value
        assert len(semantic_result.items) == 0

        # Structured query: should return results
        structured_result = await memory_bus.query(MemoryQuery(
            intent="Find hives",
            entity_type="Hive",
        ))
        assert structured_result.resolution_path == ResolutionPath.STRUCTURED.value
        assert len(structured_result.items) >= 1

    async def test_every_result_has_token_estimate(self, memory_bus):
        """Invariant 3: Every memory injection has a token budget."""
        await memory_bus.store(MemoryItem(
            content="Token budget test",
            memory_type="entity",
            properties={"type": "Test"},
            loom_id="bee_test",
        ))

        result = await memory_bus.query(MemoryQuery(
            intent="token test",
            entity_type="Test",
        ))
        assert hasattr(result, "token_estimate")
        assert isinstance(result.token_estimate, int)
        assert result.token_estimate >= 0

    async def test_store_creates_audit(self, memory_bus):
        """Every store() creates an audit log entry (provenance)."""
        node_id = await memory_bus.store(MemoryItem(
            content="Audit invariant test",
            memory_type="entity",
            properties={"type": "Test"},
            loom_id="bee_test",
        ))

        audits = await memory_bus._postgres.query_audit(action=AuditAction.PROMOTE.value)
        node_ids_in_audit = [
            a["details"].get("node_id") if isinstance(a["details"], dict) else None
            for a in audits
        ]
        assert node_id in node_ids_in_audit

    async def test_resolve_returns_confidence(self, memory_bus):
        """resolve_entity returns candidates with confidence scores."""
        await memory_bus._postgres.add_alias("ConfTest", "entity_conf_test", "bee", 0.95)

        candidates = await memory_bus.resolve_entity("ConfTest")
        assert len(candidates) >= 1
        assert "confidence" in candidates[0]
        assert 0.0 <= candidates[0]["confidence"] <= 1.0

    async def test_status_tracks_resolution_distribution(self, memory_bus):
        """Status endpoint reports query resolution path distribution."""
        # Run some queries
        await memory_bus.store(MemoryItem(
            content="Status test", memory_type="entity",
            properties={"type": "Test"}, loom_id="bee_test",
        ))
        await memory_bus.query(MemoryQuery(intent="status test", entity_type="Test"))

        status = memory_bus.status()
        assert "resolution_distribution" in status
        assert "query_count" in status
        assert status["query_count"] >= 1
