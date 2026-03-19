"""
Unit tests for hololoom.core.memory.bus — MemoryBus.

Tests the MemoryBus class methods without real Neo4j/Postgres backends.
Uses mocks for database drivers and validates:
  - Lifecycle (initialize, close, context manager)
  - Query cascade (4-level: exact → structured → graph → semantic)
  - Token budgeting and truncation
  - _build_result stats tracking
  - _estimate_tokens correctness
  - status() and pressure() reporting
  - _require_neo4j / _require_postgres guards
  - store() node creation and label mapping
  - explain() provenance chain
  - entities() browsing
  - _audit_query error handling
  - StoredItem wrapping (AP-1 anti-pattern guard)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

from hololoom.core.memory.bus import (
    MemoryBus,
    MemoryBusConfig,
    MemoryItem,
    MemoryQuery,
    MemoryResult,
)
from hololoom.core.memory.bus_config import (
    PressureTier,
    ResolutionPath,
)
from hololoom.core.memory.lite_bus import StoredItem


# =============================================================================
# Helpers
# =============================================================================

def _make_bus(neo4j_driver=None, postgres=None, config=None):
    """Create a MemoryBus with injected mocks, bypassing initialize()."""
    bus = MemoryBus(config=config or MemoryBusConfig())
    bus._neo4j_driver = neo4j_driver
    bus._postgres = postgres
    bus._initialized = True
    return bus


def _mock_neo4j_session(records=None):
    """Build a mock Neo4j driver that returns the given records from session.run()."""
    records = records or []
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter(records))
    mock_result.single.return_value = records[0] if records else None

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session
    return mock_driver, mock_session


# =============================================================================
# Data Types
# =============================================================================

class TestDataTypes:
    """Verify dataclass defaults for MemoryQuery, MemoryItem, MemoryResult."""

    def test_memory_query_defaults(self):
        q = MemoryQuery(intent="test")
        assert q.max_results == 5
        assert q.min_importance == 0.1
        assert q.entity_ids is None
        assert q.semantic_text is None

    def test_memory_item_defaults(self):
        item = MemoryItem(content="hello", memory_type="entity")
        assert item.importance == 0.5
        assert item.scope == "local"
        assert item.entity_ids is None
        assert item.loom_id is None

    def test_memory_result_defaults(self):
        q = MemoryQuery(intent="x")
        r = MemoryResult(items=[], query=q, resolution_path="exact", token_estimate=0)
        assert r.truncated is False


# =============================================================================
# Lifecycle
# =============================================================================

class TestLifecycle:
    """Test MemoryBus lifecycle methods."""

    def test_default_config(self):
        bus = MemoryBus()
        assert bus.config is not None
        assert bus._initialized is False
        assert bus._neo4j_driver is None
        assert bus._postgres is None

    def test_custom_config(self):
        cfg = MemoryBusConfig(token_budget_absolute=4000)
        bus = MemoryBus(config=cfg)
        assert bus.config.token_budget_absolute == 4000

    @pytest.mark.asyncio
    async def test_initialize_without_backends(self):
        """Initialize with no neo4j/psycopg available degrades gracefully."""
        with patch("hololoom.core.memory.bus.NEO4J_AVAILABLE", False), \
             patch("hololoom.core.memory.bus.PSYCOPG_AVAILABLE", False):
            bus = MemoryBus()
            status = await bus.initialize()
            assert status["neo4j"] == "unavailable"
            assert status["postgres"] == "unavailable"
            assert bus._initialized is True

    @pytest.mark.asyncio
    async def test_close_clears_state(self):
        mock_driver = MagicMock()
        mock_postgres = AsyncMock()
        bus = _make_bus(neo4j_driver=mock_driver, postgres=mock_postgres)

        await bus.close()
        mock_driver.close.assert_called_once()
        mock_postgres.close.assert_called_once()
        assert bus._neo4j_driver is None
        assert bus._postgres is None
        assert bus._initialized is False

    @pytest.mark.asyncio
    async def test_close_without_backends(self):
        """close() works even if no backends were connected."""
        bus = MemoryBus()
        await bus.close()  # Should not raise
        assert bus._initialized is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Async context manager calls initialize and close."""
        with patch("hololoom.core.memory.bus.NEO4J_AVAILABLE", False), \
             patch("hololoom.core.memory.bus.PSYCOPG_AVAILABLE", False):
            async with MemoryBus() as bus:
                assert bus._initialized is True
            assert bus._initialized is False


# =============================================================================
# Require Guards
# =============================================================================

class TestRequireGuards:
    """Test _require_neo4j and _require_postgres raise when backends missing."""

    def test_require_neo4j_raises(self):
        bus = MemoryBus()
        with pytest.raises(RuntimeError, match="Neo4j not available"):
            bus._require_neo4j()

    def test_require_neo4j_passes(self):
        bus = _make_bus(neo4j_driver=MagicMock())
        bus._require_neo4j()  # Should not raise

    def test_require_postgres_raises(self):
        bus = MemoryBus()
        with pytest.raises(RuntimeError, match="Postgres not available"):
            bus._require_postgres()

    def test_require_postgres_passes(self):
        bus = _make_bus(postgres=MagicMock())
        bus._require_postgres()  # Should not raise


# =============================================================================
# Token Estimation
# =============================================================================

class TestTokenEstimation:
    """Test _estimate_tokens static method."""

    def test_estimate_tokens_basic(self):
        item = {"content": "a" * 100}
        tokens = MemoryBus._estimate_tokens(item)
        # str(item) is longer than 100 due to dict repr
        assert tokens > 0

    def test_estimate_tokens_empty(self):
        tokens = MemoryBus._estimate_tokens({})
        assert tokens >= 1  # min 1

    def test_estimate_tokens_proportional(self):
        small = MemoryBus._estimate_tokens({"x": "hi"})
        large = MemoryBus._estimate_tokens({"x": "a" * 1000})
        assert large > small


# =============================================================================
# _build_result
# =============================================================================

class TestBuildResult:
    """Test _build_result token budgeting and stats."""

    def test_build_result_basic(self):
        bus = _make_bus(neo4j_driver=MagicMock())
        items = [{"id": "1", "content": "short"}]
        q = MemoryQuery(intent="test")
        result = bus._build_result(items, q, ResolutionPath.EXACT)

        assert result.resolution_path == "exact"
        assert result.token_estimate > 0
        assert result.truncated is False
        assert len(result.items) == 1

    def test_build_result_tracks_stats(self):
        bus = _make_bus(neo4j_driver=MagicMock())
        items = [{"id": "1", "content": "hello"}]
        q = MemoryQuery(intent="test")

        assert bus._query_count == 0
        bus._build_result(items, q, ResolutionPath.EXACT)
        assert bus._query_count == 1
        assert bus._resolution_counts["exact"] == 1
        assert bus._total_tokens_used > 0

    def test_build_result_truncates_over_budget(self):
        cfg = MemoryBusConfig(token_budget_absolute=10)
        bus = _make_bus(neo4j_driver=MagicMock(), config=cfg)
        # Create items that exceed the tiny budget
        items = [
            {"id": str(i), "content": "x" * 200}
            for i in range(10)
        ]
        q = MemoryQuery(intent="test")
        result = bus._build_result(items, q, ResolutionPath.STRUCTURED)

        assert result.truncated is True
        assert len(result.items) < 10
        assert result.token_estimate <= cfg.token_budget_absolute

    def test_build_result_within_budget_not_truncated(self):
        cfg = MemoryBusConfig(token_budget_absolute=100_000)
        bus = _make_bus(neo4j_driver=MagicMock(), config=cfg)
        items = [{"id": "1", "content": "small"}]
        q = MemoryQuery(intent="test")
        result = bus._build_result(items, q, ResolutionPath.EXACT)

        assert result.truncated is False
        assert len(result.items) == 1

    def test_build_result_empty_items(self):
        bus = _make_bus(neo4j_driver=MagicMock())
        q = MemoryQuery(intent="test")
        result = bus._build_result([], q, ResolutionPath.GRAPH)

        assert result.items == []
        assert result.token_estimate == 0
        assert result.truncated is False


# =============================================================================
# Query Cascade
# =============================================================================

class TestQueryCascade:
    """Test the 4-level query cascade."""

    @pytest.mark.asyncio
    async def test_query_requires_neo4j(self):
        bus = MemoryBus()
        with pytest.raises(RuntimeError, match="Neo4j not available"):
            await bus.query(MemoryQuery(intent="test"))

    @pytest.mark.asyncio
    async def test_query_exact_match(self):
        """Level 1: entity_ids triggers exact lookup."""
        record = {"e": {"id": "abc", "name": "Test"}}
        driver, session = _mock_neo4j_session([record])
        bus = _make_bus(neo4j_driver=driver)

        result = await bus.query(MemoryQuery(
            intent="find entity",
            entity_ids=["abc"],
        ))

        assert result.resolution_path == "exact"
        assert len(result.items) == 1
        session.run.assert_called()

    @pytest.mark.asyncio
    async def test_query_structured_by_type(self):
        """Level 2: entity_type triggers structured query."""
        record = {"n": {"id": "1", "type": "Hive", "timestamp": "2025-01-01"}}
        driver, session = _mock_neo4j_session([record])
        bus = _make_bus(neo4j_driver=driver)

        result = await bus.query(MemoryQuery(
            intent="find hives",
            entity_type="Hive",
        ))

        assert result.resolution_path == "structured"
        assert len(result.items) == 1

    @pytest.mark.asyncio
    async def test_query_structured_by_memory_type(self):
        """Level 2: memory_type triggers structured query."""
        record = {"n": {"id": "ep1", "summary": "test", "timestamp": "2025-01-01"}}
        driver, session = _mock_neo4j_session([record])
        bus = _make_bus(neo4j_driver=driver)

        result = await bus.query(MemoryQuery(
            intent="find episodes",
            memory_type="episodic",
        ))

        assert result.resolution_path == "structured"

    @pytest.mark.asyncio
    async def test_query_structured_by_time(self):
        """Level 2: time_after triggers structured query."""
        record = {"n": {"id": "1", "timestamp": "2025-06-01"}}
        driver, session = _mock_neo4j_session([record])
        bus = _make_bus(neo4j_driver=driver)

        result = await bus.query(MemoryQuery(
            intent="recent",
            time_after="2025-01-01",
        ))

        assert result.resolution_path == "structured"

    @pytest.mark.asyncio
    async def test_query_graph_fallback(self):
        """Level 3: intent-only falls through to graph traversal."""
        # Exact: no entity_ids, Structured: no filters → Graph via intent
        record = {"node": {"id": "1", "name": "Aurora"}, "score": 0.9}
        driver, session = _mock_neo4j_session([record])
        bus = _make_bus(neo4j_driver=driver)

        result = await bus.query(MemoryQuery(intent="Aurora"))

        assert result.resolution_path == "graph"

    @pytest.mark.asyncio
    async def test_query_semantic_deferred(self):
        """Level 4: semantic_text returns empty (deferred)."""
        # No entity_ids, no filters, empty intent → falls to semantic
        driver, _ = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        result = await bus.query(MemoryQuery(
            intent="",
            semantic_text="something about bees",
        ))

        assert result.resolution_path == "semantic"
        assert len(result.items) == 0

    @pytest.mark.asyncio
    async def test_query_no_match(self):
        """When nothing matches, resolution_path is 'none'."""
        driver, _ = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        result = await bus.query(MemoryQuery(intent=""))

        assert result.resolution_path == "none"
        assert len(result.items) == 0

    @pytest.mark.asyncio
    async def test_query_exact_empty_falls_through(self):
        """If exact lookup returns empty, cascade continues to next level."""
        driver, session = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        # entity_ids set but no records found → falls through
        # No structured filters, no intent content → ends at 'none'
        result = await bus.query(MemoryQuery(
            intent="",
            entity_ids=["nonexistent"],
        ))

        assert result.resolution_path == "none"


# =============================================================================
# Query — Graph Level 3 (fulltext fallback)
# =============================================================================

class TestQueryGraph:
    """Test the graph traversal path with fulltext index fallback."""

    @pytest.mark.asyncio
    async def test_graph_fulltext_exception_falls_back(self):
        """When fulltext index is unavailable, falls back to CONTAINS."""
        call_count = 0

        def mock_run(cypher, **params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is fulltext — raise to simulate no index
                raise Exception("No fulltext index")
            # Second call is CONTAINS fallback
            record = MagicMock()
            record.__getitem__ = lambda self, key: {"id": "1", "name": "Aurora"}
            mock_result = MagicMock()
            mock_result.__iter__ = MagicMock(return_value=iter([record]))
            return mock_result

        mock_session = MagicMock()
        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        bus = _make_bus(neo4j_driver=mock_driver)

        result = await bus.query(MemoryQuery(intent="Aurora"))
        assert result.resolution_path == "graph"
        assert len(result.items) >= 1


# =============================================================================
# Store
# =============================================================================

class TestStore:
    """Test store() method — node creation and label mapping."""

    @pytest.mark.asyncio
    async def test_store_requires_neo4j(self):
        bus = MemoryBus()
        with pytest.raises(RuntimeError, match="Neo4j not available"):
            await bus.store(MemoryItem(content="x", memory_type="entity"))

    @pytest.mark.asyncio
    async def test_store_returns_node_id(self):
        driver, session = _mock_neo4j_session()
        bus = _make_bus(neo4j_driver=driver)

        node_id = await bus.store(MemoryItem(
            content="Test Entity",
            memory_type="entity",
            properties={"type": "Hive"},
        ))

        assert node_id is not None
        assert len(node_id) == 12  # uuid4()[:12]
        session.run.assert_called()

    @pytest.mark.asyncio
    async def test_store_episodic_uses_episode_label(self):
        driver, session = _mock_neo4j_session()
        bus = _make_bus(neo4j_driver=driver)

        await bus.store(MemoryItem(
            content="An inspection",
            memory_type="episodic",
            properties={"type": "inspection"},
            importance=0.7,
        ))

        # Verify the CREATE statement references Episode label
        create_call = session.run.call_args
        assert "Episode" in create_call[0][0]

    @pytest.mark.asyncio
    async def test_store_factual_uses_fact_label(self):
        driver, session = _mock_neo4j_session()
        bus = _make_bus(neo4j_driver=driver)

        await bus.store(MemoryItem(
            content="Bees are pollinators",
            memory_type="factual",
            properties={"domain": "biology"},
            importance=0.9,
        ))

        create_call = session.run.call_args
        assert "Fact" in create_call[0][0]

    @pytest.mark.asyncio
    async def test_store_plan_uses_plan_label(self):
        driver, session = _mock_neo4j_session()
        bus = _make_bus(neo4j_driver=driver)

        await bus.store(MemoryItem(
            content="Harvest honey next week",
            memory_type="plan",
        ))

        create_call = session.run.call_args
        assert "Plan" in create_call[0][0]

    @pytest.mark.asyncio
    async def test_store_unknown_type_defaults_to_entity(self):
        driver, session = _mock_neo4j_session()
        bus = _make_bus(neo4j_driver=driver)

        await bus.store(MemoryItem(
            content="Unknown thing",
            memory_type="custom_unknown",
        ))

        create_call = session.run.call_args
        assert "Entity" in create_call[0][0]

    @pytest.mark.asyncio
    async def test_store_with_entity_ids_creates_relationships(self):
        """When entity_ids provided and label != Entity, creates ABOUT rels."""
        driver, session = _mock_neo4j_session()
        bus = _make_bus(neo4j_driver=driver)

        await bus.store(MemoryItem(
            content="Observation",
            memory_type="episodic",
            entity_ids=["ent-1", "ent-2"],
        ))

        # Should have been called for CREATE + 2x relationship
        assert session.run.call_count >= 3

    @pytest.mark.asyncio
    async def test_store_entity_type_no_about_relationships(self):
        """Entity-type items should NOT create ABOUT relationships to themselves."""
        driver, session = _mock_neo4j_session()
        bus = _make_bus(neo4j_driver=driver)

        await bus.store(MemoryItem(
            content="Some entity",
            memory_type="entity",
            entity_ids=["ent-1"],
        ))

        # Only the CREATE call, no relationship calls
        assert session.run.call_count == 1

    @pytest.mark.asyncio
    async def test_store_with_postgres_audits(self):
        """store() logs audit when Postgres is available."""
        driver, _ = _mock_neo4j_session()
        mock_pg = AsyncMock()
        bus = _make_bus(neo4j_driver=driver, postgres=mock_pg)

        await bus.store(MemoryItem(
            content="Audited",
            memory_type="entity",
        ))

        mock_pg.log_audit.assert_called_once()
        call_kwargs = mock_pg.log_audit.call_args[1]
        assert call_kwargs["action"] == "promote"

    @pytest.mark.asyncio
    async def test_store_fact_mirrors_to_postgres(self):
        """Factual items should be mirrored to Postgres fast lookup."""
        driver, _ = _mock_neo4j_session()
        mock_pg = AsyncMock()
        bus = _make_bus(neo4j_driver=driver, postgres=mock_pg)

        await bus.store(MemoryItem(
            content="Fact content",
            memory_type="factual",
            properties={"domain": "science"},
            importance=0.8,
        ))

        mock_pg.store_fact.assert_called_once()
        mock_pg.log_audit.assert_called_once()


# =============================================================================
# Status & Pressure
# =============================================================================

class TestStatusAndPressure:
    """Test status() and pressure() reporting."""

    def test_status_initial(self):
        bus = MemoryBus()
        s = bus.status()
        assert s["initialized"] is False
        assert s["neo4j"] == "unavailable"
        assert s["postgres"] == "unavailable"
        assert s["query_count"] == 0

    def test_status_with_backends(self):
        bus = _make_bus(neo4j_driver=MagicMock(), postgres=MagicMock())
        s = bus.status()
        assert s["initialized"] is True
        assert s["neo4j"] == "connected"
        assert s["postgres"] == "connected"

    def test_status_after_queries(self):
        bus = _make_bus(neo4j_driver=MagicMock())
        # Simulate some queries
        bus._query_count = 10
        bus._resolution_counts["exact"] = 5
        bus._resolution_counts["structured"] = 3
        bus._resolution_counts["semantic"] = 2
        bus._total_tokens_used = 500

        s = bus.status()
        assert s["query_count"] == 10
        assert s["resolution_distribution"]["exact"] == 5
        assert s["total_tokens_used"] == 500
        assert s["semantic_fallback_pct"] == 0.2
        assert s["semantic_target_met"] is True  # 0.2 <= 0.20

    def test_status_semantic_target_not_met(self):
        bus = _make_bus(neo4j_driver=MagicMock())
        bus._query_count = 10
        bus._resolution_counts["semantic"] = 5  # 50% semantic
        s = bus.status()
        assert s["semantic_target_met"] is False

    def test_pressure_nominal(self):
        bus = _make_bus(neo4j_driver=MagicMock())
        bus._total_tokens_used = 0
        p = bus.pressure()
        assert p["tier"] == PressureTier.NOMINAL.value
        assert p["utilization_pct"] == 0.0

    def test_pressure_elevated(self):
        cfg = MemoryBusConfig(token_budget_absolute=1000)
        bus = _make_bus(neo4j_driver=MagicMock(), config=cfg)
        bus._total_tokens_used = 600  # 60%
        p = bus.pressure()
        assert p["tier"] == PressureTier.ELEVATED.value

    def test_pressure_high(self):
        cfg = MemoryBusConfig(token_budget_absolute=1000)
        bus = _make_bus(neo4j_driver=MagicMock(), config=cfg)
        bus._total_tokens_used = 800  # 80%
        p = bus.pressure()
        assert p["tier"] == PressureTier.HIGH.value

    def test_pressure_critical(self):
        cfg = MemoryBusConfig(token_budget_absolute=1000)
        bus = _make_bus(neo4j_driver=MagicMock(), config=cfg)
        bus._total_tokens_used = 950  # 95%
        p = bus.pressure()
        assert p["tier"] == PressureTier.CRITICAL.value

    def test_pressure_budget_info(self):
        cfg = MemoryBusConfig(token_budget_absolute=8000)
        bus = _make_bus(neo4j_driver=MagicMock(), config=cfg)
        bus._total_tokens_used = 2000
        p = bus.pressure()
        assert p["token_budget"] == 8000
        assert p["context_tokens_used"] == 2000
        assert p["utilization_pct"] == 0.25


# =============================================================================
# Explain
# =============================================================================

class TestExplain:
    """Test explain() provenance chain."""

    @pytest.mark.asyncio
    async def test_explain_requires_neo4j(self):
        bus = MemoryBus()
        with pytest.raises(RuntimeError, match="Neo4j not available"):
            await bus.explain("some-id")

    @pytest.mark.asyncio
    async def test_explain_returns_structure(self):
        """explain() returns dict with expected keys."""
        # Mock three session.run calls (node, derived_from, connected_entities)
        call_count = 0

        def mock_run(cypher, **params):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            if call_count == 1:
                # Node itself
                record = MagicMock()
                record.__getitem__ = lambda self, key: {
                    "labels": ["Entity"],
                    "props": {"id": "abc", "name": "Test"},
                }.get(key)
                mock_result.single.return_value = record
            else:
                mock_result.__iter__ = MagicMock(return_value=iter([]))
                mock_result.single.return_value = None

            mock_result.__iter__ = mock_result.__iter__ if hasattr(mock_result, '__iter__') else MagicMock(return_value=iter([]))
            return mock_result

        mock_session = MagicMock()
        mock_session.run = mock_run
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        mock_driver = MagicMock()
        mock_driver.session.return_value = mock_session
        bus = _make_bus(neo4j_driver=mock_driver)

        explanation = await bus.explain("abc")

        assert explanation["node_id"] == "abc"
        assert "derived_from" in explanation
        assert "connected_entities" in explanation
        assert "audit_entries" in explanation


# =============================================================================
# Entities Browsing
# =============================================================================

class TestEntities:
    """Test entities() browsing method."""

    @pytest.mark.asyncio
    async def test_entities_requires_neo4j(self):
        bus = MemoryBus()
        with pytest.raises(RuntimeError, match="Neo4j not available"):
            await bus.entities()

    @pytest.mark.asyncio
    async def test_entities_all(self):
        """entities() without filters returns all entities."""
        record = MagicMock()
        record.__getitem__ = lambda self, key: {"id": "1", "name": "Test", "type": "Hive"}
        driver, session = _mock_neo4j_session([record])
        bus = _make_bus(neo4j_driver=driver)

        result = await bus.entities()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_entities_by_type(self):
        """entities(entity_type=...) uses type filter."""
        driver, session = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        await bus.entities(entity_type="Hive")
        cypher = session.run.call_args[0][0]
        assert "type: $type" in cypher

    @pytest.mark.asyncio
    async def test_entities_by_query(self):
        """entities(query=...) uses CONTAINS filter."""
        driver, session = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        await bus.entities(query="aurora")
        cypher = session.run.call_args[0][0]
        assert "CONTAINS" in cypher

    @pytest.mark.asyncio
    async def test_entities_connected_to(self):
        """entities(connected_to=...) uses graph traversal."""
        driver, session = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        await bus.entities(connected_to="hive-1", max_hops=3)
        cypher = session.run.call_args[0][0]
        assert "*1..3" in cypher


# =============================================================================
# Audit Query
# =============================================================================

class TestAuditQuery:
    """Test _audit_query error handling."""

    @pytest.mark.asyncio
    async def test_audit_skipped_without_postgres(self):
        """_audit_query is a no-op when Postgres is unavailable."""
        bus = _make_bus(neo4j_driver=MagicMock())
        q = MemoryQuery(intent="test")
        r = MemoryResult(items=[], query=q, resolution_path="none", token_estimate=0)

        # Should not raise
        await bus._audit_query("loop-1", q, r)

    @pytest.mark.asyncio
    async def test_audit_logs_to_postgres(self):
        mock_pg = AsyncMock()
        bus = _make_bus(neo4j_driver=MagicMock(), postgres=mock_pg)
        q = MemoryQuery(intent="test")
        r = MemoryResult(items=[], query=q, resolution_path="exact", token_estimate=10)

        await bus._audit_query("loop-1", q, r)

        mock_pg.log_audit.assert_called_once()
        call_kwargs = mock_pg.log_audit.call_args[1]
        assert call_kwargs["action"] == "query"
        assert call_kwargs["loop_id"] == "loop-1"
        assert call_kwargs["resolution_path"] == "exact"
        assert call_kwargs["tokens_used"] == 10

    @pytest.mark.asyncio
    async def test_audit_error_is_swallowed(self):
        """Audit failures don't crash the bus."""
        mock_pg = AsyncMock()
        mock_pg.log_audit.side_effect = Exception("DB down")
        bus = _make_bus(neo4j_driver=MagicMock(), postgres=mock_pg)
        q = MemoryQuery(intent="test")
        r = MemoryResult(items=[], query=q, resolution_path="none", token_estimate=0)

        # Should not raise — audit is non-critical
        await bus._audit_query("loop-1", q, r)


# =============================================================================
# Entity Resolution
# =============================================================================

class TestEntityResolution:
    """Test resolve_entity() method."""

    @pytest.mark.asyncio
    async def test_resolve_via_postgres_alias(self):
        mock_pg = AsyncMock()
        mock_pg.resolve_alias.return_value = [
            {"entity_id": "hive-1", "alias": "Aurora", "confidence": 0.95},
        ]
        bus = _make_bus(postgres=mock_pg)

        candidates = await bus.resolve_entity("Aurora")
        assert len(candidates) == 1
        assert candidates[0]["entity_id"] == "hive-1"
        assert candidates[0]["source"] == "alias_table"

    @pytest.mark.asyncio
    async def test_resolve_falls_back_to_neo4j(self):
        """When Postgres returns nothing, falls back to Neo4j name search."""
        record = MagicMock()
        record.__getitem__ = lambda self, key: {
            "id": "hive-1", "name": "Aurora", "type": "Hive"
        }.get(key)
        driver, session = _mock_neo4j_session([record])
        bus = _make_bus(neo4j_driver=driver)

        candidates = await bus.resolve_entity("Aurora")
        assert len(candidates) == 1
        assert candidates[0]["source"] == "neo4j_name"
        assert candidates[0]["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_resolve_no_match(self):
        driver, _ = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        candidates = await bus.resolve_entity("nonexistent")
        assert candidates == []

    @pytest.mark.asyncio
    async def test_resolve_audits_when_postgres_available(self):
        mock_pg = AsyncMock()
        mock_pg.resolve_alias.return_value = []
        bus = _make_bus(postgres=mock_pg)

        await bus.resolve_entity("test")
        mock_pg.log_audit.assert_called_once()


# =============================================================================
# StoredItem Anti-Pattern Guard (AP-1)
# =============================================================================

class TestStoredItemAntiPattern:
    """Guard against AP-1: bus._items must use StoredItem, not raw MemoryItem.

    StoredItem has .to_dict(); MemoryItem does not.
    """

    def test_stored_item_has_to_dict(self):
        item = StoredItem(
            id="test-1",
            content="hello",
            memory_type="entity",
        )
        d = item.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "test-1"
        assert d["content"] == "hello"

    def test_memory_item_lacks_to_dict(self):
        item = MemoryItem(content="hello", memory_type="entity")
        assert not hasattr(item, "to_dict")

    def test_stored_item_wraps_all_fields(self):
        item = StoredItem(
            id="s1",
            content="content",
            memory_type="factual",
            entity_type="Hive",
            entity_ids=["e1"],
            properties={"domain": "bio"},
            importance=0.9,
            timestamp="2025-01-01T00:00:00",
        )
        d = item.to_dict()
        assert d["memory_type"] == "factual"
        assert d["importance"] == 0.9
        assert d["entity_type"] == "Hive"
        # to_dict() spreads properties — domain comes from properties
        assert d["domain"] == "bio"
        # entity_ids live in to_snapshot(), not to_dict()
        snap = item.to_snapshot()
        assert snap["entity_ids"] == ["e1"]

    def test_stored_item_is_expired_false_by_default(self):
        item = StoredItem(
            id="s1",
            content="x",
            memory_type="entity",
            timestamp="2025-01-01T00:00:00",
        )
        # No TTL → never expires
        assert item.is_expired() is False

    def test_stored_item_touch_increments_access(self):
        item = StoredItem(
            id="s1",
            content="x",
            memory_type="entity",
            timestamp="2025-01-01T00:00:00",
        )
        assert item.access_count == 0
        item.touch()
        assert item.access_count == 1
        assert item.last_accessed is not None


# =============================================================================
# Bus Config
# =============================================================================

class TestBusConfig:
    """Test MemoryBusConfig pressure tier calculation."""

    def test_pressure_tier_nominal(self):
        cfg = MemoryBusConfig()
        assert cfg.pressure_tier_for(0.0) == PressureTier.NOMINAL
        assert cfg.pressure_tier_for(0.49) == PressureTier.NOMINAL

    def test_pressure_tier_elevated(self):
        cfg = MemoryBusConfig()
        assert cfg.pressure_tier_for(0.50) == PressureTier.ELEVATED
        assert cfg.pressure_tier_for(0.74) == PressureTier.ELEVATED

    def test_pressure_tier_high(self):
        cfg = MemoryBusConfig()
        assert cfg.pressure_tier_for(0.75) == PressureTier.HIGH
        assert cfg.pressure_tier_for(0.89) == PressureTier.HIGH

    def test_pressure_tier_critical(self):
        cfg = MemoryBusConfig()
        assert cfg.pressure_tier_for(0.90) == PressureTier.CRITICAL
        assert cfg.pressure_tier_for(1.0) == PressureTier.CRITICAL


# =============================================================================
# Structured Query Filters
# =============================================================================

class TestStructuredQueryFilters:
    """Test _query_structured builds correct Cypher from query filters."""

    def test_structured_with_time_bounds(self):
        """Both time_after and time_before produce WHERE clauses."""
        driver, session = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        bus._query_structured(MemoryQuery(
            intent="test",
            entity_type="Hive",
            time_after="2025-01-01",
            time_before="2025-12-31",
        ))

        cypher = session.run.call_args[0][0]
        assert "n.timestamp >= $time_after" in cypher
        assert "n.timestamp <= $time_before" in cypher
        assert "n.type = $entity_type" in cypher

    def test_structured_label_mapping(self):
        """memory_type selects the correct node label."""
        driver, session = _mock_neo4j_session([])
        bus = _make_bus(neo4j_driver=driver)

        # episodic → Episode
        bus._query_structured(MemoryQuery(intent="x", memory_type="episodic"))
        assert "Episode" in session.run.call_args[0][0]

        # factual → Fact
        bus._query_structured(MemoryQuery(intent="x", memory_type="factual"))
        assert "Fact" in session.run.call_args[0][0]

        # plan → Plan
        bus._query_structured(MemoryQuery(intent="x", memory_type="plan"))
        assert "Plan" in session.run.call_args[0][0]

        # default → Entity
        bus._query_structured(MemoryQuery(intent="x", memory_type="other"))
        assert "Entity" in session.run.call_args[0][0]
