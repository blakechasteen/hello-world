"""
Tests for Neo4j MCTS Store — Phase 6.

Tests the MCTSStore with graceful degradation (no Neo4j required).
When driver=None, all methods return safe defaults.
"""

import pytest
from hololoom.core.bus.neo4j_store import MCTSStore, MCTS_SCHEMA_STATEMENTS


class TestMCTSStoreDegraded:
    """Tests with Neo4j unavailable (driver=None). Must not crash."""

    @pytest.fixture
    def store(self):
        return MCTSStore(driver=None)

    def test_not_available(self, store):
        assert store.available is False

    def test_save_tree_returns_false(self, store):
        result = store.save_tree("t1", "code", "strategic", [])
        assert result is False

    def test_load_tree_returns_none(self, store):
        result = store.load_tree("code")
        assert result is None

    def test_backprop_returns_false(self, store):
        result = store.backprop_update("node-1", 0.8)
        assert result is False

    def test_record_strategy_returns_false(self, store):
        result = store.record_strategy_outcome("verified_query", "code", True, 0.9)
        assert result is False

    def test_query_best_strategy_returns_empty(self, store):
        result = store.query_best_strategy("code")
        assert result == []

    def test_get_all_pattern_domains_returns_empty(self, store):
        result = store.get_all_pattern_domains()
        assert result == []

    def test_prune_returns_zero(self, store):
        result = store.prune_low_visit_nodes("t1")
        assert result == 0


class TestMCTSSchemaStatements:
    """Verify schema statements are well-formed."""

    def test_schema_statements_are_strings(self):
        for stmt in MCTS_SCHEMA_STATEMENTS:
            assert isinstance(stmt, str)
            assert "IF NOT EXISTS" in stmt

    def test_schema_covers_all_node_types(self):
        all_stmts = " ".join(MCTS_SCHEMA_STATEMENTS)
        assert "MCTSNode" in all_stmts
        assert "MCTSTree" in all_stmts
        assert "PatternDomain" in all_stmts
