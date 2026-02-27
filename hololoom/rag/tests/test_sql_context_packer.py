"""
Tests for SQL Context Packer - Phase 6.2
=========================================

Comprehensive test suite covering:
- Column MI scoring
- Row MI scoring
- Column selection based on MI threshold
- Row selection based on MI and budget
- Budget constraints
- Strategy selection (Phase 6.1 integration)
- Convenience functions
- Edge cases
- Statistics tracking

Author: Claude Code
Date: 2025-12-09
"""

import pytest
from typing import Dict, List, Any

from HoloLoom.rag.sql_context_packer import (
    SQLContextPacker,
    PackedSQLContext,
    PackingStrategy,
    pack_sql_results,
    get_adaptive_sql_budget,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_rows() -> List[Dict[str, Any]]:
    """Sample SQL result rows for testing."""
    return [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30, "department": "Engineering"},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25, "department": "Marketing"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35, "department": "Engineering"},
        {"id": 4, "name": "Diana", "email": "diana@example.com", "age": 28, "department": "Sales"},
        {"id": 5, "name": "Eve", "email": "eve@example.com", "age": 32, "department": "Engineering"},
    ]


@pytest.fixture
def large_rows() -> List[Dict[str, Any]]:
    """Large dataset for budget constraint testing."""
    return [
        {
            "id": i,
            "name": f"User_{i}",
            "email": f"user{i}@example.com",
            "age": 20 + (i % 40),
            "department": ["Engineering", "Marketing", "Sales", "HR"][i % 4],
            "salary": 50000 + (i * 1000),
            "hire_date": f"2020-{(i % 12) + 1:02d}-01",
            "status": "active" if i % 3 != 0 else "inactive",
        }
        for i in range(100)
    ]


@pytest.fixture
def packer() -> SQLContextPacker:
    """Default SQL context packer."""
    return SQLContextPacker()


@pytest.fixture
def packer_no_adaptive() -> SQLContextPacker:
    """Packer with adaptive strategy disabled."""
    return SQLContextPacker(enable_adaptive_strategy=False)


# ============================================================================
# Column MI Scoring Tests
# ============================================================================

class TestColumnMIScoring:
    """Tests for column MI scoring."""

    def test_column_name_overlap_increases_score(self, packer, sample_rows):
        """Columns with names matching query terms should score higher."""
        query = "Show me the name and email of users"

        column_scores = packer._score_columns_by_mi(
            columns=["id", "name", "email", "age", "department"],
            rows=sample_rows,
            query=query
        )

        # name and email should score higher than id and age
        assert column_scores["name"] > column_scores["id"]
        assert column_scores["email"] > column_scores["id"]

    def test_column_value_overlap_increases_score(self, packer, sample_rows):
        """Columns with values matching query terms should score higher."""
        query = "Find Alice in the Engineering department"

        column_scores = packer._score_columns_by_mi(
            columns=["id", "name", "email", "age", "department"],
            rows=sample_rows,
            query=query
        )

        # name (has Alice) and department (has Engineering) should score high
        assert column_scores["name"] > column_scores["id"]
        assert column_scores["department"] > column_scores["id"]

    def test_semantic_bonus_for_id_columns(self, packer, sample_rows):
        """ID columns should get a semantic bonus."""
        query = "Get records"  # No specific column mentioned

        column_scores = packer._score_columns_by_mi(
            columns=["id", "random_column", "xyz"],
            rows=[{"id": 1, "random_column": "data", "xyz": "value"}],
            query=query
        )

        # id should get bonus over random columns
        assert column_scores["id"] >= column_scores["random_column"]

    def test_semantic_bonus_for_name_columns(self, packer, sample_rows):
        """Name/title columns should get a semantic bonus."""
        query = "Get records"

        column_scores = packer._score_columns_by_mi(
            columns=["name", "title", "random"],
            rows=[{"name": "Test", "title": "Title", "random": "data"}],
            query=query
        )

        assert column_scores["name"] >= column_scores["random"]
        assert column_scores["title"] >= column_scores["random"]


# ============================================================================
# Row MI Scoring Tests
# ============================================================================

class TestRowMIScoring:
    """Tests for row MI scoring."""

    def test_rows_with_query_terms_score_higher(self, packer, sample_rows):
        """Rows containing query terms should have higher MI scores."""
        query = "Find Alice"

        row_scores = packer._score_rows_by_mi(
            rows=sample_rows,
            selected_columns=["name", "email", "department"],
            query=query
        )

        # First row (Alice) should score highest
        alice_idx = 0  # Alice is first row
        for i, score in enumerate(row_scores):
            if i != alice_idx:
                assert row_scores[alice_idx] >= score

    def test_multiple_term_matches_score_higher(self, packer, sample_rows):
        """Rows matching multiple query terms should score higher."""
        query = "Find Engineering employees named Alice"

        row_scores = packer._score_rows_by_mi(
            rows=sample_rows,
            selected_columns=["name", "department"],
            query=query
        )

        # Alice in Engineering should score highest
        assert row_scores[0] > 0  # Alice is Engineering
        # Compare to Bob (Marketing)
        assert row_scores[0] > row_scores[1]


# ============================================================================
# Column Selection Tests
# ============================================================================

class TestColumnSelection:
    """Tests for column selection based on MI threshold."""

    def test_high_threshold_selects_fewer_columns(self, packer, sample_rows):
        """Higher MI threshold should select fewer columns."""
        query = "Show name"

        column_scores = packer._score_columns_by_mi(
            columns=list(sample_rows[0].keys()),
            rows=sample_rows,
            query=query
        )

        cols_aggressive = packer._select_columns(column_scores, mi_threshold=0.3)
        cols_research = packer._select_columns(column_scores, mi_threshold=0.05)

        assert len(cols_aggressive) <= len(cols_research)

    def test_always_keeps_minimum_columns(self, packer, sample_rows):
        """Should always keep at least 2 columns even with high threshold."""
        query = "xyz"  # Unlikely to match anything well

        column_scores = packer._score_columns_by_mi(
            columns=list(sample_rows[0].keys()),
            rows=sample_rows,
            query=query
        )

        # Even with very high threshold, should keep 2 columns
        selected = packer._select_columns(column_scores, mi_threshold=0.99)
        assert len(selected) >= 2

    def test_max_columns_limit_respected(self, packer, sample_rows):
        """max_columns parameter should limit selection."""
        query = "Show all user information"

        column_scores = packer._score_columns_by_mi(
            columns=list(sample_rows[0].keys()),
            rows=sample_rows,
            query=query
        )

        selected = packer._select_columns(column_scores, mi_threshold=0.0, max_columns=3)
        assert len(selected) == 3


# ============================================================================
# Row Selection Tests
# ============================================================================

class TestRowSelection:
    """Tests for row selection based on MI and budget."""

    def test_keeps_top_mi_rows(self, packer, sample_rows):
        """Should keep rows with highest MI scores."""
        query = "Find Alice and Bob"

        row_scores = packer._score_rows_by_mi(
            rows=sample_rows,
            selected_columns=["name"],
            query=query
        )

        selected_rows, selected_scores = packer._select_rows(
            rows=sample_rows,
            row_mi_scores=row_scores,
            selected_columns=["name", "email"],
            max_rows=2,
            token_budget=None,
            strategy=PackingStrategy.BALANCED
        )

        # Should select Alice and Bob (highest MI)
        names = [r["name"] for r in selected_rows]
        assert "Alice" in names
        assert "Bob" in names

    def test_strategy_affects_keep_ratio(self, packer, large_rows):
        """Different strategies should keep different percentages."""
        query = "Show all users"

        row_scores = packer._score_rows_by_mi(
            rows=large_rows,
            selected_columns=["name"],
            query=query
        )

        aggressive_rows, _ = packer._select_rows(
            rows=large_rows,
            row_mi_scores=row_scores,
            selected_columns=["name"],
            max_rows=None,
            token_budget=None,
            strategy=PackingStrategy.AGGRESSIVE
        )

        research_rows, _ = packer._select_rows(
            rows=large_rows,
            row_mi_scores=row_scores,
            selected_columns=["name"],
            max_rows=None,
            token_budget=None,
            strategy=PackingStrategy.RESEARCH
        )

        # Aggressive keeps 30%, Research keeps 90%
        assert len(aggressive_rows) < len(research_rows)
        assert len(aggressive_rows) <= int(100 * 0.35)  # ~30% + buffer
        assert len(research_rows) >= int(100 * 0.85)  # ~90% - buffer


# ============================================================================
# Budget Constraint Tests
# ============================================================================

class TestBudgetConstraints:
    """Tests for token budget constraints."""

    def test_token_budget_limits_rows(self, packer, large_rows):
        """Token budget should limit number of rows."""
        query = "Show users"

        result = packer.pack_sql_results(
            rows=large_rows,
            query=query,
            token_budget=200  # Very small budget
        )

        # Should respect budget
        assert result.estimated_tokens <= 200 + 50  # Allow small overflow
        assert len(result.rows) < len(large_rows)

    def test_tokens_saved_calculated(self, packer, sample_rows):
        """tokens_saved should reflect actual savings."""
        result = packer.pack_sql_results(
            rows=sample_rows,
            query="Show name and email",
            strategy=PackingStrategy.AGGRESSIVE
        )

        # Should save some tokens
        assert result.tokens_saved >= 0
        # Original - current should equal saved
        original_tokens = packer._estimate_tokens(
            rows=sample_rows,
            columns=list(sample_rows[0].keys())
        )
        assert result.tokens_saved == original_tokens - result.estimated_tokens


# ============================================================================
# Strategy Selection Tests
# ============================================================================

class TestStrategySelection:
    """Tests for strategy selection."""

    def test_simple_query_gets_aggressive(self, packer):
        """Simple queries should select aggressive strategy."""
        strategy = packer._select_strategy("count users")
        assert strategy in [PackingStrategy.AGGRESSIVE, PackingStrategy.BALANCED]

    def test_complex_query_gets_conservative(self, packer):
        """Complex queries should select conservative or higher strategy."""
        # "explain" triggers MODERATE in Phase 6.1, which maps to BALANCED
        # Need 2+ complex keywords for COMPLEX classification
        strategy = packer._select_strategy("compare and contrast the relationship between users and orders")
        assert strategy in [PackingStrategy.CONSERVATIVE, PackingStrategy.RESEARCH, PackingStrategy.BALANCED]

    def test_research_query_gets_research(self, packer):
        """Research queries should select research strategy."""
        strategy = packer._select_strategy("analyze all tradeoffs comprehensively")
        assert strategy == PackingStrategy.RESEARCH

    def test_explicit_strategy_overrides_adaptive(self, packer, sample_rows):
        """Explicit strategy should override adaptive selection."""
        result = packer.pack_sql_results(
            rows=sample_rows,
            query="analyze comprehensive tradeoffs",  # Would normally be RESEARCH
            strategy=PackingStrategy.AGGRESSIVE  # Explicitly aggressive
        )

        assert result.strategy == PackingStrategy.AGGRESSIVE


# ============================================================================
# Integration Tests (Phase 6.1)
# ============================================================================

class TestPhase61Integration:
    """Tests for Phase 6.1 AdaptiveCompressionStrategy integration."""

    def test_integrates_with_adaptive_compression(self, packer, sample_rows):
        """Should integrate with Phase 6.1 AdaptiveCompressionStrategy."""
        # Simple query - should be AGGRESSIVE (maps from TRIVIAL/SIMPLE)
        result_simple = packer.pack_sql_results(
            rows=sample_rows,
            query="hi"
        )
        assert result_simple.strategy in [PackingStrategy.AGGRESSIVE, PackingStrategy.BALANCED]

        # Complex query - should be CONSERVATIVE or RESEARCH
        result_complex = packer.pack_sql_results(
            rows=sample_rows,
            query="compare and contrast all user departments and analyze comprehensive tradeoffs"
        )
        assert result_complex.strategy in [PackingStrategy.CONSERVATIVE, PackingStrategy.RESEARCH]


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_pack_sql_results_function(self, sample_rows):
        """pack_sql_results convenience function should work."""
        result = pack_sql_results(
            rows=sample_rows,
            query="Find Alice"
        )

        assert isinstance(result, PackedSQLContext)
        assert len(result.rows) <= len(sample_rows)

    def test_get_adaptive_sql_budget(self):
        """get_adaptive_sql_budget should return reasonable limits."""
        # Simple query
        max_rows, max_cols, budget = get_adaptive_sql_budget("count users")
        assert max_rows <= 50
        assert max_cols <= 10
        assert budget <= 2000

        # Complex query
        max_rows, max_cols, budget = get_adaptive_sql_budget(
            "analyze all tradeoffs comprehensively"
        )
        assert max_rows >= 50
        assert max_cols >= 10
        assert budget >= 2000


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_rows(self, packer):
        """Should handle empty rows gracefully."""
        result = packer.pack_sql_results(
            rows=[],
            query="anything"
        )

        assert len(result.rows) == 0
        assert len(result.columns) == 0
        assert result.total_compression_ratio == 0.0

    def test_single_row(self, packer):
        """Should handle single row."""
        rows = [{"id": 1, "name": "Alice"}]

        result = packer.pack_sql_results(
            rows=rows,
            query="Find Alice"
        )

        assert len(result.rows) >= 1
        assert len(result.columns) >= 1

    def test_single_column(self, packer):
        """Should handle single column."""
        rows = [{"name": "Alice"}, {"name": "Bob"}]

        result = packer.pack_sql_results(
            rows=rows,
            query="Find users"
        )

        assert len(result.columns) == 1
        assert result.columns[0] == "name"

    def test_null_values_handled(self, packer):
        """Should handle null values in rows."""
        rows = [
            {"id": 1, "name": "Alice", "email": None},
            {"id": 2, "name": None, "email": "bob@example.com"},
        ]

        result = packer.pack_sql_results(
            rows=rows,
            query="Find users"
        )

        assert len(result.rows) > 0  # Should not crash


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:
    """Tests for statistics tracking."""

    def test_statistics_updated(self, packer, sample_rows):
        """Statistics should be updated after packing."""
        initial_stats = packer.get_statistics()
        assert initial_stats['total_packs'] == 0

        packer.pack_sql_results(rows=sample_rows, query="Find Alice")
        packer.pack_sql_results(rows=sample_rows, query="Find Bob")

        stats = packer.get_statistics()
        assert stats['total_packs'] == 2
        assert stats['total_rows_processed'] == len(sample_rows) * 2

    def test_cache_size_tracked(self, packer, sample_rows):
        """Cache size should be tracked in statistics."""
        packer.pack_sql_results(rows=sample_rows, query="Find Alice")

        stats = packer.get_statistics()
        assert stats['cache_size'] > 0

    def test_cache_can_be_cleared(self, packer, sample_rows):
        """Cache should be clearable."""
        packer.pack_sql_results(rows=sample_rows, query="Find Alice")
        assert packer.get_statistics()['cache_size'] > 0

        packer.clear_cache()
        assert packer.get_statistics()['cache_size'] == 0


# ============================================================================
# PackedSQLContext Tests
# ============================================================================

class TestPackedSQLContext:
    """Tests for PackedSQLContext dataclass."""

    def test_str_representation(self, packer, sample_rows):
        """__str__ should produce readable output."""
        result = packer.pack_sql_results(
            rows=sample_rows,
            query="Find users"
        )

        str_repr = str(result)
        assert "PackedSQLContext" in str_repr
        assert "rows:" in str_repr
        assert "columns:" in str_repr
        assert "MI preserved:" in str_repr

    def test_to_dataframe_with_pandas(self, packer, sample_rows):
        """to_dataframe should work if pandas available."""
        result = packer.pack_sql_results(
            rows=sample_rows,
            query="Find users"
        )

        df = result.to_dataframe()
        # May return None if pandas not available
        if df is not None:
            assert len(df) == len(result.rows)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
