"""
SQL Context Packer - MI-Aware Row/Column Optimization
=====================================================

Phase 6.2: Wires context packing into SQL RAG pipeline for row/column optimization.

Capabilities:
- Pre-pack retrieved SQL rows before generation
- MI-aware column selection (drop low-MI columns)
- Budget-aware result limiting (top-K by MI)
- Integration with Phase 6.1 adaptive strategy

MI Scoring:
- Column MI: I(Column; Query) based on column name/values overlap with query
- Row MI: I(Row; Query) based on row content overlap with query
- Budget-aware selection: Stop when token budget exhausted

Author: Claude Code
Date: 2025-12-09
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


class PackingStrategy(Enum):
    """SQL packing strategy selection."""
    AGGRESSIVE = "aggressive"    # Keep 30% rows/columns (fast queries)
    BALANCED = "balanced"        # Keep 50% rows/columns (standard)
    CONSERVATIVE = "conservative"  # Keep 70% rows/columns (complex)
    RESEARCH = "research"        # Keep 90% rows/columns (full context)


@dataclass
class PackedSQLContext:
    """
    Result of SQL context packing.

    Contains the optimized rows and columns with MI scores.
    """
    # Packed data
    rows: List[Dict[str, Any]]
    columns: List[str]

    # Original data for comparison
    original_row_count: int
    original_column_count: int

    # Compression metrics
    row_compression_ratio: float  # packed/original
    column_compression_ratio: float
    total_compression_ratio: float

    # MI scores
    column_mi_scores: Dict[str, float]  # column -> MI score
    row_mi_scores: List[float]  # MI score per packed row
    total_mi: float  # Total MI preserved

    # Token estimates
    estimated_tokens: int
    token_budget: int
    tokens_saved: int

    # Strategy used
    strategy: PackingStrategy

    def __str__(self) -> str:
        """Human-readable representation."""
        lines = [
            "PackedSQLContext(",
            f"  rows: {len(self.rows)}/{self.original_row_count} ({self.row_compression_ratio:.1%})",
            f"  columns: {len(self.columns)}/{self.original_column_count} ({self.column_compression_ratio:.1%})",
            f"  total compression: {self.total_compression_ratio:.1%}",
            f"  MI preserved: {self.total_mi:.2f} bits",
            f"  tokens: {self.estimated_tokens}/{self.token_budget} ({self.tokens_saved} saved)",
            f"  strategy: {self.strategy.value}",
            ")"
        ]
        return "\n".join(lines)

    def to_dataframe(self):
        """Convert to pandas DataFrame if available."""
        try:
            import pandas as pd
            return pd.DataFrame(self.rows, columns=self.columns)
        except ImportError:
            return None


class SQLContextPacker:
    """
    MI-aware SQL context packer.

    Optimizes SQL query results by selecting the most informative
    rows and columns based on mutual information with the query.
    """

    # Default token estimates per cell
    DEFAULT_TOKENS_PER_CELL = 5
    DEFAULT_TOKENS_PER_COLUMN_HEADER = 3

    # MI thresholds for column/row selection
    MI_THRESHOLD_AGGRESSIVE = 0.3
    MI_THRESHOLD_BALANCED = 0.2
    MI_THRESHOLD_CONSERVATIVE = 0.1
    MI_THRESHOLD_RESEARCH = 0.05

    def __init__(
        self,
        default_strategy: PackingStrategy = PackingStrategy.BALANCED,
        tokens_per_cell: int = None,
        enable_adaptive_strategy: bool = True
    ):
        """
        Initialize SQL Context Packer.

        Args:
            default_strategy: Default packing strategy if not adaptive
            tokens_per_cell: Estimated tokens per table cell
            enable_adaptive_strategy: Auto-select strategy based on query complexity
        """
        self.default_strategy = default_strategy
        self.tokens_per_cell = tokens_per_cell or self.DEFAULT_TOKENS_PER_CELL
        self.enable_adaptive_strategy = enable_adaptive_strategy

        # Caches
        self._column_mi_cache: Dict[Tuple[str, str], float] = {}

        # Statistics
        self._stats = {
            'total_packs': 0,
            'total_rows_processed': 0,
            'total_columns_processed': 0,
            'total_rows_kept': 0,
            'total_columns_kept': 0,
            'total_tokens_saved': 0,
            'avg_row_compression': 0.0,
            'avg_column_compression': 0.0,
        }

    def pack_sql_results(
        self,
        rows: List[Dict[str, Any]],
        query: str,
        max_rows: Optional[int] = None,
        max_columns: Optional[int] = None,
        token_budget: Optional[int] = None,
        strategy: Optional[PackingStrategy] = None
    ) -> PackedSQLContext:
        """
        Pack SQL results with MI-aware row/column selection.

        Args:
            rows: List of row dictionaries from SQL query
            query: Natural language query text
            max_rows: Maximum rows to keep (None = budget-based)
            max_columns: Maximum columns to keep (None = budget-based)
            token_budget: Maximum tokens for result (None = no limit)
            strategy: Packing strategy (None = auto-select)

        Returns:
            PackedSQLContext with optimized rows and columns
        """
        if not rows:
            return self._empty_result(token_budget)

        # Get all columns from first row
        all_columns = list(rows[0].keys())
        original_row_count = len(rows)
        original_column_count = len(all_columns)

        logger.debug(
            f"Packing SQL: {original_row_count} rows × {original_column_count} columns, "
            f"query: {query[:50]}..."
        )

        # Step 1: Select strategy
        if strategy is None:
            if self.enable_adaptive_strategy:
                strategy = self._select_strategy(query)
            else:
                strategy = self.default_strategy

        # Step 2: Score columns by MI
        column_mi_scores = self._score_columns_by_mi(
            columns=all_columns,
            rows=rows,
            query=query
        )

        # Step 3: Select columns based on MI threshold
        mi_threshold = self._get_mi_threshold(strategy)
        selected_columns = self._select_columns(
            column_mi_scores=column_mi_scores,
            mi_threshold=mi_threshold,
            max_columns=max_columns
        )

        # Step 4: Score rows by MI (using selected columns only)
        row_mi_scores = self._score_rows_by_mi(
            rows=rows,
            selected_columns=selected_columns,
            query=query
        )

        # Step 5: Select rows based on MI and budget
        selected_rows, selected_row_scores = self._select_rows(
            rows=rows,
            row_mi_scores=row_mi_scores,
            selected_columns=selected_columns,
            max_rows=max_rows,
            token_budget=token_budget,
            strategy=strategy
        )

        # Step 6: Calculate metrics
        row_compression = len(selected_rows) / original_row_count if original_row_count > 0 else 0.0
        column_compression = len(selected_columns) / original_column_count if original_column_count > 0 else 0.0
        total_compression = row_compression * column_compression

        # Estimate tokens
        estimated_tokens = self._estimate_tokens(
            rows=selected_rows,
            columns=selected_columns
        )

        original_tokens = self._estimate_tokens(rows=rows, columns=all_columns)
        tokens_saved = original_tokens - estimated_tokens

        # Calculate total MI preserved
        total_mi = sum(column_mi_scores.get(col, 0.0) for col in selected_columns)
        total_mi += sum(selected_row_scores)

        # Update stats
        self._update_stats(
            original_rows=original_row_count,
            original_cols=original_column_count,
            kept_rows=len(selected_rows),
            kept_cols=len(selected_columns),
            tokens_saved=tokens_saved
        )

        # Build result
        result = PackedSQLContext(
            rows=selected_rows,
            columns=selected_columns,
            original_row_count=original_row_count,
            original_column_count=original_column_count,
            row_compression_ratio=row_compression,
            column_compression_ratio=column_compression,
            total_compression_ratio=total_compression,
            column_mi_scores={col: column_mi_scores.get(col, 0.0) for col in selected_columns},
            row_mi_scores=selected_row_scores,
            total_mi=total_mi,
            estimated_tokens=estimated_tokens,
            token_budget=token_budget or estimated_tokens,
            tokens_saved=tokens_saved,
            strategy=strategy
        )

        logger.info(
            f"Packed SQL: {len(selected_rows)}/{original_row_count} rows, "
            f"{len(selected_columns)}/{original_column_count} cols, "
            f"MI={total_mi:.2f} bits, {tokens_saved} tokens saved"
        )

        return result

    def _select_strategy(self, query: str) -> PackingStrategy:
        """
        Auto-select packing strategy based on query complexity.

        Integrates with Phase 6.1 AdaptiveCompressionStrategy if available.
        """
        try:
            from hololoom.context_packing import (
                AdaptiveCompressionStrategy,
                AdaptiveComplexity
            )

            strategy_obj = AdaptiveCompressionStrategy()
            result = strategy_obj.select_strategy(query)

            # Map complexity to packing strategy
            complexity_map = {
                AdaptiveComplexity.TRIVIAL: PackingStrategy.AGGRESSIVE,
                AdaptiveComplexity.SIMPLE: PackingStrategy.AGGRESSIVE,
                AdaptiveComplexity.MODERATE: PackingStrategy.BALANCED,
                AdaptiveComplexity.COMPLEX: PackingStrategy.CONSERVATIVE,
                AdaptiveComplexity.RESEARCH: PackingStrategy.RESEARCH,
            }

            return complexity_map.get(result.complexity, PackingStrategy.BALANCED)

        except ImportError:
            # Fallback: Simple heuristics
            query_lower = query.lower()

            # Research keywords
            if any(kw in query_lower for kw in ['analyze', 'compare', 'comprehensive', 'tradeoffs']):
                return PackingStrategy.RESEARCH

            # Complex keywords
            if any(kw in query_lower for kw in ['explain', 'describe', 'how does']):
                return PackingStrategy.CONSERVATIVE

            # Simple keywords
            if any(kw in query_lower for kw in ['count', 'how many', 'list', 'show']):
                return PackingStrategy.AGGRESSIVE

            return PackingStrategy.BALANCED

    def _get_mi_threshold(self, strategy: PackingStrategy) -> float:
        """Get MI threshold for strategy."""
        thresholds = {
            PackingStrategy.AGGRESSIVE: self.MI_THRESHOLD_AGGRESSIVE,
            PackingStrategy.BALANCED: self.MI_THRESHOLD_BALANCED,
            PackingStrategy.CONSERVATIVE: self.MI_THRESHOLD_CONSERVATIVE,
            PackingStrategy.RESEARCH: self.MI_THRESHOLD_RESEARCH,
        }
        return thresholds.get(strategy, self.MI_THRESHOLD_BALANCED)

    def _score_columns_by_mi(
        self,
        columns: List[str],
        rows: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, float]:
        """
        Score columns by mutual information with query.

        MI(Column; Query) = overlap of column name/values with query terms.

        Args:
            columns: List of column names
            rows: Row data for value-based scoring
            query: Query text

        Returns:
            Dict mapping column name -> MI score (0.0-1.0)
        """
        query_terms = self._extract_terms(query)
        column_scores = {}

        for column in columns:
            # Check cache first
            cache_key = (column, query)
            if cache_key in self._column_mi_cache:
                column_scores[column] = self._column_mi_cache[cache_key]
                continue

            score = 0.0

            # 1. Column name MI (0.0-0.5)
            column_terms = self._extract_terms(column)
            name_overlap = len(query_terms & column_terms)
            name_mi = min(0.5, name_overlap * 0.25)  # Max 0.5 from name
            score += name_mi

            # 2. Value MI (0.0-0.5)
            # Sample values from column
            sample_values = []
            for row in rows[:min(20, len(rows))]:  # Sample first 20 rows
                val = row.get(column, '')
                if val is not None:
                    sample_values.append(str(val))

            # Count term overlaps in values
            value_text = ' '.join(sample_values).lower()
            value_overlaps = sum(1 for term in query_terms if term in value_text)
            value_mi = min(0.5, value_overlaps * 0.1)  # Max 0.5 from values
            score += value_mi

            # 3. Semantic relevance bonus
            # ID columns, foreign keys, and timestamps often relevant
            semantic_bonus = 0.0
            col_lower = column.lower()
            if 'id' in col_lower:
                semantic_bonus = 0.1
            elif any(kw in col_lower for kw in ['name', 'title', 'description']):
                semantic_bonus = 0.15
            elif any(kw in col_lower for kw in ['date', 'time', 'created', 'updated']):
                semantic_bonus = 0.05

            score += semantic_bonus

            # Normalize to 0-1
            score = min(1.0, score)

            column_scores[column] = score
            self._column_mi_cache[cache_key] = score

        return column_scores

    def _select_columns(
        self,
        column_mi_scores: Dict[str, float],
        mi_threshold: float,
        max_columns: Optional[int] = None
    ) -> List[str]:
        """
        Select columns based on MI scores.

        Args:
            column_mi_scores: Column -> MI score mapping
            mi_threshold: Minimum MI to keep column
            max_columns: Maximum columns (None = no limit)

        Returns:
            List of selected column names (sorted by MI descending)
        """
        # Sort by MI score descending
        sorted_columns = sorted(
            column_mi_scores.keys(),
            key=lambda c: column_mi_scores[c],
            reverse=True
        )

        # Filter by threshold
        selected = [c for c in sorted_columns if column_mi_scores[c] >= mi_threshold]

        # Always keep at least some columns for context
        # If only 1 column exists, keep it; otherwise try to keep at least 2
        if len(sorted_columns) == 1:
            # Only 1 column exists - must keep it
            if len(selected) == 0:
                selected = sorted_columns[:1]
        elif len(sorted_columns) >= 2:
            # Multiple columns exist - keep at least 2 for context
            if len(selected) < 2:
                selected = sorted_columns[:2]

        # Apply max limit
        if max_columns is not None and len(selected) > max_columns:
            selected = selected[:max_columns]

        return selected

    def _score_rows_by_mi(
        self,
        rows: List[Dict[str, Any]],
        selected_columns: List[str],
        query: str
    ) -> List[float]:
        """
        Score rows by mutual information with query.

        MI(Row; Query) = overlap of row content with query terms.

        Args:
            rows: List of row dictionaries
            selected_columns: Columns to consider for scoring
            query: Query text

        Returns:
            List of MI scores (one per row)
        """
        query_terms = self._extract_terms(query)
        row_scores = []

        for row in rows:
            # Build row text from selected columns
            row_text_parts = []
            for col in selected_columns:
                val = row.get(col, '')
                if val is not None:
                    row_text_parts.append(str(val))

            row_text = ' '.join(row_text_parts).lower()

            # Count term overlaps
            overlaps = sum(1 for term in query_terms if term in row_text)

            # Normalize by query term count
            if len(query_terms) > 0:
                score = overlaps / len(query_terms)
            else:
                score = 0.0

            row_scores.append(min(1.0, score))

        return row_scores

    def _select_rows(
        self,
        rows: List[Dict[str, Any]],
        row_mi_scores: List[float],
        selected_columns: List[str],
        max_rows: Optional[int],
        token_budget: Optional[int],
        strategy: PackingStrategy
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Select rows based on MI scores and budget.

        Args:
            rows: Original rows
            row_mi_scores: MI scores per row
            selected_columns: Columns to keep in each row
            max_rows: Maximum rows to keep
            token_budget: Token budget constraint
            strategy: Packing strategy

        Returns:
            Tuple of (selected_rows, selected_scores)
        """
        # Sort indices by MI score descending
        sorted_indices = sorted(
            range(len(rows)),
            key=lambda i: row_mi_scores[i],
            reverse=True
        )

        # Get keep ratio from strategy
        keep_ratios = {
            PackingStrategy.AGGRESSIVE: 0.3,
            PackingStrategy.BALANCED: 0.5,
            PackingStrategy.CONSERVATIVE: 0.7,
            PackingStrategy.RESEARCH: 0.9,
        }
        keep_ratio = keep_ratios.get(strategy, 0.5)

        # Calculate limits
        if max_rows is not None:
            row_limit = min(max_rows, int(len(rows) * keep_ratio))
        else:
            row_limit = max(1, int(len(rows) * keep_ratio))

        # Token budget constraint
        if token_budget is not None:
            tokens_per_row = len(selected_columns) * self.tokens_per_cell
            tokens_per_row += len(selected_columns) * self.DEFAULT_TOKENS_PER_COLUMN_HEADER
            max_rows_by_budget = max(1, token_budget // tokens_per_row)
            row_limit = min(row_limit, max_rows_by_budget)

        # Select top rows
        selected_indices = sorted_indices[:row_limit]

        # Build result (filter columns)
        selected_rows = []
        selected_scores = []

        for idx in selected_indices:
            row = rows[idx]
            filtered_row = {col: row.get(col) for col in selected_columns}
            selected_rows.append(filtered_row)
            selected_scores.append(row_mi_scores[idx])

        return selected_rows, selected_scores

    def _estimate_tokens(
        self,
        rows: List[Dict[str, Any]],
        columns: List[str]
    ) -> int:
        """
        Estimate token count for rows and columns.

        Args:
            rows: Row data
            columns: Column names

        Returns:
            Estimated token count
        """
        # Column headers
        header_tokens = len(columns) * self.DEFAULT_TOKENS_PER_COLUMN_HEADER

        # Cell tokens
        cell_tokens = len(rows) * len(columns) * self.tokens_per_cell

        return header_tokens + cell_tokens

    def _extract_terms(self, text: str) -> set:
        """Extract normalized terms from text."""
        # Lowercase and split on non-alphanumeric
        terms = re.split(r'[^a-zA-Z0-9]+', text.lower())

        # Filter stopwords and short terms
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'and', 'or', 'not', 'but', 'if', 'then', 'so', 'as', 'that',
                     'this', 'it', 'its', 'what', 'which', 'who', 'how', 'when'}

        return {t for t in terms if len(t) > 2 and t not in stopwords}

    def _empty_result(self, token_budget: Optional[int]) -> PackedSQLContext:
        """Return empty result for empty input."""
        return PackedSQLContext(
            rows=[],
            columns=[],
            original_row_count=0,
            original_column_count=0,
            row_compression_ratio=0.0,
            column_compression_ratio=0.0,
            total_compression_ratio=0.0,
            column_mi_scores={},
            row_mi_scores=[],
            total_mi=0.0,
            estimated_tokens=0,
            token_budget=token_budget or 0,
            tokens_saved=0,
            strategy=self.default_strategy
        )

    def _update_stats(
        self,
        original_rows: int,
        original_cols: int,
        kept_rows: int,
        kept_cols: int,
        tokens_saved: int
    ) -> None:
        """Update packing statistics."""
        self._stats['total_packs'] += 1
        self._stats['total_rows_processed'] += original_rows
        self._stats['total_columns_processed'] += original_cols
        self._stats['total_rows_kept'] += kept_rows
        self._stats['total_columns_kept'] += kept_cols
        self._stats['total_tokens_saved'] += tokens_saved

        # Update averages
        if self._stats['total_rows_processed'] > 0:
            self._stats['avg_row_compression'] = (
                self._stats['total_rows_kept'] / self._stats['total_rows_processed']
            )
        if self._stats['total_columns_processed'] > 0:
            self._stats['avg_column_compression'] = (
                self._stats['total_columns_kept'] / self._stats['total_columns_processed']
            )

    def get_statistics(self) -> Dict[str, Any]:
        """Get packing statistics."""
        return {
            **self._stats,
            'cache_size': len(self._column_mi_cache),
        }

    def clear_cache(self) -> None:
        """Clear column MI cache."""
        self._column_mi_cache.clear()


# Convenience functions

def pack_sql_results(
    rows: List[Dict[str, Any]],
    query: str,
    max_rows: Optional[int] = None,
    max_columns: Optional[int] = None,
    token_budget: Optional[int] = None
) -> PackedSQLContext:
    """
    Convenience function for one-shot SQL packing.

    Args:
        rows: SQL query result rows
        query: Natural language query
        max_rows: Maximum rows to keep
        max_columns: Maximum columns to keep
        token_budget: Token budget constraint

    Returns:
        PackedSQLContext with optimized results

    Example:
        >>> rows = [{'id': 1, 'name': 'Alice', 'age': 30}, ...]
        >>> result = pack_sql_results(rows, "Show users over 25")
        >>> print(result)
    """
    packer = SQLContextPacker()
    return packer.pack_sql_results(
        rows=rows,
        query=query,
        max_rows=max_rows,
        max_columns=max_columns,
        token_budget=token_budget
    )


def get_adaptive_sql_budget(query: str) -> Tuple[int, int, int]:
    """
    Get adaptive row/column/token budget based on query complexity.

    Uses Phase 6.1 AdaptiveCompressionStrategy to determine appropriate limits.

    Args:
        query: Natural language query

    Returns:
        Tuple of (max_rows, max_columns, token_budget)

    Example:
        >>> max_rows, max_cols, budget = get_adaptive_sql_budget("count users")
        >>> print(f"Limits: {max_rows} rows, {max_cols} cols, {budget} tokens")
    """
    try:
        from hololoom.context_packing import get_adaptive_mi_budget, AdaptiveComplexity
        from hololoom.context_packing.adaptive_strategy import AdaptiveCompressionStrategy

        strategy = AdaptiveCompressionStrategy()
        result = strategy.select_strategy(query)

        # Map complexity to SQL limits
        limits = {
            AdaptiveComplexity.TRIVIAL: (10, 5, 500),
            AdaptiveComplexity.SIMPLE: (20, 8, 1000),
            AdaptiveComplexity.MODERATE: (50, 10, 2000),
            AdaptiveComplexity.COMPLEX: (100, 15, 4000),
            AdaptiveComplexity.RESEARCH: (500, 20, 8000),
        }

        return limits.get(result.complexity, (50, 10, 2000))

    except ImportError:
        # Fallback defaults
        return (50, 10, 2000)
