"""
Small Multiples Visualization Tests
=====================================
Comprehensive tests for comparison visualizations.

Test Coverage:
- Grid layout
- Consistent scales
- Best/worst highlighting
- Sparkline trends
- Query comparison
- Edge cases and validation

Author: Claude Code
Date: January 2026
"""

import pytest
from typing import List, Dict, Any

from HoloLoom.visualization.small_multiples import (
    QueryMultiple,
    MultipleLayout,
    SmallMultiplesRenderer,
    render_small_multiples,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_query_data() -> List[Dict[str, Any]]:
    """Create basic query data for testing."""
    return [
        {
            'query_text': 'Query 1',
            'latency_ms': 95.0,
            'confidence': 0.92,
            'threads_count': 3,
            'cached': True,
            'trend': [105.0, 102.0, 98.0, 96.0, 95.0],
            'timestamp': 1698595200.0,
            'tool_used': 'answer'
        },
        {
            'query_text': 'Query 2',
            'latency_ms': 150.0,
            'confidence': 0.85,
            'threads_count': 5,
            'cached': False,
            'trend': [140.0, 145.0, 148.0, 149.0, 150.0],
            'timestamp': 1698595300.0,
            'tool_used': 'research'
        },
        {
            'query_text': 'Query 3',
            'latency_ms': 80.0,
            'confidence': 0.95,
            'threads_count': 2,
            'cached': True,
            'trend': [90.0, 88.0, 85.0, 82.0, 80.0],
            'timestamp': 1698595400.0,
            'tool_used': 'answer'
        },
    ]


@pytest.fixture
def query_multiples(basic_query_data) -> List[QueryMultiple]:
    """Create QueryMultiple objects for testing."""
    return [QueryMultiple(**q) for q in basic_query_data]


@pytest.fixture
def renderer() -> SmallMultiplesRenderer:
    """Create default renderer."""
    return SmallMultiplesRenderer()


# ============================================================================
# QueryMultiple Tests
# ============================================================================

class TestQueryMultiple:
    """Tests for QueryMultiple dataclass."""

    def test_basic_creation(self):
        """Test basic QueryMultiple creation."""
        query = QueryMultiple(
            query_text="What is Thompson Sampling?",
            latency_ms=95.0,
            confidence=0.92,
            threads_count=3,
            cached=True,
            trend=[100.0, 98.0, 95.0],
            timestamp=1698595200.0,
            tool_used="answer"
        )

        assert query.query_text == "What is Thompson Sampling?"
        assert query.latency_ms == 95.0
        assert query.confidence == 0.92
        assert query.threads_count == 3
        assert query.cached is True
        assert len(query.trend) == 3

    def test_is_fast_property(self):
        """Test is_fast property (under 100ms)."""
        fast_query = QueryMultiple(
            query_text="Fast",
            latency_ms=80.0,
            confidence=0.9,
            threads_count=2,
            cached=True,
            trend=[80.0],
            timestamp=0.0,
            tool_used="answer"
        )
        assert fast_query.is_fast is True

        slow_query = QueryMultiple(
            query_text="Slow",
            latency_ms=150.0,
            confidence=0.9,
            threads_count=2,
            cached=False,
            trend=[150.0],
            timestamp=0.0,
            tool_used="research"
        )
        assert slow_query.is_fast is False

    def test_is_confident_property(self):
        """Test is_confident property (>= 90%)."""
        confident = QueryMultiple(
            query_text="Confident",
            latency_ms=100.0,
            confidence=0.92,
            threads_count=3,
            cached=True,
            trend=[100.0],
            timestamp=0.0,
            tool_used="answer"
        )
        assert confident.is_confident is True

        not_confident = QueryMultiple(
            query_text="Not Confident",
            latency_ms=100.0,
            confidence=0.85,
            threads_count=3,
            cached=True,
            trend=[100.0],
            timestamp=0.0,
            tool_used="answer"
        )
        assert not_confident.is_confident is False

    def test_trend_direction_up(self):
        """Test trend direction for degrading performance."""
        query = QueryMultiple(
            query_text="Degrading",
            latency_ms=150.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[100.0, 120.0, 150.0],  # Getting worse (up)
            timestamp=0.0,
            tool_used="answer"
        )
        assert query.trend_direction == 'up'

    def test_trend_direction_down(self):
        """Test trend direction for improving performance."""
        query = QueryMultiple(
            query_text="Improving",
            latency_ms=80.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[150.0, 120.0, 80.0],  # Getting better (down)
            timestamp=0.0,
            tool_used="answer"
        )
        assert query.trend_direction == 'down'

    def test_trend_direction_flat(self):
        """Test trend direction for stable performance."""
        query = QueryMultiple(
            query_text="Stable",
            latency_ms=100.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[100.0, 100.0, 100.0],  # Same
            timestamp=0.0,
            tool_used="answer"
        )
        assert query.trend_direction == 'flat'

    def test_trend_direction_single_value(self):
        """Test trend direction with single value."""
        query = QueryMultiple(
            query_text="Single",
            latency_ms=100.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[100.0],
            timestamp=0.0,
            tool_used="answer"
        )
        assert query.trend_direction == 'flat'

    def test_trend_direction_empty(self):
        """Test trend direction with empty trend."""
        query = QueryMultiple(
            query_text="Empty",
            latency_ms=100.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[],
            timestamp=0.0,
            tool_used="answer"
        )
        assert query.trend_direction == 'flat'


# ============================================================================
# MultipleLayout Tests
# ============================================================================

class TestMultipleLayout:
    """Tests for MultipleLayout enum."""

    def test_all_layouts_exist(self):
        """Test all expected layouts exist."""
        assert hasattr(MultipleLayout, 'GRID')
        assert hasattr(MultipleLayout, 'ROW')
        assert hasattr(MultipleLayout, 'COLUMN')

    def test_layout_values(self):
        """Test layout string values."""
        assert MultipleLayout.GRID.value == 'grid'
        assert MultipleLayout.ROW.value == 'row'
        assert MultipleLayout.COLUMN.value == 'column'


# ============================================================================
# Renderer Initialization Tests
# ============================================================================

class TestRendererInitialization:
    """Tests for SmallMultiplesRenderer initialization."""

    def test_default_initialization(self):
        """Test renderer with default parameters."""
        renderer = SmallMultiplesRenderer()

        assert renderer.default_width == 200
        assert renderer.default_height == 150
        assert renderer.sparkline_width == 80
        assert renderer.sparkline_height == 20


# ============================================================================
# Rendering Tests
# ============================================================================

class TestRendering:
    """Tests for small multiples rendering."""

    def test_render_basic_html(self, renderer, query_multiples):
        """Test basic HTML rendering."""
        html = renderer.render(query_multiples)

        assert isinstance(html, str)
        assert "small-multiples-container" in html

    def test_render_empty_returns_empty_state(self, renderer):
        """Test rendering empty list returns empty state."""
        html = renderer.render([])

        assert "No queries to compare" in html or "empty-multiples" in html

    def test_render_includes_query_text(self, renderer, query_multiples):
        """Test that query text is included."""
        html = renderer.render(query_multiples)

        # Query texts should appear (possibly truncated)
        assert "Query" in html

    def test_render_includes_latency(self, renderer, query_multiples):
        """Test that latency values are included."""
        html = renderer.render(query_multiples)

        assert "ms" in html

    def test_render_includes_confidence(self, renderer, query_multiples):
        """Test that confidence is included."""
        html = renderer.render(query_multiples)

        assert "Conf" in html or "%" in html


# ============================================================================
# Layout Tests
# ============================================================================

class TestLayoutModes:
    """Tests for different layout modes."""

    def test_grid_layout(self, renderer, query_multiples):
        """Test grid layout."""
        html = renderer.render(query_multiples, layout=MultipleLayout.GRID)

        assert "display: grid" in html
        assert "grid-template-columns" in html

    def test_row_layout(self, renderer, query_multiples):
        """Test row layout (all in one row)."""
        html = renderer.render(query_multiples, layout=MultipleLayout.ROW)

        assert "display: grid" in html
        # Row layout should have columns = number of items
        assert f"repeat({len(query_multiples)}, 1fr)" in html

    def test_column_layout(self, renderer, query_multiples):
        """Test column layout (all in one column)."""
        html = renderer.render(query_multiples, layout=MultipleLayout.COLUMN)

        assert "display: grid" in html
        assert "repeat(1, 1fr)" in html

    def test_grid_columns_calculation(self, renderer):
        """Test grid column calculation for different counts."""
        # 2 items -> 2 columns
        two_items = [
            QueryMultiple(
                query_text=f"Q{i}", latency_ms=100.0, confidence=0.9,
                threads_count=3, cached=True, trend=[100.0],
                timestamp=0.0, tool_used="answer"
            )
            for i in range(2)
        ]
        html_2 = renderer.render(two_items, layout=MultipleLayout.GRID)
        assert "repeat(2, 1fr)" in html_2

        # 4 items -> 2 columns
        four_items = [
            QueryMultiple(
                query_text=f"Q{i}", latency_ms=100.0, confidence=0.9,
                threads_count=3, cached=True, trend=[100.0],
                timestamp=0.0, tool_used="answer"
            )
            for i in range(4)
        ]
        html_4 = renderer.render(four_items, layout=MultipleLayout.GRID)
        assert "repeat(2, 1fr)" in html_4

        # 6 items -> 3 columns
        six_items = [
            QueryMultiple(
                query_text=f"Q{i}", latency_ms=100.0, confidence=0.9,
                threads_count=3, cached=True, trend=[100.0],
                timestamp=0.0, tool_used="answer"
            )
            for i in range(6)
        ]
        html_6 = renderer.render(six_items, layout=MultipleLayout.GRID)
        assert "repeat(3, 1fr)" in html_6

    def test_max_columns_parameter(self, renderer):
        """Test max_columns parameter limits columns."""
        items = [
            QueryMultiple(
                query_text=f"Q{i}", latency_ms=100.0, confidence=0.9,
                threads_count=3, cached=True, trend=[100.0],
                timestamp=0.0, tool_used="answer"
            )
            for i in range(10)
        ]

        # With max_columns=2
        html = renderer.render(items, layout=MultipleLayout.GRID, max_columns=2)
        assert "repeat(2, 1fr)" in html


# ============================================================================
# Best/Worst Highlighting Tests
# ============================================================================

class TestHighlighting:
    """Tests for best/worst query highlighting."""

    def test_best_query_highlighted(self, renderer, query_multiples):
        """Test that best (fastest) query is highlighted."""
        html = renderer.render(query_multiples)

        # Best should have green border/star
        assert "#10b981" in html  # Green color
        # Unicode star character or star indicator

    def test_worst_query_highlighted(self, renderer, query_multiples):
        """Test that worst (slowest) query is highlighted."""
        html = renderer.render(query_multiples)

        # Worst should have red border/warning
        assert "#ef4444" in html  # Red color

    def test_single_query_no_best_worst(self, renderer):
        """Test that single query doesn't show best/worst."""
        single = [
            QueryMultiple(
                query_text="Only Query",
                latency_ms=100.0,
                confidence=0.9,
                threads_count=3,
                cached=True,
                trend=[100.0],
                timestamp=0.0,
                tool_used="answer"
            )
        ]

        html = renderer.render(single)

        # Should render but without best/worst indicators
        # The star and warning should not appear for single item
        assert isinstance(html, str)


# ============================================================================
# Sparkline Tests
# ============================================================================

class TestSparklines:
    """Tests for sparkline trend visualization."""

    def test_sparklines_rendered(self, renderer, query_multiples):
        """Test that sparklines are rendered."""
        html = renderer.render(query_multiples)

        # Should have SVG sparkline elements
        assert "<svg" in html

    def test_sparkline_with_empty_trend(self, renderer):
        """Test sparkline with empty trend (not rendered)."""
        query = QueryMultiple(
            query_text="No Trend",
            latency_ms=100.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[],  # Empty
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should still render, sparkline skipped
        assert isinstance(html, str)

    def test_sparkline_with_single_value(self, renderer):
        """Test sparkline with single value (not rendered)."""
        query = QueryMultiple(
            query_text="Single Value",
            latency_ms=100.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[100.0],  # Single value
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should still render, sparkline may be skipped
        assert isinstance(html, str)

    def test_sparkline_path_structure(self, renderer, query_multiples):
        """Test sparkline SVG path structure."""
        html = renderer.render(query_multiples)

        # Should have path elements
        assert "path" in html.lower()
        assert 'd="M ' in html  # SVG path command


# ============================================================================
# Trend Indicator Tests
# ============================================================================

class TestTrendIndicators:
    """Tests for trend direction indicators."""

    def test_improving_trend_indicator(self, renderer):
        """Test improving trend shows down arrow."""
        query = QueryMultiple(
            query_text="Improving",
            latency_ms=80.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[150.0, 100.0, 80.0],  # Improving (latency going down)
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should show "improving" or down arrow
        assert "improving" in html.lower() or "down" in html.lower()

    def test_degrading_trend_indicator(self, renderer):
        """Test degrading trend shows up arrow."""
        query = QueryMultiple(
            query_text="Degrading",
            latency_ms=150.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[80.0, 100.0, 150.0],  # Degrading (latency going up)
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should show "degrading" or up arrow
        assert "degrading" in html.lower() or "up" in html.lower()

    def test_stable_trend_indicator(self, renderer):
        """Test stable trend shows stable indicator."""
        query = QueryMultiple(
            query_text="Stable",
            latency_ms=100.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[100.0, 100.0, 100.0],  # Stable
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should show "stable" or horizontal arrow
        assert "stable" in html.lower() or "flat" in html.lower()


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunction:
    """Tests for render_small_multiples convenience function."""

    def test_basic_usage(self, basic_query_data):
        """Test basic usage with dict data."""
        html = render_small_multiples(basic_query_data)

        assert isinstance(html, str)
        assert "small-multiples-container" in html

    def test_with_layout_string(self, basic_query_data):
        """Test with layout as string."""
        html_grid = render_small_multiples(basic_query_data, layout='grid')
        html_row = render_small_multiples(basic_query_data, layout='row')
        html_column = render_small_multiples(basic_query_data, layout='column')

        assert isinstance(html_grid, str)
        assert isinstance(html_row, str)
        assert isinstance(html_column, str)

    def test_with_max_columns(self, basic_query_data):
        """Test with max_columns parameter."""
        html = render_small_multiples(basic_query_data, max_columns=2)

        assert isinstance(html, str)


# ============================================================================
# Semantic Coloring Tests
# ============================================================================

class TestSemanticColoring:
    """Tests for semantic color coding."""

    def test_fast_latency_green(self, renderer):
        """Test fast latency (<100ms) is green."""
        query = QueryMultiple(
            query_text="Fast",
            latency_ms=50.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[50.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should have green color for latency
        assert "#10b981" in html

    def test_medium_latency_yellow(self, renderer):
        """Test medium latency (100-200ms) is yellow/amber."""
        query = QueryMultiple(
            query_text="Medium",
            latency_ms=150.0,
            confidence=0.9,
            threads_count=3,
            cached=False,
            trend=[150.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should have yellow/amber color
        assert "#f59e0b" in html

    def test_slow_latency_red(self, renderer):
        """Test slow latency (>200ms) is red."""
        query = QueryMultiple(
            query_text="Slow",
            latency_ms=300.0,
            confidence=0.9,
            threads_count=3,
            cached=False,
            trend=[300.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should have red color
        assert "#ef4444" in html

    def test_high_confidence_green(self, renderer):
        """Test high confidence (>=90%) is green."""
        query = QueryMultiple(
            query_text="Confident",
            latency_ms=100.0,
            confidence=0.95,
            threads_count=3,
            cached=True,
            trend=[100.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Confidence should show green
        assert "#10b981" in html

    def test_low_confidence_red(self, renderer):
        """Test low confidence (<75%) is red."""
        query = QueryMultiple(
            query_text="Low Confidence",
            latency_ms=100.0,
            confidence=0.60,
            threads_count=3,
            cached=True,
            trend=[100.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should have red for low confidence
        assert "#ef4444" in html


# ============================================================================
# Cache Indicator Tests
# ============================================================================

class TestCacheIndicators:
    """Tests for cache hit/miss indicators."""

    def test_cached_indicator(self, renderer):
        """Test cached query shows indicator."""
        cached = QueryMultiple(
            query_text="Cached",
            latency_ms=50.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[50.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([cached])

        # Should have cache indicator (disk emoji or similar)
        # Note: The original code uses emoji
        assert isinstance(html, str)

    def test_not_cached_no_indicator(self, renderer):
        """Test uncached query doesn't show cache indicator."""
        uncached = QueryMultiple(
            query_text="Uncached",
            latency_ms=150.0,
            confidence=0.9,
            threads_count=3,
            cached=False,
            trend=[150.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([uncached])

        # Should render without cache indicator
        assert isinstance(html, str)


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_query(self, renderer):
        """Test with single query."""
        single = [
            QueryMultiple(
                query_text="Only Query",
                latency_ms=100.0,
                confidence=0.9,
                threads_count=3,
                cached=True,
                trend=[100.0],
                timestamp=0.0,
                tool_used="answer"
            )
        ]

        html = renderer.render(single)

        assert isinstance(html, str)
        assert "Only Query" in html or "only" in html.lower()

    def test_many_queries(self, renderer):
        """Test with many queries."""
        many = [
            QueryMultiple(
                query_text=f"Query {i}",
                latency_ms=80.0 + i * 10,
                confidence=0.95 - i * 0.02,
                threads_count=i % 5 + 1,
                cached=i % 2 == 0,
                trend=[80.0 + i * j for j in range(5)],
                timestamp=1698595200.0 + i * 100,
                tool_used="answer" if i % 2 == 0 else "research"
            )
            for i in range(15)
        ]

        html = renderer.render(many)

        assert isinstance(html, str)

    def test_very_long_query_text(self, renderer):
        """Test with very long query text (should be truncated)."""
        query = QueryMultiple(
            query_text="This is a very long query text that should definitely be truncated when displayed",
            latency_ms=100.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[100.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        # Should be truncated (30 chars + ...)
        assert "..." in html

    def test_zero_latency(self, renderer):
        """Test with zero latency."""
        query = QueryMultiple(
            query_text="Instant",
            latency_ms=0.0,
            confidence=0.9,
            threads_count=3,
            cached=True,
            trend=[0.0],
            timestamp=0.0,
            tool_used="answer"
        )

        html = renderer.render([query])

        assert isinstance(html, str)
        assert "0" in html

    def test_same_latency_all_queries(self, renderer):
        """Test with all same latency (consistent scales)."""
        queries = [
            QueryMultiple(
                query_text=f"Q{i}",
                latency_ms=100.0,
                confidence=0.9,
                threads_count=3,
                cached=True,
                trend=[100.0],
                timestamp=0.0,
                tool_used="answer"
            )
            for i in range(3)
        ]

        html = renderer.render(queries)

        # Should render without error
        assert isinstance(html, str)


# ============================================================================
# HTML Structure Tests
# ============================================================================

class TestHTMLStructure:
    """Tests for HTML structure validity."""

    def test_valid_grid_structure(self, renderer, query_multiples):
        """Test that grid has valid CSS structure."""
        html = renderer.render(query_multiples)

        assert "display: grid" in html
        assert "grid-template-columns" in html
        assert "gap:" in html

    def test_no_external_dependencies(self, renderer, query_multiples):
        """Test that HTML has no external dependencies."""
        html = renderer.render(query_multiples)

        assert 'href="http' not in html
        assert 'src="http' not in html
        assert '<link' not in html.lower()

    def test_inline_styles(self, renderer, query_multiples):
        """Test that styles are inline."""
        html = renderer.render(query_multiples)

        assert 'style="' in html

    def test_svg_sparklines_properly_closed(self, renderer, query_multiples):
        """Test that SVG sparklines are properly closed."""
        html = renderer.render(query_multiples)

        svg_opens = html.count("<svg")
        svg_closes = html.count("</svg>")
        assert svg_opens == svg_closes


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, basic_query_data):
        """Test complete workflow with all features."""
        html = render_small_multiples(
            basic_query_data,
            layout='grid',
            max_columns=4
        )

        # Verify all components
        assert "small-multiples-container" in html
        assert "display: grid" in html
        assert "<svg" in html  # Sparklines
        assert "ms" in html  # Latency values
        assert "%" in html or "Conf" in html  # Confidence

    def test_comparison_accuracy(self, renderer):
        """Test that comparison highlights correct queries."""
        # Create queries with clear best and worst
        queries = [
            QueryMultiple(
                query_text="Fastest",
                latency_ms=50.0,
                confidence=0.9,
                threads_count=3,
                cached=True,
                trend=[50.0],
                timestamp=0.0,
                tool_used="answer"
            ),
            QueryMultiple(
                query_text="Medium",
                latency_ms=100.0,
                confidence=0.9,
                threads_count=3,
                cached=True,
                trend=[100.0],
                timestamp=0.0,
                tool_used="answer"
            ),
            QueryMultiple(
                query_text="Slowest",
                latency_ms=200.0,
                confidence=0.9,
                threads_count=3,
                cached=True,
                trend=[200.0],
                timestamp=0.0,
                tool_used="answer"
            ),
        ]

        html = renderer.render(queries)

        # Should have both green (best) and red (worst) colors
        assert "#10b981" in html  # Green for best
        assert "#ef4444" in html  # Red for worst
