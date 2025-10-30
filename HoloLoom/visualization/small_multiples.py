"""
Small Multiples Renderer - Edward Tufte Visualization
======================================================
Enable comparison through repetition of consistent, compact charts.

"At the heart of quantitative reasoning is a single question: Compared to what?"
- Edward Tufte

Author: Claude Code
Date: October 29, 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class MultipleLayout(Enum):
    """Layout strategy for small multiples."""
    GRID = 'grid'  # Auto grid (2-4 columns)
    ROW = 'row'    # Horizontal row
    COLUMN = 'column'  # Vertical column


@dataclass
class QueryMultiple:
    """Single query for small multiples comparison."""
    query_text: str
    latency_ms: float
    confidence: float
    threads_count: int
    cached: bool
    trend: List[float]  # Recent latency trend
    timestamp: float
    tool_used: str

    @property
    def is_fast(self) -> bool:
        """Query completed under 100ms."""
        return self.latency_ms < 100

    @property
    def is_confident(self) -> bool:
        """Confidence above 90%."""
        return self.confidence >= 0.90

    @property
    def trend_direction(self) -> str:
        """up, down, or flat."""
        if len(self.trend) < 2:
            return 'flat'
        return 'down' if self.trend[-1] < self.trend[0] else 'up' if self.trend[-1] > self.trend[0] else 'flat'


class SmallMultiplesRenderer:
    """
    Renders small multiples for query comparison.

    Principles:
    - Consistent scales across all multiples
    - Minimal decoration (data-ink ratio)
    - Compact size (fit 3-6 on screen)
    - Highlight differences (color code extremes)
    """

    def __init__(self):
        self.default_width = 200  # px per multiple
        self.default_height = 150  # px per multiple
        self.sparkline_width = 80
        self.sparkline_height = 20

    def render(
        self,
        queries: List[QueryMultiple],
        layout: MultipleLayout = MultipleLayout.GRID,
        max_columns: int = 4
    ) -> str:
        """
        Render small multiples HTML.

        Args:
            queries: List of queries to compare
            layout: Layout strategy
            max_columns: Maximum columns for grid layout

        Returns:
            HTML string with small multiples
        """
        if not queries:
            return '<div class="empty-multiples">No queries to compare</div>'

        # Determine layout
        grid_cols = self._get_grid_columns(len(queries), max_columns, layout)

        # Find global min/max for consistent scales
        all_latencies = [q.latency_ms for q in queries]
        min_latency = min(all_latencies)
        max_latency = max(all_latencies)

        # Find best/worst for highlighting
        best_query = min(queries, key=lambda q: q.latency_ms)
        worst_query = max(queries, key=lambda q: q.latency_ms)

        # Render each multiple
        multiples_html = []
        for query in queries:
            is_best = (query == best_query) and len(queries) > 1
            is_worst = (query == worst_query) and len(queries) > 1

            multiple_html = self._render_single_multiple(
                query,
                min_latency,
                max_latency,
                is_best,
                is_worst
            )
            multiples_html.append(multiple_html)

        # Container with grid
        container_html = f"""
        <div class="small-multiples-container"
             style="display: grid;
                    grid-template-columns: repeat({grid_cols}, 1fr);
                    gap: 16px;
                    margin: 16px 0;">
            {''.join(multiples_html)}
        </div>
        """

        return container_html

    def _render_single_multiple(
        self,
        query: QueryMultiple,
        global_min: float,
        global_max: float,
        is_best: bool,
        is_worst: bool
    ) -> str:
        """Render a single small multiple."""

        # Determine border color based on performance
        if is_best:
            border_color = '#10b981'  # green
            border_width = '2px'
            indicator = '<span style="color: #10b981; font-size: 16px;">★</span>'
        elif is_worst:
            border_color = '#ef4444'  # red
            border_width = '2px'
            indicator = '<span style="color: #ef4444; font-size: 16px;">⚠</span>'
        else:
            border_color = '#e5e7eb'  # gray
            border_width = '1px'
            indicator = ''

        # Latency color (semantic)
        if query.latency_ms < 100:
            latency_color = '#10b981'  # green
        elif query.latency_ms < 200:
            latency_color = '#f59e0b'  # yellow
        else:
            latency_color = '#ef4444'  # red

        # Confidence color
        if query.confidence >= 0.90:
            conf_color = '#10b981'
        elif query.confidence >= 0.75:
            conf_color = '#f59e0b'
        else:
            conf_color = '#ef4444'

        # Trend sparkline
        sparkline_svg = self._render_sparkline(query.trend, latency_color)

        # Cache indicator
        cache_badge = '💾' if query.cached else ''

        # Truncate query text
        truncated_query = query.query_text[:30] + '...' if len(query.query_text) > 30 else query.query_text

        # Render HTML
        html = f"""
        <div class="small-multiple"
             style="border: {border_width} solid {border_color};
                    border-radius: 6px;
                    padding: 12px;
                    background: #ffffff;
                    min-width: {self.default_width}px;">

            <!-- Query Text -->
            <div style="font-size: 11px;
                        color: #6b7280;
                        margin-bottom: 8px;
                        font-weight: 500;
                        overflow: hidden;
                        text-overflow: ellipsis;
                        white-space: nowrap;"
                 title="{query.query_text}">
                {indicator} {truncated_query} {cache_badge}
            </div>

            <!-- Main Metric: Latency -->
            <div style="font-size: 28px;
                        font-weight: bold;
                        color: {latency_color};
                        margin-bottom: 4px;">
                {query.latency_ms:.0f}<span style="font-size: 14px; font-weight: normal; color: #9ca3af;">ms</span>
            </div>

            <!-- Sparkline -->
            {sparkline_svg}

            <!-- Secondary Metrics -->
            <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 11px;">
                <div>
                    <span style="color: #9ca3af;">Conf:</span>
                    <span style="color: {conf_color}; font-weight: 600;">{query.confidence * 100:.0f}%</span>
                </div>
                <div>
                    <span style="color: #9ca3af;">Threads:</span>
                    <span style="color: #374151; font-weight: 600;">{query.threads_count}</span>
                </div>
                <div>
                    <span style="color: #9ca3af;">Tool:</span>
                    <span style="color: #374151; font-weight: 600;">{query.tool_used[:6]}</span>
                </div>
            </div>

            <!-- Trend Direction Indicator -->
            <div style="margin-top: 6px; font-size: 10px; color: #9ca3af;">
                Trend: {self._get_trend_indicator(query.trend_direction)}
            </div>
        </div>
        """

        return html

    def _render_sparkline(self, values: List[float], color: str) -> str:
        """Render sparkline SVG for trend."""
        if not values or len(values) < 2:
            return ''

        width = self.sparkline_width
        height = self.sparkline_height

        # Normalize values
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val if max_val != min_val else 1

        # Generate path points
        padding = 2
        points = []
        for i, val in enumerate(values):
            x = padding + (i / (len(values) - 1)) * (width - 2 * padding)
            y = height - padding - ((val - min_val) / value_range) * (height - 2 * padding)
            points.append(f"{x:.1f},{y:.1f}")

        path_data = "M " + " L ".join(points)

        svg = f"""
        <div style="margin-top: 6px;">
            <svg width="{width}" height="{height}" style="display: block;">
                <path d="{path_data}"
                      fill="none"
                      stroke="{color}"
                      stroke-width="1"
                      opacity="0.6"/>
                <circle cx="{points[-1].split(',')[0]}"
                        cy="{points[-1].split(',')[1]}"
                        r="1.5"
                        fill="{color}"/>
            </svg>
        </div>
        """

        return svg

    def _get_trend_indicator(self, direction: str) -> str:
        """Get visual indicator for trend direction."""
        if direction == 'down':
            return '<span style="color: #10b981;">↓ improving</span>'
        elif direction == 'up':
            return '<span style="color: #ef4444;">↑ degrading</span>'
        else:
            return '<span style="color: #9ca3af;">→ stable</span>'

    def _get_grid_columns(self, num_items: int, max_columns: int, layout: MultipleLayout) -> int:
        """Determine number of columns for grid layout."""
        if layout == MultipleLayout.ROW:
            return num_items  # All in one row
        elif layout == MultipleLayout.COLUMN:
            return 1  # All in one column
        else:  # GRID
            # Auto-determine optimal columns
            if num_items <= 2:
                return num_items
            elif num_items <= 4:
                return 2
            elif num_items <= 6:
                return 3
            else:
                return min(max_columns, 4)


# Helper function for easy use
def render_small_multiples(
    queries_data: List[Dict[str, Any]],
    layout: str = 'grid',
    max_columns: int = 4
) -> str:
    """
    Convenience function to render small multiples from dict data.

    Args:
        queries_data: List of query dicts with keys:
            - query_text: str
            - latency_ms: float
            - confidence: float
            - threads_count: int
            - cached: bool
            - trend: List[float]
            - timestamp: float
            - tool_used: str
        layout: 'grid', 'row', or 'column'
        max_columns: Max columns for grid

    Returns:
        HTML string
    """
    # Convert dicts to QueryMultiple objects
    queries = [QueryMultiple(**q) for q in queries_data]

    # Map layout string to enum
    layout_enum = {
        'grid': MultipleLayout.GRID,
        'row': MultipleLayout.ROW,
        'column': MultipleLayout.COLUMN
    }.get(layout, MultipleLayout.GRID)

    # Render
    renderer = SmallMultiplesRenderer()
    return renderer.render(queries, layout_enum, max_columns)
