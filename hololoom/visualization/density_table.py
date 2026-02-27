"""
Data Density Tables - Edward Tufte Visualization
=================================================
Maximum information per square inch. Tables that don't waste space.

"Graphical excellence is that which gives to the viewer the greatest number
of ideas in the shortest time with the least ink in the smallest space."
- Edward Tufte

Author: Claude Code
Date: October 29, 2025
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class ColumnType(Enum):
    """Column data types for proper formatting."""
    TEXT = 'text'
    NUMBER = 'number'
    PERCENT = 'percent'
    DELTA = 'delta'  # Change from previous
    SPARKLINE = 'sparkline'
    INDICATOR = 'indicator'  # Boolean/status
    DURATION = 'duration'  # Time duration


class ColumnAlign(Enum):
    """Text alignment for columns."""
    LEFT = 'left'
    RIGHT = 'right'
    CENTER = 'center'


@dataclass
class Column:
    """Table column definition."""
    name: str
    type: ColumnType
    align: ColumnAlign = ColumnAlign.LEFT
    unit: Optional[str] = None  # 'ms', '%', etc.
    width: Optional[int] = None  # px or None for auto


@dataclass
class Row:
    """Table row data."""
    cells: Dict[str, Any]  # column_name -> value
    highlight: bool = False  # Highlight this row (bottleneck, error, etc.)
    highlight_color: str = '#fef2f2'  # Light red background


class DensityTableRenderer:
    """
    Renders high-density tables for dashboard metrics.

    Principles:
    - Tight spacing (minimal padding)
    - Right-align numbers, left-align text
    - Inline visualizations (sparklines)
    - Subtle gridlines (gray, low opacity)
    - Monospace numbers for alignment
    - Small font (10-11px) for density
    """

    def __init__(self):
        self.default_font_size = 11  # px
        self.header_font_size = 10  # px (smaller)
        self.row_height = 28  # px (tight)
        self.cell_padding = '4px 8px'
        self.sparkline_width = 60
        self.sparkline_height = 16

    def render(
        self,
        columns: List[Column],
        rows: List[Row],
        footer: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Render data density table HTML.

        Args:
            columns: Column definitions
            rows: Row data
            footer: Optional footer row (totals, etc.)
            title: Optional table title

        Returns:
            HTML string with density table
        """
        if not columns or not rows:
            return '<div class="empty-table">No data to display</div>'

        # Render title
        title_html = ''
        if title:
            title_html = f"""
            <div style="font-size: 12px;
                        font-weight: 600;
                        color: #374151;
                        margin-bottom: 8px;
                        padding-left: 8px;">
                {title}
            </div>
            """

        # Render header
        header_html = self._render_header(columns)

        # Render rows
        rows_html = [self._render_row(row, columns) for row in rows]

        # Render footer
        footer_html = ''
        if footer:
            footer_html = self._render_footer(footer, columns)

        # Complete table
        table_html = f"""
        <div class="density-table" style="margin: 16px 0;">
            {title_html}
            <div style="border: 1px solid #e5e7eb;
                        border-radius: 6px;
                        overflow: hidden;
                        background: #ffffff;">
                {header_html}
                {''.join(rows_html)}
                {footer_html}
            </div>
        </div>
        """

        return table_html

    def _render_header(self, columns: List[Column]) -> str:
        """Render table header."""
        cells = []
        for col in columns:
            align_style = col.align.value
            cells.append(f"""
                <div style="text-align: {align_style};
                            font-size: {self.header_font_size}px;
                            font-weight: 600;
                            color: #6b7280;
                            text-transform: uppercase;
                            letter-spacing: 0.5px;">
                    {col.name}
                </div>
            """)

        header_html = f"""
        <div style="display: grid;
                    grid-template-columns: {self._get_grid_template(columns)};
                    gap: 4px;
                    padding: {self.cell_padding};
                    background: #f9fafb;
                    border-bottom: 2px solid #e5e7eb;">
            {''.join(cells)}
        </div>
        """

        return header_html

    def _render_row(self, row: Row, columns: List[Column]) -> str:
        """Render single table row."""
        cells = []
        for col in columns:
            value = row.cells.get(col.name, '')
            cell_html = self._render_cell(value, col)
            cells.append(cell_html)

        # Highlight style
        bg_color = row.highlight_color if row.highlight else '#ffffff'
        hover_color = '#f9fafb'

        row_html = f"""
        <div style="display: grid;
                    grid-template-columns: {self._get_grid_template(columns)};
                    gap: 4px;
                    padding: {self.cell_padding};
                    background: {bg_color};
                    border-bottom: 1px solid #f3f4f6;
                    min-height: {self.row_height}px;
                    align-items: center;"
             onmouseover="this.style.background='{hover_color}'"
             onmouseout="this.style.background='{bg_color}'">
            {''.join(cells)}
        </div>
        """

        return row_html

    def _render_cell(self, value: Any, col: Column) -> str:
        """Render individual cell based on column type."""
        align_style = col.align.value

        # Format based on type
        if col.type == ColumnType.TEXT:
            formatted = str(value)
            font_family = 'inherit'
        elif col.type == ColumnType.NUMBER:
            formatted = f"{float(value):.1f}"
            if col.unit:
                formatted += col.unit
            font_family = 'monospace'
        elif col.type == ColumnType.PERCENT:
            formatted = f"{float(value):.0f}%"
            font_family = 'monospace'
        elif col.type == ColumnType.DURATION:
            formatted = f"{float(value):.0f}{col.unit or 'ms'}"
            font_family = 'monospace'
        elif col.type == ColumnType.DELTA:
            formatted = self._format_delta(value)
            font_family = 'monospace'
        elif col.type == ColumnType.SPARKLINE:
            # Render inline sparkline
            if isinstance(value, list) and len(value) >= 2:
                return self._render_sparkline(value)
            else:
                formatted = '—'
                font_family = 'inherit'
        elif col.type == ColumnType.INDICATOR:
            formatted = self._format_indicator(value)
            font_family = 'inherit'
        else:
            formatted = str(value)
            font_family = 'inherit'

        cell_html = f"""
        <div style="text-align: {align_style};
                    font-size: {self.default_font_size}px;
                    color: #374151;
                    font-family: {font_family};">
            {formatted}
        </div>
        """

        return cell_html

    def _render_footer(self, footer: Dict[str, Any], columns: List[Column]) -> str:
        """Render footer row (totals, etc.)."""
        cells = []
        for col in columns:
            value = footer.get(col.name, '')
            if value:
                cell_html = self._render_cell(value, col)
            else:
                cell_html = '<div></div>'  # Empty cell
            cells.append(cell_html)

        footer_html = f"""
        <div style="display: grid;
                    grid-template-columns: {self._get_grid_template(columns)};
                    gap: 4px;
                    padding: {self.cell_padding};
                    background: #f9fafb;
                    border-top: 2px solid #e5e7eb;
                    font-weight: 600;">
            {''.join(cells)}
        </div>
        """

        return footer_html

    def _get_grid_template(self, columns: List[Column]) -> str:
        """Generate CSS grid-template-columns value."""
        template_parts = []
        for col in columns:
            if col.width:
                template_parts.append(f"{col.width}px")
            else:
                # Auto-size based on type
                if col.type == ColumnType.TEXT:
                    template_parts.append("2fr")
                elif col.type == ColumnType.SPARKLINE:
                    template_parts.append(f"{self.sparkline_width}px")
                elif col.type == ColumnType.INDICATOR:
                    template_parts.append("80px")
                else:
                    template_parts.append("1fr")

        return " ".join(template_parts)

    def _format_delta(self, value: float) -> str:
        """Format delta with color and symbol."""
        if value > 0:
            return f'<span style="color: #ef4444;">+{value:.0f}</span>'
        elif value < 0:
            return f'<span style="color: #10b981;">{value:.0f}</span>'
        else:
            return '<span style="color: #9ca3af;">0</span>'

    def _format_indicator(self, value: Any) -> str:
        """Format boolean/status indicator."""
        if isinstance(value, bool):
            if value:
                return '<span style="color: #ef4444; font-weight: 600;">YES</span>'
            else:
                return '<span style="color: #9ca3af;">—</span>'
        else:
            return str(value)

    def _render_sparkline(self, values: List[float]) -> str:
        """Render inline sparkline SVG."""
        if not values or len(values) < 2:
            return '<div>—</div>'

        width = self.sparkline_width
        height = self.sparkline_height

        # Normalize
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val if max_val != min_val else 1

        # Generate path
        padding = 2
        points = []
        for i, val in enumerate(values):
            x = padding + (i / (len(values) - 1)) * (width - 2 * padding)
            y = height - padding - ((val - min_val) / value_range) * (height - 2 * padding)
            points.append(f"{x:.1f},{y:.1f}")

        path_data = "M " + " L ".join(points)

        # Color based on trend
        color = '#10b981' if values[-1] < values[0] else '#ef4444' if values[-1] > values[0] else '#9ca3af'

        svg_html = f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <svg width="{width}" height="{height}">
                <path d="{path_data}"
                      fill="none"
                      stroke="{color}"
                      stroke-width="1"
                      opacity="0.7"/>
                <circle cx="{points[-1].split(',')[0]}"
                        cy="{points[-1].split(',')[1]}"
                        r="1"
                        fill="{color}"/>
            </svg>
        </div>
        """

        return svg_html


# Helper function for common use case: stage timing table
def render_stage_timing_table(
    stages: List[Dict[str, Any]],
    total_duration: float,
    bottleneck_threshold: float = 0.4
) -> str:
    """
    Render stage timing table with bottleneck detection.

    Args:
        stages: List of stage dicts with keys:
            - name: str
            - duration_ms: float
            - trend: List[float] (optional)
            - delta: float (optional, change from previous)
        total_duration: Total duration for percentage calculation
        bottleneck_threshold: Percentage threshold for bottleneck (0.4 = 40%)

    Returns:
        HTML string with timing table
    """
    # Define columns
    columns = [
        Column('Stage', ColumnType.TEXT, ColumnAlign.LEFT),
        Column('Time', ColumnType.DURATION, ColumnAlign.RIGHT, unit='ms'),
        Column('%', ColumnType.PERCENT, ColumnAlign.RIGHT),
        Column('Δ', ColumnType.DELTA, ColumnAlign.RIGHT),
        Column('Trend', ColumnType.SPARKLINE, ColumnAlign.CENTER),
        Column('Bottleneck?', ColumnType.INDICATOR, ColumnAlign.LEFT),
    ]

    # Convert stages to rows
    rows = []
    for stage in stages:
        duration = stage['duration_ms']
        percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
        is_bottleneck = percentage >= (bottleneck_threshold * 100)

        row = Row(
            cells={
                'Stage': stage['name'],
                'Time': duration,
                '%': percentage,
                'Δ': stage.get('delta', 0),
                'Trend': stage.get('trend', []),
                'Bottleneck?': is_bottleneck
            },
            highlight=is_bottleneck,
            highlight_color='#fef2f2' if is_bottleneck else '#ffffff'
        )
        rows.append(row)

    # Footer with totals
    footer = {
        'Stage': 'Total',
        'Time': total_duration,
        '%': 100
    }

    # Render
    renderer = DensityTableRenderer()
    return renderer.render(
        columns=columns,
        rows=rows,
        footer=footer,
        title='Stage Timing Analysis'
    )
