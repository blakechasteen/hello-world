"""
Stage Waterfall Chart - Tufte-Style Sequential Pipeline Visualization

Shows sequential pipeline timing with horizontal stacked bars.
Highlights bottlenecks and dependencies with minimal decoration.

Author: Claude Code
Date: October 29, 2025
Principles: Edward Tufte - "Above all else show the data"
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class StageStatus(Enum):
    """Status of a pipeline stage."""
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class WaterfallStage:
    """
    A single stage in the pipeline waterfall.

    Attributes:
        name: Stage name (e.g., "Pattern Selection")
        start_ms: When stage started (relative to pipeline start)
        duration_ms: How long stage took
        status: Stage completion status
        trend: Historical durations for sparkline (optional)
        metadata: Additional stage info (dependencies, errors, etc.)
    """
    name: str
    start_ms: float
    duration_ms: float
    status: StageStatus = StageStatus.SUCCESS
    trend: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def end_ms(self) -> float:
        """Calculate when stage ended."""
        return self.start_ms + self.duration_ms


class StageWaterfallRenderer:
    """
    Render sequential pipeline timing as horizontal waterfall chart.

    Tufte Principles Applied:
    - Maximize data-ink ratio: No axes, grids, or decoration
    - Meaning first: Bottlenecks highlighted immediately
    - Data density: Show duration, %, offset, trend in compact space
    - Layering: Color, position, width, opacity convey multiple dimensions
    """

    def __init__(
        self,
        bottleneck_threshold: float = 0.4,
        show_sparklines: bool = True,
        show_percentages: bool = True
    ):
        """
        Initialize waterfall renderer.

        Args:
            bottleneck_threshold: % of total time to consider bottleneck (0.4 = 40%)
            show_sparklines: Include historical trend sparklines
            show_percentages: Show percentage of total time
        """
        self.bottleneck_threshold = bottleneck_threshold
        self.show_sparklines = show_sparklines
        self.show_percentages = show_percentages

    def render(
        self,
        stages: List[WaterfallStage],
        title: str = "Pipeline Stage Waterfall",
        total_duration_ms: Optional[float] = None
    ) -> str:
        """
        Render waterfall chart HTML.

        Args:
            stages: List of pipeline stages (in execution order)
            title: Chart title
            total_duration_ms: Total pipeline duration (auto-calculated if None)

        Returns:
            HTML string with waterfall visualization
        """
        if not stages:
            return self._render_empty()

        # Calculate total duration
        if total_duration_ms is None:
            total_duration_ms = max(s.end_ms for s in stages)

        # Detect bottlenecks
        bottlenecks = [
            s for s in stages
            if (s.duration_ms / total_duration_ms) >= self.bottleneck_threshold
        ]

        # Render header
        header_html = f"""
        <div class="waterfall-title" style="font-size: 14px; font-weight: 600;
                                             color: #374151; margin-bottom: 8px;">
            {title}
            <span style="font-size: 12px; font-weight: 400; color: #6b7280; margin-left: 8px;">
                Total: {total_duration_ms:.1f}ms
            </span>
        </div>
        """

        # Render stages
        stages_html = [self._render_stage(s, total_duration_ms, s in bottlenecks) for s in stages]

        # Render time axis
        axis_html = self._render_time_axis(total_duration_ms)

        # Combine
        return f"""
        <div class="stage-waterfall" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                                             background: #ffffff; padding: 16px; border-radius: 4px;">
            {header_html}
            <div class="waterfall-stages" style="margin: 16px 0;">
                {''.join(stages_html)}
            </div>
            {axis_html}
            {self._render_legend()}
        </div>
        """

    def _render_stage(self, stage: WaterfallStage, total_ms: float, is_bottleneck: bool) -> str:
        """Render a single stage row."""
        # Calculate percentages
        start_pct = (stage.start_ms / total_ms) * 100
        width_pct = (stage.duration_ms / total_ms) * 100
        time_pct = (stage.duration_ms / total_ms) * 100

        # Choose color based on status and bottleneck
        if stage.status == StageStatus.ERROR:
            color = "#ef4444"  # Red
            bg_color = "#fef2f2"
        elif is_bottleneck:
            color = "#f59e0b"  # Amber
            bg_color = "#fffbeb"
        elif stage.status == StageStatus.SUCCESS:
            color = "#10b981"  # Green
            bg_color = "#f0fdf4"
        elif stage.status == StageStatus.WARNING:
            color = "#f59e0b"  # Amber
            bg_color = "#fffbeb"
        else:  # SKIPPED
            color = "#9ca3af"  # Gray
            bg_color = "#f9fafb"

        # Bottleneck indicator
        bottleneck_badge = ""
        if is_bottleneck:
            bottleneck_badge = '<span style="background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 3px; font-size: 10px; margin-left: 8px; font-weight: 600;">BOTTLENECK</span>'

        # Status indicator
        status_icon = {
            StageStatus.SUCCESS: "&#10003;",  # Checkmark
            StageStatus.WARNING: "!",
            StageStatus.ERROR: "X",
            StageStatus.SKIPPED: "-"
        }.get(stage.status, "?")

        # Percentage display
        pct_display = ""
        if self.show_percentages:
            pct_display = f'<span style="color: #6b7280; font-size: 11px; margin-left: 8px;">({time_pct:.1f}%)</span>'

        # Sparkline (if available and enabled)
        sparkline_html = ""
        if self.show_sparklines and stage.trend and len(stage.trend) > 1:
            sparkline_html = self._generate_sparkline(stage.trend, color)

        # Build stage bar
        bar_html = f"""
        <div style="margin-left: {start_pct}%; width: {width_pct}%;
                    background: {color}; height: 24px; border-radius: 2px;
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-size: 11px; font-weight: 500;">
            {stage.duration_ms:.1f}ms
        </div>
        """

        # Build row
        return f"""
        <div class="waterfall-stage-row" style="margin-bottom: 12px;">
            <div style="display: flex; align-items: center; margin-bottom: 4px;">
                <span style="display: inline-block; width: 16px; height: 16px;
                             border-radius: 50%; background: {bg_color}; color: {color};
                             text-align: center; line-height: 16px; font-size: 10px; font-weight: 600;">
                    {status_icon}
                </span>
                <span style="font-size: 12px; font-weight: 500; color: #374151; margin-left: 8px;">
                    {stage.name}
                </span>
                {bottleneck_badge}
                {pct_display}
                {sparkline_html}
            </div>
            <div style="position: relative; height: 24px; background: #f3f4f6; border-radius: 2px;">
                {bar_html}
            </div>
        </div>
        """

    def _render_time_axis(self, total_ms: float) -> str:
        """Render minimal time axis with range markers."""
        # Calculate quartile markers
        markers = [0, total_ms * 0.25, total_ms * 0.5, total_ms * 0.75, total_ms]

        markers_html = []
        for i, ms in enumerate(markers):
            left_pct = (ms / total_ms) * 100
            markers_html.append(f"""
            <div style="position: absolute; left: {left_pct}%; top: 0;
                        border-left: 1px solid #d1d5db; height: 8px;">
                <div style="position: absolute; top: 10px; left: -12px;
                           font-size: 10px; color: #9ca3af; font-family: monospace;">
                    {ms:.0f}ms
                </div>
            </div>
            """)

        return f"""
        <div style="position: relative; height: 30px; margin-top: 16px; margin-bottom: 8px;">
            {''.join(markers_html)}
            <div style="position: absolute; top: 0; left: 0; right: 0;
                       border-top: 1px solid #e5e7eb;"></div>
        </div>
        """

    def _render_legend(self) -> str:
        """Render compact legend."""
        return f"""
        <div style="margin-top: 16px; padding-top: 12px; border-top: 1px solid #e5e7eb;">
            <div style="display: flex; gap: 16px; font-size: 11px; color: #6b7280;">
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 12px; height: 12px; background: #10b981; border-radius: 2px;"></div>
                    Success
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 12px; height: 12px; background: #f59e0b; border-radius: 2px;"></div>
                    Bottleneck ({self.bottleneck_threshold*100:.0f}%+)
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 12px; height: 12px; background: #ef4444; border-radius: 2px;"></div>
                    Error
                </div>
                <div style="display: flex; align-items: center; gap: 4px;">
                    <div style="width: 12px; height: 12px; background: #9ca3af; border-radius: 2px;"></div>
                    Skipped
                </div>
            </div>
        </div>
        """

    def _generate_sparkline(self, values: List[float], color: str) -> str:
        """Generate Tufte-style sparkline for trend."""
        if len(values) < 2:
            return ""

        # SVG dimensions (compact)
        width = 60
        height = 16
        padding = 2

        # Normalize values
        min_val = min(values)
        max_val = max(values)
        value_range = max_val - min_val if max_val != min_val else 1

        # Generate path points
        points = []
        for i, val in enumerate(values):
            x = padding + (i / (len(values) - 1)) * (width - 2 * padding)
            y = height - padding - ((val - min_val) / value_range) * (height - 2 * padding)
            points.append(f"{x:.1f},{y:.1f}")

        path = "M " + " L ".join(points)

        return f"""
        <svg width="{width}" height="{height}" style="margin-left: 8px; vertical-align: middle;">
            <path d="{path}" stroke="{color}" stroke-width="1.5" fill="none" opacity="0.6"/>
        </svg>
        """

    def _render_empty(self) -> str:
        """Render empty state."""
        return """
        <div style="padding: 24px; text-align: center; color: #9ca3af; font-size: 13px;">
            No pipeline stages to display
        </div>
        """


def render_pipeline_waterfall(
    stage_durations: Dict[str, float],
    stage_trends: Optional[Dict[str, List[float]]] = None,
    title: str = "Pipeline Stage Waterfall",
    bottleneck_threshold: float = 0.4
) -> str:
    """
    Convenience function to render waterfall from stage durations dict.

    Args:
        stage_durations: Dict mapping stage name -> duration_ms
        stage_trends: Optional dict mapping stage name -> historical durations
        title: Chart title
        bottleneck_threshold: Threshold for bottleneck highlighting

    Returns:
        HTML string with waterfall visualization

    Example:
        >>> durations = {
        ...     'Pattern Selection': 5.2,
        ...     'Retrieval': 50.5,
        ...     'Convergence': 30.0,
        ...     'Tool Execution': 64.3
        ... }
        >>> html = render_pipeline_waterfall(durations)
    """
    # Convert dict to WaterfallStage objects
    stages = []
    current_offset = 0.0

    for name, duration in stage_durations.items():
        trend = stage_trends.get(name) if stage_trends else None
        stage = WaterfallStage(
            name=name,
            start_ms=current_offset,
            duration_ms=duration,
            status=StageStatus.SUCCESS,
            trend=trend
        )
        stages.append(stage)
        current_offset += duration

    # Render
    renderer = StageWaterfallRenderer(
        bottleneck_threshold=bottleneck_threshold,
        show_sparklines=True,
        show_percentages=True
    )
    return renderer.render(stages, title=title)


def render_parallel_waterfall(
    stage_durations: Dict[str, float],
    parallel_groups: List[List[str]],
    title: str = "Pipeline Stage Waterfall (Parallel Execution)"
) -> str:
    """
    Render waterfall with parallel stage execution.

    Stages in the same parallel_group execute concurrently.

    Args:
        stage_durations: Dict mapping stage name -> duration_ms
        parallel_groups: List of stage name lists that execute in parallel
        title: Chart title

    Returns:
        HTML string with waterfall visualization

    Example:
        >>> durations = {
        ...     'Input Processing': 10.0,
        ...     'Feature A': 30.0,
        ...     'Feature B': 25.0,
        ...     'Feature C': 35.0,
        ...     'Decision': 20.0
        ... }
        >>> parallel = [
        ...     ['Input Processing'],
        ...     ['Feature A', 'Feature B', 'Feature C'],  # Parallel
        ...     ['Decision']
        ... ]
        >>> html = render_parallel_waterfall(durations, parallel)
    """
    stages = []
    current_offset = 0.0

    for group in parallel_groups:
        # All stages in group start at same time
        group_start = current_offset
        group_max_duration = 0.0

        for stage_name in group:
            duration = stage_durations.get(stage_name, 0.0)
            stage = WaterfallStage(
                name=stage_name,
                start_ms=group_start,
                duration_ms=duration,
                status=StageStatus.SUCCESS
            )
            stages.append(stage)
            group_max_duration = max(group_max_duration, duration)

        # Next group starts after longest stage in current group
        current_offset = group_start + group_max_duration

    renderer = StageWaterfallRenderer()
    return renderer.render(stages, title=title)
