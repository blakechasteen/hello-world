"""
Cache Effectiveness Gauge - Tufte-Style Radial Gauge

Visualize cache performance metrics with radial gauge and statistics.
Shows hit rate, time saved, and effectiveness scoring.

Author: Claude Code
Date: October 29, 2025
Principles: Edward Tufte - "Above all else show the data"

API Documentation:
    This module provides programmatic access to cache effectiveness
    visualization for automated tool calling, dashboard integration,
    and real-time monitoring systems.

    Primary Functions:
        - render_cache_gauge(): Main rendering function
        - CacheGaugeRenderer.render(): Full-featured renderer
        - calculate_cache_metrics(): Statistical calculation
        - estimate_time_saved(): Performance impact estimation

    Integration Points:
        - HoloLoom WeavingOrchestrator (via cache hit/miss tracking)
        - Dashboard Constructor (via panel data)
        - Performance monitoring (via metrics collection)
        - Alert systems (via threshold detection)

    Thread Safety:
        All functions are thread-safe for concurrent rendering.
        No shared mutable state.

    Performance:
        - Rendering: ~1-2ms
        - HTML size: ~6-8 KB
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
import math


class CacheEffectiveness(Enum):
    """
    Cache effectiveness rating based on hit rate and time saved.

    Values:
        EXCELLENT: Hit rate >80%, significant time savings
        GOOD: Hit rate 60-80%, moderate time savings
        FAIR: Hit rate 40-60%, some time savings
        POOR: Hit rate 20-40%, minimal time savings
        CRITICAL: Hit rate <20%, cache ineffective
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class CacheMetrics:
    """
    Cache performance metrics.

    Attributes:
        total_queries: Total number of queries processed
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        hit_rate: Cache hit rate [0.0, 1.0]
        avg_cached_latency_ms: Average latency for cached queries
        avg_uncached_latency_ms: Average latency for cache misses
        time_saved_ms: Estimated total time saved by cache
        effectiveness: Overall cache effectiveness rating

    Example:
        >>> metrics = CacheMetrics(
        ...     total_queries=100,
        ...     cache_hits=75,
        ...     cache_misses=25,
        ...     hit_rate=0.75,
        ...     avg_cached_latency_ms=15.0,
        ...     avg_uncached_latency_ms=120.0,
        ...     time_saved_ms=7875.0,
        ...     effectiveness=CacheEffectiveness.GOOD
        ... )
    """
    total_queries: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_cached_latency_ms: float
    avg_uncached_latency_ms: float
    time_saved_ms: float
    effectiveness: CacheEffectiveness


class CacheGaugeRenderer:
    """
    Render cache effectiveness as radial gauge with statistics.

    This renderer follows Tufte's principles:
    - Maximize data-ink ratio: Minimal decoration, focus on metrics
    - Meaning first: Effectiveness rating immediately visible
    - Data density: Hit rate + latencies + time saved in compact space
    - Direct labeling: Values shown inline, no separate legend

    Thread Safety:
        Each renderer instance is independent and thread-safe.
        Safe for concurrent rendering of different gauges.

    Performance:
        Rendering complexity: O(1) - constant time
        Memory usage: O(1) - constant memory

    Example:
        >>> renderer = CacheGaugeRenderer(show_details=True)
        >>> metrics = CacheMetrics(...)
        >>> html = renderer.render(metrics, title='Cache Performance')
    """

    def __init__(
        self,
        show_details: bool = True,
        show_recommendations: bool = True
    ):
        """
        Initialize cache gauge renderer.

        Args:
            show_details: Show detailed statistics below gauge
            show_recommendations: Show performance recommendations
        """
        self.show_details = show_details
        self.show_recommendations = show_recommendations

    def render(
        self,
        metrics: CacheMetrics,
        title: str = "Cache Effectiveness",
        subtitle: Optional[str] = None
    ) -> str:
        """
        Render cache gauge HTML.

        Args:
            metrics: Cache performance metrics
            title: Gauge title
            subtitle: Optional subtitle for context

        Returns:
            HTML string with complete gauge visualization

        Raises:
            ValueError: If metrics contain invalid values

        Example:
            >>> metrics = CacheMetrics(
            ...     total_queries=100,
            ...     cache_hits=75,
            ...     cache_misses=25,
            ...     hit_rate=0.75,
            ...     avg_cached_latency_ms=15.0,
            ...     avg_uncached_latency_ms=120.0,
            ...     time_saved_ms=7875.0,
            ...     effectiveness=CacheEffectiveness.GOOD
            ... )
            >>> html = renderer.render(metrics, title='Cache Performance')
        """
        # Validate metrics
        self._validate_metrics(metrics)

        # Render header
        header_html = self._render_header(title, subtitle, metrics)

        # Render radial gauge
        gauge_html = self._render_gauge(metrics)

        # Render statistics
        stats_html = ""
        if self.show_details:
            stats_html = self._render_statistics(metrics)

        # Render recommendations
        recommendations_html = ""
        if self.show_recommendations:
            recommendations_html = self._render_recommendations(metrics)

        # Combine
        return f"""
        <div class="cache-gauge" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                                        background: #ffffff; padding: 16px; border-radius: 4px;">
            {header_html}
            {gauge_html}
            {stats_html}
            {recommendations_html}
        </div>
        """

    def _validate_metrics(self, metrics: CacheMetrics) -> None:
        """Validate metrics for common errors."""
        if not 0.0 <= metrics.hit_rate <= 1.0:
            raise ValueError(f"hit_rate must be in [0.0, 1.0], got {metrics.hit_rate}")
        if metrics.total_queries != metrics.cache_hits + metrics.cache_misses:
            raise ValueError("total_queries must equal cache_hits + cache_misses")
        if metrics.avg_cached_latency_ms < 0 or metrics.avg_uncached_latency_ms < 0:
            raise ValueError("Latencies must be non-negative")

    def _render_header(self, title: str, subtitle: Optional[str], metrics: CacheMetrics) -> str:
        """Render gauge header with title and effectiveness rating."""
        subtitle_html = ""
        if subtitle:
            subtitle_html = f'<div style="font-size: 12px; color: #6b7280; margin-top: 4px;">{subtitle}</div>'

        # Effectiveness badge
        effectiveness_color, effectiveness_bg = self._effectiveness_colors(metrics.effectiveness)
        effectiveness_text = metrics.effectiveness.value.title()

        return f"""
        <div class="gauge-header" style="margin-bottom: 16px; text-align: center;">
            <div style="font-size: 14px; font-weight: 600; color: #374151;">{title}</div>
            {subtitle_html}
            <div style="margin-top: 8px;">
                <span style="display: inline-block; padding: 4px 12px; background: {effectiveness_bg};
                             color: {effectiveness_color}; border-radius: 12px; font-size: 11px;
                             font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">
                    {effectiveness_text}
                </span>
            </div>
        </div>
        """

    def _render_gauge(self, metrics: CacheMetrics) -> str:
        """Render SVG radial gauge."""
        # SVG dimensions
        size = 200
        center = size / 2
        radius = 70
        stroke_width = 16

        # Calculate arc for hit rate
        hit_rate_degrees = metrics.hit_rate * 270  # 270° = 0.75 of circle
        start_angle = 135  # Start at bottom-left
        end_angle = start_angle + hit_rate_degrees

        # Convert to radians
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)

        # Calculate arc path
        x1 = center + radius * math.cos(start_rad)
        y1 = center + radius * math.sin(start_rad)
        x2 = center + radius * math.cos(end_rad)
        y2 = center + radius * math.sin(end_rad)

        large_arc = 1 if hit_rate_degrees > 180 else 0

        arc_path = f"M {x1:.1f} {y1:.1f} A {radius} {radius} 0 {large_arc} 1 {x2:.1f} {y2:.1f}"

        # Background arc (full 270°)
        bg_end_angle = start_angle + 270
        bg_end_rad = math.radians(bg_end_angle)
        bg_x2 = center + radius * math.cos(bg_end_rad)
        bg_y2 = center + radius * math.sin(bg_end_rad)
        bg_arc_path = f"M {x1:.1f} {y1:.1f} A {radius} {radius} 0 1 1 {bg_x2:.1f} {bg_y2:.1f}"

        # Color based on effectiveness
        arc_color, _ = self._effectiveness_colors(metrics.effectiveness)

        # Center text
        hit_rate_text = f"{metrics.hit_rate*100:.1f}%"

        return f"""
        <div style="display: flex; justify-content: center; margin: 16px 0;">
            <svg width="{size}" height="{size}">
                <!-- Background arc -->
                <path d="{bg_arc_path}" stroke="#e5e7eb" stroke-width="{stroke_width}"
                      fill="none" stroke-linecap="round"/>

                <!-- Hit rate arc -->
                <path d="{arc_path}" stroke="{arc_color}" stroke-width="{stroke_width}"
                      fill="none" stroke-linecap="round"/>

                <!-- Center text -->
                <text x="{center}" y="{center - 10}" text-anchor="middle"
                      font-size="32" font-weight="700" fill="#374151">
                    {hit_rate_text}
                </text>
                <text x="{center}" y="{center + 15}" text-anchor="middle"
                      font-size="12" fill="#9ca3af">
                    Hit Rate
                </text>
            </svg>
        </div>
        """

    def _render_statistics(self, metrics: CacheMetrics) -> str:
        """Render detailed statistics grid."""
        # Calculate speedup
        if metrics.avg_cached_latency_ms > 0:
            speedup = metrics.avg_uncached_latency_ms / metrics.avg_cached_latency_ms
        else:
            speedup = 0.0

        # Format time saved
        if metrics.time_saved_ms > 1000:
            time_saved_display = f"{metrics.time_saved_ms / 1000:.1f}s"
        else:
            time_saved_display = f"{metrics.time_saved_ms:.0f}ms"

        return f"""
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #e5e7eb;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; font-size: 12px;">
                <div>
                    <div style="color: #9ca3af; margin-bottom: 4px;">Total Queries</div>
                    <div style="font-size: 18px; font-weight: 600; color: #374151; font-family: monospace;">
                        {metrics.total_queries}
                    </div>
                </div>
                <div>
                    <div style="color: #9ca3af; margin-bottom: 4px;">Cache Hits</div>
                    <div style="font-size: 18px; font-weight: 600; color: #10b981; font-family: monospace;">
                        {metrics.cache_hits}
                    </div>
                </div>
                <div>
                    <div style="color: #9ca3af; margin-bottom: 4px;">Cached Latency</div>
                    <div style="font-size: 18px; font-weight: 600; color: #374151; font-family: monospace;">
                        {metrics.avg_cached_latency_ms:.1f}ms
                    </div>
                </div>
                <div>
                    <div style="color: #9ca3af; margin-bottom: 4px;">Uncached Latency</div>
                    <div style="font-size: 18px; font-weight: 600; color: #374151; font-family: monospace;">
                        {metrics.avg_uncached_latency_ms:.1f}ms
                    </div>
                </div>
                <div>
                    <div style="color: #9ca3af; margin-bottom: 4px;">Time Saved</div>
                    <div style="font-size: 18px; font-weight: 600; color: #3b82f6; font-family: monospace;">
                        {time_saved_display}
                    </div>
                </div>
                <div>
                    <div style="color: #9ca3af; margin-bottom: 4px;">Speedup</div>
                    <div style="font-size: 18px; font-weight: 600; color: #3b82f6; font-family: monospace;">
                        {speedup:.1f}x
                    </div>
                </div>
            </div>
        </div>
        """

    def _render_recommendations(self, metrics: CacheMetrics) -> str:
        """Render performance recommendations based on metrics."""
        recommendations = []

        # Hit rate recommendations
        if metrics.hit_rate < 0.4:
            recommendations.append("Low hit rate - consider increasing cache size or TTL")
        elif metrics.hit_rate < 0.6:
            recommendations.append("Moderate hit rate - review cache eviction policy")

        # Latency recommendations
        if metrics.avg_cached_latency_ms > 50:
            recommendations.append("Cached queries slow - check cache storage performance")

        # Time saved recommendations
        speedup = metrics.avg_uncached_latency_ms / max(metrics.avg_cached_latency_ms, 1.0)
        if speedup < 2.0:
            recommendations.append("Low speedup - cache may not be worth the complexity")

        # Coverage recommendations
        if metrics.cache_misses > metrics.cache_hits:
            recommendations.append("More misses than hits - improve query pattern matching")

        if not recommendations:
            recommendations.append("Cache performing well - no immediate actions needed")

        recommendations_html = "\n".join([
            f'<li style="color: #374151; font-size: 12px; line-height: 1.6;">{rec}</li>'
            for rec in recommendations
        ])

        return f"""
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid #e5e7eb;">
            <div style="font-size: 12px; font-weight: 500; color: #374151; margin-bottom: 8px;">
                Recommendations
            </div>
            <ul style="margin: 0; padding-left: 20px;">
                {recommendations_html}
            </ul>
        </div>
        """

    def _effectiveness_colors(self, effectiveness: CacheEffectiveness) -> tuple:
        """Get color and background for effectiveness rating."""
        colors = {
            CacheEffectiveness.EXCELLENT: ("#065f46", "#d1fae5"),  # Dark green, light green
            CacheEffectiveness.GOOD: ("#047857", "#d1fae5"),       # Green
            CacheEffectiveness.FAIR: ("#d97706", "#fef3c7"),       # Amber
            CacheEffectiveness.POOR: ("#dc2626", "#fee2e2"),       # Red
            CacheEffectiveness.CRITICAL: ("#991b1b", "#fee2e2"),   # Dark red
        }
        return colors.get(effectiveness, ("#6b7280", "#f3f4f6"))


def calculate_cache_metrics(
    total_queries: int,
    cache_hits: int,
    cached_latencies_ms: List[float],
    uncached_latencies_ms: List[float]
) -> CacheMetrics:
    """
    Calculate cache performance metrics from raw data.

    Args:
        total_queries: Total number of queries processed
        cache_hits: Number of cache hits
        cached_latencies_ms: List of latencies for cached queries
        uncached_latencies_ms: List of latencies for cache misses

    Returns:
        CacheMetrics object with calculated statistics

    Raises:
        ValueError: If inputs are invalid

    Example:
        >>> metrics = calculate_cache_metrics(
        ...     total_queries=100,
        ...     cache_hits=75,
        ...     cached_latencies_ms=[15.0] * 75,
        ...     uncached_latencies_ms=[120.0] * 25
        ... )
        >>> assert metrics.hit_rate == 0.75
        >>> assert metrics.effectiveness == CacheEffectiveness.GOOD
    """
    # Validate inputs
    if total_queries <= 0:
        raise ValueError("total_queries must be positive")
    if cache_hits < 0 or cache_hits > total_queries:
        raise ValueError("cache_hits must be in [0, total_queries]")
    if len(cached_latencies_ms) != cache_hits:
        raise ValueError("cached_latencies_ms length must equal cache_hits")

    cache_misses = total_queries - cache_hits
    if len(uncached_latencies_ms) != cache_misses:
        raise ValueError("uncached_latencies_ms length must equal cache_misses")

    # Calculate hit rate
    hit_rate = cache_hits / total_queries

    # Calculate average latencies
    avg_cached_latency_ms = sum(cached_latencies_ms) / max(len(cached_latencies_ms), 1)
    avg_uncached_latency_ms = sum(uncached_latencies_ms) / max(len(uncached_latencies_ms), 1)

    # Estimate time saved
    time_saved_ms = estimate_time_saved(
        cache_hits,
        avg_cached_latency_ms,
        avg_uncached_latency_ms
    )

    # Determine effectiveness
    effectiveness = _determine_effectiveness(hit_rate, avg_cached_latency_ms, avg_uncached_latency_ms)

    return CacheMetrics(
        total_queries=total_queries,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        hit_rate=hit_rate,
        avg_cached_latency_ms=avg_cached_latency_ms,
        avg_uncached_latency_ms=avg_uncached_latency_ms,
        time_saved_ms=time_saved_ms,
        effectiveness=effectiveness
    )


def estimate_time_saved(
    cache_hits: int,
    avg_cached_latency_ms: float,
    avg_uncached_latency_ms: float
) -> float:
    """
    Estimate total time saved by cache.

    Formula:
        time_saved = cache_hits * (avg_uncached - avg_cached)

    Args:
        cache_hits: Number of cache hits
        avg_cached_latency_ms: Average cached query latency
        avg_uncached_latency_ms: Average uncached query latency

    Returns:
        Estimated time saved in milliseconds

    Example:
        >>> time_saved = estimate_time_saved(75, 15.0, 120.0)
        >>> assert time_saved == 7875.0  # 75 * (120 - 15)
    """
    return cache_hits * (avg_uncached_latency_ms - avg_cached_latency_ms)


def _determine_effectiveness(
    hit_rate: float,
    avg_cached_latency_ms: float,
    avg_uncached_latency_ms: float
) -> CacheEffectiveness:
    """Determine cache effectiveness rating."""
    # Calculate speedup
    speedup = avg_uncached_latency_ms / max(avg_cached_latency_ms, 1.0)

    # Excellent: High hit rate + good speedup
    if hit_rate >= 0.8 and speedup >= 4.0:
        return CacheEffectiveness.EXCELLENT

    # Good: Good hit rate + moderate speedup
    if hit_rate >= 0.6 and speedup >= 2.0:
        return CacheEffectiveness.GOOD

    # Fair: Moderate hit rate or speedup
    if hit_rate >= 0.4 or speedup >= 2.0:
        return CacheEffectiveness.FAIR

    # Poor: Low hit rate and speedup
    if hit_rate >= 0.2:
        return CacheEffectiveness.POOR

    # Critical: Very low hit rate
    return CacheEffectiveness.CRITICAL


def render_cache_gauge(
    hit_rate: float,
    total_queries: int,
    cache_hits: int,
    avg_cached_latency_ms: float = 15.0,
    avg_uncached_latency_ms: float = 120.0,
    title: str = "Cache Effectiveness",
    subtitle: Optional[str] = None,
    show_details: bool = True,
    show_recommendations: bool = True
) -> str:
    """
    Convenience function to render cache gauge from simple parameters.

    This is the primary programmatic API for automated tool calling.

    Args:
        hit_rate: Cache hit rate [0.0, 1.0]
        total_queries: Total number of queries
        cache_hits: Number of cache hits
        avg_cached_latency_ms: Average latency for cached queries
        avg_uncached_latency_ms: Average latency for cache misses
        title: Gauge title
        subtitle: Optional subtitle
        show_details: Show detailed statistics
        show_recommendations: Show performance recommendations

    Returns:
        HTML string with complete gauge visualization

    Raises:
        ValueError: If parameters are invalid

    Example (Simple):
        >>> html = render_cache_gauge(
        ...     hit_rate=0.75,
        ...     total_queries=100,
        ...     cache_hits=75
        ... )

    Example (Complete):
        >>> html = render_cache_gauge(
        ...     hit_rate=0.75,
        ...     total_queries=100,
        ...     cache_hits=75,
        ...     avg_cached_latency_ms=15.0,
        ...     avg_uncached_latency_ms=120.0,
        ...     title='Production Cache Performance',
        ...     subtitle='Last 24 hours',
        ...     show_details=True,
        ...     show_recommendations=True
        ... )

    Integration with HoloLoom:
        >>> # Track cache performance
        >>> total = 0
        >>> hits = 0
        >>> for query in queries:
        ...     spacetime = await orchestrator.weave(query)
        ...     total += 1
        ...     if spacetime.metadata.get('cache_hit'):
        ...         hits += 1
        >>>
        >>> html = render_cache_gauge(
        ...     hit_rate=hits / total,
        ...     total_queries=total,
        ...     cache_hits=hits
        ... )
    """
    # Validate inputs
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"hit_rate must be in [0.0, 1.0], got {hit_rate}")
    if total_queries <= 0:
        raise ValueError("total_queries must be positive")
    if cache_hits < 0 or cache_hits > total_queries:
        raise ValueError("cache_hits must be in [0, total_queries]")

    cache_misses = total_queries - cache_hits

    # Estimate time saved
    time_saved_ms = estimate_time_saved(
        cache_hits,
        avg_cached_latency_ms,
        avg_uncached_latency_ms
    )

    # Determine effectiveness
    effectiveness = _determine_effectiveness(
        hit_rate,
        avg_cached_latency_ms,
        avg_uncached_latency_ms
    )

    # Create metrics object
    metrics = CacheMetrics(
        total_queries=total_queries,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        hit_rate=hit_rate,
        avg_cached_latency_ms=avg_cached_latency_ms,
        avg_uncached_latency_ms=avg_uncached_latency_ms,
        time_saved_ms=time_saved_ms,
        effectiveness=effectiveness
    )

    # Render
    renderer = CacheGaugeRenderer(
        show_details=show_details,
        show_recommendations=show_recommendations
    )
    return renderer.render(metrics, title=title, subtitle=subtitle)
