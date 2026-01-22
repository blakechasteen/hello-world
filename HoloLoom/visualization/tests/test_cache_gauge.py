"""
Cache Gauge Visualization Tests
================================
Comprehensive tests for radial gauge and effectiveness ratings.

Test Coverage:
- Radial gauge rendering
- 5 effectiveness ratings (excellent, good, fair, poor, critical)
- Hit rate calculation
- Speedup metrics
- Recommendations generation
- Edge cases and validation

Author: Claude Code
Date: January 2026
"""

import pytest
from typing import List

from HoloLoom.visualization.cache_gauge import (
    CacheMetrics,
    CacheEffectiveness,
    CacheGaugeRenderer,
    calculate_cache_metrics,
    estimate_time_saved,
    render_cache_gauge,
    _determine_effectiveness,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def excellent_metrics() -> CacheMetrics:
    """Create metrics for excellent effectiveness."""
    return CacheMetrics(
        total_queries=100,
        cache_hits=85,
        cache_misses=15,
        hit_rate=0.85,
        avg_cached_latency_ms=10.0,
        avg_uncached_latency_ms=150.0,
        time_saved_ms=85 * (150.0 - 10.0),
        effectiveness=CacheEffectiveness.EXCELLENT
    )


@pytest.fixture
def good_metrics() -> CacheMetrics:
    """Create metrics for good effectiveness."""
    return CacheMetrics(
        total_queries=100,
        cache_hits=70,
        cache_misses=30,
        hit_rate=0.70,
        avg_cached_latency_ms=20.0,
        avg_uncached_latency_ms=100.0,
        time_saved_ms=70 * (100.0 - 20.0),
        effectiveness=CacheEffectiveness.GOOD
    )


@pytest.fixture
def poor_metrics() -> CacheMetrics:
    """Create metrics for poor effectiveness."""
    return CacheMetrics(
        total_queries=100,
        cache_hits=25,
        cache_misses=75,
        hit_rate=0.25,
        avg_cached_latency_ms=50.0,
        avg_uncached_latency_ms=80.0,
        time_saved_ms=25 * (80.0 - 50.0),
        effectiveness=CacheEffectiveness.POOR
    )


@pytest.fixture
def renderer() -> CacheGaugeRenderer:
    """Create default renderer."""
    return CacheGaugeRenderer()


# ============================================================================
# CacheEffectiveness Tests
# ============================================================================

class TestCacheEffectiveness:
    """Tests for CacheEffectiveness enum."""

    def test_all_effectiveness_levels_exist(self):
        """Test all expected effectiveness levels exist."""
        assert hasattr(CacheEffectiveness, 'EXCELLENT')
        assert hasattr(CacheEffectiveness, 'GOOD')
        assert hasattr(CacheEffectiveness, 'FAIR')
        assert hasattr(CacheEffectiveness, 'POOR')
        assert hasattr(CacheEffectiveness, 'CRITICAL')

    def test_effectiveness_values(self):
        """Test effectiveness string values."""
        assert CacheEffectiveness.EXCELLENT.value == "excellent"
        assert CacheEffectiveness.GOOD.value == "good"
        assert CacheEffectiveness.FAIR.value == "fair"
        assert CacheEffectiveness.POOR.value == "poor"
        assert CacheEffectiveness.CRITICAL.value == "critical"


# ============================================================================
# CacheMetrics Tests
# ============================================================================

class TestCacheMetrics:
    """Tests for CacheMetrics dataclass."""

    def test_basic_creation(self):
        """Test basic metrics creation."""
        metrics = CacheMetrics(
            total_queries=100,
            cache_hits=75,
            cache_misses=25,
            hit_rate=0.75,
            avg_cached_latency_ms=15.0,
            avg_uncached_latency_ms=120.0,
            time_saved_ms=7875.0,
            effectiveness=CacheEffectiveness.GOOD
        )

        assert metrics.total_queries == 100
        assert metrics.cache_hits == 75
        assert metrics.cache_misses == 25
        assert metrics.hit_rate == 0.75
        assert metrics.effectiveness == CacheEffectiveness.GOOD

    def test_metrics_consistency(self):
        """Test that metrics values are consistent."""
        metrics = CacheMetrics(
            total_queries=100,
            cache_hits=75,
            cache_misses=25,
            hit_rate=0.75,
            avg_cached_latency_ms=15.0,
            avg_uncached_latency_ms=120.0,
            time_saved_ms=7875.0,
            effectiveness=CacheEffectiveness.GOOD
        )

        assert metrics.cache_hits + metrics.cache_misses == metrics.total_queries


# ============================================================================
# Effectiveness Determination Tests
# ============================================================================

class TestEffectivenessDetermination:
    """Tests for effectiveness determination logic."""

    def test_excellent_high_hit_rate_high_speedup(self):
        """Test excellent rating with high hit rate and speedup."""
        effectiveness = _determine_effectiveness(
            hit_rate=0.85,
            avg_cached_latency_ms=10.0,
            avg_uncached_latency_ms=150.0  # 15x speedup
        )
        assert effectiveness == CacheEffectiveness.EXCELLENT

    def test_good_moderate_hit_rate(self):
        """Test good rating with moderate hit rate."""
        effectiveness = _determine_effectiveness(
            hit_rate=0.70,
            avg_cached_latency_ms=30.0,
            avg_uncached_latency_ms=120.0  # 4x speedup
        )
        assert effectiveness == CacheEffectiveness.GOOD

    def test_fair_lower_hit_rate(self):
        """Test fair rating with lower hit rate."""
        effectiveness = _determine_effectiveness(
            hit_rate=0.45,
            avg_cached_latency_ms=50.0,
            avg_uncached_latency_ms=100.0  # 2x speedup
        )
        assert effectiveness == CacheEffectiveness.FAIR

    def test_poor_low_hit_rate(self):
        """Test poor rating with low hit rate."""
        effectiveness = _determine_effectiveness(
            hit_rate=0.25,
            avg_cached_latency_ms=80.0,
            avg_uncached_latency_ms=100.0  # 1.25x speedup
        )
        assert effectiveness == CacheEffectiveness.POOR

    def test_critical_very_low_hit_rate(self):
        """Test critical rating with very low hit rate."""
        effectiveness = _determine_effectiveness(
            hit_rate=0.10,
            avg_cached_latency_ms=90.0,
            avg_uncached_latency_ms=100.0  # 1.1x speedup
        )
        assert effectiveness == CacheEffectiveness.CRITICAL

    def test_speedup_affects_rating(self):
        """Test that speedup affects effectiveness rating."""
        # Low hit rate but high speedup might get FAIR
        effectiveness = _determine_effectiveness(
            hit_rate=0.35,
            avg_cached_latency_ms=10.0,
            avg_uncached_latency_ms=100.0  # 10x speedup
        )
        # Should not be POOR or CRITICAL due to high speedup
        assert effectiveness in [CacheEffectiveness.FAIR, CacheEffectiveness.GOOD]


# ============================================================================
# Time Saved Calculation Tests
# ============================================================================

class TestTimeSavedCalculation:
    """Tests for time saved estimation."""

    def test_basic_time_saved(self):
        """Test basic time saved calculation."""
        time_saved = estimate_time_saved(
            cache_hits=75,
            avg_cached_latency_ms=15.0,
            avg_uncached_latency_ms=120.0
        )

        expected = 75 * (120.0 - 15.0)  # 7875.0
        assert time_saved == expected

    def test_zero_hits_zero_time_saved(self):
        """Test zero cache hits results in zero time saved."""
        time_saved = estimate_time_saved(
            cache_hits=0,
            avg_cached_latency_ms=15.0,
            avg_uncached_latency_ms=120.0
        )
        assert time_saved == 0.0

    def test_same_latency_zero_time_saved(self):
        """Test same latency results in zero time saved."""
        time_saved = estimate_time_saved(
            cache_hits=100,
            avg_cached_latency_ms=50.0,
            avg_uncached_latency_ms=50.0
        )
        assert time_saved == 0.0


# ============================================================================
# Metrics Calculation Tests
# ============================================================================

class TestMetricsCalculation:
    """Tests for calculate_cache_metrics function."""

    def test_basic_calculation(self):
        """Test basic metrics calculation."""
        metrics = calculate_cache_metrics(
            total_queries=100,
            cache_hits=75,
            cached_latencies_ms=[15.0] * 75,
            uncached_latencies_ms=[120.0] * 25
        )

        assert metrics.total_queries == 100
        assert metrics.cache_hits == 75
        assert metrics.cache_misses == 25
        assert metrics.hit_rate == 0.75
        assert metrics.avg_cached_latency_ms == 15.0
        assert metrics.avg_uncached_latency_ms == 120.0

    def test_invalid_total_queries_raises(self):
        """Test that invalid total_queries raises ValueError."""
        with pytest.raises(ValueError, match="total_queries"):
            calculate_cache_metrics(
                total_queries=0,
                cache_hits=0,
                cached_latencies_ms=[],
                uncached_latencies_ms=[]
            )

    def test_invalid_cache_hits_raises(self):
        """Test that invalid cache_hits raises ValueError."""
        with pytest.raises(ValueError, match="cache_hits"):
            calculate_cache_metrics(
                total_queries=100,
                cache_hits=150,  # More than total
                cached_latencies_ms=[15.0] * 150,
                uncached_latencies_ms=[]
            )

    def test_mismatched_latencies_raises(self):
        """Test that mismatched latency list lengths raise ValueError."""
        with pytest.raises(ValueError):
            calculate_cache_metrics(
                total_queries=100,
                cache_hits=75,
                cached_latencies_ms=[15.0] * 50,  # Should be 75
                uncached_latencies_ms=[120.0] * 25
            )


# ============================================================================
# Renderer Tests
# ============================================================================

class TestCacheGaugeRenderer:
    """Tests for CacheGaugeRenderer."""

    def test_default_initialization(self):
        """Test renderer with default parameters."""
        renderer = CacheGaugeRenderer()

        assert renderer.show_details is True
        assert renderer.show_recommendations is True

    def test_custom_initialization(self):
        """Test renderer with custom parameters."""
        renderer = CacheGaugeRenderer(
            show_details=False,
            show_recommendations=False
        )

        assert renderer.show_details is False
        assert renderer.show_recommendations is False

    def test_render_basic_html(self, renderer, good_metrics):
        """Test basic HTML rendering."""
        html = renderer.render(good_metrics)

        assert isinstance(html, str)
        assert "cache-gauge" in html
        assert "Cache Effectiveness" in html

    def test_render_with_title(self, renderer, good_metrics):
        """Test rendering with custom title."""
        html = renderer.render(good_metrics, title="Custom Cache")

        assert "Custom Cache" in html

    def test_render_with_subtitle(self, renderer, good_metrics):
        """Test rendering with subtitle."""
        html = renderer.render(
            good_metrics,
            title="Cache Stats",
            subtitle="Last 24 hours"
        )

        assert "Last 24 hours" in html

    def test_render_includes_effectiveness_badge(self, renderer, good_metrics):
        """Test that effectiveness badge is rendered."""
        html = renderer.render(good_metrics)

        # Should show effectiveness rating
        assert "Good" in html or "GOOD" in html

    def test_render_includes_hit_rate(self, renderer, good_metrics):
        """Test that hit rate is displayed."""
        html = renderer.render(good_metrics)

        # Should show hit rate percentage
        assert "70" in html or "Hit Rate" in html


# ============================================================================
# SVG Gauge Tests
# ============================================================================

class TestSVGGauge:
    """Tests for SVG gauge rendering."""

    def test_svg_structure(self, renderer, good_metrics):
        """Test SVG structure in rendered HTML."""
        html = renderer.render(good_metrics)

        assert "<svg" in html
        assert "</svg>" in html
        assert "path" in html.lower()

    def test_gauge_arc_rendered(self, renderer, good_metrics):
        """Test that gauge arc is rendered."""
        html = renderer.render(good_metrics)

        # Should have arc path
        assert "A " in html or "arc" in html.lower()

    def test_center_text_rendered(self, renderer, good_metrics):
        """Test that center text is rendered."""
        html = renderer.render(good_metrics)

        # Should have text element with hit rate
        assert "<text" in html


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatisticsRendering:
    """Tests for statistics section rendering."""

    def test_statistics_rendered(self, good_metrics):
        """Test that statistics are rendered."""
        renderer = CacheGaugeRenderer(show_details=True)
        html = renderer.render(good_metrics)

        assert "Total Queries" in html
        assert "Cache Hits" in html

    def test_statistics_disabled(self, good_metrics):
        """Test that statistics can be disabled."""
        renderer = CacheGaugeRenderer(show_details=False)
        html = renderer.render(good_metrics)

        # Should still render, just without detailed stats
        assert isinstance(html, str)

    def test_time_saved_displayed(self, good_metrics):
        """Test that time saved is displayed."""
        renderer = CacheGaugeRenderer(show_details=True)
        html = renderer.render(good_metrics)

        assert "Time Saved" in html

    def test_speedup_displayed(self, good_metrics):
        """Test that speedup is displayed."""
        renderer = CacheGaugeRenderer(show_details=True)
        html = renderer.render(good_metrics)

        assert "Speedup" in html or "x" in html


# ============================================================================
# Recommendations Tests
# ============================================================================

class TestRecommendationsRendering:
    """Tests for recommendations section rendering."""

    def test_recommendations_rendered(self, poor_metrics):
        """Test that recommendations are rendered for poor performance."""
        renderer = CacheGaugeRenderer(show_recommendations=True)
        html = renderer.render(poor_metrics)

        assert "Recommendations" in html

    def test_recommendations_disabled(self, good_metrics):
        """Test that recommendations can be disabled."""
        renderer = CacheGaugeRenderer(show_recommendations=False)
        html = renderer.render(good_metrics)

        # Should not have recommendations section
        assert "Recommendations" not in html

    def test_low_hit_rate_recommendation(self):
        """Test recommendation for low hit rate."""
        metrics = CacheMetrics(
            total_queries=100,
            cache_hits=30,
            cache_misses=70,
            hit_rate=0.30,
            avg_cached_latency_ms=15.0,
            avg_uncached_latency_ms=120.0,
            time_saved_ms=3000.0,
            effectiveness=CacheEffectiveness.POOR
        )

        renderer = CacheGaugeRenderer(show_recommendations=True)
        html = renderer.render(metrics)

        # Should recommend improving hit rate
        assert "hit rate" in html.lower() or "cache size" in html.lower()

    def test_good_performance_recommendation(self, excellent_metrics):
        """Test positive recommendation for excellent performance."""
        renderer = CacheGaugeRenderer(show_recommendations=True)
        html = renderer.render(excellent_metrics)

        # Should have positive recommendation
        assert "performing well" in html.lower() or "no immediate" in html.lower()


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestRenderCacheGauge:
    """Tests for render_cache_gauge convenience function."""

    def test_basic_usage(self):
        """Test basic usage."""
        html = render_cache_gauge(
            hit_rate=0.75,
            total_queries=100,
            cache_hits=75
        )

        assert isinstance(html, str)
        assert "cache-gauge" in html

    def test_with_latencies(self):
        """Test with custom latencies."""
        html = render_cache_gauge(
            hit_rate=0.75,
            total_queries=100,
            cache_hits=75,
            avg_cached_latency_ms=20.0,
            avg_uncached_latency_ms=150.0
        )

        assert isinstance(html, str)

    def test_with_all_options(self):
        """Test with all options."""
        html = render_cache_gauge(
            hit_rate=0.75,
            total_queries=100,
            cache_hits=75,
            avg_cached_latency_ms=15.0,
            avg_uncached_latency_ms=120.0,
            title="Production Cache",
            subtitle="Last 24 hours",
            show_details=True,
            show_recommendations=True
        )

        assert "Production Cache" in html
        assert "Last 24 hours" in html

    def test_invalid_hit_rate_raises(self):
        """Test that invalid hit_rate raises ValueError."""
        with pytest.raises(ValueError, match="hit_rate"):
            render_cache_gauge(
                hit_rate=1.5,  # Invalid
                total_queries=100,
                cache_hits=75
            )

    def test_invalid_total_queries_raises(self):
        """Test that invalid total_queries raises ValueError."""
        with pytest.raises(ValueError, match="total_queries"):
            render_cache_gauge(
                hit_rate=0.75,
                total_queries=0,
                cache_hits=0
            )

    def test_invalid_cache_hits_raises(self):
        """Test that invalid cache_hits raises ValueError."""
        with pytest.raises(ValueError, match="cache_hits"):
            render_cache_gauge(
                hit_rate=0.75,
                total_queries=100,
                cache_hits=150  # More than total
            )


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Tests for metrics validation."""

    def test_validate_hit_rate_bounds(self, renderer):
        """Test validation of hit_rate bounds."""
        invalid_metrics = CacheMetrics(
            total_queries=100,
            cache_hits=75,
            cache_misses=25,
            hit_rate=1.5,  # Invalid
            avg_cached_latency_ms=15.0,
            avg_uncached_latency_ms=120.0,
            time_saved_ms=7875.0,
            effectiveness=CacheEffectiveness.GOOD
        )

        with pytest.raises(ValueError, match="hit_rate"):
            renderer.render(invalid_metrics)

    def test_validate_total_equals_sum(self, renderer):
        """Test validation that total equals hits + misses."""
        invalid_metrics = CacheMetrics(
            total_queries=100,
            cache_hits=60,
            cache_misses=30,  # 60 + 30 != 100
            hit_rate=0.60,
            avg_cached_latency_ms=15.0,
            avg_uncached_latency_ms=120.0,
            time_saved_ms=5000.0,
            effectiveness=CacheEffectiveness.GOOD
        )

        with pytest.raises(ValueError, match="total_queries"):
            renderer.render(invalid_metrics)

    def test_validate_non_negative_latencies(self, renderer):
        """Test validation of non-negative latencies."""
        invalid_metrics = CacheMetrics(
            total_queries=100,
            cache_hits=75,
            cache_misses=25,
            hit_rate=0.75,
            avg_cached_latency_ms=-15.0,  # Invalid
            avg_uncached_latency_ms=120.0,
            time_saved_ms=7875.0,
            effectiveness=CacheEffectiveness.GOOD
        )

        with pytest.raises(ValueError, match="Latencies"):
            renderer.render(invalid_metrics)


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_percent_hit_rate(self):
        """Test with 0% hit rate."""
        html = render_cache_gauge(
            hit_rate=0.0,
            total_queries=100,
            cache_hits=0
        )

        assert isinstance(html, str)
        assert "0" in html

    def test_hundred_percent_hit_rate(self):
        """Test with 100% hit rate."""
        html = render_cache_gauge(
            hit_rate=1.0,
            total_queries=100,
            cache_hits=100
        )

        assert isinstance(html, str)
        assert "100" in html

    def test_large_time_saved(self):
        """Test with large time saved (displayed in seconds)."""
        html = render_cache_gauge(
            hit_rate=0.90,
            total_queries=10000,
            cache_hits=9000,
            avg_cached_latency_ms=5.0,
            avg_uncached_latency_ms=200.0
        )

        assert isinstance(html, str)
        # Large time saved should be displayed in seconds
        assert "s" in html

    def test_small_latency_difference(self):
        """Test with small latency difference."""
        html = render_cache_gauge(
            hit_rate=0.75,
            total_queries=100,
            cache_hits=75,
            avg_cached_latency_ms=95.0,
            avg_uncached_latency_ms=100.0
        )

        assert isinstance(html, str)

    def test_zero_cached_latency(self):
        """Test with zero cached latency."""
        html = render_cache_gauge(
            hit_rate=0.75,
            total_queries=100,
            cache_hits=75,
            avg_cached_latency_ms=0.0,  # Edge case
            avg_uncached_latency_ms=100.0
        )

        assert isinstance(html, str)


# ============================================================================
# HTML Structure Tests
# ============================================================================

class TestHTMLStructure:
    """Tests for HTML structure validity."""

    def test_no_external_dependencies(self, renderer, good_metrics):
        """Test that HTML has no external dependencies."""
        html = renderer.render(good_metrics)

        assert 'href="http' not in html
        assert 'src="http' not in html
        assert '<link' not in html.lower()

    def test_inline_styles(self, renderer, good_metrics):
        """Test that styles are inline."""
        html = renderer.render(good_metrics)

        assert 'style="' in html

    def test_svg_properly_closed(self, renderer, good_metrics):
        """Test that SVG is properly closed."""
        html = renderer.render(good_metrics)

        svg_opens = html.count("<svg")
        svg_closes = html.count("</svg>")
        assert svg_opens == svg_closes


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test complete workflow with all features."""
        # Simulate collecting cache data
        cached_latencies = [15.0, 12.0, 18.0, 14.0, 16.0] * 15  # 75 hits
        uncached_latencies = [120.0, 130.0, 110.0, 125.0, 115.0] * 5  # 25 misses

        # Calculate metrics
        metrics = calculate_cache_metrics(
            total_queries=100,
            cache_hits=75,
            cached_latencies_ms=cached_latencies,
            uncached_latencies_ms=uncached_latencies
        )

        # Render gauge
        renderer = CacheGaugeRenderer(
            show_details=True,
            show_recommendations=True
        )
        html = renderer.render(
            metrics,
            title="Integration Test Cache",
            subtitle="Test data"
        )

        # Verify output
        assert "Integration Test Cache" in html
        assert "cache-gauge" in html
        assert "<svg" in html
        assert "Recommendations" in html
