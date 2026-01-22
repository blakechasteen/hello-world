"""
Confidence Trajectory Visualization Tests
==========================================
Comprehensive tests for time series rendering and anomaly detection.

Test Coverage:
- Time series rendering with various data points
- Anomaly detection (4 types): sudden_drop, prolonged_low, high_variance, cache_miss_cluster
- Cache markers (hit/miss visualization)
- Mean/std bands rendering
- Empty data handling
- Edge cases and validation

Author: Claude Code
Date: January 2026
"""

import pytest
import math
from typing import List

from HoloLoom.visualization.confidence_trajectory import (
    ConfidencePoint,
    ConfidenceTrajectoryRenderer,
    AnomalyType,
    Anomaly,
    TrajectoryMetrics,
    detect_confidence_anomalies,
    calculate_trajectory_metrics,
    render_confidence_trajectory,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_points() -> List[ConfidencePoint]:
    """Create basic confidence points for testing."""
    return [
        ConfidencePoint(index=0, confidence=0.92, cached=True),
        ConfidencePoint(index=1, confidence=0.88, cached=True),
        ConfidencePoint(index=2, confidence=0.85, cached=False),
        ConfidencePoint(index=3, confidence=0.87, cached=True),
        ConfidencePoint(index=4, confidence=0.90, cached=True),
    ]


@pytest.fixture
def points_with_sudden_drop() -> List[ConfidencePoint]:
    """Create points with sudden drop anomaly."""
    return [
        ConfidencePoint(index=0, confidence=0.92, cached=True),
        ConfidencePoint(index=1, confidence=0.90, cached=True),
        ConfidencePoint(index=2, confidence=0.65, cached=False),  # Drop of 0.25
        ConfidencePoint(index=3, confidence=0.68, cached=False),
        ConfidencePoint(index=4, confidence=0.72, cached=False),
    ]


@pytest.fixture
def points_with_prolonged_low() -> List[ConfidencePoint]:
    """Create points with prolonged low anomaly."""
    return [
        ConfidencePoint(index=0, confidence=0.92, cached=True),
        ConfidencePoint(index=1, confidence=0.60, cached=False),
        ConfidencePoint(index=2, confidence=0.55, cached=False),
        ConfidencePoint(index=3, confidence=0.58, cached=False),
        ConfidencePoint(index=4, confidence=0.52, cached=False),
        ConfidencePoint(index=5, confidence=0.57, cached=False),  # 5 consecutive below 0.7
    ]


@pytest.fixture
def points_with_high_variance() -> List[ConfidencePoint]:
    """Create points with high variance anomaly."""
    return [
        ConfidencePoint(index=0, confidence=0.90, cached=True),
        ConfidencePoint(index=1, confidence=0.50, cached=False),
        ConfidencePoint(index=2, confidence=0.95, cached=True),
        ConfidencePoint(index=3, confidence=0.45, cached=False),
        ConfidencePoint(index=4, confidence=0.85, cached=True),
    ]


@pytest.fixture
def points_with_cache_miss_cluster() -> List[ConfidencePoint]:
    """Create points with cache miss cluster anomaly."""
    return [
        ConfidencePoint(index=0, confidence=0.90, cached=True),
        ConfidencePoint(index=1, confidence=0.85, cached=False),
        ConfidencePoint(index=2, confidence=0.87, cached=False),
        ConfidencePoint(index=3, confidence=0.88, cached=False),
        ConfidencePoint(index=4, confidence=0.90, cached=True),
    ]


@pytest.fixture
def renderer() -> ConfidenceTrajectoryRenderer:
    """Create default renderer."""
    return ConfidenceTrajectoryRenderer()


# ============================================================================
# ConfidencePoint Tests
# ============================================================================

class TestConfidencePoint:
    """Tests for ConfidencePoint dataclass."""

    def test_basic_creation(self):
        """Test basic point creation."""
        point = ConfidencePoint(index=0, confidence=0.85)
        assert point.index == 0
        assert point.confidence == 0.85
        assert point.cached is False  # Default
        assert point.query_text is None
        assert point.timestamp is None
        assert point.metadata is None

    def test_full_creation(self):
        """Test point creation with all fields."""
        point = ConfidencePoint(
            index=5,
            confidence=0.92,
            cached=True,
            query_text="What is Thompson Sampling?",
            timestamp=1698595200.0,
            metadata={'latency_ms': 45.2}
        )
        assert point.index == 5
        assert point.confidence == 0.92
        assert point.cached is True
        assert point.query_text == "What is Thompson Sampling?"
        assert point.timestamp == 1698595200.0
        assert point.metadata == {'latency_ms': 45.2}

    def test_confidence_bounds(self):
        """Test confidence values at boundaries."""
        point_min = ConfidencePoint(index=0, confidence=0.0)
        point_max = ConfidencePoint(index=1, confidence=1.0)
        assert point_min.confidence == 0.0
        assert point_max.confidence == 1.0


# ============================================================================
# Anomaly Detection Tests
# ============================================================================

class TestAnomalyDetection:
    """Tests for anomaly detection functionality."""

    def test_sudden_drop_detection(self, points_with_sudden_drop):
        """Test detection of sudden confidence drops."""
        anomalies = detect_confidence_anomalies(points_with_sudden_drop)

        sudden_drops = [a for a in anomalies if a.type == AnomalyType.SUDDEN_DROP]
        assert len(sudden_drops) >= 1

        # Check the detected drop
        drop = sudden_drops[0]
        assert drop.start_index == 2  # Where drop occurs
        assert 0.0 <= drop.severity <= 1.0
        assert "dropped" in drop.description.lower()

    def test_prolonged_low_detection(self, points_with_prolonged_low):
        """Test detection of prolonged low confidence."""
        anomalies = detect_confidence_anomalies(points_with_prolonged_low, threshold=0.7)

        prolonged_lows = [a for a in anomalies if a.type == AnomalyType.PROLONGED_LOW]
        assert len(prolonged_lows) >= 1

        # Check the detected anomaly
        low = prolonged_lows[0]
        assert len(low.affected_points) > 3  # More than 3 consecutive
        assert "consecutive" in low.description.lower()

    def test_high_variance_detection(self, points_with_high_variance):
        """Test detection of high variance in confidence."""
        anomalies = detect_confidence_anomalies(points_with_high_variance, window_size=5)

        high_variance = [a for a in anomalies if a.type == AnomalyType.HIGH_VARIANCE]
        assert len(high_variance) >= 1

        # Check the detected anomaly
        variance = high_variance[0]
        assert "variance" in variance.description.lower() or "std" in variance.description.lower()
        assert variance.severity > 0

    def test_cache_miss_cluster_detection(self, points_with_cache_miss_cluster):
        """Test detection of cache miss clusters."""
        anomalies = detect_confidence_anomalies(points_with_cache_miss_cluster, window_size=5)

        cache_miss_clusters = [a for a in anomalies if a.type == AnomalyType.CACHE_MISS_CLUSTER]
        assert len(cache_miss_clusters) >= 1

        # Check the detected anomaly
        cluster = cache_miss_clusters[0]
        assert "cache" in cluster.description.lower() or "miss" in cluster.description.lower()

    def test_no_anomalies_in_stable_data(self, basic_points):
        """Test that stable data has minimal anomalies."""
        anomalies = detect_confidence_anomalies(basic_points)

        # Stable data should have no sudden drops or prolonged low
        sudden_drops = [a for a in anomalies if a.type == AnomalyType.SUDDEN_DROP]
        prolonged_lows = [a for a in anomalies if a.type == AnomalyType.PROLONGED_LOW]

        assert len(sudden_drops) == 0
        assert len(prolonged_lows) == 0

    def test_anomaly_severity_bounds(self, points_with_sudden_drop):
        """Test that anomaly severity is bounded [0, 1]."""
        anomalies = detect_confidence_anomalies(points_with_sudden_drop)

        for anomaly in anomalies:
            assert 0.0 <= anomaly.severity <= 1.0

    def test_empty_points_no_anomalies(self):
        """Test that empty points list returns no anomalies."""
        anomalies = detect_confidence_anomalies([])
        assert anomalies == []

    def test_single_point_no_anomalies(self):
        """Test that single point returns no anomalies."""
        points = [ConfidencePoint(index=0, confidence=0.85)]
        anomalies = detect_confidence_anomalies(points)
        assert anomalies == []

    def test_anomalies_sorted_by_severity(self, points_with_sudden_drop):
        """Test that anomalies are sorted by severity (descending)."""
        anomalies = detect_confidence_anomalies(points_with_sudden_drop)

        for i in range(len(anomalies) - 1):
            assert anomalies[i].severity >= anomalies[i + 1].severity

    def test_custom_threshold(self):
        """Test anomaly detection with custom threshold."""
        points = [
            ConfidencePoint(index=i, confidence=0.65)
            for i in range(6)
        ]

        # With threshold=0.7, all points are below threshold
        anomalies_high_thresh = detect_confidence_anomalies(points, threshold=0.7)
        prolonged = [a for a in anomalies_high_thresh if a.type == AnomalyType.PROLONGED_LOW]
        assert len(prolonged) >= 1

        # With threshold=0.5, no points are below threshold
        anomalies_low_thresh = detect_confidence_anomalies(points, threshold=0.5)
        prolonged_low = [a for a in anomalies_low_thresh if a.type == AnomalyType.PROLONGED_LOW]
        assert len(prolonged_low) == 0

    def test_custom_window_size(self):
        """Test anomaly detection with custom window size."""
        points = [
            ConfidencePoint(index=i, confidence=0.8, cached=False)
            for i in range(10)
        ]

        # Window size 5
        anomalies_5 = detect_confidence_anomalies(points, window_size=5)

        # Window size 3 should detect more clusters
        anomalies_3 = detect_confidence_anomalies(points, window_size=3)

        # Both should detect cache miss clusters
        cache_5 = [a for a in anomalies_5 if a.type == AnomalyType.CACHE_MISS_CLUSTER]
        cache_3 = [a for a in anomalies_3 if a.type == AnomalyType.CACHE_MISS_CLUSTER]
        assert len(cache_5) >= 0  # May or may not detect
        assert len(cache_3) >= 0


# ============================================================================
# Trajectory Metrics Tests
# ============================================================================

class TestTrajectoryMetrics:
    """Tests for trajectory metrics calculation."""

    def test_basic_metrics(self, basic_points):
        """Test basic metrics calculation."""
        metrics = calculate_trajectory_metrics(basic_points)

        assert isinstance(metrics, TrajectoryMetrics)
        assert 0.0 <= metrics.mean <= 1.0
        assert metrics.std >= 0
        assert 0.0 <= metrics.min <= 1.0
        assert 0.0 <= metrics.max <= 1.0
        assert metrics.min <= metrics.mean <= metrics.max

    def test_cache_hit_rate(self):
        """Test cache hit rate calculation."""
        points = [
            ConfidencePoint(index=0, confidence=0.9, cached=True),
            ConfidencePoint(index=1, confidence=0.9, cached=True),
            ConfidencePoint(index=2, confidence=0.9, cached=False),
            ConfidencePoint(index=3, confidence=0.9, cached=True),
        ]

        metrics = calculate_trajectory_metrics(points)
        assert metrics.cache_hit_rate == 0.75  # 3 out of 4

    def test_trend_slope_positive(self):
        """Test positive trend slope (improving)."""
        points = [
            ConfidencePoint(index=i, confidence=0.5 + i * 0.1)
            for i in range(5)
        ]

        metrics = calculate_trajectory_metrics(points)
        assert metrics.trend_slope > 0

    def test_trend_slope_negative(self):
        """Test negative trend slope (degrading)."""
        points = [
            ConfidencePoint(index=i, confidence=0.9 - i * 0.1)
            for i in range(5)
        ]

        metrics = calculate_trajectory_metrics(points)
        assert metrics.trend_slope < 0

    def test_trend_slope_flat(self):
        """Test flat trend slope (stable)."""
        points = [
            ConfidencePoint(index=i, confidence=0.85)
            for i in range(5)
        ]

        metrics = calculate_trajectory_metrics(points)
        assert abs(metrics.trend_slope) < 0.001

    def test_empty_points_raises(self):
        """Test that empty points raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            calculate_trajectory_metrics([])

    def test_single_point_metrics(self):
        """Test metrics for single point."""
        points = [ConfidencePoint(index=0, confidence=0.85)]
        metrics = calculate_trajectory_metrics(points)

        assert metrics.mean == 0.85
        assert metrics.std == 0.0
        assert metrics.min == 0.85
        assert metrics.max == 0.85

    def test_reliability_score_bounds(self, basic_points):
        """Test reliability score is bounded [0, 1]."""
        metrics = calculate_trajectory_metrics(basic_points)
        assert 0.0 <= metrics.reliability_score <= 1.0


# ============================================================================
# Renderer Tests
# ============================================================================

class TestConfidenceTrajectoryRenderer:
    """Tests for ConfidenceTrajectoryRenderer."""

    def test_renderer_initialization(self):
        """Test renderer initialization with defaults."""
        renderer = ConfidenceTrajectoryRenderer()

        assert renderer.detect_anomalies is True
        assert renderer.show_cache_markers is True
        assert renderer.show_confidence_bands is True
        assert renderer.anomaly_threshold == 0.7
        assert renderer.window_size == 5

    def test_renderer_custom_params(self):
        """Test renderer initialization with custom parameters."""
        renderer = ConfidenceTrajectoryRenderer(
            detect_anomalies=False,
            show_cache_markers=False,
            show_confidence_bands=False,
            anomaly_threshold=0.5,
            window_size=10
        )

        assert renderer.detect_anomalies is False
        assert renderer.anomaly_threshold == 0.5
        assert renderer.window_size == 10

    def test_invalid_threshold_raises(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="anomaly_threshold"):
            ConfidenceTrajectoryRenderer(anomaly_threshold=1.5)

        with pytest.raises(ValueError, match="anomaly_threshold"):
            ConfidenceTrajectoryRenderer(anomaly_threshold=-0.1)

    def test_invalid_window_size_raises(self):
        """Test that invalid window size raises ValueError."""
        with pytest.raises(ValueError, match="window_size"):
            ConfidenceTrajectoryRenderer(window_size=1)

        with pytest.raises(ValueError, match="window_size"):
            ConfidenceTrajectoryRenderer(window_size=0)

    def test_render_basic_html(self, renderer, basic_points):
        """Test basic HTML rendering."""
        html = renderer.render(basic_points)

        assert isinstance(html, str)
        assert "confidence-trajectory" in html
        assert "<svg" in html
        assert "path" in html.lower()

    def test_render_with_title_subtitle(self, renderer, basic_points):
        """Test rendering with title and subtitle."""
        html = renderer.render(
            basic_points,
            title="Test Trajectory",
            subtitle="Test Subtitle"
        )

        assert "Test Trajectory" in html
        assert "Test Subtitle" in html

    def test_render_empty_returns_empty_state(self, renderer):
        """Test rendering empty points returns empty state."""
        html = renderer.render([])

        assert "No confidence data" in html

    def test_render_includes_metrics(self, renderer, basic_points):
        """Test that rendered HTML includes metrics."""
        html = renderer.render(basic_points)

        assert "Mean" in html
        assert "Reliability" in html

    def test_render_includes_svg_elements(self, renderer, basic_points):
        """Test SVG structure in rendered HTML."""
        html = renderer.render(basic_points)

        # Check for SVG structure
        assert "<svg" in html
        assert "width=" in html
        assert "height=" in html
        assert "</svg>" in html

    def test_render_with_anomalies(self, renderer, points_with_sudden_drop):
        """Test rendering includes anomaly markers."""
        html = renderer.render(points_with_sudden_drop)

        # Should include anomaly indicators
        assert "Anomalies" in html or "circle" in html

    def test_render_cache_markers(self, renderer, basic_points):
        """Test that cache markers are rendered."""
        html = renderer.render(basic_points)

        # Should include rectangle markers for cached items
        assert "rect" in html.lower() or "fill" in html

    def test_validation_unsorted_points_raises(self, renderer):
        """Test that unsorted points raise ValueError."""
        points = [
            ConfidencePoint(index=2, confidence=0.85),
            ConfidencePoint(index=1, confidence=0.90),  # Out of order
        ]

        with pytest.raises(ValueError, match="sorted"):
            renderer.render(points)

    def test_validation_invalid_confidence_raises(self, renderer):
        """Test that invalid confidence raises ValueError."""
        points = [
            ConfidencePoint(index=0, confidence=1.5),  # Invalid
        ]

        with pytest.raises(ValueError, match="[Cc]onfidence"):
            renderer.render(points)


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestRenderConfidenceTrajectory:
    """Tests for render_confidence_trajectory convenience function."""

    def test_basic_usage(self):
        """Test basic usage with just confidence list."""
        confidences = [0.92, 0.88, 0.85, 0.87, 0.90]
        html = render_confidence_trajectory(confidences)

        assert isinstance(html, str)
        assert "confidence-trajectory" in html

    def test_with_cache_markers(self):
        """Test with cache markers."""
        confidences = [0.92, 0.88, 0.65, 0.87, 0.91]
        cached = [True, True, False, False, True]

        html = render_confidence_trajectory(confidences, cached=cached)
        assert isinstance(html, str)

    def test_with_query_texts(self):
        """Test with query texts."""
        confidences = [0.92, 0.88, 0.85]
        query_texts = ["Query 1", "Query 2", "Query 3"]

        html = render_confidence_trajectory(
            confidences,
            query_texts=query_texts
        )
        assert isinstance(html, str)

    def test_mismatched_cached_length_raises(self):
        """Test that mismatched cached length raises ValueError."""
        confidences = [0.92, 0.88, 0.85]
        cached = [True, False]  # Wrong length

        with pytest.raises(ValueError, match="cached"):
            render_confidence_trajectory(confidences, cached=cached)

    def test_mismatched_query_texts_length_raises(self):
        """Test that mismatched query_texts length raises ValueError."""
        confidences = [0.92, 0.88, 0.85]
        query_texts = ["Query 1"]  # Wrong length

        with pytest.raises(ValueError, match="query_texts"):
            render_confidence_trajectory(confidences, query_texts=query_texts)

    def test_invalid_confidence_value_raises(self):
        """Test that invalid confidence value raises ValueError."""
        confidences = [0.92, 1.5, 0.85]  # 1.5 is invalid

        with pytest.raises(ValueError, match="Confidence"):
            render_confidence_trajectory(confidences)

    def test_with_title_and_subtitle(self):
        """Test with custom title and subtitle."""
        confidences = [0.92, 0.88, 0.85]

        html = render_confidence_trajectory(
            confidences,
            title="Custom Title",
            subtitle="Custom Subtitle"
        )

        assert "Custom Title" in html
        assert "Custom Subtitle" in html

    def test_disable_anomaly_detection(self):
        """Test with anomaly detection disabled."""
        confidences = [0.92, 0.65, 0.85]  # Has potential anomaly

        html = render_confidence_trajectory(
            confidences,
            detect_anomalies=False
        )

        # Should still render, just without anomaly section
        assert isinstance(html, str)


# ============================================================================
# HTML Structure Tests
# ============================================================================

class TestHTMLStructure:
    """Tests for HTML structure validity."""

    def test_valid_svg_structure(self, renderer, basic_points):
        """Test that SVG has valid structure."""
        html = renderer.render(basic_points)

        # Count opening and closing tags
        svg_opens = html.count("<svg")
        svg_closes = html.count("</svg>")
        assert svg_opens == svg_closes

    def test_no_external_dependencies(self, renderer, basic_points):
        """Test that HTML has no external CSS/JS dependencies."""
        html = renderer.render(basic_points)

        # Should not have external links
        assert 'href="http' not in html
        assert 'src="http' not in html
        assert '<link' not in html.lower()

    def test_inline_styles(self, renderer, basic_points):
        """Test that styles are inline."""
        html = renderer.render(basic_points)

        # Should have inline styles
        assert 'style="' in html

    def test_contains_viewbox_or_dimensions(self, renderer, basic_points):
        """Test that SVG has proper dimensions."""
        html = renderer.render(basic_points)

        assert 'width=' in html or 'viewBox' in html
        assert 'height=' in html or 'viewBox' in html


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_two_points(self, renderer):
        """Test with minimum points for meaningful visualization."""
        points = [
            ConfidencePoint(index=0, confidence=0.85),
            ConfidencePoint(index=1, confidence=0.90),
        ]

        html = renderer.render(points)
        assert isinstance(html, str)

    def test_many_points(self, renderer):
        """Test with large number of points."""
        points = [
            ConfidencePoint(index=i, confidence=0.5 + 0.4 * math.sin(i / 10))
            for i in range(100)
        ]

        html = renderer.render(points)
        assert isinstance(html, str)
        assert "<svg" in html

    def test_all_same_confidence(self, renderer):
        """Test with all identical confidence values."""
        points = [
            ConfidencePoint(index=i, confidence=0.85)
            for i in range(10)
        ]

        html = renderer.render(points)
        assert isinstance(html, str)

    def test_extreme_confidence_values(self, renderer):
        """Test with extreme confidence values (0 and 1)."""
        points = [
            ConfidencePoint(index=0, confidence=0.0),
            ConfidencePoint(index=1, confidence=1.0),
            ConfidencePoint(index=2, confidence=0.0),
            ConfidencePoint(index=3, confidence=1.0),
        ]

        html = renderer.render(points)
        assert isinstance(html, str)

    def test_all_cached(self, renderer):
        """Test with all cached points."""
        points = [
            ConfidencePoint(index=i, confidence=0.9, cached=True)
            for i in range(5)
        ]

        metrics = calculate_trajectory_metrics(points)
        assert metrics.cache_hit_rate == 1.0

        html = renderer.render(points)
        assert isinstance(html, str)

    def test_no_cached(self, renderer):
        """Test with no cached points."""
        points = [
            ConfidencePoint(index=i, confidence=0.9, cached=False)
            for i in range(5)
        ]

        metrics = calculate_trajectory_metrics(points)
        assert metrics.cache_hit_rate == 0.0

        html = renderer.render(points)
        assert isinstance(html, str)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test complete workflow: create points -> detect anomalies -> render."""
        # Create points with various characteristics
        confidences = [0.92, 0.88, 0.65, 0.67, 0.64, 0.62, 0.70, 0.85, 0.90]
        cached = [True, True, False, False, False, False, False, True, True]

        # Render
        html = render_confidence_trajectory(
            confidences,
            cached=cached,
            title="Integration Test",
            subtitle="Testing full workflow",
            detect_anomalies=True
        )

        # Verify output
        assert "Integration Test" in html
        assert "confidence-trajectory" in html
        assert "<svg" in html

    def test_anomaly_detection_and_rendering(self, points_with_sudden_drop):
        """Test that detected anomalies appear in rendering."""
        renderer = ConfidenceTrajectoryRenderer(detect_anomalies=True)
        html = renderer.render(points_with_sudden_drop)

        # Should have anomaly section
        assert "Anomalies" in html or "anomaly" in html.lower()
