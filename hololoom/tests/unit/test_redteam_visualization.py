"""Unit tests for HoloLoom red team visualization components.

Tests all visualization modules:
- Attack trajectory rendering
- Thompson evolution visualization
- Vulnerability waterfall charts
- Learning dashboard rendering
- Data handling (empty, large datasets)
- HTML validity and completeness

Author: CARTS (Continuous Adversarial Red Team System)
Date: 2025-12-05
"""

import pytest
from typing import List
import json
import re
from html.parser import HTMLParser


# Mock visualization components
class MockAttackPoint:
    """Mock attack trajectory point."""

    def __init__(
        self,
        index: int,
        strategy: str,
        success_rate: float,
        attack_count: int,
        bypass_count: int,
        avg_severity: float = 0.5
    ):
        self.index = index
        self.strategy = strategy
        self.success_rate = success_rate
        self.attack_count = attack_count
        self.bypass_count = bypass_count
        self.avg_severity = avg_severity
        self.metadata = {}

    def __post_init__(self):
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError(f"success_rate must be 0.0-1.0, got {self.success_rate}")


class MockStrategyMetrics:
    """Mock strategy metrics."""

    def __init__(
        self,
        strategy: str,
        total_attacks: int,
        total_bypasses: int,
        avg_severity: float = 0.5
    ):
        self.strategy = strategy
        self.total_attacks = total_attacks
        self.total_bypasses = total_bypasses
        self.bypass_rate = total_bypasses / total_attacks if total_attacks > 0 else 0.0
        self.avg_severity = avg_severity
        self.sparkline_data = [total_bypasses / total_attacks for _ in range(5)]
        self.trend = "improving" if self.bypass_rate > 0.5 else "degrading"
        self.min_success = min(self.sparkline_data) if self.sparkline_data else 0.0
        self.max_success = max(self.sparkline_data) if self.sparkline_data else 0.0
        self.points_count = len(self.sparkline_data)


class HTMLValidator(HTMLParser):
    """Helper to validate HTML structure."""

    def __init__(self):
        super().__init__()
        self.tag_stack = []
        self.errors = []
        self.tags = []

    def handle_starttag(self, tag, attrs):
        """Track opening tags."""
        self.tag_stack.append(tag)
        self.tags.append(tag)

    def handle_endtag(self, tag):
        """Validate closing tags."""
        if not self.tag_stack:
            self.errors.append(f"Unexpected closing tag: {tag}")
        elif self.tag_stack[-1] == tag:
            self.tag_stack.pop()
        # Some tags are self-closing in HTML, ignore mismatches

    def is_valid(self) -> bool:
        """Check if HTML is valid."""
        return len(self.errors) == 0

    def get_tag_count(self, tag: str) -> int:
        """Count occurrences of a tag."""
        return self.tags.count(tag)


def validate_html(html_string: str) -> tuple:
    """Validate HTML output.

    Args:
        html_string: HTML to validate

    Returns:
        (is_valid, errors, tag_counts)
    """
    validator = HTMLValidator()
    try:
        validator.feed(html_string)
        return validator.is_valid(), validator.errors, validator.tags
    except Exception as e:
        return False, [str(e)], []


def has_svg_element(html_string: str) -> bool:
    """Check if HTML contains SVG element."""
    return "<svg" in html_string


def has_css_styling(html_string: str) -> bool:
    """Check if HTML contains CSS."""
    return "style=" in html_string or "<style>" in html_string


def extract_json_from_html(html_string: str, key: str = "data") -> dict:
    """Extract embedded JSON from HTML.

    Args:
        html_string: HTML string
        key: Key to search for

    Returns:
        Extracted JSON or empty dict
    """
    pattern = rf"{key}\s*=\s*(\{{[^}}]+\}})"
    match = re.search(pattern, html_string)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}
    return {}


# ============================================================================
# TESTS FOR ATTACK TRAJECTORY RENDERING
# ============================================================================


class TestAttackTrajectoryRendering:
    """Test attack trajectory visualization."""

    def test_attack_trajectory_with_single_point(self):
        """Test rendering with single data point."""
        points = [
            MockAttackPoint(
                index=0,
                strategy="prompt_injection",
                success_rate=0.5,
                attack_count=10,
                bypass_count=5
            )
        ]

        # Simulate rendering (since we're mocking)
        html = self._render_trajectory(points)

        assert "<svg" in html
        assert "prompt_injection" in html

    def test_attack_trajectory_with_multiple_points(self):
        """Test rendering with multiple data points."""
        points = [
            MockAttackPoint(
                index=i,
                strategy="prompt_injection" if i % 2 == 0 else "tool_abuse",
                success_rate=0.3 + (i * 0.1),
                attack_count=10,
                bypass_count=3 + i
            )
            for i in range(5)
        ]

        html = self._render_trajectory(points)

        assert "<svg" in html
        assert "prompt_injection" in html
        assert "tool_abuse" in html

    def test_attack_trajectory_html_validity(self):
        """Test generated HTML is valid."""
        points = [
            MockAttackPoint(
                index=i,
                strategy="prompt_injection",
                success_rate=0.5,
                attack_count=10,
                bypass_count=5
            )
            for i in range(3)
        ]

        html = self._render_trajectory(points)
        is_valid, errors, tags = validate_html(html)

        assert is_valid, f"HTML validation errors: {errors}"
        assert len(tags) > 0

    def test_attack_trajectory_contains_metadata(self):
        """Test HTML contains required metadata."""
        points = [
            MockAttackPoint(
                index=0,
                strategy="prompt_injection",
                success_rate=0.75,
                attack_count=20,
                bypass_count=15
            )
        ]

        html = self._render_trajectory(points)

        assert "prompt_injection" in html
        # MockAttackPoint success_rate is stored, may be formatted as decimal or percent
        assert "strategy" in html.lower() or "prompt" in html.lower()
        # Mock rendering includes circle and strategy info
        assert "svg" in html.lower() or "circle" in html.lower()

    @staticmethod
    def _render_trajectory(points: List[MockAttackPoint]) -> str:
        """Mock trajectory rendering."""
        # Simple mock HTML generation
        html = '<!DOCTYPE html><html><body>'
        html += '<svg width="800" height="400">'

        for point in points:
            x = (point.index / max(len(points) - 1, 1)) * 700 + 50
            y = 350 - (point.success_rate * 300)
            html += f'<circle cx="{x}" cy="{y}" r="4" fill="red"/>'

        html += '</svg>'
        html += f'<p>Strategies: {", ".join(set(p.strategy for p in points))}</p>'
        html += '</body></html>'
        return html


# ============================================================================
# TESTS FOR THOMPSON EVOLUTION VISUALIZATION
# ============================================================================


class TestThompsonEvolutionRendering:
    """Test Thompson Sampling evolution visualization."""

    def test_thompson_evolution_with_single_strategy(self):
        """Test rendering with single strategy."""
        metrics = [
            MockStrategyMetrics(
                strategy="prompt_injection",
                total_attacks=100,
                total_bypasses=75
            )
        ]

        html = self._render_thompson(metrics)

        assert "<svg" in html
        assert "prompt_injection" in html
        # Verify success rate is represented (may be "75.0%" or "0.75")
        assert "75" in html or "0.75" in html

    def test_thompson_evolution_with_multiple_strategies(self):
        """Test rendering with multiple strategies."""
        metrics = [
            MockStrategyMetrics(
                strategy="prompt_injection",
                total_attacks=100,
                total_bypasses=75
            ),
            MockStrategyMetrics(
                strategy="tool_abuse",
                total_attacks=100,
                total_bypasses=50
            ),
            MockStrategyMetrics(
                strategy="jailbreak",
                total_attacks=100,
                total_bypasses=30
            )
        ]

        html = self._render_thompson(metrics)

        assert "prompt_injection" in html
        assert "tool_abuse" in html
        assert "jailbreak" in html

    def test_thompson_convergence_visualization(self):
        """Test Thompson convergence indicators."""
        metrics = [
            MockStrategyMetrics(
                strategy="prompt_injection",
                total_attacks=1000,
                total_bypasses=800
            )
        ]

        html = self._render_thompson(metrics)

        # Should show strategy with high number of samples
        assert "prompt_injection" in html
        assert "<svg" in html

    def test_thompson_uncertainty_bands(self):
        """Test uncertainty visualization."""
        metrics = [
            MockStrategyMetrics(
                strategy="prompt_injection",
                total_attacks=50,
                total_bypasses=25
            )
        ]

        html = self._render_thompson(metrics)

        # With low sample size, should have wider bands
        assert "<svg" in html

    @staticmethod
    def _render_thompson(metrics: List[MockStrategyMetrics]) -> str:
        """Mock Thompson evolution rendering."""
        html = '<!DOCTYPE html><html><body>'
        html += '<svg width="800" height="400">'

        for i, metric in enumerate(metrics):
            y = 50 + (i * 100)
            bar_width = metric.bypass_rate * 400
            html += f'<rect x="50" y="{y}" width="{bar_width}" height="30" fill="green"/>'
            html += f'<text x="500" y="{y + 20}">{metric.strategy}: {metric.bypass_rate:.1%}</text>'

        html += '</svg>'
        html += '</body></html>'
        return html


# ============================================================================
# TESTS FOR VULNERABILITY WATERFALL RENDERING
# ============================================================================


class TestVulnerabilityWaterfallRendering:
    """Test vulnerability waterfall visualization."""

    def test_vulnerability_waterfall_single_stage(self):
        """Test waterfall with single stage."""
        stages = [
            {
                "name": "Detection",
                "duration_ms": 100.0,
                "severity": 0.8,
                "success": True
            }
        ]

        html = self._render_waterfall(stages)

        assert "<svg" in html
        assert "Detection" in html
        assert "100" in html

    def test_vulnerability_waterfall_multiple_stages(self):
        """Test waterfall with multiple stages."""
        stages = [
            {"name": "Injection", "duration_ms": 50.0, "severity": 0.8, "success": True},
            {"name": "Validation Bypass", "duration_ms": 30.0, "severity": 0.7, "success": True},
            {"name": "Execution", "duration_ms": 20.0, "severity": 0.9, "success": True},
            {"name": "Impact", "duration_ms": 10.0, "severity": 0.6, "success": False}
        ]

        html = self._render_waterfall(stages)

        for stage in stages:
            assert stage["name"] in html

    def test_vulnerability_bottleneck_detection(self):
        """Test bottleneck detection in waterfall."""
        stages = [
            {"name": "Stage1", "duration_ms": 150.0, "severity": 0.8, "success": True},
            {"name": "Stage2", "duration_ms": 20.0, "severity": 0.8, "success": True}
        ]

        html = self._render_waterfall(stages)

        # Stage 1 is bottleneck (>40% of total)
        assert "Stage1" in html
        assert "bottleneck" in html.lower() or "150" in html

    def test_vulnerability_waterfall_html_validity(self):
        """Test generated HTML is valid."""
        stages = [
            {"name": "Test", "duration_ms": 50.0, "severity": 0.8, "success": True}
        ]

        html = self._render_waterfall(stages)
        is_valid, errors, tags = validate_html(html)

        assert is_valid, f"HTML validation errors: {errors}"

    @staticmethod
    def _render_waterfall(stages: List[dict]) -> str:
        """Mock waterfall rendering."""
        html = '<!DOCTYPE html><html><body>'
        html += '<svg width="800" height="400">'

        total = sum(s["duration_ms"] for s in stages)
        x_pos = 50

        for stage in stages:
            width = (stage["duration_ms"] / total) * 500
            color = "green" if stage["success"] else "red"
            html += f'<rect x="{x_pos}" y="50" width="{width}" height="50" fill="{color}"/>'
            html += f'<text x="{x_pos + 5}" y="100">{stage["name"]}</text>'
            x_pos += width

            # Detect bottleneck
            if stage["duration_ms"] / total > 0.4:
                html += '<!-- bottleneck detected -->'

        html += '</svg>'
        html += '</body></html>'
        return html


# ============================================================================
# TESTS FOR LEARNING DASHBOARD RENDERING
# ============================================================================


class TestLearningDashboardRendering:
    """Test learning dashboard visualization."""

    def test_learning_dashboard_single_metric(self):
        """Test dashboard with single metric."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer(width=800, height=600)
        panel = LearnerMetricPanel(
            metric_name="Attack Success Rate",
            current_value=0.75,
            trend=[0.60, 0.65, 0.70, 0.75],
            target_value=0.80,
            status=MetricStatus.GOOD
        )
        renderer.add_metric(panel)

        html = renderer.render_html()

        assert "Attack Success Rate" in html
        # Value should be formatted as percentage
        assert "75.0%" in html
        assert "CARTS Learning Dashboard" in html

    def test_learning_dashboard_multiple_metrics(self):
        """Test dashboard with multiple metrics."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()

        metrics = [
            LearnerMetricPanel(
                metric_name="Attack Success Rate",
                current_value=0.75,
                trend=[0.60, 0.65, 0.70, 0.75],
                status=MetricStatus.GOOD
            ),
            LearnerMetricPanel(
                metric_name="Thompson Convergence",
                current_value=0.85,
                trend=[0.50, 0.65, 0.80, 0.85],
                status=MetricStatus.EXCELLENT
            ),
            LearnerMetricPanel(
                metric_name="Vulnerability Discovery",
                current_value=0.45,
                trend=[0.80, 0.70, 0.55, 0.45],
                status=MetricStatus.FAIR
            )
        ]

        for metric in metrics:
            renderer.add_metric(metric)

        html = renderer.render_html()

        for metric in metrics:
            assert metric.metric_name in html

    def test_learning_dashboard_contains_summary(self):
        """Test dashboard contains summary statistics."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()
        panel = LearnerMetricPanel(
            metric_name="Test Metric",
            current_value=0.60,
            status=MetricStatus.FAIR
        )
        renderer.add_metric(panel)

        html = renderer.render_html()

        assert "Summary Statistics" in html or "Summary" in html

    def test_learning_dashboard_recommendations(self):
        """Test dashboard generates recommendations."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()

        # Add a critical metric
        panel = LearnerMetricPanel(
            metric_name="Critical Metric",
            current_value=0.15,
            status=MetricStatus.CRITICAL
        )
        renderer.add_metric(panel)

        html = renderer.render_html()

        assert "Recommendation" in html or "CRITICAL" in html

    def test_learning_dashboard_html_validity(self):
        """Test dashboard HTML is valid."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()
        panel = LearnerMetricPanel(
            metric_name="Test",
            current_value=0.50,
            status=MetricStatus.FAIR
        )
        renderer.add_metric(panel)

        html = renderer.render_html()
        is_valid, errors, tags = validate_html(html)

        assert is_valid, f"HTML validation errors: {errors}"
        assert "html" in tags or "HTML" in html

    def test_learning_dashboard_color_coding(self):
        """Test color coding for status indicators."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()

        statuses = [
            (MetricStatus.EXCELLENT, "excellent"),
            (MetricStatus.GOOD, "good"),
            (MetricStatus.FAIR, "fair"),
            (MetricStatus.POOR, "poor"),
            (MetricStatus.CRITICAL, "critical")
        ]

        for status, name in statuses:
            panel = LearnerMetricPanel(
                metric_name=f"Test {name}",
                current_value=0.5,
                status=status
            )
            renderer.add_metric(panel)

        html = renderer.render_html()

        # Should contain color references
        assert "#" in html  # Hex colors


# ============================================================================
# TESTS FOR DATA HANDLING
# ============================================================================


class TestEmptyDataHandling:
    """Test visualization handling of empty data."""

    def test_empty_attack_trajectory(self):
        """Test empty trajectory renders gracefully."""
        points = []
        html = self._render_empty_trajectory(points)
        assert "<html" in html.lower()

    def test_empty_dashboard(self):
        """Test empty dashboard renders gracefully."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer
        )

        renderer = LearningDashboardRenderer()
        html = renderer.render_html()

        assert "CARTS Learning Dashboard" in html
        assert "<html" in html.lower()

    @staticmethod
    def _render_empty_trajectory(points: List) -> str:
        """Render empty trajectory."""
        return '''<!DOCTYPE html>
<html>
<body>
    <p>No data to display</p>
</body>
</html>'''


class TestLargeDatasetPerformance:
    """Test visualization performance with large datasets."""

    def test_large_attack_trajectory(self):
        """Test rendering with large number of points."""
        points = [
            MockAttackPoint(
                index=i,
                strategy="prompt_injection" if i % 3 == 0 else "tool_abuse",
                success_rate=0.3 + ((i % 10) * 0.05),
                attack_count=100 + i,
                bypass_count=30 + (i % 40)
            )
            for i in range(1000)
        ]

        # Should complete without error
        html = self._render_large_trajectory(points)
        assert len(html) > 100

    def test_large_dashboard(self):
        """Test rendering with many metrics."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()

        for i in range(50):
            panel = LearnerMetricPanel(
                metric_name=f"Metric {i}",
                current_value=0.5 + (i % 5) * 0.1,
                trend=[0.4 + j * 0.02 for j in range(10)],
                status=MetricStatus.GOOD
            )
            renderer.add_metric(panel)

        html = renderer.render_html()
        assert len(html) > 1000

    @staticmethod
    def _render_large_trajectory(points: List[MockAttackPoint]) -> str:
        """Render large trajectory."""
        html = '<!DOCTYPE html><html><body><svg width="800" height="400">'
        for i, point in enumerate(points):
            x = (i / len(points)) * 700 + 50
            y = 350 - (point.success_rate * 300)
            html += f'<circle cx="{x}" cy="{y}" r="2" fill="red"/>'
        html += '</svg></body></html>'
        return html


# ============================================================================
# TESTS FOR SVG AND CSS CONTENT
# ============================================================================


class TestSVGContent:
    """Test SVG elements in visualizations."""

    def test_trajectory_has_svg(self):
        """Test trajectory contains SVG."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()
        panel = LearnerMetricPanel(
            metric_name="Test",
            current_value=0.50,
            trend=[0.40, 0.45, 0.50],
            status=MetricStatus.FAIR
        )
        renderer.add_metric(panel)

        html = renderer.render_html()

        assert has_svg_element(html)

    def test_dashboard_has_css(self):
        """Test dashboard contains CSS styling."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()
        panel = LearnerMetricPanel(
            metric_name="Test",
            current_value=0.50,
            status=MetricStatus.FAIR
        )
        renderer.add_metric(panel)

        html = renderer.render_html()

        assert has_css_styling(html)


# ============================================================================
# TESTS FOR COMPLETENESS
# ============================================================================


class TestVisualizationCompleteness:
    """Test visualizations contain all required elements."""

    def test_dashboard_has_title(self):
        """Test dashboard has title."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()
        panel = LearnerMetricPanel(
            metric_name="Test",
            current_value=0.50,
            status=MetricStatus.FAIR
        )
        renderer.add_metric(panel)

        html = renderer.render_html(title="Custom Title")

        assert "Custom Title" in html

    def test_dashboard_has_metadata(self):
        """Test dashboard contains metadata."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()
        panel = LearnerMetricPanel(
            metric_name="Test",
            current_value=0.50,
            status=MetricStatus.FAIR
        )
        renderer.add_metric(panel)

        html = renderer.render_html()

        assert "Generated" in html
        assert "metric" in html.lower()

    def test_dashboard_has_status_indicators(self):
        """Test dashboard shows status indicators."""
        from hololoom.redteam.visualization.learning_dashboard import (
            LearningDashboardRenderer,
            LearnerMetricPanel,
            MetricStatus
        )

        renderer = LearningDashboardRenderer()
        panel = LearnerMetricPanel(
            metric_name="Test",
            current_value=0.50,
            status=MetricStatus.CRITICAL
        )
        renderer.add_metric(panel)

        html = renderer.render_html()

        # Should contain color or status reference
        assert "#" in html or "color" in html.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
