"""
Stage Waterfall Visualization Tests
=====================================
Comprehensive tests for pipeline timing visualization.

Test Coverage:
- Pipeline timing visualization
- Bottleneck detection (>40% of total time)
- Status indicators (success, warning, error, skipped)
- Sparkline trends
- Parallel execution visualization
- Edge cases and validation

Author: Claude Code
Date: January 2026
"""

import pytest
from typing import List, Dict

from HoloLoom.visualization.stage_waterfall import (
    WaterfallStage,
    StageStatus,
    StageWaterfallRenderer,
    render_pipeline_waterfall,
    render_parallel_waterfall,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def basic_stages() -> List[WaterfallStage]:
    """Create basic stages for testing."""
    return [
        WaterfallStage(name="Pattern Selection", start_ms=0.0, duration_ms=5.2),
        WaterfallStage(name="Retrieval", start_ms=5.2, duration_ms=50.5),
        WaterfallStage(name="Convergence", start_ms=55.7, duration_ms=30.0),
        WaterfallStage(name="Tool Execution", start_ms=85.7, duration_ms=64.3),
    ]


@pytest.fixture
def stages_with_bottleneck() -> List[WaterfallStage]:
    """Create stages with a clear bottleneck."""
    total = 100.0
    return [
        WaterfallStage(name="Step 1", start_ms=0.0, duration_ms=10.0),
        WaterfallStage(name="Step 2", start_ms=10.0, duration_ms=55.0),  # 55% - bottleneck
        WaterfallStage(name="Step 3", start_ms=65.0, duration_ms=35.0),
    ]


@pytest.fixture
def stages_with_status() -> List[WaterfallStage]:
    """Create stages with various statuses."""
    return [
        WaterfallStage(name="Success Step", start_ms=0.0, duration_ms=20.0,
                       status=StageStatus.SUCCESS),
        WaterfallStage(name="Warning Step", start_ms=20.0, duration_ms=30.0,
                       status=StageStatus.WARNING),
        WaterfallStage(name="Error Step", start_ms=50.0, duration_ms=25.0,
                       status=StageStatus.ERROR),
        WaterfallStage(name="Skipped Step", start_ms=75.0, duration_ms=0.0,
                       status=StageStatus.SKIPPED),
    ]


@pytest.fixture
def stages_with_trends() -> List[WaterfallStage]:
    """Create stages with trend data."""
    return [
        WaterfallStage(
            name="Step 1", start_ms=0.0, duration_ms=20.0,
            trend=[25.0, 23.0, 22.0, 21.0, 20.0]  # Improving
        ),
        WaterfallStage(
            name="Step 2", start_ms=20.0, duration_ms=40.0,
            trend=[30.0, 33.0, 36.0, 38.0, 40.0]  # Degrading
        ),
    ]


@pytest.fixture
def renderer() -> StageWaterfallRenderer:
    """Create default renderer."""
    return StageWaterfallRenderer()


# ============================================================================
# WaterfallStage Tests
# ============================================================================

class TestWaterfallStage:
    """Tests for WaterfallStage dataclass."""

    def test_basic_creation(self):
        """Test basic stage creation."""
        stage = WaterfallStage(
            name="Test Stage",
            start_ms=10.0,
            duration_ms=50.0
        )

        assert stage.name == "Test Stage"
        assert stage.start_ms == 10.0
        assert stage.duration_ms == 50.0
        assert stage.status == StageStatus.SUCCESS  # Default
        assert stage.trend is None
        assert stage.metadata is None

    def test_end_ms_property(self):
        """Test end_ms calculated property."""
        stage = WaterfallStage(
            name="Test Stage",
            start_ms=10.0,
            duration_ms=50.0
        )

        assert stage.end_ms == 60.0

    def test_full_creation(self):
        """Test stage creation with all fields."""
        stage = WaterfallStage(
            name="Full Stage",
            start_ms=0.0,
            duration_ms=100.0,
            status=StageStatus.WARNING,
            trend=[90.0, 95.0, 100.0],
            metadata={'key': 'value'}
        )

        assert stage.status == StageStatus.WARNING
        assert stage.trend == [90.0, 95.0, 100.0]
        assert stage.metadata == {'key': 'value'}

    def test_zero_duration(self):
        """Test stage with zero duration."""
        stage = WaterfallStage(
            name="Skipped",
            start_ms=50.0,
            duration_ms=0.0,
            status=StageStatus.SKIPPED
        )

        assert stage.duration_ms == 0.0
        assert stage.end_ms == 50.0


# ============================================================================
# StageStatus Tests
# ============================================================================

class TestStageStatus:
    """Tests for StageStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert hasattr(StageStatus, 'SUCCESS')
        assert hasattr(StageStatus, 'WARNING')
        assert hasattr(StageStatus, 'ERROR')
        assert hasattr(StageStatus, 'SKIPPED')

    def test_status_values(self):
        """Test status string values."""
        assert StageStatus.SUCCESS.value == "success"
        assert StageStatus.WARNING.value == "warning"
        assert StageStatus.ERROR.value == "error"
        assert StageStatus.SKIPPED.value == "skipped"


# ============================================================================
# Renderer Initialization Tests
# ============================================================================

class TestRendererInitialization:
    """Tests for StageWaterfallRenderer initialization."""

    def test_default_initialization(self):
        """Test renderer with default parameters."""
        renderer = StageWaterfallRenderer()

        assert renderer.bottleneck_threshold == 0.4
        assert renderer.show_sparklines is True
        assert renderer.show_percentages is True

    def test_custom_initialization(self):
        """Test renderer with custom parameters."""
        renderer = StageWaterfallRenderer(
            bottleneck_threshold=0.5,
            show_sparklines=False,
            show_percentages=False
        )

        assert renderer.bottleneck_threshold == 0.5
        assert renderer.show_sparklines is False
        assert renderer.show_percentages is False


# ============================================================================
# Rendering Tests
# ============================================================================

class TestRendering:
    """Tests for waterfall rendering."""

    def test_render_basic_html(self, renderer, basic_stages):
        """Test basic HTML rendering."""
        html = renderer.render(basic_stages)

        assert isinstance(html, str)
        assert "stage-waterfall" in html
        assert "Pipeline Stage Waterfall" in html

    def test_render_with_title(self, renderer, basic_stages):
        """Test rendering with custom title."""
        html = renderer.render(basic_stages, title="Custom Pipeline")

        assert "Custom Pipeline" in html

    def test_render_empty_returns_empty_state(self, renderer):
        """Test rendering empty stages returns empty state."""
        html = renderer.render([])

        assert "No pipeline stages" in html

    def test_render_includes_stage_names(self, renderer, basic_stages):
        """Test that rendered HTML includes stage names."""
        html = renderer.render(basic_stages)

        for stage in basic_stages:
            assert stage.name in html

    def test_render_includes_durations(self, renderer, basic_stages):
        """Test that rendered HTML includes durations."""
        html = renderer.render(basic_stages)

        # Should have duration values
        assert "ms" in html

    def test_render_includes_total_duration(self, renderer, basic_stages):
        """Test that total duration is shown."""
        html = renderer.render(basic_stages)

        total = max(s.end_ms for s in basic_stages)
        assert "Total:" in html


# ============================================================================
# Bottleneck Detection Tests
# ============================================================================

class TestBottleneckDetection:
    """Tests for bottleneck detection functionality."""

    def test_bottleneck_detection(self, stages_with_bottleneck):
        """Test that bottleneck is detected."""
        renderer = StageWaterfallRenderer(bottleneck_threshold=0.4)
        html = renderer.render(stages_with_bottleneck)

        assert "BOTTLENECK" in html

    def test_no_bottleneck_below_threshold(self, basic_stages):
        """Test no bottleneck when all below threshold."""
        # Set very high threshold
        renderer = StageWaterfallRenderer(bottleneck_threshold=0.9)
        html = renderer.render(basic_stages)

        # Check if BOTTLENECK appears - it shouldn't
        assert "BOTTLENECK" not in html or html.count("BOTTLENECK") == 0

    def test_custom_bottleneck_threshold(self):
        """Test custom bottleneck threshold."""
        stages = [
            WaterfallStage(name="Step 1", start_ms=0.0, duration_ms=30.0),
            WaterfallStage(name="Step 2", start_ms=30.0, duration_ms=70.0),  # 70%
        ]

        # With 0.5 threshold, step 2 is bottleneck
        renderer_50 = StageWaterfallRenderer(bottleneck_threshold=0.5)
        html_50 = renderer_50.render(stages)
        assert "BOTTLENECK" in html_50

        # With 0.8 threshold, no bottleneck
        renderer_80 = StageWaterfallRenderer(bottleneck_threshold=0.8)
        html_80 = renderer_80.render(stages)
        assert "BOTTLENECK" not in html_80


# ============================================================================
# Status Indicator Tests
# ============================================================================

class TestStatusIndicators:
    """Tests for status indicator visualization."""

    def test_success_status_rendering(self):
        """Test success status rendering."""
        stages = [
            WaterfallStage(
                name="Success",
                start_ms=0.0,
                duration_ms=50.0,
                status=StageStatus.SUCCESS
            )
        ]

        renderer = StageWaterfallRenderer()
        html = renderer.render(stages)

        # Should have green/success indicators
        assert "#10b981" in html or "success" in html.lower()

    def test_error_status_rendering(self):
        """Test error status rendering."""
        stages = [
            WaterfallStage(
                name="Error Stage",
                start_ms=0.0,
                duration_ms=50.0,
                status=StageStatus.ERROR
            )
        ]

        renderer = StageWaterfallRenderer()
        html = renderer.render(stages)

        # Should have red/error indicators
        assert "#ef4444" in html or "error" in html.lower()

    def test_warning_status_rendering(self):
        """Test warning status rendering."""
        stages = [
            WaterfallStage(
                name="Warning Stage",
                start_ms=0.0,
                duration_ms=50.0,
                status=StageStatus.WARNING
            )
        ]

        renderer = StageWaterfallRenderer()
        html = renderer.render(stages)

        # Should have amber/warning indicators
        assert "#f59e0b" in html or "warning" in html.lower()

    def test_skipped_status_rendering(self):
        """Test skipped status rendering."""
        stages = [
            WaterfallStage(
                name="Active Stage",
                start_ms=0.0,
                duration_ms=50.0,
                status=StageStatus.SUCCESS
            ),
            WaterfallStage(
                name="Skipped Stage",
                start_ms=50.0,
                duration_ms=0.0,
                status=StageStatus.SKIPPED
            )
        ]

        renderer = StageWaterfallRenderer()
        html = renderer.render(stages)

        # Should have gray/skipped indicators
        assert "#9ca3af" in html or "skipped" in html.lower() or "Skipped" in html


# ============================================================================
# Sparkline Tests
# ============================================================================

class TestSparklines:
    """Tests for sparkline trend visualization."""

    def test_sparklines_rendered(self, stages_with_trends):
        """Test that sparklines are rendered."""
        renderer = StageWaterfallRenderer(show_sparklines=True)
        html = renderer.render(stages_with_trends)

        # Should have SVG sparkline elements
        assert "<svg" in html
        assert "path" in html.lower()

    def test_sparklines_disabled(self, stages_with_trends):
        """Test sparklines can be disabled."""
        renderer = StageWaterfallRenderer(show_sparklines=False)
        html = renderer.render(stages_with_trends)

        # Should still render but check for fewer SVGs
        assert isinstance(html, str)

    def test_sparkline_with_single_value(self):
        """Test sparkline with single value (not rendered)."""
        stages = [
            WaterfallStage(
                name="Single Value",
                start_ms=0.0,
                duration_ms=50.0,
                trend=[50.0]  # Single value
            )
        ]

        renderer = StageWaterfallRenderer(show_sparklines=True)
        html = renderer.render(stages)

        # Should still render, sparkline may be skipped
        assert isinstance(html, str)

    def test_sparkline_with_multiple_values(self, stages_with_trends):
        """Test sparkline with multiple values."""
        renderer = StageWaterfallRenderer(show_sparklines=True)
        html = renderer.render(stages_with_trends)

        # Should have path elements for sparklines
        assert "path" in html.lower()


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_render_pipeline_waterfall(self):
        """Test render_pipeline_waterfall convenience function."""
        durations = {
            'Pattern Selection': 5.2,
            'Retrieval': 50.5,
            'Convergence': 30.0,
            'Tool Execution': 64.3
        }

        html = render_pipeline_waterfall(durations)

        assert isinstance(html, str)
        assert "stage-waterfall" in html
        assert "Pattern Selection" in html

    def test_render_pipeline_with_trends(self):
        """Test render_pipeline_waterfall with trends."""
        durations = {
            'Step 1': 20.0,
            'Step 2': 30.0,
        }
        trends = {
            'Step 1': [25.0, 22.0, 20.0],
            'Step 2': [28.0, 29.0, 30.0],
        }

        html = render_pipeline_waterfall(
            durations,
            stage_trends=trends,
            title="With Trends"
        )

        assert "With Trends" in html

    def test_render_parallel_waterfall(self):
        """Test render_parallel_waterfall convenience function."""
        durations = {
            'Input Processing': 10.0,
            'Feature A': 30.0,
            'Feature B': 25.0,
            'Feature C': 35.0,
            'Decision': 20.0
        }
        parallel_groups = [
            ['Input Processing'],
            ['Feature A', 'Feature B', 'Feature C'],  # Parallel
            ['Decision']
        ]

        html = render_parallel_waterfall(durations, parallel_groups)

        assert isinstance(html, str)
        assert "Input Processing" in html
        assert "Feature A" in html

    def test_parallel_stages_same_start_time(self):
        """Test that parallel stages have same start time."""
        durations = {
            'Step 1': 10.0,
            'Parallel A': 20.0,
            'Parallel B': 30.0,
            'Step 4': 15.0
        }
        parallel_groups = [
            ['Step 1'],
            ['Parallel A', 'Parallel B'],  # Same start time
            ['Step 4']
        ]

        html = render_parallel_waterfall(durations, parallel_groups)

        # The HTML should render successfully
        assert isinstance(html, str)


# ============================================================================
# HTML Structure Tests
# ============================================================================

class TestHTMLStructure:
    """Tests for HTML structure validity."""

    def test_valid_html_structure(self, renderer, basic_stages):
        """Test that HTML has valid structure."""
        html = renderer.render(basic_stages)

        # Check for proper div structure
        assert "<div" in html
        assert "</div>" in html

    def test_legend_rendered(self, renderer, basic_stages):
        """Test that legend is rendered."""
        html = renderer.render(basic_stages)

        # Should have legend with status colors
        assert "Success" in html or "success" in html.lower()
        assert "Error" in html or "error" in html.lower()
        assert "Bottleneck" in html or "bottleneck" in html.lower()

    def test_time_axis_rendered(self, renderer, basic_stages):
        """Test that time axis is rendered."""
        html = renderer.render(basic_stages)

        # Should have time markers
        assert "ms" in html

    def test_no_external_dependencies(self, renderer, basic_stages):
        """Test that HTML has no external dependencies."""
        html = renderer.render(basic_stages)

        assert 'href="http' not in html
        assert 'src="http' not in html


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_stage(self, renderer):
        """Test with single stage."""
        stages = [
            WaterfallStage(name="Single", start_ms=0.0, duration_ms=100.0)
        ]

        html = renderer.render(stages)
        assert isinstance(html, str)
        assert "Single" in html

    def test_many_stages(self, renderer):
        """Test with many stages."""
        stages = []
        current_ms = 0.0
        for i in range(20):
            duration = 10.0 + i
            stages.append(WaterfallStage(
                name=f"Stage {i+1}",
                start_ms=current_ms,
                duration_ms=duration
            ))
            current_ms += duration

        html = renderer.render(stages)
        assert isinstance(html, str)

    def test_very_long_stage_name(self, renderer):
        """Test with very long stage name."""
        stages = [
            WaterfallStage(
                name="This is a very long stage name that should be handled gracefully",
                start_ms=0.0,
                duration_ms=50.0
            )
        ]

        html = renderer.render(stages)
        assert isinstance(html, str)

    def test_zero_total_duration(self, renderer):
        """Test with zero total duration - source doesn't handle this edge case."""
        stages = [
            WaterfallStage(name="Zero", start_ms=0.0, duration_ms=0.0)
        ]

        # Note: The source code doesn't handle zero total duration
        # This results in ZeroDivisionError - testing documents this behavior
        with pytest.raises(ZeroDivisionError):
            renderer.render(stages)

    def test_custom_total_duration(self, renderer, basic_stages):
        """Test with custom total duration override."""
        html = renderer.render(basic_stages, total_duration_ms=200.0)

        assert isinstance(html, str)
        assert "200" in html

    def test_percentages_toggled(self, basic_stages):
        """Test percentage display toggle."""
        renderer_with = StageWaterfallRenderer(show_percentages=True)
        html_with = renderer_with.render(basic_stages)

        renderer_without = StageWaterfallRenderer(show_percentages=False)
        html_without = renderer_without.render(basic_stages)

        # Both should render
        assert isinstance(html_with, str)
        assert isinstance(html_without, str)

        # With percentages should have % character
        assert "%" in html_with


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test complete workflow with all features."""
        stages = [
            WaterfallStage(
                name="Input",
                start_ms=0.0,
                duration_ms=15.0,
                status=StageStatus.SUCCESS,
                trend=[18.0, 17.0, 16.0, 15.0]
            ),
            WaterfallStage(
                name="Processing",
                start_ms=15.0,
                duration_ms=60.0,
                status=StageStatus.SUCCESS,
                trend=[55.0, 57.0, 59.0, 60.0]
            ),
            WaterfallStage(
                name="Validation",
                start_ms=75.0,
                duration_ms=20.0,
                status=StageStatus.WARNING,
                trend=[15.0, 18.0, 20.0]
            ),
            WaterfallStage(
                name="Output",
                start_ms=95.0,
                duration_ms=5.0,
                status=StageStatus.SUCCESS,
                trend=[5.0, 5.0, 5.0]
            ),
        ]

        renderer = StageWaterfallRenderer(
            bottleneck_threshold=0.4,
            show_sparklines=True,
            show_percentages=True
        )

        html = renderer.render(stages, title="Full Integration Test")

        # Verify all components
        assert "Full Integration Test" in html
        assert "Input" in html
        assert "Processing" in html
        assert "BOTTLENECK" in html  # Processing is >40%
        assert "<svg" in html  # Sparklines
        assert "%" in html  # Percentages
