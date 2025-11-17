"""Comprehensive test suite for AR visualization components.

Tests for:
- AR Overlay Renderer (7 overlay types)
- AR Chart Renderer (6 chart types)
- AR Heatmap Renderer (8 colormaps)

Implemented: 2025-11-17
"""

import pytest
import asyncio
import numpy as np
from typing import List, Tuple

# Import AR visualization modules
from HoloLoom.visualization.ar_overlay import (
    AROverlay,
    AROverlayRenderer,
    OverlayType,
    OverlayBlendMode,
    ProjectionMatrix,
)
from HoloLoom.visualization.ar_charts import (
    ARChart,
    ChartType,
    ChartConfig,
    ARChartRenderer,
    GaugeData,
    GaugeStyle,
    LiveChartUpdater,
)
from HoloLoom.visualization.ar_heatmap import (
    Heatmap,
    ARHeatmapRenderer,
    ColormapType,
    HeatmapAnimator,
)


# ============================================================================
# AR OVERLAY TESTS
# ============================================================================


class TestAROverlay:
    """Tests for AROverlay dataclass."""

    def test_overlay_creation(self):
        """Test creating an overlay."""
        overlay = AROverlay(
            type=OverlayType.BOUNDING_BOX,
            position=(0, 0, 1),
            color=(1.0, 0.0, 0.0, 1.0),
        )
        assert overlay.type == OverlayType.BOUNDING_BOX
        assert overlay.position == (0, 0, 1)
        assert overlay.is_expired() is False

    def test_overlay_expiration(self):
        """Test overlay expiration."""
        import time

        overlay = AROverlay(
            type=OverlayType.LABEL,
            position=(0, 0, 1),
            duration=0.1,
        )
        assert overlay.is_expired() is False
        time.sleep(0.15)
        assert overlay.is_expired() is True

    def test_overlay_with_content(self):
        """Test overlay with custom content."""
        overlay = AROverlay(
            type=OverlayType.LABEL,
            position=(0, 0, 1),
            content={"text": "Test Label", "font_scale": 0.8},
        )
        assert overlay.content["text"] == "Test Label"
        assert overlay.content["font_scale"] == 0.8

    def test_overlay_blend_modes(self):
        """Test different blend modes."""
        for mode in OverlayBlendMode:
            overlay = AROverlay(
                type=OverlayType.HIGHLIGHT,
                position=(0, 0, 1),
                blend_mode=mode,
            )
            assert overlay.blend_mode == mode


class TestAROverlayRenderer:
    """Tests for AROverlayRenderer."""

    @pytest.fixture
    def renderer(self):
        """Create renderer instance."""
        return AROverlayRenderer()

    @pytest.fixture
    def dummy_frame(self):
        """Create dummy image frame."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 255

    @pytest.fixture
    def camera_pos(self):
        """Default camera position."""
        return (0, 0, 0)

    @pytest.fixture
    def camera_rot(self):
        """Default camera rotation."""
        return (0, 0, 0)

    @pytest.mark.asyncio
    async def test_add_overlay(self, renderer):
        """Test adding overlay to renderer."""
        overlay = AROverlay(
            type=OverlayType.MARKER,
            position=(0, 0, 2),
        )
        await renderer.add_overlay(overlay)
        assert len(renderer.active_overlays) == 1

    @pytest.mark.asyncio
    async def test_remove_overlay(self, renderer):
        """Test removing overlay."""
        overlay = AROverlay(
            type=OverlayType.MARKER,
            position=(0, 0, 2),
        )
        await renderer.add_overlay(overlay)
        await renderer.remove_overlay(overlay)
        assert len(renderer.active_overlays) == 0

    @pytest.mark.asyncio
    async def test_clear_overlays(self, renderer):
        """Test clearing all overlays."""
        for i in range(5):
            overlay = AROverlay(
                type=OverlayType.MARKER,
                position=(0, 0, 2),
            )
            await renderer.add_overlay(overlay)
        assert len(renderer.active_overlays) == 5
        await renderer.clear_overlays()
        assert len(renderer.active_overlays) == 0

    @pytest.mark.asyncio
    async def test_render_bounding_box(self, renderer, dummy_frame, camera_pos, camera_rot):
        """Test rendering bounding box."""
        overlay = AROverlay(
            type=OverlayType.BOUNDING_BOX,
            position=(0, 0, 2),
            scale=(0.5, 0.5, 0.5),
            color=(1.0, 0.0, 0.0, 1.0),
        )
        await renderer.add_overlay(overlay)
        result = await renderer.render(camera_pos, camera_rot, dummy_frame)
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_label(self, renderer, dummy_frame, camera_pos, camera_rot):
        """Test rendering text label."""
        overlay = AROverlay(
            type=OverlayType.LABEL,
            position=(0, 0, 2),
            content={"text": "Test Label", "font_scale": 0.6},
            color=(1.0, 1.0, 1.0, 1.0),
        )
        await renderer.add_overlay(overlay)
        result = await renderer.render(camera_pos, camera_rot, dummy_frame)
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_marker(self, renderer, dummy_frame, camera_pos, camera_rot):
        """Test rendering point marker."""
        overlay = AROverlay(
            type=OverlayType.MARKER,
            position=(0, 0, 2),
            content={"radius": 8, "label": "Point A"},
            color=(0.0, 1.0, 0.0, 1.0),
        )
        await renderer.add_overlay(overlay)
        result = await renderer.render(camera_pos, camera_rot, dummy_frame)
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_arrow(self, renderer, dummy_frame, camera_pos, camera_rot):
        """Test rendering directional arrow."""
        overlay = AROverlay(
            type=OverlayType.ARROW,
            position=(0, 0, 2),
            content={"direction": [0, 0, 1], "length": 0.5},
            color=(1.0, 1.0, 0.0, 1.0),
        )
        await renderer.add_overlay(overlay)
        result = await renderer.render(camera_pos, camera_rot, dummy_frame)
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_path(self, renderer, dummy_frame, camera_pos, camera_rot):
        """Test rendering navigation path."""
        overlay = AROverlay(
            type=OverlayType.PATH,
            position=(0, 0, 2),
            content={
                "waypoints": [
                    (0, 0, 2),
                    (0.5, 0, 2),
                    (1, 0, 2),
                    (1, 0.5, 2),
                ],
                "marker_radius": 5,
            },
            color=(0.0, 1.0, 1.0, 1.0),
        )
        await renderer.add_overlay(overlay)
        result = await renderer.render(camera_pos, camera_rot, dummy_frame)
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_info_panel(self, renderer, dummy_frame, camera_pos, camera_rot):
        """Test rendering info panel."""
        overlay = AROverlay(
            type=OverlayType.INFO_PANEL,
            position=(0, 0, 2),
            content={
                "title": "System Info",
                "items": ["FPS: 60", "Latency: 15ms", "Memory: 512MB"],
                "width": 200,
                "height": 100,
            },
            color=(0.5, 0.5, 1.0, 1.0),
        )
        await renderer.add_overlay(overlay)
        result = await renderer.render(camera_pos, camera_rot, dummy_frame)
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_highlight(self, renderer, dummy_frame, camera_pos, camera_rot):
        """Test rendering highlight glow."""
        overlay = AROverlay(
            type=OverlayType.HIGHLIGHT,
            position=(0, 0, 2),
            scale=(0.3, 0.3, 0.3),
            content={"glow_radius": 20},
            color=(1.0, 0.5, 0.0, 1.0),
        )
        await renderer.add_overlay(overlay)
        result = await renderer.render(camera_pos, camera_rot, dummy_frame)
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_multiple_overlays(self, renderer, dummy_frame, camera_pos, camera_rot):
        """Test rendering multiple overlays."""
        for i in range(3):
            overlay = AROverlay(
                type=OverlayType.MARKER,
                position=(i * 0.5, 0, 2),
            )
            await renderer.add_overlay(overlay)

        result = await renderer.render(camera_pos, camera_rot, dummy_frame)
        assert result.shape == dummy_frame.shape
        assert len(renderer.active_overlays) == 3

    def test_projection_matrix_defaults(self):
        """Test projection matrix defaults."""
        proj = ProjectionMatrix(
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )
        assert proj.fx == 500.0
        assert proj.width == 640

    def test_get_stats(self, renderer):
        """Test getting renderer statistics."""
        stats = renderer.get_stats()
        assert "active_overlays" in stats
        assert "frame_count" in stats
        assert "by_type" in stats

    def test_max_overlays_limit(self):
        """Test maximum overlays limit."""
        renderer = AROverlayRenderer(max_overlays=10)

        async def add_many():
            for i in range(20):
                overlay = AROverlay(
                    type=OverlayType.MARKER,
                    position=(0, 0, 2),
                )
                await renderer.add_overlay(overlay)
            assert len(renderer.active_overlays) <= 10

        asyncio.run(add_many())


# ============================================================================
# AR CHART TESTS
# ============================================================================


class TestARChart:
    """Tests for ARChart dataclass."""

    def test_bar_chart_creation(self):
        """Test creating bar chart."""
        chart = ARChart(
            chart_type=ChartType.BAR,
            data=[10, 20, 15, 25],
            labels=["A", "B", "C", "D"],
        )
        assert chart.chart_type == ChartType.BAR
        assert len(chart.data) == 4

    def test_gauge_chart_creation(self):
        """Test creating gauge chart."""
        gauge_data = GaugeData(
            value=75.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
        )
        chart = ARChart(
            chart_type=ChartType.GAUGE,
            data=[gauge_data],
        )
        assert chart.chart_type == ChartType.GAUGE

    def test_chart_config(self):
        """Test chart configuration."""
        config = ChartConfig(
            title="Sales Report",
            width=400,
            height=300,
            show_legend=True,
            show_grid=True,
        )
        assert config.title == "Sales Report"
        assert config.width == 400


class TestARChartRenderer:
    """Tests for ARChartRenderer."""

    @pytest.fixture
    def renderer(self):
        """Create chart renderer."""
        return ARChartRenderer()

    @pytest.fixture
    def dummy_frame(self):
        """Create dummy frame."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 255

    @pytest.mark.asyncio
    async def test_render_bar_chart(self, renderer, dummy_frame):
        """Test rendering bar chart."""
        chart = ARChart(
            chart_type=ChartType.BAR,
            data=[10, 20, 15, 25],
            labels=["Q1", "Q2", "Q3", "Q4"],
            config=ChartConfig(title="Quarterly Sales"),
        )
        result = await renderer.render_chart(chart, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_line_chart(self, renderer, dummy_frame):
        """Test rendering line chart."""
        chart = ARChart(
            chart_type=ChartType.LINE,
            data=[10, 15, 12, 18, 22],
            config=ChartConfig(title="Trend Analysis"),
        )
        result = await renderer.render_chart(chart, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_pie_chart(self, renderer, dummy_frame):
        """Test rendering pie chart."""
        chart = ARChart(
            chart_type=ChartType.PIE,
            data=[30, 25, 20, 25],
            labels=["A", "B", "C", "D"],
            config=ChartConfig(title="Distribution"),
        )
        result = await renderer.render_chart(chart, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_gauge_chart(self, renderer, dummy_frame):
        """Test rendering gauge chart."""
        gauge_data = GaugeData(
            value=75.0,
            min_value=0.0,
            max_value=100.0,
            unit="%",
        )
        chart = ARChart(
            chart_type=ChartType.GAUGE,
            data=[gauge_data],
            config=ChartConfig(title="Progress"),
        )
        result = await renderer.render_chart(chart, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_histogram(self, renderer, dummy_frame):
        """Test rendering histogram."""
        chart = ARChart(
            chart_type=ChartType.HISTOGRAM,
            data=[5, 10, 8, 12, 7, 9],
            config=ChartConfig(title="Distribution"),
        )
        result = await renderer.render_chart(chart, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_scatter_chart(self, renderer, dummy_frame):
        """Test rendering scatter plot."""
        chart = ARChart(
            chart_type=ChartType.SCATTER,
            data=[(1, 2), (2, 3), (3, 2.5), (4, 4)],
            config=ChartConfig(title="Correlation"),
        )
        result = await renderer.render_chart(chart, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    def test_value_to_color(self, renderer):
        """Test value to color conversion."""
        # Good value (green)
        color_good = renderer._value_to_color(0.8)
        assert color_good[1] > color_good[2]  # Green > Red

        # Bad value (red)
        color_bad = renderer._value_to_color(0.2)
        assert color_bad[2] > color_bad[1]  # Red > Green


class TestLiveChartUpdater:
    """Tests for LiveChartUpdater."""

    @pytest.mark.asyncio
    async def test_register_chart(self):
        """Test registering chart."""
        updater = LiveChartUpdater()
        chart = ARChart(
            chart_type=ChartType.BAR,
            data=[10, 20, 30],
        )
        await updater.register_chart("chart1", chart)
        retrieved = await updater.get_chart("chart1")
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_update_data(self):
        """Test updating chart data."""
        updater = LiveChartUpdater()
        chart = ARChart(
            chart_type=ChartType.BAR,
            data=[10, 20, 30],
        )
        await updater.register_chart("chart1", chart)

        new_data = [15, 25, 35]
        await updater.update_data("chart1", new_data)

        updated = await updater.get_chart("chart1")
        assert updated.data == new_data


# ============================================================================
# AR HEATMAP TESTS
# ============================================================================


class TestHeatmap:
    """Tests for Heatmap dataclass."""

    def test_heatmap_creation(self):
        """Test creating heatmap."""
        grid = np.random.rand(50, 50)
        heatmap = Heatmap(
            grid=grid,
            colormap=ColormapType.HOT,
        )
        assert heatmap.grid.shape == (50, 50)
        assert heatmap.colormap == ColormapType.HOT

    def test_heatmap_with_label(self):
        """Test heatmap with label."""
        grid = np.random.rand(50, 50)
        heatmap = Heatmap(
            grid=grid,
            label="Temperature",
        )
        assert heatmap.label == "Temperature"


class TestARHeatmapRenderer:
    """Tests for ARHeatmapRenderer."""

    @pytest.fixture
    def renderer(self):
        """Create heatmap renderer."""
        return ARHeatmapRenderer()

    @pytest.fixture
    def dummy_frame(self):
        """Create dummy frame."""
        return np.ones((480, 640, 3), dtype=np.uint8) * 255

    @pytest.fixture
    def dummy_grid(self):
        """Create dummy heatmap grid."""
        return np.random.rand(30, 40)

    def test_colormap_initialization(self, renderer):
        """Test colormap initialization."""
        assert ColormapType.HOT in renderer.colormaps
        assert ColormapType.COOL in renderer.colormaps
        assert ColormapType.VIRIDIS in renderer.colormaps
        assert len(renderer.colormaps) == 8

    @pytest.mark.asyncio
    async def test_render_hot_heatmap(self, renderer, dummy_frame, dummy_grid):
        """Test rendering with HOT colormap."""
        heatmap = Heatmap(
            grid=dummy_grid,
            colormap=ColormapType.HOT,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_cool_heatmap(self, renderer, dummy_frame, dummy_grid):
        """Test rendering with COOL colormap."""
        heatmap = Heatmap(
            grid=dummy_grid,
            colormap=ColormapType.COOL,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_viridis_heatmap(self, renderer, dummy_frame, dummy_grid):
        """Test rendering with VIRIDIS colormap."""
        heatmap = Heatmap(
            grid=dummy_grid,
            colormap=ColormapType.VIRIDIS,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_plasma_heatmap(self, renderer, dummy_frame, dummy_grid):
        """Test rendering with PLASMA colormap."""
        heatmap = Heatmap(
            grid=dummy_grid,
            colormap=ColormapType.PLASMA,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_jet_heatmap(self, renderer, dummy_frame, dummy_grid):
        """Test rendering with JET colormap."""
        heatmap = Heatmap(
            grid=dummy_grid,
            colormap=ColormapType.JET,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_turbo_heatmap(self, renderer, dummy_frame, dummy_grid):
        """Test rendering with TURBO colormap."""
        heatmap = Heatmap(
            grid=dummy_grid,
            colormap=ColormapType.TURBO,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_inferno_heatmap(self, renderer, dummy_frame, dummy_grid):
        """Test rendering with INFERNO colormap."""
        heatmap = Heatmap(
            grid=dummy_grid,
            colormap=ColormapType.INFERNO,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_grayscale_heatmap(self, renderer, dummy_frame, dummy_grid):
        """Test rendering with GRAYSCALE colormap."""
        heatmap = Heatmap(
            grid=dummy_grid,
            colormap=ColormapType.GRAYSCALE,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_render_multiple_heatmaps(self, renderer, dummy_frame, dummy_grid):
        """Test rendering multiple heatmaps."""
        heatmaps = [
            Heatmap(grid=dummy_grid, colormap=ColormapType.HOT),
            Heatmap(grid=dummy_grid, colormap=ColormapType.COOL),
            Heatmap(grid=dummy_grid, colormap=ColormapType.VIRIDIS),
        ]
        result = await renderer.render_multiple(heatmaps, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_difference_heatmap(self, renderer):
        """Test creating difference heatmap."""
        grid1 = np.random.rand(30, 40)
        grid2 = np.random.rand(30, 40)

        heatmap = await renderer.create_difference_heatmap(grid1, grid2, (0, 0, 1))
        assert heatmap.grid.shape == grid1.shape
        assert heatmap.colormap == ColormapType.JET

    @pytest.mark.asyncio
    async def test_gradient_heatmap(self, renderer):
        """Test creating gradient heatmap."""
        grid = np.random.rand(30, 40)
        heatmap = await renderer.create_gradient_heatmap(grid, (0, 0, 1))
        assert heatmap.grid.shape == grid.shape
        assert heatmap.colormap == ColormapType.PLASMA

    @pytest.mark.asyncio
    async def test_heatmap_normalization(self, renderer, dummy_frame):
        """Test automatic heatmap normalization."""
        grid = np.array([[100, 200], [300, 400]])
        heatmap = Heatmap(
            grid=grid,
            auto_normalize=True,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_heatmap_with_opacity(self, renderer, dummy_frame, dummy_grid):
        """Test heatmap with custom opacity."""
        heatmap = Heatmap(
            grid=dummy_grid,
            opacity=0.5,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_heatmap_with_intensity(self, renderer, dummy_frame, dummy_grid):
        """Test heatmap with custom intensity."""
        heatmap = Heatmap(
            grid=dummy_grid,
            intensity=0.8,
        )
        result = await renderer.render(heatmap, dummy_frame, (0, 0, 0))
        assert result.shape == dummy_frame.shape


class TestHeatmapAnimator:
    """Tests for HeatmapAnimator."""

    @pytest.mark.asyncio
    async def test_register_heatmap(self):
        """Test registering heatmap."""
        animator = HeatmapAnimator()
        grid = np.random.rand(30, 40)
        heatmap = Heatmap(grid=grid)

        await animator.register_heatmap("hm1", heatmap)
        retrieved = await animator.get_heatmap("hm1")
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_unregister_heatmap(self):
        """Test unregistering heatmap."""
        animator = HeatmapAnimator()
        grid = np.random.rand(30, 40)
        heatmap = Heatmap(grid=grid)

        await animator.register_heatmap("hm1", heatmap)
        await animator.unregister_heatmap("hm1")

        retrieved = await animator.get_heatmap("hm1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_start_stop_animation(self):
        """Test starting and stopping heatmap animation."""
        animator = HeatmapAnimator()
        grid = np.random.rand(30, 40)
        heatmap = Heatmap(grid=grid)

        await animator.register_heatmap("hm1", heatmap)

        # Create update function
        async def update_func():
            return np.random.rand(30, 40)

        # Start animation
        await animator.start_animation("hm1", update_func)
        await asyncio.sleep(0.2)

        # Stop animation
        await animator.stop_animation("hm1")

        # Verify state
        retrieved = await animator.get_heatmap("hm1")
        assert retrieved is not None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests across components."""

    @pytest.mark.asyncio
    async def test_overlay_and_chart(self):
        """Test using overlays and charts together."""
        overlay_renderer = AROverlayRenderer()
        chart_renderer = ARChartRenderer()

        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Add overlay
        overlay = AROverlay(
            type=OverlayType.MARKER,
            position=(0, 0, 2),
        )
        await overlay_renderer.add_overlay(overlay)

        # Render overlay
        frame1 = await overlay_renderer.render((0, 0, 0), (0, 0, 0), dummy_frame)

        # Render chart on top
        chart = ARChart(
            chart_type=ChartType.BAR,
            data=[10, 20, 15],
        )
        frame2 = await chart_renderer.render_chart(chart, frame1, (0, 0, 0))

        assert frame2.shape == dummy_frame.shape

    @pytest.mark.asyncio
    async def test_heatmap_and_overlay(self):
        """Test using heatmaps with overlays."""
        heatmap_renderer = ARHeatmapRenderer()
        overlay_renderer = AROverlayRenderer()

        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        grid = np.random.rand(30, 40)

        # Render heatmap
        heatmap = Heatmap(grid=grid)
        frame1 = await heatmap_renderer.render(heatmap, dummy_frame, (0, 0, 0))

        # Add overlay on heatmap
        overlay = AROverlay(
            type=OverlayType.LABEL,
            position=(0, 0, 2),
            content={"text": "Heatmap Active"},
        )
        await overlay_renderer.add_overlay(overlay)
        frame2 = await overlay_renderer.render((0, 0, 0), (0, 0, 0), frame1)

        assert frame2.shape == dummy_frame.shape


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.asyncio
    async def test_render_many_overlays(self):
        """Test rendering performance with many overlays."""
        renderer = AROverlayRenderer()
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        # Add 100 overlays
        for i in range(100):
            overlay = AROverlay(
                type=OverlayType.MARKER,
                position=(i * 0.01, 0, 2),
            )
            await renderer.add_overlay(overlay)

        # Render (should complete without hanging)
        import time

        start = time.time()
        result = await renderer.render((0, 0, 0), (0, 0, 0), dummy_frame)
        elapsed = time.time() - start

        assert result.shape == dummy_frame.shape
        # Should complete in reasonable time (< 2 seconds)
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_chart_rendering_speed(self):
        """Test chart rendering speed."""
        renderer = ARChartRenderer()
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255

        chart = ARChart(
            chart_type=ChartType.BAR,
            data=[10, 20, 15, 25, 30],
        )

        import time

        start = time.time()
        result = await renderer.render_chart(chart, dummy_frame, (0, 0, 0))
        elapsed = time.time() - start

        assert result.shape == dummy_frame.shape
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
