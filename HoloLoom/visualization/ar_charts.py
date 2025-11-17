"""AR Data Visualization Charts for augmented reality.

Render data charts (bar, line, pie, gauge) in AR space with floating
3D positioning and real-time updates.

Implemented: 2025-11-17
Part of Wave 5: Advanced AR Integration
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import numpy as np
import asyncio
from datetime import datetime


class ChartType(Enum):
    """AR chart types."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SCATTER = "scatter"


class GaugeStyle(Enum):
    """Gauge visualization style."""
    CIRCULAR = "circular"
    LINEAR = "linear"
    RADIAL = "radial"


@dataclass
class ChartConfig:
    """Configuration for AR charts.

    Attributes:
        title: Chart title
        width: Width in screen pixels
        height: Height in screen pixels
        show_legend: Display legend
        show_grid: Display grid lines
        show_values: Display numeric values
        bg_opacity: Background opacity (0-1)
        font_size: Font size in pixels
    """
    title: str = ""
    width: int = 300
    height: int = 200
    show_legend: bool = True
    show_grid: bool = True
    show_values: bool = True
    bg_opacity: float = 0.8
    font_size: int = 12


@dataclass
class ARChart:
    """AR data visualization in 3D space.

    Attributes:
        chart_type: Type of chart
        data: List of numeric values or list of (x, y) tuples for scatter
        labels: Labels for data points
        position: 3D position in world space (x, y, z)
        scale: Visual scale factor
        rotation: Euler angles (yaw, pitch, roll)
        config: Chart configuration
    """
    chart_type: ChartType
    data: Union[List[float], List[Tuple[float, float]]]
    labels: List[str] = field(default_factory=list)
    position: Tuple[float, float, float] = (0, 0, 1)
    scale: float = 1.0
    rotation: Tuple[float, float, float] = (0, 0, 0)
    config: ChartConfig = field(default_factory=ChartConfig)
    colors: Optional[List[Tuple[int, int, int]]] = None


@dataclass
class GaugeData:
    """Gauge-specific data.

    Attributes:
        value: Current value
        min_value: Minimum value
        max_value: Maximum value
        unit: Unit of measurement
        style: Gauge style
    """
    value: float
    min_value: float = 0.0
    max_value: float = 100.0
    unit: str = ""
    style: GaugeStyle = GaugeStyle.CIRCULAR


class ARChartRenderer:
    """Render data charts in AR space.

    Manages rendering of various chart types with AR projection
    and real-time updates.
    """

    def __init__(self):
        """Initialize AR chart renderer."""
        self.rendered_charts: Dict[str, np.ndarray] = {}
        self._lock = asyncio.Lock()

    async def render_chart(
        self,
        chart: ARChart,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render chart onto frame.

        Args:
            chart: ARChart to render
            frame: Input image frame
            camera_position: Camera position for projection

        Returns:
            Frame with rendered chart
        """
        try:
            import cv2
        except ImportError:
            return frame

        # Render chart to buffer
        if chart.chart_type == ChartType.BAR:
            chart_img = await self._render_bar_chart(chart)
        elif chart.chart_type == ChartType.LINE:
            chart_img = await self._render_line_chart(chart)
        elif chart.chart_type == ChartType.PIE:
            chart_img = await self._render_pie_chart(chart)
        elif chart.chart_type == ChartType.GAUGE:
            chart_img = await self._render_gauge(chart)
        elif chart.chart_type == ChartType.HISTOGRAM:
            chart_img = await self._render_histogram(chart)
        elif chart.chart_type == ChartType.SCATTER:
            chart_img = await self._render_scatter_chart(chart)
        else:
            return frame

        # Project chart onto AR plane
        output = await self._project_chart_to_frame(
            frame,
            chart_img,
            chart.position,
            camera_position,
            chart.config.width,
            chart.config.height,
        )

        return output

    async def _render_bar_chart(self, chart: ARChart) -> np.ndarray:
        """Render bar chart to numpy array."""
        try:
            import cv2
        except ImportError:
            return np.zeros((chart.config.height, chart.config.width, 3), dtype=np.uint8)

        img = np.ones(
            (chart.config.height, chart.config.width, 3), dtype=np.uint8
        ) * 255

        data = chart.data
        labels = chart.labels or [str(i) for i in range(len(data))]
        config = chart.config

        if len(data) == 0:
            return img

        # Normalize data
        max_val = max(data) if data else 1
        min_val = min(data) if data else 0
        range_val = max_val - min_val if max_val > min_val else 1

        # Draw background
        cv2.rectangle(img, (0, 0), (config.width - 1, config.height - 1), (200, 200, 200), 1)

        # Draw title
        if config.title:
            cv2.putText(
                img,
                config.title,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Draw bars
        bar_width = (config.width - 40) / len(data)
        chart_height = config.height - 60

        for i, value in enumerate(data):
            # Normalize to chart height
            normalized = (value - min_val) / range_val if range_val > 0 else 0
            bar_height = int(normalized * chart_height)

            x1 = int(20 + i * bar_width + 2)
            y1 = int(config.height - 30 - bar_height)
            x2 = int(20 + (i + 1) * bar_width - 2)
            y2 = int(config.height - 30)

            # Bar color
            color = chart.colors[i % len(chart.colors)] if chart.colors else (50, 100, 200)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

            # Value text
            if config.show_values:
                cv2.putText(
                    img,
                    f"{value:.1f}",
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    1,
                )

            # Label
            label = labels[i] if i < len(labels) else str(i)
            cv2.putText(
                img,
                label[:5],
                (x1, config.height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                1,
            )

        # Draw axes
        cv2.line(img, (20, config.height - 30), (config.width - 10, config.height - 30), (0, 0, 0), 1)
        cv2.line(img, (20, 30), (20, config.height - 30), (0, 0, 0), 1)

        return img

    async def _render_line_chart(self, chart: ARChart) -> np.ndarray:
        """Render line chart to numpy array."""
        try:
            import cv2
        except ImportError:
            return np.zeros((chart.config.height, chart.config.width, 3), dtype=np.uint8)

        img = np.ones(
            (chart.config.height, chart.config.width, 3), dtype=np.uint8
        ) * 255

        data = chart.data
        config = chart.config

        if len(data) < 2:
            return img

        # Normalize data
        max_val = max(data) if data else 1
        min_val = min(data) if data else 0
        range_val = max_val - min_val if max_val > min_val else 1

        # Draw background
        cv2.rectangle(img, (0, 0), (config.width - 1, config.height - 1), (200, 200, 200), 1)

        # Draw title
        if config.title:
            cv2.putText(
                img,
                config.title,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Draw grid
        if config.show_grid:
            for y in range(50, config.height - 30, 20):
                cv2.line(img, (20, y), (config.width - 10, y), (220, 220, 220), 1)

        # Draw line
        points = []
        chart_width = config.width - 40
        chart_height = config.height - 60

        for i, value in enumerate(data):
            x = int(20 + (i / (len(data) - 1)) * chart_width) if len(data) > 1 else 20
            normalized = (value - min_val) / range_val if range_val > 0 else 0
            y = int(config.height - 30 - normalized * chart_height)
            points.append((x, y))

        # Draw line segments
        color = (0, 100, 200)
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, 2)

        # Draw points
        for point in points:
            cv2.circle(img, point, 3, color, -1)

        # Draw axes
        cv2.line(img, (20, config.height - 30), (config.width - 10, config.height - 30), (0, 0, 0), 1)
        cv2.line(img, (20, 30), (20, config.height - 30), (0, 0, 0), 1)

        return img

    async def _render_pie_chart(self, chart: ARChart) -> np.ndarray:
        """Render pie chart to numpy array."""
        try:
            import cv2
        except ImportError:
            return np.zeros((chart.config.height, chart.config.width, 3), dtype=np.uint8)

        img = np.ones(
            (chart.config.height, chart.config.width, 3), dtype=np.uint8
        ) * 255

        data = chart.data
        labels = chart.labels or [str(i) for i in range(len(data))]
        config = chart.config

        if not data:
            return img

        # Draw title
        if config.title:
            cv2.putText(
                img,
                config.title,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Draw pie
        center = (config.width // 2, config.height // 2)
        radius = min(config.width, config.height) // 3
        total = sum(data)

        start_angle = 0
        for i, value in enumerate(data):
            angle = (value / total) * 360 if total > 0 else 0
            color = chart.colors[i % len(chart.colors)] if chart.colors else (50, 100 + i * 30, 200)

            # Draw pie slice
            cv2.ellipse(
                img,
                center,
                (radius, radius),
                0,
                start_angle,
                start_angle + angle,
                color,
                -1,
            )

            # Draw label
            mid_angle = start_angle + angle / 2
            rad = np.radians(mid_angle)
            label_x = int(center[0] + (radius + 20) * np.cos(rad))
            label_y = int(center[1] + (radius + 20) * np.sin(rad))

            label = labels[i] if i < len(labels) else str(i)
            cv2.putText(
                img,
                f"{label} ({value:.0f}%)",
                (label_x - 20, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                1,
            )

            start_angle += angle

        return img

    async def _render_gauge(self, chart: ARChart) -> np.ndarray:
        """Render gauge chart to numpy array."""
        try:
            import cv2
        except ImportError:
            return np.zeros((chart.config.height, chart.config.width, 3), dtype=np.uint8)

        img = np.ones(
            (chart.config.height, chart.config.width, 3), dtype=np.uint8
        ) * 255

        # Extract gauge data
        gauge_data: GaugeData = chart.data[0] if isinstance(chart.data, list) else chart.data

        if isinstance(gauge_data, (int, float)):
            # Simple gauge
            value = gauge_data
            min_val = 0.0
            max_val = 100.0
        else:
            value = gauge_data.value
            min_val = gauge_data.min_value
            max_val = gauge_data.max_value

        config = chart.config
        center = (config.width // 2, config.height // 2)
        radius = min(config.width, config.height) // 3

        # Normalize value
        normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0
        normalized = max(0, min(1, normalized))

        # Draw background arc
        cv2.ellipse(
            img,
            center,
            (radius, radius),
            0,
            0,
            360,
            (200, 200, 200),
            15,
        )

        # Draw value arc
        angle = int(360 * normalized)
        color = (0, int(200 * normalized), int(200 * (1 - normalized)))
        cv2.ellipse(
            img,
            center,
            (radius, radius),
            0,
            0,
            angle,
            color,
            15,
        )

        # Draw center circle
        cv2.circle(img, center, 20, (255, 255, 255), -1)

        # Draw value text
        text = f"{value:.1f}"
        cv2.putText(
            img,
            text,
            (center[0] - 30, center[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        # Draw title
        if config.title:
            cv2.putText(
                img,
                config.title,
                (center[0] - 40, center[1] + radius + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return img

    async def _render_histogram(self, chart: ARChart) -> np.ndarray:
        """Render histogram to numpy array."""
        # Reuse bar chart for histogram
        return await self._render_bar_chart(chart)

    async def _render_scatter_chart(self, chart: ARChart) -> np.ndarray:
        """Render scatter plot to numpy array."""
        try:
            import cv2
        except ImportError:
            return np.zeros((chart.config.height, chart.config.width, 3), dtype=np.uint8)

        img = np.ones(
            (chart.config.height, chart.config.width, 3), dtype=np.uint8
        ) * 255

        data = chart.data
        config = chart.config

        if not data:
            return img

        # Draw title
        if config.title:
            cv2.putText(
                img,
                config.title,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Extract x, y values
        x_vals = [p[0] for p in data]
        y_vals = [p[1] for p in data]

        max_x = max(x_vals) if x_vals else 1
        min_x = min(x_vals) if x_vals else 0
        max_y = max(y_vals) if y_vals else 1
        min_y = min(y_vals) if y_vals else 0

        range_x = max_x - min_x if max_x > min_x else 1
        range_y = max_y - min_y if max_y > min_y else 1

        # Draw grid
        if config.show_grid:
            for y in range(50, config.height - 30, 20):
                cv2.line(img, (20, y), (config.width - 10, y), (220, 220, 220), 1)

        # Draw points
        chart_width = config.width - 40
        chart_height = config.height - 60

        for x, y in data:
            px = int(20 + ((x - min_x) / range_x) * chart_width) if range_x > 0 else 20
            py = int(config.height - 30 - ((y - min_y) / range_y) * chart_height) if range_y > 0 else config.height - 30

            cv2.circle(img, (px, py), 4, (0, 100, 200), -1)

        # Draw axes
        cv2.line(img, (20, config.height - 30), (config.width - 10, config.height - 30), (0, 0, 0), 1)
        cv2.line(img, (20, 30), (20, config.height - 30), (0, 0, 0), 1)

        return img

    async def _project_chart_to_frame(
        self,
        frame: np.ndarray,
        chart_img: np.ndarray,
        position_3d: Tuple[float, float, float],
        camera_position: Tuple[float, float, float],
        chart_width: int,
        chart_height: int,
    ) -> np.ndarray:
        """Project chart onto frame.

        Simple 2D placement for MVP - future: full 3D perspective projection.
        """
        try:
            import cv2
        except ImportError:
            return frame

        # Simple projection: place chart at position on screen
        # For full 3D projection, would need perspective transform
        screen_x = int(frame.shape[1] / 2)
        screen_y = int(frame.shape[0] / 2)

        output = frame.copy()

        # Resize chart to fit
        if chart_img.shape[0] > 0 and chart_img.shape[1] > 0:
            chart_resized = cv2.resize(chart_img, (chart_width, chart_height))

            # Blend chart with frame
            x1 = max(0, screen_x - chart_width // 2)
            y1 = max(0, screen_y - chart_height // 2)
            x2 = min(output.shape[1], x1 + chart_width)
            y2 = min(output.shape[0], y1 + chart_height)

            # Adjust chart size if needed
            cw = x2 - x1
            ch = y2 - y1

            if cw > 0 and ch > 0:
                chart_patch = cv2.resize(chart_img, (cw, ch))
                output[y1:y2, x1:x2] = cv2.addWeighted(
                    output[y1:y2, x1:x2], 0.3, chart_patch, 0.7, 0
                )

        return output

    def _value_to_color(self, normalized: float) -> Tuple[int, int, int]:
        """Convert normalized value (0-1) to RGB color (BGR for OpenCV).

        Green (good) -> Yellow (medium) -> Red (bad)
        """
        if normalized >= 0.66:
            # Green
            return (0, int(255 * normalized), 0)
        elif normalized >= 0.33:
            # Yellow
            return (0, 255, int(255 * (1 - normalized)))
        else:
            # Red
            return (0, int(255 * (1 - normalized)), 255)


class LiveChartUpdater:
    """Update charts with new data in real-time."""

    def __init__(self):
        """Initialize chart updater."""
        self.charts: Dict[str, ARChart] = {}
        self._lock = asyncio.Lock()

    async def register_chart(self, chart_id: str, chart: ARChart) -> None:
        """Register chart for updates."""
        async with self._lock:
            self.charts[chart_id] = chart

    async def update_data(self, chart_id: str, data: Union[List[float], List[Tuple[float, float]]]) -> None:
        """Update chart data."""
        async with self._lock:
            if chart_id in self.charts:
                self.charts[chart_id].data = data

    async def get_chart(self, chart_id: str) -> Optional[ARChart]:
        """Get chart by ID."""
        async with self._lock:
            return self.charts.get(chart_id)
