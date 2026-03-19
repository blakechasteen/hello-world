"""
Feature Activation Heatmap Visualization

Creates heatmaps showing feature activations across layers,
tokens, or time. Supports SAE feature visualization and
semantic axis projections.

Author: HoloLoom Team
Created: December 2025
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class HeatmapConfig:
    """Configuration for heatmap rendering."""
    width: int = 800
    height: int = 500
    cell_width: int = 30
    cell_height: int = 20
    colormap: str = "viridis"  # viridis, plasma, inferno, magma, coolwarm
    show_values: bool = True
    show_colorbar: bool = True
    title_font_size: int = 14
    label_font_size: int = 10


class FeatureHeatmap:
    """
    Feature activation heatmap visualization.

    Visualizes activations as colored cells in a grid,
    with support for multiple colormap options.
    """

    # Colormap definitions (CSS gradient stops)
    COLORMAPS = {
        "viridis": [
            (0.0, "#440154"),
            (0.25, "#414487"),
            (0.5, "#2a788e"),
            (0.75, "#22a884"),
            (1.0, "#7ad151"),
        ],
        "plasma": [
            (0.0, "#0d0887"),
            (0.25, "#7e03a8"),
            (0.5, "#cc4778"),
            (0.75, "#f89540"),
            (1.0, "#f0f921"),
        ],
        "inferno": [
            (0.0, "#000004"),
            (0.25, "#420a68"),
            (0.5, "#932667"),
            (0.75, "#dd513a"),
            (1.0, "#fca50a"),
        ],
        "coolwarm": [
            (0.0, "#3b4cc0"),
            (0.25, "#7b9ff9"),
            (0.5, "#f7f7f7"),
            (0.75, "#f4a582"),
            (1.0, "#b40426"),
        ],
        "magma": [
            (0.0, "#000004"),
            (0.25, "#3b0f70"),
            (0.5, "#8c2981"),
            (0.75, "#de4968"),
            (1.0, "#fec287"),
        ],
    }

    def __init__(self, config: HeatmapConfig | None = None):
        self.config = config or HeatmapConfig()

    def _interpolate_color(self, value: float, colormap: str) -> str:
        """Interpolate color from colormap based on value [0, 1]."""
        stops = self.COLORMAPS.get(colormap, self.COLORMAPS["viridis"])
        value = max(0.0, min(1.0, value))

        # Find surrounding stops
        for i in range(len(stops) - 1):
            if stops[i][0] <= value <= stops[i + 1][0]:
                t = (value - stops[i][0]) / (stops[i + 1][0] - stops[i][0])
                c1 = self._parse_hex(stops[i][1])
                c2 = self._parse_hex(stops[i + 1][1])
                r = int(c1[0] + t * (c2[0] - c1[0]))
                g = int(c1[1] + t * (c2[1] - c1[1]))
                b = int(c1[2] + t * (c2[2] - c1[2]))
                return f"#{r:02x}{g:02x}{b:02x}"

        return stops[-1][1]

    def _parse_hex(self, hex_color: str) -> tuple[int, int, int]:
        """Parse hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

    def render(
        self,
        data: np.ndarray,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        title: str = "Feature Activations",
        subtitle: str | None = None,
    ) -> str:
        """
        Render heatmap as HTML.

        Args:
            data: 2D numpy array of values
            row_labels: Labels for rows (e.g., layer names)
            col_labels: Labels for columns (e.g., token positions)
            title: Chart title
            subtitle: Optional subtitle

        Returns:
            HTML string
        """
        n_rows, n_cols = data.shape

        # Auto-generate labels if not provided
        if row_labels is None:
            row_labels = [f"Row {i}" for i in range(n_rows)]
        if col_labels is None:
            col_labels = [f"Col {i}" for i in range(n_cols)]

        # Normalize data to [0, 1]
        vmin, vmax = float(np.min(data)), float(np.max(data))
        if vmax - vmin > 0:
            normalized = (data - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data)

        # Calculate dimensions
        label_width = 80
        label_height = 40
        colorbar_width = 60 if self.config.show_colorbar else 0

        grid_width = n_cols * self.config.cell_width
        grid_height = n_rows * self.config.cell_height

        total_width = label_width + grid_width + colorbar_width + 40
        total_height = label_height + grid_height + 60

        # Build SVG
        cells_svg = self._render_cells(normalized, data, vmin, vmax)
        row_labels_svg = self._render_row_labels(row_labels, label_width, label_height)
        col_labels_svg = self._render_col_labels(col_labels, label_width, label_height)
        colorbar_svg = self._render_colorbar(vmin, vmax, grid_width, label_width, label_height) if self.config.show_colorbar else ""

        subtitle_html = f'<p style="color: #888; margin: 0;">{subtitle}</p>' if subtitle else ""

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        :root {{
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #e94560;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            color: var(--accent);
            margin-bottom: 5px;
        }}
        .heatmap-container {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            overflow-x: auto;
        }}
        .cell {{
            cursor: pointer;
        }}
        .cell:hover {{
            stroke: #fff;
            stroke-width: 2;
        }}
        .tooltip {{
            position: fixed;
            background: var(--card-bg);
            border: 1px solid #333;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            z-index: 1000;
        }}
        .label {{
            font-size: {self.config.label_font_size}px;
            fill: #aaa;
        }}
        .value-text {{
            font-size: 8px;
            fill: #fff;
            pointer-events: none;
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>{title}</h1>
    {subtitle_html}
    <div class="heatmap-container">
        <svg width="{total_width}" height="{total_height}">
            <g transform="translate({label_width}, {label_height})">
                {cells_svg}
            </g>
            {row_labels_svg}
            {col_labels_svg}
            {colorbar_svg}
        </svg>
    </div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
    const tooltip = document.getElementById('tooltip');
    document.querySelectorAll('.cell').forEach(cell => {{
        cell.addEventListener('mouseenter', (e) => {{
            const row = cell.dataset.row;
            const col = cell.dataset.col;
            const value = cell.dataset.value;
            tooltip.innerHTML = `Row: ${{row}}<br>Col: ${{col}}<br>Value: ${{value}}`;
            tooltip.style.opacity = 1;
        }});
        cell.addEventListener('mousemove', (e) => {{
            tooltip.style.left = (e.clientX + 10) + 'px';
            tooltip.style.top = (e.clientY + 10) + 'px';
        }});
        cell.addEventListener('mouseleave', () => {{
            tooltip.style.opacity = 0;
        }});
    }});
</script>
</body>
</html>
"""

    def _render_cells(
        self,
        normalized: np.ndarray,
        original: np.ndarray,
        vmin: float,
        vmax: float
    ) -> str:
        """Render heatmap cells."""
        n_rows, n_cols = normalized.shape
        cells = []

        for i in range(n_rows):
            for j in range(n_cols):
                x = j * self.config.cell_width
                y = i * self.config.cell_height
                value = float(original[i, j])
                norm_value = float(normalized[i, j])
                color = self._interpolate_color(norm_value, self.config.colormap)

                cell = (
                    f'<rect class="cell" x="{x}" y="{y}" '
                    f'width="{self.config.cell_width - 1}" '
                    f'height="{self.config.cell_height - 1}" '
                    f'fill="{color}" '
                    f'data-row="{i}" data-col="{j}" data-value="{value:.4f}"/>'
                )
                cells.append(cell)

                if self.config.show_values and self.config.cell_width >= 25:
                    text_color = "#fff" if norm_value < 0.5 else "#000"
                    text = (
                        f'<text class="value-text" x="{x + self.config.cell_width/2}" '
                        f'y="{y + self.config.cell_height/2 + 3}" '
                        f'text-anchor="middle" fill="{text_color}">'
                        f'{value:.2f}</text>'
                    )
                    cells.append(text)

        return "\n                ".join(cells)

    def _render_row_labels(self, labels: list[str], label_width: int, label_height: int) -> str:
        """Render row labels."""
        svg_parts = []
        for i, label in enumerate(labels):
            y = label_height + i * self.config.cell_height + self.config.cell_height / 2 + 4
            short_label = label[:10] + "..." if len(label) > 10 else label
            svg_parts.append(
                f'<text class="label" x="{label_width - 5}" y="{y}" '
                f'text-anchor="end">{short_label}</text>'
            )
        return "\n            ".join(svg_parts)

    def _render_col_labels(self, labels: list[str], label_width: int, label_height: int) -> str:
        """Render column labels."""
        svg_parts = []
        for j, label in enumerate(labels):
            x = label_width + j * self.config.cell_width + self.config.cell_width / 2
            short_label = label[:6] if len(label) > 6 else label
            svg_parts.append(
                f'<text class="label" x="{x}" y="{label_height - 5}" '
                f'text-anchor="middle" transform="rotate(-45 {x} {label_height - 5})">'
                f'{short_label}</text>'
            )
        return "\n            ".join(svg_parts)

    def _render_colorbar(
        self,
        vmin: float,
        vmax: float,
        grid_width: int,
        label_width: int,
        label_height: int
    ) -> str:
        """Render colorbar."""
        bar_width = 20
        bar_height = 150
        x = label_width + grid_width + 30
        y = label_height + 20

        # Gradient definition
        stops = self.COLORMAPS.get(self.config.colormap, self.COLORMAPS["viridis"])
        gradient_stops = "\n".join([
            f'<stop offset="{int((1-s[0])*100)}%" stop-color="{s[1]}"/>'
            for s in stops
        ])

        return f"""
            <defs>
                <linearGradient id="colorbar-gradient" x1="0%" y1="100%" x2="0%" y2="0%">
                    {gradient_stops}
                </linearGradient>
            </defs>
            <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}"
                  fill="url(#colorbar-gradient)" stroke="#333"/>
            <text class="label" x="{x + bar_width + 5}" y="{y + 10}">{vmax:.2f}</text>
            <text class="label" x="{x + bar_width + 5}" y="{y + bar_height/2 + 4}">{(vmin + vmax)/2:.2f}</text>
            <text class="label" x="{x + bar_width + 5}" y="{y + bar_height}">{vmin:.2f}</text>
"""


def render_activation_heatmap(
    activations: np.ndarray,
    feature_names: list[str] | None = None,
    token_names: list[str] | None = None,
    title: str = "Feature Activations",
    colormap: str = "viridis",
) -> str:
    """
    Convenience function to render feature activations.

    Args:
        activations: 2D array (features x tokens)
        feature_names: Names for each feature
        token_names: Names for each token position
        title: Chart title
        colormap: Color scheme to use

    Returns:
        HTML string
    """
    config = HeatmapConfig(colormap=colormap)
    heatmap = FeatureHeatmap(config)
    return heatmap.render(
        data=activations,
        row_labels=feature_names,
        col_labels=token_names,
        title=title,
    )


def render_layer_heatmap(
    layer_activations: dict[str, np.ndarray],
    title: str = "Layer Activations",
    colormap: str = "plasma",
) -> str:
    """
    Render heatmap comparing activations across layers.

    Args:
        layer_activations: Dict mapping layer names to activation vectors
        title: Chart title
        colormap: Color scheme

    Returns:
        HTML string
    """
    layer_names = list(layer_activations.keys())
    n_features = max(len(v) for v in layer_activations.values())

    # Build data matrix
    data = np.zeros((len(layer_names), n_features))
    for i, (name, acts) in enumerate(layer_activations.items()):
        data[i, :len(acts)] = acts

    config = HeatmapConfig(colormap=colormap, cell_width=15, cell_height=25)
    heatmap = FeatureHeatmap(config)
    return heatmap.render(
        data=data,
        row_labels=layer_names,
        col_labels=[f"F{i}" for i in range(n_features)],
        title=title,
    )
