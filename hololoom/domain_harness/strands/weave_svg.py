"""
Weave SVG Visualizer - Tensor → Visual Knots

Renders strand tensors as living SVG knots.
Tension, complexity, dependencies, retries, regressions
all influence curvature, thickness, color, and rotation.

The knot is the visual soul of the strand.
As the feature evolves, so does its knot.

Output formats:
- Individual SVG per feature
- Combined loom SVG with all knots
- Animated SVG for tension changes

Usage:
    from hololoom.domain_harness.strands.weave_svg import KnotRenderer, LoomRenderer
    
    # Single knot
    renderer = KnotRenderer()
    renderer.render_knot(strand_state, "knot_F001.svg")
    
    # Full loom
    loom_renderer = LoomRenderer()
    loom_renderer.render_loom(loom_state, "loom.svg")
"""

import math
from dataclasses import dataclass
from pathlib import Path

from hololoom.domain_harness.strands.tension_model import (
    TENSION_CRITICAL,
    TENSION_MAX,
    LoomState,
    StrandState,
)

# ============================================================================
# SVG Primitives (No external dependencies)
# ============================================================================

class SVGBuilder:
    """Simple SVG builder without external dependencies."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.elements = []
        self.defs = []

    def add_def(self, element: str):
        """Add to defs section."""
        self.defs.append(element)

    def add_element(self, element: str):
        """Add element to main body."""
        self.elements.append(element)

    def polyline(
        self,
        points: list[tuple[float, float]],
        stroke: str = "black",
        stroke_width: float = 2,
        fill: str = "none",
        opacity: float = 1.0
    ):
        """Add polyline element."""
        points_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.elements.append(
            f'<polyline points="{points_str}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" '
            f'fill="{fill}" opacity="{opacity}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )

    def circle(
        self,
        cx: float, cy: float, r: float,
        stroke: str = "black",
        stroke_width: float = 1,
        fill: str = "none"
    ):
        """Add circle element."""
        self.elements.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" fill="{fill}"/>'
        )

    def text(
        self,
        x: float, y: float, content: str,
        font_size: int = 12,
        fill: str = "black",
        anchor: str = "middle"
    ):
        """Add text element."""
        self.elements.append(
            f'<text x="{x:.2f}" y="{y:.2f}" '
            f'font-size="{font_size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-family="monospace">{content}</text>'
        )

    def rect(
        self,
        x: float, y: float, width: float, height: float,
        fill: str = "none",
        stroke: str = "black",
        stroke_width: float = 1,
        rx: float = 0
    ):
        """Add rectangle element."""
        self.elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" '
            f'width="{width:.2f}" height="{height:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'rx="{rx}"/>'
        )

    def group(self, transform: str = "") -> 'SVGGroup':
        """Start a group."""
        return SVGGroup(self, transform)

    def render(self) -> str:
        """Render complete SVG."""
        defs_content = "\n    ".join(self.defs) if self.defs else ""
        elements_content = "\n  ".join(self.elements)

        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     width="{self.width}" height="{self.height}"
     viewBox="0 0 {self.width} {self.height}">
  <defs>
    {defs_content}
  </defs>
  {elements_content}
</svg>'''

    def save(self, path: Path):
        """Save to file."""
        Path(path).write_text(self.render())


class SVGGroup:
    """Group context manager."""

    def __init__(self, builder: SVGBuilder, transform: str = ""):
        self.builder = builder
        self.transform = transform
        self.elements = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        transform_attr = f' transform="{self.transform}"' if self.transform else ""
        content = "\n    ".join(self.elements)
        self.builder.add_element(f'<g{transform_attr}>\n    {content}\n  </g>')


# ============================================================================
# Knot Geometry
# ============================================================================

def trefoil_knot(t: float, scale: float = 1.0) -> tuple[float, float]:
    """
    Parametric trefoil knot.
    t ranges from 0 to 2π
    """
    x = math.sin(t) + 2 * math.sin(2 * t)
    y = math.cos(t) - 2 * math.cos(2 * t)
    return (x * scale, y * scale)


def figure_eight_knot(t: float, scale: float = 1.0) -> tuple[float, float]:
    """
    Parametric figure-eight knot.
    """
    x = (2 + math.cos(2 * t)) * math.cos(3 * t)
    y = (2 + math.cos(2 * t)) * math.sin(3 * t)
    return (x * scale * 0.4, y * scale * 0.4)


def lissajous_knot(
    t: float,
    a: int = 3, b: int = 2,
    delta: float = math.pi / 2,
    scale: float = 1.0
) -> tuple[float, float]:
    """
    Lissajous curve (can form knots with right parameters).
    """
    x = math.sin(a * t + delta)
    y = math.sin(b * t)
    return (x * scale, y * scale)


def spiral_knot(
    t: float,
    turns: int = 3,
    scale: float = 1.0
) -> tuple[float, float]:
    """
    Spiral pattern.
    """
    r = t / (2 * math.pi * turns) * scale
    x = r * math.cos(t * turns)
    y = r * math.sin(t * turns)
    return (x, y)


# ============================================================================
# Color Mapping
# ============================================================================

def tension_to_hue(tension: float) -> int:
    """
    Map tension to HSL hue.
    Low tension (stable) = blue/green (180-120)
    High tension (critical) = red/orange (0-30)
    """
    # Normalize tension to 0-1
    normalized = min(1.0, tension / TENSION_MAX)
    # Map to hue: 180 (cyan) at 0, 0 (red) at 1
    hue = int(180 * (1 - normalized))
    return hue


def tension_to_saturation(tension: float) -> int:
    """Higher tension = more saturated."""
    normalized = min(1.0, tension / TENSION_MAX)
    return int(50 + normalized * 30)  # 50-80%


def tension_to_lightness(tension: float) -> int:
    """Critical tension = darker."""
    normalized = min(1.0, tension / TENSION_MAX)
    return int(60 - normalized * 20)  # 60-40%


def tension_to_color(tension: float) -> str:
    """Convert tension to HSL color string."""
    h = tension_to_hue(tension)
    s = tension_to_saturation(tension)
    l = tension_to_lightness(tension)
    return f"hsl({h}, {s}%, {l}%)"


# ============================================================================
# Knot Renderer
# ============================================================================

@dataclass
class KnotStyle:
    """Visual style for a knot."""
    stroke_color: str = "hsl(200, 70%, 60%)"
    stroke_width: float = 4.0
    fill: str = "none"
    opacity: float = 1.0
    glow: bool = False
    glow_color: str = "white"


class KnotRenderer:
    """
    Renders strand state as visual knot.
    
    The knot's appearance encodes:
    - Tension → color (blue=stable, red=critical)
    - Complexity → stroke width
    - Regressions → knot type (more complex for more regressions)
    - Consecutive fails → opacity reduction
    """

    def __init__(self, size: int = 200):
        self.size = size

    def strand_to_style(self, strand: StrandState) -> KnotStyle:
        """Convert strand state to visual style."""
        tension = strand.tension
        regressions = strand.regression_count
        fails = strand.consecutive_fails

        # Color from tension
        color = tension_to_color(tension)

        # Width from complexity (base + regression bonus)
        width = 3.0 + min(regressions, 5) * 0.5

        # Opacity from consecutive fails
        opacity = max(0.4, 1.0 - fails * 0.1)

        # Glow for critical tension
        glow = tension > TENSION_CRITICAL

        return KnotStyle(
            stroke_color=color,
            stroke_width=width,
            opacity=opacity,
            glow=glow,
            glow_color=tension_to_color(tension)
        )

    def select_knot_type(self, strand: StrandState) -> str:
        """Select knot complexity based on state."""
        if strand.regression_count >= 3:
            return "figure_eight"
        elif strand.consecutive_fails >= 5:
            return "lissajous"
        else:
            return "trefoil"

    def generate_knot_points(
        self,
        knot_type: str,
        center: tuple[float, float],
        scale: float,
        rotation: float = 0
    ) -> list[tuple[float, float]]:
        """Generate points for knot curve."""
        cx, cy = center
        points = []

        # Number of points
        n_points = 200

        for i in range(n_points + 1):
            t = (i / n_points) * 2 * math.pi

            if knot_type == "trefoil":
                x, y = trefoil_knot(t, scale)
            elif knot_type == "figure_eight":
                x, y = figure_eight_knot(t, scale)
            elif knot_type == "lissajous":
                x, y = lissajous_knot(t, scale=scale)
            elif knot_type == "spiral":
                x, y = spiral_knot(t, scale=scale)
            else:
                x, y = trefoil_knot(t, scale)

            # Apply rotation
            if rotation != 0:
                cos_r = math.cos(rotation)
                sin_r = math.sin(rotation)
                x, y = x * cos_r - y * sin_r, x * sin_r + y * cos_r

            points.append((x + cx, y + cy))

        return points

    def render_knot(
        self,
        strand: StrandState,
        output_path: Path | None = None,
        show_label: bool = True
    ) -> str:
        """
        Render a single strand as SVG knot.
        
        Returns SVG string and optionally saves to file.
        """
        svg = SVGBuilder(self.size, self.size)

        # Background
        svg.rect(0, 0, self.size, self.size, fill="#1a1a2e", stroke="none")

        # Style from strand
        style = self.strand_to_style(strand)
        knot_type = self.select_knot_type(strand)

        # Calculate scale from tension
        base_scale = 20 + strand.tension * 3

        # Rotation from last motion (if available)
        rotation = hash(str(strand.last_motion)) % 360 * (math.pi / 180) if strand.last_motion else 0

        # Generate knot points
        center = (self.size / 2, self.size / 2)
        points = self.generate_knot_points(knot_type, center, base_scale, rotation)

        # Add glow effect if critical
        if style.glow:
            svg.add_def('''
                <filter id="glow">
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feMerge>
                        <feMergeNode in="blur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            ''')
            # Draw glow layer
            svg.polyline(
                points,
                stroke=style.glow_color,
                stroke_width=style.stroke_width + 4,
                opacity=0.3
            )

        # Draw main knot
        svg.polyline(
            points,
            stroke=style.stroke_color,
            stroke_width=style.stroke_width,
            opacity=style.opacity
        )

        # Label
        if show_label:
            svg.text(
                self.size / 2, self.size - 15,
                strand.feature_id,
                font_size=14,
                fill="#ffffff"
            )
            svg.text(
                self.size / 2, self.size - 3,
                f"τ={strand.tension:.1f}",
                font_size=10,
                fill="#888888"
            )

        result = svg.render()

        if output_path:
            svg.save(output_path)

        return result


# ============================================================================
# Loom Renderer (All Knots)
# ============================================================================

class LoomRenderer:
    """
    Renders complete loom state as SVG grid of knots.
    """

    def __init__(self, knot_size: int = 150, padding: int = 20):
        self.knot_size = knot_size
        self.padding = padding
        self.knot_renderer = KnotRenderer(size=knot_size)

    def calculate_grid(self, n_items: int) -> tuple[int, int]:
        """Calculate grid dimensions for n items."""
        cols = math.ceil(math.sqrt(n_items))
        rows = math.ceil(n_items / cols)
        return cols, rows

    def render_loom(
        self,
        loom: LoomState,
        output_path: Path | None = None,
        title: str = "Loom State"
    ) -> str:
        """
        Render complete loom as grid of knots.
        """
        n_strands = len(loom.strands)
        if n_strands == 0:
            # Empty loom
            svg = SVGBuilder(400, 200)
            svg.rect(0, 0, 400, 200, fill="#1a1a2e")
            svg.text(200, 100, "Empty Loom", font_size=20, fill="#ffffff")
            result = svg.render()
            if output_path:
                svg.save(output_path)
            return result

        cols, rows = self.calculate_grid(n_strands)

        # Calculate total size
        total_width = cols * (self.knot_size + self.padding) + self.padding
        total_height = rows * (self.knot_size + self.padding) + self.padding + 60  # Extra for header

        svg = SVGBuilder(total_width, total_height)

        # Background
        svg.rect(0, 0, total_width, total_height, fill="#0f0f1a", stroke="none")

        # Header
        svg.text(
            total_width / 2, 25,
            title,
            font_size=18,
            fill="#ffffff"
        )
        svg.text(
            total_width / 2, 45,
            f"Avg Tension: {loom.average_tension():.2f} | Strands: {n_strands}",
            font_size=12,
            fill="#888888"
        )

        # Draw each knot
        sorted_strands = sorted(loom.strands.items())
        for i, (fid, strand) in enumerate(sorted_strands):
            col = i % cols
            row = i // cols

            x = self.padding + col * (self.knot_size + self.padding)
            y = 60 + self.padding + row * (self.knot_size + self.padding)

            # Get knot style
            style = self.knot_renderer.strand_to_style(strand)
            knot_type = self.knot_renderer.select_knot_type(strand)

            # Generate points
            base_scale = 15 + strand.tension * 2
            center = (x + self.knot_size / 2, y + self.knot_size / 2)
            points = self.knot_renderer.generate_knot_points(knot_type, center, base_scale)

            # Background for this knot
            svg.rect(
                x, y, self.knot_size, self.knot_size,
                fill="#1a1a2e",
                stroke="#333344",
                stroke_width=1,
                rx=5
            )

            # Draw knot
            svg.polyline(
                points,
                stroke=style.stroke_color,
                stroke_width=style.stroke_width,
                opacity=style.opacity
            )

            # Label
            svg.text(
                center[0], y + self.knot_size - 8,
                fid,
                font_size=11,
                fill="#ffffff"
            )

        result = svg.render()

        if output_path:
            svg.save(output_path)

        return result


# ============================================================================
# Convenience Functions
# ============================================================================

def render_strand_knot(strand: StrandState, output_path: Path) -> str:
    """Render a single strand to SVG file."""
    renderer = KnotRenderer()
    return renderer.render_knot(strand, output_path)


def render_loom_grid(loom: LoomState, output_path: Path, title: str = "Loom") -> str:
    """Render full loom to SVG file."""
    renderer = LoomRenderer()
    return renderer.render_loom(loom, output_path, title)


def tensor_to_knot_svg(
    tensor: list[float],
    feature_id: str,
    output_path: Path
) -> str:
    """
    Convert tensor [tension, complexity, deps, retries, regressions]
    to SVG knot.
    
    Compatibility with GPT handoff format.
    """
    tension, complexity, deps, retries, regressions = tensor

    # Create strand state from tensor
    strand = StrandState(feature_id=feature_id)
    strand.tension = float(tension)
    strand.regression_count = int(regressions)
    strand.consecutive_fails = int(retries)

    renderer = KnotRenderer()
    return renderer.render_knot(strand, output_path)


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    # Demo: create sample knots
    print("Generating sample knots...")

    # Low tension (stable)
    stable = StrandState(feature_id="F001")
    stable.tension = 1.5
    render_strand_knot(stable, Path("knot_stable.svg"))

    # Medium tension
    medium = StrandState(feature_id="F002")
    medium.tension = 5.0
    medium.consecutive_fails = 2
    render_strand_knot(medium, Path("knot_medium.svg"))

    # High tension (critical)
    critical = StrandState(feature_id="F003")
    critical.tension = 9.0
    critical.regression_count = 4
    render_strand_knot(critical, Path("knot_critical.svg"))

    # Full loom
    loom = LoomState()
    loom.strands["F001"] = stable
    loom.strands["F002"] = medium
    loom.strands["F003"] = critical

    render_loom_grid(loom, Path("loom_demo.svg"), "Demo Loom")

    print("Generated: knot_stable.svg, knot_medium.svg, knot_critical.svg, loom_demo.svg")
