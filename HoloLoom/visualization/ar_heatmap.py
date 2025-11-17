"""AR Heatmap Overlay Renderer for augmented reality.

Render heatmaps for activity, temperature, intensity, and other
spatial data visualizations in AR space.

Implemented: 2025-11-17
Part of Wave 5: Advanced AR Integration
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
import asyncio


class ColormapType(Enum):
    """Colormap types for heatmaps."""
    HOT = "hot"  # Black -> Red -> Yellow -> White
    COOL = "cool"  # Black -> Blue -> Cyan -> White
    VIRIDIS = "viridis"  # Blue -> Green -> Yellow
    PLASMA = "plasma"  # Purple -> Orange -> Yellow
    JET = "jet"  # Blue -> Cyan -> Green -> Yellow -> Red
    TURBO = "turbo"  # Blue -> Cyan -> Green -> Yellow -> Red -> Magenta
    INFERNO = "inferno"  # Black -> Purple -> Orange -> Yellow
    GRAYSCALE = "grayscale"  # Black -> White


@dataclass
class Heatmap:
    """Heatmap data for AR rendering.

    Attributes:
        grid: 2D numpy array of values (normalized 0-1 or raw values)
        position: 3D anchor point (x, y, z) in world space
        size: Physical size in meters (width, height)
        colormap: Colormap type
        opacity: Overall opacity (0-1)
        intensity: Visual intensity multiplier
        label: Heatmap label
    """
    grid: np.ndarray
    position: Tuple[float, float, float] = (0, 0, 1)
    size: Tuple[float, float] = (1.0, 1.0)
    colormap: ColormapType = ColormapType.HOT
    opacity: float = 0.7
    intensity: float = 1.0
    label: str = ""
    auto_normalize: bool = True


class ARHeatmapRenderer:
    """Render heatmaps in AR space.

    Supports multiple colormaps, perspective projection,
    and real-time heatmap updates.
    """

    def __init__(self):
        """Initialize AR heatmap renderer."""
        self.colormaps = self._init_colormaps()
        self._lock = asyncio.Lock()

    def _init_colormaps(self) -> Dict[ColormapType, np.ndarray]:
        """Initialize colormap lookup tables."""
        colormaps = {}

        # Hot colormap: black -> red -> yellow -> white
        hot = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 85:
                # Black to red
                hot[i] = (0, 0, int(255 * (i / 85)))
            elif i < 170:
                # Red to yellow
                hot[i] = (0, int(255 * ((i - 85) / 85)), 255)
            else:
                # Yellow to white
                hot[i] = (int(255 * ((i - 170) / 85)), 255, 255)
        colormaps[ColormapType.HOT] = hot

        # Cool colormap: black -> blue -> cyan -> white
        cool = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 85:
                # Black to blue
                cool[i] = (int(255 * (i / 85)), 0, 0)
            elif i < 170:
                # Blue to cyan
                cool[i] = (255, int(255 * ((i - 85) / 85)), 0)
            else:
                # Cyan to white
                cool[i] = (255, 255, int(255 * ((i - 170) / 85)))
        colormaps[ColormapType.COOL] = cool

        # Viridis colormap: blue -> green -> yellow
        viridis = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 85:
                # Blue to green
                viridis[i] = (int(255 * (i / 85)), int(100 + 155 * (i / 85)), 50)
            elif i < 170:
                # Green to yellow
                viridis[i] = (0, int(255 * (1 - (i - 85) / 85)), int(50 + 205 * ((i - 85) / 85)))
            else:
                # Yellow to bright
                viridis[i] = (int(255 * ((i - 170) / 85)), 255, int(255 * ((i - 170) / 85)))
        colormaps[ColormapType.VIRIDIS] = viridis

        # Plasma colormap: purple -> orange -> yellow
        plasma = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 85:
                # Purple to magenta
                plasma[i] = (int(100 + 155 * (i / 85)), 0, int(150 * (i / 85)))
            elif i < 170:
                # Magenta to orange
                plasma[i] = (255, int(100 * ((i - 85) / 85)), int(150 * (1 - (i - 85) / 85)))
            else:
                # Orange to yellow
                plasma[i] = (255, int(100 + 155 * ((i - 170) / 85)), 0)
        colormaps[ColormapType.PLASMA] = plasma

        # Jet colormap: blue -> cyan -> green -> yellow -> red
        jet = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 51:
                # Blue
                jet[i] = (int(255 * (i / 51)), 0, 255)
            elif i < 102:
                # Cyan
                jet[i] = (int(255 * ((102 - i) / 51)), 255, 255)
            elif i < 153:
                # Green
                jet[i] = (0, 255, int(255 * ((153 - i) / 51)))
            elif i < 204:
                # Yellow
                jet[i] = (0, int(255 * ((204 - i) / 51)), 0)
            else:
                # Red
                jet[i] = (int(255 * ((i - 204) / 51)), 0, 0)
        colormaps[ColormapType.JET] = jet

        # Turbo colormap: blue -> cyan -> green -> yellow -> orange -> red
        turbo = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 43:
                turbo[i] = (0, int(100 + 155 * (i / 43)), 255)
            elif i < 85:
                turbo[i] = (0, 255, int(255 * ((85 - i) / 42)))
            elif i < 127:
                turbo[i] = (int(255 * ((i - 85) / 42)), 255, 0)
            elif i < 170:
                turbo[i] = (255, int(255 * ((170 - i) / 43)), 0)
            elif i < 212:
                turbo[i] = (255, int(100 * ((212 - i) / 42)), 0)
            else:
                turbo[i] = (255, 0, int(100 * ((i - 212) / 44)))
        colormaps[ColormapType.TURBO] = turbo

        # Inferno colormap: black -> purple -> orange -> yellow
        inferno = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 85:
                inferno[i] = (int(150 * (i / 85)), 0, int(100 * (i / 85)))
            elif i < 170:
                inferno[i] = (int(150 + 105 * ((i - 85) / 85)), int(100 * ((i - 85) / 85)), 0)
            else:
                inferno[i] = (255, int(100 + 155 * ((i - 170) / 85)), 0)
        colormaps[ColormapType.INFERNO] = inferno

        # Grayscale colormap
        grayscale = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            grayscale[i] = (i, i, i)
        colormaps[ColormapType.GRAYSCALE] = grayscale

        return colormaps

    async def render(
        self,
        heatmap: Heatmap,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
        camera_rotation: Optional[Tuple[float, float, float]] = None,
    ) -> np.ndarray:
        """Render heatmap overlay onto frame.

        Args:
            heatmap: Heatmap to render
            frame: Input image frame (BGR)
            camera_position: Camera 3D position
            camera_rotation: Camera rotation (optional)

        Returns:
            Frame with heatmap overlay
        """
        try:
            import cv2
        except ImportError:
            return frame

        # Normalize heatmap data if needed
        grid = heatmap.grid.copy()

        if heatmap.auto_normalize:
            if grid.size > 0:
                grid_min = grid.min()
                grid_max = grid.max()
                if grid_max > grid_min:
                    grid = (grid - grid_min) / (grid_max - grid_min)
        else:
            # Clip to 0-1 range
            grid = np.clip(grid, 0, 1)

        # Apply colormap
        heatmap_colored = self._apply_colormap(grid, heatmap.colormap, heatmap.intensity)

        # Project heatmap to frame
        output = await self._project_heatmap_to_frame(
            frame,
            heatmap_colored,
            heatmap.position,
            camera_position,
            heatmap.size,
            heatmap.opacity,
        )

        # Add label if provided
        if heatmap.label:
            try:
                import cv2

                cv2.putText(
                    output,
                    heatmap.label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            except Exception:
                pass

        return output

    def _apply_colormap(
        self, grid: np.ndarray, colormap_type: ColormapType, intensity: float
    ) -> np.ndarray:
        """Apply colormap to grid.

        Args:
            grid: Normalized grid (0-1)
            colormap_type: Type of colormap
            intensity: Intensity multiplier (0-1)

        Returns:
            BGR image array
        """
        try:
            import cv2
        except ImportError:
            return np.zeros((*grid.shape, 3), dtype=np.uint8)

        # Resize grid if needed
        if grid.shape[0] == 0 or grid.shape[1] == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # Normalize to 0-255
        grid_8u = (np.clip(grid, 0, 1) * 255).astype(np.uint8)

        # Get colormap lookup table
        colormap_lut = self.colormaps.get(colormap_type, self.colormaps[ColormapType.HOT])

        # Apply LUT
        colored = cv2.LUT(grid_8u, colormap_lut)

        # Apply intensity
        if intensity < 1.0:
            colored = (colored * intensity).astype(np.uint8)

        return colored

    async def _project_heatmap_to_frame(
        self,
        frame: np.ndarray,
        heatmap_img: np.ndarray,
        position_3d: Tuple[float, float, float],
        camera_position: Tuple[float, float, float],
        size_3d: Tuple[float, float],
        opacity: float,
    ) -> np.ndarray:
        """Project heatmap onto frame with blending.

        Simplified projection for MVP - future: full perspective transform.
        """
        try:
            import cv2
        except ImportError:
            return frame

        output = frame.copy()

        # Determine screen placement
        # For MVP: place at center of screen
        screen_center_x = frame.shape[1] // 2
        screen_center_y = frame.shape[0] // 2

        # Heatmap display size (pixels)
        hm_width = min(heatmap_img.shape[1], frame.shape[1] // 2)
        hm_height = min(heatmap_img.shape[0], frame.shape[0] // 2)

        # Resize heatmap if needed
        if heatmap_img.shape != (hm_height, hm_width):
            heatmap_resized = cv2.resize(heatmap_img, (hm_width, hm_height))
        else:
            heatmap_resized = heatmap_img

        # Calculate placement
        x1 = max(0, screen_center_x - hm_width // 2)
        y1 = max(0, screen_center_y - hm_height // 2)
        x2 = min(output.shape[1], x1 + hm_width)
        y2 = min(output.shape[0], y1 + hm_height)

        # Adjust if out of bounds
        cw = x2 - x1
        ch = y2 - y1

        if cw > 0 and ch > 0 and heatmap_resized.shape[0] > 0 and heatmap_resized.shape[1] > 0:
            # Resize to fit bounds
            heatmap_patch = cv2.resize(heatmap_resized, (cw, ch))

            # Blend with frame using opacity
            alpha = opacity
            output[y1:y2, x1:x2] = cv2.addWeighted(
                output[y1:y2, x1:x2], 1.0 - alpha, heatmap_patch, alpha, 0
            )

        return output

    async def render_multiple(
        self,
        heatmaps: List[Heatmap],
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render multiple heatmaps (composited together).

        Args:
            heatmaps: List of heatmaps to render
            frame: Input frame
            camera_position: Camera position

        Returns:
            Frame with all heatmaps rendered
        """
        output = frame.copy()

        for heatmap in heatmaps:
            output = await self.render(
                heatmap,
                output,
                camera_position,
            )

        return output

    async def create_difference_heatmap(
        self,
        grid1: np.ndarray,
        grid2: np.ndarray,
        position: Tuple[float, float, float],
    ) -> Heatmap:
        """Create heatmap showing difference between two grids.

        Args:
            grid1: First grid
            grid2: Second grid (same shape)
            position: 3D position for heatmap

        Returns:
            Heatmap showing absolute difference
        """
        if grid1.shape != grid2.shape:
            raise ValueError("Grids must have same shape")

        diff = np.abs(grid1 - grid2)

        return Heatmap(
            grid=diff,
            position=position,
            colormap=ColormapType.JET,
            label="Difference",
            auto_normalize=True,
        )

    async def create_gradient_heatmap(
        self,
        grid: np.ndarray,
        position: Tuple[float, float, float],
    ) -> Heatmap:
        """Create heatmap showing gradient magnitude.

        Args:
            grid: Input grid
            position: 3D position

        Returns:
            Heatmap showing spatial gradient
        """
        try:
            # Compute gradient magnitude
            gy, gx = np.gradient(grid)
            gradient = np.sqrt(gx**2 + gy**2)

            return Heatmap(
                grid=gradient,
                position=position,
                colormap=ColormapType.PLASMA,
                label="Gradient",
                auto_normalize=True,
            )
        except Exception:
            # Fallback to original grid
            return Heatmap(
                grid=grid,
                position=position,
                colormap=ColormapType.HOT,
            )


class HeatmapAnimator:
    """Animate heatmaps for real-time updates."""

    def __init__(self, update_interval: float = 0.1):
        """Initialize heatmap animator.

        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.heatmaps: Dict[str, Heatmap] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def register_heatmap(self, hm_id: str, heatmap: Heatmap) -> None:
        """Register heatmap for animation."""
        async with self._lock:
            self.heatmaps[hm_id] = heatmap

    async def unregister_heatmap(self, hm_id: str) -> None:
        """Unregister heatmap from animation."""
        async with self._lock:
            if hm_id in self.heatmaps:
                del self.heatmaps[hm_id]
            if hm_id in self._tasks:
                self._tasks[hm_id].cancel()
                del self._tasks[hm_id]

    async def start_animation(
        self,
        hm_id: str,
        update_func,
    ) -> None:
        """Start animating heatmap with update function.

        Args:
            hm_id: Heatmap ID
            update_func: Async function that returns updated grid
        """
        async def animate():
            while True:
                try:
                    async with self._lock:
                        if hm_id in self.heatmaps:
                            grid = await update_func()
                            self.heatmaps[hm_id].grid = grid
                    await asyncio.sleep(self.update_interval)
                except asyncio.CancelledError:
                    break
                except Exception:
                    await asyncio.sleep(self.update_interval)

        async with self._lock:
            if hm_id in self._tasks:
                self._tasks[hm_id].cancel()
            self._tasks[hm_id] = asyncio.create_task(animate())

    async def stop_animation(self, hm_id: str) -> None:
        """Stop animating heatmap."""
        async with self._lock:
            if hm_id in self._tasks:
                self._tasks[hm_id].cancel()
                del self._tasks[hm_id]

    async def get_heatmap(self, hm_id: str) -> Optional[Heatmap]:
        """Get current heatmap state."""
        async with self._lock:
            return self.heatmaps.get(hm_id)
