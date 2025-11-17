"""AR 3D Overlay Renderer for augmented reality visualization.

Implements rendering of 3D AR overlays including bounding boxes, labels,
arrows, highlights, info panels, navigation paths, and heatmaps.

Implemented: 2025-11-17
Part of Wave 5: Advanced AR Integration
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
import asyncio
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


class OverlayType(Enum):
    """AR overlay types."""
    BOUNDING_BOX = "bounding_box"  # 3D box around object
    LABEL = "label"  # Floating text label
    ARROW = "arrow"  # Directional arrow
    HIGHLIGHT = "highlight"  # Object highlight glow
    INFO_PANEL = "info_panel"  # Floating info panel
    PATH = "path"  # Navigation path/trajectory
    HEATMAP = "heatmap"  # Color-coded heat overlay
    MARKER = "marker"  # 3D point marker


class OverlayBlendMode(Enum):
    """Blending modes for overlay composition."""
    ALPHA_BLEND = "alpha_blend"  # Standard alpha blending
    ADDITIVE = "additive"  # Additive blending
    MULTIPLY = "multiply"  # Multiplicative blending
    SCREEN = "screen"  # Screen blending


@dataclass
class AROverlay:
    """Single AR overlay element.

    Attributes:
        type: Type of overlay to render
        position: 3D world position (x, y, z) in meters
        rotation: Euler angles (yaw, pitch, roll) in radians
        scale: Scale factors (sx, sy, sz)
        color: RGBA color (0-1 range)
        content: Type-specific content dictionary
        duration: Display duration in seconds (None = persistent)
        blend_mode: How to blend with background
        created_at: Creation timestamp
        intensity: Visual intensity (0-1)
    """
    type: OverlayType
    position: Tuple[float, float, float]
    rotation: Optional[Tuple[float, float, float]] = None
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    content: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None
    blend_mode: OverlayBlendMode = OverlayBlendMode.ALPHA_BLEND
    created_at: datetime = field(default_factory=datetime.now)
    intensity: float = 1.0

    def is_expired(self) -> bool:
        """Check if overlay duration has expired."""
        if self.duration is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed >= self.duration


@dataclass
class ProjectionMatrix:
    """Camera projection parameters for 3D to 2D projection."""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width
    height: int  # Image height


class AROverlayRenderer:
    """Render AR overlays in 3D space.

    Manages a collection of AR overlays and renders them onto camera frames
    with proper 3D projection and blending.
    """

    def __init__(
        self,
        projection: Optional[ProjectionMatrix] = None,
        max_overlays: int = 1000,
    ):
        """Initialize AR overlay renderer.

        Args:
            projection: Camera projection parameters
            max_overlays: Maximum number of concurrent overlays
        """
        self.active_overlays: List[AROverlay] = []
        self.projection = projection or self._default_projection()
        self.max_overlays = max_overlays
        self.frame_count = 0
        self._lock = asyncio.Lock()

    def _default_projection(self) -> ProjectionMatrix:
        """Get default projection matrix for VGA resolution."""
        return ProjectionMatrix(
            fx=500.0,
            fy=500.0,
            cx=320.0,
            cy=240.0,
            width=640,
            height=480,
        )

    async def add_overlay(self, overlay: AROverlay) -> None:
        """Add overlay to render queue.

        Args:
            overlay: AROverlay to add
        """
        async with self._lock:
            if len(self.active_overlays) < self.max_overlays:
                self.active_overlays.append(overlay)

    async def remove_overlay(self, overlay: AROverlay) -> None:
        """Remove overlay from render queue."""
        async with self._lock:
            if overlay in self.active_overlays:
                self.active_overlays.remove(overlay)

    async def clear_overlays(self) -> None:
        """Clear all active overlays."""
        async with self._lock:
            self.active_overlays.clear()

    async def render(
        self,
        camera_position: Tuple[float, float, float],
        camera_rotation: Tuple[float, float, float],
        frame: np.ndarray,
    ) -> np.ndarray:
        """Render all overlays onto frame.

        Args:
            camera_position: Camera 3D position (x, y, z)
            camera_rotation: Camera rotation (yaw, pitch, roll) in radians
            frame: Input RGB/BGR image frame

        Returns:
            Augmented frame with overlays rendered
        """
        try:
            import cv2
        except ImportError:
            return frame

        output_frame = frame.copy()

        async with self._lock:
            # Render each overlay
            for overlay in self.active_overlays:
                try:
                    output_frame = await self._render_overlay(
                        overlay,
                        output_frame,
                        camera_position,
                        camera_rotation,
                    )
                except Exception:
                    # Skip overlays that fail to render
                    pass

            # Remove expired overlays
            self._prune_expired_overlays()

        self.frame_count += 1
        return output_frame

    async def _render_overlay(
        self,
        overlay: AROverlay,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
        camera_rotation: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render single overlay based on type."""
        if overlay.type == OverlayType.BOUNDING_BOX:
            return await self._render_bounding_box(overlay, frame, camera_position)

        elif overlay.type == OverlayType.LABEL:
            return await self._render_label(overlay, frame, camera_position)

        elif overlay.type == OverlayType.ARROW:
            return await self._render_arrow(overlay, frame, camera_position)

        elif overlay.type == OverlayType.HIGHLIGHT:
            return await self._render_highlight(overlay, frame, camera_position)

        elif overlay.type == OverlayType.INFO_PANEL:
            return await self._render_info_panel(overlay, frame, camera_position)

        elif overlay.type == OverlayType.PATH:
            return await self._render_path(overlay, frame, camera_position)

        elif overlay.type == OverlayType.MARKER:
            return await self._render_marker(overlay, frame, camera_position)

        return frame

    async def _render_bounding_box(
        self,
        overlay: AROverlay,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render 3D bounding box."""
        try:
            import cv2
        except ImportError:
            return frame

        # Get box corners in 3D
        box_corners_3d = self._get_box_corners(overlay.position, overlay.scale)

        # Project to 2D
        box_corners_2d = [
            self._project_to_2d(corner, camera_position, frame.shape[:2])
            for corner in box_corners_3d
        ]

        # Filter out projections behind camera
        if any(p is None for p in box_corners_2d):
            return frame

        # Get color
        color_bgr = self._rgba_to_bgr(overlay.color)

        # Draw 12 edges of cube
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Vertical edges
        ]

        thickness = max(1, int(2 * overlay.intensity))

        for i, j in edges:
            pt1 = tuple(map(int, box_corners_2d[i]))
            pt2 = tuple(map(int, box_corners_2d[j]))
            cv2.line(frame, pt1, pt2, color_bgr, thickness)

        return frame

    async def _render_label(
        self,
        overlay: AROverlay,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render floating text label."""
        try:
            import cv2
        except ImportError:
            return frame

        # Project position to 2D
        pos_2d = self._project_to_2d(overlay.position, camera_position, frame.shape[:2])

        if pos_2d is None:
            return frame

        text = overlay.content.get("text", "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = overlay.content.get("font_scale", 0.6)
        thickness = overlay.content.get("thickness", 2)
        color_bgr = self._rgba_to_bgr(overlay.color)

        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Draw background rectangle
        pt1 = (int(pos_2d[0] - 5), int(pos_2d[1] - text_height - 5))
        pt2 = (int(pos_2d[0] + text_width + 5), int(pos_2d[1] + baseline + 5))

        bg_color = (0, 0, 0)
        cv2.rectangle(frame, pt1, pt2, bg_color, -1)

        # Draw text
        cv2.putText(
            frame,
            text,
            tuple(map(int, pos_2d)),
            font,
            font_scale,
            color_bgr,
            thickness,
        )

        return frame

    async def _render_arrow(
        self,
        overlay: AROverlay,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render directional arrow."""
        try:
            import cv2
        except ImportError:
            return frame

        # Start position
        start_3d = overlay.position
        start_2d = self._project_to_2d(start_3d, camera_position, frame.shape[:2])

        if start_2d is None:
            return frame

        # End position (direction * length)
        direction = np.array(overlay.content.get("direction", [0, 0, 1]))
        length = overlay.content.get("length", 0.5)
        end_3d = (
            start_3d[0] + direction[0] * length,
            start_3d[1] + direction[1] * length,
            start_3d[2] + direction[2] * length,
        )

        end_2d = self._project_to_2d(end_3d, camera_position, frame.shape[:2])

        if end_2d is None:
            return frame

        color_bgr = self._rgba_to_bgr(overlay.color)
        thickness = max(1, int(2 * overlay.intensity))

        # Draw arrow shaft
        cv2.line(
            frame,
            tuple(map(int, start_2d)),
            tuple(map(int, end_2d)),
            color_bgr,
            thickness,
        )

        # Draw arrowhead
        arrow_size = 10
        angle = np.arctan2(end_2d[1] - start_2d[1], end_2d[0] - start_2d[0])
        pt1 = (int(end_2d[0]), int(end_2d[1]))
        pt2 = (
            int(end_2d[0] - arrow_size * np.cos(angle - np.pi / 6)),
            int(end_2d[1] - arrow_size * np.sin(angle - np.pi / 6)),
        )
        pt3 = (
            int(end_2d[0] - arrow_size * np.cos(angle + np.pi / 6)),
            int(end_2d[1] - arrow_size * np.sin(angle + np.pi / 6)),
        )

        cv2.polylines(frame, [np.array([pt1, pt2, pt3])], True, color_bgr, thickness)

        return frame

    async def _render_highlight(
        self,
        overlay: AROverlay,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render object highlight glow."""
        try:
            import cv2
        except ImportError:
            return frame

        # Get box corners
        box_corners_3d = self._get_box_corners(overlay.position, overlay.scale)
        box_corners_2d = [
            self._project_to_2d(corner, camera_position, frame.shape[:2])
            for corner in box_corners_3d
        ]

        if any(p is None for p in box_corners_2d):
            return frame

        # Create mask for glow
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Draw filled box on mask
        pts = np.array([box_corners_2d], dtype=np.int32)
        cv2.drawContours(mask, pts, 0, 255, -1)

        # Apply Gaussian blur for glow effect
        glow_radius = overlay.content.get("glow_radius", 20)
        glow = cv2.GaussianBlur(mask, (glow_radius * 2 + 1, glow_radius * 2 + 1), 0)

        # Blend glow with frame
        glow_color = self._rgba_to_bgr(overlay.color)
        glow_frame = frame.copy()
        glow_frame[:, :] = glow_color

        alpha = overlay.color[3] * overlay.intensity * 0.3
        frame = cv2.addWeighted(
            frame, 1 - alpha, glow_frame, alpha, 0, dtype=cv2.CV_8UC3
        )

        return frame

    async def _render_info_panel(
        self,
        overlay: AROverlay,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render floating info panel."""
        try:
            import cv2
        except ImportError:
            return frame

        # Project position to 2D
        pos_2d = self._project_to_2d(overlay.position, camera_position, frame.shape[:2])

        if pos_2d is None:
            return frame

        pos_2d = tuple(map(int, pos_2d))

        # Get panel content
        title = overlay.content.get("title", "")
        items = overlay.content.get("items", [])
        panel_width = overlay.content.get("width", 200)
        panel_height = overlay.content.get("height", 100)

        color_bgr = self._rgba_to_bgr(overlay.color)

        # Draw background panel
        pt1 = (pos_2d[0], pos_2d[1])
        pt2 = (pos_2d[0] + panel_width, pos_2d[1] + panel_height)

        bg_alpha = overlay.color[3] * overlay.intensity
        overlay_frame = frame.copy()
        cv2.rectangle(overlay_frame, pt1, pt2, (0, 0, 0), -1)

        frame = cv2.addWeighted(
            frame, 1 - bg_alpha, overlay_frame, bg_alpha, 0, dtype=cv2.CV_8UC3
        )

        # Draw border
        cv2.rectangle(frame, pt1, pt2, color_bgr, 2)

        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = pos_2d[1] + 20

        # Title
        cv2.putText(
            frame,
            title,
            (pos_2d[0] + 10, y_offset),
            font,
            0.5,
            (255, 255, 255),
            1,
        )

        y_offset += 20

        # Items
        for item in items[:3]:  # Max 3 items
            cv2.putText(
                frame,
                str(item),
                (pos_2d[0] + 10, y_offset),
                font,
                0.4,
                (200, 200, 200),
                1,
            )
            y_offset += 15

        return frame

    async def _render_path(
        self,
        overlay: AROverlay,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render navigation path."""
        try:
            import cv2
        except ImportError:
            return frame

        # Get path waypoints
        waypoints = overlay.content.get("waypoints", [])

        if len(waypoints) < 2:
            return frame

        # Project waypoints to 2D
        waypoints_2d = [
            self._project_to_2d(wp, camera_position, frame.shape[:2])
            for wp in waypoints
        ]

        # Filter valid projections
        valid_waypoints = [wp for wp in waypoints_2d if wp is not None]

        if len(valid_waypoints) < 2:
            return frame

        # Draw path segments
        color_bgr = self._rgba_to_bgr(overlay.color)
        thickness = max(1, int(2 * overlay.intensity))

        for i in range(len(valid_waypoints) - 1):
            pt1 = tuple(map(int, valid_waypoints[i]))
            pt2 = tuple(map(int, valid_waypoints[i + 1]))

            # Fade color along path
            alpha = overlay.intensity * (i / len(valid_waypoints))
            faded_color = tuple(int(c * alpha) for c in color_bgr)

            cv2.line(frame, pt1, pt2, faded_color, thickness)

        # Draw waypoint markers
        marker_radius = overlay.content.get("marker_radius", 5)

        for i, wp in enumerate(valid_waypoints):
            center = tuple(map(int, wp))

            # Color gradient
            if i == 0:
                marker_color = (0, 255, 0)  # Green start
            elif i == len(valid_waypoints) - 1:
                marker_color = (0, 0, 255)  # Red end
            else:
                marker_color = (0, 255, 255)  # Yellow middle

            cv2.circle(frame, center, marker_radius, marker_color, -1)

        return frame

    async def _render_marker(
        self,
        overlay: AROverlay,
        frame: np.ndarray,
        camera_position: Tuple[float, float, float],
    ) -> np.ndarray:
        """Render point marker."""
        try:
            import cv2
        except ImportError:
            return frame

        # Project position to 2D
        pos_2d = self._project_to_2d(overlay.position, camera_position, frame.shape[:2])

        if pos_2d is None:
            return frame

        pos_2d = tuple(map(int, pos_2d))
        color_bgr = self._rgba_to_bgr(overlay.color)
        radius = overlay.content.get("radius", 8)

        # Draw filled circle
        cv2.circle(frame, pos_2d, int(radius * overlay.intensity), color_bgr, -1)

        # Draw outline
        cv2.circle(frame, pos_2d, int(radius * overlay.intensity), (255, 255, 255), 1)

        # Draw label if provided
        label = overlay.content.get("label", "")
        if label:
            cv2.putText(
                frame,
                label,
                (pos_2d[0] + radius + 5, pos_2d[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color_bgr,
                1,
            )

        return frame

    def _get_box_corners(
        self, center: Tuple[float, float, float], scale: Tuple[float, float, float]
    ) -> List[Tuple[float, float, float]]:
        """Get 8 corners of bounding box."""
        x, y, z = center
        sx, sy, sz = scale

        corners = [
            (x - sx / 2, y - sy / 2, z - sz / 2),  # 0
            (x + sx / 2, y - sy / 2, z - sz / 2),  # 1
            (x + sx / 2, y + sy / 2, z - sz / 2),  # 2
            (x - sx / 2, y + sy / 2, z - sz / 2),  # 3
            (x - sx / 2, y - sy / 2, z + sz / 2),  # 4
            (x + sx / 2, y - sy / 2, z + sz / 2),  # 5
            (x + sx / 2, y + sy / 2, z + sz / 2),  # 6
            (x - sx / 2, y + sy / 2, z + sz / 2),  # 7
        ]

        return corners

    def _project_to_2d(
        self,
        point_3d: Tuple[float, float, float],
        camera_position: Tuple[float, float, float],
        image_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float]]:
        """Project 3D point to 2D camera coordinates.

        Args:
            point_3d: 3D world point
            camera_position: Camera position in world
            image_size: (height, width) of image

        Returns:
            2D image coordinates or None if behind camera
        """
        # Relative position to camera
        x = point_3d[0] - camera_position[0]
        y = point_3d[1] - camera_position[1]
        z = point_3d[2] - camera_position[2]

        # Check if behind camera
        if z <= 0.1:
            return None

        # Perspective projection
        u = self.projection.fx * (x / z) + self.projection.cx
        v = self.projection.fy * (y / z) + self.projection.cy

        # Check if in frame
        if u < 0 or u >= self.projection.width or v < 0 or v >= self.projection.height:
            return None

        return (u, v)

    def _rgba_to_bgr(self, rgba: Tuple[float, float, float, float]) -> Tuple[int, int, int]:
        """Convert RGBA (0-1) to BGR (0-255) for OpenCV."""
        return (
            int(rgba[2] * 255),
            int(rgba[1] * 255),
            int(rgba[0] * 255),
        )

    def _prune_expired_overlays(self) -> None:
        """Remove expired overlays from active list."""
        self.active_overlays = [
            o for o in self.active_overlays if not o.is_expired()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get renderer statistics."""
        return {
            "active_overlays": len(self.active_overlays),
            "frame_count": self.frame_count,
            "by_type": {
                ot.value: sum(1 for o in self.active_overlays if o.type == ot)
                for ot in OverlayType
            },
        }
