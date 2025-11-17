"""Demo: AR 3D Overlay Rendering

Demonstrates all 7 overlay types with real-time rendering:
- Bounding boxes (3D object detection)
- Text labels
- Directional arrows
- Object highlights with glow
- Info panels (HUD-style)
- Navigation paths
- Point markers

Implemented: 2025-11-17
Run: PYTHONPATH=. python demos/demo_ar_overlays.py
"""

import asyncio
import numpy as np
from datetime import datetime

from HoloLoom.visualization.ar_overlay import (
    AROverlay,
    AROverlayRenderer,
    OverlayType,
)


async def create_demo_overlays() -> list:
    """Create a collection of demo overlays."""
    overlays = []

    # 1. Bounding Box - Object Detection
    overlays.append(
        AROverlay(
            type=OverlayType.BOUNDING_BOX,
            position=(0.5, 0.3, 2.0),
            scale=(0.4, 0.3, 0.5),
            color=(1.0, 0.0, 0.0, 1.0),  # Red
            content={"label": "Person"},
        )
    )

    # 2. Text Label
    overlays.append(
        AROverlay(
            type=OverlayType.LABEL,
            position=(0.5, 0.1, 2.0),
            content={"text": "Distance: 2.0m", "font_scale": 0.7, "thickness": 2},
            color=(1.0, 1.0, 1.0, 1.0),  # White
        )
    )

    # 3. Directional Arrow
    overlays.append(
        AROverlay(
            type=OverlayType.ARROW,
            position=(0.0, 0.0, 1.5),
            content={"direction": [1, 0, 0], "length": 0.5},
            color=(0.0, 1.0, 0.0, 1.0),  # Green
        )
    )

    # 4. Object Highlight
    overlays.append(
        AROverlay(
            type=OverlayType.HIGHLIGHT,
            position=(0.0, 0.3, 2.0),
            scale=(0.3, 0.3, 0.3),
            content={"glow_radius": 25},
            color=(1.0, 1.0, 0.0, 0.8),  # Yellow
            intensity=0.7,
        )
    )

    # 5. Info Panel (HUD)
    overlays.append(
        AROverlay(
            type=OverlayType.INFO_PANEL,
            position=(0.5, 0.0, 1.5),
            content={
                "title": "System Status",
                "items": [
                    "FPS: 60",
                    "Latency: 12ms",
                    "Objects: 3",
                ],
                "width": 220,
                "height": 120,
            },
            color=(0.2, 0.5, 0.8, 0.9),  # Blue
        )
    )

    # 6. Navigation Path
    overlays.append(
        AROverlay(
            type=OverlayType.PATH,
            position=(0.0, 0.0, 2.0),
            content={
                "waypoints": [
                    (0.0, 0.0, 2.0),
                    (0.3, -0.2, 2.0),
                    (0.6, -0.1, 2.0),
                    (0.9, 0.1, 2.0),
                    (1.0, 0.5, 2.0),
                ],
                "marker_radius": 6,
            },
            color=(0.0, 1.0, 1.0, 1.0),  # Cyan
        )
    )

    # 7. Point Markers
    marker_positions = [
        (-0.5, 0.5, 1.5),
        (-0.3, 0.3, 1.7),
        (-0.1, 0.1, 1.9),
    ]

    for i, pos in enumerate(marker_positions):
        overlays.append(
            AROverlay(
                type=OverlayType.MARKER,
                position=pos,
                content={"radius": 6, "label": f"P{i+1}"},
                color=(1.0, 0.5, 0.0, 1.0),  # Orange
            )
        )

    return overlays


async def create_dummy_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a dummy camera frame with gradual gradient background."""
    frame = np.ones((height, width, 3), dtype=np.uint8)

    # Create gradient background (simulates camera view)
    for y in range(height):
        intensity = int(200 + 55 * (y / height))
        frame[y, :] = intensity

    # Add some features (simulating objects in view)
    # Horizontal line across middle
    cv2.line(frame, (0, height // 2), (width, height // 2), (100, 100, 100), 2)

    # Vertical line
    cv2.line(frame, (width // 2, 0), (width // 2, height), (100, 100, 100), 2)

    return frame


async def main():
    """Main demo function."""
    print("=" * 70)
    print("AR OVERLAY RENDERING DEMO")
    print("=" * 70)
    print()

    # Create renderer
    renderer = AROverlayRenderer()
    print(f"Initialized AR Overlay Renderer")
    print(f"  Projection: 640x480, fx=500, fy=500")
    print()

    # Create demo overlays
    overlays = await create_demo_overlays()
    print(f"Created {len(overlays)} demo overlays:")

    overlay_types = {}
    for overlay in overlays:
        type_name = overlay.type.value
        overlay_types[type_name] = overlay_types.get(type_name, 0) + 1

    for type_name, count in sorted(overlay_types.items()):
        print(f"  - {type_name}: {count}")
    print()

    # Add overlays to renderer
    for overlay in overlays:
        await renderer.add_overlay(overlay)

    print(f"Added {len(overlays)} overlays to render queue")
    print(f"Active overlays: {len(renderer.active_overlays)}")
    print()

    # Create dummy frame
    dummy_frame = await create_dummy_frame()
    print(f"Created dummy frame: {dummy_frame.shape}")
    print()

    # Simulate camera positions
    camera_positions = [
        (0.0, 0.0, 0.0),
        (-0.1, 0.0, -0.2),
        (0.1, -0.05, -0.1),
    ]

    camera_rotation = (0.0, 0.0, 0.0)

    print("Rendering overlays from different camera positions...")
    print()

    for i, cam_pos in enumerate(camera_positions, 1):
        print(f"Frame {i}: Camera at {cam_pos}")

        # Render frame
        result_frame = await renderer.render(cam_pos, camera_rotation, dummy_frame)

        print(f"  Output shape: {result_frame.shape}")
        print(f"  Pixel value range: [{result_frame.min()}, {result_frame.max()}]")

        # Get statistics
        stats = renderer.get_stats()
        print(f"  Active overlays: {stats['active_overlays']}")
        print(f"  Frame count: {stats['frame_count']}")

        print()

    # Test overlay expiration
    print("Testing overlay expiration...")
    temp_overlay = AROverlay(
        type=OverlayType.MARKER,
        position=(0.5, 0.5, 2.0),
        duration=0.1,  # Expires after 0.1 seconds
    )
    await renderer.add_overlay(temp_overlay)
    print(f"  Added temporary overlay (duration: 0.1s)")
    print(f"  Is expired? {temp_overlay.is_expired()}")

    import time

    time.sleep(0.15)
    print(f"  Is expired (after 0.15s)? {temp_overlay.is_expired()}")
    print()

    # Test overlay removal
    print("Testing overlay management...")
    initial_count = len(renderer.active_overlays)
    first_overlay = renderer.active_overlays[0]
    await renderer.remove_overlay(first_overlay)
    new_count = len(renderer.active_overlays)
    print(f"  Removed one overlay: {initial_count} -> {new_count}")
    print()

    # Clear all overlays
    await renderer.clear_overlays()
    print(f"Cleared all overlays")
    print(f"Active overlays: {len(renderer.active_overlays)}")
    print()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Demonstrated all {len(overlay_types)} overlay types")
    print(f"  - Rendered {stats['frame_count']} frames")
    print(f"  - Tested overlay lifecycle (add, remove, expire)")
    print()
    print("Key Features:")
    print("  ✓ 3D to 2D projection with camera parameters")
    print("  ✓ Multiple blend modes (alpha, additive, multiply, screen)")
    print("  ✓ Time-based overlay expiration")
    print("  ✓ Per-frame performance statistics")
    print("  ✓ Thread-safe overlay queue with async/await")
    print()


# Stub for cv2 if not available
try:
    import cv2
except ImportError:
    # Minimal cv2 stub for testing
    class cv2:
        @staticmethod
        def line(img, pt1, pt2, color, thickness):
            pass


if __name__ == "__main__":
    print()
    asyncio.run(main())
