"""
Jenny Demo: M5 - WebXR AR Renderer
==================================

Demonstrates AR renderer with:
- WebXR-compatible 3D scene output
- Spatial transforms and positioning
- Coordinate system validation

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot M5)
"""

import json
from typing import List

from demos.jenny import register_demo, demo_phase

# Jenny imports
try:
    from HoloLoom.visualization.jenny_spec import (
        JennySpec,
        PanelTypeJenny,
        PanelSizeJenny,
    )
    from HoloLoom.visualization.jenny_renderer import ARRenderer
    from HoloLoom.visualization.jenny_renderer_registry import RenderTarget
    AR_AVAILABLE = True
except ImportError:
    AR_AVAILABLE = False


def create_ar_specs() -> List:
    """Create sample JennySpecs for AR demonstration."""
    if not AR_AVAILABLE:
        return []

    return [
        JennySpec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.TEXT,
            title="Main Response",
            content={"text": "Thompson Sampling is a Bayesian approach to bandits."},
            size=PanelSizeJenny.LARGE,
            priority=1,
        ),
        JennySpec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.GRAPH,
            title="Knowledge Graph",
            content={
                "nodes": [
                    {"id": "thompson", "label": "Thompson Sampling"},
                    {"id": "bayesian", "label": "Bayesian Methods"},
                    {"id": "bandit", "label": "Multi-Armed Bandit"},
                ],
                "edges": [
                    {"from": "thompson", "to": "bayesian"},
                    {"from": "thompson", "to": "bandit"},
                ]
            },
            size=PanelSizeJenny.MEDIUM,
            priority=2,
        ),
        JennySpec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.METRIC,
            title="Confidence",
            content={"value": 0.92, "label": "Response Confidence"},
            size=PanelSizeJenny.SMALL,
            priority=3,
        ),
    ]


@register_demo(
    phase_id="m5",
    name="AR Renderer",
    order=5,
    requires=["m1"],  # Depends on registry
    tags=["output", "ar", "webxr", "3d"],
    description="WebXR-compatible 3D scene output"
)
@demo_phase("M5: WebXR AR Renderer", emoji="[M5]")
async def demo_m5_ar_renderer() -> dict:
    """
    M5: AR Renderer - WebXR-compatible 3D scene output.

    Returns:
        Demo results with AR scene info
    """
    if not AR_AVAILABLE:
        return {"status": "skipped", "reason": "AR renderer not available"}

    results = {
        "objects_rendered": 0,
        "coordinate_system": "",
        "transforms": [],
    }

    specs = create_ar_specs()
    renderer = ARRenderer()

    # Overview
    print("  WebXR 3D Scene Output:")
    print("  " + "-" * 50)
    print("    ARRenderer generates JSON scenes compatible with:")
    print("      - WebXR Device API")
    print("      - A-Frame (aframe.io)")
    print("      - Three.js AR modules")
    print("      - ARKit/ARCore (via WebXR)")
    print()

    # Render to AR JSON
    result = await renderer.render(specs, target=RenderTarget.AR)
    scene = json.loads(result)

    # Show scene structure
    print("  AR Scene Structure:")
    print("  " + "-" * 50)
    print(f"    Coordinate System: {scene.get('coordinate_system', 'webxr')}")
    results["coordinate_system"] = scene.get('coordinate_system', 'webxr')

    objects = scene.get('objects', scene.get('entities', []))
    print(f"    Object Count: {len(objects)}")
    results["objects_rendered"] = len(objects)
    print()

    # Show transforms
    print("  Object Transforms:")
    print("  " + "-" * 50)
    for i, obj in enumerate(objects[:3]):  # Show first 3
        transform = obj.get('transform', {})
        position = transform.get('position', [0, 0, 0])
        rotation = transform.get('rotation', [0, 0, 0, 1])
        scale = transform.get('scale', [1, 1, 1])

        print(f"    Object {i+1}: {obj.get('type', 'unknown')}")
        print(f"      Position: {position}")
        print(f"      Rotation: {rotation}")
        print(f"      Scale: {scale}")

        results["transforms"].append({
            "type": obj.get('type'),
            "position": position,
            "rotation": rotation,
            "scale": scale,
        })
    print()

    # Spatial layout info
    print("  Spatial Layout Strategy:")
    print("  " + "-" * 50)
    print("    Panels arranged in 3D space:")
    print("      - High priority: Center-front (z = -1.5)")
    print("      - Medium: Peripheral (z = -2.0)")
    print("      - Low: Background (z = -2.5)")
    print("    User can naturally focus on important panels")

    return {
        "status": "success",
        "objects_rendered": results["objects_rendered"],
        "coordinate_system": results["coordinate_system"],
        "output_bytes": len(result),
        "transforms_count": len(results["transforms"]),
    }


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_m5_ar_renderer())
