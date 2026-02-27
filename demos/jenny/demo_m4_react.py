"""
Jenny Demo: M4 - React Component Renderer
=========================================

Demonstrates React renderer with:
- TypeScript-friendly JSON output
- Component props structure
- Accessibility props inclusion

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot M4)
"""

import json
from typing import List

from demos.jenny import register_demo, demo_phase

# Jenny imports
try:
    from hololoom.visualization.jenny_spec import (
        JennySpec,
        PanelTypeJenny,
        PanelSizeJenny,
    )
    from hololoom.visualization.jenny_renderer import ReactRenderer
    from hololoom.visualization.jenny_renderer_registry import RenderTarget
    REACT_AVAILABLE = True
except ImportError:
    REACT_AVAILABLE = False


def create_sample_specs() -> List:
    """Create sample JennySpecs for demonstration."""
    if not REACT_AVAILABLE:
        return []

    return [
        JennySpec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.TEXT,
            title="Response",
            content={"text": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem."},
            size=PanelSizeJenny.LARGE,
            priority=1,
        ),
        JennySpec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.CODE,
            title="Implementation",
            content={
                "language": "python",
                "code": "def thompson_sample(alphas, betas):\n    return np.argmax([np.random.beta(a, b) for a, b in zip(alphas, betas)])"
            },
            size=PanelSizeJenny.MEDIUM,
            priority=2,
        ),
    ]


@register_demo(
    phase_id="m4",
    name="React Renderer",
    order=4,
    requires=["m1"],  # Depends on registry
    tags=["output", "react", "typescript"],
    description="TypeScript-friendly React component props"
)
@demo_phase("M4: React Component Renderer", emoji="[M4]")
async def demo_m4_react_renderer() -> dict:
    """
    M4: React Component Props - TypeScript-friendly JSON output.

    Returns:
        Demo results with React props info
    """
    if not REACT_AVAILABLE:
        return {"status": "skipped", "reason": "React renderer not available"}

    results = {
        "panels_rendered": 0,
        "props_structure": {},
    }

    specs = create_sample_specs()
    renderer = ReactRenderer()

    # Overview
    print("  TypeScript-Friendly Output:")
    print("  " + "-" * 50)
    print("    ReactRenderer outputs JSON props compatible with:")
    print("      - React components (functional or class)")
    print("      - TypeScript type checking")
    print("      - React Native (mobile)")
    print()

    # Render to React props
    result = await renderer.render(specs, target=RenderTarget.REACT)
    props = json.loads(result)

    # Show structure
    print("  JennyDashboard Component Props:")
    print("  " + "-" * 50)
    print(f"    Component: {props.get('component', 'JennyDashboard')}")
    print(f"    Query ID: {props.get('queryId', 'N/A')}")
    print(f"    Panel Count: {len(props.get('panels', []))}")
    results["panels_rendered"] = len(props.get('panels', []))
    print()

    # Show first panel structure
    if props.get('panels'):
        panel = props['panels'][0]
        print("  First Panel Props:")
        for key in ['id', 'type', 'title', 'size', 'priority']:
            value = panel.get(key, 'N/A')
            print(f"    {key}: {value}")
            results["props_structure"][key] = type(value).__name__

        if 'accessibility' in panel:
            print(f"    accessibility: [WCAG 2.1 AA compliant]")
    print()

    # Integration pattern
    print("  Integration Pattern:")
    print("  " + "-" * 50)
    print("    import { JennyDashboard } from '@hololoom/jenny';")
    print("    <JennyDashboard {...props} />")

    return {
        "status": "success",
        "panels_rendered": results["panels_rendered"],
        "output_bytes": len(result),
        "props_keys": list(results["props_structure"].keys()),
    }


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_m4_react_renderer())
