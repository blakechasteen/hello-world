"""
Jenny Demo: M1 - Renderer Registry Pattern
==========================================

Demonstrates the singleton renderer registry with:
- Auto-registration on import
- Priority-based selection
- Concurrent rendering support

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot M1)
"""

from demos.jenny import register_demo, demo_phase

# Jenny imports
try:
    from HoloLoom.visualization.jenny_renderer_registry import (
        RendererRegistry,
        RenderTarget,
    )
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False


@register_demo(
    phase_id="m1",
    name="Renderer Registry",
    order=1,
    tags=["core", "infrastructure"],
    description="Singleton registry for extensible renderer plugins"
)
@demo_phase("M1: Renderer Registry Pattern", emoji="[M1]")
async def demo_m1_renderer_registry() -> dict:
    """
    M1: Renderer Registry Pattern - Singleton, priority-based selection.

    Returns:
        Demo results with registry info
    """
    if not REGISTRY_AVAILABLE:
        return {"status": "skipped", "reason": "Registry not available"}

    results = {
        "singleton_verified": False,
        "registered_renderers": {},
        "targets_supported": [],
    }

    # Get singleton instance
    registry = RendererRegistry()
    registry2 = RendererRegistry()
    results["singleton_verified"] = (registry is registry2)
    print(f"  Registry Singleton: {type(registry).__name__}")
    print(f"  Same instance check: {results['singleton_verified']}")
    print()

    # Check registered renderers
    print("  Auto-Registered Renderers (on import):")
    for target in [RenderTarget.HTML, RenderTarget.REACT, RenderTarget.AR]:
        renderer = registry.get_for_target(target)
        if renderer:
            renderer_name = type(renderer).__name__
            results["registered_renderers"][target.value] = renderer_name
            results["targets_supported"].append(target.value)
            print(f"    {target.value:10s} -> {renderer_name}")
        else:
            print(f"    {target.value:10s} -> (not registered)")
    print()

    # Priority-based selection info
    print("  Priority-Based Selection:")
    print("    Lower priority = higher priority (first to render)")
    print("    HTML     -> Standard web rendering (default)")
    print("    React    -> TypeScript component props")
    print("    AR       -> WebXR 3D transforms")
    print()

    # Concurrent capability
    print("  Concurrent Rendering: Supported via render_concurrent()")

    return {
        "status": "success",
        "singleton_verified": results["singleton_verified"],
        "renderers_count": len(results["registered_renderers"]),
        "targets": results["targets_supported"],
    }


# Allow direct execution for testing
if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_m1_renderer_registry())
