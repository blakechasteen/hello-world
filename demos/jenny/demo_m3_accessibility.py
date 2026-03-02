"""
Jenny Demo: M3 - WCAG 2.1 AA Accessibility
==========================================

Demonstrates accessibility layer with:
- ARIA attributes and roles
- Keyboard navigation
- Focus management
- Screen reader support

Author: HoloLoom Team
Date: 2025-12-09 (Jenny Moonshot M3)
"""

from demos.jenny import register_demo, demo_phase

# Jenny imports
try:
    from hololoom.visualization.jenny_accessibility import (
        AriaRole,
        AriaLive,
        AriaRelevant,
        AriaAttributes,
        accessible_panel,
        KeyboardHandler,
        FocusManager,
        create_accessibility_layer,
    )
    ACCESSIBILITY_AVAILABLE = True
except ImportError:
    ACCESSIBILITY_AVAILABLE = False


@register_demo(
    phase_id="m3",
    name="Accessibility Layer",
    order=3,
    tags=["core", "accessibility", "wcag"],
    description="WCAG 2.1 AA compliant accessibility layer"
)
@demo_phase("M3: WCAG 2.1 AA Accessibility", emoji="[M3]")
async def demo_m3_accessibility() -> dict:
    """
    M3: WCAG 2.1 AA Accessibility - ARIA, keyboard nav, focus management.

    Returns:
        Demo results with accessibility info
    """
    if not ACCESSIBILITY_AVAILABLE:
        return {"status": "skipped", "reason": "Accessibility module not available"}

    results = {
        "aria_attributes_shown": [],
        "keyboard_shortcuts": [],
        "live_regions": [],
    }

    # ARIA Attributes
    print("  ARIA Attributes:")
    print("  " + "-" * 50)

    aria = AriaAttributes(
        role=AriaRole.REGION,
        label="Query Response Panel",
        live=AriaLive.POLITE,
        relevant=AriaRelevant.ADDITIONS,
        expanded=True,
        controls="panel-content-001",
    )

    attrs_dict = aria.to_dict()
    print("\n  Panel ARIA Attributes:")
    for key, value in attrs_dict.items():
        print(f"    {key}: {value}")
        results["aria_attributes_shown"].append(key)
    print()

    # Keyboard Navigation
    print("  Keyboard Navigation:")
    print("  " + "-" * 50)
    shortcuts = [
        ("Tab", "Move to next panel"),
        ("Shift+Tab", "Move to previous panel"),
        ("Enter", "Activate/expand panel"),
        ("Escape", "Close/collapse panel"),
        ("Arrow keys", "Navigate within panel"),
    ]
    print("\n  Focus Order:")
    for i, (key, action) in enumerate(shortcuts, 1):
        print(f"    {i}. {key:12s} -> {action}")
        results["keyboard_shortcuts"].append(key)
    print()

    # Screen Reader Announcements
    print("  Screen Reader Announcements:")
    print("  " + "-" * 50)
    live_regions = [
        ("POLITE", "Wait for idle to announce"),
        ("ASSERTIVE", "Interrupt current speech"),
        ("OFF", "Silent updates"),
    ]
    print("\n  Live Regions:")
    for mode, description in live_regions:
        print(f"    - {mode}: {description}")
        results["live_regions"].append(mode)
    print()

    # Accessibility Utilities
    print("  Accessibility Utilities:")
    print("  " + "-" * 50)
    print("    make_panel_accessible(spec) -> Adds ARIA to panel")
    print("    add_keyboard_navigation(container) -> Adds key handlers")
    print("    Enhanced contrast: minimum 4.5:1 ratio")

    return {
        "status": "success",
        "aria_attributes": len(results["aria_attributes_shown"]),
        "keyboard_shortcuts": len(results["keyboard_shortcuts"]),
        "live_regions": results["live_regions"],
    }


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_m3_accessibility())
