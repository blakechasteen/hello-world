#!/usr/bin/env python3
"""
Demo: Mobile Spatial UI
========================

Interactive demo showing HoloLoom's Phase 4 mobile-responsive spatial UI:
- Device capability detection
- Touch gesture recognition
- Responsive UI components
- HUD creation

Part of HoloLoom Phase 4: Spatial Computing Enhancement (November 2025)
"""

import logging
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Mobile spatial UI imports
from hololoom.spatial.mobile_spatial_ui import (
    MobileSpatialUIManager,
    TouchGestureRecognizer,
    create_mobile_ui_manager,
    create_default_spatial_ui,
    DeviceCapabilities,
    DeviceType,
    ScreenOrientation,
    InputMode,
    UIScale,
    UIAnchor,
    TouchGesture,
    TouchGestureType,
    UIComponent,
    UIComponentType,
    FloatingActionButton,
    BottomSheet,
    RadialMenu,
    SpatialHUD,
    ResponsiveValue
)


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


def demo_device_capabilities():
    """Demo 1: Device capability detection."""
    print_header("Demo 1: Device Capability Detection")

    # Test different device types
    devices = [
        ("iPhone 14 Pro", DeviceType.MOBILE_PHONE, 393, 852, True),
        ("iPad Pro 12.9", DeviceType.TABLET, 1024, 1366, True),
        ("Desktop Browser", DeviceType.DESKTOP, 1920, 1080, False),
        ("AR Headset", DeviceType.AR_HEADSET, 2560, 1440, True),
    ]

    for name, device_type, width, height, has_ar in devices:
        caps = DeviceCapabilities(
            device_type=device_type,
            screen_width=width,
            screen_height=height,
            has_touch=device_type in [DeviceType.MOBILE_PHONE, DeviceType.TABLET],
            has_ar_support=has_ar,
            orientation=ScreenOrientation.PORTRAIT if height > width else ScreenOrientation.LANDSCAPE
        )

        print(f"\n  {name}:")
        print(f"    Type: {caps.device_type.value}")
        print(f"    Screen: {caps.screen_width}x{caps.screen_height}")
        print(f"    Orientation: {caps.orientation.value}")
        print(f"    Has Touch: {caps.has_touch}")
        print(f"    Has AR: {caps.has_ar_support}")
        print(f"    Recommended Scale: {caps.get_recommended_scale().value}")


def demo_touch_gesture_recognition():
    """Demo 2: Touch gesture recognition."""
    print_header("Demo 2: Touch Gesture Recognition")

    recognizer = TouchGestureRecognizer()

    # Track detected gestures
    detected_gestures: List[str] = []

    def on_gesture(gesture: TouchGesture):
        detected_gestures.append(f"{gesture.gesture_type.value} at {gesture.position}")

    # Register handlers for all gesture types
    for gesture_type in [
        TouchGestureType.TAP,
        TouchGestureType.DOUBLE_TAP,
        TouchGestureType.LONG_PRESS,
        TouchGestureType.SWIPE_LEFT,
        TouchGestureType.SWIPE_RIGHT,
        TouchGestureType.PINCH_IN,
        TouchGestureType.PINCH_OUT,
    ]:
        recognizer.on(gesture_type, on_gesture)

    # Simulate TAP gesture
    print_subheader("Simulating TAP Gesture")
    import time
    recognizer.on_touch_start(0, [(0.5, 0.5)])
    time.sleep(0.05)  # Brief touch
    result = recognizer.on_touch_end(0, [(0.5, 0.5)])
    if result:
        print(f"  Detected: {result.gesture_type.value}")
        print(f"  Position: {result.position}")
        print(f"  Confidence: {result.confidence:.2f}")

    # Simulate SWIPE gesture
    print_subheader("Simulating SWIPE Gesture")
    recognizer.on_touch_start(1, [(0.2, 0.5)])
    recognizer.on_touch_move(1, [(0.8, 0.5)])  # Horizontal swipe
    result = recognizer.on_touch_end(1, [(0.8, 0.5)])
    if result:
        print(f"  Detected: {result.gesture_type.value}")
        print(f"  Velocity: {result.velocity}")

    # Simulate PINCH gesture
    print_subheader("Simulating PINCH Gesture")
    # Two-finger touch
    recognizer.on_touch_start(2, [(0.4, 0.5), (0.6, 0.5)])
    # Spread fingers apart
    result = recognizer.on_touch_move(2, [(0.2, 0.5), (0.8, 0.5)])
    if result:
        print(f"  Detected: {result.gesture_type.value}")
        print(f"  Scale: {result.scale:.2f}")
        print(f"  Rotation: {result.rotation:.2f}")


def demo_responsive_ui_components():
    """Demo 3: Responsive UI components."""
    print_header("Demo 3: Responsive UI Components")

    # Create UI manager for mobile phone
    manager = create_mobile_ui_manager(
        screen_width=393,
        screen_height=852,
        device_type=DeviceType.MOBILE_PHONE,
        has_ar=True
    )

    print_subheader("Device Configuration")
    print(f"  Device: {manager.capabilities.device_type.value}")
    print(f"  UI Scale: {manager.ui_scale.value}")
    print(f"  Render Quality: {manager.render_quality}")

    # Create floating action button
    print_subheader("Creating UI Components")
    fab = manager.create_floating_button(
        "add_memory",
        icon="plus",
        anchor=UIAnchor.BOTTOM_RIGHT
    )
    print(f"  Created FAB: {fab.component_id}")
    print(f"    Anchor: {fab.anchor.value}")
    print(f"    Size: {fab.size}px")

    # Create bottom sheet
    sheet = manager.create_bottom_sheet(
        "memory_details",
        header_label="Memory Details"
    )
    print(f"  Created BottomSheet: {sheet.component_id}")
    print(f"    Expanded Height: {sheet.expanded_height}")
    print(f"    Has Handle: {sheet.has_handle}")

    # Create radial menu
    menu = manager.create_radial_menu(
        "quick_actions",
        items=[
            {"icon": "search", "label": "Search", "action": "search"},
            {"icon": "filter", "label": "Filter", "action": "filter"},
            {"icon": "share", "label": "Share", "action": "share"},
        ]
    )
    print(f"  Created RadialMenu: {menu.component_id}")
    print(f"    Items: {len(menu.items)}")

    # Test responsive values
    print_subheader("Responsive Values")
    responsive_width = ResponsiveValue(xs=0.9, sm=0.8, md=0.6, lg=0.4)
    print(f"  Responsive width config: xs=90%, sm=80%, md=60%, lg=40%")
    print(f"    At 320px: {responsive_width.get_for_width(320)}")
    print(f"    At 480px: {responsive_width.get_for_width(480)}")
    print(f"    At 768px: {responsive_width.get_for_width(768)}")
    print(f"    At 1200px: {responsive_width.get_for_width(1200)}")


def demo_hud_creation():
    """Demo 4: HUD creation."""
    print_header("Demo 4: Spatial HUD Creation")

    manager = create_mobile_ui_manager(
        screen_width=393,
        screen_height=852,
        device_type=DeviceType.MOBILE_PHONE
    )

    # Create HUD
    hud = manager.create_hud("main_hud")
    print(f"\n  Created HUD: {hud.hud_id}")
    print(f"    Follow Gaze: {hud.follow_gaze}")
    print(f"    Distance: {hud.distance}m")

    # Add HUD elements
    print_subheader("Adding HUD Elements")

    compass = UIComponent(
        component_id="compass",
        component_type=UIComponentType.COMPASS,
        anchor=UIAnchor.TOP_RIGHT,
        width=ResponsiveValue(xs=60, md=50),
        height=ResponsiveValue(xs=60, md=50)
    )
    hud.add_element(compass)
    print(f"  Added: Compass (top-right)")

    minimap = UIComponent(
        component_id="minimap",
        component_type=UIComponentType.MINIMAP,
        anchor=UIAnchor.TOP_LEFT,
        width=ResponsiveValue(xs=100, md=80),
        height=ResponsiveValue(xs=100, md=80)
    )
    hud.add_element(minimap)
    print(f"  Added: Minimap (top-left)")

    status_bar = UIComponent(
        component_id="status",
        component_type=UIComponentType.STATUS_BAR,
        anchor=UIAnchor.TOP_CENTER
    )
    hud.add_element(status_bar)
    print(f"  Added: Status Bar (top-center)")

    print_subheader("HUD State")
    state = hud.to_dict()
    print(f"  Elements: {len(state['elements'])}")
    print(f"  Opacity: {state['opacity']}")


def demo_ui_events():
    """Demo 5: UI event handling."""
    print_header("Demo 5: UI Event Handling")

    manager = create_mobile_ui_manager(
        screen_width=393,
        screen_height=852,
        device_type=DeviceType.MOBILE_PHONE
    )

    # Track events
    events_received: List[Dict[str, Any]] = []

    # Register event handlers
    def on_zoom(data):
        events_received.append({"type": "zoom", "data": data})
        print(f"    Zoom event: scale={data['scale']}, direction={data['direction']}")

    def on_navigate(data):
        events_received.append({"type": "navigate", "data": data})
        print(f"    Navigate event: direction={data['direction']}")

    def on_drawer(data):
        events_received.append({"type": "drawer", "data": data})
        print(f"    Drawer event: side={data['side']}, action={data['action']}")

    manager.on("zoom", on_zoom)
    manager.on("navigate", on_navigate)
    manager.on("drawer", on_drawer)

    print_subheader("Simulating Gestures → Events")

    # Simulate pinch out (zoom in)
    print("  Simulating pinch out gesture...")
    manager.gesture_recognizer.on_touch_start(0, [(0.4, 0.5), (0.6, 0.5)])
    result = manager.gesture_recognizer.on_touch_move(0, [(0.2, 0.5), (0.8, 0.5)])
    if result:
        manager.gesture_recognizer._emit_gesture(result)

    # Simulate swipe left (navigate right)
    print("  Simulating swipe left gesture...")
    manager.gesture_recognizer.on_touch_start(1, [(0.8, 0.5)])
    manager.gesture_recognizer.on_touch_move(1, [(0.2, 0.5)])
    result = manager.gesture_recognizer.on_touch_end(1, [(0.2, 0.5)])
    if result:
        manager.gesture_recognizer._emit_gesture(result)

    print(f"\n  Total events received: {len(events_received)}")


def demo_performance_hints():
    """Demo 6: Performance hints for different devices."""
    print_header("Demo 6: Performance Hints")

    devices = [
        ("Low-end Phone", DeviceType.MOBILE_PHONE, 1),
        ("Mid-range Phone", DeviceType.MOBILE_PHONE, 2),
        ("High-end Tablet", DeviceType.TABLET, 3),
    ]

    for name, device_type, gpu_tier in devices:
        caps = DeviceCapabilities(
            device_type=device_type,
            screen_width=375,
            screen_height=812,
            gpu_tier=gpu_tier
        )

        manager = MobileSpatialUIManager(caps)
        manager.update_capabilities(caps)

        hints = manager.get_performance_hints()

        print(f"\n  {name} (GPU tier {gpu_tier}):")
        print(f"    Render Quality: {hints['render_quality']}")
        print(f"    Max Overlays: {hints['max_visible_overlays']}")
        print(f"    Physics: {hints['physics_enabled']}")
        print(f"    Shadow Quality: {hints['shadow_quality']}")
        print(f"    Particles: {hints['particle_count']}")


def demo_default_ui():
    """Demo 7: Create default spatial UI."""
    print_header("Demo 7: Default Spatial UI")

    manager = create_mobile_ui_manager(
        screen_width=393,
        screen_height=852,
        device_type=DeviceType.MOBILE_PHONE,
        has_ar=True
    )

    # Create default UI components
    components = create_default_spatial_ui(manager)

    print_subheader("Default Components Created")
    for comp_id, comp in components.items():
        print(f"  • {comp_id}: {comp.component_type.value}")

    print_subheader("Full UI State")
    state = manager.to_state()
    print(f"  Device: {state['device']['device_type']}")
    print(f"  UI Scale: {state['ui_scale']}")
    print(f"  Components: {len(state['components'])}")
    print(f"  HUD: {'Present' if state['hud'] else 'None'}")
    print(f"  Performance hints:")
    for key, value in state['performance'].items():
        print(f"    {key}: {value}")


def main():
    """Run the complete mobile spatial UI demo."""
    print("\n" + "=" * 60)
    print("  HoloLoom Phase 4: Mobile Spatial UI Demo")
    print("  November 2025")
    print("=" * 60)

    try:
        # Demo 1: Device capabilities
        demo_device_capabilities()

        # Demo 2: Touch gesture recognition
        demo_touch_gesture_recognition()

        # Demo 3: Responsive UI components
        demo_responsive_ui_components()

        # Demo 4: HUD creation
        demo_hud_creation()

        # Demo 5: UI events
        demo_ui_events()

        # Demo 6: Performance hints
        demo_performance_hints()

        # Demo 7: Default UI
        demo_default_ui()

        print("\n" + "=" * 60)
        print("  Demo Complete!")
        print("=" * 60)
        print("\nAll features demonstrated:")
        print("  ✓ Device capability detection")
        print("  ✓ Touch gesture recognition (tap, swipe, pinch)")
        print("  ✓ Responsive UI components (FAB, sheet, radial menu)")
        print("  ✓ Spatial HUD creation with elements")
        print("  ✓ Gesture-to-event mapping")
        print("  ✓ Performance hints per device tier")
        print("  ✓ Default spatial UI factory")

    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise


if __name__ == "__main__":
    main()
