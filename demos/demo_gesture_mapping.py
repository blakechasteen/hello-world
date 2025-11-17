"""
Gesture Mapping Demo
=====================
Demonstration of context-aware gesture-to-command mapping.

This demo shows how gestures are mapped to different commands based on
the current AR context (selected objects, scene state, etc.).

Author: HoloLoom Team (Agent M)
Date: November 17, 2025
"""

import asyncio
from HoloLoom.voice.gesture_recognition import (
    GestureType,
    create_test_hand,
    Gesture
)
from HoloLoom.voice.gesture_mapper import (
    ContextAwareGestureMapper,
    ContextType,
    GestureMappingRule,
    create_default_mapper
)
from HoloLoom.voice.ar_context import (
    create_test_context,
    ARObjectType,
    Vector3
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_context_info(context):
    """Print AR context information."""
    print("\nAR Context:")
    print(f"  User position: {context.user_position}")
    print(f"  Gaze target: {context.gaze_target}")
    print(f"  Selected object: {context.selected_object.id if context.selected_object else 'None'}")
    print(f"  Visible objects: {len(context.visible_objects)}")
    print(f"  Scene: {context.active_scene}")
    print(f"  Conversation context: '{context.conversation_context}'")


async def demo_basic_mapping():
    """Demo: Basic gesture mapping."""
    print_header("Demo 1: Basic Gesture Mapping")

    # Create mapper and context
    mapper = create_default_mapper()
    context = create_test_context()

    print_context_info(context)

    # Test POINT gesture
    print("\n" + "-" * 70)
    print("Gesture: POINT (at hive)")
    hand = create_test_hand(GestureType.POINT)
    gesture = Gesture(
        type=GestureType.POINT,
        hand=hand,
        confidence=0.9,
        target_position=(0.1, 0.0, 3.0)  # Pointing at hive_003
    )

    command = await mapper.map_gesture(gesture, context)

    if command:
        print(f"\n✓ Command mapped:")
        print(f"  Command: {command.command}")
        print(f"  Gesture: {command.gesture_type.value}")
        print(f"  Confidence: {command.confidence:.2%}")
        print(f"  Parameters: {command.parameters}")


async def demo_context_aware_mapping():
    """Demo: Context-aware gesture mapping."""
    print_header("Demo 2: Context-Aware Mapping")

    mapper = create_default_mapper()
    context = create_test_context()

    # Test 1: OPEN_PALM in different contexts
    print("\nTest: OPEN_PALM gesture in different contexts")
    print("-" * 70)

    hand = create_test_hand(GestureType.OPEN_PALM)
    gesture = Gesture(
        type=GestureType.OPEN_PALM,
        hand=hand,
        confidence=0.95
    )

    # Context 1: No selection (default)
    context.selected_object = None
    context.conversation_context = ""
    command1 = await mapper.map_gesture(gesture, context)

    print("\n1. Default context (no selection):")
    if command1:
        print(f"   → {command1.command}")

    # Context 2: Hive selected
    context.selected_object = context.visible_objects[0]  # hive_001
    command2 = await mapper.map_gesture(gesture, context)

    print("\n2. Hive selected:")
    if command2:
        print(f"   → {command2.command}")

    # Context 3: Menu open
    context.conversation_context = "menu open"
    command3 = await mapper.map_gesture(gesture, context)

    print("\n3. Menu open:")
    if command3:
        print(f"   → {command3.command}")


async def demo_swipe_gestures():
    """Demo: Swipe gesture mapping."""
    print_header("Demo 3: Swipe Gestures")

    mapper = create_default_mapper()
    context = create_test_context()

    # Select a hive
    context.selected_object = context.visible_objects[0]

    print_context_info(context)

    # Test all swipe directions
    swipe_gestures = [
        (GestureType.SWIPE_LEFT, (-1.0, 0.0, 0.0)),
        (GestureType.SWIPE_RIGHT, (1.0, 0.0, 0.0)),
        (GestureType.SWIPE_UP, (0.0, -1.0, 0.0)),
        (GestureType.SWIPE_DOWN, (0.0, 1.0, 0.0))
    ]

    print("\n" + "-" * 70)
    print("Swipe gesture mapping (with hive selected):")

    for gesture_type, direction in swipe_gestures:
        hand = create_test_hand(GestureType.GRAB)  # Reuse grab hand
        gesture = Gesture(
            type=gesture_type,
            hand=hand,
            confidence=0.85,
            direction=direction
        )

        command = await mapper.map_gesture(gesture, context)

        if command:
            print(f"\n  {gesture_type.value.upper()}:")
            print(f"    → {command.command}")
            print(f"    Direction: {direction}")


async def demo_pointing_raycast():
    """Demo: Pointing with raycast to objects."""
    print_header("Demo 4: Pointing and Raycasting")

    mapper = create_default_mapper()
    context = create_test_context()

    print_context_info(context)
    print("\nVisible objects:")
    for obj in context.visible_objects:
        print(f"  {obj.id} ({obj.type.value}) at {obj.position}")

    # Point at different objects
    test_positions = [
        ((0.1, 0.0, 3.0), "hive_003 (closest)"),
        ((0.3, 0.0, 5.0), "hive_001 (right)"),
        ((-0.3, 0.0, 5.0), "hive_002 (left)")
    ]

    print("\n" + "-" * 70)
    print("Pointing at different objects:")

    for target_pos, description in test_positions:
        hand = create_test_hand(GestureType.POINT)
        gesture = Gesture(
            type=GestureType.POINT,
            hand=hand,
            confidence=0.9,
            target_position=target_pos
        )

        command = await mapper.map_gesture(gesture, context)

        print(f"\n  Target: {description}")
        print(f"    Position: {target_pos}")
        if command and "object_id" in command.parameters:
            print(f"    → Detected: {command.parameters['object_id']}")
            print(f"    → Command: {command.command}")
        else:
            print(f"    → No object detected at this position")


async def demo_gesture_history():
    """Demo: Gesture history tracking."""
    print_header("Demo 5: Gesture History Tracking")

    mapper = create_default_mapper()
    context = create_test_context()

    # Perform multiple gestures
    gestures = [
        (GestureType.POINT, (0.1, 0.0, 3.0)),
        (GestureType.OPEN_PALM, None),
        (GestureType.SWIPE_LEFT, None),
        (GestureType.GRAB, None)
    ]

    print("\nPerforming gesture sequence:")
    for gesture_type, target in gestures:
        hand = create_test_hand(gesture_type)
        gesture = Gesture(
            type=gesture_type,
            hand=hand,
            confidence=0.9,
            target_position=target
        )

        command = await mapper.map_gesture(gesture, context)
        if command:
            print(f"  {gesture_type.value} → {command.command}")

    # Show history
    print("\n" + "-" * 70)
    print("Gesture History:")
    history = mapper.get_gesture_history()
    for i, gesture in enumerate(history, 1):
        print(f"  {i}. {gesture.type.value} (conf={gesture.confidence:.2%})")

    print("\nCommand History:")
    cmd_history = mapper.get_command_history()
    for i, command in enumerate(cmd_history, 1):
        print(f"  {i}. {command.command} ({command.gesture_type.value})")


async def demo_custom_rules():
    """Demo: Adding custom mapping rules."""
    print_header("Demo 6: Custom Mapping Rules")

    mapper = create_default_mapper()
    context = create_test_context()

    print(f"\nDefault rules: {len(mapper.mapping_rules)}")

    # Add custom rule for CIRCLE gesture
    custom_rule = GestureMappingRule(
        gesture_type=GestureType.CIRCLE,
        context_type=ContextType.HIVE_SELECTED,
        command="rotate_view",
        confidence_multiplier=0.9,
        description="Rotate view around selected hive"
    )

    mapper.add_custom_rule(custom_rule)

    print(f"After adding custom rule: {len(mapper.mapping_rules)}")

    # Test custom rule
    context.selected_object = context.visible_objects[0]

    hand = create_test_hand(GestureType.GRAB)  # Reuse grab hand
    gesture = Gesture(
        type=GestureType.CIRCLE,
        hand=hand,
        confidence=0.85
    )

    command = await mapper.map_gesture(gesture, context)

    if command:
        print(f"\n✓ Custom rule triggered:")
        print(f"  Gesture: {gesture.type.value}")
        print(f"  Command: {command.command}")
        print(f"  Confidence: {command.confidence:.2%}")


async def demo_all_mapping_rules():
    """Demo: Show all default mapping rules."""
    print_header("Demo 7: All Default Mapping Rules")

    mapper = create_default_mapper()

    print(f"\nTotal rules: {len(mapper.mapping_rules)}\n")

    # Group by gesture type
    rules_by_gesture = {}
    for rule in mapper.mapping_rules:
        gesture_type = rule.gesture_type.value
        if gesture_type not in rules_by_gesture:
            rules_by_gesture[gesture_type] = []
        rules_by_gesture[gesture_type].append(rule)

    for gesture_type in sorted(rules_by_gesture.keys()):
        print(f"{gesture_type.upper()}:")
        for rule in rules_by_gesture[gesture_type]:
            print(f"  • {rule.context_type.value.ljust(20)} → {rule.command}")
            if rule.description:
                print(f"    ({rule.description})")
        print()


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  GESTURE MAPPING DEMO".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Context-Aware Gesture-to-Command Mapping".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run demos
    await demo_basic_mapping()
    await demo_context_aware_mapping()
    await demo_swipe_gestures()
    await demo_pointing_raycast()
    await demo_gesture_history()
    await demo_custom_rules()
    await demo_all_mapping_rules()

    # Summary
    print_header("Demo Complete")
    print("\nKey Takeaways:")
    print("  ✓ Context-aware mapping (same gesture → different commands)")
    print("  ✓ 15+ default mapping rules")
    print("  ✓ Supports all 10 gesture types")
    print("  ✓ Pointing with raycast to select objects")
    print("  ✓ Gesture and command history tracking")
    print("  ✓ Custom rule support")
    print("\nContext Types:")
    print("  • DEFAULT, HIVE_SELECTED, TOOL_SELECTED")
    print("  • NAVIGATION_ACTIVE, INSPECTION_MODE")
    print("  • DATA_VIEW, MENU_OPEN")
    print("\nNext steps:")
    print("  → Run demo_multimodal_fusion.py to see voice + gesture fusion")
    print()


if __name__ == "__main__":
    asyncio.run(main())
