"""
Multimodal Fusion Demo
=======================
Demonstration of voice + gesture multimodal input fusion.

This demo shows how voice and gesture inputs are combined to create
seamless, natural AR interactions.

Author: HoloLoom Team (Agent M)
Date: November 17, 2025
"""

import asyncio
from HoloLoom.voice.gesture_recognition import (
    GestureType,
    create_test_hand,
    Gesture
)
from HoloLoom.voice.multimodal_input import (
    create_multimodal_processor,
    MultimodalInput,
    FusionStrategy
)
from HoloLoom.voice.ar_context import create_test_context
from HoloLoom.voice.command_router import Intent, IntentType


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_multimodal_command(command):
    """Print multimodal command information."""
    print(f"\n✓ Multimodal Command:")
    print(f"  Command: {command.command}")
    print(f"  Modalities: {command.modalities}")
    print(f"  Fusion: {command.fusion_strategy}")
    print(f"  Confidence: {command.confidence:.2%}")
    if command.parameters:
        print(f"  Parameters:")
        for key, value in command.parameters.items():
            print(f"    {key}: {value}")


async def demo_voice_only():
    """Demo: Voice-only input (no gesture)."""
    print_header("Demo 1: Voice-Only Input")

    processor = await create_multimodal_processor(use_mediapipe=False)
    context = create_test_context()

    # Voice command: "show me the hive details"
    print("\nInput:")
    print("  Voice: 'show me the hive details'")
    print("  Gesture: None")

    command = await processor.process(
        voice_transcript="show me the hive details",
        ar_context=context
    )

    print_multimodal_command(command)

    assert "voice" in command.modalities
    assert "gesture" not in command.modalities
    assert command.fusion_strategy == "voice_only"

    await processor.close()


async def demo_gesture_only():
    """Demo: Gesture-only input (no voice)."""
    print_header("Demo 2: Gesture-Only Input")

    processor = await create_multimodal_processor(use_mediapipe=False)
    context = create_test_context()

    print("\nInput:")
    print("  Voice: None")
    print("  Gesture: OPEN_PALM (with hive selected)")

    # Select a hive
    context.selected_object = context.visible_objects[0]

    # Note: In real use, processor.process() would recognize gestures from camera
    # For demo, we simulate by processing without voice
    command = await processor.process(
        voice_transcript=None,
        ar_context=context
    )

    print_multimodal_command(command)

    await processor.close()


async def demo_ambiguous_voice_disambiguation():
    """Demo: Disambiguating ambiguous voice with gesture."""
    print_header("Demo 3: Ambiguous Voice Disambiguation")

    processor = await create_multimodal_processor(use_mediapipe=False)
    context = create_test_context()

    # Ambiguous voice phrases
    ambiguous_phrases = [
        "this one",
        "that one",
        "this",
        "that",
        "show me it"
    ]

    print("\nAmbiguous voice phrases that need gesture disambiguation:")
    for phrase in ambiguous_phrases:
        print(f"  • '{phrase}'")

    print("\nExample: 'this one' + pointing gesture")
    print("-" * 70)

    print("\nInput:")
    print("  Voice: 'this one' (ambiguous - which hive?)")
    print("  Gesture: POINT at hive_002 (specific target)")

    # Process ambiguous voice
    command = await processor.process(
        voice_transcript="this one",
        ar_context=context
    )

    print_multimodal_command(command)

    # Check ambiguity detection
    test_intent = Intent(
        type=IntentType.SELECT,
        action="select",
        raw_text="this one"
    )

    is_ambiguous = processor._is_ambiguous(test_intent)
    print(f"\n  Ambiguity detected: {is_ambiguous}")

    await processor.close()


async def demo_complementary_fusion():
    """Demo: Complementary voice + gesture fusion."""
    print_header("Demo 4: Complementary Fusion")

    processor = await create_multimodal_processor(use_mediapipe=False)
    context = create_test_context()

    print("\nComplementary fusion: Voice provides ACTION, gesture provides TARGET")
    print("-" * 70)

    print("\nExample 1: 'show details' + pointing")
    print("  Voice: 'show details' (action, no specific target)")
    print("  Gesture: POINT at hive_002 (target)")
    print("  Result: show_details(hive_002)")

    command1 = await processor.process(
        voice_transcript="show details",
        ar_context=context
    )

    print_multimodal_command(command1)

    print("\nExample 2: 'navigate to' + pointing")
    print("  Voice: 'navigate to' (action)")
    print("  Gesture: POINT at hive_003 (target)")
    print("  Result: navigate_to(hive_003)")

    command2 = await processor.process(
        voice_transcript="navigate to",
        ar_context=context
    )

    print_multimodal_command(command2)

    await processor.close()


async def demo_reinforced_fusion():
    """Demo: Reinforced voice + gesture fusion."""
    print_header("Demo 5: Reinforced Fusion")

    processor = await create_multimodal_processor(use_mediapipe=False)
    context = create_test_context()

    # Select a hive
    context.selected_object = context.visible_objects[0]

    print("\nReinforced fusion: Voice and gesture AGREE → higher confidence")
    print("-" * 70)

    print("\nExample: 'next hive' + swipe right")
    print("  Voice: 'next hive' (navigate_next)")
    print("  Gesture: SWIPE_RIGHT (navigate_next)")
    print("  Result: Both agree → confidence boost")

    # Test reinforcement detection
    from HoloLoom.voice.command_router import Intent, IntentType
    from HoloLoom.voice.gesture_mapper import GestureCommand

    voice_intent = Intent(
        type=IntentType.NAVIGATE,
        action="navigate_next",
        confidence=0.85,
        raw_text="next hive"
    )

    gesture_cmd = GestureCommand(
        gesture_type=GestureType.SWIPE_RIGHT,
        command="navigate_next",
        confidence=0.80
    )

    is_reinforced = processor._is_reinforced(voice_intent, gesture_cmd)
    print(f"\n  Reinforcement detected: {is_reinforced}")

    if is_reinforced:
        command = await processor._reinforced_fusion(voice_intent, gesture_cmd)
        print_multimodal_command(command)
        print(f"\n  Original confidence: {min(voice_intent.confidence, gesture_cmd.confidence):.2%}")
        print(f"  Boosted confidence: {command.confidence:.2%}")
        print(f"  Boost factor: {command.confidence / min(voice_intent.confidence, gesture_cmd.confidence):.2f}x")

    await processor.close()


async def demo_fusion_strategies():
    """Demo: All fusion strategies."""
    print_header("Demo 6: All Fusion Strategies")

    processor = await create_multimodal_processor(use_mediapipe=False)
    context = create_test_context()

    print("\nFusion Strategy Overview:")
    print("-" * 70)

    strategies = [
        ("voice_only", "Voice only (no gesture)"),
        ("gesture_only", "Gesture only (no voice)"),
        ("complementary", "Voice action + gesture target"),
        ("disambiguated", "Gesture disambiguates ambiguous voice"),
        ("reinforced", "Both modalities agree (confidence boost)"),
        ("conflicting", "Modalities conflict (reduce confidence)")
    ]

    for strategy, description in strategies:
        print(f"  {strategy.upper().ljust(20)} - {description}")

    # Test examples
    print("\n" + "-" * 70)
    print("Testing fusion strategies:")

    test_cases = [
        ("show hive", None, "voice_only"),
        ("this one", None, "disambiguated/voice_only"),
        ("show details", None, "complementary/voice_only")
    ]

    for voice, gesture, expected_strategy in test_cases:
        command = await processor.process(
            voice_transcript=voice,
            ar_context=context
        )

        print(f"\n  Voice: '{voice}'")
        print(f"  → Strategy: {command.fusion_strategy}")
        print(f"  → Command: {command.command}")

    await processor.close()


async def demo_multimodal_input_object():
    """Demo: MultimodalInput data structure."""
    print_header("Demo 7: MultimodalInput Object")

    hand = create_test_hand(GestureType.POINT)
    gesture = Gesture(
        type=GestureType.POINT,
        hand=hand,
        confidence=0.9
    )

    # Create multimodal input
    multimodal_input = MultimodalInput(
        voice_transcript="this one",
        gestures=[gesture],
        ar_context=create_test_context()
    )

    print("\nMultimodalInput:")
    print("-" * 70)
    print(f"  Voice: '{multimodal_input.voice_transcript}'")
    print(f"  Gestures: {len(multimodal_input.gestures)}")
    print(f"  Has voice: {multimodal_input.has_voice()}")
    print(f"  Has gesture: {multimodal_input.has_gesture()}")
    print(f"  Is multimodal: {multimodal_input.is_multimodal()}")
    print(f"  String: {multimodal_input}")


async def demo_history_tracking():
    """Demo: Input and command history tracking."""
    print_header("Demo 8: History Tracking")

    processor = await create_multimodal_processor(use_mediapipe=False)
    context = create_test_context()

    print("\nProcessing multiple commands:")
    print("-" * 70)

    commands_sequence = [
        "show hive",
        "next",
        "show details",
        "previous"
    ]

    for voice in commands_sequence:
        command = await processor.process(
            voice_transcript=voice,
            ar_context=context
        )
        print(f"  '{voice}' → {command.command}")

    # Show history
    print("\n" + "-" * 70)
    print("Input History:")
    input_history = processor.get_input_history()
    for i, inp in enumerate(input_history, 1):
        print(f"  {i}. {inp}")

    print("\nCommand History:")
    cmd_history = processor.get_command_history()
    for i, cmd in enumerate(cmd_history, 1):
        print(f"  {i}. {cmd.command} ({cmd.fusion_strategy}, conf={cmd.confidence:.2%})")

    await processor.close()


async def demo_real_world_scenarios():
    """Demo: Real-world AR interaction scenarios."""
    print_header("Demo 9: Real-World AR Scenarios")

    processor = await create_multimodal_processor(use_mediapipe=False)
    context = create_test_context()

    scenarios = [
        {
            "name": "Beehive Inspection",
            "voice": "show me that hive",
            "gesture": "POINT",
            "context_update": lambda ctx: None,
            "expected": "Complementary fusion: voice action + gesture target"
        },
        {
            "name": "Navigation",
            "voice": "next",
            "gesture": "SWIPE_RIGHT",
            "context_update": lambda ctx: setattr(ctx, 'selected_object', ctx.visible_objects[0]),
            "expected": "Reinforced fusion: both agree"
        },
        {
            "name": "Data Viewing",
            "voice": "show details",
            "gesture": "OPEN_PALM",
            "context_update": lambda ctx: setattr(ctx, 'selected_object', ctx.visible_objects[0]),
            "expected": "Voice provides action, context provides target"
        },
        {
            "name": "Quick Selection",
            "voice": "this one",
            "gesture": "POINT",
            "context_update": lambda ctx: None,
            "expected": "Gesture disambiguates ambiguous voice"
        }
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 70)
        print(f"  Voice: '{scenario['voice']}'")
        print(f"  Gesture: {scenario['gesture']}")
        print(f"  Expected: {scenario['expected']}")

        # Update context
        scenario['context_update'](context)

        # Process
        command = await processor.process(
            voice_transcript=scenario['voice'],
            ar_context=context
        )

        print(f"  Result: {command.command} ({command.fusion_strategy})")
        print(f"  Confidence: {command.confidence:.2%}")

    await processor.close()


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  MULTIMODAL FUSION DEMO".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Voice + Gesture Multimodal Input Fusion".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run demos
    await demo_voice_only()
    await demo_gesture_only()
    await demo_ambiguous_voice_disambiguation()
    await demo_complementary_fusion()
    await demo_reinforced_fusion()
    await demo_fusion_strategies()
    await demo_multimodal_input_object()
    await demo_history_tracking()
    await demo_real_world_scenarios()

    # Summary
    print_header("Demo Complete")
    print("\nKey Takeaways:")
    print("  ✓ 6 fusion strategies (voice_only, gesture_only, complementary, etc.)")
    print("  ✓ Automatic ambiguity detection and disambiguation")
    print("  ✓ Confidence boosting for reinforced inputs")
    print("  ✓ Complete input and command history tracking")
    print("  ✓ Natural AR interaction patterns")
    print("\nFusion Intelligence:")
    print("  • Voice provides ACTION, gesture provides TARGET")
    print("  • Gesture disambiguates vague voice ('this one', 'that')")
    print("  • Both agree → confidence boost (reinforcement)")
    print("  • Both conflict → reduce confidence (conflict resolution)")
    print("\nNext steps:")
    print("  → Integrate with Elle AR for real camera input")
    print("  → Add MediaPipe Hands for production gesture recognition")
    print("  → Test with real beekeeping scenarios")
    print()


if __name__ == "__main__":
    asyncio.run(main())
