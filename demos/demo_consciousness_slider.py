r"""
Demo: Consciousness Slider - Individual -> Shared -> Universal Dream Spectrum
===========================================================================

This demo showcases the Consciousness Slider mechanics:
1. All three consciousness levels (individual, shared, universal)
2. Parameter progression across the slider range
3. Dream adjustment at each level
4. Ego reference handling and narrative voice transitions

Run:
    PYTHONPATH=. python demos/demo_consciousness_slider.py
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from NeuroHood.dreams.consciousness_slider import (
    ConsciousnessSlider,
    ConsciousnessLevel,
    PrivacyMode,
)


def print_header(text: str, level: int = 1):
    """Print formatted header."""
    if level == 1:
        print("\n" + "=" * 80)
        print(f" {text}")
        print("=" * 80)
    elif level == 2:
        print("\n" + "-" * 80)
        print(f" {text}")
        print("-" * 80)
    else:
        print(f"\n >>> {text}")


def print_parameter(name: str, value: Any, max_width: int = 40):
    """Print formatted parameter with value."""
    if isinstance(value, float):
        value_str = f"{value:.3f}"
    else:
        value_str = str(value)
    print(f"  {name:<30} : {value_str}")


def demo_slider_overview():
    """Demo 1: Overview of three consciousness levels."""
    print_header("Demo 1: Consciousness Level Overview")

    slider = ConsciousnessSlider()

    levels = [
        (0.15, "Individual Consciousness (low)"),
        (0.50, "Shared Consciousness (mid)"),
        (0.85, "Universal Consciousness (high)"),
    ]

    for slider_value, title in levels:
        print_header(title, level=2)

        settings = slider.get_settings(slider_value)

        print(f"\nSlider Position: {slider_value * 100:.1f}%")
        print(f"Consciousness Level: {settings.level.value.upper()}")
        print(f"Privacy Mode: {settings.privacy_mode.value}")

        print("\nCore Parameters:")
        print_parameter("Max Participants", settings.max_participants)
        print_parameter("Privacy Gradient", settings.privacy_gradient)
        print_parameter("Archetypal Strength", settings.archetypal_strength)
        print_parameter("Ego Dissolution", settings.ego_dissolution)
        print_parameter("Symbol Universality", settings.symbol_universality)

        print("\nDerived Parameters:")
        print_parameter("Narrative Voice", settings.narrative_voice)
        print_parameter("Symbol Filter Threshold", settings.symbol_filter_threshold)
        print_parameter("Detail Obscuration", settings.detail_obscuration)
        print_parameter(
            "Emotional Universalization", settings.emotional_universalization
        )

        print("\nDescription:")
        print(f"  {slider.get_slider_description(slider_value)}")


def demo_parameter_progression():
    """Demo 2: How parameters progress across slider."""
    print_header("Demo 2: Parameter Progression Across Slider Range")

    slider = ConsciousnessSlider()

    # Test at key positions
    positions = [0.0, 0.165, 0.33, 0.495, 0.66, 0.83, 1.0]
    position_names = [
        "Start (0%)",
        "Individual Mid (16.5%)",
        "Boundary (33%)",
        "Shared Mid (49.5%)",
        "Boundary (66%)",
        "Universal Mid (83%)",
        "End (100%)",
    ]

    print("\nParameter Progression Table:")
    print(
        f"{'Position':<20} {'Level':<12} {'Privacy':<10} {'Archetype':<10} {'Ego Diss':<10} {'Participants':<15}"
    )
    print("-" * 77)

    for pos, name in zip(positions, position_names):
        settings = slider.get_settings(pos)
        print(
            f"{name:<20} {settings.level.value:<12} "
            f"{settings.privacy_gradient:>8.2f}  {settings.archetypal_strength:>8.2f}  "
            f"{settings.ego_dissolution:>8.2f}  {settings.max_participants:>12}"
        )


def demo_dream_adjustment():
    """Demo 3: Dream adjustment at each consciousness level."""
    print_header("Demo 3: Dream Adjustment at Each Level")

    slider = ConsciousnessSlider()

    base_dream = {
        "content": "I am trapped in this situation. I feel helpless and abandoned. "
        "My identity is dissolving.",
        "participants": ["person_a", "person_b", "person_c"],
        "emotional_essence": {
            "primary_emotion": "trapped",
            "intensity": 0.9,
            "secondary_emotions": ["helpless", "abandoned"],
        },
        "symbols": [
            {"name": "cage", "universality": 0.95},
            {"name": "shadow", "universality": 0.85},
        ],
    }

    print("\nBase Dream Content:")
    print(f"  Original: \"{base_dream['content']}\"")
    print(f"  Participants: {base_dream['participants']}")
    print(f"  Primary Emotion: {base_dream['emotional_essence']['primary_emotion']}")

    levels = [
        (0.15, "INDIVIDUAL", "Personal, private dream"),
        (0.50, "SHARED", "Collaborative dream for small group"),
        (0.85, "UNIVERSAL", "Collective dream for all residents"),
    ]

    for slider_value, level_name, description in levels:
        print_header(f"{level_name} ({slider_value * 100:.0f}%)", level=2)
        print(f"Description: {description}")

        settings = slider.get_settings(slider_value)
        adjusted = slider.adjust_dream_parameters(base_dream, settings)

        print("\nAdjusted Dream Content:")
        print(f"  Modified: \"{adjusted['content']}\"")

        print("\nAdjustment Details:")
        if "consciousness_filtering" in adjusted:
            print_parameter(
                "Participants Filtered",
                f"{adjusted['consciousness_filtering']['original_count']} -> "
                f"{adjusted['consciousness_filtering']['selected_count']}",
            )
        elif "participants" in adjusted:
            print_parameter("Participants", len(adjusted["participants"]))

        if "narrative_adjustments" in adjusted:
            narr = adjusted["narrative_adjustments"]
            print_parameter("Narrative Voice", narr["narrative_voice"])
            print_parameter("Perspective", narr["perspective"])
            print_parameter(
                "Ego Prominence", f"{narr['ego_prominence']:.2f} (1.0=high)"
            )

        if "privacy_instructions" in adjusted:
            priv = adjusted["privacy_instructions"]
            print_parameter("Privacy Gradient", f"{priv['privacy_gradient']:.2f}")
            print_parameter("Privacy Mode", priv["privacy_mode"])

        if "symbol_requirements" in adjusted:
            symb = adjusted["symbol_requirements"]
            print_parameter(
                "Min Symbol Universality", f"{symb['minimum_universality']:.2f}"
            )


def demo_ego_reference_handling():
    """Demo 4: Ego reference adjustment across consciousness levels."""
    print_header("Demo 4: Ego Reference Handling")

    slider = ConsciousnessSlider()

    dreams = [
        {
            "name": "Personal Achievement",
            "content": "I accomplished my goal. I feel proud of myself.",
        },
        {
            "name": "Relational Conflict",
            "content": "I hurt my partner. I regret my actions. My behavior was selfish.",
        },
        {
            "name": "Existential Reflection",
            "content": "I am mortality. I am impermanence. My life is fleeting.",
        },
    ]

    slider_values = [0.15, 0.50, 0.85]
    level_names = ["INDIVIDUAL", "SHARED", "UNIVERSAL"]

    for dream_info in dreams:
        print_header(f"Dream: {dream_info['name']}", level=2)
        print(f"\nOriginal: \"{dream_info['content']}\"")

        print("\nTransformations Across Consciousness Levels:")
        print("-" * 70)

        for slider_value, level_name in zip(slider_values, level_names):
            settings = slider.get_settings(slider_value)
            adjusted = slider.adjust_dream_parameters(dream_info, settings)

            print(f"\n  {level_name} ({slider_value * 100:.0f}%, {settings.narrative_voice}):")
            print(f"    \"{adjusted['content']}\"")


def demo_privacy_modes():
    """Demo 5: Privacy mode transitions."""
    print_header("Demo 5: Privacy Mode Transitions")

    slider = ConsciousnessSlider()

    print("\nPrivacy Mode Progression:")
    print("-" * 70)
    print(
        f"{'Slider':<10} {'Privacy':<10} {'Mode':<15} {'Detail Level':<15} {'Privacy':<15}"
    )
    print("-" * 70)

    for slider_value in [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]:
        settings = slider.get_settings(slider_value)

        detail_level = 1.0 - settings.detail_obscuration

        trend_indicator = "down" if settings.privacy_gradient < 0.4 else "changing" if settings.privacy_gradient < 0.7 else "up"
        print(
            f"{slider_value:>6.0%}      "
            f"{settings.privacy_gradient:>8.2f}  "
            f"{settings.privacy_mode.value:<15} "
            f"{detail_level:>14.2f}  "
            f"{trend_indicator:<15}"
        )


def demo_narrative_voice_transitions():
    """Demo 6: Narrative voice transitions across consciousness levels."""
    print_header("Demo 6: Narrative Voice Transitions")

    slider = ConsciousnessSlider()

    scenario = "There is suffering in my life. This suffering connects me to all beings."

    print(f"\nBase Scenario: \"{scenario}\"")

    print("\nNarrative Voice Transformations:")
    print("-" * 70)

    for slider_value in [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]:
        settings = slider.get_settings(slider_value)

        # Manual transformation based on narrative voice
        if settings.narrative_voice == "first_person":
            transformed = scenario.replace("There is", "I experience").replace(
                "This", "My"
            )
        elif settings.narrative_voice == "third_person":
            transformed = scenario.replace("There is", "One experiences").replace(
                "This", "That"
            )
        else:  # collective
            transformed = scenario.replace("There is", "We experience").replace(
                "This", "Our"
            )

        print(
            f"\n  {slider_value * 100:>5.0f}% ({settings.narrative_voice}):"
            f"\n    \"{transformed}\""
        )


def demo_participant_filtering():
    """Demo 7: Participant filtering based on consciousness level."""
    print_header("Demo 7: Participant Filtering by Consciousness Level")

    slider = ConsciousnessSlider()

    all_participants = [
        "alice",
        "bob",
        "charlie",
        "diana",
        "eve",
        "frank",
        "grace",
        "henry",
        "iris",
        "jack",
    ]

    base_dream = {"content": "A dream", "participants": all_participants}

    print(f"\nAll Available Participants ({len(all_participants)}): {', '.join(all_participants)}")

    print("\nParticipant Selection by Consciousness Level:")
    print("-" * 70)

    levels = [
        (0.15, "INDIVIDUAL", 1),
        (0.35, "SHARED (early)", 2),
        (0.50, "SHARED (mid)", 3),
        (0.65, "SHARED (late)", 4),
        (0.80, "UNIVERSAL (early)", 6),
        (0.95, "UNIVERSAL (late)", 999),
    ]

    for slider_value, level_name, expected_max in levels:
        settings = slider.get_settings(slider_value)
        adjusted = slider.adjust_dream_parameters(base_dream, settings)

        if "consciousness_filtering" in adjusted:
            selected_count = adjusted["consciousness_filtering"]["selected_count"]
            selected = adjusted["participants"]
            participants_str = ", ".join(selected) if isinstance(selected, list) else "all"
        else:
            selected_count = len(adjusted.get("participants", all_participants))
            participants_str = ", ".join(
                adjusted.get("participants", all_participants)[:5]
            )
            if selected_count > 5:
                participants_str += f", ... ({selected_count} total)"

        print(
            f"\n  {level_name:<20} (max {settings.max_participants:>3}): {selected_count:>2} selected"
        )
        print(f"    Selected: {participants_str}")


def demo_complete_workflow():
    """Demo 8: Complete workflow from slider to adjusted dream."""
    print_header("Demo 8: Complete Workflow - Slider to Adjusted Dream")

    slider = ConsciousnessSlider()

    test_cases = [
        (
            0.1,
            "Personal Struggle",
            "I am broken. I am ashamed. I cannot face myself.",
        ),
        (
            0.55,
            "Family Secret",
            "I betrayed my family. I am afraid they will discover my deception.",
        ),
        (
            0.95,
            "Human Condition",
            "I am mortality. I am inevitable decline. I am the human condition.",
        ),
    ]

    for slider_value, title, dream_content in test_cases:
        print_header(f"Workflow: {title}", level=2)

        print(f"\nSlider Position: {slider_value * 100:.0f}%")

        # Step 1: Get settings
        settings = slider.get_settings(slider_value)
        print(f"\nStep 1 - Get Settings:")
        print(f"  Level: {settings.level.value}")
        print(f"  Max Participants: {settings.max_participants}")
        print(f"  Privacy Mode: {settings.privacy_mode.value}")
        print(f"  Narrative Voice: {settings.narrative_voice}")

        # Step 2: Create base dream
        base_dream = {
            "content": dream_content,
            "participants": ["self", "friend1", "friend2", "public"],
        }
        print(f"\nStep 2 - Base Dream:")
        print(f"  Content: \"{base_dream['content']}\"")
        print(f"  Participants: {len(base_dream['participants'])}")

        # Step 3: Adjust dream
        adjusted = slider.adjust_dream_parameters(base_dream, settings)
        print(f"\nStep 3 - Adjusted Dream:")
        print(f"  Modified Content: \"{adjusted['content']}\"")
        if "participants" in adjusted:
            print(f"  Participants: {len(adjusted['participants'])}")

        # Step 4: Summary
        print(f"\nStep 4 - Summary:")
        print(f"  Privacy: {adjusted['privacy_instructions']['privacy_mode']}")
        print(
            f"  Min Symbol Universality: "
            f"{adjusted['symbol_requirements']['minimum_universality']:.2f}"
        )
        print(
            f"  Ego Prominence: {adjusted['narrative_adjustments']['ego_prominence']:.2f}"
        )


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print(" CONSCIOUSNESS SLIDER DEMO")
    print(" Individual -> Shared -> Universal Dream Consciousness")
    print("=" * 80)

    try:
        demo_slider_overview()
        demo_parameter_progression()
        demo_dream_adjustment()
        demo_ego_reference_handling()
        demo_privacy_modes()
        demo_narrative_voice_transitions()
        demo_participant_filtering()
        demo_complete_workflow()

        print_header("Demo Complete!")
        print("\nKey Takeaways:")
        print("  1. Consciousness slider smoothly transitions from 0-1")
        print("  2. Three levels: INDIVIDUAL (0-0.33), SHARED (0.33-0.66), UNIVERSAL (0.66-1.0)")
        print("  3. All parameters interpolate smoothly within levels")
        print("  4. Ego references automatically adjust based on narrative voice")
        print("  5. Privacy gradient, archetypal strength, and other settings scale with slider")
        print("  6. Dream adjustments preserve emotional essence while transforming form")
        print("  7. Integration points: symbolic_encoder, dream_matching, narrative_generator")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
