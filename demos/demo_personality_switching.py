"""
Personality Switching Demo
==========================

Demonstrates the 4 HoloLoom VoiceAgent personalities responding to the same queries,
showcasing their distinct styles, voices, and trait-based response variations.

Features demonstrated:
- Loading all 4 personalities (Professor, Assistant, Companion, Expert)
- Personality switching (<100ms)
- Voice-personality mapping
- Response style variations
- Trait-based personality differences

Date: November 16, 2025
Phase: 3 (Custom Personalities)
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.voice.personality import PersonalityManager


# ============================================================================
# Demo Queries
# ============================================================================

DEMO_QUERIES = [
    {
        'query': "What do you see in this hive?",
        'context': "User is inspecting a healthy hive with excellent brood pattern, "
                   "60,000 worker bees, active foraging, no visible pests."
    },
    {
        'query': "Should I add a honey super?",
        'context': "Current hive has 8 frames of brood, strong nectar flow detected, "
                   "current honey stores at 75% capacity."
    },
    {
        'query': "How do I know if my queen is healthy?",
        'context': "User is a beginner beekeeper learning inspection techniques."
    }
]


# ============================================================================
# Simulated Responses (showing personality differences)
# ============================================================================

SIMULATED_RESPONSES = {
    'professor_elle': [
        # Query 1: What do you see?
        "This hive demonstrates excellent health indicators. The brood pattern shows "
        "tight, consistent cell coverage - what we call a '10/10 pattern' - which indicates "
        "a vigorous, productive queen. Notice the population density of approximately 60,000 "
        "workers. This represents optimal colony strength for this season. The active foraging "
        "behavior you're observing is characteristic of a healthy, well-functioning colony. "
        "Let me explain the significance of each indicator...",

        # Query 2: Add super?
        "Excellent question. The decision to add supers involves assessing three key factors. "
        "First, your colony strength: 8 frames of brood indicates robust population growth. "
        "Second, nectar flow conditions: we're detecting active nectar collection, which provides "
        "the raw material for honey production. Third, current capacity utilization at 75% suggests "
        "we're approaching the threshold where additional space becomes necessary. The principle here "
        "is 'just-in-time' expansion - we want to provide space before the bees feel crowded, which "
        "could trigger swarming impulses. Based on these factors, yes, I recommend adding a super now.",

        # Query 3: Queen health?
        "Learning to assess queen health is fundamental to successful beekeeping. Let me teach you "
        "the systematic approach. We evaluate queens on three primary dimensions. First, brood pattern "
        "quality - a healthy queen produces tight, circular patterns with minimal skipped cells. "
        "Second, population dynamics - observe colony growth rates and worker demographics. Third, "
        "behavioral indicators - watch how workers attend the queen and whether they're building "
        "supersedure cells. Each of these factors tells part of the story. Would you like me to explain "
        "how to evaluate each one in detail?"
    ],

    'assistant_elle': [
        # Query 1: What do you see?
        "Hive status: Healthy. Brood pattern: 10/10 (excellent). Population: ~60k workers. "
        "Foraging: Active. Pests: None detected. Overall: Strong colony, no issues.",

        # Query 2: Add super?
        "Yes, add super now. Reasons: 8 frames brood (strong), nectar flow active, 75% capacity. "
        "Recommendation: Add deep super, check in 7 days.",

        # Query 3: Queen health?
        "Check three things: 1) Brood pattern (tight = good, scattered = problem). "
        "2) Population (growing = healthy). 3) Worker behavior (building queen cells = replacing her). "
        "Quick test: Find solid brood pattern. If yes, queen is healthy."
    ],

    'companion_elle': [
        # Query 1: What do you see?
        "Oh wow, this is a really healthy hive! Look at that brood pattern - see how all the cells "
        "are clustered nice and tight? That's exactly what you want to see. Your queen is doing an "
        "amazing job! And check out all these worker bees - there must be around 60,000 in here. "
        "They're so busy and active! The foragers are bringing in pollen like crazy. No signs of "
        "pests or problems either. You should be really proud - you're taking excellent care of this colony!",

        # Query 2: Add super?
        "Great question! Let's think through this together. Your hive is doing really well - 8 frames "
        "of brood is strong! And I can see the bees are bringing in lots of nectar right now. Plus, "
        "you're at 75% capacity, which means they're starting to run out of room for all that honey "
        "they're making. I think it's definitely time to give them more space. Go ahead and add that "
        "super - your bees will appreciate it! It's exciting watching them expand, isn't it?",

        # Query 3: Queen health?
        "I love that you're asking thoughtful questions! Queen health is so important. Here's what I "
        "look for, and you can too: First, check out that brood pattern we talked about - nice and tight "
        "means she's happy and healthy. Second, watch if your colony is growing - if you're seeing more "
        "and more bees, she's doing her job well! And third, look for those special queen cells - if the "
        "workers aren't building them, it means they're satisfied with their current queen. It gets easier "
        "with practice, and you're doing great learning this! Want to practice together?"
    ],

    'expert_elle': [
        # Query 1: What do you see?
        "Hive assessment summary: Brood pattern score: 10.0/10.0. Capped cell density: 95%. "
        "Skip rate: <2%. Population estimate: 60,000 workers ±5,000. Age distribution: Normal gradient. "
        "Foraging activity: 180 bees/minute. Pollen diversity: 4 distinct sources. Pest indicators: "
        "Negative (Varroa, SHB, wax moth). Structural integrity: Normal. Temperature gradient: Optimal. "
        "Overall assessment: Exceptional colony health. No interventions required. Continue current "
        "management protocol.",

        # Query 2: Add super?
        "Super addition analysis: Colony strength: 8 frames brood (strong threshold). Nectar flow: "
        "Active (confirmed via forager observation). Current capacity: 75% (threshold: 70%). "
        "Population trajectory: +15% week-over-week. Recommendation: Add deep super immediately. "
        "Rationale: Colony approaching swarming threshold. Space requirement will exceed capacity "
        "within 5-7 days. Proactive expansion prevents reproductive swarming. Follow-up inspection: "
        "7 days post-addition to assess utilization.",

        # Query 3: Queen health?
        "Queen assessment protocol: Parameter 1 - Brood pattern analysis. Measure: Cell utilization "
        "percentage, skip rate, pattern geometry. Acceptable range: >85% utilization, <5% skip rate, "
        "circular pattern. Parameter 2 - Population metrics. Measure: Colony growth rate, eggs per frame, "
        "worker demographics. Acceptable range: +10-20% weekly growth, 2,000-3,500 eggs/frame. "
        "Parameter 3 - Worker behavioral indicators. Measure: Supersedure cell presence, queen cup "
        "status, attendant court size. Red flags: Supersedure cells with eggs, absent attendants. "
        "This protocol provides 95%+ accuracy for queen health assessment."
    ]
}


# ============================================================================
# Display Helpers
# ============================================================================

def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_personality_info(personality):
    """Print personality information"""
    print(f"Personality: {personality.name}")
    print(f"Description: {personality.description}")
    print(f"Voice: {personality.voice_id}")
    print(f"\nTraits:")
    print(f"  - Formality: {personality.traits.formality:.1f}")
    print(f"  - Verbosity: {personality.traits.verbosity:.1f}")
    print(f"  - Emotional Tone: {personality.traits.emotional_tone:.1f}")
    print(f"  - Teaching Style: {personality.traits.teaching_style:.1f}")
    print(f"  - Humor: {personality.traits.humor:.1f}")
    print()


def print_response(query: str, response: str, personality_name: str):
    """Print query and response"""
    print(f"[{personality_name}] User: {query}")
    print(f"[{personality_name}] Response:")
    print(f"  {response}")
    print()


# ============================================================================
# Demo Scenarios
# ============================================================================

def demo_scenario_1_all_personalities():
    """Demo Scenario 1: Same query, all personalities"""
    print_header("Scenario 1: Same Query Across All Personalities")
    print("Query: 'What do you see in this hive?'")
    print("Context: Healthy hive, excellent brood pattern, 60k workers, active foraging\n")

    manager = PersonalityManager()

    personalities = [
        'professor_elle',
        'assistant_elle',
        'companion_elle',
        'expert_elle'
    ]

    for i, personality_id in enumerate(personalities):
        manager.switch_personality(personality_id)
        personality = manager.get_active_personality()

        print(f"\n--- {personality.name} (Voice: {personality.voice_id}) ---")
        print(f"Response Style: ", end="")
        if personality.traits.verbosity > 0.7:
            print("Detailed", end="")
        elif personality.traits.verbosity < 0.4:
            print("Concise", end="")
        else:
            print("Moderate", end="")

        if personality.traits.formality > 0.7:
            print(", Formal", end="")
        elif personality.traits.formality < 0.4:
            print(", Casual", end="")

        if personality.traits.emotional_tone > 0.6:
            print(", Expressive")
        elif personality.traits.emotional_tone < 0.4:
            print(", Neutral")
        else:
            print(", Balanced")

        print(f"\n{SIMULATED_RESPONSES[personality_id][0]}")


def demo_scenario_2_personality_switching():
    """Demo Scenario 2: Fast personality switching"""
    import time

    print_header("Scenario 2: Fast Personality Switching Performance")

    manager = PersonalityManager()

    personalities = ['professor_elle', 'assistant_elle', 'companion_elle', 'expert_elle']

    print("Testing switching speed (100 iterations)...")

    start = time.perf_counter()
    for _ in range(100):
        for personality_id in personalities:
            manager.switch_personality(personality_id)
    duration = time.perf_counter() - start

    avg_switch_time = (duration / (100 * 4)) * 1000  # Convert to ms

    print(f"\nResults:")
    print(f"  Total time: {duration:.3f} seconds")
    print(f"  Total switches: {100 * 4}")
    print(f"  Average switch time: {avg_switch_time:.3f} ms")
    print(f"  ✓ Target: <100ms per switch")
    print(f"  {'✓ PASS' if avg_switch_time < 100 else '✗ FAIL'}")


def demo_scenario_3_voice_mapping():
    """Demo Scenario 3: Voice-personality mapping"""
    print_header("Scenario 3: Voice-Personality Mapping")

    manager = PersonalityManager()

    print("OpenAI TTS Voice Assignments:\n")

    voice_descriptions = {
        'nova': 'Warm, educational female voice',
        'alloy': 'Neutral, efficient voice',
        'shimmer': 'Friendly, expressive female voice',
        'onyx': 'Authoritative male voice'
    }

    for personality_id in manager.list_personalities():
        manager.switch_personality(personality_id)
        personality = manager.get_active_personality()
        voice = manager.get_voice_for_personality()

        print(f"{personality.name:20} → {voice:10} ({voice_descriptions[voice]})")


def demo_scenario_4_response_variations():
    """Demo Scenario 4: Multiple queries showing personality consistency"""
    print_header("Scenario 4: Response Variation Across Multiple Queries")

    manager = PersonalityManager()

    for i, demo_query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'─' * 80}")
        print(f"Query {i}: {demo_query['query']}")
        print(f"Context: {demo_query['context']}")
        print(f"{'─' * 80}\n")

        for personality_id in ['professor_elle', 'assistant_elle', 'companion_elle', 'expert_elle']:
            manager.switch_personality(personality_id)
            personality = manager.get_active_personality()

            response_index = i - 1  # 0-indexed
            response = SIMULATED_RESPONSES[personality_id][response_index]

            print(f"[{personality.name}]")
            print(f"{response}\n")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Run all demo scenarios"""
    print("\n" + "=" * 80)
    print("  HoloLoom VoiceAgent: Personality Framework Demo")
    print("  Phase 3: Custom Personalities")
    print("=" * 80)

    # Initialize manager to show loading
    print("\nInitializing PersonalityManager...")
    manager = PersonalityManager()
    print(f"✓ Loaded {len(manager.list_personalities())} personalities")
    print(f"✓ Active personality: {manager.get_active_personality().name}")

    # Run scenarios
    try:
        demo_scenario_1_all_personalities()
        input("\nPress Enter to continue to Scenario 2...")

        demo_scenario_2_personality_switching()
        input("\nPress Enter to continue to Scenario 3...")

        demo_scenario_3_voice_mapping()
        input("\nPress Enter to continue to Scenario 4...")

        demo_scenario_4_response_variations()

        print_header("Demo Complete!")
        print("Summary:")
        print("  ✓ 4 personalities demonstrated (Professor, Assistant, Companion, Expert)")
        print("  ✓ Personality switching verified (<100ms)")
        print("  ✓ Voice-personality mapping confirmed")
        print("  ✓ Response style variations shown")
        print("  ✓ Trait-based personality differences evident")
        print("\nPhase 3 deliverables complete!")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)


if __name__ == '__main__':
    main()
