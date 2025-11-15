"""
Elle Core - Basic Demo

Demonstrates Elle's behavior across different scenarios.
"""
import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from elle_core.models import (
    SceneSnapshot,
    SceneObject,
    ObjectType,
    UserState,
    UserIntent,
    EnergyLevel,
    FocusState,
    MoodState,
)
from elle_core.engine import create_engine, ElleRequest
from elle_core.core import create_llm_client


def create_workshop_scene():
    """Cluttered workshop - intervention needed."""
    return (
        SceneSnapshot(
            description="Shed workshop, sawdust covering workbench, tools scattered everywhere",
            location="shed workshop",
            objects=[
                SceneObject(id="shop_vac_1", name="shop vacuum", type=ObjectType.TOOL, accessible=True),
                SceneObject(id="workbench_1", name="workbench", type=ObjectType.FURNITURE, accessible=False),
                SceneObject(id="toolbox_red", name="red toolbox", type=ObjectType.CONTAINER),
            ],
        ),
        UserState(energy=EnergyLevel.MEDIUM, focus=FocusState.BLOCKED, mood=MoodState.FRUSTRATED),
        UserIntent(description="Build a birdhouse", explicit=False, confidence=0.8),
    )


def create_garden_scene():
    """Garden flow state - protection mode."""
    return (
        SceneSnapshot(
            description="Vegetable garden bed, user weeding steadily, bucket half full, morning sun",
            location="vegetable garden",
            objects=[
                SceneObject(id="bucket_1", name="bucket", type=ObjectType.CONTAINER),
                SceneObject(id="garden_bed_1", name="garden bed", type=ObjectType.STRUCTURE),
            ],
        ),
        UserState(energy=EnergyLevel.HIGH, focus=FocusState.DEEP, mood=MoodState.CALM),
        None,  # No explicit intent
    )


def create_fence_scene():
    """Fence reminder - silent execution."""
    return (
        SceneSnapshot(
            description="User looking at sagging fence section",
            location="fence area",
            objects=[
                SceneObject(id="fence_section_3", name="fence section 3", type=ObjectType.STRUCTURE),
            ],
        ),
        UserState(energy=EnergyLevel.MEDIUM, focus=FocusState.SHALLOW, mood=MoodState.NEUTRAL),
        UserIntent(description="Remind me about this fence next week", explicit=True),
    )


def create_hive_scene():
    """Hive inspection - deferred observation."""
    return (
        SceneSnapshot(
            description="Bee hive, smoker lit, user carefully lifting frames and checking brood",
            location="apiary",
            objects=[
                SceneObject(id="hive_3", name="hive 3", type=ObjectType.STRUCTURE),
                SceneObject(id="smoker_1", name="smoker", type=ObjectType.EQUIPMENT),
            ],
        ),
        UserState(energy=EnergyLevel.HIGH, focus=FocusState.DEEP, mood=MoodState.ENGAGED),
        None,
    )


async def run_scenario(engine, name: str, scene, user_state, user_intent):
    """Run a single scenario and display results."""
    print("\n" + "=" * 70)
    print(f"  Scenario: {name}")
    print("=" * 70)

    print(f"\n📍 Location: {scene.location}")
    print(f"🎬 Scene: {scene.description}")
    print(f"👤 User state: {user_state}")
    if user_intent:
        print(f"🎯 {user_intent}")

    # Create request
    request = ElleRequest(
        scene=scene,
        user_state=user_state,
        user_intent=user_intent,
    )

    # Get Elle's response
    print("\n⚙️ Calling Elle...\n")
    action = await engine.handle(request)

    # Display response
    mode_emoji = {
        "idle": "😶",
        "focus_assist": "🤝",
        "silent_indicator": "🔵",
        "deferred": "⏸️",
    }
    emoji = mode_emoji.get(action.mode.value, "🤖")

    print(f"Mode: {emoji} {action.mode.value} (priority: {action.priority.value})")

    if action.utterance:
        print(f'Elle says: "{action.utterance}"')
    else:
        print("Elle: (silent)")

    if action.visuals:
        print(f"Visuals: {', '.join(str(v) for v in action.visuals)}")

    if action.followups:
        print(f"Followups: {', '.join(str(f) for f in action.followups)}")

    print(f"\n💭 Reasoning: {action.reasoning}")

    input("\nPress Enter to continue...")


async def main():
    """Run all demo scenarios."""
    print("\n" + "=" * 70)
    print("  Elle Core - Demo")
    print("  Calm AR Companion - Behavior Demonstrations")
    print("=" * 70)

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  Warning: No API key found!")
        print("\nSet one of these environment variables:")
        print("  export ANTHROPIC_API_KEY='your-key'  # For Claude")
        print("  export OPENAI_API_KEY='your-key'     # For OpenAI")
        print("\nOr run with Ollama (local):")
        print("  # Start Ollama with: ollama serve")
        print("  # Then this demo will use local models")
        print("\n" + "=" * 70)

        # Try to use Ollama as fallback
        provider = "ollama"
        print("\n🔄 Attempting to use Ollama (local)...")
    elif os.getenv("ANTHROPIC_API_KEY"):
        provider = "claude"
        print("\n✅ Using Claude (Anthropic)")
    else:
        provider = "openai"
        print("\n✅ Using OpenAI")

    # Create LLM client
    try:
        llm_client = create_llm_client(provider=provider)
    except Exception as e:
        print(f"\n❌ Failed to create LLM client: {e}")
        print("\nMake sure you have:")
        print("  1. API key set (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        print("  2. Required package installed:")
        print("     pip install anthropic  # For Claude")
        print("     pip install openai     # For OpenAI")
        print("     pip install aiohttp    # For Ollama")
        return

    # Create engine
    engine = create_engine(llm_client=llm_client)

    print(f"\n🚀 Elle engine initialized with {provider}")
    print("\nRunning 4 scenarios to demonstrate Elle's behavior:\n")

    # Run scenarios
    scenarios = [
        ("Cluttered Workshop", *create_workshop_scene()),
        ("Garden Flow State", *create_garden_scene()),
        ("Fence Reminder", *create_fence_scene()),
        ("Hive Inspection", *create_hive_scene()),
    ]

    for name, scene, user_state, user_intent in scenarios:
        await run_scenario(engine, name, scene, user_state, user_intent)

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("  • Elle protects flow states (garden, hive)")
    print("  • Elle intervenes when helpful (workshop)")
    print("  • Elle executes silently when appropriate (fence)")
    print("  • Elle defers by default when uncertain")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
