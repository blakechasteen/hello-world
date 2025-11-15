"""
Elle Core - CLI Adapter

Command-line interface for testing Elle.
"""
import asyncio
from typing import Optional
from datetime import datetime

from ..models import (
    SceneSnapshot,
    SceneObject,
    ObjectType,
    UserState,
    UserIntent,
    EnergyLevel,
    FocusState,
    MoodState,
)
from ..engine import ElleEngine, ElleRequest


class CLI:
    """
    Interactive command-line interface for Elle.

    Allows manual creation of scenes and observation of Elle's responses.
    """

    def __init__(self, engine: ElleEngine):
        self.engine = engine

    async def run_interactive(self):
        """Run interactive session."""
        print("\n" + "=" * 60)
        print("  Elle Core - Interactive CLI")
        print("=" * 60)
        print("\nDescribe scenes and watch Elle respond.\n")
        print("Commands:")
        print("  /quit - Exit")
        print("  /help - Show this help")
        print("  /scenario <name> - Run a predefined scenario")
        print("=" * 60 + "\n")

        while True:
            try:
                command = input("\n> ").strip()

                if not command:
                    continue

                if command == "/quit":
                    print("\nGoodbye! 👋")
                    break

                if command == "/help":
                    self._print_help()
                    continue

                if command.startswith("/scenario"):
                    parts = command.split(maxsplit=1)
                    scenario_name = parts[1] if len(parts) > 1 else "workshop"
                    await self._run_scenario(scenario_name)
                    continue

                # Otherwise, treat as a scene description
                await self._process_description(command)

            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def _print_help(self):
        """Print help text."""
        print("\nHow to use:")
        print("  1. Describe a scene in natural language")
        print("  2. Elle will respond based on the context")
        print("  3. Try different scenarios to see Elle's behavior\n")
        print("Example:")
        print('  > Cluttered workshop, sawdust everywhere, frustrated\n')
        print("Predefined scenarios:")
        print("  - workshop: Cluttered shed workshop")
        print("  - garden: Weeding in flow state")
        print("  - fence: Reminder request")
        print("  - hive: Bee hive inspection")

    async def _process_description(self, description: str):
        """Process a natural language scene description."""
        print(f"\n🎬 Processing: {description}\n")

        # Simple heuristic parsing (in production, use NLP)
        scene, user_state, intent = self._parse_description(description)

        # Show what we understood
        print("📋 Interpreted as:")
        print(f"   Location: {scene.location}")
        print(f"   Scene: {scene.description}")
        print(f"   User state: {user_state}")
        if intent:
            print(f"   {intent}")
        print()

        # Create request and get Elle's response
        request = ElleRequest(
            scene=scene,
            user_state=user_state,
            user_intent=intent,
        )

        action = await self.engine.handle(request)

        # Display Elle's response
        self._display_action(action)

    def _parse_description(self, description: str) -> tuple:
        """
        Simple heuristic parsing of scene description.

        In production, this would use NLP to extract structure.
        """
        desc_lower = description.lower()

        # Infer location
        if any(word in desc_lower for word in ["workshop", "shed", "garage"]):
            location = "shed workshop"
        elif any(word in desc_lower for word in ["garden", "weeding", "bed"]):
            location = "vegetable garden"
        elif any(word in desc_lower for word in ["fence", "gate"]):
            location = "fence area"
        elif any(word in desc_lower for word in ["hive", "bee", "smoker"]):
            location = "apiary"
        else:
            location = "unknown location"

        # Infer user state
        if any(word in desc_lower for word in ["frustrated", "annoyed", "stuck"]):
            mood = MoodState.FRUSTRATED
            focus = FocusState.BLOCKED
            energy = EnergyLevel.MEDIUM
        elif any(word in desc_lower for word in ["flow", "steady", "rhythm", "focused"]):
            mood = MoodState.CALM
            focus = FocusState.DEEP
            energy = EnergyLevel.HIGH
        elif any(word in desc_lower for word in ["tired", "low energy"]):
            mood = MoodState.NEUTRAL
            focus = FocusState.SCATTERED
            energy = EnergyLevel.LOW
        else:
            mood = MoodState.NEUTRAL
            focus = FocusState.SHALLOW
            energy = EnergyLevel.MEDIUM

        user_state = UserState(energy=energy, focus=focus, mood=mood)

        # Check for explicit intent
        intent = None
        if "remind me" in desc_lower:
            intent = UserIntent(
                description="Set a reminder for later",
                explicit=True,
            )

        # Create scene
        scene = SceneSnapshot(
            description=description,
            location=location,
        )

        # Add some default objects based on location
        if "workshop" in location:
            scene.objects.extend([
                SceneObject(id="shop_vac_1", name="shop vacuum", type=ObjectType.TOOL),
                SceneObject(id="workbench_1", name="workbench", type=ObjectType.FURNITURE),
            ])
        elif "garden" in location:
            scene.objects.extend([
                SceneObject(id="bucket_1", name="bucket", type=ObjectType.CONTAINER),
                SceneObject(id="garden_bed_1", name="garden bed", type=ObjectType.STRUCTURE),
            ])

        return scene, user_state, intent

    def _display_action(self, action):
        """Display Elle's action in a nice format."""
        print("⚙️ Elle responds:\n")

        # Mode and priority
        mode_emoji = {
            "idle": "😶",
            "focus_assist": "🤝",
            "silent_indicator": "🔵",
            "deferred": "⏸️",
        }
        emoji = mode_emoji.get(action.mode.value, "🤖")
        print(f"   Mode: {emoji} {action.mode.value} (priority: {action.priority.value})")

        # Utterance
        if action.utterance:
            print(f'   Says: "{action.utterance}"')
        else:
            print("   Says: (silent)")

        # Visuals
        if action.visuals:
            print(f"   Visuals: {', '.join(str(v) for v in action.visuals)}")

        # Followups
        if action.followups:
            print(f"   Followups: {', '.join(str(f) for f in action.followups)}")

        # Reasoning
        print(f"\n   💭 Reasoning: {action.reasoning}")

    async def _run_scenario(self, name: str):
        """Run a predefined scenario."""
        scenarios = {
            "workshop": (
                "Cluttered shed workshop, sawdust covering workbench, tools scattered, "
                "looking frustrated and can't find workspace",
                MoodState.FRUSTRATED,
                FocusState.BLOCKED,
                EnergyLevel.MEDIUM,
            ),
            "garden": (
                "Weeding vegetable garden bed, steady rhythm, bucket half full, "
                "morning sun, in flow state",
                MoodState.CALM,
                FocusState.DEEP,
                EnergyLevel.HIGH,
            ),
            "fence": (
                "Looking at sagging fence section",
                MoodState.NEUTRAL,
                FocusState.SHALLOW,
                EnergyLevel.MEDIUM,
            ),
            "hive": (
                "Conducting bee hive inspection, smoker lit, carefully lifting frames",
                MoodState.ENGAGED,
                FocusState.DEEP,
                EnergyLevel.HIGH,
            ),
        }

        if name not in scenarios:
            print(f"\n❌ Unknown scenario: {name}")
            print(f"Available: {', '.join(scenarios.keys())}")
            return

        desc, mood, focus, energy = scenarios[name]

        print(f"\n🎬 Running scenario: {name}")
        print(f"   {desc}\n")

        await self._process_description(desc)
