"""
Ollama-Powered Neighborhood Demo

Run the neighborhood simulation with real LLM-powered conversations
using a local Ollama instance.

Requirements:
- Ollama running locally: ollama serve
- A model pulled: ollama pull llama3.2:3b

Author: Claude (NeuroHood Extension)
Date: November 2025
"""

import asyncio
from pathlib import Path

from neighborhood_engine import NeighborhoodEngine
from ollama_client import OllamaClient, OllamaDialogueGenerator, OllamaConfig


class OllamaDialogueGeneratorAdapter:
    """Adapter that wraps OllamaDialogueGenerator for NeighborhoodEngine."""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama_gen = OllamaDialogueGenerator(ollama_client)
        self.fallback_gen = None  # Will be set after init

    async def generate_line(self, speaker, listener, context, conversation_history):
        """Generate dialogue using Ollama with fallback."""
        try:
            return await self.ollama_gen.generate_line(
                speaker=speaker,
                listener=listener,
                context=context,
                conversation_history=conversation_history
            )
        except Exception as e:
            print(f"[Ollama] Error: {e}")
            if self.fallback_gen:
                return await self.fallback_gen.generate_line(
                    speaker, listener, context, conversation_history
                )
            return "..."


class OllamaNeighborhoodEngine(NeighborhoodEngine):
    """NeighborhoodEngine with Ollama LLM integration."""

    def __init__(self, ollama_config: OllamaConfig = None):
        # Create Ollama client
        self.ollama_client = OllamaClient(ollama_config or OllamaConfig())

        # Create dialogue adapter
        self._ollama_dialogue = OllamaDialogueGeneratorAdapter(self.ollama_client)

        # Initialize engine (will create template fallback)
        super().__init__()

        # Link fallback
        self._ollama_dialogue.fallback_gen = self.dialogue_gen

        # Track availability
        self._use_ollama = False

    async def initialize(self, load_path: Path = None):
        """Initialize with Ollama availability check."""
        await super().initialize(load_path)

        # Check if Ollama is available
        available = await self.ollama_client.is_available()
        if available:
            self._use_ollama = True
            # Replace dialogue generator with Ollama version
            self.dialogue_gen = self._ollama_dialogue
            models = await self.ollama_client.list_models()
            print(f"[Ollama] Connected! Available models: {models}")
        else:
            print("[Ollama] Not available - using template dialogue")
            print("  Start Ollama with: ollama serve")
            print("  Pull a model with: ollama pull llama3.2:3b")

    async def close(self):
        """Clean up Ollama client."""
        await self.ollama_client.close()


async def demo_ollama_conversations():
    """
    Demonstrate LLM-powered conversations in the neighborhood.
    """
    print("=" * 60)
    print("OLLAMA-POWERED NEIGHBORHOOD DEMO")
    print("=" * 60)

    # Create engine with Ollama
    config = OllamaConfig(
        model="llama3.2:3b",  # Fast, good for dialogue
        temperature=0.8,  # Creative
        max_tokens=100,  # Short responses
        timeout=30.0
    )

    engine = OllamaNeighborhoodEngine(config)
    await engine.initialize()

    # Add user
    user = engine.add_user("Blake")
    print(f"\nYou join the neighborhood as {user.name}")

    # Simulate some activity first
    print("\n--- NEIGHBORHOOD WAKING UP ---")
    for _ in range(5):
        events = engine.simulate_step()
        for e in events[:2]:  # Show first 2 events per step
            print(f"  {e['description']}")

    # Go to the cafe
    print("\n--- YOU GO TO THE CORNER CAFE ---")
    result = engine.user_go_to("corner_cafe")
    print(result)

    # Find someone to talk to
    agents_here = engine.get_agents_at("corner_cafe")
    residents = [a for a in agents_here if not a.is_user]

    if residents:
        other = residents[0]
        print(f"\n--- CONVERSATION WITH {other.name.upper()} ---")
        print(f"(Personality: {describe_personality(other.personality)})")
        print(f"(Mood: {describe_mood(other.emotions)})")

        # Start conversation
        conv = await engine.user_talk_to(other.agent_id, "Hey! Nice to see you here.")

        for line in conv.lines:
            print(f"\n  {line.speaker_name}: \"{line.text}\"")

        # Continue the conversation a few rounds
        user_lines = [
            "What brings you here today?",
            "That's interesting. How do you like the neighborhood?",
            "Any recommendations for places to visit?",
        ]

        for user_line in user_lines:
            print(f"\n  {user.name}: \"{user_line}\"")
            response = await engine.continue_conversation(conv, user_line)
            print(f"  {other.name}: \"{response}\"")

        print(f"\n[Conversation ended with {len(conv.lines)} exchanges]")

    else:
        print("\nThe cafe is empty right now.")

        # Try the garden
        print("\n--- TRYING THE COMMUNITY GARDEN ---")
        result = engine.user_go_to("community_garden")
        print(result)

        agents_here = engine.get_agents_at("community_garden")
        residents = [a for a in agents_here if not a.is_user]

        if residents:
            other = residents[0]
            print(f"\n--- CONVERSATION WITH {other.name.upper()} ---")
            conv = await engine.user_talk_to(other.agent_id, "Beautiful day for gardening!")

            for line in conv.lines:
                print(f"\n  {line.speaker_name}: \"{line.text}\"")

    # Run night cycle
    print("\n--- NIGHT FALLS ---")
    dreams = await engine.run_night_cycle()

    print("\nDreams of the neighborhood:")
    for agent_id, dream in list(dreams.items())[:3]:  # Show first 3
        agent = engine.get_agent(agent_id)
        print(f"\n  {agent.name}'s dream:")
        print(f"    Seeds: {dream['seed_symbols']}")
        if dream['description']:
            print(f"    \"{dream['description'][:80]}...\"")

    # Save state
    save_path = Path("./ollama_neighborhood.json")
    engine.save(save_path)

    # Cleanup
    await engine.close()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


def describe_personality(p: dict) -> str:
    """Describe personality in natural language."""
    traits = []
    if p.get("openness", 0.5) > 0.6:
        traits.append("creative")
    if p.get("extraversion", 0.5) > 0.6:
        traits.append("outgoing")
    elif p.get("extraversion", 0.5) < 0.4:
        traits.append("quiet")
    if p.get("agreeableness", 0.5) > 0.6:
        traits.append("friendly")
    if p.get("neuroticism", 0.5) > 0.6:
        traits.append("sensitive")
    return ", ".join(traits) if traits else "balanced"


def describe_mood(e: dict) -> str:
    """Describe mood in natural language."""
    moods = []
    if e.get("happiness", 0) > 0.3:
        moods.append("happy")
    elif e.get("happiness", 0) < -0.3:
        moods.append("sad")
    if e.get("anxiety", 0) > 0.3:
        moods.append("anxious")
    if e.get("energy", 0.5) < 0.3:
        moods.append("tired")
    return ", ".join(moods) if moods else "neutral"


async def interactive_mode():
    """
    Interactive mode - talk to residents freely.
    """
    print("=" * 60)
    print("INTERACTIVE NEIGHBORHOOD MODE")
    print("=" * 60)
    print("\nCommands:")
    print("  go <location>  - Visit a location (cafe, garden, park, etc.)")
    print("  talk <name>    - Talk to someone")
    print("  look           - Look around")
    print("  residents      - List all residents")
    print("  quit           - Exit")
    print()

    engine = OllamaNeighborhoodEngine()
    await engine.initialize()
    user = engine.add_user("You")

    current_conversation = None
    current_partner = None

    # Location shortcuts
    location_aliases = {
        "cafe": "corner_cafe",
        "garden": "community_garden",
        "park": "oak_park",
        "street": "main_street",
        "reading": "reading_corner",
        "center": "community_center",
        "music": "music_gazebo",
        "fountain": "plaza_fountain",
    }

    while True:
        try:
            if current_conversation:
                prompt = f"[talking to {current_partner}] > "
            else:
                prompt = "> "

            cmd = input(prompt).strip()

            if not cmd:
                continue

            if cmd.lower() == "quit":
                break

            # If in conversation, treat input as dialogue
            if current_conversation:
                if cmd.lower() == "bye" or cmd.lower() == "goodbye":
                    print(f"\n[You end the conversation with {current_partner}]")
                    current_conversation = None
                    current_partner = None
                    continue

                response = await engine.continue_conversation(current_conversation, cmd)
                print(f"\n  {current_partner}: \"{response}\"")
                continue

            parts = cmd.split(maxsplit=1)
            action = parts[0].lower()

            if action == "go":
                if len(parts) < 2:
                    print("Go where? (cafe, garden, park, street, reading, center, music, fountain)")
                    continue
                loc = parts[1].lower()
                loc_id = location_aliases.get(loc, loc)
                result = engine.user_go_to(loc_id)
                print(result)

            elif action == "talk":
                if len(parts) < 2:
                    # Show who's here
                    if user.location_id:
                        agents = engine.get_agents_at(user.location_id)
                        residents = [a for a in agents if not a.is_user]
                        if residents:
                            print("People here: " + ", ".join(a.name for a in residents))
                        else:
                            print("No one else is here.")
                    else:
                        print("Go somewhere first!")
                    continue

                name = parts[1].title()
                # Find agent by name
                target = None
                for aid, agent in engine.state.agents.items():
                    if agent.name.lower() == name.lower():
                        target = agent
                        break

                if not target:
                    print(f"Don't know anyone named {name}.")
                    continue

                print(f"\n[You approach {target.name}]")
                current_conversation = await engine.user_talk_to(target.agent_id, "Hey!")
                current_partner = target.name

                for line in current_conversation.lines:
                    print(f"  {line.speaker_name}: \"{line.text}\"")

            elif action == "look":
                if user.location_id:
                    loc = engine.state.locations.get(user.location_id, {})
                    print(f"\n{loc.get('name', 'Unknown')}")
                    print(loc.get('description', ''))

                    agents = engine.get_agents_at(user.location_id)
                    residents = [a for a in agents if not a.is_user]
                    if residents:
                        print(f"\nPeople here: {', '.join(a.name for a in residents)}")
                else:
                    print("You're not anywhere specific yet.")

            elif action == "residents":
                print("\nNeighborhood residents:")
                for aid, agent in engine.state.agents.items():
                    if not agent.is_user:
                        loc_id = engine.state.agent_locations.get(aid)
                        loc_name = engine.state.locations.get(loc_id, {}).get('name', 'Unknown') if loc_id else 'Unknown'
                        print(f"  {agent.name} - at {loc_name}")

            else:
                print(f"Unknown command: {action}")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    await engine.close()
    print("\nGoodbye!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_mode())
    else:
        asyncio.run(demo_ollama_conversations())
