"""
Simple Text Adventure powered by Elle's Game Engine

A minimal but complete example showing how to use Elle's narrative
intelligence to power a text-based game.

Run the service first:
    python -m apps.elle_game_engine.service

Then run this demo:
    python demos/demo_text_adventure_with_elle.py
"""

import asyncio
import httpx
from typing import Optional


class TextAdventureGame:
    """Simple text adventure using Elle for narrative."""

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.client = httpx.AsyncClient(base_url=api_url, timeout=30.0)

        # Game state
        self.current_scene = "dark_forest"
        self.player_name = "Adventurer"
        self.npcs_encountered = set()

        # Define NPCs per scene
        self.scene_npcs = {
            "dark_forest": [
                {"id": "old_wizard", "name": "Merlin", "role": "wizard", "mood": "mysterious", "location": "clearing"}
            ],
            "village_square": [
                {"id": "innkeeper", "name": "Bob", "role": "innkeeper", "mood": "friendly", "location": "inn"},
                {"id": "guard", "name": "Captain Williams", "role": "guard", "mood": "alert", "location": "gate"}
            ],
            "ancient_ruins": [
                {"id": "ghost", "name": "Ancient Guardian", "role": "spirit", "mood": "solemn", "location": "altar"}
            ]
        }

        # World state
        self.time_of_day = "night"
        self.weather = "fog"
        self.tension_level = "calm"

    async def get_elle_action(self, intent_type: str, target_npc_id: Optional[str] = None, raw_input: Optional[str] = None):
        """Call Elle's game engine API."""

        # Build game state
        game_state = {
            "scene_id": self.current_scene,
            "npcs": self.scene_npcs.get(self.current_scene, []),
            "player": {
                "name": self.player_name,
                "location": self.current_scene,
                "quest_stage": "intro",
                "reputation": "outsider"
            },
            "world": {
                "time_of_day": self.time_of_day,
                "weather": self.weather,
                "tension_level": self.tension_level
            },
            "tags": list(self.npcs_encountered)
        }

        # Build player intent
        player_intent = {
            "type": intent_type,
        }

        if target_npc_id:
            player_intent["target_npc_id"] = target_npc_id
        if raw_input:
            player_intent["raw_input"] = raw_input

        # Call API
        try:
            response = await self.client.post(
                "/elle/game/action",
                json={"game_state": game_state, "player_intent": player_intent}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"❌ Error calling Elle API: {e}")
            return None

    async def enter_scene(self):
        """Enter a new scene."""
        action = await self.get_elle_action("enter_scene")

        if action and action.get("world_reaction"):
            wr = action["world_reaction"]
            print(f"\n🌍 {wr['description']}")
        elif action and action.get("dialogue"):
            # Sometimes NPCs greet you
            for d in action["dialogue"]:
                print(f"\n💬 {d['text']}")

    async def talk_to_npc(self, npc_id: str, player_message: Optional[str] = None):
        """Talk to an NPC."""
        action = await self.get_elle_action("talk_to_npc", target_npc_id=npc_id, raw_input=player_message)

        if action and action.get("dialogue"):
            self.npcs_encountered.add(f"talked_to_{npc_id}")
            for d in action["dialogue"]:
                npc_name = next((npc["name"] for npc in self.scene_npcs.get(self.current_scene, []) if npc["id"] == d["npc_id"]), "NPC")
                tone_emoji = {"warm": "😊", "stern": "😐", "cryptic": "🤔", "mysterious": "🔮"}.get(d.get("tone"), "💬")
                print(f"\n{tone_emoji} {npc_name}: \"{d['text']}\"")

    async def get_hint(self):
        """Request a hint."""
        action = await self.get_elle_action("request_hint", raw_input="I'm not sure what to do next")

        if action and action.get("hint_text"):
            print(f"\n💡 Hint: {action['hint_text']}")

    async def play(self):
        """Main game loop."""
        print("=" * 60)
        print("🎮 ELLE'S TEXT ADVENTURE")
        print("=" * 60)
        print("\nA simple adventure powered by Elle's narrative engine.")
        print("Elle is the calm, observant AI that brings the world to life.")
        print("\nCommands:")
        print("  talk <npc_id>  - Talk to an NPC in the scene")
        print("  look           - Look around the current scene")
        print("  hint           - Get a gentle hint")
        print("  move <scene>   - Move to another scene")
        print("  quit           - Exit the game")
        print("\nAvailable scenes: dark_forest, village_square, ancient_ruins")
        print("=" * 60)

        # Enter initial scene
        print(f"\n📍 You find yourself in: {self.current_scene.replace('_', ' ').title()}")
        await self.enter_scene()

        # Show NPCs
        npcs = self.scene_npcs.get(self.current_scene, [])
        if npcs:
            print(f"\n👥 Present: {', '.join(npc['name'] for npc in npcs)}")

        # Main loop
        while True:
            try:
                command = input("\n> ").strip().lower()

                if not command:
                    continue

                if command == "quit":
                    print("\n👋 Thanks for playing!")
                    break

                elif command == "look":
                    await self.enter_scene()
                    npcs = self.scene_npcs.get(self.current_scene, [])
                    if npcs:
                        print(f"\n👥 Present: {', '.join(npc['name'] for npc in npcs)}")

                elif command == "hint":
                    await self.get_hint()

                elif command.startswith("talk "):
                    npc_id = command.split(" ", 1)[1].strip()

                    # Check if NPC exists in scene
                    npcs = self.scene_npcs.get(self.current_scene, [])
                    npc_ids = [npc["id"] for npc in npcs]

                    if npc_id in npc_ids:
                        # Optional: ask for message
                        print("(What do you want to say? Press Enter for default greeting)")
                        message = input("  Say: ").strip()
                        await self.talk_to_npc(npc_id, message if message else None)
                    else:
                        print(f"❌ No NPC with ID '{npc_id}' in this scene.")
                        print(f"   Available: {', '.join(npc_ids)}")

                elif command.startswith("move "):
                    scene = command.split(" ", 1)[1].strip()

                    if scene in self.scene_npcs:
                        self.current_scene = scene
                        print(f"\n🚶 Moving to: {scene.replace('_', ' ').title()}...")
                        await self.enter_scene()
                        npcs = self.scene_npcs.get(self.current_scene, [])
                        if npcs:
                            print(f"\n👥 Present: {', '.join(npc['name'] for npc in npcs)}")
                    else:
                        print(f"❌ Unknown scene '{scene}'")
                        print("   Available: dark_forest, village_square, ancient_ruins")

                else:
                    print("❌ Unknown command. Try: talk, look, hint, move, quit")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()


async def main():
    """Run the game."""
    async with TextAdventureGame() as game:
        await game.play()


if __name__ == "__main__":
    print("\n🎮 Starting Elle's Text Adventure...")
    print("📡 Make sure the Elle Game Engine service is running on http://localhost:8000")
    print("   Start it with: python -m apps.elle_game_engine.service\n")

    asyncio.run(main())
