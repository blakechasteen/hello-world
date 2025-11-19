"""
Game Actions for Elle Tavern Demo

Defines all available player actions and action handlers.

Created: 2025-11-16
"""

from typing import Dict, List, Callable, Any, Optional
from games.elle_tavern_demo.world_data import (
    get_location, get_npc, get_npcs_at_location,
    get_available_exits, get_location_description
)
from apps.elle_game_engine.emotion import EmotionEngine


# ============================================================================
# AVAILABLE ACTIONS
# ============================================================================

ACTIONS = {
    "talk": {
        "name": "Talk",
        "description": "Talk to an NPC",
        "usage": "talk <npc_name>",
        "requires_target": True
    },
    "move": {
        "name": "Move",
        "description": "Move to a different location",
        "usage": "move <location>",
        "requires_target": True
    },
    "look": {
        "name": "Look",
        "description": "Look around current location",
        "usage": "look",
        "requires_target": False
    },
    "quest": {
        "name": "Quests",
        "description": "Check active quests",
        "usage": "quest",
        "requires_target": False
    },
    "inventory": {
        "name": "Inventory",
        "description": "Check inventory",
        "usage": "inventory",
        "requires_target": False
    },
    "gift": {
        "name": "Gift",
        "description": "Give gold or item to NPC",
        "usage": "gift <npc_name> <item/gold>",
        "requires_target": True
    },
    "insult": {
        "name": "Insult",
        "description": "Insult an NPC (for emotion demo)",
        "usage": "insult <npc_name>",
        "requires_target": True
    },
    "help": {
        "name": "Help",
        "description": "Help an NPC with a task",
        "usage": "help <npc_name>",
        "requires_target": True
    },
    "save": {
        "name": "Save",
        "description": "Save game",
        "usage": "save [filename]",
        "requires_target": False
    },
    "load": {
        "name": "Load",
        "description": "Load game",
        "usage": "load [filename]",
        "requires_target": False
    },
    "status": {
        "name": "Status",
        "description": "Show character status",
        "usage": "status",
        "requires_target": False
    },
    "quit": {
        "name": "Quit",
        "description": "Quit game",
        "usage": "quit",
        "requires_target": False
    }
}


# ============================================================================
# ACTION HANDLERS
# ============================================================================

class ActionHandler:
    """Handles player actions and updates game state."""

    def __init__(self):
        self.emotion_engine = EmotionEngine()

    def handle_action(
        self,
        action: str,
        target: Optional[str],
        game_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle a player action and return the result.

        Args:
            action: Action name (e.g., "talk", "move")
            target: Target of action (e.g., NPC name, location)
            game_state: Current game state

        Returns:
            Dict with keys:
                - success: bool
                - message: str
                - game_state: updated game state
                - use_elle: bool (whether to call Elle for response)
        """
        action_method = getattr(self, f"action_{action}", None)

        if not action_method:
            return {
                "success": False,
                "message": f"Unknown action: {action}",
                "game_state": game_state,
                "use_elle": False
            }

        return action_method(target, game_state)

    def action_talk(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Talk to an NPC - uses Elle for response."""
        if not target:
            return {
                "success": False,
                "message": "Who do you want to talk to?",
                "game_state": game_state,
                "use_elle": False
            }

        # Find NPC at current location
        current_location = game_state["player"]["location"]
        npcs_here = get_npcs_at_location(current_location)

        # Match NPC by name or ID
        npc_match = None
        for npc in npcs_here:
            if target.lower() in npc["name"].lower() or target == npc["id"]:
                npc_match = npc
                break

        if not npc_match:
            available_npcs = ", ".join([npc["name"] for npc in npcs_here])
            return {
                "success": False,
                "message": f"Nobody by that name is here. Available: {available_npcs or 'None'}",
                "game_state": game_state,
                "use_elle": False
            }

        # Mark interaction
        game_state["world"]["flags"][f"talked_to_{npc_match['id']}"] = True

        # Use Elle for dialogue
        return {
            "success": True,
            "message": f"Talking to {npc_match['name']}...",
            "game_state": game_state,
            "use_elle": True,
            "npc_id": npc_match["id"]
        }

    def action_move(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Move to a different location."""
        if not target:
            return {
                "success": False,
                "message": "Where do you want to go?",
                "game_state": game_state,
                "use_elle": False
            }

        current_location = game_state["player"]["location"]
        available_exits = get_available_exits(current_location, game_state["world"]["flags"])

        # Match location by name or ID
        location_match = None
        for exit_id in available_exits:
            location = get_location(exit_id)
            if location and (target.lower() in location["name"].lower() or target == exit_id):
                location_match = exit_id
                break

        if not location_match:
            exit_names = ", ".join([get_location(e)["name"] for e in available_exits])
            return {
                "success": False,
                "message": f"You can't go there. Available exits: {exit_names}",
                "game_state": game_state,
                "use_elle": False
            }

        # Move player
        game_state["player"]["location"] = location_match
        location_desc = get_location_description(location_match, game_state["world"]["time_of_day"])

        return {
            "success": True,
            "message": f"You travel to {get_location(location_match)['name']}.\n\n{location_desc}",
            "game_state": game_state,
            "use_elle": False
        }

    def action_look(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Look around current location."""
        current_location = game_state["player"]["location"]
        location = get_location(current_location)

        description = get_location_description(current_location, game_state["world"]["time_of_day"])

        # List NPCs
        npcs_here = get_npcs_at_location(current_location)
        if npcs_here:
            npc_names = ", ".join([npc["name"] for npc in npcs_here])
            description += f"\n\nYou see: {npc_names}"

        # List exits
        available_exits = get_available_exits(current_location, game_state["world"]["flags"])
        if available_exits:
            exit_names = ", ".join([get_location(e)["name"] for e in available_exits])
            description += f"\n\nExits: {exit_names}"

        return {
            "success": True,
            "message": description,
            "game_state": game_state,
            "use_elle": False
        }

    def action_quest(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Show active quests."""
        quests = game_state.get("quests", {"active": [], "completed": [], "failed": []})

        message = "=== QUESTS ===\n\n"

        if quests["active"]:
            message += "Active Quests:\n"
            for quest in quests["active"]:
                message += f"  - {quest.get('title', 'Untitled Quest')}\n"
        else:
            message += "No active quests.\n"

        if quests["completed"]:
            message += f"\nCompleted: {len(quests['completed'])}"

        return {
            "success": True,
            "message": message,
            "game_state": game_state,
            "use_elle": False
        }

    def action_inventory(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Show player inventory."""
        inventory = game_state["player"]["inventory"]
        gold = game_state["player"]["gold"]

        message = "=== INVENTORY ===\n\n"
        message += f"Gold: {gold}\n\n"

        if inventory:
            message += "Items:\n"
            for item in inventory:
                message += f"  - {item}\n"
        else:
            message += "No items."

        return {
            "success": True,
            "message": message,
            "game_state": game_state,
            "use_elle": False
        }

    def action_gift(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Give gold or item to NPC - improves emotion."""
        if not target:
            return {
                "success": False,
                "message": "Who do you want to give a gift to?",
                "game_state": game_state,
                "use_elle": False
            }

        # For demo purposes, giving a gift improves NPC emotion
        # In full version, would parse item/gold amount

        current_location = game_state["player"]["location"]
        npcs_here = get_npcs_at_location(current_location)

        npc_match = None
        for npc in npcs_here:
            if target.lower() in npc["name"].lower() or target == npc["id"]:
                npc_match = npc
                break

        if not npc_match:
            return {
                "success": False,
                "message": "Nobody by that name is here.",
                "game_state": game_state,
                "use_elle": False
            }

        # Update NPC emotion (gift action)
        from apps.elle_game_engine.emotion import EmotionalState
        current_emotion = game_state["npcs"][npc_match["id"]]["current_emotion"]
        emotional_state = EmotionalState(**current_emotion)

        updated_emotion = self.emotion_engine.process_player_action(
            emotional_state,
            action_type="gift",
            npc_id=npc_match["id"]
        )

        game_state["npcs"][npc_match["id"]]["current_emotion"] = {
            "valence": updated_emotion.valence,
            "arousal": updated_emotion.arousal,
            "dominance": updated_emotion.dominance,
            "trust": updated_emotion.trust
        }

        return {
            "success": True,
            "message": f"You give a gift to {npc_match['name']}. They seem pleased!",
            "game_state": game_state,
            "use_elle": False
        }

    def action_insult(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Insult an NPC - worsens emotion (demo feature)."""
        if not target:
            return {
                "success": False,
                "message": "Who do you want to insult?",
                "game_state": game_state,
                "use_elle": False
            }

        current_location = game_state["player"]["location"]
        npcs_here = get_npcs_at_location(current_location)

        npc_match = None
        for npc in npcs_here:
            if target.lower() in npc["name"].lower() or target == npc["id"]:
                npc_match = npc
                break

        if not npc_match:
            return {
                "success": False,
                "message": "Nobody by that name is here.",
                "game_state": game_state,
                "use_elle": False
            }

        # Update NPC emotion (insult action)
        from apps.elle_game_engine.emotion import EmotionalState
        current_emotion = game_state["npcs"][npc_match["id"]]["current_emotion"]
        emotional_state = EmotionalState(**current_emotion)

        updated_emotion = self.emotion_engine.process_player_action(
            emotional_state,
            action_type="insult",
            npc_id=npc_match["id"]
        )

        game_state["npcs"][npc_match["id"]]["current_emotion"] = {
            "valence": updated_emotion.valence,
            "arousal": updated_emotion.arousal,
            "dominance": updated_emotion.dominance,
            "trust": updated_emotion.trust
        }

        return {
            "success": True,
            "message": f"You insult {npc_match['name']}. They look offended!",
            "game_state": game_state,
            "use_elle": False
        }

    def action_help(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Help an NPC - improves emotion."""
        # Similar to gift, but represents helping with a task
        if not target:
            return {
                "success": False,
                "message": "Who do you want to help?",
                "game_state": game_state,
                "use_elle": False
            }

        current_location = game_state["player"]["location"]
        npcs_here = get_npcs_at_location(current_location)

        npc_match = None
        for npc in npcs_here:
            if target.lower() in npc["name"].lower() or target == npc["id"]:
                npc_match = npc
                break

        if not npc_match:
            return {
                "success": False,
                "message": "Nobody by that name is here.",
                "game_state": game_state,
                "use_elle": False
            }

        # Update NPC emotion (help action)
        from apps.elle_game_engine.emotion import EmotionalState
        current_emotion = game_state["npcs"][npc_match["id"]]["current_emotion"]
        emotional_state = EmotionalState(**current_emotion)

        updated_emotion = self.emotion_engine.process_player_action(
            emotional_state,
            action_type="help",
            npc_id=npc_match["id"]
        )

        game_state["npcs"][npc_match["id"]]["current_emotion"] = {
            "valence": updated_emotion.valence,
            "arousal": updated_emotion.arousal,
            "dominance": updated_emotion.dominance,
            "trust": updated_emotion.trust
        }

        return {
            "success": True,
            "message": f"You help {npc_match['name']}. They look grateful!",
            "game_state": game_state,
            "use_elle": False
        }

    def action_status(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Show player status."""
        player = game_state["player"]
        location = get_location(player["location"])

        message = "=== CHARACTER STATUS ===\n\n"
        message += f"Name: {player['name']}\n"
        message += f"Level: {player['level']}\n"
        message += f"Health: {player['health']}/{player['max_health']}\n"
        message += f"Gold: {player['gold']}\n"
        message += f"Experience: {player.get('experience', 0)}\n"
        message += f"\nLocation: {location['name']}\n"
        message += f"Time: {game_state['world']['time_of_day']}\n"
        message += f"Day: {game_state['world']['day']}\n"

        return {
            "success": True,
            "message": message,
            "game_state": game_state,
            "use_elle": False
        }

    def action_save(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Save game state to file."""
        import json
        from pathlib import Path

        filename = target or "savegame.json"
        save_dir = Path("games/elle_tavern_demo/saves")
        save_dir.mkdir(exist_ok=True)

        save_path = save_dir / filename

        try:
            with open(save_path, 'w') as f:
                json.dump(game_state, f, indent=2)

            return {
                "success": True,
                "message": f"Game saved to {save_path}",
                "game_state": game_state,
                "use_elle": False
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to save game: {e}",
                "game_state": game_state,
                "use_elle": False
            }

    def action_load(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Load game state from file."""
        import json
        from pathlib import Path

        filename = target or "savegame.json"
        save_path = Path("games/elle_tavern_demo/saves") / filename

        if not save_path.exists():
            return {
                "success": False,
                "message": f"Save file not found: {save_path}",
                "game_state": game_state,
                "use_elle": False
            }

        try:
            with open(save_path, 'r') as f:
                loaded_state = json.load(f)

            return {
                "success": True,
                "message": f"Game loaded from {save_path}",
                "game_state": loaded_state,
                "use_elle": False
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to load game: {e}",
                "game_state": game_state,
                "use_elle": False
            }

    def action_quit(self, target: Optional[str], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Quit the game."""
        return {
            "success": True,
            "message": "Thanks for playing!",
            "game_state": game_state,
            "use_elle": False,
            "quit": True
        }


def get_action_help() -> str:
    """Get formatted help text for all actions."""
    help_text = "=== AVAILABLE ACTIONS ===\n\n"

    for action_id, action_data in ACTIONS.items():
        help_text += f"{action_data['name']}: {action_data['description']}\n"
        help_text += f"  Usage: {action_data['usage']}\n\n"

    return help_text


if __name__ == "__main__":
    # Test action handler
    from games.elle_tavern_demo.initial_state import create_initial_game_state

    handler = ActionHandler()
    state = create_initial_game_state()

    print("✅ Action handler created")
    print(f"\n{len(ACTIONS)} actions available:")
    for action_id, action_data in ACTIONS.items():
        print(f"  - {action_id}: {action_data['description']}")

    print("\n📋 Testing 'look' action:")
    result = handler.handle_action("look", None, state)
    print(f"Success: {result['success']}")
    print(f"Message:\n{result['message']}")
