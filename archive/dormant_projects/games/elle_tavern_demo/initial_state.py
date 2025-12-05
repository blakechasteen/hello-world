"""
Initial Game State for Elle Tavern Demo

Provides functions to create a fresh game state with default values.

Created: 2025-11-16
"""

from typing import Dict, Any
from games.elle_tavern_demo.world_data import LOCATIONS, NPCS, create_npc_emotional_state


def create_initial_game_state() -> Dict[str, Any]:
    """
    Create the initial game state for a new game.

    Returns:
        Complete game state dict that can be used with Elle Game Engine
    """
    return {
        "player": {
            "name": "Traveler",
            "location": "tavern_inside",
            "level": 1,
            "health": 100,
            "max_health": 100,
            "inventory": ["rusty_sword", "healing_potion"],
            "gold": 10,
            "experience": 0,
            "reputation": {}  # NPC-specific reputation scores
        },

        "world": {
            "time_of_day": "evening",
            "weather": "clear",
            "day": 1,
            "tension": 0.0,  # 0.0 = peaceful, 1.0 = critical
            "flags": {
                # Quest flags
                "forest_path_discovered": False,
                "rat_quest_started": False,
                "rat_quest_complete": False,
                "cat_quest_started": False,
                "cat_quest_complete": False,
                "shipment_quest_started": False,
                "shipment_quest_complete": False,
                "bandit_quest_started": False,
                "bandit_quest_complete": False,

                # Story flags
                "talked_to_innkeeper": False,
                "talked_to_guard": False,
                "talked_to_merchant": False,
                "talked_to_stranger": False,
                "talked_to_child": False,
                "stranger_trusts_player": False,

                # World state
                "tavern_reputation": "unknown",
                "guard_reputation": "unknown",
            }
        },

        "npcs": create_initial_npc_states(),
        "locations": {loc_id: dict(loc_data) for loc_id, loc_data in LOCATIONS.items()},

        "quests": {
            "active": [],
            "completed": [],
            "failed": []
        },

        "session": {
            "turns_played": 0,
            "dialogue_history": [],
            "decisions_made": []
        }
    }


def create_initial_npc_states() -> Dict[str, Dict[str, Any]]:
    """
    Create initial NPC states with emotional states initialized.

    Returns:
        Dict mapping NPC IDs to their full state data
    """
    npc_states = {}

    for npc_id, npc_data in NPCS.items():
        # Copy NPC data
        npc_state = dict(npc_data)

        # Create emotional state object
        emotional_state = create_npc_emotional_state(npc_id)

        # Convert to dict for serialization
        npc_state["current_emotion"] = {
            "valence": emotional_state.valence,
            "arousal": emotional_state.arousal,
            "dominance": emotional_state.dominance,
            "trust": emotional_state.trust
        }

        # Initialize interaction tracking
        npc_state["interactions_count"] = 0
        npc_state["last_interaction_turn"] = 0
        npc_state["topics_discussed"] = []
        npc_state["player_reputation"] = 0.0  # -1.0 to 1.0

        npc_states[npc_id] = npc_state

    return npc_states


def reset_game_state(existing_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reset an existing game state to initial values.

    Args:
        existing_state: Current game state

    Returns:
        Fresh game state
    """
    return create_initial_game_state()


def save_game_state_to_dict(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare game state for serialization (e.g., to JSON).

    Args:
        state: Current game state

    Returns:
        Serializable game state dict
    """
    # Already in dict format, but this function allows for future
    # transformations if needed (e.g., converting objects to dicts)
    return state


def load_game_state_from_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load game state from deserialized dict (e.g., from JSON).

    Args:
        state_dict: Deserialized game state

    Returns:
        Game state ready for use
    """
    # Add validation here if needed
    required_keys = ["player", "world", "npcs", "locations", "quests", "session"]
    for key in required_keys:
        if key not in state_dict:
            raise ValueError(f"Invalid game state: missing '{key}' key")

    return state_dict


if __name__ == "__main__":
    # Test state creation
    state = create_initial_game_state()

    print("✅ Initial game state created successfully")
    print(f"\nPlayer: {state['player']['name']}")
    print(f"Location: {state['locations'][state['player']['location']]['name']}")
    print(f"Gold: {state['player']['gold']}")
    print(f"Inventory: {', '.join(state['player']['inventory'])}")

    print(f"\n📊 NPCs loaded: {len(state['npcs'])}")
    for npc_id, npc_state in state['npcs'].items():
        emotion = npc_state['current_emotion']
        print(f"  - {npc_state['name']}: valence={emotion['valence']:.2f}, trust={emotion['trust']:.2f}")

    print(f"\n🗺️  Locations: {len(state['locations'])}")
    for loc_id in state['locations']:
        print(f"  - {state['locations'][loc_id]['name']}")

    print(f"\n🎯 World flags: {len(state['world']['flags'])}")
    print(f"   Time: {state['world']['time_of_day']}")
    print(f"   Weather: {state['world']['weather']}")
    print(f"   Day: {state['world']['day']}")
