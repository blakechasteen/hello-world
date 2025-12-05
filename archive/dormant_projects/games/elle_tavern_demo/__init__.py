"""
Elle Tavern Demo

A fantasy tavern and town demo showcasing Elle Game Engine's capabilities.

Created: 2025-11-16
"""

from games.elle_tavern_demo.world_data import (
    LOCATIONS,
    NPCS,
    get_location,
    get_npc,
    get_npcs_at_location,
    get_available_exits,
    get_location_description,
    create_npc_emotional_state,
    get_npc_sample_dialogue,
    validate_world_data,
)

from games.elle_tavern_demo.initial_state import (
    create_initial_game_state,
    create_initial_npc_states,
    reset_game_state,
    save_game_state_to_dict,
    load_game_state_from_dict,
)

from games.elle_tavern_demo.actions import (
    ACTIONS,
    ActionHandler,
    get_action_help,
)

__version__ = "1.0.0"

__all__ = [
    # World data
    "LOCATIONS",
    "NPCS",
    "get_location",
    "get_npc",
    "get_npcs_at_location",
    "get_available_exits",
    "get_location_description",
    "create_npc_emotional_state",
    "get_npc_sample_dialogue",
    "validate_world_data",
    
    # Game state
    "create_initial_game_state",
    "create_initial_npc_states",
    "reset_game_state",
    "save_game_state_to_dict",
    "load_game_state_from_dict",
    
    # Actions
    "ACTIONS",
    "ActionHandler",
    "get_action_help",
]
