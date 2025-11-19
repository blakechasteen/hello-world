"""
Elle Game Engine Integration

A self-contained microservice that provides LLM-driven narrative intelligence
for video games. Built on Elle's core philosophy but specialized for games.

This module treats the LLM as a "plugin brain" for:
- NPC dialogue generation
- Quest hooks and narrative progression
- World reactions and environmental storytelling
- Developer narrative brainstorming tools
"""

__version__ = "0.1.0"

from .models import (
    NPCState,
    PlayerState,
    WorldState,
    GameStateSnapshot,
    PlayerIntent,
    PlayerIntentType,
    ElleGameAction,
    ActionMode,
    DialogueLine,
    WorldChange,
)

__all__ = [
    "NPCState",
    "PlayerState",
    "WorldState",
    "GameStateSnapshot",
    "PlayerIntent",
    "PlayerIntentType",
    "ElleGameAction",
    "ActionMode",
    "DialogueLine",
    "WorldChange",
]
