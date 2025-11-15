"""
Data models for Elle Game Engine.

These models define the contract between game engines and Elle's narrative intelligence.
Designed to be engine-agnostic - works with Unity, Godot, Unreal, or any engine
that can send HTTP/JSON.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


# ============================================================================
# Game State Models
# ============================================================================

@dataclass
class NPCState:
    """State of a non-player character in the game world."""

    id: str
    name: str
    role: str  # e.g. "merchant", "guard", "villager", "quest_giver"
    mood: Optional[str] = None  # "neutral", "annoyed", "grateful", "curious", "hostile"
    location: str = ""  # coarse label: "village_square", "inn", "forest_edge"
    flags: Dict[str, bool] = field(default_factory=dict)  # custom world flags

    def __post_init__(self):
        """Validate NPC state."""
        if not self.id:
            raise ValueError("NPC id cannot be empty")
        if not self.name:
            raise ValueError("NPC name cannot be empty")
        if not self.role:
            raise ValueError("NPC role cannot be empty")


@dataclass
class PlayerState:
    """State of the player character."""

    name: str
    location: str
    quest_stage: Optional[str] = None  # e.g. "intro", "mid", "late", "climax"
    reputation: Optional[str] = None  # e.g. "hero", "outsider", "villain", "neutral"
    traits: Dict[str, int] = field(default_factory=dict)  # e.g. {"brave": 3, "kind": 2, "cunning": 1}
    inventory_tags: List[str] = field(default_factory=list)  # "healing_herb", "ancient_key", "sword"

    def __post_init__(self):
        """Validate player state."""
        if not self.name:
            raise ValueError("Player name cannot be empty")
        if not self.location:
            raise ValueError("Player location cannot be empty")


@dataclass
class WorldState:
    """Ambient world conditions."""

    time_of_day: str  # "morning", "afternoon", "evening", "night", "dawn", "dusk"
    weather: Optional[str] = None  # "clear", "rain", "storm", "fog", "snow"
    tension_level: Optional[str] = None  # "calm", "uneasy", "tense", "high", "critical"

    def __post_init__(self):
        """Validate world state."""
        if not self.time_of_day:
            raise ValueError("time_of_day cannot be empty")


@dataclass
class GameStateSnapshot:
    """
    Complete snapshot of game state for Elle's decision-making.

    This is lean and extensible - game engines provide only what Elle needs
    to make intelligent narrative decisions, not the entire game state.
    """

    scene_id: str  # e.g. "village_square", "dark_forest", "throne_room"
    npcs: List[NPCState] = field(default_factory=list)
    player: PlayerState = field(default_factory=lambda: PlayerState(name="Player", location="unknown"))
    world: WorldState = field(default_factory=lambda: WorldState(time_of_day="day"))
    tags: List[str] = field(default_factory=list)  # custom tags: "post_battle", "festival", "first_visit"

    def __post_init__(self):
        """Validate game state."""
        if not self.scene_id:
            raise ValueError("scene_id cannot be empty")

    @property
    def npc_count(self) -> int:
        """Number of NPCs in the scene."""
        return len(self.npcs)

    def get_npc_by_id(self, npc_id: str) -> Optional[NPCState]:
        """Find NPC by ID."""
        for npc in self.npcs:
            if npc.id == npc_id:
                return npc
        return None

    def has_tag(self, tag: str) -> bool:
        """Check if snapshot has a specific tag."""
        return tag in self.tags


# ============================================================================
# Player Intent Models
# ============================================================================

class PlayerIntentType(str, Enum):
    """High-level player intent categories."""

    TALK_TO_NPC = "talk_to_npc"  # Player initiating dialogue
    ENTER_SCENE = "enter_scene"  # Player entering a new location
    REQUEST_HINT = "request_hint"  # Player asking for guidance
    DEBUG_SUMMARY = "debug_summary"  # Dev requesting narrative summary


@dataclass
class PlayerIntent:
    """
    What the player is doing / what we want from Elle.

    This is the high-level "question" we're asking Elle's narrative engine.
    """

    type: PlayerIntentType
    target_npc_id: Optional[str] = None  # For TALK_TO_NPC
    raw_input: Optional[str] = None  # Player's dialogue text or paraphrased question

    def __post_init__(self):
        """Validate intent."""
        if self.type == PlayerIntentType.TALK_TO_NPC and not self.target_npc_id:
            raise ValueError("target_npc_id required for TALK_TO_NPC intent")


# ============================================================================
# Action Response Models
# ============================================================================

class ActionMode(str, Enum):
    """Type of action Elle is returning."""

    NPC_DIALOGUE = "npc_dialogue"  # NPC speaking
    HINT = "hint"  # Non-spoilery guidance
    WORLD_REACTION = "world_reaction"  # Environmental response
    DEV_DEBUG = "dev_debug"  # Developer-facing narrative notes


@dataclass
class DialogueLine:
    """A line of NPC dialogue."""

    npc_id: str
    text: str
    tone: Optional[str] = None  # "warm", "stern", "cryptic", "excited", "sad"

    def __post_init__(self):
        """Validate dialogue."""
        if not self.npc_id:
            raise ValueError("npc_id cannot be empty")
        if not self.text:
            raise ValueError("dialogue text cannot be empty")


@dataclass
class WorldChange:
    """A change to the world state triggered by Elle's decision."""

    description: str  # Human-readable description
    flag_changes: Dict[str, bool] = field(default_factory=dict)  # Flags to set/unset

    def __post_init__(self):
        """Validate world change."""
        if not self.description:
            raise ValueError("description cannot be empty")


@dataclass
class ElleGameAction:
    """
    Complete structured action Elle returns to the game.

    This is what the game engine receives and processes.
    One clear action per request - no overwhelming the player.
    """

    mode: ActionMode
    priority: str  # "low" | "medium" | "high"

    # Optional response fields (depending on mode)
    dialogue: List[DialogueLine] = field(default_factory=list)
    hint_text: Optional[str] = None
    world_reaction: Optional[WorldChange] = None
    debug_notes: Optional[str] = None  # For devs, not shown to players

    def __post_init__(self):
        """Validate action."""
        if self.priority not in ["low", "medium", "high"]:
            raise ValueError(f"Invalid priority: {self.priority}")

        # Validate that we have content for the specified mode
        if self.mode == ActionMode.NPC_DIALOGUE and not self.dialogue:
            raise ValueError("NPC_DIALOGUE mode requires dialogue")
        if self.mode == ActionMode.HINT and not self.hint_text:
            raise ValueError("HINT mode requires hint_text")
        if self.mode == ActionMode.WORLD_REACTION and not self.world_reaction:
            raise ValueError("WORLD_REACTION mode requires world_reaction")

    @property
    def has_dialogue(self) -> bool:
        """Does this action include dialogue?"""
        return len(self.dialogue) > 0

    @property
    def has_world_changes(self) -> bool:
        """Does this action change the world?"""
        return self.world_reaction is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "mode": self.mode.value,
            "priority": self.priority,
            "dialogue": [
                {"npc_id": d.npc_id, "text": d.text, "tone": d.tone}
                for d in self.dialogue
            ],
            "hint_text": self.hint_text,
            "world_reaction": {
                "description": self.world_reaction.description,
                "flag_changes": self.world_reaction.flag_changes,
            } if self.world_reaction else None,
            "debug_notes": self.debug_notes,
        }
