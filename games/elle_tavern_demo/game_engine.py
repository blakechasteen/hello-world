"""
The Rusty Mug Tavern - Game Engine

Main game loop and state management for the Elle Tavern demo.
Orchestrates game state, NPC interactions, quests, and Elle API integration.

Created: 2025-11-16
"""

import asyncio
import uuid
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import aiohttp


# ============================================================================
# Game State Models
# ============================================================================

@dataclass
class NPCData:
    """NPC state in the game."""
    id: str
    name: str
    role: str
    description: str
    location: str
    current_emotion: Dict[str, float] = field(default_factory=lambda: {
        "valence": 0.0,  # -1 (negative) to 1 (positive)
        "arousal": 0.0,  # -1 (calm) to 1 (excited)
        "dominance": 0.0,  # -1 (submissive) to 1 (dominant)
        "trust": 0.5,  # 0 (no trust) to 1 (full trust)
    })
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    flags: Dict[str, bool] = field(default_factory=dict)

    def get_mood_label(self) -> str:
        """Convert PAD emotion to mood label."""
        v, a, d = self.current_emotion["valence"], self.current_emotion["arousal"], self.current_emotion["dominance"]

        if v > 0.5 and a > 0.3:
            return "excited"
        elif v > 0.3:
            return "happy"
        elif v < -0.5 and a > 0.3:
            return "angry"
        elif v < -0.3:
            return "sad"
        elif a > 0.5:
            return "surprised"
        else:
            return "neutral"


@dataclass
class PlayerData:
    """Player state in the game."""
    name: str = "Traveler"
    level: int = 1
    gold: int = 10
    location: str = "rusty_mug_tavern"
    inventory: List[str] = field(default_factory=list)
    active_quests: List[str] = field(default_factory=list)
    completed_quests: List[str] = field(default_factory=list)
    reputation: Dict[str, int] = field(default_factory=dict)  # npc_id -> reputation score


@dataclass
class QuestData:
    """Quest state."""
    id: str
    title: str
    description: str
    giver_npc_id: str
    status: str = "offered"  # offered, active, completed, failed
    objectives: List[Dict[str, Any]] = field(default_factory=list)
    rewards: Dict[str, int] = field(default_factory=lambda: {"gold": 0, "xp": 0})


@dataclass
class GameState:
    """Complete game state."""
    session_id: str
    player: PlayerData
    npcs: Dict[str, NPCData]
    quests: Dict[str, QuestData]
    world_state: Dict[str, Any] = field(default_factory=lambda: {
        "time_of_day": "evening",
        "weather": "clear",
        "tension_level": "calm"
    })
    flags: Dict[str, bool] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Game Engine
# ============================================================================

class TavernGameEngine:
    """
    Main game engine for The Rusty Mug Tavern.

    Responsibilities:
    - Game state management
    - NPC interaction via Elle API
    - Quest system
    - Save/load functionality
    - Voice synthesis integration
    """

    def __init__(self, elle_api_url: str = "http://localhost:8000"):
        self.game_state: Optional[GameState] = None
        self.elle_api_url = elle_api_url
        self.session_id: Optional[str] = None
        self.http_session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.http_session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.http_session:
            await self.http_session.close()

    # ========================================================================
    # Initialization
    # ========================================================================

    async def initialize(self) -> Dict[str, Any]:
        """Initialize new game with fresh state."""
        self.session_id = str(uuid.uuid4())

        # Create NPCs
        npcs = {
            "greta": NPCData(
                id="greta",
                name="Greta",
                role="bartender",
                description="A sturdy woman with a no-nonsense attitude and a heart of gold. She's run The Rusty Mug for 20 years.",
                location="rusty_mug_tavern",
                current_emotion={
                    "valence": 0.3,
                    "arousal": 0.2,
                    "dominance": 0.5,
                    "trust": 0.6
                }
            ),
            "aldric": NPCData(
                id="aldric",
                name="Aldric the Wanderer",
                role="quest_giver",
                description="A mysterious cloaked figure sitting in the corner. His eyes gleam with secrets.",
                location="rusty_mug_tavern",
                current_emotion={
                    "valence": 0.0,
                    "arousal": -0.2,
                    "dominance": 0.3,
                    "trust": 0.3
                }
            ),
            "pip": NPCData(
                id="pip",
                name="Pip",
                role="local",
                description="A cheerful halfling who seems to know everyone's business.",
                location="rusty_mug_tavern",
                current_emotion={
                    "valence": 0.7,
                    "arousal": 0.4,
                    "dominance": -0.2,
                    "trust": 0.7
                }
            ),
            "marcus": NPCData(
                id="marcus",
                name="Marcus the Guard",
                role="guard",
                description="A town guard taking a break. Looks tired but alert.",
                location="rusty_mug_tavern",
                current_emotion={
                    "valence": 0.1,
                    "arousal": 0.3,
                    "dominance": 0.4,
                    "trust": 0.5
                }
            )
        }

        # Create initial game state
        self.game_state = GameState(
            session_id=self.session_id,
            player=PlayerData(),
            npcs=npcs,
            quests={},
            flags={"first_visit": True}
        )

        return {
            "status": "success",
            "message": "Welcome to The Rusty Mug!",
            "game_state": self.get_public_state()
        }

    # ========================================================================
    # State Management
    # ========================================================================

    def get_public_state(self) -> Dict[str, Any]:
        """Get public-facing game state for UI."""
        if not self.game_state:
            return {}

        return {
            "session_id": self.game_state.session_id,
            "player": asdict(self.game_state.player),
            "npcs": {npc_id: asdict(npc) for npc_id, npc in self.game_state.npcs.items()},
            "quests": {q_id: asdict(q) for q_id, q in self.game_state.quests.items()},
            "world_state": self.game_state.world_state,
            "location": {
                "name": "The Rusty Mug Tavern",
                "description": "A cozy tavern filled with the smell of ale and roasting meat. Wooden beams cross the ceiling, and a crackling fire warms the room. Patrons chat quietly at various tables."
            }
        }

    async def save_game(self, save_file: str) -> Dict[str, Any]:
        """Save game to file."""
        if not self.game_state:
            return {"status": "error", "message": "No active game to save"}

        try:
            os.makedirs(os.path.dirname(save_file), exist_ok=True)

            with open(save_file, 'w') as f:
                json.dump(asdict(self.game_state), f, indent=2)

            return {
                "status": "success",
                "message": f"Game saved to {save_file}",
                "save_file": save_file
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to save: {str(e)}"}

    async def load_game(self, save_file: str) -> Dict[str, Any]:
        """Load game from file."""
        try:
            with open(save_file, 'r') as f:
                data = json.load(f)

            # Reconstruct game state
            self.game_state = GameState(
                session_id=data["session_id"],
                player=PlayerData(**data["player"]),
                npcs={npc_id: NPCData(**npc_data) for npc_id, npc_data in data["npcs"].items()},
                quests={q_id: QuestData(**q_data) for q_id, q_data in data["quests"].items()},
                world_state=data["world_state"],
                flags=data["flags"],
                created_at=data["created_at"]
            )
            self.session_id = self.game_state.session_id

            return {
                "status": "success",
                "message": f"Game loaded from {save_file}",
                "game_state": self.get_public_state()
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to load: {str(e)}"}

    # ========================================================================
    # Action Handlers
    # ========================================================================

    async def process_player_action(self, action_type: str, **kwargs) -> Dict[str, Any]:
        """
        Main action dispatcher.

        Actions:
        - talk: Talk to NPC
        - look: Look around current location
        - move: Move to different location
        - quest_accept: Accept a quest
        - quest_complete: Complete a quest
        - give_gift: Give item/gold to NPC
        """
        if not self.game_state:
            return {"status": "error", "message": "No active game"}

        handlers = {
            "talk": self.handle_talk,
            "look": self.handle_look,
            "move": self.handle_move,
            "quest_accept": self.handle_quest_accept,
            "quest_complete": self.handle_quest_complete,
            "give_gift": self.handle_give_gift
        }

        handler = handlers.get(action_type)
        if not handler:
            return {"status": "error", "message": f"Unknown action: {action_type}"}

        try:
            result = await handler(**kwargs)
            # Update game state in result
            result["game_state"] = self.get_public_state()
            return result
        except Exception as e:
            return {"status": "error", "message": f"Action failed: {str(e)}"}

    async def handle_talk(self, npc_id: str, player_message: str) -> Dict[str, Any]:
        """Handle talking to NPC via Elle."""
        npc = self.game_state.npcs.get(npc_id)
        if not npc:
            return {"status": "error", "message": f"NPC {npc_id} not found"}

        # Build Elle request
        elle_request = self.build_elle_request(npc_id, player_message)

        # Call Elle API
        elle_response = await self.call_elle_api(elle_request)

        if elle_response.get("status") == "error":
            # Fallback response if Elle unavailable
            return self.generate_fallback_response(npc, player_message)

        # Extract response data
        npc_response = elle_response.get("response", "...")
        emotion_data = elle_response.get("emotion", {})
        audio_url = elle_response.get("audio_url")
        quest_data = elle_response.get("quest_offered")

        # Update NPC emotion
        if emotion_data:
            npc.current_emotion.update(emotion_data)

        # Add to conversation history
        npc.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "player": player_message,
            "npc": npc_response
        })

        # Check for quest offers
        quest_offered = None
        if quest_data:
            quest_offered = self.process_quest_offer(npc_id, quest_data)

        return {
            "status": "success",
            "npc_id": npc_id,
            "npc_name": npc.name,
            "npc_response": npc_response,
            "npc_emotion": npc.current_emotion,
            "npc_mood": npc.get_mood_label(),
            "audio_url": audio_url,
            "quest_offered": quest_offered
        }

    async def handle_look(self) -> Dict[str, Any]:
        """Look around current location."""
        location_desc = {
            "rusty_mug_tavern": "A cozy tavern filled with the smell of ale and roasting meat. Wooden beams cross the ceiling, and a crackling fire warms the room."
        }

        npcs_here = [
            {"id": npc.id, "name": npc.name, "description": npc.description}
            for npc in self.game_state.npcs.values()
            if npc.location == self.game_state.player.location
        ]

        return {
            "status": "success",
            "location": self.game_state.player.location,
            "description": location_desc.get(self.game_state.player.location, "An unknown place."),
            "npcs": npcs_here
        }

    async def handle_move(self, destination: str) -> Dict[str, Any]:
        """Move to different location."""
        # For this demo, we only have the tavern
        return {
            "status": "error",
            "message": "The tavern is the only location in this demo."
        }

    async def handle_quest_accept(self, quest_id: str) -> Dict[str, Any]:
        """Accept a quest."""
        quest = self.game_state.quests.get(quest_id)
        if not quest:
            return {"status": "error", "message": f"Quest {quest_id} not found"}

        if quest.status != "offered":
            return {"status": "error", "message": "Quest already accepted or completed"}

        quest.status = "active"
        self.game_state.player.active_quests.append(quest_id)

        return {
            "status": "success",
            "message": f"Quest accepted: {quest.title}",
            "quest": asdict(quest)
        }

    async def handle_quest_complete(self, quest_id: str) -> Dict[str, Any]:
        """Complete a quest."""
        quest = self.game_state.quests.get(quest_id)
        if not quest:
            return {"status": "error", "message": f"Quest {quest_id} not found"}

        if quest.status != "active":
            return {"status": "error", "message": "Quest not active"}

        # Check if objectives met (simplified)
        quest.status = "completed"
        self.game_state.player.active_quests.remove(quest_id)
        self.game_state.player.completed_quests.append(quest_id)

        # Give rewards
        self.game_state.player.gold += quest.rewards.get("gold", 0)

        return {
            "status": "success",
            "message": f"Quest completed: {quest.title}",
            "rewards": quest.rewards
        }

    async def handle_give_gift(self, npc_id: str, gift_type: str, amount: int = 1) -> Dict[str, Any]:
        """Give gift to NPC (affects emotion/trust)."""
        npc = self.game_state.npcs.get(npc_id)
        if not npc:
            return {"status": "error", "message": f"NPC {npc_id} not found"}

        if gift_type == "gold":
            if self.game_state.player.gold < amount:
                return {"status": "error", "message": "Not enough gold"}
            self.game_state.player.gold -= amount

            # Improve trust and valence
            npc.current_emotion["trust"] = min(1.0, npc.current_emotion["trust"] + 0.1)
            npc.current_emotion["valence"] = min(1.0, npc.current_emotion["valence"] + 0.2)

            return {
                "status": "success",
                "message": f"{npc.name} gratefully accepts your {amount} gold.",
                "emotion_change": "positive"
            }
        else:
            return {"status": "error", "message": f"Unknown gift type: {gift_type}"}

    # ========================================================================
    # Elle API Integration
    # ========================================================================

    def build_elle_request(self, npc_id: str, player_message: str) -> Dict[str, Any]:
        """Build request payload for Elle API."""
        npc = self.game_state.npcs[npc_id]

        return {
            "game_state": {
                "scene_id": self.game_state.player.location,
                "npcs": [{
                    "id": npc.id,
                    "name": npc.name,
                    "role": npc.role,
                    "mood": npc.get_mood_label(),
                    "location": npc.location,
                    "flags": npc.flags
                }],
                "player": {
                    "name": self.game_state.player.name,
                    "location": self.game_state.player.location,
                    "quest_stage": "intro" if len(self.game_state.player.active_quests) == 0 else "mid",
                    "reputation": "neutral",
                    "traits": {},
                    "inventory_tags": self.game_state.player.inventory
                },
                "world": self.game_state.world_state,
                "tags": list(self.game_state.flags.keys())
            },
            "player_intent": {
                "type": "talk_to_npc",
                "target_npc_id": npc_id,
                "raw_input": player_message
            }
        }

    async def call_elle_api(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call Elle API with game state."""
        if not self.http_session:
            return {"status": "error", "message": "HTTP session not initialized"}

        try:
            async with self.http_session.post(
                f"{self.elle_api_url}/elle/game/action",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self.parse_elle_response(data)
                else:
                    return {"status": "error", "message": f"Elle API returned {response.status}"}
        except aiohttp.ClientError as e:
            return {"status": "error", "message": f"Elle API error: {str(e)}"}
        except asyncio.TimeoutError:
            return {"status": "error", "message": "Elle API timeout"}

    def parse_elle_response(self, elle_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Elle API response."""
        # Extract dialogue
        dialogue = elle_data.get("dialogue", [])
        npc_response = dialogue[0]["text"] if dialogue else "..."

        # Extract emotion if present
        emotion = elle_data.get("emotion", {})

        # Extract audio URL if present
        audio_url = elle_data.get("audio_url")

        # Extract quest data if present
        quest_offered = None
        if elle_data.get("mode") == "quest_offer":
            quest_offered = elle_data.get("quest_data")

        return {
            "status": "success",
            "response": npc_response,
            "emotion": emotion,
            "audio_url": audio_url,
            "quest_offered": quest_offered
        }

    def generate_fallback_response(self, npc: NPCData, player_message: str) -> Dict[str, Any]:
        """Generate fallback response when Elle is unavailable."""
        # Simple fallback responses based on NPC role
        fallback_responses = {
            "bartender": [
                "What'll it be, friend?",
                "The ale's fresh today.",
                "You look like you've had a long day."
            ],
            "quest_giver": [
                "I have much to tell, if you're willing to listen.",
                "Curious, are you?",
                "The world is full of mysteries."
            ],
            "local": [
                "Hey there! New in town?",
                "Did you hear the latest gossip?",
                "It's a beautiful day, isn't it?"
            ],
            "guard": [
                "Keep the peace, citizen.",
                "All's quiet today.",
                "Anything I can help you with?"
            ]
        }

        import random
        responses = fallback_responses.get(npc.role, ["Hello."])

        return {
            "status": "success",
            "npc_id": npc.id,
            "npc_name": npc.name,
            "npc_response": random.choice(responses),
            "npc_emotion": npc.current_emotion,
            "npc_mood": npc.get_mood_label(),
            "audio_url": None,
            "quest_offered": None,
            "fallback": True
        }

    # ========================================================================
    # Quest System
    # ========================================================================

    def process_quest_offer(self, npc_id: str, quest_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a quest offer from Elle."""
        quest_id = quest_data.get("id", f"quest_{len(self.game_state.quests) + 1}")

        quest = QuestData(
            id=quest_id,
            title=quest_data.get("title", "Untitled Quest"),
            description=quest_data.get("description", ""),
            giver_npc_id=npc_id,
            status="offered",
            objectives=quest_data.get("objectives", []),
            rewards=quest_data.get("rewards", {"gold": 10, "xp": 50})
        )

        self.game_state.quests[quest_id] = quest

        return asdict(quest)
