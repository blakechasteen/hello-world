"""
FastAPI service for Elle Game Engine.

Provides HTTP/JSON API for game engines to get narrative intelligence
from Elle without tight coupling.

Endpoints:
- POST /elle/game/action - Get action from game state + player intent
- GET /health - Health check
"""

from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "FastAPI not installed. Install with: pip install fastapi uvicorn"
    )

from .models import (
    NPCState,
    PlayerState,
    WorldState,
    GameStateSnapshot,
    PlayerIntent,
    PlayerIntentType,
    ElleGameAction,
)
from .llm_client import create_llm_client
from .policy import GamePolicy, PolicyError


# ============================================================================
# Pydantic Request/Response Models
# ============================================================================

class NPCStateRequest(BaseModel):
    """Pydantic model for NPC state in requests."""
    id: str
    name: str
    role: str
    mood: Optional[str] = None
    location: str = ""
    flags: Dict[str, bool] = Field(default_factory=dict)


class PlayerStateRequest(BaseModel):
    """Pydantic model for player state in requests."""
    name: str
    location: str
    quest_stage: Optional[str] = None
    reputation: Optional[str] = None
    traits: Dict[str, int] = Field(default_factory=dict)
    inventory_tags: List[str] = Field(default_factory=list)


class WorldStateRequest(BaseModel):
    """Pydantic model for world state in requests."""
    time_of_day: str
    weather: Optional[str] = None
    tension_level: Optional[str] = None


class GameStateRequest(BaseModel):
    """Pydantic model for game state in requests."""
    scene_id: str
    npcs: List[NPCStateRequest] = Field(default_factory=list)
    player: PlayerStateRequest
    world: WorldStateRequest
    tags: List[str] = Field(default_factory=list)


class PlayerIntentRequest(BaseModel):
    """Pydantic model for player intent in requests."""
    type: PlayerIntentType
    target_npc_id: Optional[str] = None
    raw_input: Optional[str] = None


class GameActionRequest(BaseModel):
    """Complete request body."""
    game_state: GameStateRequest
    player_intent: PlayerIntentRequest


class DialogueLineResponse(BaseModel):
    """Pydantic model for dialogue line in responses."""
    npc_id: str
    text: str
    tone: Optional[str] = None


class WorldChangeResponse(BaseModel):
    """Pydantic model for world change in responses."""
    description: str
    flag_changes: Dict[str, bool] = Field(default_factory=dict)


class GameActionResponse(BaseModel):
    """Complete response body."""
    mode: str
    priority: str
    dialogue: List[DialogueLineResponse] = Field(default_factory=list)
    hint_text: Optional[str] = None
    world_reaction: Optional[WorldChangeResponse] = None
    debug_notes: Optional[str] = None


# ============================================================================
# Service
# ============================================================================

# Global policy instance (initialized on startup)
_policy: Optional[GamePolicy] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _policy

    # Startup: Create LLM client and policy
    llm_client = create_llm_client("dummy")
    _policy = GamePolicy(llm_client)

    yield

    # Shutdown: cleanup if needed
    _policy = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Elle Game Engine",
        description="LLM-driven narrative intelligence for video games",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "elle_game_engine"}

    @app.post("/elle/game/action", response_model=GameActionResponse)
    async def get_action(request: GameActionRequest) -> GameActionResponse:
        """
        Get narrative action from Elle based on game state and player intent.

        This is the main endpoint game engines call to get intelligent
        narrative responses.

        Args:
            request: Game state snapshot + player intent

        Returns:
            Structured action (dialogue, hint, world reaction, or debug notes)

        Raises:
            HTTPException: If policy execution fails
        """
        if _policy is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Policy not initialized",
            )

        try:
            # Convert Pydantic models to dataclasses
            game_state = _pydantic_to_game_state(request.game_state)
            player_intent = _pydantic_to_player_intent(request.player_intent)

            # Get action from policy
            action = await _policy.decide(game_state, player_intent)

            # Convert to response model
            response = _action_to_response(action)

            return response

        except PolicyError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Policy error: {str(e)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error: {str(e)}",
            )

    return app


# ============================================================================
# Conversion Helpers
# ============================================================================

def _pydantic_to_game_state(req: GameStateRequest) -> GameStateSnapshot:
    """Convert Pydantic request to dataclass."""
    return GameStateSnapshot(
        scene_id=req.scene_id,
        npcs=[
            NPCState(
                id=npc.id,
                name=npc.name,
                role=npc.role,
                mood=npc.mood,
                location=npc.location,
                flags=npc.flags,
            )
            for npc in req.npcs
        ],
        player=PlayerState(
            name=req.player.name,
            location=req.player.location,
            quest_stage=req.player.quest_stage,
            reputation=req.player.reputation,
            traits=req.player.traits,
            inventory_tags=req.player.inventory_tags,
        ),
        world=WorldState(
            time_of_day=req.world.time_of_day,
            weather=req.world.weather,
            tension_level=req.world.tension_level,
        ),
        tags=req.tags,
    )


def _pydantic_to_player_intent(req: PlayerIntentRequest) -> PlayerIntent:
    """Convert Pydantic request to dataclass."""
    return PlayerIntent(
        type=req.type,
        target_npc_id=req.target_npc_id,
        raw_input=req.raw_input,
    )


def _action_to_response(action: ElleGameAction) -> GameActionResponse:
    """Convert dataclass action to Pydantic response."""
    return GameActionResponse(
        mode=action.mode.value,
        priority=action.priority,
        dialogue=[
            DialogueLineResponse(
                npc_id=d.npc_id,
                text=d.text,
                tone=d.tone,
            )
            for d in action.dialogue
        ],
        hint_text=action.hint_text,
        world_reaction=WorldChangeResponse(
            description=action.world_reaction.description,
            flag_changes=action.world_reaction.flag_changes,
        ) if action.world_reaction else None,
        debug_notes=action.debug_notes,
    )


# Create app instance
app = create_app()


# ============================================================================
# CLI Runner (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("🎮 Starting Elle Game Engine service...")
    print("📡 API documentation: http://localhost:8000/docs")
    print("❤️  Health check: http://localhost:8000/health")

    uvicorn.run(
        "apps.elle_game_engine.service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
