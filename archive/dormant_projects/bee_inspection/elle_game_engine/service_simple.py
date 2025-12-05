"""
FastAPI service for Elle Game Engine with session management and production features.

Provides HTTP/JSON API for game engines to get narrative intelligence
from Elle without tight coupling.

Endpoints:
- POST /elle/game/action - Get action from game state + player intent
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""

from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import os
import time

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import JSONResponse, PlainTextResponse
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
from .session import GameSession, InMemorySessionStore, JSONSessionStore, SessionStore
from .cache import create_cache, ResponseCache
from .metrics import initialize_metrics, get_metrics
from .middleware import create_rate_limiter


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
    session_id: Optional[str] = None  # Optional session ID for state persistence
    player_id: Optional[str] = None  # Optional player ID for new sessions


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
    session_id: str  # Session ID for subsequent requests


# ============================================================================
# Service
# ============================================================================

# Global instances (initialized on startup)
_policy: Optional[GamePolicy] = None
_session_store: Optional[SessionStore] = None
_cache: Optional[ResponseCache] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _policy, _session_store, _cache

    # Initialize metrics
    initialize_metrics()

    # Create response cache
    _cache = create_cache()

    # Startup: Create session store
    session_backend = os.getenv("ELLE_SESSION_BACKEND", "memory").lower()
    if session_backend == "file":
        session_path = os.getenv("ELLE_SESSION_PATH", "./sessions")
        _session_store = JSONSessionStore(storage_path=session_path)
        print(f"💾 Using file-based session storage: {session_path}")
    else:
        _session_store = InMemorySessionStore()
        print("💾 Using in-memory session storage (non-persistent)")

    # Startup: Create LLM client based on environment variables
    provider = os.getenv("ELLE_LLM_PROVIDER", "dummy").lower()

    # Configure client based on provider
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required for Anthropic provider")
        model = os.getenv("ELLE_LLM_MODEL", "claude-3-5-sonnet-20241022")
        llm_client = create_llm_client("anthropic", api_key=api_key, model=model)

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI provider")
        model = os.getenv("ELLE_LLM_MODEL", "gpt-4o-mini")
        llm_client = create_llm_client("openai", api_key=api_key, model=model)

    elif provider == "local":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("ELLE_LLM_MODEL", "llama3.2:3b")
        llm_client = create_llm_client("local", base_url=base_url, model=model)

    else:  # dummy (default)
        llm_client = create_llm_client("dummy")

    # Create policy
    _policy = GamePolicy(llm_client)

    # Log configuration
    print("🎮 Elle Game Engine started")
    print(f"📡 LLM Provider: {provider}")
    if provider != "dummy":
        print(f"📡 Model: {model if 'model' in locals() else 'default'}")

    # Production config
    cache_stats = _cache.get_stats()
    print(f"💾 Cache: {cache_stats['max_size']} entries, TTL {cache_stats['ttl_seconds']}s")

    rate_limit_per_min = int(os.getenv("ELLE_RATE_LIMIT_PER_MINUTE", "60"))
    rate_limit_per_hour = int(os.getenv("ELLE_RATE_LIMIT_PER_HOUR", "100"))
    print(f"🚦 Rate Limits: {rate_limit_per_min}/min per IP, {rate_limit_per_hour}/hour per session")
    print(f"📊 Metrics: /metrics endpoint enabled")

    yield

    # Shutdown: Save sessions and cleanup
    if _session_store is not None:
        _session_store.save()
        print("💾 Sessions saved")

    _policy = None
    _session_store = None
    _cache = None
    print("🛑 Elle Game Engine shutting down")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Elle Game Engine",
        description="LLM-driven narrative intelligence for video games",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add rate limiting middleware
    rate_limiter = create_rate_limiter()
    app.add_middleware(
        type(rate_limiter),
        per_minute_limit=rate_limiter.per_minute_limit,
        per_hour_limit=rate_limiter.per_hour_limit,
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "elle_game_engine"}

    @app.get("/metrics")
    async def metrics():
        """
        Prometheus-style metrics endpoint.

        Returns:
            Plain text metrics in Prometheus format
        """
        metrics_collector = get_metrics()
        return PlainTextResponse(
            content=metrics_collector.to_prometheus_format(),
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/elle/game/action", response_model=GameActionResponse)
    async def get_action(request: GameActionRequest) -> GameActionResponse:
        """
        Get narrative action from Elle based on game state and player intent.

        This is the main endpoint game engines call to get intelligent
        narrative responses.

        Args:
            request: Game state snapshot + player intent + optional session_id

        Returns:
            Structured action (dialogue, hint, world reaction, or debug notes)
            with session_id for state continuity

        Raises:
            HTTPException: If policy execution fails
        """
        if _policy is None or _session_store is None or _cache is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized",
            )

        start_time = time.time()
        cache_hit = False
        provider = os.getenv("ELLE_LLM_PROVIDER", "dummy")

        try:
            # 1. Load or create session
            if request.session_id:
                session = _session_store.get_session(request.session_id)
                if session is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Session {request.session_id} not found",
                    )
            else:
                # Create new session
                session = _session_store.create_session(player_id=request.player_id)

            # 2. Convert Pydantic models to dataclasses
            game_state = _pydantic_to_game_state(request.game_state)
            player_intent = _pydantic_to_player_intent(request.player_intent)

            # 3. Check cache (skip for debug_summary intent)
            skip_cache = player_intent.type == PlayerIntentType.DEBUG_SUMMARY
            cached_action = None if skip_cache else _cache.get(game_state, player_intent)

            if cached_action is not None:
                # Cache hit
                cache_hit = True
                action = cached_action
            else:
                # Cache miss - get conversation context and call policy
                conversation_context = session.get_conversation_context(max_exchanges=5)

                # Get action from policy (with conversation history)
                action = await _policy.decide(
                    game_state,
                    player_intent,
                    conversation_context=conversation_context,
                )

                # Cache response
                _cache.set(game_state, player_intent, action, skip_cache=skip_cache)

            # 4. Update session with this exchange
            player_query = player_intent.raw_input or f"{player_intent.type.value}"
            elle_response = _extract_response_text(action)
            session.add_exchange(
                player_query=player_query,
                elle_response=elle_response,
                npc_id=player_intent.target_npc_id,
            )

            # 5. Update world flags if action changes world
            if action.world_reaction and action.world_reaction.flag_changes:
                session.update_world_flags(action.world_reaction.flag_changes)

            # 6. Update NPC relationships if dialogue occurred
            if action.has_dialogue and player_intent.target_npc_id:
                # Simple reputation heuristic based on tone
                tone = action.dialogue[0].tone if action.dialogue else None
                reputation_delta = 0
                if tone in ["warm", "grateful", "excited"]:
                    reputation_delta = 5
                elif tone in ["stern", "hostile", "annoyed"]:
                    reputation_delta = -5

                session.update_npc_relationship(
                    npc_id=player_intent.target_npc_id,
                    reputation_delta=reputation_delta,
                    mood=tone,
                )

            # 7. Save session
            _session_store.update_session(session)

            # 8. Convert to response model (include session_id)
            response = _action_to_response(action, session_id=session.session_id)

            # Add cache hit metadata
            if not response.debug_notes:
                response.debug_notes = ""
            response.debug_notes += f" [cache_hit={cache_hit}]"

            # 9. Record metrics
            duration_ms = (time.time() - start_time) * 1000
            metrics_collector = get_metrics()
            metrics_collector.record_request(
                intent_type=player_intent.type.value,
                provider=provider,
                duration_ms=duration_ms,
                cache_hit=cache_hit,
            )

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


def _action_to_response(action: ElleGameAction, session_id: str) -> GameActionResponse:
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
        session_id=session_id,
    )


def _extract_response_text(action: ElleGameAction) -> str:
    """Extract response text from action for conversation history."""
    if action.dialogue:
        return action.dialogue[0].text
    elif action.hint_text:
        return action.hint_text
    elif action.world_reaction:
        return action.world_reaction.description
    elif action.debug_notes:
        return action.debug_notes
    else:
        return "(no response)"


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
    print("📊 Metrics: http://localhost:8000/metrics")

    uvicorn.run(
        "apps.elle_game_engine.service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
