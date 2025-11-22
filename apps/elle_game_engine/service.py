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
import os
import time

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
from .cache import create_cache, ResponseCache
from .metrics import initialize_metrics, get_metrics
from .middleware import create_rate_limiter
from .pool import create_pool, LLMConnectionPool
from .streaming import create_streaming_client, StreamChunk
from .quest import QuestGenerator, Quest, QuestStatus, QuestGenerationError
from .emotion import EmotionalState, EmotionEngine
from .voice import create_voice_engine, VoiceEngine, VoiceProfile, Emotion, AudioFormat
from .multiplayer import SharedWorldState, MultiplayerWebSocketManager, SessionCoordinator


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
    audio_url: Optional[str] = None  # URL to synthesized audio
    audio_data: Optional[str] = None  # Base64-encoded audio data


class VoiceProfileRequest(BaseModel):
    """Pydantic model for voice profile in requests."""
    voice_id: str
    pitch: float = 1.0
    speed: float = 1.0
    emotion: str = "neutral"
    stability: float = 0.5
    similarity_boost: float = 0.75


class VoiceSynthesisRequest(BaseModel):
    """Request for voice synthesis."""
    text: str
    voice_profile: VoiceProfileRequest
    format: str = "mp3"  # wav, mp3, ogg, opus
    emotion_override: Optional[str] = None


class VoiceSynthesisResponse(BaseModel):
    """Response from voice synthesis."""
    audio_data: str  # Base64-encoded audio bytes
    format: str
    duration_seconds: float
    text: str
    cached: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Service
# ============================================================================

# Global instances (initialized on startup)
_policy: Optional[GamePolicy] = None
_cache: Optional[ResponseCache] = None
_pool: Optional[LLMConnectionPool] = None
_quest_generator: Optional[QuestGenerator] = None
_emotion_engine: Optional[EmotionEngine] = None
_voice_engine: Optional[VoiceEngine] = None

# Quest storage (in-memory for demo - use database in production)
_active_quests: Dict[str, Quest] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global _policy, _cache, _pool, _quest_generator, _emotion_engine, _voice_engine

    # Initialize metrics
    initialize_metrics()

    # Create response cache
    _cache = create_cache()

    # Startup: Create LLM client based on environment variables
    provider = os.getenv("ELLE_LLM_PROVIDER", "dummy").lower()
    pool_enabled = os.getenv("ELLE_ENABLE_POOL", "true").lower() == "true"
    pool_size = int(os.getenv("ELLE_POOL_SIZE", "10"))

    # Configure client based on provider
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable required for Anthropic provider")
        model = os.getenv("ELLE_LLM_MODEL", "claude-3-5-sonnet-20241022")
        client_kwargs = {"api_key": api_key, "model": model}

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI provider")
        model = os.getenv("ELLE_LLM_MODEL", "gpt-4o-mini")
        client_kwargs = {"api_key": api_key, "model": model}

    elif provider == "local":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("ELLE_LLM_MODEL", "llama3.2:3b")
        client_kwargs = {"base_url": base_url, "model": model}

    else:  # dummy (default)
        client_kwargs = {}

    # Create connection pool or single client
    if pool_enabled:
        _pool = await create_pool(provider, pool_size=pool_size, **client_kwargs)
        # Get a client from pool for policy (backward compatibility)
        async with _pool.checkout() as llm_client:
            _policy = GamePolicy(llm_client)
            _quest_generator = QuestGenerator(llm_client)
    else:
        llm_client = create_llm_client(provider, **client_kwargs)
        _policy = GamePolicy(llm_client)
        _quest_generator = QuestGenerator(llm_client)

    # Initialize emotion engine
    _emotion_engine = EmotionEngine()

    # Initialize voice engine
    voice_backend = os.getenv("ELLE_VOICE_BACKEND", "dummy")
    _voice_engine = create_voice_engine(backend=voice_backend, enable_cache=True)

    # Log configuration
    print("🎮 Elle Game Engine started")
    print(f"📡 LLM Provider: {provider}")
    if provider != "dummy":
        print(f"📡 Model: {client_kwargs.get('model', 'default')}")

    # Production config
    cache_stats = _cache.get_stats()
    print(f"💾 Cache: {cache_stats['max_size']} entries, TTL {cache_stats['ttl_seconds']}s")

    if pool_enabled:
        pool_stats = _pool.get_stats()
        print(f"🏊 Connection Pool: {pool_stats.pool_size} clients, {pool_stats.available_connections} available")

    rate_limit_per_min = int(os.getenv("ELLE_RATE_LIMIT_PER_MINUTE", "60"))
    rate_limit_per_hour = int(os.getenv("ELLE_RATE_LIMIT_PER_HOUR", "100"))
    print(f"🚦 Rate Limits: {rate_limit_per_min}/min per IP, {rate_limit_per_hour}/hour per session")
    print(f"📊 Metrics: /metrics endpoint enabled")

    # Voice synthesis
    voice_cache_stats = _voice_engine.get_cache_stats()
    if voice_cache_stats.get("enabled", False):
        print(f"🔊 Voice Synthesis: {voice_backend} backend, cache enabled ({voice_cache_stats.get('total_size_mb', 0):.1f}MB)")
    else:
        print(f"🔊 Voice Synthesis: {voice_backend} backend")

    yield

    # Shutdown: cleanup
    if _pool:
        await _pool.close()
    _policy = None
    _cache = None
    _pool = None
    _voice_engine = None
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
        from fastapi.responses import PlainTextResponse

        metrics_collector = get_metrics()
        return PlainTextResponse(
            content=metrics_collector.to_prometheus_format(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/pool/stats")
    async def pool_stats():
        """
        Get connection pool statistics.

        Returns:
            JSON with pool metrics (or 503 if pooling disabled)
        """
        if _pool is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Connection pooling not enabled",
            )

        stats = _pool.get_stats()
        return stats.to_dict()

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
        if _policy is None or _cache is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized",
            )

        start_time = time.time()
        cache_hit = False
        provider = os.getenv("ELLE_LLM_PROVIDER", "dummy")

        try:
            # Convert Pydantic models to dataclasses
            game_state = _pydantic_to_game_state(request.game_state)
            player_intent = _pydantic_to_player_intent(request.player_intent)

            # Check cache (skip for debug_summary intent)
            skip_cache = player_intent.type == PlayerIntentType.DEBUG_SUMMARY
            cached_action = None if skip_cache else _cache.get(game_state, player_intent)

            if cached_action is not None:
                # Cache hit
                cache_hit = True
                action = cached_action
            else:
                # Cache miss - call policy
                action = await _policy.decide(game_state, player_intent)

                # Cache response
                _cache.set(game_state, player_intent, action, skip_cache=skip_cache)

            # Convert to response model
            response = _action_to_response(action)

            # Add metadata about cache hit
            if not response.debug_notes:
                response.debug_notes = ""
            response.debug_notes += f" [cache_hit={cache_hit}]"

            # Record metrics
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

    @app.get("/elle/game/action/stream")
    async def get_action_stream(
        scene_id: str,
        player_name: str,
        player_location: str,
        intent_type: str,
        target_npc_id: Optional[str] = None,
        raw_input: Optional[str] = None,
    ):
        """
        Get narrative action with Server-Sent Events (SSE) streaming.

        This endpoint streams LLM responses token-by-token for lower perceived latency.

        Query Parameters:
            scene_id: Scene identifier
            player_name: Player name
            player_location: Player location
            intent_type: Intent type (talk_to_npc, enter_scene, etc.)
            target_npc_id: Optional NPC ID (for talk_to_npc)
            raw_input: Optional raw player input

        Returns:
            Server-Sent Events stream with:
            - event: token, data: <token text>
            - event: complete, data: <final JSON>

        Example:
            GET /elle/game/action/stream?scene_id=tavern&player_name=Hero&player_location=tavern&intent_type=talk_to_npc&target_npc_id=innkeeper
        """
        from fastapi.responses import StreamingResponse

        if _pool is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Streaming requires connection pooling to be enabled",
            )

        provider = os.getenv("ELLE_LLM_PROVIDER", "dummy")

        async def event_generator():
            """Generate SSE events."""
            try:
                # Build minimal game state from query params
                game_state = GameStateSnapshot(
                    scene_id=scene_id,
                    npcs=[],  # Simple version - no NPCs in query params
                    player=PlayerState(
                        name=player_name,
                        location=player_location,
                    ),
                    world=WorldState(time_of_day="day"),
                )

                # Build player intent
                try:
                    intent_enum = PlayerIntentType(intent_type)
                except ValueError:
                    yield StreamChunk(
                        event="error",
                        data=f"Invalid intent_type: {intent_type}"
                    ).to_sse_format()
                    return

                player_intent = PlayerIntent(
                    type=intent_enum,
                    target_npc_id=target_npc_id,
                    raw_input=raw_input,
                )

                # Build prompt
                prompt = _policy._build_user_prompt(game_state, player_intent)

                # Stream from pool
                async with _pool.checkout(streaming=True) as streaming_client:
                    accumulated_response = ""

                    async for chunk in streaming_client.stream_complete(
                        prompt=prompt,
                        system_prompt=_policy.core_prompt,
                        max_tokens=500,
                        temperature=0.7,
                    ):
                        # Forward chunk to client
                        yield chunk.to_sse_format()

                        # Accumulate for final parsing
                        if chunk.event == "token":
                            accumulated_response += chunk.data

                    # Parse final response
                    try:
                        action = _policy._parse_response(accumulated_response)

                        # Send final action as JSON
                        response = _action_to_response(action)
                        final_data = response.model_dump_json()

                        yield StreamChunk(
                            event="action",
                            data=final_data,
                            metadata={"provider": provider}
                        ).to_sse_format()

                    except Exception as parse_error:
                        yield StreamChunk(
                            event="error",
                            data=f"Failed to parse response: {parse_error}"
                        ).to_sse_format()

            except Exception as e:
                yield StreamChunk(
                    event="error",
                    data=f"Streaming error: {str(e)}"
                ).to_sse_format()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # ========================================================================
    # Quest Endpoints (2025-11-16)
    # ========================================================================

    @app.post("/elle/game/quest/generate")
    async def generate_quest(
        npc_id: str,
        npc_name: str,
        npc_role: str,
        emotional_state_data: Optional[Dict] = None,
        player_level: int = 1,
        player_reputation: Optional[str] = None,
        world_tension: str = "calm",
        world_time: str = "day",
        recent_events: Optional[str] = None,
    ):
        """
        Generate a contextual quest based on NPC emotion and world state.

        Args:
            npc_id: NPC ID
            npc_name: NPC name
            npc_role: NPC role (merchant, guard, etc.)
            emotional_state_data: Optional emotional state dict
            player_level: Player level
            player_reputation: Player reputation
            world_tension: World tension level
            world_time: Time of day
            recent_events: Recent narrative events

        Returns:
            Generated quest data
        """
        if _quest_generator is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Quest generator not initialized",
            )

        try:
            # Parse emotional state
            if emotional_state_data:
                emotional_state = EmotionalState.from_dict(emotional_state_data)
            else:
                emotional_state = EmotionalState()

            # Generate quest
            quest = await _quest_generator.generate_quest(
                npc_id=npc_id,
                npc_name=npc_name,
                npc_role=npc_role,
                emotional_state=emotional_state,
                player_level=player_level,
                player_reputation=player_reputation,
                world_tension=world_tension,
                world_time=world_time,
                recent_events=recent_events,
            )

            # Store quest
            _active_quests[quest.id] = quest

            return JSONResponse(content=quest.to_dict())

        except QuestGenerationError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Quest generation failed: {str(e)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error: {str(e)}",
            )

    @app.get("/elle/game/quest/{quest_id}")
    async def get_quest(quest_id: str):
        """
        Get quest by ID.

        Args:
            quest_id: Quest ID

        Returns:
            Quest data
        """
        if quest_id not in _active_quests:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Quest {quest_id} not found",
            )

        quest = _active_quests[quest_id]

        # Check if quest expired
        quest.check_time_limit()

        return JSONResponse(content=quest.to_dict())

    @app.post("/elle/game/quest/{quest_id}/complete")
    async def complete_quest(quest_id: str):
        """
        Mark quest as completed.

        Args:
            quest_id: Quest ID

        Returns:
            Updated quest data
        """
        if quest_id not in _active_quests:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Quest {quest_id} not found",
            )

        quest = _active_quests[quest_id]

        try:
            quest.complete()
            return JSONResponse(content=quest.to_dict())
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    @app.post("/elle/game/quest/{quest_id}/fail")
    async def fail_quest(quest_id: str):
        """
        Mark quest as failed.

        Args:
            quest_id: Quest ID

        Returns:
            Updated quest data
        """
        if quest_id not in _active_quests:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Quest {quest_id} not found",
            )

        quest = _active_quests[quest_id]

        try:
            quest.fail()
            return JSONResponse(content=quest.to_dict())
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

    @app.post("/elle/game/voice/synthesize", response_model=VoiceSynthesisResponse)
    async def synthesize_voice(request: VoiceSynthesisRequest) -> VoiceSynthesisResponse:
        """
        Synthesize speech from text using configured TTS backend.

        This endpoint converts text to speech with customizable voice profiles
        and emotion control. Supports multiple backends (ElevenLabs, OpenAI, etc.)
        and automatic caching for repeated phrases.

        Args:
            request: Voice synthesis request with text and voice profile

        Returns:
            Synthesized audio as base64-encoded bytes

        Raises:
            HTTPException: If voice synthesis fails
        """
        if _voice_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Voice synthesis not initialized",
            )

        import base64

        try:
            # Convert Pydantic model to voice module types
            voice_profile = VoiceProfile(
                voice_id=request.voice_profile.voice_id,
                pitch=request.voice_profile.pitch,
                speed=request.voice_profile.speed,
                emotion=Emotion(request.voice_profile.emotion),
                stability=request.voice_profile.stability,
                similarity_boost=request.voice_profile.similarity_boost,
            )

            # Parse emotion override
            emotion_override = None
            if request.emotion_override:
                emotion_override = Emotion(request.emotion_override)

            # Parse format
            audio_format = AudioFormat(request.format)

            # Synthesize speech
            result = _voice_engine.synthesize(
                text=request.text,
                voice_profile=voice_profile,
                emotion=emotion_override,
                format=audio_format
            )

            # Convert audio bytes to base64
            audio_b64 = base64.b64encode(result.audio_data).decode('utf-8')

            return VoiceSynthesisResponse(
                audio_data=audio_b64,
                format=result.format.value,
                duration_seconds=result.duration_seconds,
                text=result.text,
                cached=result.cached,
                metadata=result.metadata
            )

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid request: {str(e)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Voice synthesis error: {str(e)}",
            )

    @app.get("/elle/game/voice/cache/stats")
    async def voice_cache_stats():
        """
        Get voice cache statistics.

        Returns:
            JSON with cache metrics
        """
        if _voice_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Voice synthesis not initialized",
            )

        return _voice_engine.get_cache_stats()

    @app.post("/elle/game/voice/cache/clear")
    async def clear_voice_cache():
        """
        Clear voice synthesis cache.

        Returns:
            Success message
        """
        if _voice_engine is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Voice synthesis not initialized",
            )

        _voice_engine.clear_cache()
        return {"status": "success", "message": "Voice cache cleared"}

    # ========================================================================
    # Multi-NPC Conversation Endpoints
    # ========================================================================

    from .conversation import ConversationCoordinator, ConversationMode

    _conversation_coordinator: Optional[ConversationCoordinator] = None

    def _get_conversation_coordinator() -> ConversationCoordinator:
        """Get or create conversation coordinator."""
        nonlocal _conversation_coordinator
        if _conversation_coordinator is None:
            _conversation_coordinator = ConversationCoordinator(_policy)
        return _conversation_coordinator

    @app.post("/elle/game/conversation/start")
    async def start_conversation(
        npc_ids: List[str],
        topic: Optional[str] = None,
        mode: Optional[str] = None
    ):
        """
        Start a multi-NPC conversation.

        Args:
            npc_ids: List of NPC IDs (2+)
            topic: Optional conversation topic
            mode: Optional mode ("two_npc", "group", "player_mediated")

        Returns:
            Conversation ID and initial state
        """
        coordinator = _get_conversation_coordinator()

        # Parse mode
        conv_mode = None
        if mode:
            try:
                conv_mode = ConversationMode(mode)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid mode: {mode}"
                )

        try:
            conversation_id = coordinator.start_conversation(
                npc_ids=npc_ids,
                topic=topic,
                mode=conv_mode
            )

            conversation = coordinator.get_conversation(conversation_id)

            return {
                "conversation_id": conversation_id,
                "participant_npc_ids": list(conversation.participant_npc_ids),
                "mode": conversation.mode.value,
                "topic": conversation.topic,
                "status": "active"
            }

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

    @app.post("/elle/game/conversation/{conversation_id}/turn")
    async def conversation_turn(
        conversation_id: str,
        request: GameActionRequest,
        player_input: Optional[str] = None
    ):
        """
        Get next turn in conversation.

        Args:
            conversation_id: Conversation identifier
            request: Game state
            player_input: Optional player input

        Returns:
            Next NPC's dialogue and updated conversation state
        """
        coordinator = _get_conversation_coordinator()

        conversation = coordinator.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )

        if not conversation.is_active():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Conversation {conversation_id} is no longer active"
            )

        try:
            game_state = _pydantic_to_game_state(request.game_state)
            action = await coordinator.next_turn(
                conversation_id,
                game_state,
                player_input
            )

            return {
                "action": _game_action_to_response(action),
                "turn_number": len(conversation.turns),
                "is_active": conversation.is_active(),
                "conversation_id": conversation_id
            }

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating turn: {str(e)}"
            )

    @app.get("/elle/game/conversation/{conversation_id}")
    async def get_conversation(conversation_id: str):
        """
        Get conversation state.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Conversation state including all turns
        """
        coordinator = _get_conversation_coordinator()

        conversation = coordinator.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )

        return {
            "conversation_id": conversation.conversation_id,
            "participant_npc_ids": list(conversation.participant_npc_ids),
            "mode": conversation.mode.value,
            "topic": conversation.topic,
            "turn_count": len(conversation.turns),
            "is_active": conversation.is_active(),
            "turns": [
                {
                    "speaker_npc_id": turn.speaker_npc_id,
                    "listener_npc_ids": turn.listener_npc_ids,
                    "text": turn.text,
                    "tone": turn.tone,
                    "timestamp": turn.timestamp.isoformat(),
                    "is_player": turn.is_player
                }
                for turn in conversation.turns
            ],
            "started_at": conversation.started_at.isoformat(),
            "last_activity": conversation.last_activity.isoformat()
        }

    @app.post("/elle/game/conversation/{conversation_id}/end")
    async def end_conversation(conversation_id: str):
        """
        End a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Success message and final conversation state
        """
        coordinator = _get_conversation_coordinator()

        conversation = coordinator.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {conversation_id} not found"
            )

        success = coordinator.end_conversation(conversation_id)

        return {
            "status": "ended",
            "conversation_id": conversation_id,
            "total_turns": len(conversation.turns),
            "success": success
        }

    @app.get("/elle/game/conversation/active")
    async def list_active_conversations():
        """
        List all active conversations.

        Returns:
            List of active conversation IDs and participant counts
        """
        coordinator = _get_conversation_coordinator()

        # Cleanup inactive first
        coordinator.cleanup_inactive_conversations()

        return {
            "active_conversations": [
                {
                    "conversation_id": conv_id,
                    "participant_count": len(conv.participant_npc_ids),
                    "turn_count": len(conv.turns),
                    "mode": conv.mode.value,
                    "is_active": conv.is_active()
                }
                for conv_id, conv in coordinator.active_conversations.items()
            ],
            "total_active": len(coordinator.active_conversations)
        }

    # ========================================================================
    # Fine-Tuning Export Endpoints
    # ========================================================================

    from .fine_tuning import (
        FineTuningExporter,
        FineTuningDataset,
        DatasetQuality,
        ModelVersionManager
    )

    _finetuning_exporter = FineTuningExporter()
    _model_version_manager = ModelVersionManager()

    @app.post("/elle/game/fine-tuning/export")
    async def export_for_finetuning(
        game_state: GameStateRequest,
        player_message: str,
        npc_response: str,
        npc_id: str,
        quality: str = "medium"
    ):
        """
        Export single conversation for fine-tuning.

        Args:
            game_state: Game state context
            player_message: Player's message
            npc_response: NPC's response
            npc_id: NPC identifier
            quality: Quality rating ("high", "medium", "low")

        Returns:
            Formatted training example
        """
        try:
            quality_enum = DatasetQuality(quality)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid quality: {quality}"
            )

        example = _finetuning_exporter.export_conversation(
            game_state=game_state.dict(),
            player_message=player_message,
            npc_response=npc_response,
            npc_id=npc_id,
            quality=quality_enum
        )

        return {
            "messages": example.messages,
            "quality": example.quality.value,
            "metadata": example.metadata,
            "openai_format": example.to_jsonl(),
            "anthropic_format": example.to_anthropic_format()
        }

    @app.post("/elle/game/fine-tuning/dataset/create")
    async def create_dataset(
        name: str,
        examples: List[Dict[str, Any]],
        provider: str = "openai"
    ):
        """
        Create fine-tuning dataset from examples.

        Args:
            name: Dataset name
            examples: List of conversation examples
            provider: "openai" or "anthropic"

        Returns:
            Dataset statistics and export path
        """
        dataset = FineTuningDataset(name=name)

        for ex_data in examples:
            from .fine_tuning import ConversationExample
            example = ConversationExample(
                messages=ex_data["messages"],
                quality=DatasetQuality(ex_data.get("quality", "medium"))
            )
            dataset.add_example(example)

        # Export
        output_dir = _finetuning_exporter.output_dir
        output_file = output_dir / f"{name}_{provider}.jsonl"

        if provider == "openai":
            dataset.export_openai_jsonl(str(output_file))
        else:
            dataset.export_anthropic_jsonl(str(output_file))

        return {
            "dataset_name": name,
            "output_file": str(output_file),
            "statistics": dataset.get_statistics(),
            "provider": provider
        }

    @app.get("/elle/game/fine-tuning/models/active")
    async def get_active_model():
        """Get currently active fine-tuned model."""
        active = _model_version_manager.get_active_model()

        if not active:
            return {"active_model": None}

        return {
            "active_model": active.to_dict()
        }

    @app.get("/elle/game/fine-tuning/models/list")
    async def list_models():
        """List all fine-tuned model versions."""
        return {
            "models": [
                version.to_dict()
                for version in _model_version_manager.versions.values()
            ]
        }

    @app.post("/elle/game/fine-tuning/models/{model_id}/activate")
    async def activate_model(model_id: str):
        """Set model as active for production use."""
        _model_version_manager.set_active(model_id)

        return {
            "status": "success",
            "active_model_id": model_id
        }

    @app.post("/elle/game/fine-tuning/models/{model_id}/metrics")
    async def update_model_metrics(
        model_id: str,
        metrics: Dict[str, float]
    ):
        """
        Update model performance metrics.

        Args:
            model_id: Model identifier
            metrics: Performance metrics (e.g., {"accuracy": 0.95, "latency_ms": 150})
        """
        _model_version_manager.update_metrics(model_id, metrics)

        return {
            "status": "success",
            "model_id": model_id,
            "updated_metrics": metrics
        }

    @app.get("/elle/game/fine-tuning/models/compare")
    async def compare_models(model_a: str, model_b: str):
        """Compare two model versions."""
        try:
            comparison = _model_version_manager.compare_models(model_a, model_b)
            return comparison
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )

    # ========================================================================
    # Multiplayer Endpoints (Phase 4A)
    # ========================================================================

    # Initialize multiplayer components
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    _world_state = None
    _ws_manager = None
    _session_coordinator = None

    async def get_world_state():
        """Get or create SharedWorldState instance"""
        nonlocal _world_state
        if _world_state is None:
            _world_state = SharedWorldState(redis_url=redis_url)
            await _world_state.connect()
        return _world_state

    async def get_ws_manager():
        """Get or create MultiplayerWebSocketManager instance"""
        nonlocal _ws_manager
        if _ws_manager is None:
            _ws_manager = MultiplayerWebSocketManager(redis_url=redis_url)
            await _ws_manager.connect()
            await _ws_manager.start_redis_listener()
            await _ws_manager.start_heartbeat(interval=30)
        return _ws_manager

    async def get_session_coordinator():
        """Get or create SessionCoordinator instance"""
        nonlocal _session_coordinator
        if _session_coordinator is None:
            _session_coordinator = SessionCoordinator(redis_url=redis_url)
            await _session_coordinator.connect()
        return _session_coordinator

    # World State Endpoints

    @app.get("/elle/game/multiplayer/world")
    async def get_world_state_endpoint():
        """
        Get current world state (flags, active players, etc.)

        Returns:
            - flags: All world flags
            - active_players: Set of active player IDs
            - statistics: World state statistics
        """
        world = await get_world_state()

        flags = await world.get_all_flags()
        active_players = await world.get_active_players()
        stats = await world.get_statistics()

        return {
            "flags": {key: flag.to_dict() for key, flag in flags.items()},
            "active_players": list(active_players),
            "statistics": stats
        }

    @app.post("/elle/game/multiplayer/world/flag")
    async def set_world_flag(flag: str, value: bool, player_id: str = "system"):
        """
        Set world flag and broadcast to all players.

        Args:
            flag: Flag name (e.g., "gate_opened", "boss_defeated")
            value: Flag value (True/False)
            player_id: Player who set the flag (default: "system")
        """
        world = await get_world_state()
        flag_obj = await world.set_flag(flag, value, player_id)

        return {
            "status": "success",
            "flag": flag_obj.to_dict()
        }

    # NPC Lock Endpoints

    @app.post("/elle/game/multiplayer/npc/{npc_id}/lock")
    async def acquire_npc_lock_endpoint(npc_id: str, player_id: str, timeout: int = 30):
        """
        Acquire exclusive lock on NPC for conversation.

        Args:
            npc_id: NPC identifier
            player_id: Player requesting lock
            timeout: Lock timeout in seconds (default: 30)

        Returns:
            - acquired: True if lock acquired
            - message: Status message
        """
        coordinator = await get_session_coordinator()
        acquired = await coordinator.acquire_npc_lock(npc_id, player_id, timeout)

        if acquired:
            return {
                "acquired": True,
                "npc_id": npc_id,
                "player_id": player_id,
                "expires_in_seconds": timeout,
                "message": f"Lock acquired on {npc_id}"
            }
        else:
            lock_info = await coordinator.get_npc_lock_info(npc_id)
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "acquired": False,
                    "message": f"NPC {npc_id} is currently in conversation",
                    "locked_by": lock_info.get("player_id") if lock_info else None,
                    "remaining_seconds": lock_info.get("remaining_seconds") if lock_info else None
                }
            )

    @app.post("/elle/game/multiplayer/npc/{npc_id}/unlock")
    async def release_npc_lock_endpoint(npc_id: str, player_id: str):
        """
        Release NPC lock.

        Args:
            npc_id: NPC identifier
            player_id: Player releasing lock
        """
        coordinator = await get_session_coordinator()
        released = await coordinator.release_npc_lock(npc_id, player_id)

        if released:
            return {
                "status": "success",
                "message": f"Lock released on {npc_id}"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No lock found for NPC {npc_id} or lock owned by different player"
            )

    @app.get("/elle/game/multiplayer/npc/{npc_id}/available")
    async def check_npc_availability(npc_id: str):
        """
        Check if NPC is available for interaction.

        Args:
            npc_id: NPC identifier

        Returns:
            - available: True if NPC is not locked
            - lock_info: Lock information if locked
        """
        coordinator = await get_session_coordinator()
        available = await coordinator.is_npc_available(npc_id)

        result = {"available": available}

        if not available:
            lock_info = await coordinator.get_npc_lock_info(npc_id)
            result["lock_info"] = lock_info

        return result

    # WebSocket Endpoint

    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/elle/game/multiplayer/ws/{player_id}")
    async def multiplayer_websocket(websocket: WebSocket, player_id: str):
        """
        WebSocket endpoint for real-time multiplayer updates.

        Clients connect here to receive:
        - World flag changes
        - NPC state updates
        - World events
        - Player join/leave notifications

        Args:
            player_id: Player identifier
        """
        ws_manager = await get_ws_manager()
        world = await get_world_state()

        # Connect player
        await ws_manager.connect_player(websocket, player_id)
        await world.add_player(player_id)

        # Define message handler
        async def handle_message(player_id: str, data: dict):
            """Handle messages from player"""
            msg_type = data.get("type")

            if msg_type == "heartbeat":
                await world.player_heartbeat(player_id)
            elif msg_type == "broadcast":
                # Player wants to broadcast something
                await ws_manager.broadcast({
                    "type": "player_message",
                    "player_id": player_id,
                    "data": data.get("data"),
                    "timestamp": time.time()
                }, exclude=player_id)

        # Listen for player messages
        try:
            await ws_manager.handle_player_message(player_id, websocket, handle_message)
        finally:
            # Cleanup
            await world.remove_player(player_id)

    # Statistics Endpoint

    @app.get("/elle/game/multiplayer/stats")
    async def get_multiplayer_stats():
        """
        Get multiplayer statistics.

        Returns:
            - world: World state statistics
            - websockets: WebSocket connection statistics
            - session_coordinator: Session coordination statistics
        """
        world = await get_world_state()
        ws_manager = await get_ws_manager()
        coordinator = await get_session_coordinator()

        return {
            "world": await world.get_statistics(),
            "websockets": await ws_manager.get_statistics(),
            "session_coordinator": await coordinator.get_statistics()
        }

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
