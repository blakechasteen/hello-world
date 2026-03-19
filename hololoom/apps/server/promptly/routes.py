"""
FastAPI route handlers for the Promptly Chat API.

Creates the ``router`` that is re-exported by ``__init__.py`` and consumed
by ``agentic_api.py``.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time

import httpx
from fastapi import APIRouter, HTTPException

from .config import (
    JENNY_CONVERSATION_ENABLED,
    JENNY_ENABLED,
    JENNY_SPATIAL_ENABLED,
    MAX_HISTORY_TURNS,
    MAX_PASSES,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    REFINEMENT_ENABLED,
)
from .hololoom_bridge import is_hololoom_available
from .jenny import (
    _gc_conversation_graphs,
    _gc_spatial_dispatchers,
    _run_jenny_conversation_pass,
    _run_jenny_pass,
    _run_spatial_update,
    _update_conversation_graph,
    get_conversation_graphs,
    get_spatial_dispatchers,
)
from .llm import _strip_thinking, call_model
from .memory import conv_memory
from .models import (
    ChatRequest,
    ChatResponse,
    ProposeRequest,
    RefinementInfo,
    RefinementResponse,
    RoutingInfo,
    TTSRequest,
)
from .refinement import (
    _emit_pass1_to_bus,
    _estimate_complexity,
    _get_refinement_bus,
    _run_refinement_pass,
    refinements,
)
from .soul import get_system_prompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/promptly", tags=["promptly-chat"])

# Late import helper — model_router lives in the parent server package
from ..model_router import get_router

# ============================================================================
# /chat
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with Promptly (multi-pass, router-aware).

    Pass 1: Routed to best available model (favors speed), returned immediately.
    Pass 2+: Background refinement via HoloLoomLite + escalated model selection.

    The response includes a refinement_id that the caller can poll via
    GET /promptly/chat/refinement/{id}. Each completed pass includes
    a next_refinement_id if another pass is chained.
    """
    start = time.time()

    # Route to best model for this query (pass 1 favors speed)
    model_router = get_router()
    try:
        decision = await model_router.route(request.text, speed_weight=0.7)
        model_spec = decision.model
        intent = decision.intent.value
    except Exception as e:
        logger.warning("Router failed, using default: %s", e)
        model_spec = None
        intent = None
        decision = None

    # Build messages: system + history + new message (with vault context)
    history = conv_memory.get_messages(request.room_id)
    system_prompt = get_system_prompt()

    # Enrich with vault context if available
    try:
        from .. import vault_bridge
        vault_ctx = vault_bridge.vault_context_block(request.text)
        if vault_ctx:
            system_prompt += vault_ctx
            logger.debug("Vault context injected (%d chars)", len(vault_ctx))
    except Exception as e:
        logger.debug("Vault context unavailable: %s", e)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": request.text})

    try:
        data = await call_model(messages, model_spec=model_spec)
    except (httpx.TimeoutException, HTTPException):
        # Try fallback model if primary failed
        if decision and decision.fallback:
            logger.warning("Primary model failed, trying fallback: %s",
                          decision.fallback.id)
            model_spec = decision.fallback
            data = await call_model(messages, model_spec=model_spec)
        else:
            raise HTTPException(status_code=504, detail="All models failed")

    content = _strip_thinking(data.get("message", {}).get("content", ""))
    tokens = (data.get("eval_count", 0) + data.get("prompt_eval_count", 0))
    duration_ms = (time.time() - start) * 1000

    # Store in conversation memory
    if content:
        conv_memory.add_turn(request.room_id, request.text, content)

    # Write observation to memory bus + emit signal (cross-agent searchable)
    if content:
        from ..kit.emit import emit as kit_emit
        kit_emit("promptly", request.text, content, request.room_id, tokens, duration_ms)

    # Update conversation graph (Stage 2)
    if JENNY_CONVERSATION_ENABLED and content:
        _update_conversation_graph(request.room_id, request.text, content)
        _gc_conversation_graphs()
        _gc_spatial_dispatchers()

    # Build routing transparency
    routing_info = None
    if decision:
        routing_info = RoutingInfo(
            intent=decision.intent.value,
            intent_confidence=round(decision.confidence, 3),
            model_id=decision.model.id,
            reason=decision.reason,
            fallback_model=decision.fallback.id if decision.fallback else None,
            speed_weight=0.7,
        )

    # Decide whether to fire refinement passes
    refinement_id = None
    complexity = _estimate_complexity(request.text) if REFINEMENT_ENABLED and content else None
    refinement_info = RefinementInfo(
        triggered=False,
        complexity_score=round(complexity, 3) if complexity is not None else None,
        threshold=0.25,
        max_passes=MAX_PASSES,
    )
    if REFINEMENT_ENABLED and content and complexity is not None:
        if complexity >= 0.25:
            # Generate deterministic refinement ID
            raw = f"{request.room_id}:{request.text}:{start}"
            refinement_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

            refinements.create(refinement_id, request.text, content, request.room_id)

            # Emit to bus if available, fall back to direct task
            bus = _get_refinement_bus()
            if bus is not None:
                from hololoom.core.bus import Signal, SignalKind
                asyncio.ensure_future(bus.emit(Signal(
                    kind=SignalKind.REFINEMENT_START,
                    source="promptly",
                    payload={
                        "refinement_id": refinement_id,
                        "query": request.text,
                        "prior_response": content,
                        "room_id": request.room_id,
                        "history": history,
                        "complexity": complexity,
                    },
                )))
            else:
                # Fallback: direct task (no bus available)
                asyncio.create_task(
                    _run_refinement_pass(
                        refinement_id, request.text, content,
                        request.room_id, history,
                        pass_number=2, complexity=complexity,
                    )
                )
            refinement_info.triggered = True
            logger.info(
                "Multi-pass refinement %s queued (complexity=%.2f, max_passes=%d, bus=%s)",
                refinement_id, complexity, MAX_PASSES, bus is not None,
            )

    # Fire Jenny visualization pass (parallel to refinement, independent)
    jenny_id = None
    if JENNY_ENABLED and content:
        raw_jenny = f"jenny:{request.room_id}:{request.text}:{start}"
        jenny_id = "j-" + hashlib.sha256(raw_jenny.encode()).hexdigest()[:10]
        refinements.create(jenny_id, request.text, content, request.room_id, pass_number=0)

        # Stage 2: conversation-aware pass when graph has enough data
        conversation_graphs = get_conversation_graphs()
        graph = conversation_graphs.get(request.room_id)
        if (JENNY_CONVERSATION_ENABLED and graph
                and graph.turn_count >= 3 and graph.has_significant_change()):
            asyncio.create_task(
                _run_jenny_conversation_pass(jenny_id, graph, request.room_id)
            )
            logger.info("Jenny conversation pass %s queued (turn=%d)", jenny_id, graph.turn_count)
        else:
            asyncio.create_task(
                _run_jenny_pass(jenny_id, request.text, content, request.room_id, duration_ms)
            )
            logger.info("Jenny visualization %s queued", jenny_id)

    # Fire spatial scene update (Stage 3, parallel to Jenny)
    if JENNY_SPATIAL_ENABLED and content:
        conversation_graphs = get_conversation_graphs()
        graph = conversation_graphs.get(request.room_id)
        if graph and graph.has_significant_change():
            asyncio.create_task(_run_spatial_update(request.room_id, graph))

    logger.info(
        "Promptly chat: room=%s tokens=%d duration=%.0fms turns=%d refine=%s",
        request.room_id, tokens, duration_ms,
        conv_memory.turn_count(request.room_id), refinement_id or "none",
    )

    # Update bandit with outcome
    if decision:
        model_router.feedback(decision, success=bool(content))

    # Emit pass 1 to bus (observability + strategy learning)
    _emit_pass1_to_bus(request.text, content, request.room_id, tokens, duration_ms)

    return ChatResponse(
        response=content,
        room_id=request.room_id,
        model=model_spec.id if model_spec else OLLAMA_MODEL,
        intent=intent,
        tokens=tokens,
        duration_ms=round(duration_ms, 1),
        turns_in_memory=conv_memory.turn_count(request.room_id),
        refinement_id=refinement_id,
        jenny_id=jenny_id,
        routing=routing_info,
        refinement_info=refinement_info,
    )


# ============================================================================
# /chat/refinement/{id}
# ============================================================================

@router.get("/chat/refinement/{refinement_id}", response_model=RefinementResponse)
async def get_refinement(refinement_id: str):
    """
    Poll for a refinement result.

    Returns status:
    - "pending": still processing
    - "complete": refinement ready (check next_refinement_id for chained pass)
    - "skipped": this pass was skipped
    - "converged": model determined no further refinement needed
    """
    entry = refinements.get(refinement_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Refinement not found or expired")

    return RefinementResponse(
        status=entry["status"],
        refinement=entry.get("refinement"),
        reason=entry.get("reason"),
        pass_number=entry.get("pass"),
        max_passes=MAX_PASSES,
        next_refinement_id=entry.get("next_refinement_id"),
        model=entry.get("model"),
        strategy=entry.get("strategy"),
    )


# ============================================================================
# /spatial/{room_id}
# ============================================================================

@router.get("/spatial/{room_id}")
async def get_spatial_scene(room_id: str):
    """
    Get current spatial scene state for a room (debug/polling fallback).

    Returns the positioned overlay scene derived from the conversation graph.
    Requires PROMPTLY_JENNY_SPATIAL=true.
    """
    if not JENNY_SPATIAL_ENABLED:
        raise HTTPException(status_code=404, detail="Spatial visualization not enabled")

    spatial_dispatchers = get_spatial_dispatchers()
    dispatcher = spatial_dispatchers.get(room_id)
    if not dispatcher:
        raise HTTPException(status_code=404, detail="No spatial scene for this room")

    state = dispatcher.get_current_state()
    if not state:
        raise HTTPException(status_code=404, detail="No scene data available")

    state["room_id"] = room_id
    return state


# ============================================================================
# /status
# ============================================================================

@router.get("/status")
async def status():
    """Promptly status and configuration."""
    model_rtr = get_router()
    try:
        from .. import vault_bridge
        vault_available = vault_bridge._is_available()
        vault_root = str(vault_bridge.VAULT_ROOT)
    except Exception:
        vault_available = False
        vault_root = None
    return {
        "name": "Promptly",
        "default_model": OLLAMA_MODEL,
        "ollama_url": OLLAMA_URL,
        "timeout_seconds": OLLAMA_TIMEOUT,
        "max_history_turns": MAX_HISTORY_TURNS,
        "active_rooms": conv_memory.room_count(),
        "refinement_enabled": REFINEMENT_ENABLED,
        "max_passes": MAX_PASSES,
        "hololoom_available": is_hololoom_available(),
        "vault_available": vault_available,
        "vault_root": vault_root,
        "models": model_rtr.list_models(),
        "bandit_stats": model_rtr.bandit_stats(),
    }


# ============================================================================
# Vault API — read-only search + federation proposals
# ============================================================================

@router.get("/vault/search")
async def vault_search(q: str, top_k: int = 5):
    """Search the PARA vault by keyword."""
    from .. import vault_bridge
    results = vault_bridge.search(q, top_k=top_k)
    return {"query": q, "count": len(results), "results": results}


@router.get("/vault/note")
async def vault_read(path: str):
    """Read a vault note by relative path."""
    from .. import vault_bridge
    content = vault_bridge.read_note(path)
    if content is None:
        raise HTTPException(status_code=404, detail=f"Note not found: {path}")
    return {"path": path, "content": content}


@router.get("/vault/list")
async def vault_list(section: str | None = None):
    """List vault notes, optionally filtered by PARA section."""
    from .. import vault_bridge
    notes = vault_bridge.list_notes(section)
    return {"count": len(notes), "notes": notes}


@router.post("/vault/propose")
async def vault_propose(req: ProposeRequest):
    """Propose a note to the vault inbox (or department inbox) for review."""
    from .. import vault_bridge
    result = vault_bridge.propose_note(
        req.title, req.content, req.links,
        department=req.department,
    )
    if "error" in result:
        code = result.get("code", "")
        status_code = 429 if code == "rate_limited" else 400
        raise HTTPException(status_code=status_code, detail=result["error"])
    return result


@router.get("/vault/federation/stats")
async def vault_federation_stats(agent: str = "promptly"):
    """Get federation stats for an agent (promotion rate, frequency tier)."""
    try:
        from ..vault_bridge import get_tracker
        return get_tracker().get_stats(agent)
    except Exception as e:
        return {"error": str(e), "agent": agent}


# ============================================================================
# TTS — Chatterbox voice synthesis (opt-in)
# ============================================================================

@router.post("/tts")
async def tts_synthesize(req: TTSRequest):
    """Synthesize text to speech via Chatterbox on the inference rig.

    Returns WAV audio as binary response. 404 if Chatterbox is unavailable.
    """
    from fastapi.responses import Response

    from .. import tts_client

    audio = await tts_client.synthesize(req.text, voice=req.voice)
    if audio is None:
        raise HTTPException(status_code=503, detail="TTS unavailable")

    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=promptly-tts.wav"},
    )


@router.get("/tts/status")
async def tts_status():
    """Check if Chatterbox TTS is available."""
    from .. import tts_client
    available = await tts_client.is_available()
    return {
        "available": available,
        "url": tts_client.CHATTERBOX_URL,
        "default_voice": tts_client.DEFAULT_VOICE,
    }
