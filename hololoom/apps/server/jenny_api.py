#!/usr/bin/env python3
from __future__ import annotations
"""
Jenny Generative UI API
========================
FastAPI router for the Jenny adaptive visualization runtime.

Endpoints:
- POST /jenny/ask          — Generate panels from a query
- GET  /jenny/panels       — List active panels
- GET  /jenny/panels/{id}  — Get a specific panel
- POST /jenny/panels/{id}/act — Execute an action on a panel
- GET  /jenny/stats        — Runtime statistics
- GET  /jenny/history      — Provenance history
- WS   /jenny/ws           — WebSocket for streaming panel updates
"""

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

router = APIRouter(prefix="/jenny", tags=["jenny"])
logger = logging.getLogger(__name__)

# Import Jenny components
try:
    from hololoom.visualization import (
        BindingMode,
        JennyConfig,
        JennyPanel,
        JennyRuntime,
        LifecycleStage,
        PanelTypeJenny,
    )
    JENNY_AVAILABLE = True
except ImportError as e:
    JENNY_AVAILABLE = False
    logger.warning(f"Jenny components not available: {e}")

# Singleton runtime — started on first request, stopped on shutdown
_runtime: Optional["JennyRuntime"] = None


async def get_runtime() -> "JennyRuntime":
    """Get or create the singleton JennyRuntime."""
    global _runtime
    if _runtime is None:
        if not JENNY_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Jenny visualization module not available",
            )
        _runtime = JennyRuntime(JennyConfig(
            auto_cleanup=True,
            enable_ledger=True,
            enable_streaming=True,
        ))
        await _runtime.start()
        logger.info("Jenny runtime started")
    return _runtime


async def shutdown_runtime():
    """Shutdown the Jenny runtime (call from app shutdown event)."""
    global _runtime
    if _runtime is not None:
        await _runtime.stop()
        _runtime = None
        logger.info("Jenny runtime stopped")


# ============================================================================
# Request / Response Models
# ============================================================================

class AskRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    context: dict[str, Any] | None = Field(
        default=None, description="Additional context for compilation"
    )
    panel_type: str | None = Field(
        default=None,
        description="Force panel type (text, code, graph, table, confidence, metric)",
    )
    binding_mode: str = Field(
        default="static",
        description="Data binding: static, reactive, or streaming",
    )


class ActRequest(BaseModel):
    action: str = Field(..., description="Action name: pin, dismiss, why, expand, copy")
    context: dict[str, Any] | None = Field(
        default=None, description="Additional action context"
    )


class PanelResponse(BaseModel):
    spec_id: str
    title: str
    panel_type: str
    lifecycle: str
    html: str
    json_data: dict[str, Any]

    @classmethod
    def from_panel(cls, panel: "JennyPanel") -> "PanelResponse":
        return cls(
            spec_id=panel.id,
            title=panel.title,
            panel_type=panel.panel_type.value,
            lifecycle=panel.lifecycle.value,
            html=panel.html,
            json_data=panel.json_data,
        )


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/ask", response_model=list[PanelResponse])
async def jenny_ask(req: AskRequest):
    """
    Generate Jenny panels from a natural language query.

    Returns one or more panels with rendered HTML and JSON data.
    Panel type is auto-detected from the query unless explicitly specified.
    """
    runtime = await get_runtime()

    # Resolve optional panel type
    ptype = None
    if req.panel_type:
        try:
            ptype = PanelTypeJenny(req.panel_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown panel type: {req.panel_type}. "
                f"Valid: {[t.value for t in PanelTypeJenny]}",
            )

    # Resolve binding mode
    try:
        binding = BindingMode(req.binding_mode)
    except ValueError:
        binding = BindingMode.STATIC

    panel = await runtime.ask(
        query=req.query,
        context=req.context,
        panel_type=ptype,
        binding_mode=binding,
    )

    return [PanelResponse.from_panel(panel)]


@router.get("/panels", response_model=list[PanelResponse])
async def list_panels(lifecycle: str | None = None):
    """List active panels, optionally filtered by lifecycle stage."""
    runtime = await get_runtime()

    stage = None
    if lifecycle:
        try:
            stage = LifecycleStage(lifecycle)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown lifecycle stage: {lifecycle}. "
                f"Valid: {[s.value for s in LifecycleStage]}",
            )

    panels = await runtime.list_panels(lifecycle=stage)
    return [PanelResponse.from_panel(p) for p in panels]


@router.get("/panels/{panel_id}", response_model=PanelResponse)
async def get_panel(panel_id: str):
    """Get a specific panel by ID."""
    runtime = await get_runtime()
    panel = await runtime.get_panel(panel_id)
    if not panel:
        raise HTTPException(status_code=404, detail="Panel not found")
    return PanelResponse.from_panel(panel)


@router.post("/panels/{panel_id}/act")
async def act_on_panel(panel_id: str, req: ActRequest):
    """
    Execute an action on a panel (pin, dismiss, why, expand, copy).

    Pin stabilizes a panel. Dismiss dissolves it. Why generates a meta-panel
    explaining the UI choice.
    """
    runtime = await get_runtime()

    result = await runtime.act(
        panel=panel_id,
        action=req.action,
        context=req.context,
    )

    return {
        "action_id": result.action_id,
        "handler": result.handler,
        "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
        "spec_id": result.spec_id,
        "error": result.error,
        "data": result.data,
    }


@router.get("/stats")
async def get_stats():
    """Get Jenny runtime statistics."""
    runtime = await get_runtime()
    stats = runtime.get_stats()
    # Serialize datetime for JSON
    if stats.get("start_time"):
        stats["start_time"] = stats["start_time"].isoformat()
    return stats


@router.get("/history")
async def get_history(limit: int = 100):
    """Get complete provenance history (panel creations, actions, transitions)."""
    runtime = await get_runtime()
    return runtime.history(limit=limit)


# ============================================================================
# WebSocket — streaming panel updates
# ============================================================================

@router.websocket("/ws")
async def jenny_websocket(websocket: WebSocket):
    """
    WebSocket for real-time panel updates.

    Client sends: {"subscribe": "<panel_id>"} or {"unsubscribe": "<sub_id>"}
    Server sends: panel updates as they occur.
    """
    await websocket.accept()
    runtime = await get_runtime()
    subscriptions: dict[str, str] = {}  # sub_id -> panel_id

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if "subscribe" in msg:
                panel_id = msg["subscribe"]

                async def on_update(update):
                    try:
                        await websocket.send_json({
                            "type": "update",
                            "panel_id": panel_id,
                            "update_type": update.update_type.value
                            if hasattr(update.update_type, 'value')
                            else str(update.update_type),
                            "data": update.data if hasattr(update, 'data') else {},
                        })
                    except Exception:
                        pass  # Connection may have closed

                try:
                    sub_id = await runtime.subscribe(panel_id, on_update)
                    subscriptions[sub_id] = panel_id
                    await websocket.send_json({
                        "type": "subscribed",
                        "subscription_id": sub_id,
                        "panel_id": panel_id,
                    })
                except (ValueError, RuntimeError) as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })

            elif "unsubscribe" in msg:
                sub_id = msg["unsubscribe"]
                if sub_id in subscriptions:
                    await runtime.unsubscribe(sub_id)
                    del subscriptions[sub_id]
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "subscription_id": sub_id,
                    })

    except WebSocketDisconnect:
        # Clean up subscriptions
        for sub_id in subscriptions:
            await runtime.unsubscribe(sub_id)
    except Exception as e:
        logger.error(f"Jenny WebSocket error: {e}")
        for sub_id in subscriptions:
            await runtime.unsubscribe(sub_id)
