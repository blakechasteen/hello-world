#!/usr/bin/env python3
"""
HoloLoom Agentic API - Multi-Threaded Conversations
===================================================

Complete multi-threaded conversation system with:
- Multiple simultaneous conversation threads
- Per-thread WebSocket connections
- Cross-thread breakthrough sharing
- Persistent agent pool
- Adversarial negotiation (creative vs QC)

Usage:
    PYTHONPATH=. python HoloLoom/server/agentic_api_multithreaded.py

Then open:
    http://localhost:8002/multithreaded_chat.html

Architecture:
    - ConversationThreadManager handles thread lifecycle
    - AdversarialOrchestrationSystem manages agents
    - WebSocket per thread + global notification stream
    - Breakthrough feed-forward across all threads
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Agentic reasoning
from hololoom.agentic import ReasoningMode

# HoloLoom imports
from hololoom.documentation.types import Query
from hololoom.embedding.spectral import MatryoshkaEmbeddings
from hololoom.memory.graph import KG

# Adversarial orchestration
from hololoom.web_dashboard.adversarial_orchestration import (
    AdversarialOrchestrationSystem,
    create_adversarial_orchestration_system,
)

# Multi-threaded conversation system
from hololoom.web_dashboard.conversation_thread_manager import ConversationThreadManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Global State
# ============================================================================

class ServerState:
    """Global server state"""

    def __init__(self):
        self.thread_manager: ConversationThreadManager | None = None
        self.orchestration_system: AdversarialOrchestrationSystem | None = None
        self.kg: KG | None = None
        self.emb: MatryoshkaEmbeddings | None = None

        # Active WebSocket connections
        self.thread_websockets: dict[str, WebSocket] = {}  # thread_id → WebSocket
        self.notification_websockets: set[WebSocket] = set()  # Global notification subscribers

        # Statistics
        self.total_threads_created = 0
        self.total_messages_sent = 0
        self.total_breakthroughs = 0


state = ServerState()


# ============================================================================
# Pydantic Models
# ============================================================================

class CreateThreadRequest(BaseModel):
    """Request to create new conversation thread"""
    user_id: str = Field(..., description="User ID")
    agent_name: str = Field(..., description="Agent name (budget, research, architecture, etc.)")
    initial_message: str | None = Field(None, description="Optional initial message")


class QueryThreadRequest(BaseModel):
    """Request to query a thread"""
    message: str = Field(..., description="Message text")
    mode: str = Field("verify", description="Reasoning mode (direct, verify, research, plan_execute)")


class ThreadInfo(BaseModel):
    """Thread information"""
    thread_id: str
    user_id: str
    agent_name: str
    message_count: int
    created_at: float
    last_activity: float


class AgentInfo(BaseModel):
    """Agent information"""
    name: str
    active_conversations: int
    total_queries: int
    success_rate: float
    breakthroughs: int


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Multi-Threaded Conversation API",
    description="Multi-threaded conversation system with breakthrough sharing",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Initialization
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize systems on startup"""
    logger.info("Starting HoloLoom Multi-Threaded Conversation API...")

    # Create knowledge graph and embeddings
    state.kg = KG()
    state.emb = MatryoshkaEmbeddings(model_name='all-MiniLM-L6-v2', scales=[96, 192, 384])

    # Create adversarial orchestration system
    logger.info("Creating adversarial orchestration system...")
    state.orchestration_system = await create_adversarial_orchestration_system(
        kg=state.kg,
        emb=state.emb,
        enable_breakthrough_sharing=True,
        default_creativity=0.8,
        default_strictness=0.8
    )

    # Create thread manager
    logger.info("Creating conversation thread manager...")
    state.thread_manager = ConversationThreadManager(
        kg=state.kg,
        emb=state.emb,
        enable_breakthrough_sharing=True
    )

    # Subscribe to breakthroughs
    state.thread_manager.on_breakthrough = _broadcast_breakthrough

    logger.info("✅ Multi-threaded conversation system ready!")
    logger.info(f"   Orchestration: {state.orchestration_system is not None}")
    logger.info(f"   Thread Manager: {state.thread_manager is not None}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")

    # Close all threads
    if state.thread_manager:
        for thread_id in list(state.thread_manager.threads.keys()):
            await state.thread_manager.close_thread(thread_id)

    # Stop orchestration system
    if state.orchestration_system:
        await state.orchestration_system.stop()


# ============================================================================
# Thread Management Endpoints
# ============================================================================

@app.post("/threads/create", response_model=ThreadInfo)
async def create_thread(request: CreateThreadRequest):
    """Create new conversation thread"""
    try:
        # Create thread
        thread = await state.thread_manager.create_thread(
            user_id=request.user_id,
            agent_name=request.agent_name,
            websocket=None  # WebSocket connected later
        )

        state.total_threads_created += 1

        # Process initial message if provided
        if request.initial_message:
            await state.thread_manager.query_thread(
                thread_id=thread.thread_id,
                query=Query(text=request.initial_message)
            )

        return ThreadInfo(
            thread_id=thread.thread_id,
            user_id=thread.user_id,
            agent_name=thread.agent_name,
            message_count=thread.message_count,
            created_at=thread.created_at,
            last_activity=thread.last_activity
        )

    except Exception as e:
        logger.error(f"Failed to create thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/list")
async def list_threads(user_id: str | None = None) -> dict[str, Any]:
    """List all threads (optionally filtered by user)"""
    try:
        threads = []

        for thread_id, thread in state.thread_manager.threads.items():
            # Filter by user if specified
            if user_id and thread.user_id != user_id:
                continue

            threads.append(ThreadInfo(
                thread_id=thread.thread_id,
                user_id=thread.user_id,
                agent_name=thread.agent_name,
                message_count=thread.message_count,
                created_at=thread.created_at,
                last_activity=thread.last_activity
            ))

        return {
            "threads": threads,
            "total_count": len(threads)
        }

    except Exception as e:
        logger.error(f"Failed to list threads: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    """Delete conversation thread"""
    try:
        await state.thread_manager.close_thread(thread_id)

        # Close WebSocket if active
        if thread_id in state.thread_websockets:
            ws = state.thread_websockets[thread_id]
            await ws.close()
            del state.thread_websockets[thread_id]

        return {"status": "deleted", "thread_id": thread_id}

    except Exception as e:
        logger.error(f"Failed to delete thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str) -> ThreadInfo:
    """Get thread information"""
    try:
        thread = state.thread_manager.threads.get(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        return ThreadInfo(
            thread_id=thread.thread_id,
            user_id=thread.user_id,
            agent_name=thread.agent_name,
            message_count=thread.message_count,
            created_at=thread.created_at,
            last_activity=thread.last_activity
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Agent Management Endpoints
# ============================================================================

@app.get("/agents/list")
async def list_agents() -> dict[str, Any]:
    """List available agents with statistics"""
    try:
        agents_info = []

        # Get agents from orchestration system
        for agent_name, agent in state.orchestration_system.agents.items():
            agents_info.append(AgentInfo(
                name=agent.agent_name,
                active_conversations=len(agent.active_conversations),
                total_queries=agent.total_queries_processed,
                success_rate=agent.success_rate,
                breakthroughs=agent.total_breakthroughs
            ))

        return {
            "agents": agents_info,
            "total_count": len(agents_info)
        }

    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Statistics Endpoints
# ============================================================================

@app.get("/stats")
async def get_stats() -> dict[str, Any]:
    """Get global statistics"""
    try:
        # Orchestration stats
        orchestration_stats = state.orchestration_system.get_global_stats() if state.orchestration_system else {}

        # Thread manager stats
        thread_stats = {
            "active_threads": len(state.thread_manager.threads),
            "total_threads_created": state.total_threads_created,
            "total_messages": state.total_messages_sent,
            "total_breakthroughs": state.total_breakthroughs
        }

        # WebSocket stats
        ws_stats = {
            "active_thread_connections": len(state.thread_websockets),
            "active_notification_subscribers": len(state.notification_websockets)
        }

        return {
            "orchestration": orchestration_stats,
            "threads": thread_stats,
            "websockets": ws_stats
        }

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Endpoints
# ============================================================================

@app.websocket("/ws/thread/{thread_id}")
async def thread_websocket(websocket: WebSocket, thread_id: str):
    """WebSocket for specific conversation thread"""
    await websocket.accept()

    # Get thread
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        await websocket.close(code=4004, reason="Thread not found")
        return

    # Store WebSocket
    state.thread_websockets[thread_id] = websocket
    thread.websocket = websocket

    logger.info(f"WebSocket connected to thread {thread_id}")

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            message_text = data.get("message", "")
            mode = data.get("mode", "verify")

            # Send "thinking" status
            await websocket.send_json({
                "type": "status",
                "status": "thinking",
                "message": f"{thread.agent_name} is thinking..."
            })

            # Process query through thread
            mode_enum = {
                "direct": ReasoningMode.DIRECT,
                "verify": ReasoningMode.VERIFY,
                "research": ReasoningMode.RESEARCH,
                "plan_execute": ReasoningMode.PLAN_EXECUTE,
            }.get(mode, ReasoningMode.VERIFY)

            result = await state.thread_manager.query_thread(
                thread_id=thread_id,
                query=Query(text=message_text),
                mode=mode_enum
            )

            state.total_messages_sent += 1

            # Send result
            await websocket.send_json({
                "type": "message",
                "role": "assistant",
                "content": result.spacetime.metadata.get("response", "No response"),
                "confidence": result.spacetime.confidence,
                "mode": result.reasoning_mode.value if hasattr(result, 'reasoning_mode') else mode,
                "thread_id": thread_id,
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from thread {thread_id}")
    except Exception as e:
        logger.error(f"Error in thread WebSocket: {e}")
    finally:
        # Cleanup
        if thread_id in state.thread_websockets:
            del state.thread_websockets[thread_id]
        thread.websocket = None


@app.websocket("/ws/notifications")
async def notifications_websocket(websocket: WebSocket):
    """WebSocket for global notifications (breakthroughs, etc.)"""
    await websocket.accept()
    state.notification_websockets.add(websocket)

    logger.info("WebSocket connected to notification stream")

    try:
        while True:
            # Keep alive (receive ping)
            await websocket.receive_text()

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected from notification stream")
    except Exception as e:
        logger.error(f"Error in notification WebSocket: {e}")
    finally:
        # Cleanup
        state.notification_websockets.discard(websocket)


# ============================================================================
# Helper Functions
# ============================================================================

async def _broadcast_breakthrough(breakthrough: dict[str, Any]):
    """Broadcast breakthrough to all notification subscribers"""
    state.total_breakthroughs += 1

    message = {
        "type": "breakthrough",
        "thread_id": breakthrough.get("thread_id"),
        "agent_name": breakthrough.get("agent_name"),
        "description": breakthrough.get("description", "New breakthrough discovered"),
        "impact_score": breakthrough.get("impact_score", 0.0),
        "timestamp": datetime.now().isoformat()
    }

    # Send to all notification subscribers
    for ws in list(state.notification_websockets):
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send breakthrough notification: {e}")
            state.notification_websockets.discard(ws)


# ============================================================================
# Static Files (UI)
# ============================================================================

# Serve UI from ui/ directory
ui_dir = Path(__file__).parent.parent.parent / "ui"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve multi-threaded chat UI"""
    ui_file = ui_dir / "multithreaded_chat.html"

    if ui_file.exists():
        return FileResponse(ui_file)
    else:
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>HoloLoom Multi-Threaded Chat</h1>
                    <p>UI file not found at: {ui_file}</p>
                    <p>Create ui/multithreaded_chat.html to enable the UI.</p>
                </body>
            </html>
            """,
            status_code=404
        )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 70)
    logger.info("HoloLoom Multi-Threaded Conversation API")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Features:")
    logger.info("  ✓ Multiple simultaneous conversation threads")
    logger.info("  ✓ Per-thread WebSocket connections")
    logger.info("  ✓ Cross-thread breakthrough sharing")
    logger.info("  ✓ Persistent agent pool")
    logger.info("  ✓ Adversarial negotiation (creative vs QC)")
    logger.info("")
    logger.info("Starting server on http://localhost:8002")
    logger.info("Open UI at: http://localhost:8002")
    logger.info("")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
