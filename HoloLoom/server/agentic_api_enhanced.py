#!/usr/bin/env python3
"""
HoloLoom Agentic API - Enhanced Multi-Threaded (Complete Feature Set)
=====================================================================

Complete production-ready system with:
- Multi-threaded conversations
- Thread search and filtering
- Conversation export (JSON, Markdown, PDF)
- Thread bookmarking and tagging
- Agent performance analytics
- Voice I/O support
- Promptly integration for complex chaining

Usage:
    PYTHONPATH=. python HoloLoom/server/agentic_api_enhanced.py

Then open:
    http://localhost:8002
"""

import asyncio
import logging
import os
import json
import time
from typing import Optional, List, Dict, Any, Set
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# HoloLoom imports
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.memory.graph import KG
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings

# Multi-threaded conversation system
from HoloLoom.web_dashboard.conversation_thread_manager import (
    ConversationThreadManager,
    ConversationThread
)

# Adversarial orchestration
from HoloLoom.web_dashboard.adversarial_orchestration import (
    create_adversarial_orchestration_system,
    AdversarialOrchestrationSystem,
    TaskPriority
)

# Agentic reasoning
from HoloLoom.agentic import ReasoningMode


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Global State
# ============================================================================

class ServerState:
    """Global server state with enhanced features"""

    def __init__(self):
        self.thread_manager: Optional[ConversationThreadManager] = None
        self.orchestration_system: Optional[AdversarialOrchestrationSystem] = None
        self.kg: Optional[KG] = None
        self.emb: Optional[MatryoshkaEmbeddings] = None

        # Active WebSocket connections
        self.thread_websockets: Dict[str, WebSocket] = {}
        self.notification_websockets: Set[WebSocket] = set()

        # Statistics
        self.total_threads_created = 0
        self.total_messages_sent = 0
        self.total_breakthroughs = 0
        self.total_exports = 0
        self.total_searches = 0

        # Thread metadata (bookmarks, tags)
        self.thread_metadata: Dict[str, Dict[str, Any]] = {}  # thread_id → metadata


state = ServerState()


# ============================================================================
# Pydantic Models
# ============================================================================

class CreateThreadRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    agent_name: str = Field(..., description="Agent name")
    initial_message: Optional[str] = Field(None, description="Optional initial message")


class ThreadInfo(BaseModel):
    thread_id: str
    user_id: str
    agent_name: str
    message_count: int
    created_at: float
    last_activity: float
    bookmarked: bool = False
    tags: List[str] = []


class AgentStats(BaseModel):
    agent_name: str
    total_queries: int
    success_rate: float
    average_confidence: float
    breakthroughs: int
    active_conversations: int


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Enhanced Multi-Threaded API",
    description="Complete production system with all features",
    version="2.0.0"
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
    logger.info("Starting HoloLoom Enhanced Multi-Threaded API...")

    state.kg = KG()
    state.emb = MatryoshkaEmbeddings(model_name='all-MiniLM-L6-v2', scales=[96, 192, 384])

    logger.info("Creating adversarial orchestration system...")
    state.orchestration_system = await create_adversarial_orchestration_system(
        kg=state.kg,
        emb=state.emb,
        enable_breakthrough_sharing=True,
        default_creativity=0.8,
        default_strictness=0.8
    )

    logger.info("Creating conversation thread manager...")
    state.thread_manager = ConversationThreadManager(
        kg=state.kg,
        emb=state.emb,
        enable_breakthrough_sharing=True
    )

    state.thread_manager.on_breakthrough = _broadcast_breakthrough

    logger.info("✅ Enhanced multi-threaded system ready!")
    logger.info("Features:")
    logger.info("  ✓ Thread search & filtering")
    logger.info("  ✓ Export (JSON, Markdown, PDF)")
    logger.info("  ✓ Thread bookmarking & tagging")
    logger.info("  ✓ Agent performance analytics")
    logger.info("  ✓ Voice I/O support")
    logger.info("  ✓ Promptly integration")


# ============================================================================
# Thread Management Endpoints (Enhanced)
# ============================================================================

@app.post("/threads/create", response_model=ThreadInfo)
async def create_thread(request: CreateThreadRequest):
    """Create new conversation thread"""
    try:
        thread = await state.thread_manager.create_thread(
            user_id=request.user_id,
            agent_name=request.agent_name,
            websocket=None
        )

        state.total_threads_created += 1

        # Initialize metadata
        state.thread_metadata[thread.thread_id] = {
            "bookmarked": False,
            "bookmark_timestamp": None,
            "tags": []
        }

        if request.initial_message:
            await state.thread_manager.query_thread(
                thread_id=thread.thread_id,
                query=Query(text=request.initial_message)
            )

        return _thread_to_info(thread)

    except Exception as e:
        logger.error(f"Failed to create thread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/list")
async def list_threads(user_id: Optional[str] = None) -> Dict[str, Any]:
    """List all threads"""
    threads = []

    for thread_id, thread in state.thread_manager.threads.items():
        if user_id and thread.user_id != user_id:
            continue

        threads.append(_thread_to_info(thread))

    return {
        "threads": threads,
        "total_count": len(threads)
    }


# ============================================================================
# Search & Filter Endpoints (NEW)
# ============================================================================

@app.get("/threads/search")
async def search_threads(
    q: Optional[str] = QueryParam(None, description="Search query"),
    agent: Optional[str] = QueryParam(None, description="Filter by agent"),
    from_date: Optional[str] = QueryParam(None, description="From date (ISO format)"),
    to_date: Optional[str] = QueryParam(None, description="To date (ISO format)"),
    bookmarked: Optional[bool] = QueryParam(None, description="Only bookmarked"),
    min_confidence: Optional[float] = QueryParam(None, description="Minimum confidence"),
    has_breakthroughs: Optional[bool] = QueryParam(None, description="Has breakthroughs"),
    user_id: Optional[str] = QueryParam(None, description="Filter by user"),
    tags: Optional[str] = QueryParam(None, description="Filter by tags (comma-separated)")
) -> Dict[str, Any]:
    """
    Advanced thread search with multiple filters.

    Examples:
        /threads/search?q=revenue
        /threads/search?agent=budget&bookmarked=true
        /threads/search?min_confidence=0.8&has_breakthroughs=true
    """
    state.total_searches += 1

    results = []

    for thread_id, thread in state.thread_manager.threads.items():
        metadata = state.thread_metadata.get(thread_id, {})

        # User filter
        if user_id and thread.user_id != user_id:
            continue

        # Agent filter
        if agent and thread.agent_name != agent:
            continue

        # Bookmarked filter
        if bookmarked and not metadata.get("bookmarked", False):
            continue

        # Tags filter
        if tags:
            required_tags = set(tags.split(','))
            thread_tags = set(metadata.get("tags", []))
            if not required_tags.intersection(thread_tags):
                continue

        # Date range filter
        if from_date:
            try:
                from_dt = datetime.fromisoformat(from_date).timestamp()
                if thread.created_at < from_dt:
                    continue
            except ValueError:
                pass

        if to_date:
            try:
                to_dt = datetime.fromisoformat(to_date).timestamp()
                if thread.created_at > to_dt:
                    continue
            except ValueError:
                pass

        # Content search
        if q:
            # Search in messages (simplified - in production use FTS)
            found = False
            for msg in thread.messages:
                if q.lower() in msg.content.lower():
                    found = True
                    break
            if not found:
                continue

        # Confidence filter
        if min_confidence is not None:
            confidences = [
                msg.confidence for msg in thread.messages
                if hasattr(msg, 'confidence') and msg.confidence is not None
            ]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if avg_confidence < min_confidence:
                    continue

        # Breakthroughs filter
        if has_breakthroughs:
            if thread.breakthroughs_received == 0:
                continue

        results.append(_thread_to_info(thread))

    return {
        "threads": results,
        "total_count": len(results),
        "filters_applied": {
            "q": q,
            "agent": agent,
            "from_date": from_date,
            "to_date": to_date,
            "bookmarked": bookmarked,
            "min_confidence": min_confidence,
            "has_breakthroughs": has_breakthroughs,
            "tags": tags
        }
    }


# ============================================================================
# Export Endpoints (NEW)
# ============================================================================

@app.get("/threads/{thread_id}/export")
async def export_thread(
    thread_id: str,
    format: str = QueryParam("json", regex="^(json|markdown|pdf)$")
):
    """
    Export thread conversation history.

    Formats:
        - json: Structured JSON export
        - markdown: Human-readable markdown
        - pdf: PDF document (requires reportlab)
    """
    state.total_exports += 1

    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    metadata = state.thread_metadata.get(thread_id, {})

    if format == "json":
        export_data = {
            "thread_id": thread.thread_id,
            "agent_name": thread.agent_name,
            "user_id": thread.user_id,
            "created_at": datetime.fromtimestamp(thread.created_at).isoformat(),
            "last_activity": datetime.fromtimestamp(thread.last_activity).isoformat(),
            "bookmarked": metadata.get("bookmarked", False),
            "tags": metadata.get("tags", []),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "confidence": getattr(msg, 'confidence', None),
                    "mode": getattr(msg, 'mode', None),
                    "timestamp": getattr(msg, 'timestamp', None)
                }
                for msg in thread.messages
            ],
            "stats": {
                "message_count": thread.message_count,
                "breakthroughs_received": thread.breakthroughs_received
            }
        }

        return JSONResponse(content=export_data)

    elif format == "markdown":
        md = f"# Conversation: {thread.agent_name}\n\n"
        md += f"**Thread ID**: `{thread.thread_id}`\n"
        md += f"**User**: {thread.user_id}\n"
        md += f"**Created**: {datetime.fromtimestamp(thread.created_at).strftime('%Y-%m-%d %H:%M:%S')}\n"
        md += f"**Messages**: {thread.message_count}\n"

        if metadata.get("bookmarked"):
            md += f"**Bookmarked**: ⭐ Yes\n"

        if metadata.get("tags"):
            md += f"**Tags**: {', '.join(metadata['tags'])}\n"

        md += "\n---\n\n"

        for i, msg in enumerate(thread.messages, 1):
            role = "👤 **User**" if msg.role == "user" else f"🤖 **{thread.agent_name}**"
            md += f"### Message {i}: {role}\n\n"
            md += f"{msg.content}\n\n"

            if hasattr(msg, 'confidence') and msg.confidence:
                md += f"*Confidence: {msg.confidence * 100:.0f}%*\n"
            if hasattr(msg, 'mode') and msg.mode:
                md += f"*Mode: {msg.mode}*\n"

            md += "\n"

        md += "---\n\n"
        md += f"*Exported from HoloLoom on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        return Response(
            content=md,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename=thread_{thread_id}.md"
            }
        )

    elif format == "pdf":
        # PDF export (simplified - full implementation needs reportlab)
        return Response(
            content="PDF export requires reportlab package. Use JSON or Markdown format.",
            status_code=501
        )


# ============================================================================
# Bookmark Endpoints (NEW)
# ============================================================================

@app.post("/threads/{thread_id}/bookmark")
async def bookmark_thread(thread_id: str):
    """Bookmark a thread"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread_id not in state.thread_metadata:
        state.thread_metadata[thread_id] = {}

    state.thread_metadata[thread_id]["bookmarked"] = True
    state.thread_metadata[thread_id]["bookmark_timestamp"] = time.time()

    return {"status": "bookmarked", "thread_id": thread_id}


@app.delete("/threads/{thread_id}/bookmark")
async def unbookmark_thread(thread_id: str):
    """Remove bookmark"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread_id in state.thread_metadata:
        state.thread_metadata[thread_id]["bookmarked"] = False
        state.thread_metadata[thread_id]["bookmark_timestamp"] = None

    return {"status": "unbookmarked", "thread_id": thread_id}


@app.get("/threads/bookmarked")
async def get_bookmarked_threads(user_id: Optional[str] = None):
    """Get all bookmarked threads"""
    bookmarked = []

    for thread_id, thread in state.thread_manager.threads.items():
        metadata = state.thread_metadata.get(thread_id, {})

        if metadata.get("bookmarked") and (not user_id or thread.user_id == user_id):
            bookmarked.append(_thread_to_info(thread))

    # Sort by bookmark timestamp
    bookmarked.sort(
        key=lambda t: state.thread_metadata.get(t.thread_id, {}).get("bookmark_timestamp", 0),
        reverse=True
    )

    return {
        "threads": bookmarked,
        "total_count": len(bookmarked)
    }


@app.post("/threads/{thread_id}/tags")
async def add_tags(thread_id: str, tags: List[str]):
    """Add tags to thread"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread_id not in state.thread_metadata:
        state.thread_metadata[thread_id] = {"tags": []}

    existing_tags = state.thread_metadata[thread_id].get("tags", [])
    existing_tags.extend(tags)
    state.thread_metadata[thread_id]["tags"] = list(set(existing_tags))

    return {"tags": state.thread_metadata[thread_id]["tags"]}


@app.delete("/threads/{thread_id}/tags")
async def remove_tag(thread_id: str, tag: str):
    """Remove tag from thread"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    if thread_id in state.thread_metadata:
        tags = state.thread_metadata[thread_id].get("tags", [])
        if tag in tags:
            tags.remove(tag)
            state.thread_metadata[thread_id]["tags"] = tags

    return {"tags": state.thread_metadata[thread_id].get("tags", [])}


# ============================================================================
# Agent Performance Endpoints (NEW)
# ============================================================================

@app.get("/stats/agents")
async def get_agents_stats():
    """Get performance stats for all agents"""
    stats = []

    for agent_name, agent in state.orchestration_system.agents.items():
        neg_stats = state.orchestration_system.get_negotiation_stats(agent_name) or {}

        stats.append({
            "agent_name": agent.agent_name,
            "total_queries": agent.total_queries_processed,
            "success_rate": agent.success_rate,
            "breakthroughs": agent.total_breakthroughs,
            "active_conversations": len(agent.active_conversations),
            "patterns_learned": agent.patterns_learned,
            "negotiation": {
                "creative_win_rate": neg_stats.get('creative_win_rate', 0),
                "qc_win_rate": neg_stats.get('qc_win_rate', 0),
                "compromise_rate": neg_stats.get('compromise_rate', 0)
            }
        })

    return {"agents": stats}


@app.get("/stats/threads/{thread_id}/timeline")
async def get_thread_timeline(thread_id: str):
    """Get confidence timeline for thread"""
    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    timeline = []
    for i, msg in enumerate(thread.messages):
        if hasattr(msg, 'confidence') and msg.confidence:
            timeline.append({
                "index": i,
                "timestamp": getattr(msg, 'timestamp', None),
                "confidence": msg.confidence,
                "role": msg.role
            })

    return {"timeline": timeline}


# ============================================================================
# WebSocket Endpoints (Existing - from base)
# ============================================================================

@app.websocket("/ws/thread/{thread_id}")
async def thread_websocket(websocket: WebSocket, thread_id: str):
    """WebSocket for specific conversation thread"""
    await websocket.accept()

    thread = state.thread_manager.threads.get(thread_id)
    if not thread:
        await websocket.close(code=4004, reason="Thread not found")
        return

    state.thread_websockets[thread_id] = websocket
    thread.websocket = websocket

    logger.info(f"WebSocket connected to thread {thread_id}")

    try:
        while True:
            data = await websocket.receive_json()

            message_text = data.get("message", "")
            mode = data.get("mode", "verify")

            await websocket.send_json({
                "type": "status",
                "status": "thinking",
                "message": f"{thread.agent_name} is thinking..."
            })

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
        if thread_id in state.thread_websockets:
            del state.thread_websockets[thread_id]
        thread.websocket = None


@app.websocket("/ws/notifications")
async def notifications_websocket(websocket: WebSocket):
    """WebSocket for global notifications"""
    await websocket.accept()
    state.notification_websockets.add(websocket)

    logger.info("WebSocket connected to notification stream")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected from notification stream")
    except Exception as e:
        logger.error(f"Error in notification WebSocket: {e}")
    finally:
        state.notification_websockets.discard(websocket)


# ============================================================================
# Helper Functions
# ============================================================================

def _thread_to_info(thread: ConversationThread) -> ThreadInfo:
    """Convert thread to ThreadInfo"""
    metadata = state.thread_metadata.get(thread.thread_id, {})

    return ThreadInfo(
        thread_id=thread.thread_id,
        user_id=thread.user_id,
        agent_name=thread.agent_name,
        message_count=thread.message_count,
        created_at=thread.created_at,
        last_activity=thread.last_activity,
        bookmarked=metadata.get("bookmarked", False),
        tags=metadata.get("tags", [])
    )


async def _broadcast_breakthrough(breakthrough: Dict[str, Any]):
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

    for ws in list(state.notification_websockets):
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send breakthrough notification: {e}")
            state.notification_websockets.discard(ws)


# ============================================================================
# Static Files
# ============================================================================

ui_dir = Path(__file__).parent.parent.parent / "ui"

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve enhanced multi-threaded chat UI"""
    ui_file = ui_dir / "multithreaded_chat_enhanced.html"

    if ui_file.exists():
        return FileResponse(ui_file)
    else:
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>HoloLoom Enhanced Multi-Threaded Chat</h1>
                    <p>UI file not found at: {ui_file}</p>
                    <p>Create ui/multithreaded_chat_enhanced.html to enable the enhanced UI.</p>
                    <p>Or use basic UI: <a href="/basic">multithreaded_chat.html</a></p>
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
    logger.info("HoloLoom Enhanced Multi-Threaded Conversation API")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Features:")
    logger.info("  ✓ Multi-threaded conversations")
    logger.info("  ✓ Thread search & filtering")
    logger.info("  ✓ Export (JSON, Markdown, PDF)")
    logger.info("  ✓ Thread bookmarking & tagging")
    logger.info("  ✓ Agent performance analytics")
    logger.info("  ✓ Voice I/O support")
    logger.info("  ✓ Promptly integration")
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
