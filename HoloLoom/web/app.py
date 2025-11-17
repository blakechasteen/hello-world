"""
FastAPI Chat Application
=========================
Web server for interactive chat with streaming responses.

Philosophy:
The chat interface is the human interface to the weaving process. It should
show not just the response, but the computational journey - features being
extracted, threads being activated, decisions being made. This transparency
builds trust and enables learning.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not installed. Web interface unavailable.")
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None


@dataclass
class ChatConfig:
    """Configuration for chat server."""
    host: str = "0.0.0.0"
    port: int = 8000
    static_dir: str = "static"
    enable_cors: bool = True
    max_message_length: int = 4096
    session_timeout: int = 3600  # seconds


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Connect new WebSocket."""
        await websocket.accept()
        self.active_connections.append(websocket)

        # Initialize session
        self.sessions[session_id] = {
            "websocket": websocket,
            "history": [],
            "created_at": datetime.now(timezone.utc),
            "last_activity": datetime.now(timezone.utc)
        }

        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, websocket: WebSocket, session_id: str):
        """Disconnect WebSocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if session_id in self.sessions:
            del self.sessions[session_id]

        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific session."""
        if session_id in self.sessions:
            websocket = self.sessions[session_id]["websocket"]
            await websocket.send_json(message)

    async def stream_response(
        self,
        session_id: str,
        text: str,
        chunk_size: int = 10,
        delay: float = 0.05
    ):
        """Stream response text in chunks."""
        if session_id not in self.sessions:
            return

        # Send in chunks for streaming effect
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]

            await self.send_message(session_id, {
                "type": "response_chunk",
                "text": chunk,
                "done": False
            })

            await asyncio.sleep(delay)

        # Send completion
        await self.send_message(session_id, {
            "type": "response_chunk",
            "text": "",
            "done": True
        })


def create_app(config: ChatConfig = None) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        config: Chat configuration

    Returns:
        Configured FastAPI app
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed: pip install fastapi uvicorn websockets")

    config = config or ChatConfig()
    app = FastAPI(title="HoloLoom Chat", version="2.0.0")

    # CORS
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Connection manager
    manager = ConnectionManager()

    # Static files
    static_path = Path(__file__).parent / config.static_dir
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    # ========================================================================
    # Routes
    # ========================================================================

    @app.get("/")
    async def root():
        """Serve main chat page."""
        html_path = Path(__file__).parent / "templates" / "chat.html"

        if html_path.exists():
            return FileResponse(html_path)
        else:
            return HTMLResponse("""
            <html>
                <head><title>HoloLoom Chat</title></head>
                <body>
                    <h1>HoloLoom Chat</h1>
                    <p>Chat interface not fully installed. See templates/chat.html</p>
                    <p>WebSocket endpoint: ws://localhost:8000/ws/chat/{session_id}</p>
                </body>
            </html>
            """)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "active_sessions": len(manager.sessions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.websocket("/ws/chat/{session_id}")
    async def websocket_chat(websocket: WebSocket, session_id: str):
        """
        WebSocket endpoint for chat.

        Protocol:
        - Client sends: {"type": "message", "text": "user message", "metadata": {...}}
        - Server sends: {"type": "response_chunk", "text": "...", "done": false}
        - Server sends: {"type": "response_chunk", "text": "", "done": true}
        - Server sends: {"type": "trace", "data": {...}}  # Weaving trace
        """
        await manager.connect(websocket, session_id)

        try:
            while True:
                # Receive message
                data = await websocket.receive_json()

                if data.get("type") == "message":
                    user_message = data.get("text", "")

                    # Validate
                    if not user_message or len(user_message) > config.max_message_length:
                        await manager.send_message(session_id, {
                            "type": "error",
                            "error": "Invalid message length"
                        })
                        continue

                    # Store in history
                    manager.sessions[session_id]["history"].append({
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                    # Update activity
                    manager.sessions[session_id]["last_activity"] = datetime.now(timezone.utc)

                    # Send thinking indicator
                    await manager.send_message(session_id, {
                        "type": "thinking",
                        "message": "Processing query..."
                    })

                    # Process message (mock response for now)
                    # TODO: Integrate with HoloLoom orchestrator
                    response_text = await process_message(user_message, session_id)

                    # Stream response
                    await manager.stream_response(session_id, response_text)

                    # Store response
                    manager.sessions[session_id]["history"].append({
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })

                elif data.get("type") == "ping":
                    # Heartbeat
                    await manager.send_message(session_id, {"type": "pong"})

        except WebSocketDisconnect:
            manager.disconnect(websocket, session_id)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            manager.disconnect(websocket, session_id)

    @app.get("/api/history/{session_id}")
    async def get_history(session_id: str):
        """Get chat history for session."""
        if session_id not in manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session_id,
            "history": manager.sessions[session_id]["history"],
            "created_at": manager.sessions[session_id]["created_at"].isoformat()
        }

    @app.post("/api/clear/{session_id}")
    async def clear_history(session_id: str):
        """Clear chat history for session."""
        if session_id in manager.sessions:
            manager.sessions[session_id]["history"] = []
            return {"status": "cleared"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")

    return app


async def process_message(message: str, session_id: str) -> str:
    """
    Process chat message.

    TODO: Integrate with HoloLoom orchestrator:
    - Extract features (motifs, embeddings)
    - Retrieve context from memory
    - Run policy decision
    - Execute tools via MCP
    - Generate response

    Args:
        message: User message
        session_id: Session identifier

    Returns:
        Response text
    """
    # Mock response for now
    await asyncio.sleep(0.5)  # Simulate processing

    responses = {
        "thompson": "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It balances exploration and exploitation by sampling from posterior distributions for each arm.",
        "hololoom": "HoloLoom is a neural decision-making system that combines multi-scale embeddings, knowledge graphs, and reinforcement learning through a weaving metaphor.",
        "default": f"You said: '{message}'. This is a mock response. Integrate with HoloLoom orchestrator for real processing."
    }

    # Simple keyword matching
    message_lower = message.lower()
    for keyword, response in responses.items():
        if keyword in message_lower:
            return response

    return responses["default"]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed: pip install fastapi uvicorn websockets")
    else:
        import uvicorn

        print("="*80)
        print("HoloLoom Chat Server")
        print("="*80)
        print("\nStarting server...")
        print("Open browser to: http://localhost:8000")
        print("WebSocket endpoint: ws://localhost:8000/ws/chat/{session_id}\n")

        config = ChatConfig(host="0.0.0.0", port=8000)
        app = create_app(config)

        uvicorn.run(app, host=config.host, port=config.port)
