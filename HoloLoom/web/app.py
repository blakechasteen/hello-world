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
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Form, Query
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not installed. Web interface unavailable.")
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None

# Import auth module
try:
    from .auth import (
        user_db,
        create_access_token,
        verify_token_for_websocket,
        get_current_user,
        User,
        get_demo_credentials,
        session_manager
    )
    AUTH_AVAILABLE = True
except ImportError:
    logger.warning("Auth module unavailable")
    AUTH_AVAILABLE = False


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

    # ========================================================================
    # Authentication Routes
    # ========================================================================

    @app.get("/login")
    async def login_page():
        """Serve login page."""
        html_path = Path(__file__).parent / "templates" / "login.html"

        if html_path.exists():
            return FileResponse(html_path)
        else:
            # Inline login page
            return HTMLResponse("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>HoloLoom - Login</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .login-container {
                        background: white;
                        border-radius: 12px;
                        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                        padding: 40px;
                        width: 400px;
                        max-width: 90%;
                    }
                    h1 {
                        color: #333;
                        margin-bottom: 10px;
                        font-size: 28px;
                    }
                    .subtitle {
                        color: #666;
                        margin-bottom: 30px;
                        font-size: 14px;
                    }
                    .form-group {
                        margin-bottom: 20px;
                    }
                    label {
                        display: block;
                        color: #444;
                        margin-bottom: 8px;
                        font-weight: 500;
                    }
                    input {
                        width: 100%;
                        padding: 12px;
                        border: 2px solid #e0e0e0;
                        border-radius: 6px;
                        font-size: 14px;
                        transition: border-color 0.3s;
                    }
                    input:focus {
                        outline: none;
                        border-color: #667eea;
                    }
                    button {
                        width: 100%;
                        padding: 12px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: transform 0.2s;
                    }
                    button:hover {
                        transform: translateY(-2px);
                    }
                    .demo-info {
                        margin-top: 20px;
                        padding: 15px;
                        background: #f5f5f5;
                        border-radius: 6px;
                        font-size: 13px;
                        color: #666;
                    }
                    .demo-info strong {
                        color: #333;
                    }
                    .error {
                        color: #e74c3c;
                        margin-top: 10px;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <div class="login-container">
                    <h1>🌀 HoloLoom</h1>
                    <p class="subtitle">Neural Decision-Making System</p>

                    <form id="loginForm">
                        <div class="form-group">
                            <label for="username">Username</label>
                            <input type="text" id="username" name="username" required autofocus>
                        </div>
                        <div class="form-group">
                            <label for="password">Password</label>
                            <input type="password" id="password" name="password" required>
                        </div>
                        <button type="submit">Login</button>
                        <div class="error" id="error" style="display: none;"></div>
                    </form>

                    <div class="demo-info">
                        <strong>Demo Accounts:</strong><br>
                        • admin / admin123<br>
                        • demo / demo123
                    </div>
                </div>

                <script>
                    document.getElementById('loginForm').addEventListener('submit', async (e) => {
                        e.preventDefault();

                        const username = document.getElementById('username').value;
                        const password = document.getElementById('password').value;
                        const errorDiv = document.getElementById('error');

                        try {
                            const response = await fetch('/api/auth/login', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({username, password})
                            });

                            if (response.ok) {
                                const data = await response.json();
                                localStorage.setItem('access_token', data.access_token);
                                window.location.href = '/';
                            } else {
                                const error = await response.json();
                                errorDiv.textContent = error.detail || 'Login failed';
                                errorDiv.style.display = 'block';
                            }
                        } catch (error) {
                            errorDiv.textContent = 'Network error. Please try again.';
                            errorDiv.style.display = 'block';
                        }
                    });
                </script>
            </body>
            </html>
            """)

    @app.post("/api/auth/login")
    async def login(username: str = Form(), password: str = Form(None)):
        """Login endpoint."""
        if not AUTH_AVAILABLE:
            raise HTTPException(status_code=500, detail="Authentication not available")

        # Support both form and JSON
        if password is None:
            # JSON request
            from fastapi import Request
            request = Request
            # Will be handled by form parameter above

        # Authenticate
        user = user_db.authenticate_user(username, password)

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password"
            )

        # Create token
        access_token = create_access_token(data={"sub": user.username})

        # Create session
        session_id = session_manager.create_session(user.username, access_token)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": user.username,
            "session_id": session_id
        }

    @app.post("/api/auth/login_json")
    async def login_json(credentials: dict):
        """Login endpoint for JSON requests."""
        if not AUTH_AVAILABLE:
            raise HTTPException(status_code=500, detail="Authentication not available")

        username = credentials.get("username")
        password = credentials.get("password")

        if not username or not password:
            raise HTTPException(status_code=400, detail="Missing credentials")

        user = user_db.authenticate_user(username, password)

        if not user:
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password"
            )

        access_token = create_access_token(data={"sub": user.username})
        session_id = session_manager.create_session(user.username, access_token)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": user.username,
            "session_id": session_id
        }

    @app.post("/api/auth/logout")
    async def logout(user: User = Depends(get_current_user)):
        """Logout endpoint."""
        # In a real app, invalidate token on server side
        return {"status": "logged_out", "username": user.username}

    @app.get("/api/auth/me")
    async def get_current_user_info(user: User = Depends(get_current_user)):
        """Get current user info (protected endpoint example)."""
        return user.to_dict()

    @app.websocket("/ws/chat/{session_id}")
    async def websocket_chat(websocket: WebSocket, session_id: str, token: str = Query(None)):
        """
        WebSocket endpoint for chat (protected with JWT).

        Protocol:
        - Connect with: ws://host/ws/chat/{session_id}?token={jwt_token}
        - Client sends: {"type": "message", "text": "user message", "metadata": {...}}
        - Server sends: {"type": "response_chunk", "text": "...", "done": false}
        - Server sends: {"type": "response_chunk", "text": "", "done": true}
        - Server sends: {"type": "trace", "data": {...}}  # Weaving trace
        """
        # Verify authentication
        if AUTH_AVAILABLE and token:
            username = verify_token_for_websocket(token)
            if not username:
                await websocket.close(code=1008, reason="Invalid or expired token")
                return
            logger.info(f"Authenticated WebSocket for user: {username}")
        elif AUTH_AVAILABLE:
            await websocket.close(code=1008, reason="Authentication required")
            return

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
