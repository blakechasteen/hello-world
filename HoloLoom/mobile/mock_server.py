"""
Mock Mobile API Server
======================
Simple mock server for testing mobile clients during development.

Usage:
    python mock_server.py --port 8000

Features:
- Mock REST API endpoints
- Mock WebSocket server
- Simulated streaming responses
- No database required
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Form, File, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    print("FastAPI not installed: pip install fastapi uvicorn websockets")
    FASTAPI_AVAILABLE = False
    exit(1)


app = FastAPI(
    title="HoloLoom Mock API",
    version="1.0.0",
    description="Mock server for mobile client development"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data
MOCK_TOKEN = "mock_jwt_token_12345"
MOCK_SESSION_ID = "mock_session_123"
MOCK_MESSAGES: List[Dict] = []


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }


# Authentication
@app.post("/auth/login")
@app.post("/api/auth/login")
async def login(username: str = Form(), password: str = Form()):
    """Mock login."""
    return {
        "access_token": MOCK_TOKEN,
        "refresh_token": "mock_refresh_token",
        "token_type": "bearer",
        "expires_in": 86400,
        "username": username,
        "session_id": MOCK_SESSION_ID,
        "user": {
            "username": username,
            "email": f"{username}@example.com",
            "full_name": username.title(),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }


@app.post("/api/auth/login_json")
async def login_json(credentials: dict):
    """Mock login (JSON)."""
    username = credentials.get("username")
    return {
        "access_token": MOCK_TOKEN,
        "refresh_token": "mock_refresh_token",
        "token_type": "bearer",
        "expires_in": 86400,
        "username": username,
        "session_id": MOCK_SESSION_ID,
        "user": {
            "username": username,
            "email": f"{username}@example.com",
            "full_name": username.title(),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    }


@app.post("/api/auth/logout")
async def logout():
    """Mock logout."""
    return {"status": "logged_out"}


@app.get("/api/auth/me")
async def get_me():
    """Mock get current user."""
    return {
        "username": "demo",
        "email": "demo@example.com",
        "full_name": "Demo User",
        "created_at": datetime.now(timezone.utc).isoformat()
    }


# Chat Sessions
@app.get("/api/chat/sessions")
async def list_sessions(limit: int = 20, offset: int = 0):
    """Mock list sessions."""
    return {
        "sessions": [
            {
                "session_id": "session_1",
                "title": "Conversation 1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "message_count": 5,
                "metadata": {}
            },
            {
                "session_id": "session_2",
                "title": "Conversation 2",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "message_count": 3,
                "metadata": {}
            }
        ],
        "total": 2,
        "limit": limit,
        "offset": offset
    }


@app.post("/api/chat/sessions")
async def create_session(body: dict):
    """Mock create session."""
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    return {
        "session_id": session_id,
        "title": body.get("title", "New conversation"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_activity": datetime.now(timezone.utc).isoformat(),
        "message_count": 0,
        "metadata": body.get("metadata", {})
    }


@app.get("/api/chat/sessions/{session_id}/messages")
async def get_messages(session_id: str, limit: int = 50):
    """Mock get messages."""
    return {
        "messages": MOCK_MESSAGES,
        "has_more": False
    }


@app.post("/api/chat/sessions/{session_id}/messages")
async def send_message(session_id: str, body: dict):
    """Mock send message (non-streaming)."""
    text = body.get("text", "")

    user_message = {
        "message_id": f"msg_{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "role": "user",
        "content": text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attachments": [],
        "metadata": {},
        "status": "sent"
    }

    # Generate mock response
    response_text = generate_mock_response(text)

    assistant_message = {
        "message_id": f"msg_{uuid.uuid4().hex[:8]}",
        "session_id": session_id,
        "role": "assistant",
        "content": response_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attachments": [],
        "metadata": {},
        "status": "sent"
    }

    MOCK_MESSAGES.extend([user_message, assistant_message])

    return {
        "user_message": user_message,
        "assistant_message": assistant_message,
        "trace": {
            "motifs": ["question", "technical"],
            "embeddings": {
                "scales": [96, 192, 384],
                "similarity_scores": [0.82, 0.85, 0.87]
            },
            "tool_selection": {
                "selected_tool": "knowledge_search",
                "confidence": 0.87,
                "strategy": "epsilon_greedy"
            },
            "execution_time_ms": 123.4
        }
    }


# File Upload
@app.post("/api/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Mock file upload."""
    file_id = f"file_{uuid.uuid4().hex[:8]}"

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "file_type": "image",
        "content_type": file.content_type,
        "file_size": 1024 * 50,  # Mock 50KB
        "url": f"/api/files/{file_id}",
        "upload_time": datetime.now(timezone.utc).isoformat()
    }


# Sync
@app.post("/api/sync/pull")
async def sync_pull(body: dict):
    """Mock sync pull."""
    return {
        "messages": [],
        "sessions": [],
        "deleted_ids": {
            "messages": [],
            "sessions": []
        },
        "sync_timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/sync/push")
async def sync_push(body: dict):
    """Mock sync push."""
    return {
        "status": "synced",
        "conflicts": [],
        "sync_timestamp": datetime.now(timezone.utc).isoformat()
    }


# Push Notifications
@app.post("/api/push/register")
async def register_push(body: dict):
    """Mock push registration."""
    device_id = body.get("device_id", f"device_{uuid.uuid4().hex[:8]}")
    return {
        "status": "registered",
        "device_id": device_id
    }


@app.post("/api/push/unregister")
async def unregister_push(body: dict):
    """Mock push unregistration."""
    return {
        "status": "unregistered"
    }


# User Profile
@app.get("/api/user/profile")
async def get_profile():
    """Mock get profile."""
    return {
        "username": "demo",
        "email": "demo@example.com",
        "full_name": "Demo User",
        "avatar_url": None,
        "bio": None,
        "preferences": {}
    }


@app.put("/api/user/profile")
async def update_profile(body: dict):
    """Mock update profile."""
    return body


@app.get("/api/user/settings")
async def get_settings():
    """Mock get settings."""
    return {
        "theme": "auto",
        "language": "en",
        "notifications_enabled": True,
        "sound_enabled": True,
        "typing_indicators": True
    }


# WebSocket Chat
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """Mock WebSocket chat with streaming."""
    await websocket.accept()

    # Send connected message
    await websocket.send_json({
        "type": "connected",
        "session_id": session_id,
        "server_time": datetime.now(timezone.utc).isoformat()
    })

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            if data.get("type") == "message":
                user_message = data.get("text", "")

                # Send thinking
                await websocket.send_json({
                    "type": "thinking",
                    "message": "Processing query...",
                    "stage": "feature_extraction"
                })

                await asyncio.sleep(0.5)

                # Generate response
                response_text = generate_mock_response(user_message)

                # Stream response in chunks
                chunk_size = 10
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i:i + chunk_size]

                    await websocket.send_json({
                        "type": "response_chunk",
                        "text": chunk,
                        "done": False,
                        "message_id": f"msg_{uuid.uuid4().hex[:8]}"
                    })

                    await asyncio.sleep(0.05)  # Simulate streaming

                # Send completion
                await websocket.send_json({
                    "type": "response_chunk",
                    "text": "",
                    "done": True,
                    "message_id": f"msg_{uuid.uuid4().hex[:8]}"
                })

                # Send complete response with trace
                await websocket.send_json({
                    "type": "response_complete",
                    "message": {
                        "message_id": f"msg_{uuid.uuid4().hex[:8]}",
                        "session_id": session_id,
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "metadata": {}
                    },
                    "trace": {
                        "motifs": ["question", "technical"],
                        "embeddings": {
                            "scales": [96, 192, 384],
                            "similarity_scores": [0.82, 0.85, 0.87]
                        },
                        "tool_selection": {
                            "selected_tool": "knowledge_search",
                            "confidence": 0.87,
                            "strategy": "epsilon_greedy"
                        },
                        "execution_time_ms": 234.5
                    }
                })

            elif data.get("type") == "ping":
                # Heartbeat response
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print(f"WebSocket disconnected: {session_id}")


def generate_mock_response(user_message: str) -> str:
    """Generate a mock response based on user message."""
    message_lower = user_message.lower()

    if "thompson" in message_lower:
        return (
            "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
            "It balances exploration and exploitation by sampling from posterior distributions "
            "for each arm, making it particularly effective for online learning scenarios."
        )
    elif "hololoom" in message_lower:
        return (
            "HoloLoom is a neural decision-making system that combines multi-scale embeddings, "
            "knowledge graph memory, and reinforcement learning. It uses a weaving metaphor "
            "where independent 'warp thread' modules are coordinated by an orchestrator."
        )
    elif "hello" in message_lower or "hi" in message_lower:
        return (
            "Hello! I'm the HoloLoom mock assistant. I can help you test the mobile API. "
            "Try asking about Thompson Sampling, HoloLoom, or any other topic!"
        )
    else:
        return (
            f"You said: '{user_message}'. This is a mock response from the test server. "
            f"The real HoloLoom system would process this through the full pipeline "
            f"with feature extraction, memory retrieval, policy decision, and tool execution."
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HoloLoom Mock API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print("=" * 80)
    print("HoloLoom Mock API Server")
    print("=" * 80)
    print(f"\nServer running at: http://{args.host}:{args.port}")
    print(f"API docs: http://localhost:{args.port}/docs")
    print(f"WebSocket: ws://localhost:{args.port}/ws/chat/{{session_id}}\n")
    print("Demo credentials:")
    print("  Username: admin")
    print("  Password: admin123")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port)
