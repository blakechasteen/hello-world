"""
Mobile Client API - REST API for iOS/Android O2 Clients

Provides HTTP REST API for mobile clients to interact with O2 Platform.

Features:
- Authentication (Matrix login)
- Query HoloLoom (chat interface)
- Governance (proposals, voting)
- Memory management (export, share)
- Real-time updates (WebSocket)
- Push notifications
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import asyncio
import logging
import jwt
import os

logger = logging.getLogger('o2-bot.mobile_api')

# FastAPI app
app = FastAPI(
    title="O2 Mobile API",
    description="REST API for O2 Platform mobile clients",
    version="1.0.0"
)

# CORS (allow mobile apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv('JWT_SECRET', 'changeme-generate-secret')

# ============================================
# Request/Response Models
# ============================================

class LoginRequest(BaseModel):
    """Login request."""
    user_id: str
    password: str


class LoginResponse(BaseModel):
    """Login response with JWT token."""
    token: str
    user_id: str
    expires_at: str


class QueryRequest(BaseModel):
    """HoloLoom query request."""
    query: str
    mode: Optional[str] = "direct"  # direct, verify, research, plan_execute


class QueryResponse(BaseModel):
    """HoloLoom query response."""
    response: str
    confidence: float
    sources: List[str]
    reasoning_mode: str


class ProposalRequest(BaseModel):
    """Create proposal request."""
    room_id: str
    title: str
    description: Optional[str] = ""
    proposal_type: Optional[str] = "policy"
    threshold: Optional[float] = 0.67
    duration_hours: Optional[int] = 72


class ProposalResponse(BaseModel):
    """Proposal response."""
    proposal_id: int
    title: str
    status: str
    created_at: str


class VoteRequest(BaseModel):
    """Vote on proposal request."""
    proposal_id: int
    vote: str  # yes, no, abstain


class VoteResponse(BaseModel):
    """Vote response."""
    success: bool
    message: str


class MemoryExportResponse(BaseModel):
    """Memory export response."""
    export_url: str
    size_bytes: int
    created_at: str


# ============================================
# Dependency: Authentication
# ============================================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Extract user from JWT token.

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        User ID

    Raises:
        HTTPException if token invalid
    """
    token = credentials.credentials

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get('user_id')

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        return user_id

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ============================================
# Authentication Endpoints
# ============================================

@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login to O2 Platform.

    Validates Matrix credentials and returns JWT token.
    """
    # TODO: Validate Matrix credentials
    # For now, accept any login (development only!)

    # Generate JWT token
    from datetime import timedelta
    expires = datetime.now() + timedelta(days=7)

    payload = {
        'user_id': request.user_id,
        'exp': expires
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    return LoginResponse(
        token=token,
        user_id=request.user_id,
        expires_at=expires.isoformat()
    )


@app.post("/auth/logout")
async def logout(user_id: str = Depends(get_current_user)):
    """Logout (invalidate token)."""
    # TODO: Add token blacklist
    return {"message": "Logged out successfully"}


# ============================================
# Query Endpoints
# ============================================

@app.post("/query", response_model=QueryResponse)
async def query_hololoom(
    request: QueryRequest,
    user_id: str = Depends(get_current_user)
):
    """
    Query HoloLoom.

    Sends query to user's HoloLoom instance and returns response.
    """
    # TODO: Integrate with actual HoloLoom
    # For now, return mock response

    return QueryResponse(
        response=f"Mock response to: {request.query}",
        confidence=0.85,
        sources=["Source 1", "Source 2"],
        reasoning_mode=request.mode
    )


# ============================================
# Governance Endpoints
# ============================================

@app.get("/proposals")
async def list_proposals(
    room_id: Optional[str] = None,
    status: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """List proposals."""
    # TODO: Integrate with governance engine
    return {
        "proposals": [
            {
                "proposal_id": 1,
                "title": "Example Proposal",
                "status": "active",
                "created_at": datetime.now().isoformat()
            }
        ]
    }


@app.post("/proposals", response_model=ProposalResponse)
async def create_proposal(
    request: ProposalRequest,
    user_id: str = Depends(get_current_user)
):
    """Create new proposal."""
    # TODO: Integrate with governance engine

    return ProposalResponse(
        proposal_id=1,
        title=request.title,
        status="active",
        created_at=datetime.now().isoformat()
    )


@app.post("/proposals/{proposal_id}/vote", response_model=VoteResponse)
async def vote_on_proposal(
    proposal_id: int,
    request: VoteRequest,
    user_id: str = Depends(get_current_user)
):
    """Vote on proposal."""
    # TODO: Integrate with governance engine

    return VoteResponse(
        success=True,
        message=f"Vote '{request.vote}' recorded on proposal #{proposal_id}"
    )


@app.get("/proposals/{proposal_id}")
async def get_proposal(
    proposal_id: int,
    user_id: str = Depends(get_current_user)
):
    """Get proposal details."""
    # TODO: Integrate with governance engine

    return {
        "proposal_id": proposal_id,
        "title": "Example Proposal",
        "description": "This is an example proposal",
        "status": "active",
        "votes": {
            "yes": 5,
            "no": 2,
            "abstain": 1
        }
    }


# ============================================
# Memory Endpoints
# ============================================

@app.post("/memory/export", response_model=MemoryExportResponse)
async def export_memory(user_id: str = Depends(get_current_user)):
    """Export user's memory graph."""
    # TODO: Integrate with federated memory manager

    return MemoryExportResponse(
        export_url="/memory/exports/user_123.o2.json",
        size_bytes=1024000,
        created_at=datetime.now().isoformat()
    )


@app.get("/memory/shared")
async def list_shared_memories(user_id: str = Depends(get_current_user)):
    """List memories shared with user."""
    # TODO: Integrate with memory sharing manager

    return {
        "shared_memories": [
            {
                "memory_id": "memory_1",
                "owner_id": "@alice:matrix.org",
                "permissions": ["read"],
                "shared_at": datetime.now().isoformat()
            }
        ]
    }


# ============================================
# Real-time WebSocket
# ============================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        """Remove WebSocket connection."""
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        """Send message to specific user."""
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def broadcast(self, message: str):
        """Broadcast message to all connected users."""
        for connection in self.active_connections.values():
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Clients can connect to receive:
    - New proposals
    - Vote updates
    - Swarm task progress
    - Memory share notifications
    """
    # TODO: Authenticate WebSocket connection
    user_id = "anonymous"  # Extract from query params or token

    await manager.connect(user_id, websocket)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()

            # Echo back (for now)
            await manager.send_personal_message(f"Echo: {data}", user_id)

    except WebSocketDisconnect:
        manager.disconnect(user_id)


# ============================================
# Health & Info
# ============================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info")
async def info():
    """Platform info."""
    return {
        "platform": "O2",
        "description": "Platform Anarchism meets Agentic Intelligence",
        "version": "1.0.0",
        "features": [
            "Democratic Governance",
            "Federated Memory",
            "Agentic Swarms",
            "User Data Sovereignty"
        ],
        "documentation": "https://docs.o2-platform.org"
    }


# ============================================
# Run Server
# ============================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "mobile_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
