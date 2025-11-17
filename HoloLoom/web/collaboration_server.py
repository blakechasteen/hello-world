"""
Collaborative WebSocket Server
================================
Real-time collaboration server with CRDT synchronization.

Philosophy:
Multiple shuttles weaving the same fabric in real-time. The server is
the loom itself - coordinating threads, preventing tangles, ensuring
eventual convergence of all weavers.

Protocol:
- join: User joins session
- leave: User leaves session
- update: State change (CRDT delta)
- cursor: Cursor position update
- presence: Presence status change
- sync: Full state synchronization
- activity: Activity feed event
"""

import logging
import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Depends
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not installed. Collaboration server unavailable.")
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None

# Import collaboration modules
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from collaboration.session import SessionManager, CollaborativeSession
    from collaboration.presence import PresenceTracker, CursorPosition, PresenceStatus
    from collaboration.sync import CRDTEngine, Delta, OperationType
    from collaboration.permissions import PermissionManager, Role, Permission
    from collaboration.activity import ActivityLogger, EventType, ActivityEvent
    from web.auth import verify_token_for_websocket, User, get_current_user

    COLLABORATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Collaboration modules unavailable: {e}")
    COLLABORATION_AVAILABLE = False


@dataclass
class CollaborationMessage:
    """WebSocket message structure."""
    type: str                           # Message type (join, leave, update, etc.)
    session_id: str                     # Session ID
    user_id: str                        # Sender user ID
    data: Dict[str, Any] = field(default_factory=dict)  # Message payload
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CollaborationMessage':
        """Create from dictionary."""
        return CollaborationMessage(
            type=data["type"],
            session_id=data["session_id"],
            user_id=data["user_id"],
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc)
        )


class CollaborationConnectionManager:
    """
    Manages WebSocket connections for collaboration.

    Features:
    - Connection lifecycle
    - Message broadcasting
    - Room management (sessions)
    - Presence tracking
    """

    def __init__(self):
        """Initialize connection manager."""
        # WebSocket connections: session_id -> user_id -> websocket
        self.connections: Dict[str, Dict[str, WebSocket]] = {}

        logger.info("CollaborationConnectionManager initialized")

    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: str
    ):
        """
        Connect user to session.

        Args:
            websocket: WebSocket connection
            session_id: Session ID
            user_id: User ID
        """
        await websocket.accept()

        if session_id not in self.connections:
            self.connections[session_id] = {}

        self.connections[session_id][user_id] = websocket

        logger.info(f"User {user_id} connected to session {session_id}")

    def disconnect(self, session_id: str, user_id: str):
        """Disconnect user from session."""
        if session_id in self.connections:
            if user_id in self.connections[session_id]:
                del self.connections[session_id][user_id]

                # Clean up empty sessions
                if not self.connections[session_id]:
                    del self.connections[session_id]

                logger.info(f"User {user_id} disconnected from session {session_id}")

    async def broadcast_to_session(
        self,
        session_id: str,
        message: Dict[str, Any],
        exclude_user: Optional[str] = None
    ):
        """
        Broadcast message to all users in session.

        Args:
            session_id: Session ID
            message: Message to broadcast
            exclude_user: Optional user to exclude from broadcast
        """
        if session_id not in self.connections:
            return

        disconnected = []

        for user_id, websocket in self.connections[session_id].items():
            if exclude_user and user_id == exclude_user:
                continue

            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to {user_id}: {e}")
                disconnected.append(user_id)

        # Clean up disconnected users
        for user_id in disconnected:
            self.disconnect(session_id, user_id)

    async def send_to_user(
        self,
        session_id: str,
        user_id: str,
        message: Dict[str, Any]
    ):
        """Send message to specific user."""
        if session_id not in self.connections:
            return

        if user_id not in self.connections[session_id]:
            return

        websocket = self.connections[session_id][user_id]

        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending to {user_id}: {e}")
            self.disconnect(session_id, user_id)

    def get_session_users(self, session_id: str) -> List[str]:
        """Get list of connected users in session."""
        if session_id not in self.connections:
            return []

        return list(self.connections[session_id].keys())

    def get_total_connections(self) -> int:
        """Get total number of active connections."""
        return sum(len(users) for users in self.connections.values())


class CollaborationServer:
    """
    Main collaboration server.

    Integrates all collaboration components:
    - Session management
    - Presence tracking
    - CRDT synchronization
    - Permissions
    - Activity logging
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize collaboration server.

        Args:
            storage_dir: Directory for persistence
        """
        if not COLLABORATION_AVAILABLE:
            raise RuntimeError("Collaboration modules not available")

        storage_dir = storage_dir or Path("./data/collaboration")

        # Initialize components
        self.connection_manager = CollaborationConnectionManager()
        self.session_manager = SessionManager(storage_dir=storage_dir / "sessions")
        self.presence_tracker = PresenceTracker(idle_threshold_seconds=60, timeout_seconds=30)
        self.permission_manager = PermissionManager()
        self.activity_logger = ActivityLogger(storage_dir=storage_dir / "activity")

        # CRDT engines per session: session_id -> user_id -> engine
        self.crdt_engines: Dict[str, Dict[str, CRDTEngine]] = {}

        logger.info(f"CollaborationServer initialized with storage: {storage_dir}")

    def get_or_create_crdt_engine(self, session_id: str, user_id: str) -> CRDTEngine:
        """Get or create CRDT engine for user in session."""
        if session_id not in self.crdt_engines:
            self.crdt_engines[session_id] = {}

        if user_id not in self.crdt_engines[session_id]:
            self.crdt_engines[session_id][user_id] = CRDTEngine(user_id=user_id)

        return self.crdt_engines[session_id][user_id]

    async def handle_join(
        self,
        websocket: WebSocket,
        session_id: str,
        user_id: str,
        username: str
    ):
        """Handle user joining session."""
        # Get or create session
        session = self.session_manager.get_session(session_id)
        if not session:
            # Create new session
            session = self.session_manager.create_session(
                name=f"Session {session_id[:8]}",
                owner_id=user_id,
                owner_username=username,
                session_id=session_id
            )
        else:
            # Add user to existing session
            user = session.get_user(user_id)
            if not user:
                session.add_user(user_id, username, role="editor")

        # Connect WebSocket
        await self.connection_manager.connect(websocket, session_id, user_id)

        # Add to presence tracker
        self.presence_tracker.add_user(session_id, user_id, username)

        # Initialize CRDT engine
        engine = self.get_or_create_crdt_engine(session_id, user_id)

        # Log activity
        self.activity_logger.log_event(
            EventType.USER_JOINED,
            session_id,
            user_id,
            username
        )

        # Send full state to new user
        full_state = self._get_session_state(session_id)
        await self.connection_manager.send_to_user(
            session_id,
            user_id,
            {
                "type": "sync",
                "data": {
                    "state": full_state,
                    "users": [u.to_dict() for u in session.users.values()],
                    "presence": self.presence_tracker.get_session_summary(session_id)
                }
            }
        )

        # Notify others
        await self.connection_manager.broadcast_to_session(
            session_id,
            {
                "type": "user_joined",
                "data": {
                    "user_id": user_id,
                    "username": username
                }
            },
            exclude_user=user_id
        )

        logger.info(f"User {username} joined session {session_id}")

    async def handle_leave(self, session_id: str, user_id: str, username: str):
        """Handle user leaving session."""
        # Remove from presence
        self.presence_tracker.remove_user(session_id, user_id)

        # Remove from session
        self.session_manager.leave_session(session_id, user_id)

        # Disconnect WebSocket
        self.connection_manager.disconnect(session_id, user_id)

        # Log activity
        self.activity_logger.log_event(
            EventType.USER_LEFT,
            session_id,
            user_id,
            username
        )

        # Notify others
        await self.connection_manager.broadcast_to_session(
            session_id,
            {
                "type": "user_left",
                "data": {
                    "user_id": user_id,
                    "username": username
                }
            }
        )

        logger.info(f"User {username} left session {session_id}")

    async def handle_update(
        self,
        session_id: str,
        user_id: str,
        username: str,
        delta_data: Dict[str, Any]
    ):
        """Handle state update (CRDT delta)."""
        # Check permission
        session = self.session_manager.get_session(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return

        user = session.get_user(user_id)
        if not user:
            logger.error(f"User {user_id} not in session")
            return

        role = Role(user.role)
        check = self.permission_manager.can_perform_action(
            user_id,
            role,
            "update_state",
            session_id
        )

        if not check.allowed:
            logger.warning(f"Permission denied: {check.reason}")
            return

        # Create delta
        delta = Delta.from_dict(delta_data)

        # Apply to user's CRDT engine
        engine = self.get_or_create_crdt_engine(session_id, user_id)
        engine.apply_delta(delta)

        # Broadcast to all other users
        await self.connection_manager.broadcast_to_session(
            session_id,
            {
                "type": "update",
                "data": {
                    "delta": delta.to_dict(),
                    "user_id": user_id,
                    "username": username
                }
            },
            exclude_user=user_id
        )

        # Apply delta to other users' engines
        for uid in self.connection_manager.get_session_users(session_id):
            if uid != user_id:
                other_engine = self.get_or_create_crdt_engine(session_id, uid)
                other_engine.apply_delta(delta)

        # Log activity
        self.activity_logger.log_event(
            EventType.DELTA_APPLIED,
            session_id,
            user_id,
            username,
            data={"delta": delta.to_dict()}
        )

        logger.debug(f"Applied delta from {username} in session {session_id}")

    async def handle_cursor(
        self,
        session_id: str,
        user_id: str,
        cursor_data: Dict[str, Any]
    ):
        """Handle cursor position update."""
        cursor = CursorPosition.from_dict(cursor_data)

        # Update presence
        self.presence_tracker.update_cursor(session_id, user_id, cursor)

        # Broadcast to others
        await self.connection_manager.broadcast_to_session(
            session_id,
            {
                "type": "cursor",
                "data": {
                    "user_id": user_id,
                    "cursor": cursor.to_dict()
                }
            },
            exclude_user=user_id
        )

    async def handle_presence(
        self,
        session_id: str,
        user_id: str,
        status: str
    ):
        """Handle presence status update."""
        # Update presence
        presence = self.presence_tracker.get_user_presence(session_id, user_id)
        if presence:
            presence.status = PresenceStatus(status)

        # Broadcast to others
        await self.connection_manager.broadcast_to_session(
            session_id,
            {
                "type": "presence",
                "data": {
                    "user_id": user_id,
                    "status": status
                }
            },
            exclude_user=user_id
        )

    async def handle_heartbeat(self, session_id: str, user_id: str):
        """Handle heartbeat ping."""
        # Update presence
        self.presence_tracker.update_heartbeat(session_id, user_id)

        # Update session activity
        session = self.session_manager.get_session(session_id)
        if session:
            session.update_user_activity(user_id)

    def _get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get current session state."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return {}

        # Get state from any user's CRDT engine
        if session_id in self.crdt_engines:
            engines = self.crdt_engines[session_id]
            if engines:
                # Use first available engine
                engine = list(engines.values())[0]
                return engine.get_full_state()

        return session.state

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None

        return {
            "session": session.to_dict(),
            "presence": self.presence_tracker.get_session_summary(session_id),
            "activity": self.activity_logger.get_session_summary(session_id),
            "connected_users": self.connection_manager.get_session_users(session_id)
        }


def create_collaboration_app(storage_dir: Optional[Path] = None) -> FastAPI:
    """
    Create FastAPI app with collaboration endpoints.

    Args:
        storage_dir: Storage directory

    Returns:
        FastAPI app
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed")

    if not COLLABORATION_AVAILABLE:
        raise RuntimeError("Collaboration modules not available")

    app = FastAPI(title="HoloLoom Collaboration Server", version="1.0.0")

    # Create collaboration server
    collab_server = CollaborationServer(storage_dir=storage_dir)

    # Start presence monitoring
    collab_server.presence_tracker.start_monitoring(check_interval=10)

    @app.get("/health")
    async def health():
        """Health check."""
        return {
            "status": "healthy",
            "total_connections": collab_server.connection_manager.get_total_connections(),
            "total_sessions": len(collab_server.session_manager.sessions)
        }

    @app.get("/api/sessions")
    async def list_sessions(user: User = Depends(get_current_user)):
        """List all sessions for user."""
        sessions = collab_server.session_manager.list_sessions(user_id=user.username)

        return {
            "sessions": [s.to_dict() for s in sessions]
        }

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str, user: User = Depends(get_current_user)):
        """Get session details."""
        info = collab_server.get_session_info(session_id)

        if not info:
            return JSONResponse(
                status_code=404,
                content={"error": "Session not found"}
            )

        return info

    @app.websocket("/ws/collaborate/{session_id}")
    async def websocket_collaborate(
        websocket: WebSocket,
        session_id: str,
        token: str = Query(None)
    ):
        """
        WebSocket endpoint for collaboration.

        Protocol:
        - Client sends: {"type": "join", "user_id": "...", "username": "..."}
        - Client sends: {"type": "update", "data": {"delta": ...}}
        - Client sends: {"type": "cursor", "data": {"line": ..., "column": ...}}
        - Client sends: {"type": "presence", "data": {"status": "online|idle|offline"}}
        - Client sends: {"type": "heartbeat"}
        - Client sends: {"type": "leave"}

        - Server sends: {"type": "sync", "data": {"state": ..., "users": ..., "presence": ...}}
        - Server sends: {"type": "update", "data": {"delta": ..., "user_id": ..., "username": ...}}
        - Server sends: {"type": "user_joined", "data": {"user_id": ..., "username": ...}}
        - Server sends: {"type": "user_left", "data": {"user_id": ..., "username": ...}}
        - Server sends: {"type": "cursor", "data": {"user_id": ..., "cursor": ...}}
        - Server sends: {"type": "presence", "data": {"user_id": ..., "status": ...}}
        """
        # Verify authentication
        username = verify_token_for_websocket(token) if token else None
        if not username:
            await websocket.close(code=1008, reason="Authentication required")
            return

        user_id = username  # Use username as user_id
        joined = False

        try:
            while True:
                # Receive message
                data = await websocket.receive_json()
                msg_type = data.get("type")

                if msg_type == "join":
                    await collab_server.handle_join(
                        websocket,
                        session_id,
                        user_id,
                        username
                    )
                    joined = True

                elif msg_type == "leave":
                    await collab_server.handle_leave(session_id, user_id, username)
                    break

                elif msg_type == "update":
                    delta_data = data.get("data", {}).get("delta")
                    if delta_data:
                        await collab_server.handle_update(
                            session_id,
                            user_id,
                            username,
                            delta_data
                        )

                elif msg_type == "cursor":
                    cursor_data = data.get("data", {})
                    if cursor_data:
                        await collab_server.handle_cursor(
                            session_id,
                            user_id,
                            cursor_data
                        )

                elif msg_type == "presence":
                    status = data.get("data", {}).get("status")
                    if status:
                        await collab_server.handle_presence(
                            session_id,
                            user_id,
                            status
                        )

                elif msg_type == "heartbeat":
                    await collab_server.handle_heartbeat(session_id, user_id)

                else:
                    logger.warning(f"Unknown message type: {msg_type}")

        except WebSocketDisconnect:
            if joined:
                await collab_server.handle_leave(session_id, user_id, username)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if joined:
                await collab_server.handle_leave(session_id, user_id, username)

    return app


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE or not COLLABORATION_AVAILABLE:
        print("Required dependencies not available")
        print("Install: pip install fastapi uvicorn websockets")
    else:
        import uvicorn

        print("="*80)
        print("HoloLoom Collaboration Server")
        print("="*80)
        print("\nStarting server...")
        print("WebSocket endpoint: ws://localhost:8001/ws/collaborate/{session_id}?token={jwt}")
        print("API endpoints: http://localhost:8001/docs\n")

        app = create_collaboration_app()
        uvicorn.run(app, host="0.0.0.0", port=8001)
