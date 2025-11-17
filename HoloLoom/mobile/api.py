"""
Mobile API Endpoints
====================
Additional REST API endpoints specifically for mobile clients.

Features:
- User registration and profile management
- File upload/download
- Offline sync (pull/push)
- Push notification registration
- Session management
"""

import logging
import asyncio
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try imports with graceful fallback
try:
    from fastapi import (
        FastAPI, HTTPException, Depends, File, UploadFile,
        Form, Body, Query, status
    )
    from fastapi.responses import JSONResponse, FileResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not installed. Mobile API unavailable.")
    FASTAPI_AVAILABLE = False


@dataclass
class MobileConfig:
    """Configuration for mobile API."""
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    upload_dir: str = "./uploads"
    allowed_file_types: List[str] = field(default_factory=lambda: [
        "image/jpeg", "image/png", "image/gif",
        "audio/mpeg", "audio/wav", "audio/ogg",
        "application/pdf", "text/plain"
    ])
    enable_push: bool = True
    enable_sync: bool = True


class FileManager:
    """Manage file uploads and downloads."""

    def __init__(self, config: MobileConfig):
        self.config = config
        self.upload_dir = Path(config.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # In-memory file metadata (use DB in production)
        self.files: Dict[str, Dict[str, Any]] = {}

        logger.info(f"FileManager initialized (upload_dir={self.upload_dir})")

    async def upload_file(
        self,
        file: UploadFile,
        user_id: str,
        file_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a file.

        Args:
            file: Uploaded file
            user_id: User identifier
            file_type: Optional file type override

        Returns:
            File metadata dictionary
        """
        # Validate file type
        content_type = file.content_type or "application/octet-stream"
        if content_type not in self.config.allowed_file_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {content_type} not allowed"
            )

        # Read file
        contents = await file.read()
        file_size = len(contents)

        # Validate size
        if file_size > self.config.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {self.config.max_file_size} bytes)"
            )

        # Generate unique file ID
        file_id = f"file_{uuid.uuid4().hex[:12]}"

        # Compute hash
        file_hash = hashlib.sha256(contents).hexdigest()

        # Determine file extension
        file_name = file.filename or "upload"
        extension = Path(file_name).suffix or ""

        # Save to disk
        file_path = self.upload_dir / f"{file_id}{extension}"
        with open(file_path, "wb") as f:
            f.write(contents)

        # Store metadata
        metadata = {
            "file_id": file_id,
            "file_name": file_name,
            "file_type": file_type or self._infer_file_type(content_type),
            "content_type": content_type,
            "file_size": file_size,
            "file_hash": file_hash,
            "user_id": user_id,
            "upload_time": datetime.now(timezone.utc).isoformat(),
            "url": f"/files/{file_id}",
            "path": str(file_path)
        }

        self.files[file_id] = metadata
        logger.info(f"Uploaded file: {file_id} ({file_size} bytes)")

        return metadata

    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata."""
        return self.files.get(file_id)

    def delete_file(self, file_id: str, user_id: str) -> bool:
        """Delete a file."""
        if file_id not in self.files:
            return False

        metadata = self.files[file_id]

        # Check ownership
        if metadata["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this file"
            )

        # Delete from disk
        file_path = Path(metadata["path"])
        if file_path.exists():
            file_path.unlink()

        # Remove metadata
        del self.files[file_id]
        logger.info(f"Deleted file: {file_id}")

        return True

    def _infer_file_type(self, content_type: str) -> str:
        """Infer file type category from content type."""
        if content_type.startswith("image/"):
            return "image"
        elif content_type.startswith("audio/"):
            return "audio"
        elif content_type.startswith("video/"):
            return "video"
        elif content_type in ["application/pdf", "text/plain"]:
            return "document"
        else:
            return "other"


class SyncManager:
    """Manage offline data synchronization."""

    def __init__(self):
        # In-memory sync state (use DB in production)
        self.user_sync_state: Dict[str, Dict[str, Any]] = {}
        logger.info("SyncManager initialized")

    async def pull_updates(
        self,
        user_id: str,
        last_sync: Optional[datetime],
        entity_types: List[str]
    ) -> Dict[str, Any]:
        """
        Pull updates from server.

        Args:
            user_id: User identifier
            last_sync: Last sync timestamp
            entity_types: Types of entities to sync

        Returns:
            Sync data dictionary
        """
        sync_data = {
            "messages": [],
            "sessions": [],
            "deleted_ids": {
                "messages": [],
                "sessions": []
            },
            "sync_timestamp": datetime.now(timezone.utc).isoformat()
        }

        # TODO: Query database for updates since last_sync
        # For now, return empty sync data

        logger.info(f"Pull sync for user {user_id} (entities={entity_types})")
        return sync_data

    async def push_updates(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Push updates to server.

        Args:
            user_id: User identifier
            messages: Messages to sync
            sessions: Sessions to sync

        Returns:
            Sync result with conflicts
        """
        conflicts = []

        # TODO: Merge updates into database
        # Handle conflicts (server_wins, client_wins, merged)

        logger.info(f"Push sync for user {user_id} (messages={len(messages)}, sessions={len(sessions)})")

        return {
            "status": "synced",
            "conflicts": conflicts,
            "sync_timestamp": datetime.now(timezone.utc).isoformat()
        }


class PushNotificationManager:
    """Manage push notification registration."""

    def __init__(self):
        # In-memory device registry (use DB in production)
        self.devices: Dict[str, Dict[str, Any]] = {}
        logger.info("PushNotificationManager initialized")

    async def register_device(
        self,
        user_id: str,
        device_token: str,
        platform: str,
        device_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register device for push notifications.

        Args:
            user_id: User identifier
            device_token: FCM token (Android) or APNs token (iOS)
            platform: "ios" or "android"
            device_id: Optional device identifier

        Returns:
            Registration result
        """
        if platform not in ["ios", "android"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Platform must be 'ios' or 'android'"
            )

        # Generate device ID if not provided
        if not device_id:
            device_id = f"{platform}_{uuid.uuid4().hex[:12]}"

        # Store device
        self.devices[device_id] = {
            "device_id": device_id,
            "user_id": user_id,
            "device_token": device_token,
            "platform": platform,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "last_active": datetime.now(timezone.utc).isoformat()
        }

        logger.info(f"Registered {platform} device for user {user_id}: {device_id}")

        return {
            "status": "registered",
            "device_id": device_id
        }

    async def unregister_device(self, device_id: str, user_id: str) -> bool:
        """Unregister device."""
        if device_id not in self.devices:
            return False

        device = self.devices[device_id]
        if device["user_id"] != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to unregister this device"
            )

        del self.devices[device_id]
        logger.info(f"Unregistered device: {device_id}")

        return True

    async def send_notification(
        self,
        user_id: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Send push notification to all user devices.

        Args:
            user_id: User identifier
            title: Notification title
            body: Notification body
            data: Optional data payload
        """
        # Find user devices
        user_devices = [
            d for d in self.devices.values()
            if d["user_id"] == user_id
        ]

        if not user_devices:
            logger.warning(f"No devices registered for user {user_id}")
            return

        # TODO: Send via FCM/APNs
        # For now, just log
        logger.info(f"Sending notification to {len(user_devices)} devices: {title}")


def add_mobile_routes(app: FastAPI, config: MobileConfig = None):
    """
    Add mobile API routes to FastAPI app.

    Args:
        app: FastAPI application
        config: Mobile configuration

    Usage:
        app = FastAPI()
        add_mobile_routes(app)
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not installed")

    config = config or MobileConfig()

    # Initialize managers
    file_manager = FileManager(config)
    sync_manager = SyncManager()
    push_manager = PushNotificationManager()

    # Import auth dependencies
    try:
        from HoloLoom.web.auth import get_current_user, User, user_db, create_access_token
        AUTH_AVAILABLE = True
    except ImportError:
        logger.warning("Auth module unavailable, using mock auth")
        AUTH_AVAILABLE = False

        # Mock dependencies for testing
        @dataclass
        class User:
            username: str
            email: Optional[str] = None

        async def get_current_user() -> User:
            return User(username="test_user", email="test@example.com")

    # ========================================================================
    # User Registration & Profile
    # ========================================================================

    @app.post("/api/auth/register", tags=["Authentication"])
    async def register_user(
        username: str = Body(...),
        password: str = Body(...),
        email: str = Body(...),
        full_name: Optional[str] = Body(None)
    ):
        """Register new user account."""
        if not AUTH_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication not available"
            )

        try:
            user = user_db.create_user(
                username=username,
                password=password,
                email=email,
                full_name=full_name
            )

            return {
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "created_at": user.created_at.isoformat()
            }

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )

    @app.get("/api/user/profile", tags=["User"])
    async def get_user_profile(user: User = Depends(get_current_user)):
        """Get user profile."""
        return {
            "username": user.username,
            "email": user.email,
            "full_name": getattr(user, "full_name", None),
            "avatar_url": None,
            "bio": None,
            "preferences": {}
        }

    @app.put("/api/user/profile", tags=["User"])
    async def update_user_profile(
        user: User = Depends(get_current_user),
        full_name: Optional[str] = Body(None),
        email: Optional[str] = Body(None),
        avatar_url: Optional[str] = Body(None),
        preferences: Optional[Dict[str, Any]] = Body(None)
    ):
        """Update user profile."""
        # TODO: Update user in database
        return {
            "username": user.username,
            "full_name": full_name or getattr(user, "full_name", None),
            "email": email or user.email,
            "avatar_url": avatar_url,
            "preferences": preferences or {}
        }

    @app.get("/api/user/settings", tags=["User"])
    async def get_user_settings(user: User = Depends(get_current_user)):
        """Get user settings."""
        return {
            "theme": "auto",
            "language": "en",
            "notifications_enabled": True,
            "sound_enabled": True,
            "typing_indicators": True
        }

    @app.put("/api/user/settings", tags=["User"])
    async def update_user_settings(
        settings: Dict[str, Any] = Body(...),
        user: User = Depends(get_current_user)
    ):
        """Update user settings."""
        # TODO: Save settings to database
        return settings

    # ========================================================================
    # File Upload/Download
    # ========================================================================

    @app.post("/api/files/upload", tags=["Files"])
    async def upload_file(
        file: UploadFile = File(...),
        file_type: Optional[str] = Form(None),
        user: User = Depends(get_current_user)
    ):
        """Upload a file."""
        metadata = await file_manager.upload_file(file, user.username, file_type)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=metadata
        )

    @app.get("/api/files/{file_id}", tags=["Files"])
    async def get_file(
        file_id: str,
        user: User = Depends(get_current_user)
    ):
        """Download a file."""
        metadata = file_manager.get_file(file_id)

        if not metadata:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )

        file_path = Path(metadata["path"])
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found on disk"
            )

        return FileResponse(
            path=file_path,
            filename=metadata["file_name"],
            media_type=metadata["content_type"]
        )

    @app.delete("/api/files/{file_id}", tags=["Files"])
    async def delete_file(
        file_id: str,
        user: User = Depends(get_current_user)
    ):
        """Delete a file."""
        success = file_manager.delete_file(file_id, user.username)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )

        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)

    # ========================================================================
    # Offline Sync
    # ========================================================================

    @app.post("/api/sync/pull", tags=["Sync"])
    async def sync_pull(
        last_sync: Optional[str] = Body(None),
        entity_types: List[str] = Body(["messages", "sessions"]),
        user: User = Depends(get_current_user)
    ):
        """Pull updates from server."""
        last_sync_dt = None
        if last_sync:
            try:
                last_sync_dt = datetime.fromisoformat(last_sync.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid last_sync timestamp"
                )

        sync_data = await sync_manager.pull_updates(
            user.username,
            last_sync_dt,
            entity_types
        )

        return sync_data

    @app.post("/api/sync/push", tags=["Sync"])
    async def sync_push(
        messages: List[Dict[str, Any]] = Body([]),
        sessions: List[Dict[str, Any]] = Body([]),
        user: User = Depends(get_current_user)
    ):
        """Push updates to server."""
        result = await sync_manager.push_updates(
            user.username,
            messages,
            sessions
        )

        return result

    # ========================================================================
    # Push Notifications
    # ========================================================================

    @app.post("/api/push/register", tags=["Push"])
    async def register_push_device(
        device_token: str = Body(...),
        platform: str = Body(...),
        device_id: Optional[str] = Body(None),
        user: User = Depends(get_current_user)
    ):
        """Register device for push notifications."""
        result = await push_manager.register_device(
            user.username,
            device_token,
            platform,
            device_id
        )

        return result

    @app.post("/api/push/unregister", tags=["Push"])
    async def unregister_push_device(
        device_id: str = Body(...),
        user: User = Depends(get_current_user)
    ):
        """Unregister device from push notifications."""
        success = await push_manager.unregister_device(device_id, user.username)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )

        return {"status": "unregistered"}

    @app.get("/api/push/preferences", tags=["Push"])
    async def get_push_preferences(user: User = Depends(get_current_user)):
        """Get push notification preferences."""
        return {
            "enabled": True,
            "new_messages": True,
            "mentions": True,
            "quiet_hours": {
                "enabled": False,
                "start_time": "22:00",
                "end_time": "08:00"
            }
        }

    @app.put("/api/push/preferences", tags=["Push"])
    async def update_push_preferences(
        preferences: Dict[str, Any] = Body(...),
        user: User = Depends(get_current_user)
    ):
        """Update push notification preferences."""
        # TODO: Save preferences to database
        return preferences

    # ========================================================================
    # Chat Sessions (Mobile-specific)
    # ========================================================================

    @app.get("/api/chat/sessions", tags=["Chat"])
    async def list_chat_sessions(
        limit: int = Query(20, le=100),
        offset: int = Query(0, ge=0),
        user: User = Depends(get_current_user)
    ):
        """List chat sessions."""
        # TODO: Query database for user sessions
        return {
            "sessions": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }

    @app.post("/api/chat/sessions", tags=["Chat"])
    async def create_chat_session(
        title: Optional[str] = Body(None),
        metadata: Optional[Dict[str, Any]] = Body(None),
        user: User = Depends(get_current_user)
    ):
        """Create new chat session."""
        session_id = f"{user.username}_{datetime.now(timezone.utc).timestamp()}"

        session = {
            "session_id": session_id,
            "title": title or "New conversation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_activity": datetime.now(timezone.utc).isoformat(),
            "message_count": 0,
            "metadata": metadata or {}
        }

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=session
        )

    @app.get("/api/chat/sessions/{session_id}", tags=["Chat"])
    async def get_chat_session(
        session_id: str,
        user: User = Depends(get_current_user)
    ):
        """Get chat session details."""
        # TODO: Query database
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    @app.delete("/api/chat/sessions/{session_id}", tags=["Chat"])
    async def delete_chat_session(
        session_id: str,
        user: User = Depends(get_current_user)
    ):
        """Delete chat session."""
        # TODO: Delete from database
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)

    @app.get("/api/chat/sessions/{session_id}/messages", tags=["Chat"])
    async def get_session_messages(
        session_id: str,
        limit: int = Query(50, le=200),
        before: Optional[str] = Query(None),
        user: User = Depends(get_current_user)
    ):
        """Get messages from session."""
        # TODO: Query database
        return {
            "messages": [],
            "has_more": False
        }

    @app.post("/api/chat/sessions/{session_id}/messages", tags=["Chat"])
    async def send_message(
        session_id: str,
        text: str = Body(...),
        metadata: Optional[Dict[str, Any]] = Body(None),
        attachments: List[str] = Body([]),
        user: User = Depends(get_current_user)
    ):
        """Send message (non-streaming)."""
        # TODO: Process message through HoloLoom orchestrator

        # Mock response
        user_message = {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "role": "user",
            "content": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attachments": [],
            "metadata": metadata or {},
            "status": "sent"
        }

        assistant_message = {
            "message_id": f"msg_{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "role": "assistant",
            "content": f"You said: {text}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attachments": [],
            "metadata": {},
            "status": "sent"
        }

        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "trace": {
                "motifs": [],
                "embeddings": {"scales": [96, 192, 384]},
                "tool_selection": {"selected_tool": "echo"},
                "execution_time_ms": 10.0
            }
        }

    logger.info("Mobile API routes added")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed: pip install fastapi uvicorn python-multipart")
    else:
        import uvicorn
        from fastapi import FastAPI

        print("="*80)
        print("HoloLoom Mobile API Server")
        print("="*80)
        print("\nStarting server...")
        print("OpenAPI docs: http://localhost:8000/docs")
        print("OpenAPI JSON: http://localhost:8000/openapi.json\n")

        app = FastAPI(
            title="HoloLoom Mobile API",
            version="1.0.0",
            description="REST API for HoloLoom mobile clients"
        )

        config = MobileConfig()
        add_mobile_routes(app, config)

        uvicorn.run(app, host="0.0.0.0", port=8000)
