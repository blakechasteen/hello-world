"""
Presence Tracking
=================
Tracks user presence, activity, and cursor positions in real-time.

Like watching shuttles move across the loom - each user's presence is
a visible thread showing where they are and what they're doing.
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class PresenceStatus(str, Enum):
    """User presence status."""
    ONLINE = "online"      # Actively using the system
    IDLE = "idle"          # Connected but inactive
    OFFLINE = "offline"    # Disconnected


@dataclass
class CursorPosition:
    """User cursor/selection position."""
    line: int = 0
    column: int = 0
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "line": self.line,
            "column": self.column,
            "selection_start": self.selection_start,
            "selection_end": self.selection_end
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CursorPosition':
        """Create from dictionary."""
        return CursorPosition(
            line=data.get("line", 0),
            column=data.get("column", 0),
            selection_start=data.get("selection_start"),
            selection_end=data.get("selection_end")
        )


@dataclass
class UserPresence:
    """
    User presence information.

    Tracks:
    - Online/idle/offline status
    - Last activity timestamp
    - Cursor position
    - Current action/context
    """
    user_id: str
    username: str
    status: PresenceStatus = PresenceStatus.OFFLINE
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cursor: Optional[CursorPosition] = None
    current_action: Optional[str] = None  # "typing", "editing", "viewing", etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_activity(self, action: Optional[str] = None):
        """Update last activity timestamp."""
        self.last_activity = datetime.now(timezone.utc)
        if action:
            self.current_action = action

    def heartbeat(self):
        """Record heartbeat (periodic ping)."""
        self.last_heartbeat = datetime.now(timezone.utc)
        self.update_activity()

    def set_cursor(self, cursor: CursorPosition):
        """Update cursor position."""
        self.cursor = cursor
        self.update_activity("editing")

    def is_idle(self, idle_threshold_seconds: int = 60) -> bool:
        """Check if user is idle."""
        now = datetime.now(timezone.utc)
        inactive_time = (now - self.last_activity).total_seconds()
        return inactive_time > idle_threshold_seconds

    def is_online(self, timeout_seconds: int = 30) -> bool:
        """Check if user is online (based on heartbeat)."""
        now = datetime.now(timezone.utc)
        heartbeat_age = (now - self.last_heartbeat).total_seconds()
        return heartbeat_age < timeout_seconds

    def update_status(self, idle_threshold: int = 60, timeout: int = 30):
        """Update status based on activity."""
        if not self.is_online(timeout):
            self.status = PresenceStatus.OFFLINE
        elif self.is_idle(idle_threshold):
            self.status = PresenceStatus.IDLE
        else:
            self.status = PresenceStatus.ONLINE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "status": self.status.value,
            "last_activity": self.last_activity.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "cursor": self.cursor.to_dict() if self.cursor else None,
            "current_action": self.current_action,
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'UserPresence':
        """Create from dictionary."""
        cursor = None
        if data.get("cursor"):
            cursor = CursorPosition.from_dict(data["cursor"])

        return UserPresence(
            user_id=data["user_id"],
            username=data["username"],
            status=PresenceStatus(data.get("status", "offline")),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            cursor=cursor,
            current_action=data.get("current_action"),
            metadata=data.get("metadata", {})
        )


class PresenceTracker:
    """
    Tracks user presence across sessions.

    Features:
    - Real-time presence updates
    - Automatic idle detection
    - Heartbeat monitoring
    - Cursor tracking
    - Presence broadcasting
    """

    def __init__(
        self,
        idle_threshold_seconds: int = 60,
        timeout_seconds: int = 30,
        heartbeat_interval: int = 10
    ):
        """
        Initialize presence tracker.

        Args:
            idle_threshold_seconds: Time before user marked as idle
            timeout_seconds: Time before user marked as offline
            heartbeat_interval: Expected heartbeat interval
        """
        self.presences: Dict[str, Dict[str, UserPresence]] = {}  # session_id -> user_id -> presence
        self.idle_threshold = idle_threshold_seconds
        self.timeout = timeout_seconds
        self.heartbeat_interval = heartbeat_interval
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.info(
            f"PresenceTracker initialized "
            f"(idle={idle_threshold_seconds}s, timeout={timeout_seconds}s)"
        )

    def get_session_presences(self, session_id: str) -> Dict[str, UserPresence]:
        """Get all presences for a session."""
        return self.presences.get(session_id, {})

    def get_user_presence(
        self,
        session_id: str,
        user_id: str
    ) -> Optional[UserPresence]:
        """Get presence for specific user in session."""
        return self.presences.get(session_id, {}).get(user_id)

    def add_user(
        self,
        session_id: str,
        user_id: str,
        username: str
    ) -> UserPresence:
        """Add user to presence tracking."""
        if session_id not in self.presences:
            self.presences[session_id] = {}

        if user_id in self.presences[session_id]:
            presence = self.presences[session_id][user_id]
            presence.status = PresenceStatus.ONLINE
            presence.heartbeat()
            return presence

        presence = UserPresence(
            user_id=user_id,
            username=username,
            status=PresenceStatus.ONLINE
        )

        self.presences[session_id][user_id] = presence

        logger.info(f"User {username} ({user_id}) added to session {session_id}")
        return presence

    def remove_user(self, session_id: str, user_id: str) -> bool:
        """Remove user from presence tracking."""
        if session_id not in self.presences:
            return False

        if user_id not in self.presences[session_id]:
            return False

        username = self.presences[session_id][user_id].username
        del self.presences[session_id][user_id]

        # Clean up empty sessions
        if not self.presences[session_id]:
            del self.presences[session_id]

        logger.info(f"User {username} ({user_id}) removed from session {session_id}")
        return True

    def update_heartbeat(self, session_id: str, user_id: str):
        """Update user heartbeat."""
        presence = self.get_user_presence(session_id, user_id)
        if presence:
            presence.heartbeat()

    def update_cursor(
        self,
        session_id: str,
        user_id: str,
        cursor: CursorPosition
    ):
        """Update user cursor position."""
        presence = self.get_user_presence(session_id, user_id)
        if presence:
            presence.set_cursor(cursor)

    def update_action(
        self,
        session_id: str,
        user_id: str,
        action: str
    ):
        """Update user's current action."""
        presence = self.get_user_presence(session_id, user_id)
        if presence:
            presence.update_activity(action)

    def get_online_users(self, session_id: str) -> List[UserPresence]:
        """Get all online users in session."""
        presences = self.get_session_presences(session_id)
        return [
            p for p in presences.values()
            if p.status == PresenceStatus.ONLINE
        ]

    def get_active_users(self, session_id: str) -> List[UserPresence]:
        """Get all active (online + idle) users in session."""
        presences = self.get_session_presences(session_id)
        return [
            p for p in presences.values()
            if p.status in [PresenceStatus.ONLINE, PresenceStatus.IDLE]
        ]

    def update_all_statuses(self, session_id: Optional[str] = None):
        """Update status for all users based on activity."""
        sessions = [session_id] if session_id else list(self.presences.keys())

        for sid in sessions:
            if sid not in self.presences:
                continue

            for presence in self.presences[sid].values():
                presence.update_status(self.idle_threshold, self.timeout)

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get presence summary for session."""
        presences = self.get_session_presences(session_id)

        online = sum(1 for p in presences.values() if p.status == PresenceStatus.ONLINE)
        idle = sum(1 for p in presences.values() if p.status == PresenceStatus.IDLE)
        offline = sum(1 for p in presences.values() if p.status == PresenceStatus.OFFLINE)

        return {
            "session_id": session_id,
            "total_users": len(presences),
            "online": online,
            "idle": idle,
            "offline": offline,
            "users": [p.to_dict() for p in presences.values()]
        }

    async def start_heartbeat_monitor(self, check_interval: int = 5):
        """
        Start background task to monitor heartbeats.

        Args:
            check_interval: How often to check (seconds)
        """
        logger.info(f"Starting heartbeat monitor (interval={check_interval}s)")

        while True:
            try:
                await asyncio.sleep(check_interval)

                # Update all statuses
                self.update_all_statuses()

                # Log status changes
                for session_id in self.presences:
                    summary = self.get_session_summary(session_id)
                    if summary["total_users"] > 0:
                        logger.debug(
                            f"Session {session_id}: "
                            f"{summary['online']} online, "
                            f"{summary['idle']} idle, "
                            f"{summary['offline']} offline"
                        )

            except asyncio.CancelledError:
                logger.info("Heartbeat monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")

    def start_monitoring(self, check_interval: int = 5):
        """Start background heartbeat monitoring."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(
                self.start_heartbeat_monitor(check_interval)
            )
            logger.info("Presence monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("Presence monitoring stopped")

    def cleanup_offline_users(self, session_id: Optional[str] = None):
        """Remove offline users from tracking."""
        sessions = [session_id] if session_id else list(self.presences.keys())
        removed = 0

        for sid in sessions:
            if sid not in self.presences:
                continue

            offline = [
                uid for uid, p in self.presences[sid].items()
                if p.status == PresenceStatus.OFFLINE
            ]

            for uid in offline:
                self.remove_user(sid, uid)
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} offline users")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import time

    async def demo():
        print("="*80)
        print("Presence Tracking")
        print("="*80 + "\n")

        # Create tracker
        tracker = PresenceTracker(idle_threshold_seconds=5, timeout_seconds=10)

        # Add users
        print("1. Adding users...")
        session_id = "demo-session"
        tracker.add_user(session_id, "user1", "Alice")
        tracker.add_user(session_id, "user2", "Bob")
        tracker.add_user(session_id, "user3", "Charlie")
        print(f"   ✓ {len(tracker.get_session_presences(session_id))} users")

        # Update cursors
        print("\n2. Updating cursors...")
        tracker.update_cursor(session_id, "user1", CursorPosition(line=10, column=5))
        tracker.update_cursor(session_id, "user2", CursorPosition(line=20, column=15))
        print("   ✓ Cursors updated")

        # Heartbeat
        print("\n3. Sending heartbeats...")
        for _ in range(3):
            tracker.update_heartbeat(session_id, "user1")
            tracker.update_heartbeat(session_id, "user2")
            # user3 doesn't send heartbeat
            await asyncio.sleep(1)
        print("   ✓ Heartbeats sent")

        # Check status
        print("\n4. Checking status (after 5s idle)...")
        await asyncio.sleep(5)
        tracker.update_all_statuses(session_id)

        for user_id in ["user1", "user2", "user3"]:
            presence = tracker.get_user_presence(session_id, user_id)
            if presence:
                print(f"   • {presence.username}: {presence.status.value}")

        # Summary
        print("\n5. Session summary:")
        summary = tracker.get_session_summary(session_id)
        print(f"   • Total: {summary['total_users']}")
        print(f"   • Online: {summary['online']}")
        print(f"   • Idle: {summary['idle']}")
        print(f"   • Offline: {summary['offline']}")

        # Cleanup
        print("\n6. Cleaning up offline users...")
        tracker.cleanup_offline_users(session_id)
        remaining = len(tracker.get_session_presences(session_id))
        print(f"   ✓ {remaining} users remaining")

        print("\n✓ Presence tracking ready!")

    asyncio.run(demo())
