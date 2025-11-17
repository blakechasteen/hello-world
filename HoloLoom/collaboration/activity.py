"""
Activity Feed & Version History
================================
Logs all collaboration events with version history and rollback capability.

Philosophy:
Every thread woven leaves a trace. The activity log is the complete
history of the fabric - who wove what, when, and why. This enables
reflection, learning, and time-travel debugging.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import difflib

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of collaboration events."""
    # Session events
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    SESSION_DELETED = "session_deleted"

    # User events
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    USER_ROLE_CHANGED = "user_role_changed"

    # Content events
    MESSAGE_SENT = "message_sent"
    MESSAGE_EDITED = "message_edited"
    MESSAGE_DELETED = "message_deleted"

    # State events
    STATE_UPDATED = "state_updated"
    STATE_CLEARED = "state_cleared"

    # Presence events
    USER_ONLINE = "user_online"
    USER_IDLE = "user_idle"
    USER_OFFLINE = "user_offline"
    CURSOR_MOVED = "cursor_moved"

    # Sync events
    DELTA_APPLIED = "delta_applied"
    STATE_SYNCED = "state_synced"


@dataclass
class ActivityEvent:
    """
    Single activity event.

    Captures:
    - What happened (event type)
    - Who did it (user)
    - When (timestamp)
    - Details (data)
    - State before/after (for rollback)
    """
    event_id: str
    event_type: EventType
    session_id: str
    user_id: str
    username: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "username": self.username,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ActivityEvent':
        """Create from dictionary."""
        return ActivityEvent(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            session_id=data["session_id"],
            user_id=data["user_id"],
            username=data["username"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            data=data.get("data", {}),
            before_state=data.get("before_state"),
            after_state=data.get("after_state"),
            metadata=data.get("metadata", {})
        )

    def get_summary(self) -> str:
        """Get human-readable event summary."""
        summaries = {
            EventType.SESSION_CREATED: f"{self.username} created session",
            EventType.USER_JOINED: f"{self.username} joined",
            EventType.USER_LEFT: f"{self.username} left",
            EventType.MESSAGE_SENT: f"{self.username} sent message",
            EventType.MESSAGE_EDITED: f"{self.username} edited message",
            EventType.MESSAGE_DELETED: f"{self.username} deleted message",
            EventType.STATE_UPDATED: f"{self.username} updated state",
            EventType.USER_ROLE_CHANGED: f"{self.username} role changed to {self.data.get('new_role', 'unknown')}",
        }

        return summaries.get(
            self.event_type,
            f"{self.username} performed {self.event_type.value}"
        )


@dataclass
class Version:
    """
    Version snapshot.

    Captures complete state at a point in time.
    """
    version_id: str
    session_id: str
    timestamp: datetime
    state: Dict[str, Any]
    event_id: str  # Event that created this version
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state,
            "event_id": self.event_id,
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Version':
        """Create from dictionary."""
        return Version(
            version_id=data["version_id"],
            session_id=data["session_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state=data["state"],
            event_id=data["event_id"],
            metadata=data.get("metadata", {})
        )


class ActivityLogger:
    """
    Activity logging and version history manager.

    Features:
    - Event logging
    - Version snapshots
    - Rollback capability
    - Activity filtering and search
    - Diff generation
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize activity logger.

        Args:
            storage_dir: Directory for persistence
        """
        self.events: Dict[str, List[ActivityEvent]] = {}  # session_id -> events
        self.versions: Dict[str, List[Version]] = {}  # session_id -> versions
        self.storage_dir = storage_dir or Path("./data/activity")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"ActivityLogger initialized with storage: {self.storage_dir}")

    def log_event(
        self,
        event_type: EventType,
        session_id: str,
        user_id: str,
        username: str,
        data: Optional[Dict[str, Any]] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None
    ) -> ActivityEvent:
        """
        Log an activity event.

        Args:
            event_type: Type of event
            session_id: Session ID
            user_id: User ID
            username: Username
            data: Event data
            before_state: State before event
            after_state: State after event

        Returns:
            Created event
        """
        import uuid

        event = ActivityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            session_id=session_id,
            user_id=user_id,
            username=username,
            data=data or {},
            before_state=before_state,
            after_state=after_state
        )

        # Store event
        if session_id not in self.events:
            self.events[session_id] = []

        self.events[session_id].append(event)

        logger.info(
            f"Logged {event_type.value} by {username} in session {session_id}"
        )

        return event

    def create_version(
        self,
        session_id: str,
        state: Dict[str, Any],
        event_id: str
    ) -> Version:
        """
        Create version snapshot.

        Args:
            session_id: Session ID
            state: Current state
            event_id: Event that triggered this version

        Returns:
            Created version
        """
        import uuid

        version = Version(
            version_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=datetime.now(timezone.utc),
            state=state.copy(),
            event_id=event_id
        )

        # Store version
        if session_id not in self.versions:
            self.versions[session_id] = []

        self.versions[session_id].append(version)

        logger.info(f"Created version {version.version_id} for session {session_id}")

        return version

    def get_events(
        self,
        session_id: str,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ActivityEvent]:
        """
        Get events for session with optional filtering.

        Args:
            session_id: Session ID
            event_type: Filter by event type
            user_id: Filter by user
            limit: Maximum number of events

        Returns:
            List of events (newest first)
        """
        if session_id not in self.events:
            return []

        events = self.events[session_id]

        # Filter by type
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Filter by user
        if user_id:
            events = [e for e in events if e.user_id == user_id]

        # Sort by timestamp (newest first)
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)

        # Limit
        if limit:
            events = events[:limit]

        return events

    def get_versions(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Version]:
        """
        Get versions for session.

        Args:
            session_id: Session ID
            limit: Maximum number of versions

        Returns:
            List of versions (newest first)
        """
        if session_id not in self.versions:
            return []

        versions = sorted(
            self.versions[session_id],
            key=lambda v: v.timestamp,
            reverse=True
        )

        if limit:
            versions = versions[:limit]

        return versions

    def get_version(self, session_id: str, version_id: str) -> Optional[Version]:
        """Get specific version by ID."""
        if session_id not in self.versions:
            return None

        for version in self.versions[session_id]:
            if version.version_id == version_id:
                return version

        return None

    def rollback_to_version(
        self,
        session_id: str,
        version_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Rollback to specific version.

        Args:
            session_id: Session ID
            version_id: Version to rollback to

        Returns:
            State at that version, or None if not found
        """
        version = self.get_version(session_id, version_id)
        if not version:
            logger.warning(f"Version {version_id} not found")
            return None

        logger.info(
            f"Rolling back session {session_id} to version {version_id} "
            f"from {version.timestamp.isoformat()}"
        )

        return version.state.copy()

    def compute_diff(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute diff between two states.

        Args:
            before: Before state
            after: After state

        Returns:
            Diff information
        """
        # Convert to JSON strings for comparison
        before_str = json.dumps(before, sort_keys=True, indent=2)
        after_str = json.dumps(after, sort_keys=True, indent=2)

        # Compute line-by-line diff
        diff = list(difflib.unified_diff(
            before_str.splitlines(keepends=True),
            after_str.splitlines(keepends=True),
            lineterm='',
            n=3
        ))

        # Analyze changes
        added_keys = set(after.keys()) - set(before.keys())
        removed_keys = set(before.keys()) - set(after.keys())
        modified_keys = set()

        for key in set(before.keys()) & set(after.keys()):
            if before[key] != after[key]:
                modified_keys.add(key)

        return {
            "added_keys": list(added_keys),
            "removed_keys": list(removed_keys),
            "modified_keys": list(modified_keys),
            "diff_lines": diff,
            "total_changes": len(added_keys) + len(removed_keys) + len(modified_keys)
        }

    def get_event_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get formatted event history.

        Args:
            session_id: Session ID
            limit: Maximum events to return

        Returns:
            List of formatted events
        """
        events = self.get_events(session_id, limit=limit)

        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp.isoformat(),
                "summary": e.get_summary(),
                "user": e.username,
                "type": e.event_type.value,
                "data": e.data
            }
            for e in events
        ]

    def search_events(
        self,
        session_id: str,
        query: str
    ) -> List[ActivityEvent]:
        """
        Search events by text query.

        Args:
            session_id: Session ID
            query: Search query

        Returns:
            Matching events
        """
        if session_id not in self.events:
            return []

        query_lower = query.lower()
        matches = []

        for event in self.events[session_id]:
            # Search in username, event type, data
            searchable = f"{event.username} {event.event_type.value} {json.dumps(event.data)}"

            if query_lower in searchable.lower():
                matches.append(event)

        return sorted(matches, key=lambda e: e.timestamp, reverse=True)

    def save_session_activity(self, session_id: str):
        """Persist session activity to disk."""
        session_dir = self.storage_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save events
        events_file = session_dir / "events.json"
        events_data = [e.to_dict() for e in self.events.get(session_id, [])]

        with open(events_file, 'w') as f:
            json.dump(events_data, f, indent=2)

        # Save versions
        versions_file = session_dir / "versions.json"
        versions_data = [v.to_dict() for v in self.versions.get(session_id, [])]

        with open(versions_file, 'w') as f:
            json.dump(versions_data, f, indent=2)

        logger.info(f"Saved activity for session {session_id}")

    def load_session_activity(self, session_id: str):
        """Load session activity from disk."""
        session_dir = self.storage_dir / session_id

        if not session_dir.exists():
            logger.warning(f"No activity found for session {session_id}")
            return

        # Load events
        events_file = session_dir / "events.json"
        if events_file.exists():
            with open(events_file, 'r') as f:
                events_data = json.load(f)

            self.events[session_id] = [
                ActivityEvent.from_dict(e) for e in events_data
            ]

        # Load versions
        versions_file = session_dir / "versions.json"
        if versions_file.exists():
            with open(versions_file, 'r') as f:
                versions_data = json.load(f)

            self.versions[session_id] = [
                Version.from_dict(v) for v in versions_data
            ]

        logger.info(f"Loaded activity for session {session_id}")

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary statistics for session."""
        events = self.events.get(session_id, [])
        versions = self.versions.get(session_id, [])

        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Count events by user
        user_counts = {}
        for event in events:
            user_counts[event.username] = user_counts.get(event.username, 0) + 1

        return {
            "session_id": session_id,
            "total_events": len(events),
            "total_versions": len(versions),
            "event_counts": event_counts,
            "user_counts": user_counts,
            "first_event": events[-1].timestamp.isoformat() if events else None,
            "last_event": events[0].timestamp.isoformat() if events else None
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import uuid

    print("="*80)
    print("Activity Feed & Version History")
    print("="*80 + "\n")

    # Create logger
    with tempfile.TemporaryDirectory() as tmpdir:
        logger_instance = ActivityLogger(storage_dir=Path(tmpdir))
        session_id = "demo-session"

        # Log events
        print("1. Logging events...")
        logger_instance.log_event(
            EventType.SESSION_CREATED,
            session_id,
            "user1",
            "Alice",
            data={"name": "AI Research"}
        )

        logger_instance.log_event(
            EventType.USER_JOINED,
            session_id,
            "user2",
            "Bob"
        )

        state_v1 = {"topic": "Neural Networks", "messages": []}
        logger_instance.log_event(
            EventType.STATE_UPDATED,
            session_id,
            "user1",
            "Alice",
            after_state=state_v1
        )

        print("   ✓ Logged 3 events")

        # Create versions
        print("\n2. Creating versions...")
        event1 = logger_instance.events[session_id][0]
        logger_instance.create_version(session_id, state_v1, event1.event_id)

        state_v2 = {"topic": "Deep Learning", "messages": ["Hello"]}
        event2 = logger_instance.events[session_id][-1]
        logger_instance.create_version(session_id, state_v2, event2.event_id)

        print("   ✓ Created 2 versions")

        # Get history
        print("\n3. Event history:")
        history = logger_instance.get_event_history(session_id)
        for h in history:
            print(f"   • {h['timestamp'][:19]} - {h['summary']}")

        # Compute diff
        print("\n4. Computing diff between versions...")
        diff = logger_instance.compute_diff(state_v1, state_v2)
        print(f"   • Modified keys: {diff['modified_keys']}")
        print(f"   • Total changes: {diff['total_changes']}")

        # Search
        print("\n5. Searching events...")
        results = logger_instance.search_events(session_id, "Alice")
        print(f"   ✓ Found {len(results)} events matching 'Alice'")

        # Summary
        print("\n6. Session summary:")
        summary = logger_instance.get_session_summary(session_id)
        print(f"   • Total events: {summary['total_events']}")
        print(f"   • Total versions: {summary['total_versions']}")
        print(f"   • Event types: {summary['event_counts']}")

        # Save/load
        print("\n7. Persistence:")
        logger_instance.save_session_activity(session_id)
        print("   ✓ Saved to disk")

        logger_instance.events.clear()
        logger_instance.load_session_activity(session_id)
        print(f"   ✓ Loaded {len(logger_instance.events[session_id])} events")

        print("\n✓ Activity logging ready!")
