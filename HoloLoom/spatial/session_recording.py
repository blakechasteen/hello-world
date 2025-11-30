"""
Session Recording and Playback System for HoloLoom Spatial Computing.

Record and replay XR sessions including:
- User poses and movements
- Gaze data
- Hand gestures
- Voice commands
- Graph state changes
- UI interactions

Useful for training, demos, debugging, and collaboration.

Created: November 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, BinaryIO
from enum import Enum
from datetime import datetime
import json
import gzip
import io


class RecordingState(Enum):
    """State of recording/playback."""
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    PLAYING = "playing"
    SEEKING = "seeking"


class EventType(Enum):
    """Types of recordable events."""
    # User tracking
    HEAD_POSE = "head_pose"
    LEFT_HAND_POSE = "left_hand_pose"
    RIGHT_HAND_POSE = "right_hand_pose"
    BODY_POSE = "body_pose"
    GAZE_RAY = "gaze_ray"

    # Interactions
    GESTURE = "gesture"
    VOICE_COMMAND = "voice_command"
    UI_INTERACTION = "ui_interaction"
    NODE_SELECTION = "node_selection"

    # Graph changes
    NODE_CREATED = "node_created"
    NODE_DELETED = "node_deleted"
    NODE_MOVED = "node_moved"
    EDGE_CREATED = "edge_created"
    EDGE_DELETED = "edge_deleted"
    GRAPH_LAYOUT = "graph_layout"

    # System events
    QUERY_SUBMITTED = "query_submitted"
    RESPONSE_RECEIVED = "response_received"
    ERROR_OCCURRED = "error_occurred"
    MARKER = "marker"  # User-defined markers

    # Session
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"


@dataclass
class RecordedEvent:
    """A single recorded event."""
    event_id: str
    event_type: EventType
    timestamp: float  # Seconds from session start
    data: Dict[str, Any]
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.event_id,
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "userId": self.user_id,
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "RecordedEvent":
        return RecordedEvent(
            event_id=data["id"],
            event_type=EventType(data["type"]),
            timestamp=data["timestamp"],
            data=data["data"],
            user_id=data.get("userId"),
            metadata=data.get("metadata", {})
        )


@dataclass
class RecordingBookmark:
    """A named bookmark in a recording."""
    bookmark_id: str
    name: str
    timestamp: float
    description: Optional[str] = None
    thumbnail_data: Optional[str] = None  # Base64 encoded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.bookmark_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "description": self.description,
            "thumbnail": self.thumbnail_data
        }


@dataclass
class SessionMetadata:
    """Metadata about a recorded session."""
    session_id: str
    title: str
    description: Optional[str] = None
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    duration: float = 0.0
    event_count: int = 0
    user_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    thumbnail_data: Optional[str] = None
    graph_snapshot: Optional[Dict[str, Any]] = None
    version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "title": self.title,
            "description": self.description,
            "createdAt": self.created_at,
            "duration": self.duration,
            "eventCount": self.event_count,
            "userIds": self.user_ids,
            "tags": self.tags,
            "thumbnail": self.thumbnail_data,
            "graphSnapshot": self.graph_snapshot,
            "version": self.version
        }


@dataclass
class Recording:
    """A complete session recording."""
    metadata: SessionMetadata
    events: List[RecordedEvent] = field(default_factory=list)
    bookmarks: List[RecordingBookmark] = field(default_factory=list)

    def add_event(self, event: RecordedEvent):
        """Add an event to the recording."""
        self.events.append(event)
        self.metadata.event_count = len(self.events)
        if event.timestamp > self.metadata.duration:
            self.metadata.duration = event.timestamp

    def add_bookmark(self, bookmark: RecordingBookmark):
        """Add a bookmark."""
        self.bookmarks.append(bookmark)

    def get_events_in_range(self, start: float, end: float) -> List[RecordedEvent]:
        """Get events within a time range."""
        return [e for e in self.events if start <= e.timestamp <= end]

    def get_events_by_type(self, event_type: EventType) -> List[RecordedEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.event_type == event_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "bookmarks": [b.to_dict() for b in self.bookmarks]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Recording":
        metadata = SessionMetadata(
            session_id=data["metadata"]["sessionId"],
            title=data["metadata"]["title"],
            description=data["metadata"].get("description"),
            created_at=data["metadata"].get("createdAt", 0),
            duration=data["metadata"].get("duration", 0),
            event_count=data["metadata"].get("eventCount", 0),
            user_ids=data["metadata"].get("userIds", []),
            tags=data["metadata"].get("tags", []),
            thumbnail_data=data["metadata"].get("thumbnail"),
            graph_snapshot=data["metadata"].get("graphSnapshot"),
            version=data["metadata"].get("version", "1.0")
        )

        events = [RecordedEvent.from_dict(e) for e in data.get("events", [])]
        bookmarks = []

        for b in data.get("bookmarks", []):
            bookmarks.append(RecordingBookmark(
                bookmark_id=b["id"],
                name=b["name"],
                timestamp=b["timestamp"],
                description=b.get("description"),
                thumbnail_data=b.get("thumbnail")
            ))

        recording = Recording(metadata=metadata, bookmarks=bookmarks)
        recording.events = events
        return recording


class SessionRecorder:
    """
    Records XR session events.

    Captures user movements, interactions, and graph changes
    for later playback.
    """

    def __init__(
        self,
        session_id: str,
        title: str = "Untitled Recording",
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        auto_bookmark_interval: float = 0.0  # 0 = disabled
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.auto_bookmark_interval = auto_bookmark_interval

        self.recording = Recording(
            metadata=SessionMetadata(
                session_id=session_id,
                title=title,
                description=description,
                user_ids=[user_id] if user_id else []
            )
        )

        self.state = RecordingState.IDLE
        self.start_time: Optional[float] = None
        self.pause_time: Optional[float] = None
        self.total_pause_duration: float = 0.0

        self._event_counter = 0
        self._bookmark_counter = 0
        self._last_auto_bookmark = 0.0

        # Event filters
        self.enabled_event_types: set = set(EventType)
        self.sample_rates: Dict[EventType, float] = {}  # Events per second
        self._last_sample_times: Dict[EventType, float] = {}

        # Callbacks
        self.on_event_recorded: List[Callable[[RecordedEvent], None]] = []

    def start(self):
        """Start recording."""
        if self.state == RecordingState.IDLE:
            self.start_time = datetime.now().timestamp()
            self.state = RecordingState.RECORDING

            # Record session start
            self.record_event(
                EventType.SESSION_START,
                {"sessionId": self.session_id}
            )

    def pause(self):
        """Pause recording."""
        if self.state == RecordingState.RECORDING:
            self.pause_time = datetime.now().timestamp()
            self.state = RecordingState.PAUSED

    def resume(self):
        """Resume recording after pause."""
        if self.state == RecordingState.PAUSED and self.pause_time:
            pause_duration = datetime.now().timestamp() - self.pause_time
            self.total_pause_duration += pause_duration
            self.pause_time = None
            self.state = RecordingState.RECORDING

    def stop(self) -> Recording:
        """Stop recording and return the recording."""
        if self.state in [RecordingState.RECORDING, RecordingState.PAUSED]:
            self.record_event(EventType.SESSION_END, {})
            self.state = RecordingState.IDLE

        return self.recording

    def get_elapsed_time(self) -> float:
        """Get recording time in seconds."""
        if not self.start_time:
            return 0.0

        if self.state == RecordingState.PAUSED and self.pause_time:
            return self.pause_time - self.start_time - self.total_pause_duration

        return datetime.now().timestamp() - self.start_time - self.total_pause_duration

    def set_sample_rate(self, event_type: EventType, rate: float):
        """Set sample rate for an event type (events per second)."""
        self.sample_rates[event_type] = rate

    def enable_event_type(self, event_type: EventType, enabled: bool = True):
        """Enable or disable recording of an event type."""
        if enabled:
            self.enabled_event_types.add(event_type)
        else:
            self.enabled_event_types.discard(event_type)

    def _should_record_event(self, event_type: EventType) -> bool:
        """Check if event should be recorded based on filters and sample rate."""
        if event_type not in self.enabled_event_types:
            return False

        if event_type in self.sample_rates:
            rate = self.sample_rates[event_type]
            if rate <= 0:
                return False

            min_interval = 1.0 / rate
            last_time = self._last_sample_times.get(event_type, 0)
            current_time = self.get_elapsed_time()

            if current_time - last_time < min_interval:
                return False

            self._last_sample_times[event_type] = current_time

        return True

    def record_event(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[RecordedEvent]:
        """Record an event."""
        if self.state != RecordingState.RECORDING:
            return None

        if not self._should_record_event(event_type):
            return None

        self._event_counter += 1
        event = RecordedEvent(
            event_id=f"evt_{self._event_counter}",
            event_type=event_type,
            timestamp=self.get_elapsed_time(),
            data=data,
            user_id=user_id or self.user_id,
            metadata=metadata or {}
        )

        self.recording.add_event(event)

        # Auto-bookmark
        if self.auto_bookmark_interval > 0:
            if event.timestamp - self._last_auto_bookmark >= self.auto_bookmark_interval:
                self.add_bookmark(f"Auto {int(event.timestamp / 60)}:{int(event.timestamp % 60):02d}")
                self._last_auto_bookmark = event.timestamp

        # Callbacks
        for callback in self.on_event_recorded:
            callback(event)

        return event

    def record_head_pose(self, position: tuple, rotation: tuple, user_id: Optional[str] = None):
        """Record head pose."""
        self.record_event(EventType.HEAD_POSE, {
            "position": {"x": position[0], "y": position[1], "z": position[2]},
            "rotation": {"x": rotation[0], "y": rotation[1], "z": rotation[2], "w": rotation[3] if len(rotation) > 3 else 1.0}
        }, user_id)

    def record_hand_pose(self, hand: str, joints: List[Dict], user_id: Optional[str] = None):
        """Record hand pose (25 joints)."""
        event_type = EventType.LEFT_HAND_POSE if hand == "left" else EventType.RIGHT_HAND_POSE
        self.record_event(event_type, {"joints": joints}, user_id)

    def record_gaze(self, origin: tuple, direction: tuple, hit_node_id: Optional[str] = None):
        """Record gaze ray."""
        self.record_event(EventType.GAZE_RAY, {
            "origin": {"x": origin[0], "y": origin[1], "z": origin[2]},
            "direction": {"x": direction[0], "y": direction[1], "z": direction[2]},
            "hitNodeId": hit_node_id
        })

    def record_gesture(self, gesture_type: str, hand: str, confidence: float):
        """Record a hand gesture."""
        self.record_event(EventType.GESTURE, {
            "gestureType": gesture_type,
            "hand": hand,
            "confidence": confidence
        })

    def record_voice_command(self, command: str, transcript: str, confidence: float):
        """Record a voice command."""
        self.record_event(EventType.VOICE_COMMAND, {
            "command": command,
            "transcript": transcript,
            "confidence": confidence
        })

    def record_ui_interaction(self, element_id: str, interaction_type: str, data: Dict = None):
        """Record UI interaction."""
        self.record_event(EventType.UI_INTERACTION, {
            "elementId": element_id,
            "interactionType": interaction_type,
            "data": data or {}
        })

    def record_node_selection(self, node_id: str, selected: bool):
        """Record node selection."""
        self.record_event(EventType.NODE_SELECTION, {
            "nodeId": node_id,
            "selected": selected
        })

    def record_query(self, query_text: str, query_id: str):
        """Record query submission."""
        self.record_event(EventType.QUERY_SUBMITTED, {
            "queryId": query_id,
            "queryText": query_text
        })

    def record_response(self, query_id: str, response_text: str, confidence: float):
        """Record query response."""
        self.record_event(EventType.RESPONSE_RECEIVED, {
            "queryId": query_id,
            "responseText": response_text,
            "confidence": confidence
        })

    def add_bookmark(self, name: str, description: Optional[str] = None, thumbnail: Optional[str] = None):
        """Add a bookmark at current time."""
        self._bookmark_counter += 1
        bookmark = RecordingBookmark(
            bookmark_id=f"bm_{self._bookmark_counter}",
            name=name,
            timestamp=self.get_elapsed_time(),
            description=description,
            thumbnail_data=thumbnail
        )
        self.recording.add_bookmark(bookmark)
        return bookmark

    def add_marker(self, label: str, data: Optional[Dict] = None):
        """Add a custom marker event."""
        self.record_event(EventType.MARKER, {
            "label": label,
            "data": data or {}
        })

    def set_graph_snapshot(self, graph_data: Dict[str, Any]):
        """Store initial graph state."""
        self.recording.metadata.graph_snapshot = graph_data


class SessionPlayer:
    """
    Plays back recorded XR sessions.

    Supports variable speed playback, seeking, and event callbacks.
    """

    def __init__(self, recording: Recording):
        self.recording = recording
        self.state = RecordingState.IDLE

        self.current_time: float = 0.0
        self.playback_speed: float = 1.0
        self.loop: bool = False

        self._event_index: int = 0
        self._last_update_time: Optional[float] = None

        # Event callbacks by type
        self.event_handlers: Dict[EventType, List[Callable[[RecordedEvent], None]]] = {
            event_type: [] for event_type in EventType
        }

        # General callbacks
        self.on_event: List[Callable[[RecordedEvent], None]] = []
        self.on_playback_complete: List[Callable[[], None]] = []
        self.on_bookmark_reached: List[Callable[[RecordingBookmark], None]] = []

    def play(self):
        """Start playback."""
        if self.state == RecordingState.IDLE:
            self.current_time = 0.0
            self._event_index = 0

        self.state = RecordingState.PLAYING
        self._last_update_time = datetime.now().timestamp()

    def pause(self):
        """Pause playback."""
        if self.state == RecordingState.PLAYING:
            self.state = RecordingState.PAUSED

    def resume(self):
        """Resume playback."""
        if self.state == RecordingState.PAUSED:
            self.state = RecordingState.PLAYING
            self._last_update_time = datetime.now().timestamp()

    def stop(self):
        """Stop playback and reset."""
        self.state = RecordingState.IDLE
        self.current_time = 0.0
        self._event_index = 0

    def seek(self, time: float):
        """Seek to a specific time."""
        self.current_time = max(0.0, min(time, self.recording.metadata.duration))

        # Find event index for this time
        self._event_index = 0
        for i, event in enumerate(self.recording.events):
            if event.timestamp > self.current_time:
                break
            self._event_index = i

    def seek_to_bookmark(self, bookmark_id: str):
        """Seek to a bookmark."""
        for bookmark in self.recording.bookmarks:
            if bookmark.bookmark_id == bookmark_id:
                self.seek(bookmark.timestamp)
                return True
        return False

    def set_speed(self, speed: float):
        """Set playback speed (1.0 = normal)."""
        self.playback_speed = max(0.1, min(10.0, speed))

    def update(self) -> List[RecordedEvent]:
        """Update playback, return events that occurred since last update."""
        if self.state != RecordingState.PLAYING:
            return []

        current_real_time = datetime.now().timestamp()
        if self._last_update_time is None:
            self._last_update_time = current_real_time
            return []

        # Calculate time advance
        delta = (current_real_time - self._last_update_time) * self.playback_speed
        self._last_update_time = current_real_time

        old_time = self.current_time
        self.current_time += delta

        # Check for end of recording
        if self.current_time >= self.recording.metadata.duration:
            if self.loop:
                self.current_time = 0.0
                self._event_index = 0
            else:
                self.state = RecordingState.IDLE
                for callback in self.on_playback_complete:
                    callback()
                return []

        # Get events in this time slice
        triggered_events = []
        while self._event_index < len(self.recording.events):
            event = self.recording.events[self._event_index]
            if event.timestamp <= self.current_time:
                triggered_events.append(event)
                self._fire_event(event)
                self._event_index += 1
            else:
                break

        # Check for bookmarks
        for bookmark in self.recording.bookmarks:
            if old_time < bookmark.timestamp <= self.current_time:
                for callback in self.on_bookmark_reached:
                    callback(bookmark)

        return triggered_events

    def _fire_event(self, event: RecordedEvent):
        """Fire callbacks for an event."""
        # Type-specific handlers
        for handler in self.event_handlers.get(event.event_type, []):
            handler(event)

        # General handlers
        for handler in self.on_event:
            handler(event)

    def register_handler(self, event_type: EventType, handler: Callable[[RecordedEvent], None]):
        """Register a handler for a specific event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    def get_events_at_time(self, time: float, tolerance: float = 0.1) -> List[RecordedEvent]:
        """Get events near a specific time."""
        return [
            e for e in self.recording.events
            if abs(e.timestamp - time) <= tolerance
        ]

    def get_state_at_time(self, time: float) -> Dict[str, Any]:
        """Reconstruct state at a specific time."""
        state = {
            "head_pose": None,
            "left_hand": None,
            "right_hand": None,
            "gaze": None,
            "selected_nodes": set()
        }

        # Replay events up to time
        for event in self.recording.events:
            if event.timestamp > time:
                break

            if event.event_type == EventType.HEAD_POSE:
                state["head_pose"] = event.data
            elif event.event_type == EventType.LEFT_HAND_POSE:
                state["left_hand"] = event.data
            elif event.event_type == EventType.RIGHT_HAND_POSE:
                state["right_hand"] = event.data
            elif event.event_type == EventType.GAZE_RAY:
                state["gaze"] = event.data
            elif event.event_type == EventType.NODE_SELECTION:
                if event.data.get("selected"):
                    state["selected_nodes"].add(event.data["nodeId"])
                else:
                    state["selected_nodes"].discard(event.data["nodeId"])

        state["selected_nodes"] = list(state["selected_nodes"])
        return state

    def get_progress(self) -> float:
        """Get playback progress (0.0 to 1.0)."""
        if self.recording.metadata.duration <= 0:
            return 0.0
        return self.current_time / self.recording.metadata.duration

    def to_state(self) -> Dict[str, Any]:
        """Export player state for UI."""
        return {
            "state": self.state.value,
            "currentTime": self.current_time,
            "duration": self.recording.metadata.duration,
            "progress": self.get_progress(),
            "speed": self.playback_speed,
            "loop": self.loop,
            "eventIndex": self._event_index,
            "totalEvents": len(self.recording.events)
        }


class RecordingStorage:
    """
    Storage utilities for recordings.

    Supports saving/loading to JSON with optional gzip compression.
    """

    @staticmethod
    def save_to_json(recording: Recording, path: str, compress: bool = True):
        """Save recording to JSON file."""
        data = recording.to_dict()
        json_str = json.dumps(data, indent=2 if not compress else None)

        if compress:
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)

    @staticmethod
    def load_from_json(path: str) -> Recording:
        """Load recording from JSON file."""
        try:
            # Try gzip first
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        except gzip.BadGzipFile:
            # Fall back to uncompressed
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        return Recording.from_dict(data)

    @staticmethod
    def save_to_buffer(recording: Recording, compress: bool = True) -> bytes:
        """Save recording to bytes buffer."""
        data = recording.to_dict()
        json_str = json.dumps(data)

        if compress:
            return gzip.compress(json_str.encode('utf-8'))
        return json_str.encode('utf-8')

    @staticmethod
    def load_from_buffer(data: bytes) -> Recording:
        """Load recording from bytes buffer."""
        try:
            json_str = gzip.decompress(data).decode('utf-8')
        except gzip.BadGzipFile:
            json_str = data.decode('utf-8')

        return Recording.from_dict(json.loads(json_str))

    @staticmethod
    def get_recording_info(path: str) -> Optional[SessionMetadata]:
        """Get metadata without loading full recording."""
        try:
            try:
                with gzip.open(path, 'rt', encoding='utf-8') as f:
                    # Only parse first few lines to get metadata
                    partial = ""
                    for line in f:
                        partial += line
                        if '"events"' in partial:
                            break
            except gzip.BadGzipFile:
                with open(path, 'r', encoding='utf-8') as f:
                    partial = ""
                    for line in f:
                        partial += line
                        if '"events"' in partial:
                            break

            # Extract metadata section
            import re
            match = re.search(r'"metadata"\s*:\s*({[^}]+})', partial)
            if match:
                metadata_json = match.group(1)
                metadata_data = json.loads(metadata_json)
                return SessionMetadata(
                    session_id=metadata_data.get("sessionId", ""),
                    title=metadata_data.get("title", ""),
                    description=metadata_data.get("description"),
                    created_at=metadata_data.get("createdAt", 0),
                    duration=metadata_data.get("duration", 0),
                    event_count=metadata_data.get("eventCount", 0)
                )
        except Exception:
            pass

        return None
