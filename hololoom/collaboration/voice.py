"""
WebRTC Voice/Video Communication System.

Provides real-time voice and video chat for collaborative sessions.

Phase 3: Multi-User Collaboration - Voice/video infrastructure.

Created: November 2025
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Set
import asyncio
import logging

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Types of media streams."""
    AUDIO = "audio"
    VIDEO = "video"
    SCREEN = "screen"


class StreamQuality(Enum):
    """Stream quality levels."""
    LOW = "low"        # 720p15 or audio-only
    MEDIUM = "medium"  # 720p30
    HIGH = "high"      # 1080p30
    HD = "hd"          # 1080p60


class ConnectionState(Enum):
    """WebRTC connection states."""
    NEW = "new"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    FAILED = "failed"
    CLOSED = "closed"


class SignalingType(Enum):
    """WebRTC signaling message types."""
    OFFER = "offer"
    ANSWER = "answer"
    ICE_CANDIDATE = "ice_candidate"
    MEDIA_STATE = "media_state"
    QUALITY_CHANGE = "quality_change"


@dataclass
class MediaTrack:
    """A single media track (audio or video)."""
    track_id: str
    media_type: MediaType
    user_id: str
    enabled: bool = True
    muted: bool = False
    quality: StreamQuality = StreamQuality.MEDIUM

    # Track metadata
    label: Optional[str] = None
    device_id: Optional[str] = None
    codec: Optional[str] = None
    bitrate: Optional[int] = None

    # Stats
    packets_sent: int = 0
    packets_lost: int = 0
    bytes_sent: int = 0
    jitter: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "media_type": self.media_type.value,
            "user_id": self.user_id,
            "enabled": self.enabled,
            "muted": self.muted,
            "quality": self.quality.value,
            "label": self.label,
            "device_id": self.device_id,
            "codec": self.codec,
            "bitrate": self.bitrate,
            "stats": {
                "packets_sent": self.packets_sent,
                "packets_lost": self.packets_lost,
                "bytes_sent": self.bytes_sent,
                "jitter": self.jitter,
                "latency_ms": self.latency_ms
            }
        }


@dataclass
class PeerConnection:
    """A WebRTC peer connection to another participant."""
    connection_id: str
    local_user_id: str
    remote_user_id: str
    state: ConnectionState = ConnectionState.NEW
    created_at: datetime = field(default_factory=datetime.now)
    connected_at: Optional[datetime] = None

    # Tracks
    local_tracks: Dict[str, MediaTrack] = field(default_factory=dict)
    remote_tracks: Dict[str, MediaTrack] = field(default_factory=dict)

    # ICE state
    ice_connection_state: str = "new"
    ice_gathering_state: str = "new"

    # SDP state
    local_description: Optional[str] = None
    remote_description: Optional[str] = None

    # Stats
    quality_score: float = 1.0  # 0-1, 1 = excellent
    reconnect_count: int = 0

    def add_local_track(self, track: MediaTrack):
        """Add a local media track."""
        self.local_tracks[track.track_id] = track

    def add_remote_track(self, track: MediaTrack):
        """Add a remote media track."""
        self.remote_tracks[track.track_id] = track

    def remove_track(self, track_id: str):
        """Remove a track."""
        self.local_tracks.pop(track_id, None)
        self.remote_tracks.pop(track_id, None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "local_user_id": self.local_user_id,
            "remote_user_id": self.remote_user_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "local_tracks": {k: v.to_dict() for k, v in self.local_tracks.items()},
            "remote_tracks": {k: v.to_dict() for k, v in self.remote_tracks.items()},
            "ice_connection_state": self.ice_connection_state,
            "ice_gathering_state": self.ice_gathering_state,
            "quality_score": self.quality_score,
            "reconnect_count": self.reconnect_count
        }


@dataclass
class SignalingMessage:
    """WebRTC signaling message."""
    message_id: str
    message_type: SignalingType
    from_user: str
    to_user: str
    connection_id: str
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "from_user": self.from_user,
            "to_user": self.to_user,
            "connection_id": self.connection_id,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class VoiceRoomSettings:
    """Settings for a voice room."""
    max_participants: int = 25
    allow_video: bool = True
    allow_screen_share: bool = True
    default_muted: bool = False
    default_video_off: bool = True
    require_push_to_talk: bool = False
    auto_gain_control: bool = True
    noise_suppression: bool = True
    echo_cancellation: bool = True
    spatial_audio: bool = False
    max_video_quality: StreamQuality = StreamQuality.HIGH
    record_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_participants": self.max_participants,
            "allow_video": self.allow_video,
            "allow_screen_share": self.allow_screen_share,
            "default_muted": self.default_muted,
            "default_video_off": self.default_video_off,
            "require_push_to_talk": self.require_push_to_talk,
            "auto_gain_control": self.auto_gain_control,
            "noise_suppression": self.noise_suppression,
            "echo_cancellation": self.echo_cancellation,
            "spatial_audio": self.spatial_audio,
            "max_video_quality": self.max_video_quality.value,
            "record_enabled": self.record_enabled
        }


@dataclass
class VoiceRoomParticipant:
    """A participant in a voice room."""
    user_id: str
    display_name: str
    joined_at: datetime = field(default_factory=datetime.now)

    # Media state
    is_muted: bool = False
    is_video_on: bool = False
    is_screen_sharing: bool = False
    is_speaking: bool = False
    speaking_volume: float = 0.0  # 0-1 for volume indicator

    # Connection quality
    connection_quality: float = 1.0  # 0-1, 1 = excellent
    latency_ms: float = 0.0

    # Active tracks
    audio_track_id: Optional[str] = None
    video_track_id: Optional[str] = None
    screen_track_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "display_name": self.display_name,
            "joined_at": self.joined_at.isoformat(),
            "is_muted": self.is_muted,
            "is_video_on": self.is_video_on,
            "is_screen_sharing": self.is_screen_sharing,
            "is_speaking": self.is_speaking,
            "speaking_volume": self.speaking_volume,
            "connection_quality": self.connection_quality,
            "latency_ms": self.latency_ms,
            "audio_track_id": self.audio_track_id,
            "video_track_id": self.video_track_id,
            "screen_track_id": self.screen_track_id
        }


class VoiceRoom:
    """
    A voice/video room for real-time communication.

    Manages participants, connections, and media states.
    """

    def __init__(
        self,
        room_id: str,
        session_id: str,
        settings: Optional[VoiceRoomSettings] = None
    ):
        self.room_id = room_id
        self.session_id = session_id
        self.settings = settings or VoiceRoomSettings()
        self.created_at = datetime.now()

        self.participants: Dict[str, VoiceRoomParticipant] = {}
        self.connections: Dict[str, PeerConnection] = {}
        self.pending_messages: List[SignalingMessage] = []

        self._event_handlers: Dict[str, List[Callable]] = {}
        self._speaking_detector_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the voice room."""
        self._running = True
        self._speaking_detector_task = asyncio.create_task(self._speaking_detection_loop())
        logger.info(f"Voice room {self.room_id} started")

    async def stop(self):
        """Stop the voice room."""
        self._running = False
        if self._speaking_detector_task:
            self._speaking_detector_task.cancel()
            try:
                await self._speaking_detector_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        for conn in list(self.connections.values()):
            await self._close_connection(conn.connection_id)

        logger.info(f"Voice room {self.room_id} stopped")

    async def _speaking_detection_loop(self):
        """Periodically update speaking indicators (would integrate with actual audio levels)."""
        while self._running:
            await asyncio.sleep(0.1)  # 100ms updates

            # In real implementation, this would check audio levels
            # Here we just reset speaking state after a timeout
            for participant in self.participants.values():
                if participant.is_speaking and participant.speaking_volume < 0.1:
                    participant.is_speaking = False
                    self._emit_event("speaking_stopped", {"user_id": participant.user_id})

    def add_participant(
        self,
        user_id: str,
        display_name: str
    ) -> Optional[VoiceRoomParticipant]:
        """Add a participant to the voice room."""
        if len(self.participants) >= self.settings.max_participants:
            logger.warning(f"Voice room {self.room_id} is full")
            return None

        if user_id in self.participants:
            return self.participants[user_id]

        participant = VoiceRoomParticipant(
            user_id=user_id,
            display_name=display_name,
            is_muted=self.settings.default_muted,
            is_video_on=not self.settings.default_video_off
        )

        self.participants[user_id] = participant
        self._emit_event("participant_joined", {"participant": participant.to_dict()})

        return participant

    def remove_participant(self, user_id: str) -> bool:
        """Remove a participant from the voice room."""
        if user_id not in self.participants:
            return False

        participant = self.participants.pop(user_id)

        # Close connections involving this user
        for conn_id in list(self.connections.keys()):
            conn = self.connections.get(conn_id)
            if conn and (conn.local_user_id == user_id or conn.remote_user_id == user_id):
                asyncio.create_task(self._close_connection(conn_id))

        self._emit_event("participant_left", {"user_id": user_id})
        return True

    async def toggle_mute(self, user_id: str, muted: Optional[bool] = None) -> bool:
        """Toggle or set mute state."""
        participant = self.participants.get(user_id)
        if not participant:
            return False

        if muted is None:
            participant.is_muted = not participant.is_muted
        else:
            participant.is_muted = muted

        self._emit_event("mute_changed", {
            "user_id": user_id,
            "is_muted": participant.is_muted
        })
        return True

    async def toggle_video(self, user_id: str, video_on: Optional[bool] = None) -> bool:
        """Toggle or set video state."""
        if not self.settings.allow_video:
            return False

        participant = self.participants.get(user_id)
        if not participant:
            return False

        if video_on is None:
            participant.is_video_on = not participant.is_video_on
        else:
            participant.is_video_on = video_on

        self._emit_event("video_changed", {
            "user_id": user_id,
            "is_video_on": participant.is_video_on
        })
        return True

    async def start_screen_share(self, user_id: str) -> bool:
        """Start screen sharing."""
        if not self.settings.allow_screen_share:
            return False

        participant = self.participants.get(user_id)
        if not participant:
            return False

        # Only one screen share at a time
        current_sharer = next(
            (p for p in self.participants.values() if p.is_screen_sharing),
            None
        )
        if current_sharer and current_sharer.user_id != user_id:
            return False

        participant.is_screen_sharing = True
        self._emit_event("screen_share_started", {"user_id": user_id})
        return True

    async def stop_screen_share(self, user_id: str) -> bool:
        """Stop screen sharing."""
        participant = self.participants.get(user_id)
        if not participant:
            return False

        participant.is_screen_sharing = False
        self._emit_event("screen_share_stopped", {"user_id": user_id})
        return True

    def update_speaking(self, user_id: str, volume: float):
        """Update speaking indicator (called by audio level detector)."""
        participant = self.participants.get(user_id)
        if not participant:
            return

        was_speaking = participant.is_speaking
        participant.speaking_volume = volume
        participant.is_speaking = volume > 0.1  # Threshold

        if participant.is_speaking and not was_speaking:
            self._emit_event("speaking_started", {"user_id": user_id})

    async def create_connection(
        self,
        local_user_id: str,
        remote_user_id: str
    ) -> Optional[PeerConnection]:
        """Create a peer connection between two participants."""
        from uuid import uuid4

        if local_user_id not in self.participants or remote_user_id not in self.participants:
            return None

        connection_id = f"conn_{uuid4().hex[:12]}"

        connection = PeerConnection(
            connection_id=connection_id,
            local_user_id=local_user_id,
            remote_user_id=remote_user_id
        )

        self.connections[connection_id] = connection
        self._emit_event("connection_created", {
            "connection_id": connection_id,
            "local_user_id": local_user_id,
            "remote_user_id": remote_user_id
        })

        return connection

    async def _close_connection(self, connection_id: str):
        """Close a peer connection."""
        connection = self.connections.pop(connection_id, None)
        if connection:
            connection.state = ConnectionState.CLOSED
            self._emit_event("connection_closed", {"connection_id": connection_id})

    def handle_signaling(self, message: SignalingMessage):
        """Handle incoming signaling message."""
        connection = self.connections.get(message.connection_id)

        if message.message_type == SignalingType.OFFER:
            if connection:
                connection.remote_description = message.payload.get("sdp")
                connection.state = ConnectionState.CONNECTING

        elif message.message_type == SignalingType.ANSWER:
            if connection:
                connection.remote_description = message.payload.get("sdp")
                connection.state = ConnectionState.CONNECTED
                connection.connected_at = datetime.now()

        elif message.message_type == SignalingType.ICE_CANDIDATE:
            # Forward ICE candidate to peer
            pass

        elif message.message_type == SignalingType.MEDIA_STATE:
            user_id = message.from_user
            if user_id in self.participants:
                participant = self.participants[user_id]
                if "is_muted" in message.payload:
                    participant.is_muted = message.payload["is_muted"]
                if "is_video_on" in message.payload:
                    participant.is_video_on = message.payload["is_video_on"]

        self._emit_event("signaling_received", {"message": message.to_dict()})

    def send_signaling(
        self,
        message_type: SignalingType,
        from_user: str,
        to_user: str,
        connection_id: str,
        payload: Dict[str, Any]
    ) -> SignalingMessage:
        """Create and queue a signaling message."""
        from uuid import uuid4

        message = SignalingMessage(
            message_id=f"sig_{uuid4().hex[:8]}",
            message_type=message_type,
            from_user=from_user,
            to_user=to_user,
            connection_id=connection_id,
            payload=payload
        )

        self.pending_messages.append(message)
        self._emit_event("signaling_sent", {"message": message.to_dict()})

        return message

    def get_pending_messages(self, for_user: str) -> List[SignalingMessage]:
        """Get pending signaling messages for a user."""
        messages = [m for m in self.pending_messages if m.to_user == for_user]
        self.pending_messages = [m for m in self.pending_messages if m.to_user != for_user]
        return messages

    def update_connection_quality(self, connection_id: str, stats: Dict[str, Any]):
        """Update connection quality from WebRTC stats."""
        connection = self.connections.get(connection_id)
        if not connection:
            return

        # Calculate quality score from stats
        packet_loss = stats.get("packet_loss", 0)
        jitter = stats.get("jitter", 0)
        latency = stats.get("latency", 0)

        # Simple quality calculation
        quality = 1.0
        quality -= min(packet_loss * 0.1, 0.3)  # Packet loss penalty
        quality -= min(jitter * 0.01, 0.2)      # Jitter penalty
        quality -= min(latency / 500, 0.3)      # Latency penalty (500ms = -0.3)

        connection.quality_score = max(0.0, quality)

        # Update participant quality
        participant = self.participants.get(connection.remote_user_id)
        if participant:
            participant.connection_quality = connection.quality_score
            participant.latency_ms = latency

    def get_active_speakers(self, limit: int = 4) -> List[VoiceRoomParticipant]:
        """Get currently speaking participants (for spotlight view)."""
        speaking = [p for p in self.participants.values() if p.is_speaking]
        speaking.sort(key=lambda p: p.speaking_volume, reverse=True)
        return speaking[:limit]

    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    def _emit_event(self, event: str, data: Dict[str, Any]):
        """Emit event to handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(event, data)
            except Exception as e:
                logger.error(f"Voice room event handler error: {e}")

    def to_state(self) -> Dict[str, Any]:
        """Export voice room state."""
        return {
            "room_id": self.room_id,
            "session_id": self.session_id,
            "settings": self.settings.to_dict(),
            "created_at": self.created_at.isoformat(),
            "participants": {k: v.to_dict() for k, v in self.participants.items()},
            "connections": {k: v.to_dict() for k, v in self.connections.items()},
            "pending_messages": len(self.pending_messages),
            "active_speakers": [p.user_id for p in self.get_active_speakers()]
        }


class VoiceManager:
    """
    Manages voice rooms across collaboration sessions.
    """

    def __init__(self):
        self.rooms: Dict[str, VoiceRoom] = {}
        self.user_rooms: Dict[str, str] = {}  # user_id -> room_id

    async def create_room(
        self,
        session_id: str,
        settings: Optional[VoiceRoomSettings] = None
    ) -> VoiceRoom:
        """Create a voice room for a session."""
        from uuid import uuid4

        room_id = f"voice_{uuid4().hex[:8]}"
        room = VoiceRoom(
            room_id=room_id,
            session_id=session_id,
            settings=settings
        )

        await room.start()
        self.rooms[room_id] = room

        logger.info(f"Created voice room {room_id} for session {session_id}")
        return room

    async def get_room(self, room_id: str) -> Optional[VoiceRoom]:
        """Get a voice room by ID."""
        return self.rooms.get(room_id)

    async def get_room_for_session(self, session_id: str) -> Optional[VoiceRoom]:
        """Get voice room for a session."""
        for room in self.rooms.values():
            if room.session_id == session_id:
                return room
        return None

    async def join_room(
        self,
        room_id: str,
        user_id: str,
        display_name: str
    ) -> Optional[VoiceRoomParticipant]:
        """Join a voice room."""
        room = self.rooms.get(room_id)
        if not room:
            return None

        # Leave current room if in one
        current_room_id = self.user_rooms.get(user_id)
        if current_room_id and current_room_id != room_id:
            await self.leave_room(current_room_id, user_id)

        participant = room.add_participant(user_id, display_name)
        if participant:
            self.user_rooms[user_id] = room_id

        return participant

    async def leave_room(self, room_id: str, user_id: str) -> bool:
        """Leave a voice room."""
        room = self.rooms.get(room_id)
        if not room:
            return False

        result = room.remove_participant(user_id)
        if result:
            self.user_rooms.pop(user_id, None)

        return result

    async def close_room(self, room_id: str) -> bool:
        """Close a voice room."""
        room = self.rooms.pop(room_id, None)
        if not room:
            return False

        await room.stop()

        # Clean up user mappings
        for user_id, rid in list(self.user_rooms.items()):
            if rid == room_id:
                del self.user_rooms[user_id]

        return True

    def to_state(self) -> Dict[str, Any]:
        """Export manager state."""
        return {
            "total_rooms": len(self.rooms),
            "total_users_in_voice": len(self.user_rooms),
            "rooms": {k: v.to_state() for k, v in self.rooms.items()}
        }


# Factory functions
def create_voice_room(
    session_id: str,
    settings: Optional[VoiceRoomSettings] = None
) -> VoiceRoom:
    """Create a voice room."""
    from uuid import uuid4
    room_id = f"voice_{uuid4().hex[:8]}"
    return VoiceRoom(room_id=room_id, session_id=session_id, settings=settings)


def create_voice_manager() -> VoiceManager:
    """Create a voice manager."""
    return VoiceManager()
