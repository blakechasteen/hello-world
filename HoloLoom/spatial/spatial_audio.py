"""
HoloLoom Phase 4+: Spatial Audio System
November 2025

3D audio for immersive knowledge exploration.
Audio cues help users navigate and understand
spatial relationships between concepts.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import json
import math


class AudioType(Enum):
    """Types of spatial audio."""
    AMBIENT = "ambient"          # Background atmosphere
    NOTIFICATION = "notification"  # Alerts and pings
    INTERACTION = "interaction"  # UI feedback
    NAVIGATION = "navigation"    # Movement cues
    VOICE = "voice"              # Speech/narration
    MUSIC = "music"              # Background music


class AudioState(Enum):
    """State of an audio source."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    FADING_IN = "fading_in"
    FADING_OUT = "fading_out"


@dataclass
class Vector3:
    """3D position/direction."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def distance_to(self, other: 'Vector3') -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def normalize(self) -> 'Vector3':
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if length < 0.0001:
            return Vector3(0, 0, 1)
        return Vector3(self.x/length, self.y/length, self.z/length)

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class AudioSource:
    """A 3D audio source in space."""
    source_id: str
    audio_type: AudioType
    position: Vector3 = field(default_factory=Vector3)

    # Audio file/stream
    audio_url: str = ""
    audio_buffer_id: Optional[str] = None  # For pre-loaded buffers

    # Playback state
    state: AudioState = AudioState.STOPPED
    volume: float = 1.0                    # 0.0 - 1.0
    pitch: float = 1.0                     # Playback rate
    loop: bool = False
    current_time: float = 0.0              # Seconds

    # Spatialization
    is_spatial: bool = True
    min_distance: float = 1.0              # Full volume within this distance
    max_distance: float = 20.0             # Inaudible beyond this
    rolloff_factor: float = 1.0            # How quickly volume drops
    cone_inner_angle: float = 360.0        # Full volume cone
    cone_outer_angle: float = 360.0        # Zero volume outside
    cone_outer_gain: float = 0.0           # Volume outside cone

    # Direction (for directional audio)
    direction: Optional[Vector3] = None

    # Linked node (for knowledge graph)
    linked_node_id: Optional[str] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "source_id": self.source_id,
            "audio_type": self.audio_type.value,
            "position": self.position.to_dict(),
            "audio_url": self.audio_url,
            "state": self.state.value,
            "volume": self.volume,
            "pitch": self.pitch,
            "loop": self.loop,
            "is_spatial": self.is_spatial,
            "min_distance": self.min_distance,
            "max_distance": self.max_distance,
            "rolloff_factor": self.rolloff_factor,
            "linked_node_id": self.linked_node_id,
            "tags": self.tags
        }


@dataclass
class AudioListener:
    """The user's audio listener (ears) in 3D space."""
    position: Vector3 = field(default_factory=Vector3)
    forward: Vector3 = field(default_factory=lambda: Vector3(0, 0, -1))
    up: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))

    # Master volume
    master_volume: float = 1.0

    # Per-type volume
    type_volumes: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "position": self.position.to_dict(),
            "forward": self.forward.to_dict(),
            "up": self.up.to_dict(),
            "master_volume": self.master_volume,
            "type_volumes": self.type_volumes
        }


@dataclass
class SoundEffect:
    """A predefined sound effect."""
    effect_id: str
    name: str
    audio_url: str
    audio_type: AudioType
    default_volume: float = 1.0
    is_spatial: bool = True
    duration_ms: float = 0.0  # 0 = unknown/streaming


class SpatialAudioManager:
    """
    Manages 3D spatial audio for XR knowledge exploration.

    Features:
    - 3D positioned audio sources
    - Distance-based attenuation
    - Directional audio (cones)
    - Audio linked to knowledge nodes
    - Sound effect library
    - Ambient soundscapes
    """

    # Default sound effects library
    DEFAULT_EFFECTS = {
        # Navigation
        "teleport": SoundEffect("teleport", "Teleport", "/audio/teleport.mp3", AudioType.NAVIGATION, 0.7),
        "step": SoundEffect("step", "Footstep", "/audio/step.mp3", AudioType.NAVIGATION, 0.3),
        "whoosh": SoundEffect("whoosh", "Movement Whoosh", "/audio/whoosh.mp3", AudioType.NAVIGATION, 0.5),

        # Interaction
        "select": SoundEffect("select", "Select", "/audio/select.mp3", AudioType.INTERACTION, 0.5),
        "hover": SoundEffect("hover", "Hover", "/audio/hover.mp3", AudioType.INTERACTION, 0.3),
        "grab": SoundEffect("grab", "Grab", "/audio/grab.mp3", AudioType.INTERACTION, 0.4),
        "release": SoundEffect("release", "Release", "/audio/release.mp3", AudioType.INTERACTION, 0.4),
        "connect": SoundEffect("connect", "Connect", "/audio/connect.mp3", AudioType.INTERACTION, 0.6),

        # Notifications
        "ping": SoundEffect("ping", "Ping", "/audio/ping.mp3", AudioType.NOTIFICATION, 0.6),
        "alert": SoundEffect("alert", "Alert", "/audio/alert.mp3", AudioType.NOTIFICATION, 0.8),
        "success": SoundEffect("success", "Success", "/audio/success.mp3", AudioType.NOTIFICATION, 0.7),
        "error": SoundEffect("error", "Error", "/audio/error.mp3", AudioType.NOTIFICATION, 0.7),
        "user_joined": SoundEffect("user_joined", "User Joined", "/audio/join.mp3", AudioType.NOTIFICATION, 0.5),
        "user_left": SoundEffect("user_left", "User Left", "/audio/leave.mp3", AudioType.NOTIFICATION, 0.5),

        # Ambient
        "knowledge_hum": SoundEffect("knowledge_hum", "Knowledge Hum", "/audio/hum.mp3", AudioType.AMBIENT, 0.2, False),
        "activation": SoundEffect("activation", "Activation Pulse", "/audio/activation.mp3", AudioType.AMBIENT, 0.3),
    }

    def __init__(self):
        self.sources: Dict[str, AudioSource] = {}
        self.listener = AudioListener()
        self.effects: Dict[str, SoundEffect] = dict(self.DEFAULT_EFFECTS)
        self._source_counter = 0

        # Callbacks for WebXR client sync
        self.on_source_created: Optional[Callable[[AudioSource], None]] = None
        self.on_source_updated: Optional[Callable[[AudioSource], None]] = None
        self.on_source_removed: Optional[Callable[[str], None]] = None
        self.on_effect_played: Optional[Callable[[str, Vector3, float], None]] = None

    def _generate_source_id(self) -> str:
        self._source_counter += 1
        return f"audio_{self._source_counter:06d}"

    # === Source Management ===

    def create_source(
        self,
        audio_type: AudioType,
        position: Vector3,
        audio_url: str = "",
        volume: float = 1.0,
        loop: bool = False,
        spatial: bool = True,
        linked_node_id: str = None,
        tags: List[str] = None
    ) -> AudioSource:
        """Create a new audio source."""
        source_id = self._generate_source_id()

        source = AudioSource(
            source_id=source_id,
            audio_type=audio_type,
            position=position,
            audio_url=audio_url,
            volume=volume,
            loop=loop,
            is_spatial=spatial,
            linked_node_id=linked_node_id,
            tags=tags or []
        )

        self.sources[source_id] = source

        if self.on_source_created:
            self.on_source_created(source)

        return source

    def remove_source(self, source_id: str) -> bool:
        """Remove an audio source."""
        if source_id not in self.sources:
            return False

        del self.sources[source_id]

        if self.on_source_removed:
            self.on_source_removed(source_id)

        return True

    def get_source(self, source_id: str) -> Optional[AudioSource]:
        """Get a source by ID."""
        return self.sources.get(source_id)

    def get_sources_for_node(self, node_id: str) -> List[AudioSource]:
        """Get all audio sources linked to a knowledge node."""
        return [s for s in self.sources.values() if s.linked_node_id == node_id]

    # === Playback Control ===

    def play(self, source_id: str, fade_in_ms: float = 0) -> bool:
        """Start playback of a source."""
        source = self.sources.get(source_id)
        if not source:
            return False

        if fade_in_ms > 0:
            source.state = AudioState.FADING_IN
        else:
            source.state = AudioState.PLAYING

        if self.on_source_updated:
            self.on_source_updated(source)

        return True

    def pause(self, source_id: str) -> bool:
        """Pause a source."""
        source = self.sources.get(source_id)
        if not source:
            return False

        source.state = AudioState.PAUSED

        if self.on_source_updated:
            self.on_source_updated(source)

        return True

    def stop(self, source_id: str, fade_out_ms: float = 0) -> bool:
        """Stop a source."""
        source = self.sources.get(source_id)
        if not source:
            return False

        if fade_out_ms > 0:
            source.state = AudioState.FADING_OUT
        else:
            source.state = AudioState.STOPPED
            source.current_time = 0.0

        if self.on_source_updated:
            self.on_source_updated(source)

        return True

    def set_volume(self, source_id: str, volume: float) -> bool:
        """Set volume for a source."""
        source = self.sources.get(source_id)
        if not source:
            return False

        source.volume = max(0.0, min(1.0, volume))

        if self.on_source_updated:
            self.on_source_updated(source)

        return True

    def update_position(self, source_id: str, position: Vector3) -> bool:
        """Update source position."""
        source = self.sources.get(source_id)
        if not source:
            return False

        source.position = position

        if self.on_source_updated:
            self.on_source_updated(source)

        return True

    # === Sound Effects ===

    def play_effect(
        self,
        effect_id: str,
        position: Vector3 = None,
        volume: float = None
    ) -> Optional[AudioSource]:
        """Play a one-shot sound effect."""
        effect = self.effects.get(effect_id)
        if not effect:
            return None

        pos = position or Vector3()
        vol = volume if volume is not None else effect.default_volume

        # Create temporary source
        source = self.create_source(
            audio_type=effect.audio_type,
            position=pos,
            audio_url=effect.audio_url,
            volume=vol,
            loop=False,
            spatial=effect.is_spatial,
            tags=["effect", effect_id]
        )

        source.state = AudioState.PLAYING

        if self.on_effect_played:
            self.on_effect_played(effect_id, pos, vol)

        return source

    def register_effect(self, effect: SoundEffect):
        """Register a custom sound effect."""
        self.effects[effect.effect_id] = effect

    # === Listener ===

    def update_listener(
        self,
        position: Vector3 = None,
        forward: Vector3 = None,
        up: Vector3 = None
    ):
        """Update listener position and orientation."""
        if position:
            self.listener.position = position
        if forward:
            self.listener.forward = forward.normalize()
        if up:
            self.listener.up = up.normalize()

    def set_master_volume(self, volume: float):
        """Set master volume."""
        self.listener.master_volume = max(0.0, min(1.0, volume))

    def set_type_volume(self, audio_type: AudioType, volume: float):
        """Set volume for a specific audio type."""
        self.listener.type_volumes[audio_type.value] = max(0.0, min(1.0, volume))

    # === Spatial Audio Calculations ===

    def calculate_volume(self, source: AudioSource) -> float:
        """Calculate effective volume based on distance and settings."""
        if not source.is_spatial:
            return source.volume * self.listener.master_volume

        distance = source.position.distance_to(self.listener.position)

        # Distance attenuation
        if distance <= source.min_distance:
            distance_factor = 1.0
        elif distance >= source.max_distance:
            distance_factor = 0.0
        else:
            # Inverse distance rolloff
            ref_distance = source.min_distance
            distance_factor = ref_distance / (
                ref_distance + source.rolloff_factor * (distance - ref_distance)
            )

        # Type volume
        type_vol = self.listener.type_volumes.get(source.audio_type.value, 1.0)

        return source.volume * distance_factor * type_vol * self.listener.master_volume

    def calculate_pan(self, source: AudioSource) -> float:
        """Calculate stereo pan (-1 = left, 1 = right)."""
        if not source.is_spatial:
            return 0.0

        # Vector from listener to source
        to_source = Vector3(
            source.position.x - self.listener.position.x,
            source.position.y - self.listener.position.y,
            source.position.z - self.listener.position.z
        )

        # Cross product with up to get right vector
        right = Vector3(
            self.listener.forward.y * self.listener.up.z - self.listener.forward.z * self.listener.up.y,
            self.listener.forward.z * self.listener.up.x - self.listener.forward.x * self.listener.up.z,
            self.listener.forward.x * self.listener.up.y - self.listener.forward.y * self.listener.up.x
        )

        # Dot product with right vector for pan
        distance = to_source.distance_to(Vector3())
        if distance < 0.001:
            return 0.0

        pan = (to_source.x * right.x + to_source.y * right.y + to_source.z * right.z) / distance
        return max(-1.0, min(1.0, pan))

    def get_audible_sources(self) -> List[AudioSource]:
        """Get all sources that should be audible."""
        audible = []
        for source in self.sources.values():
            if source.state not in [AudioState.PLAYING, AudioState.FADING_IN, AudioState.FADING_OUT]:
                continue

            volume = self.calculate_volume(source)
            if volume > 0.01:  # Audibility threshold
                audible.append(source)

        return audible

    # === Ambient Soundscapes ===

    def create_ambient_layer(
        self,
        name: str,
        audio_url: str,
        volume: float = 0.3
    ) -> AudioSource:
        """Create a non-spatial ambient layer."""
        return self.create_source(
            audio_type=AudioType.AMBIENT,
            position=Vector3(),
            audio_url=audio_url,
            volume=volume,
            loop=True,
            spatial=False,
            tags=["ambient", name]
        )

    def set_ambient_intensity(self, intensity: float):
        """Set overall ambient audio intensity."""
        for source in self.sources.values():
            if source.audio_type == AudioType.AMBIENT:
                source.volume = source.volume * intensity

    # === Node-Based Audio ===

    def attach_audio_to_node(
        self,
        node_id: str,
        position: Vector3,
        audio_type: AudioType = AudioType.AMBIENT,
        audio_url: str = "",
        volume: float = 0.5
    ) -> AudioSource:
        """Attach an audio source to a knowledge node."""
        return self.create_source(
            audio_type=audio_type,
            position=position,
            audio_url=audio_url,
            volume=volume,
            loop=True,
            linked_node_id=node_id,
            tags=["node_audio", node_id]
        )

    def play_activation_sound(self, position: Vector3, activation_level: float):
        """Play activation sound (when a node is activated)."""
        volume = 0.3 + (activation_level * 0.4)  # 0.3 - 0.7
        self.play_effect("activation", position, volume)

    # === Export ===

    def export_state(self) -> str:
        """Export audio state for WebXR client."""
        data = {
            "listener": self.listener.to_dict(),
            "sources": [s.to_dict() for s in self.sources.values()],
            "effects": [
                {"id": e.effect_id, "name": e.name, "url": e.audio_url}
                for e in self.effects.values()
            ]
        }
        return json.dumps(data, indent=2)

    def get_statistics(self) -> Dict:
        """Get audio system statistics."""
        playing = sum(1 for s in self.sources.values() if s.state == AudioState.PLAYING)
        by_type = {}
        for source in self.sources.values():
            t = source.audio_type.value
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total_sources": len(self.sources),
            "playing": playing,
            "by_type": by_type,
            "effects_registered": len(self.effects)
        }


# Quick test
if __name__ == "__main__":
    print("Testing Spatial Audio Manager...")

    manager = SpatialAudioManager()

    # Create ambient layer
    ambient = manager.create_ambient_layer("background", "/audio/space_ambient.mp3")
    manager.play(ambient.source_id)
    print(f"Ambient source: {ambient.source_id}")

    # Attach audio to nodes
    node_audio = manager.attach_audio_to_node(
        "thompson_sampling",
        Vector3(1.0, 1.5, 2.0),
        volume=0.4
    )
    print(f"Node audio: {node_audio.source_id}")

    # Update listener position
    manager.update_listener(position=Vector3(0, 1.6, 0))

    # Play sound effects
    effect = manager.play_effect("select", Vector3(1.0, 1.5, 2.0))
    print(f"Effect played: {effect.source_id if effect else 'None'}")

    # Calculate volume at distance
    vol = manager.calculate_volume(node_audio)
    pan = manager.calculate_pan(node_audio)
    print(f"Node audio - Volume: {vol:.2f}, Pan: {pan:.2f}")

    # Get audible sources
    audible = manager.get_audible_sources()
    print(f"Audible sources: {len(audible)}")

    # Statistics
    stats = manager.get_statistics()
    print(f"Stats: {stats}")
