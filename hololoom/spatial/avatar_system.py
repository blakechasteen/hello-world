# HoloLoom Phase 4: Avatar System
# November 2025
#
# Customizable user avatars for spatial presence
# - Body/head/hand customization
# - Avatar animations and poses
# - Name tags and status indicators
# - Inverse kinematics for tracked input
# - Expression system

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


class AvatarStyle(Enum):
    """Overall avatar visual style."""
    REALISTIC = auto()      # Photorealistic appearance
    STYLIZED = auto()       # Stylized cartoon look
    ABSTRACT = auto()       # Abstract geometric shapes
    ROBOT = auto()          # Robotic/mechanical
    HOLOGRAM = auto()       # Holographic projection
    MINIMAL = auto()        # Minimalist (head + hands only)
    SILHOUETTE = auto()     # Shadow/silhouette form
    CUSTOM = auto()         # Fully custom mesh


class BodyType(Enum):
    """Avatar body type presets."""
    AVERAGE = auto()
    TALL = auto()
    SHORT = auto()
    ATHLETIC = auto()
    HEAVY = auto()
    SLIM = auto()
    CUSTOM = auto()


class HeadShape(Enum):
    """Avatar head shape options."""
    ROUND = auto()
    OVAL = auto()
    SQUARE = auto()
    HEART = auto()
    DIAMOND = auto()
    CUSTOM = auto()


class HandStyle(Enum):
    """Hand appearance style."""
    REALISTIC = auto()
    GLOVES = auto()
    ROBOT = auto()
    ABSTRACT = auto()
    TRANSPARENT = auto()
    CUSTOM = auto()


class ExpressionType(Enum):
    """Facial expression types."""
    NEUTRAL = auto()
    HAPPY = auto()
    SAD = auto()
    SURPRISED = auto()
    ANGRY = auto()
    THINKING = auto()
    CONFUSED = auto()
    EXCITED = auto()
    SLEEPY = auto()
    FOCUSED = auto()


class AnimationState(Enum):
    """Avatar animation states."""
    IDLE = auto()
    WALKING = auto()
    RUNNING = auto()
    POINTING = auto()
    WAVING = auto()
    NODDING = auto()
    SHAKING_HEAD = auto()
    THUMBS_UP = auto()
    CLAPPING = auto()
    THINKING = auto()
    PRESENTING = auto()
    GRABBING = auto()
    TYPING = auto()
    CUSTOM = auto()


class StatusType(Enum):
    """User status indicators."""
    AVAILABLE = auto()
    BUSY = auto()
    AWAY = auto()
    DO_NOT_DISTURB = auto()
    IN_MEETING = auto()
    PRESENTING = auto()
    SPECTATING = auto()
    INVISIBLE = auto()


class IKTarget(Enum):
    """Inverse kinematics target types."""
    HEAD = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()
    LEFT_FOOT = auto()
    RIGHT_FOOT = auto()
    HIPS = auto()
    CHEST = auto()


@dataclass
class Color:
    """RGBA color."""
    r: float = 1.0
    g: float = 1.0
    b: float = 1.0
    a: float = 1.0

    def to_hex(self) -> str:
        """Convert to hex string."""
        return f"#{int(self.r*255):02x}{int(self.g*255):02x}{int(self.b*255):02x}"

    def to_dict(self) -> dict[str, float]:
        return {"r": self.r, "g": self.g, "b": self.b, "a": self.a}

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        """Create from hex string."""
        hex_str = hex_str.lstrip('#')
        r = int(hex_str[0:2], 16) / 255
        g = int(hex_str[2:4], 16) / 255
        b = int(hex_str[4:6], 16) / 255
        return cls(r, g, b, 1.0)

    # Common color presets
    @classmethod
    def white(cls) -> "Color": return cls(1.0, 1.0, 1.0)
    @classmethod
    def black(cls) -> "Color": return cls(0.0, 0.0, 0.0)
    @classmethod
    def red(cls) -> "Color": return cls(1.0, 0.0, 0.0)
    @classmethod
    def green(cls) -> "Color": return cls(0.0, 1.0, 0.0)
    @classmethod
    def blue(cls) -> "Color": return cls(0.0, 0.0, 1.0)
    @classmethod
    def cyan(cls) -> "Color": return cls(0.0, 1.0, 1.0)
    @classmethod
    def magenta(cls) -> "Color": return cls(1.0, 0.0, 1.0)
    @classmethod
    def yellow(cls) -> "Color": return cls(1.0, 1.0, 0.0)


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Vector3":
        mag = self.magnitude()
        if mag == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x/mag, self.y/mag, self.z/mag)


@dataclass
class Quaternion:
    """Quaternion rotation."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}

    @classmethod
    def identity(cls) -> "Quaternion":
        return cls(0, 0, 0, 1)

    @classmethod
    def from_euler(cls, pitch: float, yaw: float, roll: float) -> "Quaternion":
        """Create quaternion from Euler angles (radians)."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        return cls(
            x=sr * cp * cy - cr * sp * sy,
            y=cr * sp * cy + sr * cp * sy,
            z=cr * cp * sy - sr * sp * cy,
            w=cr * cp * cy + sr * sp * sy
        )


@dataclass
class BodySettings:
    """Avatar body customization."""
    body_type: BodyType = BodyType.AVERAGE
    height: float = 1.7  # meters
    shoulder_width: float = 0.45
    arm_length: float = 0.6
    torso_length: float = 0.5
    leg_length: float = 0.9

    # Colors
    skin_color: Color = field(default_factory=lambda: Color(0.9, 0.75, 0.65))
    clothing_primary: Color = field(default_factory=lambda: Color(0.2, 0.4, 0.8))
    clothing_secondary: Color = field(default_factory=lambda: Color(0.3, 0.3, 0.3))

    # Clothing
    shirt_style: str = "casual"
    pants_style: str = "casual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_type": self.body_type.name,
            "height": self.height,
            "shoulder_width": self.shoulder_width,
            "arm_length": self.arm_length,
            "torso_length": self.torso_length,
            "leg_length": self.leg_length,
            "skin_color": self.skin_color.to_dict(),
            "clothing_primary": self.clothing_primary.to_dict(),
            "clothing_secondary": self.clothing_secondary.to_dict(),
            "shirt_style": self.shirt_style,
            "pants_style": self.pants_style
        }


@dataclass
class HeadSettings:
    """Avatar head customization."""
    shape: HeadShape = HeadShape.OVAL
    scale: float = 1.0

    # Hair
    hair_style: str = "short"
    hair_color: Color = field(default_factory=lambda: Color(0.2, 0.15, 0.1))

    # Eyes
    eye_color: Color = field(default_factory=lambda: Color(0.3, 0.5, 0.8))
    eye_size: float = 1.0
    eye_spacing: float = 0.5

    # Features
    eyebrow_style: str = "normal"
    nose_size: float = 1.0
    mouth_size: float = 1.0

    # Accessories
    glasses: str | None = None
    hat: str | None = None
    earrings: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape.name,
            "scale": self.scale,
            "hair_style": self.hair_style,
            "hair_color": self.hair_color.to_dict(),
            "eye_color": self.eye_color.to_dict(),
            "eye_size": self.eye_size,
            "eye_spacing": self.eye_spacing,
            "eyebrow_style": self.eyebrow_style,
            "nose_size": self.nose_size,
            "mouth_size": self.mouth_size,
            "glasses": self.glasses,
            "hat": self.hat,
            "earrings": self.earrings
        }


@dataclass
class HandSettings:
    """Avatar hand customization."""
    style: HandStyle = HandStyle.REALISTIC
    size: float = 1.0

    # Appearance
    skin_color: Color | None = None  # None = match body
    glove_color: Color = field(default_factory=lambda: Color(0.2, 0.2, 0.2))

    # Accessories
    rings: list[str] = field(default_factory=list)
    bracelet: str | None = None
    watch: str | None = None

    # Effects
    trail_enabled: bool = False
    trail_color: Color = field(default_factory=lambda: Color(0.5, 0.8, 1.0, 0.5))

    def to_dict(self) -> dict[str, Any]:
        return {
            "style": self.style.name,
            "size": self.size,
            "skin_color": self.skin_color.to_dict() if self.skin_color else None,
            "glove_color": self.glove_color.to_dict(),
            "rings": self.rings,
            "bracelet": self.bracelet,
            "watch": self.watch,
            "trail_enabled": self.trail_enabled,
            "trail_color": self.trail_color.to_dict()
        }


@dataclass
class Expression:
    """Facial expression configuration."""
    expression_type: ExpressionType
    intensity: float = 1.0  # 0.0 to 1.0

    # Blend shape values (0-1)
    brow_raise: float = 0.0
    brow_furrow: float = 0.0
    eye_squint: float = 0.0
    eye_wide: float = 0.0
    smile: float = 0.0
    frown: float = 0.0
    mouth_open: float = 0.0
    lip_pucker: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression_type": self.expression_type.name,
            "intensity": self.intensity,
            "blend_shapes": {
                "brow_raise": self.brow_raise,
                "brow_furrow": self.brow_furrow,
                "eye_squint": self.eye_squint,
                "eye_wide": self.eye_wide,
                "smile": self.smile,
                "frown": self.frown,
                "mouth_open": self.mouth_open,
                "lip_pucker": self.lip_pucker
            }
        }

    @classmethod
    def neutral(cls) -> "Expression":
        return cls(ExpressionType.NEUTRAL)

    @classmethod
    def happy(cls, intensity: float = 1.0) -> "Expression":
        return cls(
            ExpressionType.HAPPY,
            intensity=intensity,
            smile=0.8 * intensity,
            eye_squint=0.3 * intensity
        )

    @classmethod
    def sad(cls, intensity: float = 1.0) -> "Expression":
        return cls(
            ExpressionType.SAD,
            intensity=intensity,
            frown=0.7 * intensity,
            brow_furrow=0.4 * intensity
        )

    @classmethod
    def surprised(cls, intensity: float = 1.0) -> "Expression":
        return cls(
            ExpressionType.SURPRISED,
            intensity=intensity,
            brow_raise=0.9 * intensity,
            eye_wide=0.8 * intensity,
            mouth_open=0.5 * intensity
        )

    @classmethod
    def thinking(cls, intensity: float = 1.0) -> "Expression":
        return cls(
            ExpressionType.THINKING,
            intensity=intensity,
            brow_raise=0.3 * intensity,
            brow_furrow=0.2 * intensity,
            eye_squint=0.2 * intensity
        )


@dataclass
class IKState:
    """Inverse kinematics state for avatar."""
    head_position: Vector3 = field(default_factory=Vector3)
    head_rotation: Quaternion = field(default_factory=Quaternion.identity)

    left_hand_position: Vector3 = field(default_factory=Vector3)
    left_hand_rotation: Quaternion = field(default_factory=Quaternion.identity)
    left_hand_tracking: bool = False

    right_hand_position: Vector3 = field(default_factory=Vector3)
    right_hand_rotation: Quaternion = field(default_factory=Quaternion.identity)
    right_hand_tracking: bool = False

    # Optional foot tracking (full body)
    left_foot_position: Vector3 | None = None
    left_foot_rotation: Quaternion | None = None
    right_foot_position: Vector3 | None = None
    right_foot_rotation: Quaternion | None = None

    # Hip tracking (for full body)
    hip_position: Vector3 | None = None
    hip_rotation: Quaternion | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "head": {
                "position": self.head_position.to_dict(),
                "rotation": self.head_rotation.to_dict()
            },
            "left_hand": {
                "position": self.left_hand_position.to_dict(),
                "rotation": self.left_hand_rotation.to_dict(),
                "tracking": self.left_hand_tracking
            },
            "right_hand": {
                "position": self.right_hand_position.to_dict(),
                "rotation": self.right_hand_rotation.to_dict(),
                "tracking": self.right_hand_tracking
            }
        }

        if self.left_foot_position:
            result["left_foot"] = {
                "position": self.left_foot_position.to_dict(),
                "rotation": self.left_foot_rotation.to_dict() if self.left_foot_rotation else None
            }
        if self.right_foot_position:
            result["right_foot"] = {
                "position": self.right_foot_position.to_dict(),
                "rotation": self.right_foot_rotation.to_dict() if self.right_foot_rotation else None
            }
        if self.hip_position:
            result["hips"] = {
                "position": self.hip_position.to_dict(),
                "rotation": self.hip_rotation.to_dict() if self.hip_rotation else None
            }

        return result


@dataclass
class NameTag:
    """Floating name tag above avatar."""
    display_name: str
    title: str | None = None  # e.g., "Team Lead", "Guest"

    # Appearance
    background_color: Color = field(default_factory=lambda: Color(0.1, 0.1, 0.1, 0.8))
    text_color: Color = field(default_factory=Color.white)
    title_color: Color = field(default_factory=lambda: Color(0.7, 0.7, 0.7))

    # Position
    offset_y: float = 0.25  # Above head
    face_camera: bool = True

    # Visibility
    visible: bool = True
    visible_distance: float = 20.0  # Fade out beyond this

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "title": self.title,
            "background_color": self.background_color.to_dict(),
            "text_color": self.text_color.to_dict(),
            "title_color": self.title_color.to_dict(),
            "offset_y": self.offset_y,
            "face_camera": self.face_camera,
            "visible": self.visible,
            "visible_distance": self.visible_distance
        }


@dataclass
class StatusIndicator:
    """Status indicator for avatar."""
    status: StatusType = StatusType.AVAILABLE
    custom_message: str | None = None

    # Appearance
    show_indicator: bool = True
    indicator_position: str = "name_tag"  # "name_tag", "chest", "hand"

    def get_color(self) -> Color:
        """Get color for status type."""
        colors = {
            StatusType.AVAILABLE: Color(0.2, 0.8, 0.2),      # Green
            StatusType.BUSY: Color(0.8, 0.2, 0.2),           # Red
            StatusType.AWAY: Color(0.8, 0.8, 0.2),           # Yellow
            StatusType.DO_NOT_DISTURB: Color(0.6, 0.0, 0.0), # Dark red
            StatusType.IN_MEETING: Color(0.8, 0.4, 0.0),     # Orange
            StatusType.PRESENTING: Color(0.4, 0.4, 0.8),     # Purple
            StatusType.SPECTATING: Color(0.4, 0.6, 0.8),     # Light blue
            StatusType.INVISIBLE: Color(0.5, 0.5, 0.5, 0.3)  # Faded gray
        }
        return colors.get(self.status, Color.white())

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.name,
            "custom_message": self.custom_message,
            "show_indicator": self.show_indicator,
            "indicator_position": self.indicator_position,
            "color": self.get_color().to_dict()
        }


@dataclass
class AnimationClip:
    """Single animation clip."""
    clip_id: str
    name: str
    state: AnimationState
    duration: float  # seconds
    loop: bool = False
    blend_in: float = 0.2  # Transition time
    blend_out: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "name": self.name,
            "state": self.state.name,
            "duration": self.duration,
            "loop": self.loop,
            "blend_in": self.blend_in,
            "blend_out": self.blend_out
        }


@dataclass
class Avatar:
    """Complete avatar configuration."""
    avatar_id: str
    user_id: str

    # Style
    style: AvatarStyle = AvatarStyle.STYLIZED

    # Customization
    body: BodySettings = field(default_factory=BodySettings)
    head: HeadSettings = field(default_factory=HeadSettings)
    left_hand: HandSettings = field(default_factory=HandSettings)
    right_hand: HandSettings = field(default_factory=HandSettings)

    # State
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion.identity)
    expression: Expression = field(default_factory=Expression.neutral)
    animation_state: AnimationState = AnimationState.IDLE

    # IK
    ik_state: IKState = field(default_factory=IKState)
    ik_enabled: bool = True

    # UI Elements
    name_tag: NameTag | None = None
    status: StatusIndicator = field(default_factory=StatusIndicator)

    # Visibility
    visible: bool = True
    opacity: float = 1.0

    # Effects
    outline_enabled: bool = False
    outline_color: Color = field(default_factory=lambda: Color(0.2, 0.6, 1.0))
    glow_enabled: bool = False
    glow_color: Color = field(default_factory=lambda: Color(0.5, 0.8, 1.0, 0.3))

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "avatar_id": self.avatar_id,
            "user_id": self.user_id,
            "style": self.style.name,
            "body": self.body.to_dict(),
            "head": self.head.to_dict(),
            "left_hand": self.left_hand.to_dict(),
            "right_hand": self.right_hand.to_dict(),
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "expression": self.expression.to_dict(),
            "animation_state": self.animation_state.name,
            "ik_state": self.ik_state.to_dict(),
            "ik_enabled": self.ik_enabled,
            "name_tag": self.name_tag.to_dict() if self.name_tag else None,
            "status": self.status.to_dict(),
            "visible": self.visible,
            "opacity": self.opacity,
            "outline_enabled": self.outline_enabled,
            "outline_color": self.outline_color.to_dict(),
            "glow_enabled": self.glow_enabled,
            "glow_color": self.glow_color.to_dict()
        }


class AvatarAnimator:
    """Manages avatar animations."""

    def __init__(self):
        self.clips: dict[str, AnimationClip] = {}
        self.current_clip: AnimationClip | None = None
        self.blend_weight: float = 0.0
        self.playback_time: float = 0.0

        # Register default clips
        self._register_default_clips()

    def _register_default_clips(self):
        """Register built-in animation clips."""
        defaults = [
            AnimationClip("idle", "Idle", AnimationState.IDLE, 2.0, loop=True),
            AnimationClip("wave", "Wave", AnimationState.WAVING, 1.5, loop=False),
            AnimationClip("nod", "Nod", AnimationState.NODDING, 0.8, loop=False),
            AnimationClip("shake_head", "Shake Head", AnimationState.SHAKING_HEAD, 1.0, loop=False),
            AnimationClip("thumbs_up", "Thumbs Up", AnimationState.THUMBS_UP, 1.2, loop=False),
            AnimationClip("clap", "Clap", AnimationState.CLAPPING, 0.6, loop=True),
            AnimationClip("think", "Thinking", AnimationState.THINKING, 3.0, loop=True),
            AnimationClip("present", "Presenting", AnimationState.PRESENTING, 2.0, loop=True),
            AnimationClip("point", "Pointing", AnimationState.POINTING, 1.0, loop=False),
            AnimationClip("grab", "Grabbing", AnimationState.GRABBING, 0.3, loop=False),
            AnimationClip("walk", "Walking", AnimationState.WALKING, 1.0, loop=True),
            AnimationClip("run", "Running", AnimationState.RUNNING, 0.6, loop=True),
        ]

        for clip in defaults:
            self.clips[clip.clip_id] = clip

    def play(self, clip_id: str) -> bool:
        """Start playing an animation clip."""
        if clip_id not in self.clips:
            return False

        self.current_clip = self.clips[clip_id]
        self.playback_time = 0.0
        self.blend_weight = 0.0
        return True

    def update(self, delta_time: float) -> AnimationState | None:
        """Update animation playback."""
        if not self.current_clip:
            return None

        # Update blend weight
        if self.blend_weight < 1.0:
            self.blend_weight = min(1.0, self.blend_weight + delta_time / self.current_clip.blend_in)

        # Update playback time
        self.playback_time += delta_time

        # Check if clip finished
        if not self.current_clip.loop and self.playback_time >= self.current_clip.duration:
            finished_state = self.current_clip.state
            self.current_clip = self.clips.get("idle")
            self.playback_time = 0.0
            return finished_state

        # Loop if needed
        if self.current_clip.loop:
            self.playback_time = self.playback_time % self.current_clip.duration

        return self.current_clip.state

    def get_state(self) -> dict[str, Any]:
        """Get current animation state."""
        return {
            "current_clip": self.current_clip.clip_id if self.current_clip else None,
            "playback_time": self.playback_time,
            "blend_weight": self.blend_weight,
            "available_clips": list(self.clips.keys())
        }


class AvatarIKSolver:
    """Simple IK solver for avatar limbs."""

    def __init__(self, avatar: Avatar):
        self.avatar = avatar
        self.arm_length = avatar.body.arm_length
        self.shoulder_width = avatar.body.shoulder_width

    def solve_arm(
        self,
        shoulder_pos: Vector3,
        target_pos: Vector3,
        is_left: bool
    ) -> tuple[Quaternion, Quaternion]:
        """
        Solve IK for arm from shoulder to hand target.
        Returns (upper_arm_rotation, forearm_rotation).
        """
        # Calculate direction to target
        dx = target_pos.x - shoulder_pos.x
        dy = target_pos.y - shoulder_pos.y
        dz = target_pos.z - shoulder_pos.z

        distance = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Clamp to arm length
        upper_arm = self.arm_length * 0.5
        forearm = self.arm_length * 0.5
        max_reach = upper_arm + forearm

        if distance > max_reach * 0.99:
            # Fully extended
            scale = max_reach * 0.99 / distance
            dx *= scale
            dy *= scale
            dz *= scale
            distance = max_reach * 0.99

        # Calculate elbow bend using law of cosines
        cos_angle = (upper_arm*upper_arm + forearm*forearm - distance*distance) / (2 * upper_arm * forearm)
        cos_angle = max(-1, min(1, cos_angle))
        elbow_angle = math.acos(cos_angle)

        # Calculate shoulder rotation (simplified)
        yaw = math.atan2(dx, dz)
        pitch = math.atan2(dy, math.sqrt(dx*dx + dz*dz))

        upper_rot = Quaternion.from_euler(pitch, yaw, 0)
        forearm_rot = Quaternion.from_euler(math.pi - elbow_angle, 0, 0)

        return upper_rot, forearm_rot

    def update_from_tracking(self, ik_state: IKState) -> dict[str, Quaternion]:
        """Update avatar bone rotations from IK state."""
        results = {}

        # Head rotation
        results["head"] = ik_state.head_rotation

        # Calculate shoulder positions based on body
        height = self.avatar.body.height
        shoulder_y = height * 0.85

        # Left arm IK
        if ik_state.left_hand_tracking:
            left_shoulder = Vector3(
                -self.shoulder_width / 2,
                shoulder_y,
                0
            )
            upper, forearm = self.solve_arm(
                left_shoulder,
                ik_state.left_hand_position,
                is_left=True
            )
            results["left_upper_arm"] = upper
            results["left_forearm"] = forearm
            results["left_hand"] = ik_state.left_hand_rotation

        # Right arm IK
        if ik_state.right_hand_tracking:
            right_shoulder = Vector3(
                self.shoulder_width / 2,
                shoulder_y,
                0
            )
            upper, forearm = self.solve_arm(
                right_shoulder,
                ik_state.right_hand_position,
                is_left=False
            )
            results["right_upper_arm"] = upper
            results["right_forearm"] = forearm
            results["right_hand"] = ik_state.right_hand_rotation

        return results


class AvatarManager:
    """Manages multiple avatars in a session."""

    def __init__(self):
        self.avatars: dict[str, Avatar] = {}
        self.animators: dict[str, AvatarAnimator] = {}
        self.ik_solvers: dict[str, AvatarIKSolver] = {}
        self.local_avatar_id: str | None = None

    def create_avatar(
        self,
        user_id: str,
        display_name: str,
        style: AvatarStyle = AvatarStyle.STYLIZED,
        is_local: bool = False
    ) -> Avatar:
        """Create a new avatar for a user."""
        avatar_id = str(uuid.uuid4())[:8]

        avatar = Avatar(
            avatar_id=avatar_id,
            user_id=user_id,
            style=style,
            name_tag=NameTag(display_name=display_name)
        )

        self.avatars[avatar_id] = avatar
        self.animators[avatar_id] = AvatarAnimator()
        self.ik_solvers[avatar_id] = AvatarIKSolver(avatar)

        if is_local:
            self.local_avatar_id = avatar_id

        return avatar

    def get_avatar(self, avatar_id: str) -> Avatar | None:
        """Get avatar by ID."""
        return self.avatars.get(avatar_id)

    def get_local_avatar(self) -> Avatar | None:
        """Get the local user's avatar."""
        if self.local_avatar_id:
            return self.avatars.get(self.local_avatar_id)
        return None

    def remove_avatar(self, avatar_id: str) -> bool:
        """Remove an avatar."""
        if avatar_id in self.avatars:
            del self.avatars[avatar_id]
            if avatar_id in self.animators:
                del self.animators[avatar_id]
            if avatar_id in self.ik_solvers:
                del self.ik_solvers[avatar_id]
            if self.local_avatar_id == avatar_id:
                self.local_avatar_id = None
            return True
        return False

    def update_position(
        self,
        avatar_id: str,
        position: Vector3,
        rotation: Quaternion | None = None
    ) -> bool:
        """Update avatar position and rotation."""
        avatar = self.avatars.get(avatar_id)
        if not avatar:
            return False

        avatar.position = position
        if rotation:
            avatar.rotation = rotation
        avatar.last_updated = datetime.now()
        return True

    def update_ik(self, avatar_id: str, ik_state: IKState) -> bool:
        """Update avatar IK state from tracking data."""
        avatar = self.avatars.get(avatar_id)
        if not avatar or not avatar.ik_enabled:
            return False

        avatar.ik_state = ik_state
        avatar.last_updated = datetime.now()

        # Solve IK
        solver = self.ik_solvers.get(avatar_id)
        if solver:
            solver.update_from_tracking(ik_state)

        return True

    def set_expression(
        self,
        avatar_id: str,
        expression: Expression
    ) -> bool:
        """Set avatar facial expression."""
        avatar = self.avatars.get(avatar_id)
        if not avatar:
            return False

        avatar.expression = expression
        avatar.last_updated = datetime.now()
        return True

    def play_animation(
        self,
        avatar_id: str,
        animation: str
    ) -> bool:
        """Play animation on avatar."""
        animator = self.animators.get(avatar_id)
        avatar = self.avatars.get(avatar_id)

        if not animator or not avatar:
            return False

        if animator.play(animation):
            avatar.last_updated = datetime.now()
            return True
        return False

    def set_status(
        self,
        avatar_id: str,
        status: StatusType,
        custom_message: str | None = None
    ) -> bool:
        """Set avatar status."""
        avatar = self.avatars.get(avatar_id)
        if not avatar:
            return False

        avatar.status.status = status
        avatar.status.custom_message = custom_message
        avatar.last_updated = datetime.now()
        return True

    def update(self, delta_time: float):
        """Update all avatars (animations, etc.)."""
        for avatar_id, animator in self.animators.items():
            new_state = animator.update(delta_time)
            if new_state:
                avatar = self.avatars.get(avatar_id)
                if avatar:
                    avatar.animation_state = new_state

    def customize_avatar(
        self,
        avatar_id: str,
        body: BodySettings | None = None,
        head: HeadSettings | None = None,
        left_hand: HandSettings | None = None,
        right_hand: HandSettings | None = None
    ) -> bool:
        """Customize avatar appearance."""
        avatar = self.avatars.get(avatar_id)
        if not avatar:
            return False

        if body:
            avatar.body = body
            # Update IK solver with new proportions
            self.ik_solvers[avatar_id] = AvatarIKSolver(avatar)
        if head:
            avatar.head = head
        if left_hand:
            avatar.left_hand = left_hand
        if right_hand:
            avatar.right_hand = right_hand

        avatar.last_updated = datetime.now()
        return True

    def to_state(self) -> dict[str, Any]:
        """Export state for WebXR client."""
        return {
            "avatars": {
                avatar_id: avatar.to_dict()
                for avatar_id, avatar in self.avatars.items()
            },
            "animations": {
                avatar_id: animator.get_state()
                for avatar_id, animator in self.animators.items()
            },
            "local_avatar_id": self.local_avatar_id,
            "avatar_count": len(self.avatars)
        }


# Preset avatar templates
class AvatarPresets:
    """Pre-configured avatar templates."""

    @staticmethod
    def professional(user_id: str, name: str) -> Avatar:
        """Professional business avatar."""
        avatar = Avatar(
            avatar_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            style=AvatarStyle.STYLIZED
        )
        avatar.body.clothing_primary = Color(0.15, 0.2, 0.3)  # Dark suit
        avatar.body.clothing_secondary = Color.white()  # White shirt
        avatar.body.shirt_style = "formal"
        avatar.body.pants_style = "formal"
        avatar.name_tag = NameTag(display_name=name, title="Professional")
        return avatar

    @staticmethod
    def casual(user_id: str, name: str) -> Avatar:
        """Casual everyday avatar."""
        avatar = Avatar(
            avatar_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            style=AvatarStyle.STYLIZED
        )
        avatar.body.clothing_primary = Color(0.2, 0.5, 0.8)  # Blue
        avatar.body.clothing_secondary = Color(0.3, 0.3, 0.3)  # Gray
        avatar.name_tag = NameTag(display_name=name)
        return avatar

    @staticmethod
    def robot(user_id: str, name: str) -> Avatar:
        """Robot-style avatar."""
        avatar = Avatar(
            avatar_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            style=AvatarStyle.ROBOT
        )
        avatar.body.skin_color = Color(0.6, 0.65, 0.7)  # Metallic
        avatar.head.eye_color = Color(0.2, 0.8, 1.0)  # Cyan glow
        avatar.glow_enabled = True
        avatar.glow_color = Color(0.2, 0.6, 1.0, 0.2)
        avatar.name_tag = NameTag(
            display_name=name,
            background_color=Color(0.0, 0.2, 0.3, 0.9)
        )
        return avatar

    @staticmethod
    def hologram(user_id: str, name: str) -> Avatar:
        """Holographic avatar."""
        avatar = Avatar(
            avatar_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            style=AvatarStyle.HOLOGRAM
        )
        avatar.opacity = 0.8
        avatar.body.skin_color = Color(0.3, 0.5, 1.0, 0.8)  # Blue tint
        avatar.outline_enabled = True
        avatar.outline_color = Color(0.4, 0.6, 1.0)
        avatar.glow_enabled = True
        avatar.name_tag = NameTag(
            display_name=name,
            background_color=Color(0.0, 0.1, 0.2, 0.6)
        )
        return avatar

    @staticmethod
    def minimal(user_id: str, name: str) -> Avatar:
        """Minimalist avatar (head and hands only)."""
        avatar = Avatar(
            avatar_id=str(uuid.uuid4())[:8],
            user_id=user_id,
            style=AvatarStyle.MINIMAL
        )
        avatar.left_hand.style = HandStyle.ABSTRACT
        avatar.right_hand.style = HandStyle.ABSTRACT
        avatar.name_tag = NameTag(display_name=name)
        return avatar
