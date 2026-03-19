"""
Haptic Feedback System for HoloLoom Spatial Computing.

Advanced haptic feedback for XR controllers and hand tracking,
providing tactile responses for interactions.

Features:
- Controller vibration patterns
- Hand haptic simulation (for devices that support it)
- Haptic curves and envelopes
- Spatial haptics (location-based feedback)
- Collision feedback
- UI interaction haptics
- Custom haptic patterns

Created: November 2025
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class HapticDevice(Enum):
    """Target haptic device."""
    LEFT_CONTROLLER = "left_controller"
    RIGHT_CONTROLLER = "right_controller"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"
    BOTH_CONTROLLERS = "both_controllers"
    BOTH_HANDS = "both_hands"
    ALL = "all"


class HapticIntensity(Enum):
    """Preset intensity levels."""
    NONE = 0.0
    SUBTLE = 0.15
    LIGHT = 0.3
    MEDIUM = 0.5
    STRONG = 0.7
    INTENSE = 0.85
    MAXIMUM = 1.0


class HapticWaveform(Enum):
    """Haptic waveform types."""
    CONSTANT = "constant"      # Steady vibration
    SINE = "sine"              # Smooth oscillation
    SQUARE = "square"          # On/off pulses
    SAWTOOTH = "sawtooth"      # Ramping pattern
    TRIANGLE = "triangle"      # Up-down ramp
    NOISE = "noise"            # Random texture
    PULSE = "pulse"            # Quick burst
    CUSTOM = "custom"          # Custom envelope


@dataclass
class HapticEnvelope:
    """
    Attack-Decay-Sustain-Release envelope for haptic intensity.

    Shapes the intensity over time for more natural haptics.
    """
    attack_time: float = 0.05   # Time to reach peak (seconds)
    decay_time: float = 0.1     # Time from peak to sustain
    sustain_level: float = 0.7  # Sustained intensity (0-1)
    release_time: float = 0.1   # Time to fade out

    def get_intensity(self, time: float, duration: float) -> float:
        """Get intensity at a specific time in the envelope."""
        if time < 0:
            return 0.0

        # Attack phase
        if time < self.attack_time:
            return time / self.attack_time if self.attack_time > 0 else 1.0

        # Decay phase
        decay_start = self.attack_time
        if time < decay_start + self.decay_time:
            t = (time - decay_start) / self.decay_time if self.decay_time > 0 else 1.0
            return 1.0 - (1.0 - self.sustain_level) * t

        # Sustain phase
        sustain_end = duration - self.release_time
        if time < sustain_end:
            return self.sustain_level

        # Release phase
        if time < duration:
            t = (time - sustain_end) / self.release_time if self.release_time > 0 else 1.0
            return self.sustain_level * (1.0 - t)

        return 0.0


@dataclass
class HapticPulse:
    """A single haptic pulse/event."""
    device: HapticDevice
    intensity: float  # 0.0 to 1.0
    duration: float   # Seconds
    frequency: float = 150.0  # Hz (typical range 50-300Hz)
    waveform: HapticWaveform = HapticWaveform.CONSTANT
    envelope: HapticEnvelope | None = None
    delay: float = 0.0  # Delay before starting

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device.value,
            "intensity": self.intensity,
            "duration": self.duration,
            "frequency": self.frequency,
            "waveform": self.waveform.value,
            "envelope": {
                "attack": self.envelope.attack_time,
                "decay": self.envelope.decay_time,
                "sustain": self.envelope.sustain_level,
                "release": self.envelope.release_time
            } if self.envelope else None,
            "delay": self.delay
        }


@dataclass
class HapticPattern:
    """A sequence of haptic pulses forming a pattern."""
    pattern_id: str
    name: str
    pulses: list[HapticPulse] = field(default_factory=list)
    loop: bool = False
    loop_count: int = 0  # 0 = infinite
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_total_duration(self) -> float:
        """Get total duration of pattern."""
        if not self.pulses:
            return 0.0
        return max(p.delay + p.duration for p in self.pulses)

    def add_pulse(
        self,
        intensity: float,
        duration: float,
        device: HapticDevice = HapticDevice.BOTH_CONTROLLERS,
        delay: float = 0.0,
        frequency: float = 150.0,
        waveform: HapticWaveform = HapticWaveform.CONSTANT
    ) -> "HapticPattern":
        """Add a pulse to the pattern (fluent interface)."""
        self.pulses.append(HapticPulse(
            device=device,
            intensity=intensity,
            duration=duration,
            frequency=frequency,
            waveform=waveform,
            delay=delay
        ))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.pattern_id,
            "name": self.name,
            "pulses": [p.to_dict() for p in self.pulses],
            "loop": self.loop,
            "loopCount": self.loop_count,
            "totalDuration": self.get_total_duration()
        }


@dataclass
class SpatialHapticZone:
    """A zone in 3D space that triggers haptics when entered/touched."""
    zone_id: str
    position: tuple[float, float, float]
    radius: float
    pattern: HapticPattern
    trigger_on_enter: bool = True
    trigger_on_exit: bool = False
    trigger_continuous: bool = False
    intensity_falloff: bool = True  # Intensity decreases with distance
    enabled: bool = True


@dataclass
class CollisionHaptic:
    """Haptic feedback configuration for collisions."""
    velocity_threshold: float = 0.1  # Minimum velocity to trigger
    max_intensity: float = 1.0
    intensity_curve: str = "linear"  # linear, quadratic, logarithmic
    duration_base: float = 0.05
    duration_scale: float = 0.1  # Scales with velocity
    frequency_base: float = 100.0
    frequency_scale: float = 50.0  # Higher freq for harder hits


class HapticPatternLibrary:
    """Library of preset haptic patterns."""

    @staticmethod
    def click() -> HapticPattern:
        """Short click feedback."""
        return HapticPattern(
            pattern_id="click",
            name="Click"
        ).add_pulse(0.5, 0.02)

    @staticmethod
    def double_click() -> HapticPattern:
        """Double click feedback."""
        return HapticPattern(
            pattern_id="double_click",
            name="Double Click"
        ).add_pulse(0.5, 0.02, delay=0.0).add_pulse(0.5, 0.02, delay=0.1)

    @staticmethod
    def success() -> HapticPattern:
        """Success/confirmation feedback."""
        pattern = HapticPattern(pattern_id="success", name="Success")
        pattern.add_pulse(0.3, 0.05, delay=0.0)
        pattern.add_pulse(0.5, 0.05, delay=0.08)
        pattern.add_pulse(0.7, 0.1, delay=0.16)
        return pattern

    @staticmethod
    def error() -> HapticPattern:
        """Error/warning feedback."""
        pattern = HapticPattern(pattern_id="error", name="Error")
        pattern.add_pulse(0.8, 0.15, frequency=80)
        pattern.add_pulse(0.8, 0.15, delay=0.2, frequency=80)
        pattern.add_pulse(0.8, 0.15, delay=0.4, frequency=80)
        return pattern

    @staticmethod
    def notification() -> HapticPattern:
        """Gentle notification."""
        pattern = HapticPattern(pattern_id="notification", name="Notification")
        pulse = HapticPulse(
            device=HapticDevice.BOTH_CONTROLLERS,
            intensity=0.4,
            duration=0.3,
            frequency=120,
            envelope=HapticEnvelope(0.05, 0.1, 0.3, 0.15)
        )
        pattern.pulses.append(pulse)
        return pattern

    @staticmethod
    def grab() -> HapticPattern:
        """Object grab feedback."""
        return HapticPattern(
            pattern_id="grab",
            name="Grab"
        ).add_pulse(0.6, 0.08, waveform=HapticWaveform.PULSE)

    @staticmethod
    def release() -> HapticPattern:
        """Object release feedback."""
        pattern = HapticPattern(pattern_id="release", name="Release")
        pulse = HapticPulse(
            device=HapticDevice.BOTH_CONTROLLERS,
            intensity=0.3,
            duration=0.1,
            envelope=HapticEnvelope(0.0, 0.0, 0.3, 0.1)
        )
        pattern.pulses.append(pulse)
        return pattern

    @staticmethod
    def hover() -> HapticPattern:
        """Hover over interactive element."""
        return HapticPattern(
            pattern_id="hover",
            name="Hover"
        ).add_pulse(0.15, 0.03)

    @staticmethod
    def scroll_tick() -> HapticPattern:
        """Scrolling tick feedback."""
        return HapticPattern(
            pattern_id="scroll_tick",
            name="Scroll Tick"
        ).add_pulse(0.2, 0.01, frequency=200)

    @staticmethod
    def impact_light() -> HapticPattern:
        """Light impact/collision."""
        return HapticPattern(
            pattern_id="impact_light",
            name="Impact Light"
        ).add_pulse(0.4, 0.05, frequency=100)

    @staticmethod
    def impact_heavy() -> HapticPattern:
        """Heavy impact/collision."""
        pattern = HapticPattern(pattern_id="impact_heavy", name="Impact Heavy")
        pulse = HapticPulse(
            device=HapticDevice.BOTH_CONTROLLERS,
            intensity=0.9,
            duration=0.15,
            frequency=60,
            envelope=HapticEnvelope(0.0, 0.05, 0.5, 0.1)
        )
        pattern.pulses.append(pulse)
        return pattern

    @staticmethod
    def heartbeat() -> HapticPattern:
        """Heartbeat pattern."""
        pattern = HapticPattern(pattern_id="heartbeat", name="Heartbeat")
        pattern.add_pulse(0.7, 0.08, delay=0.0)
        pattern.add_pulse(0.5, 0.06, delay=0.15)
        pattern.loop = True
        pattern.loop_count = 0
        return pattern

    @staticmethod
    def texture_rough() -> HapticPattern:
        """Rough texture simulation."""
        pattern = HapticPattern(pattern_id="texture_rough", name="Texture Rough")
        pattern.add_pulse(0.3, 0.5, waveform=HapticWaveform.NOISE, frequency=180)
        pattern.loop = True
        return pattern

    @staticmethod
    def texture_smooth() -> HapticPattern:
        """Smooth texture simulation."""
        pattern = HapticPattern(pattern_id="texture_smooth", name="Texture Smooth")
        pattern.add_pulse(0.1, 0.5, waveform=HapticWaveform.SINE, frequency=50)
        pattern.loop = True
        return pattern

    @staticmethod
    def countdown_tick() -> HapticPattern:
        """Countdown timer tick."""
        return HapticPattern(
            pattern_id="countdown_tick",
            name="Countdown Tick"
        ).add_pulse(0.4, 0.03, frequency=150)

    @staticmethod
    def teleport() -> HapticPattern:
        """Teleportation effect."""
        pattern = HapticPattern(pattern_id="teleport", name="Teleport")
        # Rising intensity
        for i in range(5):
            pattern.add_pulse(0.2 + i * 0.15, 0.05, delay=i * 0.06)
        return pattern

    @staticmethod
    def boundary_warning() -> HapticPattern:
        """Guardian/boundary warning."""
        pattern = HapticPattern(pattern_id="boundary_warning", name="Boundary Warning")
        pattern.add_pulse(0.6, 0.2, waveform=HapticWaveform.SINE, frequency=100)
        pattern.loop = True
        return pattern


class HapticFeedbackManager:
    """
    Main haptic feedback management system.

    Coordinates haptic output across devices, manages patterns,
    and handles spatial haptics.
    """

    def __init__(
        self,
        default_device: HapticDevice = HapticDevice.BOTH_CONTROLLERS,
        master_intensity: float = 1.0,
        enable_spatial: bool = True
    ):
        self.default_device = default_device
        self.master_intensity = master_intensity
        self.enable_spatial = enable_spatial

        # Pattern library
        self.patterns: dict[str, HapticPattern] = {}
        self._load_preset_patterns()

        # Spatial zones
        self.spatial_zones: dict[str, SpatialHapticZone] = {}

        # Active haptics
        self.active_pulses: list[dict[str, Any]] = []

        # Collision configuration
        self.collision_config = CollisionHaptic()

        # Device availability
        self.available_devices: dict[HapticDevice, bool] = dict.fromkeys(HapticDevice, True)

        # Haptic history for analytics
        self.haptic_history: list[dict[str, Any]] = []
        self.max_history = 1000

        # Callbacks
        self.on_haptic_start: list[Callable[[str, HapticDevice], None]] = []
        self.on_haptic_complete: list[Callable[[str], None]] = []

    def _load_preset_patterns(self):
        """Load preset haptic patterns."""
        library = HapticPatternLibrary()
        presets = [
            library.click(),
            library.double_click(),
            library.success(),
            library.error(),
            library.notification(),
            library.grab(),
            library.release(),
            library.hover(),
            library.scroll_tick(),
            library.impact_light(),
            library.impact_heavy(),
            library.heartbeat(),
            library.texture_rough(),
            library.texture_smooth(),
            library.countdown_tick(),
            library.teleport(),
            library.boundary_warning(),
        ]

        for pattern in presets:
            self.patterns[pattern.pattern_id] = pattern

    def register_pattern(self, pattern: HapticPattern):
        """Register a custom haptic pattern."""
        self.patterns[pattern.pattern_id] = pattern

    def play_pattern(
        self,
        pattern_id: str,
        device: HapticDevice | None = None,
        intensity_scale: float = 1.0
    ) -> bool:
        """Play a registered haptic pattern."""
        if pattern_id not in self.patterns:
            return False

        pattern = self.patterns[pattern_id]
        target_device = device or self.default_device

        if not self._is_device_available(target_device):
            return False

        # Scale intensity
        effective_intensity = self.master_intensity * intensity_scale

        # Queue all pulses
        for pulse in pattern.pulses:
            self._queue_pulse(pulse, target_device, effective_intensity)

        # Record history
        self._record_haptic(pattern_id, target_device, pattern.get_total_duration())

        # Fire callbacks
        for callback in self.on_haptic_start:
            callback(pattern_id, target_device)

        return True

    def play_pulse(
        self,
        intensity: float,
        duration: float,
        device: HapticDevice | None = None,
        frequency: float = 150.0,
        waveform: HapticWaveform = HapticWaveform.CONSTANT
    ) -> bool:
        """Play a single haptic pulse."""
        target_device = device or self.default_device

        if not self._is_device_available(target_device):
            return False

        pulse = HapticPulse(
            device=target_device,
            intensity=intensity * self.master_intensity,
            duration=duration,
            frequency=frequency,
            waveform=waveform
        )

        self._queue_pulse(pulse, target_device, 1.0)
        self._record_haptic("custom_pulse", target_device, duration)

        return True

    def play_click(self, device: HapticDevice | None = None) -> bool:
        """Convenience: Play click haptic."""
        return self.play_pattern("click", device)

    def play_success(self, device: HapticDevice | None = None) -> bool:
        """Convenience: Play success haptic."""
        return self.play_pattern("success", device)

    def play_error(self, device: HapticDevice | None = None) -> bool:
        """Convenience: Play error haptic."""
        return self.play_pattern("error", device)

    def play_grab(self, device: HapticDevice | None = None) -> bool:
        """Convenience: Play grab haptic."""
        return self.play_pattern("grab", device)

    def play_release(self, device: HapticDevice | None = None) -> bool:
        """Convenience: Play release haptic."""
        return self.play_pattern("release", device)

    def play_collision(
        self,
        velocity: float,
        device: HapticDevice | None = None
    ) -> bool:
        """Play collision haptic based on velocity."""
        cfg = self.collision_config

        if velocity < cfg.velocity_threshold:
            return False

        # Calculate intensity based on velocity
        normalized_v = min(1.0, velocity / 5.0)  # Normalize to 0-1

        if cfg.intensity_curve == "quadratic":
            intensity = normalized_v ** 2
        elif cfg.intensity_curve == "logarithmic":
            intensity = math.log1p(normalized_v * 9) / math.log(10)
        else:  # linear
            intensity = normalized_v

        intensity = min(cfg.max_intensity, intensity)

        # Calculate duration
        duration = cfg.duration_base + normalized_v * cfg.duration_scale

        # Calculate frequency (higher for harder hits)
        frequency = cfg.frequency_base + normalized_v * cfg.frequency_scale

        return self.play_pulse(
            intensity=intensity,
            duration=duration,
            device=device,
            frequency=frequency
        )

    def add_spatial_zone(
        self,
        zone_id: str,
        position: tuple[float, float, float],
        radius: float,
        pattern_id: str,
        trigger_on_enter: bool = True,
        trigger_continuous: bool = False
    ) -> bool:
        """Add a spatial haptic zone."""
        if pattern_id not in self.patterns:
            return False

        self.spatial_zones[zone_id] = SpatialHapticZone(
            zone_id=zone_id,
            position=position,
            radius=radius,
            pattern=self.patterns[pattern_id],
            trigger_on_enter=trigger_on_enter,
            trigger_continuous=trigger_continuous
        )
        return True

    def remove_spatial_zone(self, zone_id: str):
        """Remove a spatial haptic zone."""
        self.spatial_zones.pop(zone_id, None)

    def check_spatial_zones(
        self,
        hand_position: tuple[float, float, float],
        device: HapticDevice = HapticDevice.RIGHT_CONTROLLER
    ) -> list[str]:
        """Check if position triggers any spatial zones."""
        if not self.enable_spatial:
            return []

        triggered = []

        for zone_id, zone in self.spatial_zones.items():
            if not zone.enabled:
                continue

            # Calculate distance
            dx = hand_position[0] - zone.position[0]
            dy = hand_position[1] - zone.position[1]
            dz = hand_position[2] - zone.position[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)

            if distance <= zone.radius:
                # Calculate intensity based on distance
                intensity_scale = 1.0
                if zone.intensity_falloff:
                    intensity_scale = 1.0 - (distance / zone.radius)

                # Play pattern
                for pulse in zone.pattern.pulses:
                    scaled_pulse = HapticPulse(
                        device=device,
                        intensity=pulse.intensity * intensity_scale,
                        duration=pulse.duration,
                        frequency=pulse.frequency,
                        waveform=pulse.waveform,
                        envelope=pulse.envelope,
                        delay=pulse.delay
                    )
                    self._queue_pulse(scaled_pulse, device, 1.0)

                triggered.append(zone_id)

        return triggered

    def set_master_intensity(self, intensity: float):
        """Set master intensity (0.0 to 1.0)."""
        self.master_intensity = max(0.0, min(1.0, intensity))

    def set_device_available(self, device: HapticDevice, available: bool):
        """Set device availability."""
        self.available_devices[device] = available

    def stop_all(self):
        """Stop all active haptics."""
        self.active_pulses.clear()

    def _is_device_available(self, device: HapticDevice) -> bool:
        """Check if device is available."""
        if device == HapticDevice.ALL:
            return any(self.available_devices.values())
        if device == HapticDevice.BOTH_CONTROLLERS:
            return (self.available_devices.get(HapticDevice.LEFT_CONTROLLER, False) or
                    self.available_devices.get(HapticDevice.RIGHT_CONTROLLER, False))
        if device == HapticDevice.BOTH_HANDS:
            return (self.available_devices.get(HapticDevice.LEFT_HAND, False) or
                    self.available_devices.get(HapticDevice.RIGHT_HAND, False))
        return self.available_devices.get(device, False)

    def _queue_pulse(
        self,
        pulse: HapticPulse,
        target_device: HapticDevice,
        intensity_scale: float
    ):
        """Queue a pulse for playback."""
        self.active_pulses.append({
            "pulse": pulse,
            "device": target_device,
            "intensity_scale": intensity_scale,
            "start_time": datetime.now().timestamp(),
            "played": False
        })

    def _record_haptic(self, pattern_id: str, device: HapticDevice, duration: float):
        """Record haptic to history."""
        self.haptic_history.append({
            "pattern_id": pattern_id,
            "device": device.value,
            "duration": duration,
            "timestamp": datetime.now().timestamp()
        })

        # Trim history
        if len(self.haptic_history) > self.max_history:
            self.haptic_history = self.haptic_history[-self.max_history:]

    def get_statistics(self) -> dict[str, Any]:
        """Get haptic usage statistics."""
        pattern_counts: dict[str, int] = {}
        device_counts: dict[str, int] = {}
        total_duration = 0.0

        for record in self.haptic_history:
            pattern_id = record["pattern_id"]
            pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1

            device = record["device"]
            device_counts[device] = device_counts.get(device, 0) + 1

            total_duration += record["duration"]

        return {
            "total_haptics": len(self.haptic_history),
            "total_duration": total_duration,
            "patterns_used": pattern_counts,
            "devices_used": device_counts,
            "registered_patterns": len(self.patterns),
            "spatial_zones": len(self.spatial_zones),
            "master_intensity": self.master_intensity
        }

    def to_state(self) -> dict[str, Any]:
        """Export state for WebXR client."""
        return {
            "type": "haptic_state",
            "master_intensity": self.master_intensity,
            "available_devices": {d.value: a for d, a in self.available_devices.items()},
            "active_pulses": len(self.active_pulses),
            "registered_patterns": list(self.patterns.keys()),
            "spatial_zones": [
                {
                    "id": z.zone_id,
                    "position": z.position,
                    "radius": z.radius,
                    "enabled": z.enabled
                }
                for z in self.spatial_zones.values()
            ],
            "timestamp": datetime.now().isoformat()
        }

    def create_pattern(
        self,
        pattern_id: str,
        name: str
    ) -> HapticPattern:
        """Create and register a new empty pattern (fluent interface)."""
        pattern = HapticPattern(pattern_id=pattern_id, name=name)
        self.patterns[pattern_id] = pattern
        return pattern
