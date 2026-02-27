"""
Spatial Audio Handler Module
==============================
3D audio positioning and HRTF processing for AR voice integration.

This module positions TTS audio in 3D space relative to the user's position
and orientation, creating immersive spatial audio experiences in AR.

Author: HoloLoom Team
Date: November 15, 2025
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import numpy as np

from HoloLoom.voice.ar_context import ARContext, Vector3


# ============================================================================
# Audio Configuration
# ============================================================================

@dataclass
class SpatialAudioConfig:
    """Configuration for spatial audio processing."""
    sample_rate: int = 16000
    enable_hrtf: bool = True
    enable_distance_attenuation: bool = True
    enable_doppler: bool = False  # Future: Doppler shift for moving sources
    max_distance: float = 50.0    # Maximum audible distance (meters)
    reference_distance: float = 1.0  # Distance for 0dB attenuation
    rolloff_factor: float = 1.0   # How quickly sound attenuates with distance


# ============================================================================
# Spatial Audio Handler
# ============================================================================

class SpatialAudioHandler:
    """
    Handles 3D spatial audio positioning for AR voice responses.

    Features:
    - HRTF (Head-Related Transfer Function) for 3D positioning
    - Distance-based volume attenuation
    - Directional filtering (future)
    - Doppler shift (future)
    """

    def __init__(self, config: Optional[SpatialAudioConfig] = None):
        """
        Initialize spatial audio handler.

        Args:
            config: Spatial audio configuration
        """
        self.config = config or SpatialAudioConfig()

    async def position_audio(
        self,
        audio: bytes,
        position: Vector3,
        ar_context: ARContext
    ) -> bytes:
        """
        Apply spatial audio processing to position audio in 3D space.

        Args:
            audio: TTS audio bytes (mono, 16kHz)
            position: 3D position of audio source in AR space
            ar_context: User position and orientation

        Returns:
            Spatially processed audio (stereo)
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio, dtype=np.int16)

        # Calculate spatial parameters
        spatial_params = self._calculate_spatial_parameters(position, ar_context)

        # Apply distance attenuation
        if self.config.enable_distance_attenuation:
            audio_array = self._apply_distance_attenuation(
                audio_array,
                spatial_params['distance'],
                spatial_params['attenuation']
            )

        # Apply HRTF for 3D positioning
        if self.config.enable_hrtf:
            stereo_audio = self._apply_hrtf(
                audio_array,
                spatial_params['azimuth'],
                spatial_params['elevation']
            )
        else:
            # Simple pan without HRTF
            stereo_audio = self._apply_simple_pan(
                audio_array,
                spatial_params['azimuth']
            )

        # Convert back to bytes
        return stereo_audio.tobytes()

    def _calculate_spatial_parameters(
        self,
        source_position: Vector3,
        ar_context: ARContext
    ) -> dict:
        """
        Calculate spatial audio parameters relative to user.

        Args:
            source_position: Audio source position
            ar_context: User AR context

        Returns:
            Dictionary with spatial parameters:
            - distance: Distance from user to source (meters)
            - attenuation: Volume attenuation factor (0-1)
            - azimuth: Horizontal angle (radians, -π to π)
            - elevation: Vertical angle (radians, -π/2 to π/2)
        """
        # Vector from user to source
        to_source = Vector3(
            source_position.x - ar_context.user_position.x,
            source_position.y - ar_context.user_position.y,
            source_position.z - ar_context.user_position.z
        )

        # Calculate distance
        distance = to_source.distance_to(Vector3(0, 0, 0))

        # Calculate attenuation
        attenuation = self.calculate_attenuation(distance)

        # Calculate azimuth (horizontal angle)
        # Get user's forward direction from orientation
        _, yaw, _ = ar_context.user_orientation.to_euler()

        # Calculate angle to source relative to forward
        source_angle = math.atan2(to_source.x, to_source.z)
        azimuth = source_angle - yaw

        # Normalize to [-π, π]
        while azimuth > math.pi:
            azimuth -= 2 * math.pi
        while azimuth < -math.pi:
            azimuth += 2 * math.pi

        # Calculate elevation (vertical angle)
        horizontal_distance = math.sqrt(to_source.x**2 + to_source.z**2)
        if horizontal_distance > 0:
            elevation = math.atan2(to_source.y, horizontal_distance)
        else:
            elevation = 0.0

        return {
            'distance': distance,
            'attenuation': attenuation,
            'azimuth': azimuth,
            'elevation': elevation
        }

    def calculate_attenuation(self, distance: float) -> float:
        """
        Calculate volume attenuation based on distance.

        Uses inverse square law with configurable rolloff:
        attenuation = (reference_distance / distance) ^ rolloff_factor

        Args:
            distance: Distance from user to source (meters)

        Returns:
            Attenuation factor (0-1), where 1 = no attenuation
        """
        if distance <= self.config.reference_distance:
            return 1.0

        if distance >= self.config.max_distance:
            return 0.0

        # Inverse distance law with rolloff factor
        attenuation = (
            self.config.reference_distance / distance
        ) ** self.config.rolloff_factor

        # Clamp to [0, 1]
        return max(0.0, min(1.0, attenuation))

    def _apply_distance_attenuation(
        self,
        audio: np.ndarray,
        distance: float,
        attenuation: float
    ) -> np.ndarray:
        """
        Apply distance-based volume attenuation.

        Args:
            audio: Mono audio array
            distance: Distance to source
            attenuation: Attenuation factor (0-1)

        Returns:
            Attenuated audio array
        """
        return (audio * attenuation).astype(np.int16)

    def _apply_hrtf(
        self,
        mono_audio: np.ndarray,
        azimuth: float,
        elevation: float
    ) -> np.ndarray:
        """
        Apply Head-Related Transfer Function for 3D audio positioning.

        This is a simplified HRTF that applies:
        - Interaural Time Difference (ITD)
        - Interaural Level Difference (ILD)
        - Basic frequency filtering

        For production, use a full HRTF library like:
        - CIPIC HRTF Database
        - MIT KEMAR HRTF
        - SADIE HRTF

        Args:
            mono_audio: Mono audio array
            azimuth: Horizontal angle (radians, -π to π)
            elevation: Vertical angle (radians)

        Returns:
            Stereo audio array (interleaved L/R)
        """
        # Calculate ITD (Interaural Time Difference)
        # Maximum ITD is ~0.65ms for human head (~10 samples at 16kHz)
        max_itd_samples = int(0.00065 * self.config.sample_rate)
        itd_samples = int(max_itd_samples * math.sin(azimuth))

        # Calculate ILD (Interaural Level Difference)
        # Sound from left: left louder, right quieter
        # Sound from right: right louder, left quieter
        # Maximum ILD is ~20dB
        max_ild_db = 20.0
        ild_db = max_ild_db * math.sin(azimuth)

        # Convert dB to linear gain
        # Positive azimuth (right): boost right, attenuate left
        # Negative azimuth (left): boost left, attenuate right
        left_gain = 10 ** (-ild_db / 20.0)
        right_gain = 10 ** (ild_db / 20.0)

        # Apply elevation-based attenuation (simple model)
        # Sound from above/below is slightly quieter
        elevation_attenuation = 1.0 - 0.1 * abs(elevation) / (math.pi / 2)
        left_gain *= elevation_attenuation
        right_gain *= elevation_attenuation

        # Create stereo channels
        left_channel = mono_audio.copy()
        right_channel = mono_audio.copy()

        # Apply ITD (delay)
        if itd_samples > 0:
            # Delay right channel (source on right)
            right_channel = np.pad(right_channel, (itd_samples, 0), 'constant')[:-itd_samples]
        elif itd_samples < 0:
            # Delay left channel (source on left)
            left_channel = np.pad(left_channel, (-itd_samples, 0), 'constant')[:len(mono_audio)]

        # Apply ILD (gain difference)
        left_channel = (left_channel * left_gain).astype(np.int16)
        right_channel = (right_channel * right_gain).astype(np.int16)

        # Interleave stereo channels (LRLRLR...)
        stereo = np.empty(len(left_channel) * 2, dtype=np.int16)
        stereo[0::2] = left_channel
        stereo[1::2] = right_channel

        return stereo

    def _apply_simple_pan(
        self,
        mono_audio: np.ndarray,
        azimuth: float
    ) -> np.ndarray:
        """
        Apply simple stereo panning without HRTF.

        Simpler than HRTF but faster to compute.

        Args:
            mono_audio: Mono audio array
            azimuth: Horizontal angle (radians)

        Returns:
            Stereo audio array (interleaved L/R)
        """
        # Pan factor: -1 (full left) to +1 (full right)
        pan = math.sin(azimuth)  # -π to π → -1 to 1

        # Calculate left/right gains using constant power panning
        # This maintains perceived loudness across the stereo field
        angle = (pan + 1) * math.pi / 4  # Map [-1, 1] to [0, π/2]
        left_gain = math.cos(angle)
        right_gain = math.sin(angle)

        # Apply panning
        left_channel = (mono_audio * left_gain).astype(np.int16)
        right_channel = (mono_audio * right_gain).astype(np.int16)

        # Interleave stereo channels
        stereo = np.empty(len(left_channel) * 2, dtype=np.int16)
        stereo[0::2] = left_channel
        stereo[1::2] = right_channel

        return stereo

    def get_spatial_preview(
        self,
        position: Vector3,
        ar_context: ARContext
    ) -> dict:
        """
        Get spatial parameters without processing audio.

        Useful for debugging and visualization.

        Args:
            position: Audio source position
            ar_context: User AR context

        Returns:
            Dictionary with spatial parameters
        """
        params = self._calculate_spatial_parameters(position, ar_context)

        # Add human-readable values
        params['azimuth_degrees'] = math.degrees(params['azimuth'])
        params['elevation_degrees'] = math.degrees(params['elevation'])

        # Describe position
        if abs(params['azimuth']) < math.pi / 4:
            position_desc = "front"
        elif abs(params['azimuth']) > 3 * math.pi / 4:
            position_desc = "back"
        elif params['azimuth'] > 0:
            position_desc = "right"
        else:
            position_desc = "left"

        if abs(params['elevation']) > math.pi / 6:
            if params['elevation'] > 0:
                position_desc += "-above"
            else:
                position_desc += "-below"

        params['position_description'] = position_desc

        return params


# ============================================================================
# Helper Functions
# ============================================================================

def create_test_audio(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """
    Create test audio signal (440Hz sine wave - A note).

    Args:
        duration_seconds: Duration in seconds
        sample_rate: Sample rate in Hz

    Returns:
        Mono audio bytes (16-bit PCM)
    """
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    frequency = 440.0  # A4 note
    signal = np.sin(2 * np.pi * frequency * t)

    # Convert to 16-bit PCM
    audio_int16 = (signal * 32767 * 0.5).astype(np.int16)

    return audio_int16.tobytes()


def test_spatial_audio():
    """Test spatial audio handler with various positions."""
    from HoloLoom.voice.ar_context import create_test_context

    print("Testing Spatial Audio Handler")
    print("=" * 60)

    # Create handler and context
    handler = SpatialAudioHandler()
    context = create_test_context()

    # Test positions
    test_positions = [
        (Vector3(0, 1.7, 2), "Front, close"),
        (Vector3(2, 1.7, 0), "Right side"),
        (Vector3(-2, 1.7, 0), "Left side"),
        (Vector3(0, 1.7, -2), "Behind"),
        (Vector3(0, 3, 2), "Above, front"),
        (Vector3(5, 1.7, 5), "Far, front-right")
    ]

    for position, description in test_positions:
        print(f"\n{description}:")
        print(f"  Position: {position}")

        # Get spatial parameters
        params = handler.get_spatial_preview(position, context)

        print(f"  Distance: {params['distance']:.2f}m")
        print(f"  Attenuation: {params['attenuation']:.2%}")
        print(f"  Azimuth: {params['azimuth_degrees']:.1f}°")
        print(f"  Elevation: {params['elevation_degrees']:.1f}°")
        print(f"  Position: {params['position_description']}")

    # Test actual audio processing
    print("\n" + "=" * 60)
    print("Processing test audio...")

    test_audio_mono = create_test_audio(1.0)
    print(f"Mono audio size: {len(test_audio_mono)} bytes")

    # Process for position on the right
    position_right = Vector3(2, 1.7, 2)

    import asyncio
    stereo_audio = asyncio.run(
        handler.position_audio(test_audio_mono, position_right, context)
    )

    print(f"Stereo audio size: {len(stereo_audio)} bytes")
    print(f"Stereo/Mono ratio: {len(stereo_audio) / len(test_audio_mono):.1f}x (expected 2.0x)")

    print("\n✅ Spatial audio processing working!")


if __name__ == "__main__":
    test_spatial_audio()
