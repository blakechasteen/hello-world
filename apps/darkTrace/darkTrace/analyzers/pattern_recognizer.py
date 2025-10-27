"""
Pattern Recognizer
==================
Detect recurring patterns in semantic trajectories.

Patterns include:
- Oscillations (periodic movement between semantic regions)
- Spirals (gradual convergence with rotation)
- Jumps (sudden semantic shifts)
- Plateaus (stable semantic regions)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from enum import Enum
import logging

from darkTrace.observers.trajectory_recorder import Trajectory
from darkTrace.observers.semantic_observer import StateSnapshot


logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of detected patterns."""
    OSCILLATION = "oscillation"
    SPIRAL = "spiral"
    JUMP = "jump"
    PLATEAU = "plateau"
    DRIFT = "drift"
    LOOP = "loop"


@dataclass
class Pattern:
    """A detected semantic pattern."""

    # Pattern metadata
    pattern_type: PatternType
    confidence: float  # 0-1

    # Location in trajectory
    start_index: int
    end_index: int
    duration_tokens: int

    # Pattern characteristics
    frequency: Optional[float] = None  # For oscillations
    amplitude: Optional[float] = None  # For oscillations/spirals
    jump_magnitude: Optional[float] = None  # For jumps
    plateau_variance: Optional[float] = None  # For plateaus

    # Involved dimensions
    dominant_dimensions: List[str] = None

    # Description
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "confidence": self.confidence,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "duration_tokens": self.duration_tokens,
            "frequency": self.frequency,
            "amplitude": self.amplitude,
            "jump_magnitude": self.jump_magnitude,
            "plateau_variance": self.plateau_variance,
            "dominant_dimensions": self.dominant_dimensions,
            "description": self.description,
        }


class PatternRecognizer:
    """
    Detect recurring patterns in semantic trajectories.

    Usage:
        recognizer = PatternRecognizer(min_pattern_length=5)

        # Detect patterns
        patterns = recognizer.detect(trajectory)

        # Filter by type
        oscillations = [p for p in patterns if p.pattern_type == PatternType.OSCILLATION]
    """

    def __init__(
        self,
        min_pattern_length: int = 3,
        oscillation_threshold: float = 0.3,
        jump_threshold: float = 2.0,
        plateau_threshold: float = 0.1,
    ):
        """
        Initialize pattern recognizer.

        Args:
            min_pattern_length: Minimum length for pattern detection
            oscillation_threshold: Threshold for oscillation detection
            jump_threshold: Threshold for jump detection (in velocity units)
            plateau_threshold: Variance threshold for plateau detection
        """
        self.min_pattern_length = min_pattern_length
        self.oscillation_threshold = oscillation_threshold
        self.jump_threshold = jump_threshold
        self.plateau_threshold = plateau_threshold

        logger.info("PatternRecognizer initialized")

    def detect(self, trajectory: Trajectory) -> List[Pattern]:
        """
        Detect patterns in trajectory.

        Args:
            trajectory: Trajectory to analyze

        Returns:
            List of detected Pattern objects
        """
        if len(trajectory.snapshots) < self.min_pattern_length:
            return []

        patterns = []

        # Extract position sequence
        positions = np.array([s.position for s in trajectory.snapshots])
        velocities = np.array([
            s.velocity_magnitude for s in trajectory.snapshots
            if s.velocity is not None
        ])

        # Detect different pattern types
        patterns.extend(self._detect_oscillations(trajectory, positions))
        patterns.extend(self._detect_jumps(trajectory, velocities))
        patterns.extend(self._detect_plateaus(trajectory, positions))
        patterns.extend(self._detect_spirals(trajectory, positions))
        patterns.extend(self._detect_loops(trajectory, positions))

        # Sort by start index
        patterns.sort(key=lambda p: p.start_index)

        logger.info(f"Detected {len(patterns)} patterns")
        return patterns

    def _detect_oscillations(
        self,
        trajectory: Trajectory,
        positions: np.ndarray
    ) -> List[Pattern]:
        """Detect oscillatory patterns."""
        patterns = []

        # Use sliding window
        window_size = max(self.min_pattern_length, 10)

        for i in range(len(positions) - window_size):
            window = positions[i:i+window_size]

            # Compute centroid
            centroid = np.mean(window, axis=0)

            # Compute distances from centroid
            distances = np.linalg.norm(window - centroid, axis=1)

            # Check for oscillation (alternating near/far from centroid)
            crossings = 0
            mean_dist = np.mean(distances)

            for j in range(len(distances) - 1):
                if (distances[j] < mean_dist and distances[j+1] > mean_dist) or \
                   (distances[j] > mean_dist and distances[j+1] < mean_dist):
                    crossings += 1

            # Oscillation if multiple crossings
            if crossings >= 3:
                amplitude = float(np.std(distances))
                frequency = crossings / window_size

                # Get dominant dimensions
                dominant_dims = self._get_dominant_dimensions(
                    trajectory.snapshots[i:i+window_size]
                )

                pattern = Pattern(
                    pattern_type=PatternType.OSCILLATION,
                    confidence=min(1.0, crossings / 5),
                    start_index=i,
                    end_index=i + window_size,
                    duration_tokens=window_size,
                    frequency=frequency,
                    amplitude=amplitude,
                    dominant_dimensions=dominant_dims,
                    description=f"Oscillation with {crossings} cycles, amplitude {amplitude:.3f}"
                )

                patterns.append(pattern)

                # Skip ahead to avoid overlapping detections
                i += window_size // 2

        return patterns

    def _detect_jumps(
        self,
        trajectory: Trajectory,
        velocities: np.ndarray
    ) -> List[Pattern]:
        """Detect sudden semantic jumps."""
        patterns = []

        if len(velocities) < 2:
            return patterns

        # Find velocity spikes
        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)

        for i, vel in enumerate(velocities):
            if vel > mean_velocity + self.jump_threshold * std_velocity:
                # Significant jump detected
                pattern = Pattern(
                    pattern_type=PatternType.JUMP,
                    confidence=min(1.0, (vel - mean_velocity) / (3 * std_velocity)),
                    start_index=i,
                    end_index=i + 1,
                    duration_tokens=1,
                    jump_magnitude=float(vel),
                    description=f"Semantic jump (velocity {vel:.3f})"
                )

                patterns.append(pattern)

        return patterns

    def _detect_plateaus(
        self,
        trajectory: Trajectory,
        positions: np.ndarray
    ) -> List[Pattern]:
        """Detect plateau regions (low variance)."""
        patterns = []

        window_size = max(self.min_pattern_length, 5)

        for i in range(len(positions) - window_size):
            window = positions[i:i+window_size]

            # Compute variance
            variance = np.var(window, axis=0).mean()

            if variance < self.plateau_threshold:
                # Plateau detected
                dominant_dims = self._get_dominant_dimensions(
                    trajectory.snapshots[i:i+window_size]
                )

                pattern = Pattern(
                    pattern_type=PatternType.PLATEAU,
                    confidence=min(1.0, (self.plateau_threshold - variance) / self.plateau_threshold),
                    start_index=i,
                    end_index=i + window_size,
                    duration_tokens=window_size,
                    plateau_variance=float(variance),
                    dominant_dimensions=dominant_dims,
                    description=f"Plateau (variance {variance:.4f})"
                )

                patterns.append(pattern)

                # Skip ahead
                i += window_size // 2

        return patterns

    def _detect_spirals(
        self,
        trajectory: Trajectory,
        positions: np.ndarray
    ) -> List[Pattern]:
        """Detect spiral patterns (convergence with rotation)."""
        patterns = []

        window_size = max(self.min_pattern_length, 15)

        for i in range(len(positions) - window_size):
            window = positions[i:i+window_size]

            # Compute distances from centroid
            centroid = np.mean(window, axis=0)
            distances = np.linalg.norm(window - centroid, axis=1)

            # Check for decreasing distance (convergence)
            is_converging = np.polyfit(range(len(distances)), distances, 1)[0] < -0.01

            # Check for rotation (angular momentum)
            if is_converging and len(window) > 5:
                # Simple rotation check: consecutive angles change sign
                angles_change = 0
                for j in range(2, len(window)):
                    v1 = window[j-1] - window[j-2]
                    v2 = window[j] - window[j-1]

                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        # Dot product indicates angle
                        dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        if dot < 0.7:  # Angle > 45 degrees
                            angles_change += 1

                if angles_change >= 3:
                    pattern = Pattern(
                        pattern_type=PatternType.SPIRAL,
                        confidence=min(1.0, angles_change / 5),
                        start_index=i,
                        end_index=i + window_size,
                        duration_tokens=window_size,
                        amplitude=float(distances[0] - distances[-1]),
                        description=f"Spiral pattern (converging with rotation)"
                    )

                    patterns.append(pattern)

                    # Skip ahead
                    i += window_size // 2

        return patterns

    def _detect_loops(
        self,
        trajectory: Trajectory,
        positions: np.ndarray
    ) -> List[Pattern]:
        """Detect loop patterns (return to previous state)."""
        patterns = []

        # Check for returns to previous positions
        for i in range(self.min_pattern_length, len(positions)):
            current_pos = positions[i]

            # Look for similar previous position
            for j in range(max(0, i - 50), i - self.min_pattern_length):
                prev_pos = positions[j]
                distance = np.linalg.norm(current_pos - prev_pos)

                # If close to previous position, we have a loop
                if distance < 0.5:
                    loop_size = i - j

                    pattern = Pattern(
                        pattern_type=PatternType.LOOP,
                        confidence=min(1.0, (0.5 - distance) / 0.5),
                        start_index=j,
                        end_index=i,
                        duration_tokens=loop_size,
                        description=f"Loop of length {loop_size} tokens"
                    )

                    patterns.append(pattern)
                    break

        return patterns

    def _get_dominant_dimensions(
        self,
        snapshots: List[StateSnapshot]
    ) -> List[str]:
        """Get dominant dimensions in a window."""
        dim_scores = {}

        for snapshot in snapshots:
            for dim, score in snapshot.dimension_scores.items():
                dim_scores[dim] = dim_scores.get(dim, 0.0) + score

        sorted_dims = sorted(dim_scores.items(), key=lambda x: x[1], reverse=True)
        return [dim for dim, _ in sorted_dims[:5]]
